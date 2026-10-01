"""Isaac Lab FEM soft-body wrapper used by RoboDojo scene management."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from isaacsim.core.utils.prims import delete_prim
from isaaclab.assets import DeformableObject as IsaacLabDeformableObject
from isaaclab.assets import DeformableObjectCfg
import isaaclab.sim as sim_utils
from omegaconf import DictConfig, OmegaConf

from utils.deformable_metrics import validate_deformable_config


class DeformableObject:
    """Wrap one Isaac Lab volume-deformable object for RoboDojo.

    FEM assets are only supported by the GPU simulation backend in the
    supported Isaac Lab release. Isaac Lab creates the sponge's cuboid mesh
    and applies deformable schemas when the mesh is spawned.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str | None,
        inst_config: DictConfig | dict[str, Any],
        env_origin: torch.Tensor | np.ndarray,
        default_pos: tuple[float, float, float],
        default_ori: tuple[float, float, float, float],
        scale: tuple[float, float, float],
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Isaac Lab FEM deformables require GPU simulation; set the RoboDojo device to CUDA.")
        self.prim_path = prim_path
        self.usd_prim_path = prim_path
        # Kept for compatibility with SceneManager's shared wrapper signature.
        # FEM sponge geometry is generated from the task's configured size.
        self.usd_path = usd_path
        self.instance_name = prim_path.rstrip("/").split("/")[-1]
        self.category_name = prim_path.rstrip("/").split("/")[-2]
        self.instance_config = inst_config
        if isinstance(inst_config, DictConfig):
            config = OmegaConf.to_container(inst_config, resolve=True)
        else:
            config = dict(inst_config)
        if "color" not in config:
            config["color"] = config.get("visual", {}).get("color", (0.78, 0.70, 0.43, 1.0))
        self.physics_config = dict(config.get("physics", {}))
        self.deformable_config = validate_deformable_config(config)
        self.env_origin = np.asarray(env_origin.detach().cpu() if isinstance(env_origin, torch.Tensor) else env_origin, dtype=np.float32)
        self.default_pos = tuple(float(value) for value in default_pos)
        self.default_ori = tuple(float(value) for value in default_ori)
        self.scale = tuple(float(value) for value in scale)
        if len(self.default_pos) != 3 or len(self.default_ori) != 4 or len(self.scale) != 3:
            raise ValueError("FEM initial position, orientation, and scale must have 3, 4, and 3 values")

        physics = self.deformable_config["physics"]
        spawn_cfg = sim_utils.MeshCuboidCfg(
            size=self.deformable_config["size"],
            scale=self.scale,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                deformable_enabled=True,
                kinematic_enabled=False,
                collision_simplification=True,
                collision_simplification_remeshing=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.deformable_config["color"][:3]),
        )
        # Some RoboDojo Isaac Lab builds omit ``physics_material`` from the
        # spawner constructor while the spawn function still reads it. Attach
        # it after construction to preserve FEM material settings across builds.
        spawn_cfg.physics_material = sim_utils.DeformableBodyMaterialCfg(
            density=physics["density"],
            youngs_modulus=physics["youngs_modulus"],
            poissons_ratio=physics["poissons_ratio"],
            elasticity_damping=physics["elasticity_damping"],
            dynamic_friction=physics["dynamic_friction"],
        )
        self.cfg = DeformableObjectCfg(
            prim_path=self.prim_path,
            spawn=spawn_cfg,
            init_state=DeformableObjectCfg.InitialStateCfg(pos=self.default_pos, rot=self.default_ori),
            debug_vis=False,
        )
        self.asset = IsaacLabDeformableObject(cfg=self.cfg)
        self._default_nodal_state_w: torch.Tensor | None = None

    def initialize(self) -> None:
        """Cache the default state after Isaac Lab's physics-ready callback."""
        if not self.asset.is_initialized:
            raise RuntimeError(
                f"Isaac Lab has not initialized FEM object {self.prim_path}; "
                "create it before the simulation starts playing."
            )
        self._default_nodal_state_w = self.asset.data.default_nodal_state_w.clone()

    def update(self, dt: float) -> None:
        """Refresh Isaac Lab's node buffers after one simulation step."""
        if not self.asset.is_initialized:
            raise RuntimeError(f"FEM object {self.prim_path} is not initialized")
        self.asset.update(float(dt))

    def get_nodal_positions_w(self) -> torch.Tensor:
        """Return the simulated FEM node positions in world coordinates."""
        return self.asset.data.nodal_pos_w[0]

    def get_state(self, is_relative: bool = False) -> dict[str, torch.Tensor]:
        """Return an approximate center pose plus the complete nodal state.

        A volume deformable has no rigid root pose. ``root_pose`` is provided
        only for RoboDojo utilities that expect a pose; task success checks use
        ``nodal_pos_w`` directly.
        """
        nodes_w = self.get_nodal_positions_w()
        center_w = nodes_w.mean(dim=0)
        if is_relative:
            center_w = center_w - torch.as_tensor(self.env_origin, device=center_w.device, dtype=center_w.dtype)
        orientation = torch.as_tensor(self.default_ori, device=center_w.device, dtype=center_w.dtype)
        return {"root_pose": torch.cat((center_w, orientation)), "nodal_pos_w": nodes_w}

    def get_bbox(self, is_relative: bool = True):
        """Return center, nominal orientation, and current node bounds."""
        nodes = self.get_nodal_positions_w().detach().cpu().numpy()
        center = nodes.mean(axis=0)
        if is_relative:
            center = center - self.env_origin
        local_nodes = nodes - nodes.mean(axis=0)
        bounds = np.concatenate((local_nodes.min(axis=0), local_nodes.max(axis=0)))
        return center, np.asarray(self.default_ori, dtype=np.float32), bounds

    def apply_saved_pose(self) -> None:
        """Restore the initial nodal state and release all kinematic targets."""
        if self._default_nodal_state_w is None:
            raise RuntimeError(f"FEM object {self.prim_path} is not initialized")
        nodal_state = self._default_nodal_state_w.clone()
        self.asset.write_nodal_state_to_sim(nodal_state)
        targets = self.asset.data.nodal_kinematic_target.clone()
        targets[..., :3] = nodal_state[..., :3]
        targets[..., 3] = 1.0
        self.asset.write_nodal_kinematic_target_to_sim(targets)
        self.asset.reset()

    def reset(self) -> None:
        self.apply_saved_pose()

    def destroy(self) -> None:
        """Delete this instance's USD prim from the active stage."""
        delete_prim(self.prim_path)
        self.asset = None
