"""Runtime aerodynamic drag for movable RoboDojo task bodies.

Pure geometry/force helpers stay importable without Isaac Sim. Isaac Sim APIs
are imported only by :func:`create_rigid_wind_controller`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


def _vector3(value: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3D vector")
    return result


def compute_drag_forces(
    wind: Iterable[float],
    velocities: Iterable[Iterable[float]],
    areas: Iterable[float],
    air_density: float = 1.225,
    drag_coefficient: float = 1.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Compute quadratic drag forces for bodies in world coordinates.

    ``wind`` is the ambient air velocity in m/s. ``scale`` is exposed as a
    tuning factor because RoboDojo assets do not author calibrated aero data.
    """
    wind_vector = _vector3(wind, "wind")
    velocity_array = np.asarray(velocities, dtype=np.float64)
    area_array = np.asarray(areas, dtype=np.float64)
    if velocity_array.ndim != 2 or velocity_array.shape[1] != 3:
        raise ValueError("velocities must have shape (N, 3)")
    if area_array.shape != (velocity_array.shape[0],):
        raise ValueError("areas must have shape (N,) matching velocities")
    if not np.all(np.isfinite(velocity_array)):
        raise ValueError("velocities must contain only finite values")
    if not np.all(np.isfinite(area_array)) or np.any(area_array < 0.0):
        raise ValueError("areas must contain finite non-negative values")
    coefficients = np.asarray((air_density, drag_coefficient, scale), dtype=np.float64)
    if not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0.0):
        raise ValueError("air_density, drag_coefficient, and scale must be finite and non-negative")

    relative_air_velocity = wind_vector[None, :] - velocity_array
    speed = np.linalg.norm(relative_air_velocity, axis=1)
    return (
        0.5
        * float(air_density)
        * float(drag_coefficient)
        * float(scale)
        * area_array[:, None]
        * speed[:, None]
        * relative_air_velocity
    )


def box_projected_area(
    local_extents: Iterable[float],
    orientation: Iterable[Iterable[float]],
    wind: Iterable[float],
) -> float:
    """Estimate box area projected onto a plane normal to the wind."""
    extents = _vector3(local_extents, "local_extents")
    rotation = np.asarray(orientation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("orientation must be a finite 3x3 rotation matrix")
    if np.any(extents < 0.0):
        raise ValueError("local_extents must be non-negative")
    wind_vector = _vector3(wind, "wind")
    wind_norm = float(np.linalg.norm(wind_vector))
    if wind_norm == 0.0:
        return 0.0

    local_wind = rotation.T @ (wind_vector / wind_norm)
    face_areas = np.asarray(
        (extents[1] * extents[2], extents[0] * extents[2], extents[0] * extents[1]),
        dtype=np.float64,
    )
    return float(np.dot(np.abs(local_wind), face_areas))


def quaternion_wxyz_to_matrix(quaternion: Iterable[float]) -> np.ndarray:
    """Convert an Isaac Sim scalar-first quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must be a finite 4D vector")
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = q / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def capsule_projected_area(
    radius: float,
    cylinder_length: float,
    axis: Iterable[float],
    wind: Iterable[float],
) -> float:
    """Estimate a capsule's projected area, including its hemispherical ends."""
    radius = float(radius)
    cylinder_length = float(cylinder_length)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and non-negative")
    if not math.isfinite(cylinder_length) or cylinder_length < 0.0:
        raise ValueError("cylinder_length must be finite and non-negative")
    axis_vector = _vector3(axis, "axis")
    wind_vector = _vector3(wind, "wind")
    axis_norm = float(np.linalg.norm(axis_vector))
    wind_norm = float(np.linalg.norm(wind_vector))
    if axis_norm == 0.0:
        raise ValueError("axis must be non-zero")
    if wind_norm == 0.0:
        return 0.0
    alignment = float(np.dot(axis_vector / axis_norm, wind_vector / wind_norm))
    broadside_factor = math.sqrt(max(0.0, 1.0 - alignment * alignment))
    return 2.0 * radius * cylinder_length * broadside_factor + math.pi * radius * radius


def sphere_projected_area(radius: float) -> float:
    """Estimate the frontal area of a sphere."""
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and non-negative")
    return math.pi * radius * radius


def is_wind_body_candidate(object_kind: str, fixed_base: bool = False, is_robot: bool = False) -> bool:
    """Return whether a scene-managed physical object may receive wind."""
    return object_kind in {"rigid", "dynamic", "articulation"} and not fixed_base and not is_robot


@dataclass(frozen=True)
class RigidBodySpec:
    prim_path: str
    label: str
    local_extents: tuple[float, float, float]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


class RigidWindController:
    """Apply computed drag forces to an Isaac Sim rigid body view."""

    def __init__(
        self,
        view: Any,
        bodies: Iterable[RigidBodySpec],
        wind: Iterable[float],
        air_density: float = 1.225,
        drag_coefficient: float = 1.0,
        force_scale: float = 1.0,
    ):
        self.view = view
        self.bodies = tuple(bodies)
        self.wind = _vector3(wind, "wind")
        self.air_density = float(air_density)
        self.drag_coefficient = float(drag_coefficient)
        self.force_scale = float(force_scale)
        self.last_diagnostics: dict[str, Any] = {
            "body_count": len(self.bodies),
            "applied_body_count": 0,
            "bodies": [],
        }

    def step(self) -> dict[str, Any]:
        if not self.bodies or not np.any(self.wind):
            self.last_diagnostics = {
                "body_count": len(self.bodies),
                "applied_body_count": 0,
                "bodies": [],
            }
            return self.last_diagnostics

        velocities = _to_numpy(self.view.get_linear_velocities())
        _, quaternions = self.view.get_world_poses()
        quaternions = _to_numpy(quaternions)
        if velocities.shape != (len(self.bodies), 3) or quaternions.shape != (len(self.bodies), 4):
            raise RuntimeError(
                "Rigid body view state shape does not match selected paths: "
                f"velocities={velocities.shape}, orientations={quaternions.shape}, bodies={len(self.bodies)}"
            )

        areas = np.asarray(
            [
                box_projected_area(
                    body.local_extents,
                    quaternion_wxyz_to_matrix(quaternion),
                    self.wind,
                )
                for body, quaternion in zip(self.bodies, quaternions)
            ],
            dtype=np.float64,
        )
        forces = compute_drag_forces(
            self.wind,
            velocities,
            areas,
            air_density=self.air_density,
            drag_coefficient=self.drag_coefficient,
            scale=self.force_scale,
        )
        self.view.apply_forces(forces.astype(np.float32), is_global=True)
        body_rows = []
        for body, area, force in zip(self.bodies, areas, forces):
            body_rows.append(
                {
                    "prim_path": body.prim_path,
                    "label": body.label,
                    "projected_area_m2": float(area),
                    "force_n": [float(value) for value in force],
                    "force_norm_n": float(np.linalg.norm(force)),
                }
            )
        self.last_diagnostics = {
            "body_count": len(self.bodies),
            "applied_body_count": len(body_rows),
            "bodies": body_rows,
        }
        return self.last_diagnostics


def install_sim_step_wind_hook(env: Any, controller: RigidWindController) -> Any:
    """Apply task-body wind immediately before each underlying physics step."""
    original_sim_step = env.sim_step

    def sim_step_with_wind(*args: Any, **kwargs: Any) -> Any:
        controller.step()
        return original_sim_step(*args, **kwargs)

    env.sim_step = sim_step_with_wind
    return original_sim_step


def _prim_local_extents(prim: Any, bbox_cache: Any) -> tuple[float, float, float] | None:
    try:
        size = bbox_cache.ComputeLocalBound(prim).GetRange().GetSize()
        extents = np.asarray(size, dtype=np.float64)
    except Exception:
        return None
    if extents.shape != (3,) or not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
        return None
    return tuple(float(value) for value in extents)


def discover_wind_bodies(scene_manager: Any) -> tuple[list[RigidBodySpec], list[dict[str, str]]]:
    """Find dynamic rigid bodies in RoboDojo task-object registries.

    Robot assets are held by ``robot_manager``, not these scene registries.
    Static Geometry and the room/table registries are intentionally omitted.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = scene_manager.stage
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    specs: list[RigidBodySpec] = []
    skipped: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    object_groups = (
        ("rigid", getattr(scene_manager, "_rigid_and_dynamic_objects", [])),
        ("articulation", getattr(scene_manager, "_articulation_objects", [])),
    )
    for object_kind, per_env in object_groups:
        for objects in per_env:
            for object_key, obj in objects.items():
                physics_cfg = getattr(obj, "physics_config", getattr(obj, "physics_cfg", {})) or {}
                root_path = getattr(obj, "usd_prim_path", getattr(obj, "_prim_path", None))
                is_robot = bool(getattr(obj, "is_robot", False)) or (root_path and "/robot" in root_path.lower())
                fixed_base = bool(physics_cfg.get("fixed_base", False)) if object_kind == "articulation" else False
                if not is_wind_body_candidate(object_kind, fixed_base=fixed_base, is_robot=is_robot):
                    skipped.append({"label": str(object_key), "reason": "robot_or_fixed_base"})
                    continue
                if not root_path:
                    skipped.append({"label": str(object_key), "reason": "missing_prim_path"})
                    continue
                root_prim = stage.GetPrimAtPath(root_path)
                if not root_prim or not root_prim.IsValid():
                    skipped.append({"label": str(object_key), "reason": "invalid_prim_path"})
                    continue

                found_dynamic_body = False
                for prim in Usd.PrimRange(root_prim):
                    path = prim.GetPath().pathString
                    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        continue
                    found_dynamic_body = True
                    if path in seen_paths:
                        continue
                    rigid_api = UsdPhysics.RigidBodyAPI(prim)
                    kinematic_attr = rigid_api.GetKinematicEnabledAttr()
                    if kinematic_attr and kinematic_attr.Get():
                        skipped.append({"label": path, "reason": "kinematic_body"})
                        continue
                    extents = _prim_local_extents(prim, bbox_cache)
                    if extents is None:
                        skipped.append({"label": path, "reason": "missing_or_invalid_bounds"})
                        continue
                    specs.append(RigidBodySpec(path, f"{object_key}/{prim.GetName()}", extents))
                    seen_paths.add(path)
                if not found_dynamic_body:
                    skipped.append({"label": str(object_key), "reason": "no_rigid_body_api"})

    return specs, skipped


def create_rigid_wind_controller(
    scene_manager: Any,
    wind: Iterable[float],
    air_density: float = 1.225,
    drag_coefficient: float = 1.0,
    force_scale: float = 1.0,
) -> tuple[RigidWindController | None, list[dict[str, str]]]:
    """Create a live Isaac Sim rigid view for eligible task-body links."""
    specs, skipped = discover_wind_bodies(scene_manager)
    if not specs:
        return None, skipped

    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager

    view = RigidPrim(
        prim_paths_expr=[body.prim_path for body in specs],
        name="robodojo_constant_wind_task_bodies",
        reset_xform_properties=False,
        prepare_contact_sensors=False,
    )
    view.initialize(physics_sim_view=SimulationManager.get_physics_sim_view())
    return (
        RigidWindController(
            view=view,
            bodies=specs,
            wind=wind,
            air_density=air_density,
            drag_coefficient=drag_coefficient,
            force_scale=force_scale,
        ),
        skipped,
    )
