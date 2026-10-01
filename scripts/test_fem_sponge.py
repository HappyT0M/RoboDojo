"""Check that the project FEM sponge loads, advances, and resets in Isaac Sim.

Run from the RoboDojo root in the Isaac Sim environment::

    python scripts/test_fem_sponge.py --device_id 0 --steps 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description="Smoke-test RoboDojo's Isaac Lab FEM sponge mesh.")
parser.add_argument("--steps", type=int, default=30, help="Physics steps between load and reset")
parser.add_argument("--device_id", type=int, default=0, help="GPU index already visible to this process")
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.device_id < 0:
    parser.error("--device_id must be a non-negative integer")
args.device = f"cuda:{args.device_id}"
print(f"[fem-smoke] Isaac Sim device={args.device}")
app = AppLauncher(args).app


def main() -> None:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    from env.scene_manager.objects.deformable import DeformableObject

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FEM smoke test requires a visible CUDA GPU; pass --device cuda:<id>.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    sim = SimulationContext(sim_utils.SimulationCfg(device=str(device), dt=1.0 / 60.0))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sponge_config = {
        "size": [0.06, 0.05, 0.03],
        "physics": {
            "type": "deformable",
            "density": 250.0,
            "youngs_modulus": 120000.0,
            "poissons_ratio": 0.35,
            "elasticity_damping": 0.2,
            "dynamic_friction": 0.6,
        },
        "visual": {"color": [0.78, 0.70, 0.43, 1.0]},
    }
    sponge = DeformableObject(
        prim_path="/World/envs/env_0/Deformable/sponge/sponge_0_1",
        # Empty path verifies that the wrapper creates geometry procedurally.
        usd_path="",
        inst_config=sponge_config,
        env_origin=np.zeros(3, dtype=np.float32),
        default_pos=(0.0, 0.0, 0.3),
        default_ori=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )

    sim.reset()
    sponge.initialize()
    initial = sponge.get_nodal_positions_w().detach().clone()
    if initial.ndim != 2 or initial.shape[-1] != 3 or initial.shape[0] < 4:
        raise AssertionError(f"Unexpected FEM nodal position shape: {tuple(initial.shape)}")
    if not torch.isfinite(initial).all():
        raise AssertionError("FEM nodal positions contain NaN or infinity")

    for _ in range(args.steps):
        sim.step(render=False)
        sponge.update(sim.get_physics_dt())
    advanced = sponge.get_nodal_positions_w().detach().clone()
    sponge.reset()
    reset = sponge.get_nodal_positions_w().detach().clone()
    if not torch.isfinite(reset).all():
        raise AssertionError("FEM nodal positions became invalid after reset")
    reset_error = torch.max(torch.abs(reset - initial)).item()
    if reset_error > 1.0e-3:
        raise AssertionError(f"FEM reset error is too large: {reset_error:.6g} m")

    print(
        "FEM sponge smoke test passed: "
        f"nodes={initial.shape[0]}, "
        f"initial_z=[{initial[:, 2].min().item():.4f}, {initial[:, 2].max().item():.4f}], "
        f"motion={torch.linalg.vector_norm(advanced - initial).item():.6g} m, "
        f"reset_error={reset_error:.6g} m"
    )
    sponge.destroy()
    sim.stop()


try:
    main()
finally:
    app.close()
