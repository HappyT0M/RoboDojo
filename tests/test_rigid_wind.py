import numpy as np
import pytest
import torch

from utils.rigid_wind import (
    RigidBodySpec,
    RigidWindController,
    box_projected_area,
    capsule_projected_area,
    compute_drag_forces,
    install_sim_step_wind_hook,
    is_wind_body_candidate,
    sphere_projected_area,
)


def test_drag_force_follows_relative_wind_and_scales_quadratically():
    force = compute_drag_forces(
        wind=(2.0, 0.0, 0.0),
        velocities=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        areas=(1.0, 1.0),
        air_density=1.0,
        drag_coefficient=1.0,
    )

    np.testing.assert_allclose(force, ((2.0, 0.0, 0.0), (0.5, 0.0, 0.0)))


def test_drag_force_is_zero_without_relative_wind_or_ambient_wind():
    no_relative_wind = compute_drag_forces(
        wind=(1.0, -2.0, 0.5),
        velocities=((1.0, -2.0, 0.5),),
        areas=(1.0,),
        air_density=1.225,
        drag_coefficient=1.0,
    )
    no_ambient_wind = compute_drag_forces(
        wind=(0.0, 0.0, 0.0),
        velocities=((0.0, 0.0, 0.0),),
        areas=(0.5,),
        air_density=1.225,
        drag_coefficient=1.0,
    )

    np.testing.assert_array_equal(no_relative_wind, ((0.0, 0.0, 0.0),))
    np.testing.assert_array_equal(no_ambient_wind, ((0.0, 0.0, 0.0),))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wind": (1.0, 0.0), "velocities": ((0.0, 0.0, 0.0),), "areas": (1.0,)},
        {"wind": (1.0, 0.0, 0.0), "velocities": ((0.0, 0.0, 0.0),), "areas": (-1.0,)},
        {"wind": (1.0, 0.0, 0.0), "velocities": ((0.0, 0.0, 0.0),), "areas": (float("nan"),)},
    ],
)
def test_drag_force_rejects_invalid_vectors_and_areas(kwargs):
    with pytest.raises(ValueError):
        compute_drag_forces(air_density=1.225, drag_coefficient=1.0, **kwargs)


def test_box_projected_area_uses_world_orientation():
    extents = (2.0, 3.0, 4.0)
    identity = np.eye(3)
    rotated_90deg_z = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))

    assert box_projected_area(extents, identity, (1.0, 0.0, 0.0)) == pytest.approx(12.0)
    assert box_projected_area(extents, identity, (0.0, 1.0, 0.0)) == pytest.approx(8.0)
    assert box_projected_area(extents, rotated_90deg_z, (1.0, 0.0, 0.0)) == pytest.approx(8.0)


def test_capsule_projected_area_depends_on_capsule_axis():
    along_capsule = capsule_projected_area(radius=1.0, cylinder_length=2.0, axis=(1.0, 0.0, 0.0), wind=(1.0, 0.0, 0.0))
    across_capsule = capsule_projected_area(radius=1.0, cylinder_length=2.0, axis=(1.0, 0.0, 0.0), wind=(0.0, 1.0, 0.0))

    assert along_capsule == pytest.approx(np.pi)
    assert across_capsule == pytest.approx(4.0 + np.pi)
    assert sphere_projected_area(radius=0.5) == pytest.approx(np.pi * 0.25)


@pytest.mark.parametrize(
    "kind,fixed_base,is_robot,expected",
    [
        ("rigid", False, False, True),
        ("dynamic", False, False, True),
        ("articulation", False, False, True),
        ("articulation", True, False, False),
        ("articulation", False, True, False),
        ("geometry", False, False, False),
        ("garment", False, False, False),
        ("fluid", False, False, False),
    ],
)
def test_wind_candidate_filter_excludes_robot_and_non_rigid_categories(kind, fixed_base, is_robot, expected):
    assert is_wind_body_candidate(kind, fixed_base=fixed_base, is_robot=is_robot) is expected


class _FakeRigidView:
    def __init__(self, velocities):
        self.velocities = np.asarray(velocities, dtype=np.float64)
        self.orientations = np.tile((1.0, 0.0, 0.0, 0.0), (len(self.velocities), 1))
        self.applied = []

    def get_linear_velocities(self):
        return self.velocities

    def get_world_poses(self):
        return np.zeros_like(self.velocities), self.orientations

    def apply_forces(self, forces, is_global=True):
        self.applied.append((forces, is_global))


def test_controller_reads_body_state_and_applies_global_forces():
    view = _FakeRigidView(((0.0, 0.0, 0.0),))
    controller = RigidWindController(
        view=view,
        bodies=(RigidBodySpec("/World/rope/segment_00", "segment_00", (2.0, 3.0, 4.0)),),
        wind=(1.0, 0.0, 0.0),
        air_density=1.0,
        drag_coefficient=1.0,
        force_scale=1.0,
    )

    diagnostics = controller.step()

    assert len(view.applied) == 1
    assert isinstance(view.applied[0][0], torch.Tensor)
    assert view.applied[0][0].device.type == "cpu"
    np.testing.assert_allclose(view.applied[0][0].numpy(), ((6.0, 0.0, 0.0),))
    assert view.applied[0][1] is True
    assert diagnostics["body_count"] == 1
    assert diagnostics["bodies"][0]["prim_path"] == "/World/rope/segment_00"


def test_controller_skips_force_api_when_wind_is_zero():
    view = _FakeRigidView(((0.0, 0.0, 0.0),))
    controller = RigidWindController(
        view=view,
        bodies=(RigidBodySpec("/World/rope/ball", "ball", (1.0, 1.0, 1.0)),),
        wind=(0.0, 0.0, 0.0),
    )

    diagnostics = controller.step()

    assert view.applied == []
    assert diagnostics["applied_body_count"] == 0


def test_sim_step_hook_applies_wind_before_physics_step_and_returns_original():
    events = []

    class _Env:
        def sim_step(self, render=True):
            events.append(("physics", render))
            return "step-result"

    class _Controller:
        def step(self):
            events.append(("wind", None))

    env = _Env()
    original = env.sim_step
    install_sim_step_wind_hook(env, _Controller())

    assert env.sim_step(render=False) == "step-result"
    assert events == [("wind", None), ("physics", False)]
    assert original(render=True) == "step-result"
