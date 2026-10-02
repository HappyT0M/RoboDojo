import math
from pathlib import Path

import yaml

from utils.rope_task_metrics import (
    ConsecutiveSuccess,
    ball_in_directed_far_zone,
    crosses_ring_aperture,
    gripper_holds_rope,
    point_inside_oriented_box,
    polyline_wraps_post,
    resolve_physx_gpu_override,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "put_rope_ball_in_basket",
    "pass_rope_through_ring",
    "route_rope_around_posts",
)


def test_three_rope_tasks_are_registered_as_cpu_only_articulations():
    task_index = yaml.safe_load((ROOT / "task/RoboDojo/config/_task.yml").read_text(encoding="utf-8"))
    for task in TASKS:
        assert task in task_index["tasks"]
        assert task_index["tasks"][task]["physics_device"] == "cpu"
        assert task_index["tasks"][task]["gpu_dynamics_enabled"] is False

        task_config = yaml.safe_load(
            (ROOT / f"task/RoboDojo/config/{task}.yml").read_text(encoding="utf-8")
        )
        assert "Articulation" in task_config
        assert "rope_chain" in task_config["Articulation"][0]["category"][0]["name"]
        assert "Deformable" not in task_config
        assert "Garment" not in task_config
        assert "Fluid" not in task_config

        layout_path = ROOT / f"Assets/Eval_Layout/RoboDojo/arx_x5/0/{task}_0.json"
        assert layout_path.is_file()


def test_cpu_task_settings_override_sim_device_and_keep_default_gpu_behavior():
    from utils.pipeline_utils import apply_task_physics_settings

    cfg = {"sim": {"device": "cuda:0", "gpu_dynamics_enabled": True}}
    apply_task_physics_settings(cfg, {"physics_device": "cpu", "gpu_dynamics_enabled": False})
    assert cfg["sim"]["device"] == "cpu"
    assert cfg["sim"]["gpu_dynamics_enabled"] is False

    legacy_cfg = {"sim": {"device": "cpu"}}
    apply_task_physics_settings(legacy_cfg, {})
    assert legacy_cfg["sim"]["device"] == "cpu"
    assert resolve_physx_gpu_override(legacy_cfg["sim"]) == 1
    assert resolve_physx_gpu_override(cfg["sim"]) == 0

    from utils.pipeline_utils import process_config

    full_cfg = {
        "sim": {"device": "cuda:0"},
        "eval_cfg": {},
        "scene": {},
        "camera": {},
        "robot": {},
    }
    configured, _ = process_config(full_cfg, TASKS[0])
    assert configured["sim"]["device"] == "cpu"
    assert configured["sim"]["gpu_dynamics_enabled"] is False


def test_generated_rope_usd_assets_and_seed_layouts_are_present():
    rope_usd = ROOT / "Assets/Object/RoboDojo/Articulation/rope_chain/00000/object.usd"
    assert rope_usd.is_file()
    contents = rope_usd.read_text(encoding="utf-8")
    assert contents.count('def Capsule "collision"') == 12
    assert contents.count("def PhysicsRevoluteJoint") == 11
    assert 'def PhysicsFixedJoint "joint_ball"' in contents
    assert 'bool physxArticulation:enabledSelfCollisions = false' in contents

    for category in ("rope_basket", "rope_ring", "rope_post", "rope_slot"):
        asset_dir = ROOT / f"Assets/Object/RoboDojo/Geometry/{category}/00000"
        assert (asset_dir / "object.usd").is_file()
        metadata = yaml.safe_load((asset_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["geometry"]["vertices"] > 0
    ring_metadata = yaml.safe_load(
        (ROOT / "Assets/Object/RoboDojo/Geometry/rope_ring/00000/metadata.json").read_text(encoding="utf-8")
    )
    assert ring_metadata["geometry"]["aligned_bbox"]["extents"][1] < 0.02


def test_basket_containment_respects_box_pose_and_bounds():
    assert point_inside_oriented_box(
        point=(0.05, -0.02, 0.04),
        box_position=(0.0, 0.0, 0.0),
        box_quaternion_wxyz=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        bounds=((-0.1, 0.1), (-0.05, 0.05), (0.0, 0.1)),
    )
    assert not point_inside_oriented_box(
        point=(0.2, 0.0, 0.04),
        box_position=(0.0, 0.0, 0.0),
        box_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        bounds=((-0.1, 0.1), (-0.05, 0.05), (0.0, 0.1)),
    )


def test_ring_crossing_requires_correct_direction_and_clearance():
    assert crosses_ring_aperture(
        previous=(-0.1, 0.0, 0.0),
        current=(0.1, 0.0, 0.0),
        center=(0.0, 0.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        inner_radius=0.08,
        margin=0.01,
    )
    assert not crosses_ring_aperture(
        previous=(-0.1, 0.0, 0.1),
        current=(0.1, 0.0, 0.1),
        center=(0.0, 0.0, 0.0),
        normal=(1.0, 0.0, 0.0),
        inner_radius=0.08,
        margin=0.01,
    )


def test_ring_goal_requires_ball_inside_far_side_target_zone():
    assert ball_in_directed_far_zone(
        position=(0.02, 0.16, 0.03),
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        min_distance=0.10,
        max_distance=0.22,
        tangential_radius=0.08,
    )
    assert not ball_in_directed_far_zone(
        position=(0.10, 0.16, 0.0),
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        min_distance=0.10,
        max_distance=0.22,
        tangential_radius=0.08,
    )
    assert not ball_in_directed_far_zone(
        position=(0.0, -0.16, 0.0),
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        min_distance=0.10,
        max_distance=0.22,
        tangential_radius=0.08,
    )


def test_post_wrap_requires_chain_arc_close_to_post():
    angles = [math.radians(value) for value in range(-100, 101, 20)]
    wrapped = [(0.04 * math.cos(angle), 0.04 * math.sin(angle), 0.0) for angle in angles]
    straight = [(-0.1 + index * 0.02, 0.04, 0.0) for index in range(11)]
    assert polyline_wraps_post(wrapped, (0.0, 0.0), post_radius=0.03)
    assert not polyline_wraps_post(straight, (0.0, 0.0), post_radius=0.03)


def test_t1_requires_closed_gripper_near_a_lifted_rope_link():
    assert gripper_holds_rope(
        gripper_position=(0.0, 0.0, 0.86),
        rope_points=((0.0, 0.0, 0.86), (0.1, 0.0, 0.80)),
        gripper_open_fraction=0.1,
        table_height=0.765,
    )
    assert not gripper_holds_rope(
        gripper_position=(0.0, 0.0, 0.86),
        rope_points=((0.0, 0.0, 0.86), (0.1, 0.0, 0.80)),
        gripper_open_fraction=0.9,
        table_height=0.765,
    )
    assert not gripper_holds_rope(
        gripper_position=(0.0, 0.0, 0.78),
        rope_points=((0.0, 0.0, 0.78),),
        gripper_open_fraction=0.1,
        table_height=0.765,
    )


def test_success_window_requires_consecutive_stable_steps():
    success = ConsecutiveSuccess(required_steps=3)
    assert [success.update(value) for value in (True, False, True, True, True)] == [False, False, False, False, True]
    success.reset()
    assert success.count == 0
