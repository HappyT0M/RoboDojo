import json

import numpy as np
import pytest

from utils.teleop_controls import apply_translation_keys, parse_wind, select_gripper_command, teleop_record_row


def test_translation_keys_move_xyz_without_changing_orientation():
    pose = [0.1, -0.2, 1.0, 1.0, 0.0, 0.0, 0.0]
    result = apply_translation_keys(
        pose,
        {"W", "D", "R"},
        step_m=0.01,
        bounds=((-1, 1), (-1, 1), (0, 2)),
    )

    np.testing.assert_allclose(result, [0.11, -0.19, 1.01, 1, 0, 0, 0])


def test_opposing_translation_keys_cancel_and_bounds_clip():
    result = apply_translation_keys(
        [0.99, 0, 1, 1, 0, 0, 0],
        {"W", "S", "D"},
        step_m=0.05,
        bounds=((-1, 1), (-1, 1), (0, 2)),
    )

    np.testing.assert_allclose(result[:3], [0.99, 0.05, 1])


def test_gripper_key_selects_open_or_closed_fraction():
    assert select_gripper_command({"C"}, current=1.0) == 0.0
    assert select_gripper_command({"O"}, current=0.0) == 1.0
    assert select_gripper_command(set(), current=0.4) == 0.4


def test_parse_wind_requires_finite_three_vector():
    assert parse_wind((0, 0.5, 0)) == (0.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        parse_wind((0, float("nan"), 0))
    with pytest.raises(ValueError):
        parse_wind((1, 2))


def test_record_row_is_json_serializable_and_contains_replay_context():
    row = teleop_record_row(
        step=3,
        held_keys={"W"},
        arm="left",
        target_pose=[0, 0, 1, 1, 0, 0, 0],
        gripper=1.0,
        wind=(0, 0, 0),
        task="put_rope_ball_in_basket",
        seed=0,
    )

    decoded = json.loads(json.dumps(row))
    assert decoded["step"] == 3
    assert decoded["arm"] == "left"
    assert decoded["held_keys"] == ["W"]
    assert decoded["wind"] == [0, 0, 0]
