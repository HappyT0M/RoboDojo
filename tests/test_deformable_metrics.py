import numpy as np
import pytest

from utils.deformable_metrics import (
    ConsecutiveConditionCounter,
    compute_deformable_containment,
    validate_deformable_config,
)


def test_containment_uses_bowl_local_frame():
    angle = np.pi / 4
    bowl_quat_w = np.array([np.cos(angle), 0.0, 0.0, np.sin(angle)])
    points_w = np.array([[0.0, 0.03, 0.05], [0.0, 0.02, 0.05]])

    metrics = compute_deformable_containment(
        points_w=points_w,
        bowl_pos_w=np.zeros(3),
        bowl_quat_w=bowl_quat_w,
        interior_bounds=((-0.04, -0.02, 0.01), (0.04, 0.02, 0.10)),
        rim_z=0.10,
    )

    assert metrics.center_inside
    assert metrics.contained_fraction == 1.0


def test_nodes_above_rim_or_outside_footprint_do_not_count_as_contained():
    metrics = compute_deformable_containment(
        points_w=np.array([[0.0, 0.0, 0.05], [0.06, 0.0, 0.05], [0.0, 0.0, 0.11]]),
        bowl_pos_w=np.zeros(3),
        bowl_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
        interior_bounds=((-0.04, -0.04, 0.01), (0.04, 0.04, 0.10)),
        rim_z=0.10,
    )

    assert metrics.center_inside
    assert metrics.contained_count == 1
    assert metrics.contained_fraction == pytest.approx(1 / 3)


def test_containment_rejects_empty_or_invalid_inputs():
    args = {
        "bowl_pos_w": np.zeros(3),
        "bowl_quat_w": np.array([1.0, 0.0, 0.0, 0.0]),
        "interior_bounds": ((-0.04, -0.04, 0.01), (0.04, 0.04, 0.10)),
        "rim_z": 0.10,
    }

    with pytest.raises(ValueError, match="at least one"):
        compute_deformable_containment(points_w=np.empty((0, 3)), **args)

    with pytest.raises(ValueError, match="quaternion"):
        compute_deformable_containment(points_w=np.zeros((1, 3)), bowl_quat_w=np.zeros(4), **{k: v for k, v in args.items() if k != "bowl_quat_w"})


def test_consecutive_condition_counter_resets_after_a_miss():
    counter = ConsecutiveConditionCounter(required_steps=3)

    assert counter.update(True) is False
    assert counter.update(True) is False
    assert counter.update(False) is False
    assert counter.count == 0
    assert counter.update(True) is False
    assert counter.update(True) is False
    assert counter.update(True) is True


def test_consecutive_condition_counter_requires_positive_window():
    with pytest.raises(ValueError, match="positive"):
        ConsecutiveConditionCounter(required_steps=0)


def test_deformable_config_normalizes_size_and_material_values():
    settings = validate_deformable_config(
        {
            "size": [0.06, 0.06, 0.03],
            "physics": {
                "density": 250.0,
                "youngs_modulus": 120000.0,
                "poissons_ratio": 0.35,
                "elasticity_damping": 0.2,
            },
        }
    )

    assert settings["size"] == (0.06, 0.06, 0.03)
    assert settings["physics"]["density"] == 250.0
    assert settings["physics"]["poissons_ratio"] == 0.35


@pytest.mark.parametrize(
    "config",
    [
        {"size": [0.06, 0.0, 0.03]},
        {"size": [0.06, 0.06, 0.03], "physics": {"density": 0.0}},
        {"size": [0.06, 0.06, 0.03], "physics": {"youngs_modulus": -1.0}},
        {"size": [0.06, 0.06, 0.03], "physics": {"poissons_ratio": 0.5}},
    ],
)
def test_deformable_config_rejects_invalid_dimensions_or_material(config):
    with pytest.raises(ValueError):
        validate_deformable_config(config)
