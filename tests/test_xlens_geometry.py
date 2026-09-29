import sys
from pathlib import Path

import numpy as np


ROBODOJO_ROOT = Path(__file__).resolve().parents[1]
if str(ROBODOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBODOJO_ROOT))

from utils.xlens_geometry import fit_oriented_bbox, transform_points  # noqa: E402


def make_grid_box(center=(1.0, 2.0, 3.0), size=(0.4, 0.2, 0.1)):
    axes = [np.linspace(-length / 2, length / 2, count) for length, count in zip(size, (9, 7, 5))]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid + np.asarray(center, dtype=np.float32)


def test_fit_oriented_bbox_recovers_metric_dimensions_and_center():
    points = make_grid_box()

    result = fit_oriented_bbox(points, min_points=20)

    np.testing.assert_allclose(result["center"], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(result["size"], [0.4, 0.2, 0.1], atol=1e-6)
    assert result["status"] == "ok"
    assert result["num_points"] == len(points)


def test_transform_points_applies_camera_to_world_translation():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [0.5, -0.25, 1.0]

    transformed = transform_points(points, c2w)

    np.testing.assert_allclose(transformed, points + [0.5, -0.25, 1.0])


def test_fit_oriented_bbox_reports_insufficient_points_without_crashing():
    result = fit_oriented_bbox(np.zeros((3, 3), dtype=np.float32), min_points=20)

    assert result["status"] == "insufficient_points"
    assert result["num_points"] == 3

