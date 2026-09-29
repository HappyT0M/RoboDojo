import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROBODOJO_ROOT = Path(__file__).resolve().parents[1]
if str(ROBODOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBODOJO_ROOT))

from utils.xlens_geometry import load_scene_manifest  # noqa: E402


def test_manifest_resolves_relative_rgb_mask_and_calibration_paths(tmp_path):
    (tmp_path / "rgb.npy").write_bytes(b"")
    (tmp_path / "mask.npy").write_bytes(b"")
    (tmp_path / "rays.npy").write_bytes(b"")
    manifest_path = tmp_path / "scene.json"
    manifest_path.write_text(
        json.dumps(
            {
                "task_name": "build_tower",
                "coordinate_frame": "robot_base",
                "views": [
                    {
                        "name": "cam_head",
                        "rgb": "rgb.npy",
                        "mask": "mask.npy",
                        "dcam": "rays.npy",
                        "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "c2w": np.eye(4).tolist(),
                    }
                ],
                "objects": [{"name": "target", "category": "cube", "label": 7}],
            }
        ),
        encoding="utf-8",
    )

    result = load_scene_manifest(manifest_path)

    assert result["views"][0]["rgb"] == str(tmp_path / "rgb.npy")
    assert result["views"][0]["mask"] == str(tmp_path / "mask.npy")
    assert result["views"][0]["dcam"] == str(tmp_path / "rays.npy")
    assert result["views"][0]["c2w"] == np.eye(4).tolist()


def test_manifest_requires_camera_pose_and_calibration(tmp_path):
    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text(
        json.dumps(
            {
                "views": [{"name": "cam_head", "rgb": "rgb.npy"}],
                "objects": [{"name": "target", "label": 1}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="c2w"):
        load_scene_manifest(manifest_path)

