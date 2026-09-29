import json
import sys
from pathlib import Path

import numpy as np


ROBODOJO_ROOT = Path(__file__).resolve().parents[1]
if str(ROBODOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBODOJO_ROOT))

from utils.xlens_snapshot import (  # noqa: E402
    populate_manifest_objects,
    save_robodojo_snapshot,
)


class FakeObservationManager:
    def __init__(self):
        self.render_called = False

    def render_for_capture(self):
        self.render_called = True


class FakeCameraManager:
    num_cams = 1
    camera_names = [["cam_head"]]

    def get_camera_intrinsics(self, cam_id, env_id):
        return np.eye(3)

    def get_camera_extrinsics(self, cam_id, env_id):
        return np.eye(4)


class FakeCaptureManager:
    def __init__(self, rgb, mask):
        self.rgb = rgb
        self.mask = mask

    def step(self, env_ids):
        return [
            {
                "rgb": [{"data": self.rgb}],
                "instance_id_segmentation_fast": [{"data": self.mask}],
            }
        ]


class FakeEnv:
    def __init__(self, rgb, mask):
        self.obs_manager = FakeObservationManager()
        self.camera_manager = FakeCameraManager()
        self.capture_manager = FakeCaptureManager(rgb, mask)


def test_save_robodojo_snapshot_exports_manifest_and_masks(tmp_path):
    rgb = np.zeros((4, 5, 4), dtype=np.uint8)
    rgb[..., :3] = [10, 20, 30]
    mask = np.zeros((4, 5), dtype=np.uint32)
    mask[1:3, 1:4] = 7
    env = FakeEnv(rgb, mask)

    manifest_path = save_robodojo_snapshot(
        env,
        tmp_path,
        task_name="build_tower",
        objects=[{"name": "target", "category": "cube", "label": 7}],
    )

    assert env.obs_manager.render_called is True
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["task_name"] == "build_tower"
    assert manifest["views"][0]["name"] == "cam_head"
    assert Path(tmp_path / "cam_head" / "rgb.png").exists()
    assert Path(tmp_path / "cam_head" / "instance_mask.npy").exists()
    assert manifest["views"][0]["mask"] == "cam_head/instance_mask.npy"


def test_populate_manifest_objects_discovers_visible_instance_labels(tmp_path):
    manifest_path = tmp_path / "scene.json"
    mask_path = tmp_path / "cam_head" / "instance_mask.npy"
    mask_path.parent.mkdir()
    np.save(mask_path, np.array([[0, 3, 3], [0, 5, 5]], dtype=np.uint32))
    manifest_path.write_text(
        json.dumps(
            {
                "views": [{"mask": "cam_head/instance_mask.npy"}],
                "objects": [],
            }
        ),
        encoding="utf-8",
    )

    objects = populate_manifest_objects(manifest_path, min_pixels=2)

    assert objects == [
        {"name": "instance_3", "category": "segmentation_instance", "label": 3, "pixel_count": 2},
        {"name": "instance_5", "category": "segmentation_instance", "label": 5, "pixel_count": 2},
    ]
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["objects"] == objects
