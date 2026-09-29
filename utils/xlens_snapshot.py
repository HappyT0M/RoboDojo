"""Export one synchronized RoboDojo camera snapshot for XLens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    elif hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    image = _to_numpy(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"RGB image must have shape (H, W, >=3), got {image.shape}")
    Image.fromarray(image[..., :3].astype(np.uint8), mode="RGB").save(path)


def _first_capture(capture_data: dict[str, Any], key: str, env_index: int) -> np.ndarray:
    entries = capture_data.get(key)
    if entries is None:
        raise KeyError(key)
    if not entries:
        raise ValueError(f"Capture annotator '{key}' returned no data")
    return _to_numpy(entries[0 if env_index == 0 else env_index]["data"])


def populate_manifest_objects(
    manifest_path: str | Path,
    *,
    min_pixels: int = 50,
    background_labels: tuple[int, ...] = (0,),
) -> list[dict[str, Any]]:
    """Populate ``scene.json`` with generic objects found in segmentation masks."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_pixel_counts: dict[int, int] = {}

    for view in manifest.get("views", []):
        mask_path = Path(view["mask"])
        if not mask_path.is_absolute():
            mask_path = manifest_path.parent / mask_path
        mask = np.asarray(np.load(mask_path)).squeeze()
        labels, counts = np.unique(mask, return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist()):
            label = int(label)
            if label in background_labels:
                continue
            label_pixel_counts[label] = label_pixel_counts.get(label, 0) + int(count)

    objects = [
        {
            "name": f"instance_{label}",
            "category": "segmentation_instance",
            "label": label,
            "pixel_count": pixel_count,
        }
        for label, pixel_count in sorted(label_pixel_counts.items())
        if pixel_count >= min_pixels
    ]
    manifest["objects"] = objects
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return objects


def save_robodojo_snapshot(
    env: Any,
    output_dir: str | Path,
    *,
    task_name: str,
    objects: list[dict[str, Any]],
    env_id: int = 0,
    coordinate_frame: str = "world",
) -> str:
    """Capture RGB and segmentation data from a live RoboDojo environment.

    The environment must have RGB plus one of the following annotators enabled:
    ``instance_id_segmentation_fast``, ``instance_segmentation_fast`` or
    ``semantic_segmentation``.  The function intentionally only writes a
    snapshot; XLens inference is performed by ``run_xlens_geometry.py``.
    """
    if hasattr(env, "obs_manager") and hasattr(env.obs_manager, "render_for_capture"):
        env.obs_manager.render_for_capture()
    if not hasattr(env, "capture_manager") or not hasattr(env, "camera_manager"):
        raise ValueError("env must expose capture_manager and camera_manager")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = env.capture_manager.step(env_ids=[env_id])
    num_cams = int(env.camera_manager.num_cams)
    views = []

    mask_candidates = (
        ("instance_id_segmentation_fast", "instance_id"),
        ("instance_segmentation_fast", "instance"),
        ("semantic_segmentation", "semantic"),
    )
    for cam_id in range(num_cams):
        camera_name = str(env.camera_manager.camera_names[env_id][cam_id])
        cam_data = capture[cam_id]
        rgb_key = "rgb" if "rgb" in cam_data else "rgba" if "rgba" in cam_data else None
        if rgb_key is None:
            raise ValueError(f"Camera '{camera_name}' has no RGB annotator enabled")
        rgb = _first_capture(cam_data, rgb_key, 0)
        camera_dir = output_dir / camera_name
        camera_dir.mkdir(parents=True, exist_ok=True)
        _write_rgb(camera_dir / "rgb.png", rgb)

        mask_key = next((key for key, _ in mask_candidates if key in cam_data), None)
        if mask_key is None:
            enabled = ", ".join(key for key, _ in mask_candidates)
            raise ValueError(f"Camera '{camera_name}' has no segmentation annotator; enable one of {enabled}")
        mask_type = next(kind for key, kind in mask_candidates if key == mask_key)
        mask = _first_capture(cam_data, mask_key, 0)
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"Camera '{camera_name}' mask must be HxW, got {mask.shape}")
        np.save(camera_dir / "instance_mask.npy", mask.astype(np.int64, copy=False))

        K = np.asarray(env.camera_manager.get_camera_intrinsics(cam_id, env_id), dtype=np.float64)
        c2w = np.asarray(env.camera_manager.get_camera_extrinsics(cam_id, env_id), dtype=np.float64)
        views.append(
            {
                "name": camera_name,
                "rgb": f"{camera_name}/rgb.png",
                "mask": f"{camera_name}/instance_mask.npy",
                "mask_type": mask_type,
                "K": K.tolist(),
                "c2w": c2w.tolist(),
            }
        )

    manifest = {
        "task_name": task_name,
        "env_id": int(env_id),
        "coordinate_frame": coordinate_frame,
        "views": views,
        "objects": objects,
    }
    manifest_path = output_dir / "scene.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(manifest_path)
