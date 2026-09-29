"""XLens-backed geometry estimation for RoboDojo scene snapshots.

The module deliberately keeps the object-geometry math independent from Isaac Sim.
The simulator only needs to export RGB images, masks, camera intrinsics and camera
to world poses into the manifest format documented in
``XLens_RoboDojo尺寸估计技术文档.md``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_array(path: str | Path) -> np.ndarray:
    """Load a NumPy or image array from disk."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if "data" in archive:
            return archive["data"]
        if len(archive.files) != 1:
            raise ValueError(f"NPZ file must contain one array or a 'data' array: {path}")
        return archive[archive.files[0]]
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        from PIL import Image

        return np.asarray(Image.open(path))
    raise ValueError(f"Unsupported array format: {path}")


def _aligned_image_size(
    height: int,
    width: int,
    requested: tuple[int, int] | list[int] | None = None,
    *,
    patch_size: int = 14,
) -> tuple[int, int]:
    """Return an XLens-compatible ``(H, W)`` resolution.

    XLens uses 14x14 patch tokens for both the RGB and ray-map branches.  If
    no target is supplied, use the largest resolution not exceeding the input
    size whose dimensions are divisible by 14.
    """
    if requested is None:
        target_h = height - height % patch_size
        target_w = width - width % patch_size
    else:
        if len(requested) != 2:
            raise ValueError("image_size must contain [height, width]")
        target_h, target_w = (int(requested[0]), int(requested[1]))
        if target_h <= 0 or target_w <= 0:
            raise ValueError("image_size values must be positive")
        if target_h % patch_size != 0 or target_w % patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size={patch_size}, got ({target_h}, {target_w})"
            )
    if target_h < patch_size or target_w < patch_size:
        raise ValueError(f"Input resolution ({height}, {width}) is too small for patch_size={patch_size}")
    return target_h, target_w


def _resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    mask = np.asarray(mask).squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Segmentation mask must be 2D, got {mask.shape}")
    image = Image.fromarray(mask.astype(np.int32, copy=False), mode="I")
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST), dtype=np.int64)


def _resolve_path(base_dir: Path, value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _validate_matrix(value: Any, shape: tuple[int, int], name: str) -> list[list[float]]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    return matrix.tolist()


def load_scene_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a RoboDojo XLens snapshot manifest.

    Paths inside ``views`` are resolved relative to the JSON manifest.  The
    returned object remains JSON-serializable.
    """
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("Manifest must contain a non-empty 'views' list")
    objects = manifest.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Manifest 'objects' must be a list")

    for index, view in enumerate(views):
        if not isinstance(view, dict):
            raise ValueError(f"View {index} must be an object")
        if not view.get("rgb"):
            raise ValueError(f"View {index} is missing 'rgb'")
        if "c2w" not in view:
            raise ValueError(f"View {index} is missing 'c2w'")
        if "K" not in view and "dcam" not in view and "lut" not in view:
            raise ValueError(f"View {index} needs one of 'K', 'dcam' or 'lut'")

        view["rgb"] = _resolve_path(manifest_path.parent, view["rgb"])
        view["mask"] = _resolve_path(manifest_path.parent, view.get("mask"))
        view["dcam"] = _resolve_path(manifest_path.parent, view.get("dcam"))
        view["lut"] = _resolve_path(manifest_path.parent, view.get("lut"))
        view["c2w"] = _validate_matrix(view["c2w"], (4, 4), f"views[{index}].c2w")
        if "K" in view:
            view["K"] = _validate_matrix(view["K"], (3, 3), f"views[{index}].K")

    for index, obj in enumerate(objects):
        if not isinstance(obj, dict) or not obj.get("name"):
            raise ValueError(f"Object {index} must contain a non-empty 'name'")
        if "label" not in obj:
            raise ValueError(f"Object {index} is missing 'label'")

    manifest["manifest_path"] = str(manifest_path)
    return manifest


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to an ``(N, 3)`` point array."""
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    return (points @ transform[:3, :3].T) + transform[:3, 3]


def _empty_bbox(status: str, count: int = 0) -> dict[str, Any]:
    return {
        "status": status,
        "center": None,
        "size": None,
        "orientation": None,
        "aabb_min": None,
        "aabb_max": None,
        "num_points": int(count),
        "num_views": 0,
        "mean_depth_confidence": None,
        "confidence": 0.0,
    }


def fit_oriented_bbox(
    points: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    min_points: int = 20,
) -> dict[str, Any]:
    """Fit a PCA-oriented bounding box and return dimensions in metres.

    Dimensions are sorted from largest to smallest for a stable downstream
    representation.  This is an observed point-cloud box; it does not infer
    hidden geometry behind severe occlusion.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)[finite]
    if len(points) < min_points:
        return _empty_bbox("insufficient_points", len(points))

    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = np.cov(centered, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1

    projected = centered @ axes
    lower = projected.min(axis=0)
    upper = projected.max(axis=0)
    raw_size = upper - lower
    size_order = np.argsort(raw_size)[::-1]
    raw_size = raw_size[size_order]
    axes = axes[:, size_order]
    local_center = (lower + upper) / 2.0
    box_center = centroid + (eigenvectors[:, order] @ local_center)
    aabb_min = points.min(axis=0)
    aabb_max = points.max(axis=0)

    mean_confidence = float(np.clip(np.mean(confidence), 0.0, 1.0)) if confidence is not None else 1.0
    density_score = float(np.clip(len(points) / 1000.0, 0.0, 1.0))
    result = {
        "status": "ok",
        "center": box_center.tolist(),
        "size": raw_size.tolist(),
        "orientation": axes.tolist(),
        "aabb_min": aabb_min.tolist(),
        "aabb_max": aabb_max.tolist(),
        "num_points": int(len(points)),
        "num_views": 0,
        "mean_depth_confidence": mean_confidence,
        "confidence": float(0.5 * mean_confidence + 0.5 * density_score),
    }
    return result


def _label_mask(mask: np.ndarray, label: int | list[int] | tuple[int, ...]) -> np.ndarray:
    mask = np.asarray(mask)
    labels = label if isinstance(label, (list, tuple)) else [label]
    return np.isin(mask, np.asarray(labels))


def estimate_object_from_views(
    points_by_view: list[np.ndarray],
    masks_by_view: list[np.ndarray],
    confidences_by_view: list[np.ndarray] | None,
    label: int | list[int] | tuple[int, ...],
    *,
    min_points: int = 20,
) -> dict[str, Any]:
    """Extract an object point cloud from per-view masks and fit its 3D box."""
    if len(points_by_view) != len(masks_by_view):
        raise ValueError("points_by_view and masks_by_view must have equal lengths")
    if confidences_by_view is not None and len(confidences_by_view) != len(points_by_view):
        raise ValueError("confidences_by_view must match the number of views")

    selected_points: list[np.ndarray] = []
    selected_confidences: list[np.ndarray] = []
    visible_views = 0
    for view_index, (points, mask) in enumerate(zip(points_by_view, masks_by_view)):
        points = np.asarray(points)
        mask = np.asarray(mask)
        if points.shape[:2] != mask.shape[:2]:
            raise ValueError(f"View {view_index} point/mask spatial shapes do not match")
        selected = _label_mask(mask, label)
        if np.any(selected):
            visible_views += 1
            selected_points.append(points[selected])
            if confidences_by_view is not None:
                selected_confidences.append(np.asarray(confidences_by_view[view_index])[selected])

    if not selected_points:
        result = _empty_bbox("label_not_visible")
        result["mask_label"] = label
        return result

    points = np.concatenate(selected_points, axis=0)
    confidences = np.concatenate(selected_confidences, axis=0) if selected_confidences else None
    result = fit_oriented_bbox(points, confidences, min_points=min_points)
    result["num_views"] = visible_views
    result["mask_label"] = label
    return result


def _load_xlens_components(xlens_root: str | Path):
    xlens_root = str(Path(xlens_root).resolve())
    if xlens_root not in sys.path:
        sys.path.insert(0, xlens_root)
    try:
        from xlens.inference import XLensInference
        from xlens.inference.geometry import fuse_point_cloud
        from xlens.inference.geometry import unproject_to_world
        from xlens.inference.preprocess import assemble_batch, load_fisheye_lut, pinhole_d_cam, resize_d_cam
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "XLens dependencies are unavailable. Install XLens requirements before running scene inference."
        ) from exc

    return (
        XLensInference,
        assemble_batch,
        load_fisheye_lut,
        pinhole_d_cam,
        resize_d_cam,
        fuse_point_cloud,
        unproject_to_world,
    )


def run_xlens_scene(
    manifest: dict[str, Any],
    *,
    checkpoint: str | Path,
    xlens_root: str | Path,
    config: str | Path | None = None,
    device: str = "cuda",
    image_size: tuple[int, int] | list[int] | None = None,
    max_depth: float | None = None,
    conf_drop_pct: float = 10.0,
    fov_max: float | None = 85.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run XLens and estimate all objects declared in a manifest.

    Returns ``(geometry_summary, runtime_artifacts)``.  The latter contains
    NumPy arrays and point clouds for the CLI to save.
    """
    (
        XLensInference,
        assemble_batch,
        load_fisheye_lut,
        pinhole_d_cam,
        resize_d_cam,
        fuse_point_cloud,
        unproject_to_world,
    ) = _load_xlens_components(xlens_root)
    from PIL import Image

    # XLensNet uses a patch size of 14.  Keep an explicitly configured target
    # if provided; otherwise align the native RoboDojo resolution downward.
    requested_size = image_size if image_size is not None else manifest.get("image_size")
    target_hw: tuple[int, int] | None = None

    images: list[np.ndarray] = []
    d_cams: list[np.ndarray] = []
    cam_types: list[int] = []
    world_c2w: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for index, view in enumerate(manifest["views"]):
        image = np.asarray(Image.open(view["rgb"]).convert("RGB"))
        source_height, source_width = image.shape[:2]
        if target_hw is None:
            target_hw = _aligned_image_size(source_height, source_width, requested_size)
        target_height, target_width = target_hw
        if (source_height, source_width) != target_hw:
            image = _resize_rgb(image, target_height, target_width)
        height, width = image.shape[:2]
        images.append(image)
        if view.get("dcam"):
            d_cam = load_array(view["dcam"]).astype(np.float32)
            cam_type = int(view.get("cam_type", 1))
            if d_cam.shape != (source_height, source_width, 3):
                raise ValueError(
                    f"View {index} dcam shape must be {(source_height, source_width, 3)}, got {d_cam.shape}"
                )
            if (source_height, source_width) != target_hw:
                d_cam = resize_d_cam(d_cam, target_height, target_width)
        elif view.get("lut"):
            d_cam = load_fisheye_lut(view["lut"], source_height, source_width)
            cam_type = 0
            if (source_height, source_width) != target_hw:
                d_cam = resize_d_cam(d_cam, target_height, target_width)
        elif view.get("K"):
            K = np.asarray(view["K"], dtype=np.float32).copy()
            if (source_height, source_width) != target_hw:
                scale_x = target_width / source_width
                scale_y = target_height / source_height
                K[0, 0] *= scale_x
                K[0, 2] *= scale_x
                K[1, 1] *= scale_y
                K[1, 2] *= scale_y
            d_cam = pinhole_d_cam(K, height, width)
            cam_type = 1
        else:
            raise ValueError(f"View {index} has no usable camera calibration")
        if d_cam.shape != (height, width, 3):
            raise ValueError(f"View {index} calibration shape must be {(height, width, 3)}, got {d_cam.shape}")
        d_cams.append(d_cam)
        cam_types.append(cam_type)
        world_c2w.append(np.asarray(view["c2w"], dtype=np.float64))
        if not view.get("mask"):
            raise ValueError(f"View {index} is missing a segmentation mask")
        mask = load_array(view["mask"])
        mask = np.asarray(mask).squeeze()
        if mask.shape == (source_height, source_width) and (source_height, source_width) != target_hw:
            mask = _resize_mask(mask, target_height, target_width)
        elif mask.shape != target_hw:
            raise ValueError(f"View {index} mask shape must be {target_hw}, got {mask.shape}")
        masks.append(mask)

    world_c2w_array = np.stack(world_c2w, axis=0)
    view0_from_world = np.linalg.inv(world_c2w_array[0])
    relative_c2w = np.stack([view0_from_world @ pose for pose in world_c2w_array], axis=0)
    batch = assemble_batch(images, d_cams, cam_types, c2w=relative_c2w, device=device)
    os.environ.setdefault("DINOV2_CKPT_DIR", str((Path(xlens_root).resolve() / "checkpoints")))
    model = XLensInference(str(checkpoint), device=device, config=str(config) if config else None)
    output = model(batch)
    depth = output["depth_metric"][0].detach().cpu().numpy()
    confidence = output["depth_conf"][0].detach().cpu().numpy()

    points_by_view = []
    for view_index in range(len(images)):
        points_view0 = unproject_to_world(depth[view_index], d_cams[view_index], relative_c2w[view_index])
        points_world = transform_points(points_view0.reshape(-1, 3), world_c2w_array[0]).reshape(
            points_view0.shape
        )
        points_by_view.append(points_world)

    points_view0, colors = fuse_point_cloud(
        depth,
        np.stack(d_cams, axis=0),
        relative_c2w,
        rgb=np.stack(images, axis=0),
        conf=confidence,
        conf_drop_pct=conf_drop_pct,
        max_depth=max_depth,
        fov_max_deg=fov_max,
    )
    points_world = transform_points(points_view0, world_c2w_array[0])

    objects = []
    for object_spec in manifest.get("objects", []):
        estimated = estimate_object_from_views(
            points_by_view,
            masks,
            [confidence[index] for index in range(len(images))],
            object_spec["label"],
        )
        estimated.update(
            {
                "name": object_spec["name"],
                "category": object_spec.get("category", "unknown"),
            }
        )
        objects.append(estimated)

    geometry = {
        "schema_version": "1.0",
        "task_name": manifest.get("task_name", "unknown"),
        "coordinate_frame": manifest.get("coordinate_frame", "world"),
        "num_views": len(images),
        "metric_scale_valid": True,
        "objects": objects,
        "scene_bounds": {
            "min": points_world.min(axis=0).tolist() if len(points_world) else None,
            "max": points_world.max(axis=0).tolist() if len(points_world) else None,
        },
        "quality": {
            "point_count": int(len(points_world)),
            "mean_depth_confidence": float(np.mean(confidence)),
        },
    }
    artifacts = {
        "depth_metric": depth,
        "depth_conf": confidence,
        "points": points_world,
        "colors": colors,
        "points_by_view": points_by_view,
        "masks": masks,
    }
    return geometry, artifacts
