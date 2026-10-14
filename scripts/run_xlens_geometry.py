"""Run one-shot XLens geometry estimation on a RoboDojo snapshot manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make ``python scripts/run_xlens_geometry.py`` work when launched from the
# RoboDojo repository root, without requiring an editable install.
ROBODOJO_ROOT = Path(__file__).resolve().parents[1]
if str(ROBODOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBODOJO_ROOT))

from utils.xlens_geometry import load_scene_manifest, run_xlens_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate metric object and fixture geometry from a RoboDojo multi-view snapshot."
    )
    parser.add_argument("--manifest", required=True, help="Path to a RoboDojo XLens scene manifest JSON")
    parser.add_argument("--ckpt", required=True, help="Path to an XLens checkpoint")
    parser.add_argument("--xlens-root", required=True, help="Path to the checked-out XLens repository")
    parser.add_argument("--config", default=None, help="XLens architecture YAML, required for bare .safetensors")
    parser.add_argument("--out-dir", required=True, help="Directory for geometry and point-cloud outputs")
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device, e.g. cuda, cuda:0, cuda:2, or cpu. If --device-id is given, it takes precedence.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="CUDA device index; equivalent to --device cuda:<id>.",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=None,
        help="Optional XLens input resolution; both values must be divisible by 14. Default: align native size downward.",
    )
    parser.add_argument("--max-depth", type=float, default=None, help="Optional Euclidean point range in metres")
    parser.add_argument("--conf-drop-pct", type=float, default=10.0)
    parser.add_argument("--fov-max", type=float, default=85.0)
    return parser


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    has_color = colors is not None and len(colors) == len(points)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3) if has_color else None
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}"]
    header.extend(["property float x", "property float y", "property float z"])
    if has_color:
        header.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header.append("end_header\n")
    if has_color:
        dtype = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
        )
        buffer = np.empty(len(points), dtype=dtype)
        buffer["x"], buffer["y"], buffer["z"] = points.T
        buffer["r"], buffer["g"], buffer["b"] = colors.T
    else:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        buffer = np.empty(len(points), dtype=dtype)
        buffer["x"], buffer["y"], buffer["z"] = points.T
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        buffer.tofile(handle)


def _select_label_points(points_by_view, masks_by_view, label):
    labels = label if isinstance(label, (list, tuple)) else [label]
    chunks = []
    for points, mask in zip(points_by_view, masks_by_view):
        selected = np.isin(mask, np.asarray(labels))
        if np.any(selected):
            chunks.append(points[selected])
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = f"cuda:{args.device_id}" if args.device_id is not None else args.device
    manifest = load_scene_manifest(args.manifest)
    geometry, artifacts = run_xlens_scene(
        manifest,
        checkpoint=args.ckpt,
        xlens_root=args.xlens_root,
        config=args.config,
        device=device,
        image_size=args.image_size,
        max_depth=args.max_depth,
        conf_drop_pct=args.conf_drop_pct,
        fov_max=args.fov_max,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "depth_metric.npy", artifacts["depth_metric"])
    np.save(out_dir / "depth_conf.npy", artifacts["depth_conf"])
    _save_ply(out_dir / "points.ply", artifacts["points"], artifacts["colors"])

    for object_spec in manifest.get("objects", []):
        object_points = _select_label_points(
            artifacts["points_by_view"],
            artifacts["masks"],
            object_spec["label"],
        )
        safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in object_spec["name"])
        _save_ply(out_dir / f"object_{safe_name}.ply", object_points)

    (out_dir / "geometry.json").write_text(
        json.dumps(geometry, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(geometry, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
