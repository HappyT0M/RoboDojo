"""Export the objects and saved scenes used by the eight custom tasks for HF.

This creates a local dataset directory only. Robot, room, ground, and shared
material dependencies remain in the official RoboDojo asset installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "put_rope_ball_in_basket", "pass_rope_through_ring", "route_rope_around_posts",
    "fold_clothes_deformable", "flatten_clothes", "hang_clothes",
    "put_clothes_in_bin", "drape_cloth_over_rack",
)
OBJECT_TYPES = ("Rigid", "Dynamic", "Geometry", "Garment", "Articulation", "Fluid")
SKIP = {".git", ".cache", ".DS_Store", "__pycache__"}
UPSTREAM = "RoboDojo-Benchmark/RoboDojo"


def inside_assets(root: Path, relative: Path) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset dependency escapes Assets: {relative}")
    return path


def collect(assets: Path, tasks=TASKS, layout_count=10):
    files = set()
    dependencies = {}
    scenes = {"rooms": set(), "backgrounds": set(), "table_materials": set()}
    layout_dir = Path("Eval_Layout/RoboDojo/arx_x5/0")
    for task in tasks:
        used = set()
        for index in range(layout_count):
            relative = layout_dir / f"{task}_{index}.json"
            layout = json.loads(inside_assets(assets, relative).read_text(encoding="utf-8"))
            files.add(relative)
            for kind in OBJECT_TYPES:
                for category, entries in layout.get(kind, {}).items():
                    for entry in entries:
                        if entry.get("category", category) != category:
                            raise ValueError(f"Conflicting category in {relative}: {entry}")
                        asset_dir = Path("Object/RoboDojo") / kind / category / f'{int(entry["category_idx"]):05d}'
                        inside_assets(assets, asset_dir)
                        used.add(asset_dir)
            for key, section, field in (
                ("rooms", "Room", "default"),
                ("backgrounds", "Background", "category_name"),
                ("table_materials", "Table", "default"),
            ):
                if layout.get(section, {}).get(field):
                    scenes[key].add(layout[section][field])
        dependencies[task] = sorted(p.as_posix() for p in used)

    object_dirs = sorted({p for paths in dependencies.values() for p in paths})
    for relative in object_dirs:
        directory = inside_assets(assets, Path(relative))
        if not (directory / "metadata.json").is_file() or not any(
            (directory / name).is_file() for name in ("object.usd", "object.usdz")
        ):
            raise FileNotFoundError(f"Missing model or metadata: {directory}")
        # Preserve asset-side textures, annotations, and other sidecar files.
        for file in sorted(directory.rglob("*")):
            rel = file.relative_to(assets)
            if not SKIP.intersection(rel.parts) and file.is_file():
                inside_assets(assets, rel)
                files.add(rel)
    return sorted(files), dependencies, {k: sorted(v) for k, v in scenes.items()}


def export_bundle(assets: Path, output: Path, *, tasks=TASKS, layout_count=10, base_revision=None):
    assets, output = assets.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new export directory to avoid mixing releases: {output}")
    if output.is_relative_to(assets):
        raise ValueError("Export output must be outside the source Assets directory")
    files, dependencies, scene_dependencies = collect(assets, tasks, layout_count)
    # Resolve every required model before creating the export directory.
    output.mkdir(parents=True)
    entries = []
    for relative in files:
        source = assets / relative
        destination = output / "Assets" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append({
            "path": destination.relative_to(output).as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
    manifest = {
        "schema_version": 1,
        "tasks": list(tasks),
        "layouts_per_task": layout_count,
        "object_directories": sorted({p for values in dependencies.values() for p in values}),
        "task_objects": dependencies,
        "base_assets": {
            "repo_id": UPSTREAM,
            "revision": base_revision,
            "required": ["Robots", "Room", "Material", "Background"],
            "scene_dependencies": scene_dependencies,
            "note": "Install official base assets first. External USD dependencies are not recursively resolved by this exporter.",
        },
        "files": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# RoboDojo deformable task assets\n\n"
        f"Objects and {len(tasks) * layout_count} saved layouts for the custom task set.\n\n"
        "Install the official RoboDojo base assets first, then merge this repository's "
        "Assets directory into the local RoboDojo Assets directory, preserving paths. "
        "Robot, room, background and shared material assets come from the official installation.\n\n"
        f"Upstream asset source: https://huggingface.co/datasets/{UPSTREAM}\n\n"
        "This export contains both custom and reused official assets. Retain upstream "
        "attribution and license terms when publishing the reused assets. No new license "
        "for upstream assets is assigned by this export.\n\n"
        "See manifest.json for task dependencies, file sizes and SHA-256 checksums.\n\n"
        + "\n".join(f"- `{task}`" for task in tasks) + "\n",
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "Assets")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the HF dataset contents")
    parser.add_argument("--base-revision", help="Known official HF asset commit SHA, recorded for reproducibility")
    args = parser.parse_args()
    manifest = export_bundle(args.assets_dir, args.output, base_revision=args.base_revision)
    total = sum(f["size_bytes"] for f in manifest["files"])
    print(f'Exported {len(manifest["object_directories"])} object directories, '
          f'{len(manifest["tasks"]) * manifest["layouts_per_task"]} layouts, '
          f'{total / 1024**2:.2f} MiB to {args.output.resolve()}')


if __name__ == "__main__":
    main()
