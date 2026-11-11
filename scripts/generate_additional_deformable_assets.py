"""Generate/check the cover, pull and wipe assets and 30 deterministic layouts."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import create_rope_assets as rope

ASSETS = ROOT / "Assets"
TASKS = ("cover_object_with_cloth", "pull_rope_to_retrieve_ball", "wipe_marked_area_with_cloth")
CLOTHS = {"cover_cloth": (.48, .40, 25, 21), "wiping_cloth": (.20, .16, 15, 13)}
COLORS = ((.24, .36, .46), (.42, .34, .25), (.28, .40, .33), (.45, .29, .25), (.34, .31, .43))


def cloth_mesh(width, depth, nx, ny):
    points = [(-width/2 + width*x/(nx-1), -depth/2 + depth*y/(ny-1), 0.)
              for y in range(ny) for x in range(nx)]
    faces = []
    for y in range(ny-1):
        for x in range(nx-1):
            a = y*nx+x
            faces.extend(((a, a+1, a+nx+1), (a, a+nx+1, a+nx)))
    return points, faces


def cloth_physics():
    return {
        "type": "garment",
        "garment_config": {"particle_mass": .00025, "stretch_stiffness": 10000.,
                           "bend_stiffness": .2, "shear_stiffness": 100., "spring_damping": .2,
                           "self_collision": True, "self_collision_filter": True},
        "particle_system": {"contact_offset": .008, "rest_offset": .004,
                            "particle_contact_offset": .008, "solid_rest_offset": .004,
                            "solver_position_iteration_count": 16, "wind": [0., 0., 0.]},
        "particle_material": {"friction": .5, "particle_friction_scale": .5,
                              "adhesion": 0., "drag": .02, "lift": .005},
    }


def entry(category, label, pos, *, yaw=0, scale=(1, 1, 1), kind="geometry"):
    value = rope._entry(category, label, pos, (math.cos(yaw/2), 0, 0, math.sin(yaw/2)), kind, scale)
    if kind == "garment":
        value["physics"] = cloth_physics()
    elif kind == "geometry":
        value["physics"].update(mesh_collision_approximation="none", static_friction=.5, dynamic_friction=.4)
    return value


def make_layout(task, index):
    if task not in TASKS or not 0 <= index < 10:
        raise ValueError("Expected a new task and a layout index 0–9")
    dx = (-.04, -.02, 0., .02, .04)[index % 5]
    dy = (-.03, .03)[index // 5]
    layout = rope._layout({"Geometry": {}})
    layout["Room"]["default"] = ("Simple_Room_nolight", "Simple_Room_orange", "Simple_Room_teal")[index % 3]
    layout["Table"]["scale"][:2] = [(1.35, 1.4, 1.45)[index % 3], (1.05, 1.1, 1.15)[(index // 2) % 3]]
    layout["Background"]["intensity"] = (850, 1000, 1150)[index % 3]
    top = rope.table_top(layout)
    yaw = (-1, 0, 1)[index % 3] * math.pi/18
    if task == "pull_rope_to_retrieve_ball":
        x = (-.18, -.09, 0., .09, .18)[index % 5]
        layout["Articulation"] = {"rope_chain": [entry(
            "rope_chain", "rope", (x, -.18 + dy*.5, top + .025), yaw=math.pi/2 + yaw*.5, kind="articulation") ]}
        layout["Geometry"]["rope_retrieval_zone"] = [entry("rope_retrieval_zone", "goal", (x, .15 + dy*.5, top))]
        # Side rails vary in position and size without blocking the pull corridor.
        if index % 2:
            layout["Geometry"]["cloth_block"] = [entry("cloth_block", "distractor", (x+.28, .22, top), scale=(.5, 2., 1.))]
    else:
        cloth_name = "cover_cloth" if task == "cover_object_with_cloth" else "wiping_cloth"
        cloth = entry(cloth_name, "cloth", (-.34+dx*.4, -.13+dy, top+.016), yaw=yaw, kind="garment")
        cloth["visual"] = {"color": list(COLORS[index % len(COLORS)]), "roughness": .95, "metallic": 0.}
        layout["Garment"] = {cloth_name: [cloth]}
        if task == "cover_object_with_cloth":
            scale = (.85, 1., 1.15)[index % 3]
            layout["Geometry"]["cover_block"] = [entry("cover_block", "goal", (.25+dx, .10+dy, top),
                                                         yaw=yaw, scale=(scale, scale, 1.+.15*(index % 2)))]
        else:
            layout["Geometry"]["wipe_board"] = [entry("wipe_board", "goal", (.18+dx, .08+dy, top), yaw=yaw)]
        if index % 2:
            layout["Geometry"]["cloth_block"] = [entry("cloth_block", "distractor", (-.43, .34, top), scale=(1., 1., .6+.2*(index % 3)))]
    return layout


def generated_files():
    files = {}
    for name, dimensions in CLOTHS.items():
        points, faces = cloth_mesh(*dimensions)
        # Cloth gets PhysX particle schemas at load time, not static triangle collision.
        usd = rope._mesh_usd("asset", points, faces, COLORS[0]).replace(
            '["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"]', '["MaterialBindingAPI"]'
        ).replace('    bool physics:collisionEnabled = true\n', '').replace('    uniform token physics:approximation = "none"\n', '')
        directory = Path("Object/RoboDojo/Garment") / name / "00000"
        files[directory / "object.usd"] = usd
        files[directory / "metadata.json"] = json.dumps(rope._metadata(points, faces, name), indent=2) + "\n"
    for name, size, color in (
        ("cover_block", (.18, .14, .05), (.58, .42, .20)),
        ("wipe_board", (.40, .28, .006), (.22, .42, .40)),
        ("rope_retrieval_zone", (.22, .16, .001), (.18, .48, .36)),
    ):
        points, faces = [], []
        rope._add_box(points, faces, (0, 0, size[2]/2), size)
        usd = rope._mesh_usd("asset", points, faces, color)
        directory = Path("Object/RoboDojo/Geometry") / name / "00000"
        files[directory / "object.usd"] = usd
        files[directory / "metadata.json"] = json.dumps(rope._metadata(points, faces, name), indent=2) + "\n"
    for task in TASKS:
        for index in range(10):
            files[Path("Eval_Layout/RoboDojo/arx_x5/0") / f"{task}_{index}.json"] = json.dumps(make_layout(task, index), indent=2) + "\n"
    return files


def generate(assets=ASSETS, check=False):
    files = generated_files()
    for relative, contents in files.items():
        path = assets / relative
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != contents:
                raise ValueError(f"Missing or changed generated file: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only verification of assets and all 30 layouts")
    parser.add_argument("--assets-dir", type=Path, default=ASSETS)
    args = parser.parse_args()
    count = generate(args.assets_dir, args.check)
    print(f"{'Verified' if args.check else 'Generated'} 5 assets and 30 layouts ({count} files).")
