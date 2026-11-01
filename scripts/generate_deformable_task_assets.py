"""Create lightweight fixtures and ten fixed evaluation scenes for garment tasks."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import create_rope_assets as rope  # noqa: E402

ASSETS = ROOT / "Assets/Object/RoboDojo"
LAYOUTS = ROOT / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
GARMENTS = (1, 4, 6, 7, 9, 11)
TASKS = (
    "fold_clothes_deformable",
    "flatten_clothes",
    "hang_clothes",
    "put_clothes_in_bin",
)
GARMENT_COLORS = (
    (0.20, 0.32, 0.44), (0.36, 0.42, 0.46), (0.48, 0.42, 0.34),
    (0.30, 0.40, 0.34), (0.46, 0.30, 0.24), (0.18, 0.24, 0.32),
)


def box(points, faces, center, size):
    x, y, z = center
    sx, sy, sz = (v / 2 for v in size)
    start = len(points)
    points.extend([(x-sx,y-sy,z-sz),(x+sx,y-sy,z-sz),(x+sx,y+sy,z-sz),(x-sx,y+sy,z-sz),
                   (x-sx,y-sy,z+sz),(x+sx,y-sy,z+sz),(x+sx,y+sy,z+sz),(x-sx,y+sy,z+sz)])
    for f in ((0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),
              (2,3,7),(2,7,6),(3,0,4),(3,4,7)):
        faces.append(tuple(start+i for i in f))


def write_geometry(name, pieces, color):
    points, faces = [], []
    for center, size in pieces:
        box(points, faces, center, size)
    rope._write_geometry_asset(name, points, faces, color)


def entry(category, label, pos, quat=(1,0,0,0), typ="geometry", scale=(1,1,1), idx=0):
    value = rope._entry(category, label, pos, quat, typ, scale, idx)
    if typ == "geometry":
        # Keep the bin opening and the space under the rack traversable.
        value["physics"]["mesh_collision_approximation"] = "none"
    return value


def scene(objects, i, table=(1.4,1.1,0.05)):
    value = rope._layout(objects)
    value["Table"]["scale"] = [table[0], table[1], table[2]]
    value["Table"]["default_pos"][1] = -0.05 + (i % 3 - 1)*0.025
    value["Room"]["default"] = ("Simple_Room_nolight", "Simple_Room_orange", "Simple_Room_teal")[i % 3]
    return value


def save_layout(task, i, value):
    if task not in TASKS:
        raise ValueError(f"Refusing to overwrite layouts outside the custom task set: {task}")
    LAYOUTS.mkdir(parents=True, exist_ok=True)
    (LAYOUTS / f"{task}_{i}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def garment_scene(task, i):
    gidx = GARMENTS[i % len(GARMENTS)]
    target_x = (-0.08, 0.0, 0.08, -0.05, 0.05)[i % 5]
    target_y = (-0.02, 0.02, 0.0, 0.04, -0.04)[(i // 2) % 5]
    gpos = (target_x - 0.16 + (i % 3)*0.02, target_y - 0.12, 0.775)
    if task in ("hang_clothes", "put_clothes_in_bin"):
        # Keep the initial cloth entirely to the left of the now-solid fixtures.
        gpos = (-0.33 + (i % 3)*0.01, -0.18 + target_y*0.5, 0.775)
    yaw = (0, math.pi/6, -math.pi/6, math.pi/3, -math.pi/3)[i % 5]
    garment_q = (math.cos(yaw/2), 0, 0, math.sin(yaw/2))
    objects = {"Garment": {"Top_Long": [entry("Top_Long", "target", gpos, garment_q, "garment", idx=gidx)]},
               "Geometry": {}}
    if task == "fold_clothes_deformable":
        objects["Geometry"]["folding_board"] = [entry("folding_board", "target_area", (target_x,target_y,0.768))]
    elif task == "flatten_clothes":
        objects["Geometry"]["folding_board"] = [entry("folding_board", "target_area", (target_x,target_y,0.768))]
    elif task == "hang_clothes":
        objects["Geometry"]["garment_rack"] = [entry("garment_rack", "rack", (0.32+target_x*.25, 0.12+target_y, 0.765))]
    elif task == "put_clothes_in_bin":
        objects["Geometry"]["garment_bin"] = [entry("garment_bin", "bin", (0.26+target_x*.25, 0.12+target_y, 0.765))]
    # Clutter/fixtures vary independently from the garment pose, changing the scene geometry.
    if i % 2:
        objects["Geometry"]["cloth_block"] = [entry("cloth_block", "obstacle", (-0.43 + .08*(i%3), .37, .765), scale=(1,1,1))]
    if task == "put_clothes_in_bin" and i % 3 == 0:
        objects["Geometry"]["cloth_block"] = [entry("cloth_block", "obstacle", (-0.35,.37,.765))]
    value = scene(objects, i)
    garment = objects["Garment"]["Top_Long"][0]
    # Override inconsistent imported shaders only in these custom saved scenes.
    garment["visual"] = {"color": list(GARMENT_COLORS[i % len(GARMENT_COLORS)]),
                         "roughness": 0.9, "metallic": 0.0}
    support_z = rope.table_top(value)
    if "folding_board" in objects["Geometry"]:
        board = objects["Geometry"]["folding_board"][0]
        board["default_pos"][2] = support_z
        support_z += 0.024
    # Native garments are centered, with vertices below local Z=0. Once the
    # fixtures are visible/collidable, spawning at 0.775 would bury cloth in them.
    lowest = rope.transformed_min_z("Garment", "Top_Long", gidx, garment_q, garment["scale"])
    garment["default_pos"][2] = round(support_z - lowest + 0.01, 6)
    return value


def make_assets():
    write_geometry("folding_board", [((0,0,.012),(.68,.50,.024))], (.48,.43,.35))
    # Bar center matches GarmentTask.bar_height; feet rest on the tabletop.
    write_geometry("garment_rack", [((-.22,0,.1275),(.025,.035,.255)), ((.22,0,.1275),(.025,.035,.255)),
                                    ((0,0,.255),(.48,.035,.018)),
                                    ((-.22,0,.012),(.06,.14,.024)), ((.22,0,.012),(.06,.14,.024))], (.24,.27,.30))
    write_geometry("garment_bin", [((0,0,.008),(.34,.26,.016)), ((0,-.126,.085),(.34,.008,.17)),
                                   ((0,.126,.085),(.34,.008,.17)), ((-.166,0,.085),(.008,.25,.17)),
                                   ((.166,0,.085),(.008,.25,.17))], (.30,.40,.34))
    write_geometry("cloth_block", [((0,0,.025),(.12,.12,.05))], (.46,.36,.26))


def make_new_task_layouts():
    for task in TASKS:
        for i in range(10):
            save_layout(task, i, garment_scene(task, i))


if __name__ == "__main__":
    make_assets()
    make_new_task_layouts()
    print("Generated four reusable fixtures and 40 garment-task layouts.")
