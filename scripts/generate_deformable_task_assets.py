"""Create lightweight fixtures and ten fixed evaluation scenes for garment tasks."""

from __future__ import annotations

import json
import math
import re
import sys
import uuid
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
    "drape_cloth_over_rack",
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


def write_towel():
    name, nx, ny, width, depth = "square_towel", 16, 12, 0.42, 0.34
    points = [(-width/2 + width*i/(nx-1), -depth/2 + depth*j/(ny-1), 0.0)
              for j in range(ny) for i in range(nx)]
    faces = []
    for j in range(ny-1):
        for i in range(nx-1):
            a = j*nx+i
            faces.extend(((a,a+1,a+nx+1),(a,a+nx+1,a+nx)))
    directory = ASSETS / "Garment" / name / "00000"
    directory.mkdir(parents=True, exist_ok=True)
    text = rope._mesh_usd(name, points, faces, (0.24, 0.62, 0.78))
    text = re.sub(r'def Mesh "square_towel" \(\s*prepend apiSchemas = \["PhysicsCollisionAPI"\]\s*\)',
                  'def Mesh "square_towel"', text)
    (directory / "object.usd").write_text(text, encoding="utf-8")
    metadata = rope._metadata(points, faces, name)
    metadata["uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "robodojo://garment/square-towel"))
    metadata["physics"] = {"type": "cloth"}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def entry(category, label, pos, quat=(1,0,0,0), typ="geometry", scale=(1,1,1), idx=0):
    return rope._entry(category, label, pos, quat, typ, scale, idx)


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
    yaw = (0, math.pi/6, -math.pi/6, math.pi/3, -math.pi/3)[i % 5]
    garment_q = (math.cos(yaw/2), 0, 0, math.sin(yaw/2))
    objects = {"Garment": {"Top_Long": [entry("Top_Long", "target", gpos, garment_q, "garment", idx=gidx)]},
               "Geometry": {}}
    if task == "fold_clothes_deformable":
        objects["Geometry"]["folding_board"] = [entry("folding_board", "target_area", (target_x,target_y,0.768))]
    elif task == "flatten_clothes":
        objects["Geometry"]["folding_board"] = [entry("folding_board", "target_area", (target_x,target_y,0.768))]
    elif task in ("hang_clothes", "drape_cloth_over_rack"):
        category = "square_towel" if task == "drape_cloth_over_rack" else "Top_Long"
        if task == "drape_cloth_over_rack":
            objects["Garment"] = {category: [entry(category, "target", (-0.28, -0.20, 0.775), garment_q, "garment")]}
        objects["Geometry"]["garment_rack"] = [entry("garment_rack", "rack", (0.20, 0.10, 0.765))]
    elif task == "put_clothes_in_bin":
        objects["Geometry"]["garment_bin"] = [entry("garment_bin", "bin", (0.22, 0.12, 0.765))]
    if task in ("fold_clothes_deformable", "flatten_clothes"):
        objects["Garment"]["Top_Long"][0]["default_pos"] = list(gpos)
    # Clutter/fixtures vary independently from the garment pose, changing the scene geometry.
    if i % 2:
        objects["Geometry"]["cloth_block"] = [entry("cloth_block", "obstacle", (-0.43 + .08*(i%3), .23, .79), scale=(1,1,1))]
    if task == "put_clothes_in_bin" and i % 3 == 0:
        objects["Geometry"]["cloth_block"] = [entry("cloth_block", "obstacle", (-0.35,.22,.79))]
    return scene(objects, i)


def make_assets():
    write_geometry("folding_board", [((0,0,.012),(.68,.50,.024))], (.38,.28,.16))
    # Two uprights and one crossbar; bar center is 25.5 cm above tabletop.
    write_geometry("garment_rack", [((-.22,0,.1275),(.025,.035,.255)), ((.22,0,.1275),(.025,.035,.255)),
                                    ((0,0,.2475),(.48,.035,.018))], (.30,.32,.36))
    write_geometry("garment_bin", [((0,0,.008),(.34,.26,.016)), ((0,-.126,.085),(.34,.008,.17)),
                                   ((0,.126,.085),(.34,.008,.17)), ((-.166,0,.085),(.008,.25,.17)),
                                   ((.166,0,.085),(.008,.25,.17))], (.20,.46,.32))
    write_geometry("cloth_block", [((0,0,.025),(.12,.12,.05))], (.56,.38,.20))
    write_towel()


def make_new_task_layouts():
    for task in TASKS:
        for i in range(10):
            save_layout(task, i, garment_scene(task, i))


if __name__ == "__main__":
    make_assets()
    make_new_task_layouts()
    print("Generated four reusable fixtures, a simulated cloth towel, and 50 garment-task layouts.")
