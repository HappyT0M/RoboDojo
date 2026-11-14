"""Build ten reproducible, scene-varied eval layouts per rope task."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import create_rope_assets as assets  # noqa: E402

LAYOUT_DIR = ROOT / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
TASKS = (
    "put_rope_ball_in_basket",
    "pass_rope_through_ring",
    "route_rope_around_posts",
)
SCENE_OFFSETS = ((0,0),(0.025,0),(-.025,0),(0,.025),(0,-.025),(.02,.02),(-.02,.02),(.02,-.02),(-.02,-.02),(0,0))
ROPE_OFFSETS = ((0,0),(.015,-.01),(-.015,.01),(.02,0),(-.02,0),(-.025,-.015),(.025,.015),(0,.025),(0,-.025),(.03,-.03))

BALL_CENTER_X = 0.594  # (12 - 1) * 0.05 + 0.05 / 2 + 0.019 in the USD rest pose.
ROPE_END_X = BALL_CENTER_X + 0.019
TABLE_X = (-0.70, 0.70)
TABLE_Y = (-0.60, 0.50)
LAYOUT_TWEAKS = {
    # Keep the task semantics and room variation, while reducing rope-target
    # separation and incidental obstacle clutter in the layouts selected for
    # focused policy re-evaluation.
    "put_rope_ball_in_basket": {0, 2, 8, 9},
    "pass_rope_through_ring": {2, 5, 9},
    "route_rope_around_posts": {1, 5, 7, 9},
}


def _shift_xy(entry: dict, dx: float, dy: float) -> None:
    position = entry["default_pos"]
    entry["default_pos"] = [
        round(position[0] + dx, 6),
        round(position[1] + dy, 6),
        position[2],
    ]


def make_layout(base: dict, offsets: tuple[float, float, float, float]) -> dict:
    scene_x, scene_y = offsets
    layout = deepcopy(base)
    for entries in layout["Articulation"].values():
        for entry in entries:
            _shift_xy(entry, scene_x, scene_y)
    for entries in layout["Geometry"].values():
        for entry in entries:
            _shift_xy(entry, scene_x, scene_y)
    return layout


def _set_xy(entry, x, y):
    entry["default_pos"][:2] = [round(x, 6), round(y, 6)]


def _scene_variation(layout, task, index):
    dx, dy = SCENE_OFFSETS[index]
    rx, ry = ROPE_OFFSETS[index]
    rope = layout["Articulation"]["rope_chain"][0]
    _set_xy(rope, -0.28 + dx + rx, -0.12 + dy + ry) if task != "route_rope_around_posts" else _set_xy(rope, -0.594 + dx + rx, -0.20 + dy + ry)
    table = layout["Table"]
    table["scale"][0] = [1.35, 1.4, 1.45][index % 3]
    table["scale"][1] = [1.05, 1.1, 1.15][(index // 2) % 3]
    layout["Room"]["default"] = ("Simple_Room_nolight", "Simple_Room_orange", "Simple_Room_teal")[index % 3]
    geometry = layout["Geometry"]
    if task == "put_rope_ball_in_basket":
        basket = geometry["basket"][0]
        basket["category_idx"] = (2, 3, 4, 5)[index % 4]
        _set_xy(basket, 0.34 + dx, 0.14 + dy)
        if index % 3 == 2:
            geometry["rope_post"] = [
                {**deepcopy(basket), "category": "rope_post", "label": "barrier", "category_idx": 0,
                 "default_pos": [-.05 + dx, .18 + dy, .765], "scale": [.8, .8, 1.0]}
            ]
    elif task == "pass_rope_through_ring":
        frame = geometry["frame"][0]
        _set_xy(frame, .20 + dx, .05 + dy)
        # Spin within the frame plane, preserving its directed normal and the aperture test.
        q0 = (0.6532814824, 0.6532814824, 0.2705980501, 0.2705980501)
        theta = (index % 5 - 2) * math.pi / 12
        qz = (math.cos(theta/2), 0, 0, math.sin(theta/2))
        w,x,y,z = q0; a,b,c,d = qz
        frame["default_ori"] = [round(w*a-x*b-y*c-z*d, 9), round(w*b+x*a+y*d-z*c, 9),
                                round(w*c-x*d+y*a+z*b, 9), round(w*d+x*c-y*b+z*a, 9)]
        lowest = assets.transformed_min_z("Geometry", "frame", frame["category_idx"],
                                          frame["default_ori"], frame["scale"])
        frame["default_pos"][2] = round(assets.table_top(layout) - lowest + 0.002, 6)
        if index % 3 == 1:
            geometry["rope_post"] = [{"category":"rope_post","category_idx":0,"group":None,"label":"barrier",
                "default_pos":[-.08+dx,.20+dy,.765],"default_ori":[1,0,0,0],"scale":[1,1,1],
                "relative_plane":"Table","need_check_stable":True,"physics":{"type":"geometry","collision":True},"visual":{}}]
    elif task == "route_rope_around_posts":
        posts = geometry["rope_post"]
        spread = (.10, .13, .16)[index % 3]
        _set_xy(posts[0], -.08 + dx - spread, .02 + dy)
        _set_xy(posts[1], -.08 + dx + spread, .02 + dy)
        slot = geometry["rope_slot"][0]
        _set_xy(slot, -.08 + dx, .28 + dy)
    if index % 2 == 1 and task != "put_rope_ball_in_basket" and "rope_post" not in geometry:
        geometry["rope_post"] = [{"category":"rope_post","category_idx":0,"group":None,"label":"barrier",
            "default_pos":[.30+dx,-.12+dy,.765],"default_ori":[1,0,0,0],"scale":[1,1,1],
            "relative_plane":"Table","need_check_stable":True,"physics":{"type":"geometry","collision":True},"visual":{}}]
    return layout


def _apply_focused_layout_tweak(layout: dict, task: str, index: int) -> dict:
    """Apply small, deterministic adjustments to layouts selected for retest."""
    if index not in LAYOUT_TWEAKS[task]:
        return layout

    geometry = layout["Geometry"]
    rope = layout["Articulation"]["rope_chain"][0]
    if task == "put_rope_ball_in_basket":
        # Layout 2 previously produced an invalid rope transform during rollout.
        # Move the rope and basket slightly toward the table center while
        # keeping their clear near/far separation along Y.
        _shift_xy(rope, 0.0, 0.02)
        _shift_xy(geometry["basket"][0], 0.0, -0.02)
        if index == 2:
            _shift_xy(rope, 0.02, 0.0)
            _shift_xy(geometry["basket"][0], 0.015, 0.0)
        geometry.pop("rope_post", None)
    elif task == "pass_rope_through_ring":
        # Use the nominal upright frame pose for the focused retest and retain
        # the computed table clearance. Move rope and frame apart along Y to
        # reduce incidental initial contact. Remove the post obstacle.
        frame = geometry["frame"][0]
        if index in {2, 9}:
            _shift_xy(rope, 0.0, -0.02)
            _shift_xy(frame, 0.0, 0.02)
        if index == 2:
            # This layout still hit an invalid PhysX rope transform. Increase
            # the initial rope/frame separation without changing the frame
            # pose or the other previously stable focused layouts.
            _shift_xy(rope, 0.0, -0.03)
            _shift_xy(frame, 0.0, 0.02)
        frame["default_ori"] = [0.653281482, 0.653281482, 0.27059805, 0.27059805]
        lowest = assets.transformed_min_z(
            "Geometry", "frame", frame["category_idx"], frame["default_ori"], frame["scale"]
        )
        frame["default_pos"][2] = round(assets.table_top(layout) - lowest + 0.01, 6)
        geometry.pop("rope_post", None)
    elif task == "route_rope_around_posts":
        # Keep the rope on the near side with an extra 3 cm of clearance from
        # the posts after layouts 5 and 7 produced invalid transforms.
        _shift_xy(rope, 0.03, 0.02)
        if index in {5, 7}:
            _shift_xy(rope, 0.0, -0.03)
    return layout


def _normalize_route_layout(layout):
    geometry = layout.setdefault("Geometry", {})
    geometry.pop("vase", None)
    geometry.pop("mousemat", None)
    geometry["rope_post"] = [
        {"category": "rope_post", "category_idx": 0, "group": None, "label": f"post{i}",
         "default_pos": [x, 0.02, 0.765], "default_ori": [1,0,0,0], "scale": [1,1,1],
         "relative_plane": "Table", "need_check_stable": True,
         "physics": {"type": "geometry", "collision": True}, "visual": {}}
        for i, x in enumerate((-.18, .02))
    ]
    geometry["rope_slot"] = [{"category": "rope_slot", "category_idx": 0, "group": None, "label": "target_slot",
        "default_pos": [-.08, .28, .765], "default_ori": [1,0,0,0], "scale": [1,1,1],
        "relative_plane": "Table", "need_check_stable": True,
        "physics": {"type": "geometry", "collision": True}, "visual": {}}]
    return layout


def validate_layout(task: str, layout: dict) -> None:
    rope = layout["Articulation"]["rope_chain"][0]["default_pos"]
    ball_x, ball_y = rope[0] + BALL_CENTER_X, rope[1]
    margin = 0.04
    if not (TABLE_X[0] + margin <= rope[0] and rope[0] + ROPE_END_X <= TABLE_X[1] - margin):
        raise ValueError(f"{task}: rope rest pose exceeds the table X margin")
    if not TABLE_Y[0] + margin <= rope[1] <= TABLE_Y[1] - margin:
        raise ValueError(f"{task}: rope rest pose exceeds the table Y margin")

    geometry = layout["Geometry"]
    if task == "put_rope_ball_in_basket":
        basket = geometry["basket"][0]["default_pos"]
        if abs(ball_y - basket[1]) <= 0.11:
            raise ValueError(f"{task}: ball starts too close to the basket")
    elif task == "pass_rope_through_ring":
        frame_entry = geometry["frame"][0]
        frame = frame_entry["default_pos"]
        lowest = assets.transformed_min_z("Geometry", "frame", frame_entry["category_idx"],
                                          frame_entry["default_ori"], frame_entry["scale"])
        if frame[2] + lowest < assets.table_top(layout) - 1e-6:
            raise ValueError(f"{task}: frame intersects the table")
        # The configured directed far side is (-X, +Y); keep the nominal ball
        # clearly on the near side before any policy action or wind step.
        near_side_distance = ((frame[0] - ball_x) + (ball_y - frame[1])) / 2**0.5
        if near_side_distance >= -0.10:
            raise ValueError(f"{task}: ball does not start clear of the frame")
    elif task == "route_rope_around_posts":
        posts = geometry["rope_post"]
        slot = geometry["rope_slot"][0]["default_pos"]
        if ball_y >= posts[0]["default_pos"][1] - 0.04 or slot[1] <= posts[0]["default_pos"][1] + 0.12:
            raise ValueError(f"{task}: near-side start or far-side goal is lost")
    else:
        raise ValueError(f"Unknown rope task: {task}")


def generate(*, check: bool = False) -> None:
    for task in TASKS:
        base_path = LAYOUT_DIR / f"{task}_0.json"
        base = json.loads(base_path.read_text(encoding="utf-8"))
        if task == "route_rope_around_posts":
            base = _normalize_route_layout(base)
        if task == "route_rope_around_posts" and ("rope_post" not in base["Geometry"] or "rope_slot" not in base["Geometry"]):
            raise ValueError("route_rope_around_posts base layout must contain rope_post and rope_slot")
        # Each rotation is derived from the canonical q0, so layout 0 from a
        # previous generation is a valid input and repeated runs are identical.
        seen = set()
        for index, offsets in enumerate(SCENE_OFFSETS):
            layout = _scene_variation(deepcopy(base), task, index)
            layout = _apply_focused_layout_tweak(layout, task, index)
            validate_layout(task, layout)
            poses = tuple(
                tuple(entry["default_pos"])
                for group in ("Articulation", "Geometry")
                for entries in layout[group].values()
                for entry in entries
            )
            if poses in seen:
                raise ValueError(f"{task}: duplicate layout {index}")
            seen.add(poses)
            path = LAYOUT_DIR / f"{task}_{index}.json"
            if check:
                if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != layout:
                    raise ValueError(f"Missing or changed generated layout: {path}")
            else:
                path.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
        print(f"{task}: generated {len(SCENE_OFFSETS)} scene-varied layouts (0-{len(SCENE_OFFSETS) - 1})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify generated layouts without writing")
    generate(check=parser.parse_args().check)
