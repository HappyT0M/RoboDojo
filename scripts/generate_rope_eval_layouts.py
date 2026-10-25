"""Build ten reproducible eval layouts per CPU rope task from layout 0.

Only tabletop XY poses change. Fixture orientations, rope articulation state,
task geometry, and the rest of the scene stay identical. Run with ``--check``
to verify the generated files without writing them.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAYOUT_DIR = ROOT / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
TASKS = (
    "put_rope_ball_in_basket",
    "pass_rope_through_ring",
    "route_rope_around_posts",
)
BASE_POSES = {
    "put_rope_ball_in_basket": {
        "rope_chain": (-0.28, -0.12, 0.79),
        "basket": (0.34, 0.14, 0.8035),
    },
    "pass_rope_through_ring": {
        "rope_chain": (-0.28, -0.12, 0.79),
        "frame": (0.2, 0.05, 0.915),
    },
    "route_rope_around_posts": {
        "rope_chain": (-0.594, -0.20, 0.79),
        "vase": (0.0, 0.0, 0.84053),
        "mousemat": (0.0, 0.24, 0.76865),
    },
}

# (whole-scene X, whole-scene Y, rope-relative X, rope-relative Y), in metres.
# The original layout is index 0. Translation of every fixture together keeps
# the existing task-space reward geometry valid; the rope offset also changes
# the initial grasp and the rope-to-goal relationship.
OFFSETS = (
    (0.000, 0.000, 0.000, 0.000),
    (0.025, 0.000, 0.015, -0.010),
    (-0.025, 0.000, -0.015, 0.010),
    (0.000, 0.025, 0.020, 0.000),
    (0.000, -0.025, -0.020, 0.000),
    (0.020, 0.020, -0.025, -0.015),
    (-0.020, 0.020, 0.025, 0.015),
    (0.020, -0.020, 0.000, 0.025),
    (-0.020, -0.020, 0.000, -0.025),
    (0.000, 0.000, 0.030, -0.030),
)

BALL_CENTER_X = 0.594  # (12 - 1) * 0.05 + 0.05 / 2 + 0.019 in the USD rest pose.
ROPE_END_X = BALL_CENTER_X + 0.019
TABLE_X = (-0.70, 0.70)
TABLE_Y = (-0.60, 0.50)


def _shift_xy(entry: dict, dx: float, dy: float) -> None:
    position = entry["default_pos"]
    entry["default_pos"] = [
        round(position[0] + dx, 6),
        round(position[1] + dy, 6),
        position[2],
    ]


def make_layout(base: dict, offsets: tuple[float, float, float, float]) -> dict:
    scene_x, scene_y, rope_x, rope_y = offsets
    layout = deepcopy(base)
    for entries in layout["Articulation"].values():
        for entry in entries:
            _shift_xy(entry, scene_x + rope_x, scene_y + rope_y)
    for entries in layout["Geometry"].values():
        for entry in entries:
            _shift_xy(entry, scene_x, scene_y)
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
        frame = geometry["frame"][0]["default_pos"]
        # The configured directed far side is (-X, +Y); keep the nominal ball
        # clearly on the near side before any policy action or wind step.
        near_side_distance = ((frame[0] - ball_x) + (ball_y - frame[1])) / 2**0.5
        if near_side_distance >= -0.10:
            raise ValueError(f"{task}: ball does not start clear of the frame")
    elif task == "route_rope_around_posts":
        vase = geometry["vase"][0]["default_pos"]
        mat = geometry["mousemat"][0]["default_pos"]
        if ball_y >= vase[1] - 0.12 or mat[1] <= vase[1] + 0.15:
            raise ValueError(f"{task}: near-side start or far-side goal is lost")
    else:
        raise ValueError(f"Unknown rope task: {task}")


def generate(*, check: bool = False) -> None:
    for task in TASKS:
        base_path = LAYOUT_DIR / f"{task}_0.json"
        base = json.loads(base_path.read_text(encoding="utf-8"))
        for category, position in BASE_POSES[task].items():
            group = "Articulation" if category == "rope_chain" else "Geometry"
            actual = base[group][category][0]["default_pos"]
            if any(abs(a - b) > 1e-6 for a, b in zip(actual, position)):
                raise ValueError(f"Unexpected base pose for {task}/{category}: {actual}")
        if task == "pass_rope_through_ring":
            expected_orientation = (0.6532814824, 0.6532814824, 0.2705980501, 0.2705980501)
            actual_orientation = base["Geometry"]["frame"][0]["default_ori"]
            if any(abs(a - b) > 1e-6 for a, b in zip(actual_orientation, expected_orientation)):
                raise ValueError(f"Unexpected frame orientation: {actual_orientation}")
        seen = set()
        for index, offsets in enumerate(OFFSETS):
            layout = make_layout(base, offsets)
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
            if index == 0:
                if layout != base:
                    raise ValueError(f"{task}: layout 0 was modified")
                continue
            if check:
                if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != layout:
                    raise ValueError(f"Missing or changed generated layout: {path}")
            else:
                if path.exists() and json.loads(path.read_text(encoding="utf-8")) != layout:
                    raise ValueError(f"Refusing to overwrite changed layout: {path}")
                path.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
        print(f"{task}: verified {len(OFFSETS)} layouts (0-{len(OFFSETS) - 1})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify generated layouts without writing")
    generate(check=parser.parse_args().check)
