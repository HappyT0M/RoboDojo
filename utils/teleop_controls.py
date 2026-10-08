"""Pure Python helpers for keyboard-based RoboDojo teleoperation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


MOVEMENT_KEYS = frozenset({"W", "S", "A", "D", "R", "F"})
GRIPPER_KEYS = frozenset({"C", "O"})

_TRANSLATION_DIRECTIONS = {
    "W": (0, 1),
    "S": (0, -1),
    "D": (1, 1),
    "A": (1, -1),
    "R": (2, 1),
    "F": (2, -1),
}


def apply_translation_keys(
    target_pose: Iterable[float],
    held_keys: Iterable[str],
    step_m: float,
    bounds: Iterable[Iterable[float]],
) -> np.ndarray:
    """Apply one Cartesian translation increment while preserving orientation.

    W/S move along +/-X, D/A along +/-Y, and R/F along +/-Z. Opposing keys
    cancel. Bounds are absolute minimum/maximum values for each XYZ coordinate.
    """
    pose = np.asarray(target_pose, dtype=np.float64).copy()
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("target_pose must be a finite 7D pose")
    if not math.isfinite(float(step_m)) or float(step_m) <= 0.0:
        raise ValueError("step_m must be positive and finite")

    limits = np.asarray(bounds, dtype=np.float64)
    if (
        limits.shape != (3, 2)
        or not np.all(np.isfinite(limits))
        or np.any(limits[:, 0] > limits[:, 1])
    ):
        raise ValueError("bounds must contain three finite (minimum, maximum) pairs")

    delta = np.zeros(3, dtype=np.float64)
    for key in set(held_keys):
        direction = _TRANSLATION_DIRECTIONS.get(str(key).upper())
        if direction is not None:
            axis, sign = direction
            delta[axis] += sign * float(step_m)

    pose[:3] = np.clip(pose[:3] + delta, limits[:, 0], limits[:, 1])
    return pose


def select_gripper_command(held_keys: Iterable[str], current: float) -> float:
    """Return normalized gripper opening: C closes, O opens, otherwise hold."""
    keys = {str(key).upper() for key in held_keys}
    if "C" in keys and "O" not in keys:
        return 0.0
    if "O" in keys and "C" not in keys:
        return 1.0
    return float(current)


def parse_wind(values: Iterable[float]) -> tuple[float, float, float]:
    """Validate and normalize a world-space constant wind velocity vector."""
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("wind must be a finite 3D vector") from exc
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError("wind must be a finite 3D vector")
    return result


def teleop_record_row(
    step: int,
    held_keys: Iterable[str],
    arm: str,
    target_pose: Iterable[float],
    gripper: float,
    wind: Iterable[float],
    task: str,
    seed: int,
) -> dict:
    """Build a JSON-serializable control sample for the teleop JSONL log."""
    pose = [float(value) for value in target_pose]
    if len(pose) != 7 or not all(math.isfinite(value) for value in pose):
        raise ValueError("target_pose must be a finite 7D pose")
    gripper = float(gripper)
    if not math.isfinite(gripper):
        raise ValueError("gripper command must be finite")
    return {
        "step": int(step),
        "held_keys": sorted({str(key).upper() for key in held_keys}),
        "arm": str(arm),
        "target_pose": pose,
        "gripper": gripper,
        "wind": list(parse_wind(wind)),
        "task": str(task),
        "seed": int(seed),
    }
