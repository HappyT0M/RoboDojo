"""Pure geometry helpers for deformable-object task metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeformableContainmentMetrics:
    """Containment measurements expressed in the bowl's local frame."""

    center_local: np.ndarray
    center_inside: bool
    contained_fraction: float
    contained_count: int
    valid_count: int


class ConsecutiveConditionCounter:
    """Track how many consecutive updates satisfy a condition."""

    def __init__(self, required_steps: int):
        if not isinstance(required_steps, (int, np.integer)) or required_steps <= 0:
            raise ValueError("required_steps must be a positive integer")
        self.required_steps = int(required_steps)
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def update(self, condition: bool) -> bool:
        self.count = self.count + 1 if bool(condition) else 0
        return self.count >= self.required_steps


def validate_deformable_config(config: dict) -> dict:
    """Validate and normalize dimensions and FEM material configuration."""
    try:
        size = tuple(float(value) for value in config["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("deformable size must contain three positive values") from exc
    if len(size) != 3 or not np.isfinite(size).all() or any(value <= 0 for value in size):
        raise ValueError("deformable size must contain three positive values")

    physics = dict(config.get("physics", {}))
    normalized_physics = {
        "density": float(physics.get("density", 250.0)),
        "youngs_modulus": float(physics.get("youngs_modulus", 120000.0)),
        "poissons_ratio": float(physics.get("poissons_ratio", 0.35)),
        "elasticity_damping": float(physics.get("elasticity_damping", 0.2)),
        "dynamic_friction": float(physics.get("dynamic_friction", 0.6)),
    }
    if not np.isfinite(list(normalized_physics.values())).all():
        raise ValueError("deformable physics values must be finite")
    if normalized_physics["density"] <= 0 or normalized_physics["youngs_modulus"] <= 0:
        raise ValueError("deformable density and youngs_modulus must be positive")
    if not 0.0 < normalized_physics["poissons_ratio"] < 0.5:
        raise ValueError("deformable poissons_ratio must be between 0 and 0.5")
    if normalized_physics["elasticity_damping"] < 0 or normalized_physics["dynamic_friction"] < 0:
        raise ValueError("deformable damping and friction must be non-negative")

    color = tuple(float(value) for value in config.get("color", (0.78, 0.70, 0.43)))
    if len(color) not in (3, 4) or not np.isfinite(color).all() or any(not 0.0 <= value <= 1.0 for value in color):
        raise ValueError("deformable color must contain three or four values in [0, 1]")
    if len(color) == 3:
        color += (1.0,)
    return {"size": size, "physics": normalized_physics, "color": color}


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("bowl_quat_w must be a finite quaternion with shape (4,)")
    norm = np.linalg.norm(quaternion)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("bowl_quat_w quaternion norm must be non-zero")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compute_deformable_containment(
    points_w: np.ndarray,
    bowl_pos_w: np.ndarray,
    bowl_quat_w: np.ndarray,
    interior_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    rim_z: float,
) -> DeformableContainmentMetrics:
    """Measure soft-body node containment using bowl-local interior bounds.

    Args:
        points_w: FEM node positions in world coordinates, shaped ``(N, 3)``.
        bowl_pos_w: Bowl origin in world coordinates.
        bowl_quat_w: Bowl orientation in ``(w, x, y, z)`` order.
        interior_bounds: Lower and upper bowl-interior bounds in bowl-local meters.
        rim_z: Bowl-rim height in bowl-local meters. Nodes above this height do
            not count as contained, even if the upper bound extends farther.

    Returns:
        Center, center-in-interior result, and the fraction of finite FEM nodes
        inside the bowl's horizontal footprint and vertical interior.

    Raises:
        ValueError: If node positions, bowl pose, bounds, or rim height are invalid.
    """
    points_w = np.asarray(points_w, dtype=np.float64)
    bowl_pos_w = np.asarray(bowl_pos_w, dtype=np.float64)
    if points_w.ndim != 2 or points_w.shape[1] != 3 or points_w.shape[0] == 0:
        raise ValueError("points_w must contain at least one point with shape (N, 3)")
    if not np.isfinite(points_w).all():
        raise ValueError("points_w must contain only finite coordinates")
    if bowl_pos_w.shape != (3,) or not np.isfinite(bowl_pos_w).all():
        raise ValueError("bowl_pos_w must be a finite vector with shape (3,)")
    try:
        bounds = np.asarray(interior_bounds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("interior_bounds must contain finite lower and upper 3D bounds") from exc
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError("interior_bounds must contain finite lower and upper 3D bounds")
    if not np.isfinite(rim_z) or rim_z <= bounds[0, 2]:
        raise ValueError("rim_z must be finite and above the bowl interior floor")

    rotation_local_to_world = _quaternion_wxyz_to_matrix(bowl_quat_w)
    points_local = (points_w - bowl_pos_w) @ rotation_local_to_world
    center_local = points_local.mean(axis=0)
    lower, upper = bounds
    effective_upper_z = min(upper[2], float(rim_z))
    inside = np.logical_and(points_local >= lower, points_local <= upper).all(axis=1)
    inside &= points_local[:, 2] <= effective_upper_z
    center_inside = bool(
        np.all(center_local[:2] >= lower[:2])
        and np.all(center_local[:2] <= upper[:2])
        and lower[2] <= center_local[2] <= effective_upper_z
    )
    contained_count = int(inside.sum())
    return DeformableContainmentMetrics(
        center_local=center_local,
        center_inside=center_inside,
        contained_fraction=contained_count / points_local.shape[0],
        contained_count=contained_count,
        valid_count=int(points_local.shape[0]),
    )
