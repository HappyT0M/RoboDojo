"""Geometry and state helpers for the CPU rigid-link rope tasks."""

from dataclasses import dataclass
import math
from typing import Sequence


def _vec3(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) < 3:
        raise ValueError("Expected a 3D vector")
    return (float(value[0]), float(value[1]), float(value[2]))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _normalize(value: Sequence[float]) -> tuple[float, float, float]:
    vector = _vec3(value)
    magnitude = math.sqrt(_dot(vector, vector))
    if magnitude <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return tuple(component / magnitude for component in vector)


def _rotate_by_quaternion(vector: Sequence[float], quaternion_wxyz: Sequence[float]) -> tuple[float, float, float]:
    """Rotate a vector by a unit quaternion in Isaac's wxyz convention."""
    x, y, z = _vec3(vector)
    if len(quaternion_wxyz) != 4:
        raise ValueError("Expected a quaternion in wxyz order")
    w, qx, qy, qz = (float(value) for value in quaternion_wxyz)
    # q*v*q^-1, expressed without constructing temporary quaternions.
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + w * tx + (qy * tz - qz * ty),
        y + w * ty + (qz * tx - qx * tz),
        z + w * tz + (qx * ty - qy * tx),
    )


def point_inside_oriented_box(
    point: Sequence[float],
    box_position: Sequence[float],
    box_quaternion_wxyz: Sequence[float],
    bounds: Sequence[Sequence[float]],
) -> bool:
    """Return whether a world-space point is inside local xyz bounds."""
    if len(bounds) != 3 or any(len(axis) != 2 for axis in bounds):
        raise ValueError("bounds must contain [min, max] for each of three axes")
    delta = tuple(a - b for a, b in zip(_vec3(point), _vec3(box_position)))
    w, x, y, z = (float(value) for value in box_quaternion_wxyz)
    local = _rotate_by_quaternion(delta, (w, -x, -y, -z))
    return all(float(low) <= value <= float(high) for value, (low, high) in zip(local, bounds))


def crosses_ring_aperture(
    previous: Sequence[float],
    current: Sequence[float],
    center: Sequence[float],
    normal: Sequence[float],
    inner_radius: float,
    margin: float = 0.0,
) -> bool:
    """Check a directed segment crossing the ring plane through its aperture."""
    if inner_radius <= 0 or margin < 0 or margin >= inner_radius:
        raise ValueError("Require inner_radius > margin >= 0")
    prev = _vec3(previous)
    curr = _vec3(current)
    center = _vec3(center)
    normal = _normalize(normal)
    d0 = _dot(tuple(a - b for a, b in zip(prev, center)), normal)
    d1 = _dot(tuple(a - b for a, b in zip(curr, center)), normal)
    if d0 >= 0 or d1 < 0 or d1 - d0 <= 1e-12:
        return False
    ratio = -d0 / (d1 - d0)
    crossing = tuple(a + ratio * (b - a) for a, b in zip(prev, curr))
    offset = tuple(a - b for a, b in zip(crossing, center))
    normal_component = _dot(offset, normal)
    radial = tuple(value - normal_component * axis for value, axis in zip(offset, normal))
    return math.sqrt(_dot(radial, radial)) <= inner_radius - margin


def ball_in_directed_far_zone(
    position: Sequence[float],
    center: Sequence[float],
    normal: Sequence[float],
    min_distance: float,
    max_distance: float,
    tangential_radius: float,
) -> bool:
    """Check the ball lies in a bounded target zone beyond a ring plane."""
    if min_distance < 0 or max_distance <= min_distance or tangential_radius <= 0:
        raise ValueError("Invalid far-side target-zone dimensions")
    normal = _normalize(normal)
    offset = tuple(a - b for a, b in zip(_vec3(position), _vec3(center)))
    signed_distance = _dot(offset, normal)
    tangent = tuple(value - signed_distance * axis for value, axis in zip(offset, normal))
    return (
        min_distance <= signed_distance <= max_distance
        and math.sqrt(_dot(tangent, tangent)) <= tangential_radius
    )


def polyline_wraps_post(
    points: Sequence[Sequence[float]],
    post_center_xy: Sequence[float],
    post_radius: float,
    rope_radius: float = 0.0045,
    min_arc_radians: float = math.radians(100.0),
) -> bool:
    """Approximate a rope wrap by a contiguous chain arc around a post."""
    if len(post_center_xy) < 2:
        raise ValueError("post_center_xy must contain x and y")
    if post_radius <= 0 or rope_radius < 0 or not 0 < min_arc_radians <= 2 * math.pi:
        raise ValueError("Invalid post wrap geometry")
    cx, cy = float(post_center_xy[0]), float(post_center_xy[1])
    min_centerline_radius = max(0.0, post_radius - rope_radius * 1.5)
    max_centerline_radius = post_radius + rope_radius * 3.0 + 0.015
    angles = []
    for point in points:
        x, y, _ = _vec3(point)
        dx, dy = x - cx, y - cy
        radius = math.hypot(dx, dy)
        if min_centerline_radius <= radius <= max_centerline_radius:
            angles.append(math.atan2(dy, dx))
    if len(angles) < 3:
        return False

    unwrapped = [angles[0]]
    for angle in angles[1:]:
        delta = (angle - unwrapped[-1] + math.pi) % (2.0 * math.pi) - math.pi
        unwrapped.append(unwrapped[-1] + delta)
    return abs(unwrapped[-1] - unwrapped[0]) >= min_arc_radians


def gripper_holds_rope(
    gripper_position: Sequence[float],
    rope_points: Sequence[Sequence[float]],
    gripper_open_fraction: float,
    table_height: float,
    max_contact_distance: float = 0.055,
    max_open_fraction: float = 0.25,
    min_lift_height: float = 0.025,
) -> bool:
    """Approximate a grasp using a closed gripper near a lifted rope link."""
    if not 0.0 <= gripper_open_fraction <= 1.0:
        raise ValueError("gripper_open_fraction must be in [0, 1]")
    if not rope_points:
        return False
    gripper = _vec3(gripper_position)
    return (
        gripper_open_fraction <= max_open_fraction
        and any(
            point[2] >= table_height + min_lift_height
            and math.dist(gripper, _vec3(point)) <= max_contact_distance
            for point in rope_points
        )
    )


@dataclass
class ConsecutiveSuccess:
    """Track whether a condition has held for a consecutive step window."""

    required_steps: int = 20
    count: int = 0

    def __post_init__(self):
        if self.required_steps <= 0:
            raise ValueError("required_steps must be positive")

    def update(self, condition: bool) -> bool:
        self.count = self.count + 1 if condition else 0
        return self.count >= self.required_steps

    def reset(self) -> None:
        self.count = 0


def resolve_physx_gpu_override(sim_config) -> int:
    """Map the optional task setting to PhysX's CPU/GPU override value."""
    return 0 if sim_config.get("gpu_dynamics_enabled", True) is False else 1
