"""Simulator-independent geometry and episode state for cover, pull and wipe."""

import math

from utils.rope_task_metrics import ConsecutiveSuccess, _rotate_by_quaternion


def local_points(points, position, quaternion, scale=(1, 1, 1)):
    norm = math.sqrt(sum(float(v) ** 2 for v in quaternion))
    if norm < 1e-9 or any(float(s) <= 0 for s in scale):
        raise ValueError("Invalid target transform")
    w, x, y, z = (float(v) / norm for v in quaternion)
    return [tuple(v / float(s) for v, s in zip(
        _rotate_by_quaternion(tuple(float(a) - float(b) for a, b in zip(point, position)), (w, -x, -y, -z)),
        scale,
    )) for point in points]


def rectangular_samples(half_extents, shape=(11, 9)):
    """Include the boundary, so uncovered edges cannot hide between cell centers."""
    hx, hy = half_extents
    nx, ny = shape
    return [(-hx + 2 * hx * x / (nx - 1), -hy + 2 * hy * y / (ny - 1))
            for x in range(nx) for y in range(ny)]


def surface_coverage(points, triangles, samples, z_min, z_max):
    """Vertical ray/triangle intersections; a cloth's convex hull is insufficient."""
    covered = set()
    for face in triangles:
        a, b, c = (points[index] for index in face)
        determinant = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(determinant) < 1e-12:
            continue
        if min(a[2], b[2], c[2]) > z_max or max(a[2], b[2], c[2]) < z_min:
            continue
        xmin, xmax = min(a[0], b[0], c[0]), max(a[0], b[0], c[0])
        ymin, ymax = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])
        for index, (x, y) in enumerate(samples):
            if index in covered or not (xmin - 1e-9 <= x <= xmax + 1e-9 and ymin - 1e-9 <= y <= ymax + 1e-9):
                continue
            u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / determinant
            v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / determinant
            t = 1 - u - v
            if min(u, v, t) >= -1e-8 and z_min <= u * a[2] + v * b[2] + t * c[2] <= z_max:
                covered.add(index)
    return covered


def held_by(points, grippers, distance=0.045, max_open=0.25):
    """Closed gripper near cloth/rope vertices: geometric grasp proxy."""
    for index, (position, openness) in enumerate(grippers):
        if openness <= max_open and any(math.dist(position, p) <= distance for p in points):
            return index
    return None


def finite_points(points):
    return bool(len(points)) and all(len(p) == 3 and all(math.isfinite(float(v)) for v in p) for p in points)


class CoverProgress:
    def __init__(self, config):
        self.config = config
        self.samples = rectangular_samples(config["target_half_extents"])
        self.counter = ConsecutiveSuccess(int(config.get("stable_steps", 15)))
        self.previous = None
        self.manipulated = False
        self.coverage = 0.0

    def update(self, points, triangles, grippers):
        if not finite_points(points):
            self.previous = None
            return self.counter.update(False)
        held = held_by(points, grippers) is not None
        self.manipulated |= held
        top = float(self.config["surface_height"])
        covered = surface_coverage(points, triangles, self.samples, top - .003, top + .025)
        self.coverage = len(covered) / len(self.samples)
        stable = self.previous is not None and len(points) == len(self.previous) and max(
            math.dist(a, b) for a, b in zip(points, self.previous)
        ) <= float(self.config.get("maximum_step_motion", .008))
        self.previous = [tuple(p) for p in points]
        return self.counter.update(self.manipulated and not held and stable and
                                   self.coverage >= float(self.config.get("minimum_coverage", .98)))


class WipeProgress:
    def __init__(self, config):
        self.config = config
        self.samples = rectangular_samples(config["target_half_extents"], (15, 11))
        self.covered = set()
        self.path_length = 0.0
        self.previous = None
        self.previous_grippers = []
        self.previous_holder = None

    def update(self, points, triangles, grippers):
        holder = held_by(points, grippers) if finite_points(points) else None
        top = float(self.config["surface_height"])
        tolerance = float(self.config.get("contact_height_tolerance", .015))
        if (holder is not None and holder == self.previous_holder and self.previous is not None
                and len(self.previous) == len(points) and holder < len(self.previous_grippers)):
            hand_motion = math.dist(grippers[holder][0][:2], self.previous_grippers[holder][0][:2])
            contacts = [i for i, p in enumerate(points)
                        if top - .003 <= p[2] <= top + tolerance
                        and top - .003 <= self.previous[i][2] <= top + tolerance]
            cloth_motion = (sum(math.dist(points[i][:2], self.previous[i][:2]) for i in contacts)
                            / len(contacts)) if contacts else 0.0
            minimum = float(self.config.get("minimum_step_motion", .001))
            if len(contacts) >= 3 and min(hand_motion, cloth_motion) >= minimum:
                # Do not bridge unobserved jumps or accumulate stationary/wind-only coverage.
                if max(hand_motion, cloth_motion) <= .08:
                    self.path_length += min(hand_motion, cloth_motion)
                    self.covered.update(surface_coverage(points, triangles, self.samples, top - .003, top + tolerance))
        self.previous = [tuple(p) for p in points] if finite_points(points) else None
        self.previous_grippers = [(tuple(p), v) for p, v in grippers]
        self.previous_holder = holder
        return (len(self.covered) / len(self.samples) >= float(self.config.get("minimum_coverage", .90))
                and self.path_length >= float(self.config.get("minimum_wipe_distance", .15)))


class PullProgress:
    def __init__(self, config):
        self.config = config
        self.counter = ConsecutiveSuccess(int(config.get("stable_steps", 15)))
        self.initial_ball = None
        self.previous_ball = None
        self.previous_rope = None
        self.previous_grippers = []
        self.previous_holder = None
        self.pull_distance = 0.0
        self.touched_ball = False

    def update(self, rope_points, ball, grippers):
        if not finite_points([*rope_points, ball]):
            self.previous_ball = None
            self.previous_holder = None
            return self.counter.update(False)
        if self.initial_ball is None:
            self.initial_ball = tuple(ball)
        self.touched_ball |= any(math.dist(p, ball) < float(self.config.get("ball_exclusion_radius", .06))
                                 for p, _ in grippers)
        free_end = rope_points[:int(self.config.get("free_end_links", 3))]
        holder = held_by(free_end, grippers)
        if (holder is not None and holder == self.previous_holder and self.previous_ball is not None
                and holder < len(self.previous_grippers)):
            # Reward only simultaneous movement of hand, free rope end and ball toward the goal.
            hand_motion = math.dist(grippers[holder][0], self.previous_grippers[holder][0])
            rope_motion = math.dist(rope_points[0], self.previous_rope)
            progress = math.hypot(*self.previous_ball[:2]) - math.hypot(*ball[:2])
            if min(hand_motion, rope_motion) >= .001 and progress > 0:
                self.pull_distance += min(progress, hand_motion, rope_motion)
        self.previous_ball = tuple(ball)
        self.previous_rope = tuple(rope_points[0])
        self.previous_grippers = [(tuple(p), v) for p, v in grippers]
        self.previous_holder = holder
        hx, hy = self.config["target_half_extents"]
        low, high = self.config.get("ball_height_range", [.012, .055])
        inside = abs(ball[0]) <= hx and abs(ball[1]) <= hy and low <= ball[2] <= high
        moved = math.dist(self.initial_ball[:2], ball[:2]) >= float(self.config.get("minimum_ball_travel", .18))
        return self.counter.update(inside and moved and not self.touched_ball and
                                   self.pull_distance >= float(self.config.get("minimum_pull_distance", .12)))
