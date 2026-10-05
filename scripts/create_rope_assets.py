"""Generate compact, CPU-friendly USD assets and fixed seed layouts for rope tasks.

Uses only Python's standard library so it can run with the RoboDojo Python
environment before Isaac Sim starts. Generated USD files are ASCII USD.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "Assets" / "Object" / "RoboDojo"
LAYOUTS = ROOT / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
ROPE_SEGMENTS = 12
SEGMENT_LENGTH = 0.05
ROPE_RADIUS = 0.0045
BALL_RADIUS = 0.019


def _fmt_vec(values):
    return "(" + ", ".join(f"{float(value):.7g}" for value in values) + ")"


def _mesh_usd(name, points, faces, color):
    counts = ", ".join("3" for _ in faces)
    indices = ", ".join(str(index) for face in faces for index in face)
    point_text = ",\n        ".join(_fmt_vec(point) for point in points)
    return f'''#usda 1.0
(
    defaultPrim = "{name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Mesh "{name}" (
    prepend apiSchemas = ["PhysicsCollisionAPI"]
)
{{
    uniform bool doubleSided = true
    uniform token subdivisionScheme = "none"
    uniform token orientation = "rightHanded"
    point3f[] points = [
        {point_text}
    ]
    int[] faceVertexCounts = [{counts}]
    int[] faceVertexIndices = [{indices}]
    color3f[] primvars:displayColor = [{_fmt_vec(color)}]
    uniform token[] xformOpOrder = []
}}
'''


def _add_box(points, faces, center, size):
    cx, cy, cz = center
    sx, sy, sz = (value / 2 for value in size)
    start = len(points)
    points.extend(
        [
            (cx - sx, cy - sy, cz - sz),
            (cx + sx, cy - sy, cz - sz),
            (cx + sx, cy + sy, cz - sz),
            (cx - sx, cy + sy, cz - sz),
            (cx - sx, cy - sy, cz + sz),
            (cx + sx, cy - sy, cz + sz),
            (cx + sx, cy + sy, cz + sz),
            (cx - sx, cy + sy, cz + sz),
        ]
    )
    for face in (
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ):
        faces.append(tuple(start + index for index in face))


def _slot_mesh(width=0.22, depth=0.14, height=0.06, wall=0.008):
    points, faces = [], []
    _add_box(points, faces, (0, 0, wall / 2), (width, depth, wall))
    _add_box(points, faces, (0, depth / 2 - wall / 2, height / 2), (width, wall, height))
    _add_box(points, faces, (-width / 2 + wall / 2, 0, height / 2), (wall, depth, height))
    _add_box(points, faces, (width / 2 - wall / 2, 0, height / 2), (wall, depth, height))
    return points, faces


def _ring_mesh(major_radius=0.105, tube_radius=0.009, major_steps=64, minor_steps=12):
    points, faces = [], []
    for i in range(major_steps):
        theta = 2.0 * math.pi * i / major_steps
        for j in range(minor_steps):
            phi = 2.0 * math.pi * j / minor_steps
            radial = major_radius + tube_radius * math.cos(phi)
            # Ring plane is XZ; its normal points along +Y.
            points.append((radial * math.cos(theta), tube_radius * math.sin(phi), radial * math.sin(theta)))
    for i in range(major_steps):
        for j in range(minor_steps):
            a = i * minor_steps + j
            b = ((i + 1) % major_steps) * minor_steps + j
            c = ((i + 1) % major_steps) * minor_steps + (j + 1) % minor_steps
            d = i * minor_steps + (j + 1) % minor_steps
            faces.extend(((a, b, c), (a, c, d)))
    return points, faces


def _post_mesh(radius=0.024, height=0.14, steps=24):
    points, faces = [], []
    for z in (0.0, height):
        points.extend(
            (radius * math.cos(2 * math.pi * i / steps), radius * math.sin(2 * math.pi * i / steps), z)
            for i in range(steps)
        )
    for i in range(steps):
        nxt = (i + 1) % steps
        faces.extend(((i, nxt, steps + nxt), (i, steps + nxt, steps + i)))
    for i in range(1, steps - 1):
        faces.append((0, i + 1, i))
        faces.append((steps, steps + i, steps + i + 1))
    return points, faces


def _metadata(points, faces, asset_name):
    mins = [min(point[axis] for point in points) for axis in range(3)]
    maxs = [max(point[axis] for point in points) for axis in range(3)]
    vertices = [
        (x, y, z)
        for x in (mins[0], maxs[0])
        for y in (mins[1], maxs[1])
        for z in (mins[2], maxs[2])
    ]
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"robodojo://cpu-rope/{asset_name}")),
        "geometry": {
            "faces": len(faces),
            "vertices": len(points),
            "aligned_bbox": {
                "vertices": vertices,
                "extents": [maxs[i] - mins[i] for i in range(3)],
            },
        },
        "physics": {},
        "visual": {},
        "active": {},
        "passive": {},
    }


def _write_geometry_asset(name, points, faces, color):
    directory = ASSETS / "Geometry" / name / "00000"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "object.usd").write_text(_mesh_usd("asset", points, faces, color), encoding="utf-8")
    (directory / "metadata.json").write_text(
        json.dumps(_metadata(points, faces, name), indent=2), encoding="utf-8"
    )


def _link_xform(name, x, radius, color, mass, shape="capsule"):
    if shape == "capsule":
        geom = f'''def Capsule "collision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            uniform token axis = "X"
            double radius = {radius}
            double height = {SEGMENT_LENGTH - 2 * radius}
            color3f[] primvars:displayColor = [{_fmt_vec(color)}]
        }}'''
    else:
        geom = f'''def Sphere "collision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            double radius = {radius}
            color3f[] primvars:displayColor = [{_fmt_vec(color)}]
        }}'''
    return f'''    def Xform "{name}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        bool physics:rigidBodyEnabled = true
        float physics:mass = {mass}
        double3 xformOp:translate = ({x:.7g}, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        {geom}
    }}
'''


def _rope_usd():
    rope_color = (0.82, 0.78, 0.68)
    ball_color = (0.12, 0.45, 0.82)
    pieces = [
        '''#usda 1.0
(
    defaultPrim = "rope_chain"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "rope_chain" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxSchemaPhysxArticulationAPI"]
)
{
    bool physxArticulation:enabledSelfCollisions = false
'''
    ]
    for index in range(ROPE_SEGMENTS):
        pieces.append(
            _link_xform(
                f"segment_{index:02d}",
                index * SEGMENT_LENGTH,
                ROPE_RADIUS,
                rope_color,
                0.001,
            )
        )
    ball_x = (ROPE_SEGMENTS - 1) * SEGMENT_LENGTH + SEGMENT_LENGTH / 2 + BALL_RADIUS
    pieces.append(_link_xform("ball", ball_x, BALL_RADIUS, ball_color, 0.018, shape="sphere"))

    for index in range(ROPE_SEGMENTS - 1):
        axis = "Y" if index % 2 == 0 else "Z"
        pieces.append(
            f'''    def PhysicsRevoluteJoint "joint_{index:02d}" (
        prepend apiSchemas = ["PhysicsJointAPI"]
    )
    {{
        rel physics:body0 = </rope_chain/segment_{index:02d}>
        rel physics:body1 = </rope_chain/segment_{index + 1:02d}>
        point3f physics:localPos0 = ({SEGMENT_LENGTH / 2}, 0, 0)
        point3f physics:localPos1 = ({-SEGMENT_LENGTH / 2}, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
        uniform token physics:axis = "{axis}"
        float physics:lowerLimit = -100
        float physics:upperLimit = 100
    }}
'''
        )
    pieces.append(
        f'''    def PhysicsFixedJoint "joint_ball" (
        prepend apiSchemas = ["PhysicsJointAPI"]
    )
    {{
        rel physics:body0 = </rope_chain/segment_{ROPE_SEGMENTS - 1:02d}>
        rel physics:body1 = </rope_chain/ball>
        point3f physics:localPos0 = ({SEGMENT_LENGTH / 2}, 0, 0)
        point3f physics:localPos1 = ({-BALL_RADIUS}, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
    }}
}}
'''
    )
    return "".join(pieces)


def _write_rope_asset():
    directory = ASSETS / "Articulation" / "rope_chain" / "00000"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "object.usd").write_text(_rope_usd(), encoding="utf-8")
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, "robodojo://cpu-rope/rope-chain")),
                "geometry": {
                    "vertices": ROPE_SEGMENTS * 12 + 32,
                    "faces": ROPE_SEGMENTS * 20 + 64,
                    "aligned_bbox": {
                        "vertices": [
                            [x, y, z]
                            for x in (0, 0.62)
                            for y in (-0.01, 0.01)
                            for z in (-0.01, 0.01)
                        ],
                        "extents": [0.62, 0.02, 0.02],
                    },
                },
                "physics": {"link_count": ROPE_SEGMENTS, "rope_radius": ROPE_RADIUS, "ball_radius": BALL_RADIUS},
                "visual": {},
                "active": {},
                "passive": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _entry(
    category,
    label,
    position,
    quaternion=(1, 0, 0, 0),
    object_type="geometry",
    scale=(1, 1, 1),
    category_idx=0,
):
    entry = {
        "category": category,
        "category_idx": category_idx,
        "group": None,
        "label": label,
        "default_pos": list(position),
        "default_ori": list(quaternion),
        "scale": list(scale),
        "relative_plane": "Table",
        "need_check_stable": object_type != "articulation",
        "physics": {"type": object_type},
        "visual": {},
    }
    if object_type == "articulation":
        entry["physics"]["reset_drive"] = True
    if object_type == "geometry":
        entry["physics"]["collision"] = True
    return entry


def _layout(objects):
    return {
        **objects,
        "Room": {
            "default": "Simple_Room_nolight",
            "default_pos": [0.0, 0.0, 0.0],
            "default_rot": [1.0, 0.0, 0.0, 0.0],
            "scale": [0.5, 0.5, 0.5],
        },
        "Table": {
            "default": "material_0122",
            "scale": [1.4, 1.1, 0.05],
            "default_pos": [0.0, -0.05, 0.74],
            "default_ori": [1.0, 0.0, 0.0, 0.0],
        },
        "Ground": {
            "geometry": "cube",
            "thickness": 0.1,
            "axis": "Z",
            "default_pos": [0.0, 0.0, 0.0],
            "materials": {"type": "mdl", "default": "material_0564", "random": False},
            "physics_material": {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0,
                "friction_combine_mode": "multiply",
                "restitution_combine_mode": "multiply",
                "compliant_contact_stiffness": 0.0,
                "compliant_contact_damping": 0.0,
            },
        },
        "Background": {"intensity": 1000, "category_name": "brown_photostudio_02_4k.hdr"},
    }


def _write_layout(task_name, objects):
    LAYOUTS.mkdir(parents=True, exist_ok=True)
    (LAYOUTS / f"{task_name}_0.json").write_text(json.dumps(_layout(objects), indent=2), encoding="utf-8")


def generate_assets():
    _write_rope_asset()
    ring, ring_faces = _ring_mesh()
    _write_geometry_asset("rope_ring", ring, ring_faces, (0.78, 0.78, 0.78))
    post, post_faces = _post_mesh()
    _write_geometry_asset("rope_post", post, post_faces, (0.34, 0.38, 0.42))
    slot, slot_faces = _slot_mesh()
    _write_geometry_asset("rope_slot", slot, slot_faces, (0.18, 0.48, 0.36))

    _write_layout(
        "put_rope_ball_in_basket",
        {
            "Articulation": {"rope_chain": [_entry("rope_chain", "rope", [-0.28, -0.12, 0.79], object_type="articulation")]},
            "Geometry": {
                "basket": [_entry("basket", "basket", [0.34, 0.14, 0.8035], category_idx=2)],
            },
        },
    )
    _write_layout(
        "pass_rope_through_ring",
        {
            "Articulation": {
                "rope_chain": [
                    _entry("rope_chain", "rope", [-0.15, -0.50, 0.79], (math.sqrt(0.5), 0, 0, math.sqrt(0.5)), "articulation")
                ]
            },
            "Geometry": {"rope_ring": [_entry("rope_ring", "ring", [-0.15, 0.15, 0.87])]},
        },
    )
    _write_layout(
        "route_rope_around_posts",
        {
            "Articulation": {"rope_chain": [_entry("rope_chain", "rope", [-0.34, -0.18, 0.79], object_type="articulation")]},
            "Geometry": {
                "rope_post": [
                    _entry("rope_post", "post0", [-0.15, -0.10, 0.765]),
                    _entry("rope_post", "post1", [0.06, -0.10, 0.765]),
                ],
                "rope_slot": [_entry("rope_slot", "target_slot", [0.30, 0.15, 0.765])],
            },
        },
    )
    print("Generated CPU rope chain, three fixtures, metadata, and three seed-0 layouts.")


if __name__ == "__main__":
    generate_assets()
