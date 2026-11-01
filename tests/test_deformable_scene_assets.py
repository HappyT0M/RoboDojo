"""Fixture composition and placement regressions; no Isaac Sim startup needed."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import create_rope_assets as rope
from scripts import generate_deformable_task_assets as garment
from scripts import generate_rope_eval_layouts as layouts
from utils.rope_task_metrics import _rotate_by_quaternion

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    Usd = None


class DeformableSceneAssetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.assets = self.root / "Object"
        self.layouts = self.root / "Layout"
        self.layouts.mkdir()
        for module, key, value in ((rope, "ASSETS", self.assets),
                                   (garment.rope, "ASSETS", self.assets),
                                   (layouts, "LAYOUT_DIR", self.layouts)):
            patcher = patch.object(module, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Official frame bounds, in source asset metres before nonuniform scale.
        path = self.assets / "Geometry/frame/00000/metadata.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"geometry": {"aligned_bbox": {"vertices": [
            [x, y, z] for x in (-.12, .12) for y in (-.05, .05) for z in (0, .0003)
        ]}}}))
        bases = {
            "put_rope_ball_in_basket": {"basket": [rope._entry("basket", "basket", (.34, .14, .8035), category_idx=2)]},
            "pass_rope_through_ring": {"frame": [rope.frame_target_entry()]},
            "route_rope_around_posts": {},
        }
        for task, geometry in bases.items():
            layout = rope._layout({
                "Geometry": geometry,
                "Articulation": {"rope_chain": [rope._entry("rope_chain", "rope", (-.28, -.12, .79), object_type="articulation")]},
            })
            (self.layouts / f"{task}_0.json").write_text(json.dumps(layout))

    def test_rotated_frame_bottom_clears_table_and_normal_is_preserved(self):
        base = json.loads((self.layouts / "pass_rope_through_ring_0.json").read_text())
        normal = _rotate_by_quaternion((0, 0, -1), rope.frame_target_entry()["default_ori"])
        for index in range(10):
            layout = layouts._scene_variation(json.loads(json.dumps(base)), "pass_rope_through_ring", index)
            frame = layout["Geometry"]["frame"][0]
            q = frame["default_ori"]
            # Independently rotate all eight scaled corners, rather than reuse
            # the placement helper whose sign/order this regression is checking.
            bottom = min(_rotate_by_quaternion((x*1.5, y*3, z*50), q)[2]
                         for x in (-.12, .12) for y in (-.05, .05) for z in (0, .0003))
            self.assertAlmostEqual(frame["default_pos"][2]+bottom, .767, places=6)
            for actual, expected in zip(_rotate_by_quaternion((0, 0, -1), q), normal):
                self.assertAlmostEqual(actual, expected, places=7)
            layouts.validate_layout("pass_rope_through_ring", layout)
            frame["default_pos"][2] -= .01
            with self.assertRaisesRegex(ValueError, "intersects the table"):
                layouts.validate_layout("pass_rope_through_ring", layout)

    def test_rope_generation_is_repeatable_from_generated_layout_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            layouts.generate()
            first = {p.name: p.read_bytes() for p in self.layouts.glob("*.json")}
            layouts.generate()
            layouts.generate(check=True)
        self.assertEqual(len(first), 30)
        self.assertEqual(first, {p.name: p.read_bytes() for p in self.layouts.glob("*.json")})

    def make_fixtures(self):
        rope.update_fixture_assets()
        garment.make_assets()
        return sorted(self.assets.glob("Geometry/*/00000/object.usd"))

    def test_fixture_assets_have_child_mesh_and_concave_collisions(self):
        paths = self.make_fixtures()
        self.assertEqual(len(paths), 6)
        for path in paths:
            text = path.read_text()
            self.assertIn('def Xform "asset"', text)
            self.assertIn('def Mesh "mesh"', text)
            self.assertNotIn('def Mesh "asset"', text)
            self.assertIn('uniform token physics:approximation = "none"', text)
            self.assertIn('rel material:binding = </asset/Looks/Surface>', text)
            self.assertIn('float inputs:roughness = 0.8', text)

    @unittest.skipIf(Usd is None, "OpenUSD (pxr) unavailable; run this test in the simulation environment")
    def test_real_usd_reference_keeps_render_mesh_collision_and_material(self):
        for path in self.make_fixtures():
            with self.subTest(asset=path.parent.parent.name):
                # Reproduce add_reference_to_stage's explicitly typed Xform.
                stage = Usd.Stage.CreateInMemory()
                root = stage.DefinePrim("/World/fixture", "Xform")
                root.GetReferences().AddReference(str(path))
                meshes = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
                self.assertEqual(len(meshes), 1)
                mesh = meshes[0]
                self.assertTrue(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
                self.assertTrue(UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get())
                self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get(), "none")
                material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
                self.assertTrue(material)
                self.assertEqual(str(material.GetPath()), "/World/fixture/Looks/Surface")


if __name__ == "__main__":
    unittest.main()
