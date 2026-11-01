"""Guard against replacing the native benchmark's saved scenes."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import generate_deformable_task_assets as generator


class DeformableLayoutGenerationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.layouts = Path(directory.name)
        patcher = patch.object(generator, "LAYOUTS", self.layouts)
        patcher.start()
        self.addCleanup(patcher.stop)
        # A centered native garment has vertices below its origin. Test without
        # requiring a local copy of the official multi-GB asset installation.
        self.assets = self.layouts / "objects"
        for index in generator.GARMENTS:
            path = self.assets / "Garment/Top_Long" / f"{index:05d}" / "metadata.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"geometry": {"aligned_bbox": {"vertices": [
                [x, y, z] for x in (-.30, .30) for y in (-.17, .17) for z in (-.08, .12)
            ]}}}))
        patcher = patch.object(generator.rope, "ASSETS", self.assets)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_custom_scene_generation_preserves_native_layouts(self):
        native_files = {
            "fold_clothes_0.json": b'{"native": 0}\n',
            "fold_clothes_9.json": b'{"native": 9}\n',
            "fold_clothes_10.json": b'{"native": 10}\n',
            "fold_clothes_random_0.json": b'{"native_random": 0}\n',
        }
        for name, data in native_files.items():
            (self.layouts / name).write_bytes(data)

        # Regeneration must remain safe with both native and custom files present.
        for _ in range(2):
            generator.make_new_task_layouts()
            for name, data in native_files.items():
                self.assertEqual((self.layouts / name).read_bytes(), data)

        generated_names = {p.name for p in self.layouts.glob("*.json")} - native_files.keys()
        self.assertEqual(len(generated_names), 40)
        folding_paths = sorted(self.layouts.glob("fold_clothes_deformable_*.json"))
        self.assertEqual(len(folding_paths), 10)
        for path in folding_paths:
            layout = json.loads(path.read_text())
            self.assertIn("folding_board", layout["Geometry"])

    def test_cloth_starts_above_support_with_matte_nonmetallic_material(self):
        for task in generator.TASKS:
            for index in range(10):
                with self.subTest(task=task, index=index):
                    layout = generator.garment_scene(task, index)
                    cloth = layout["Garment"]["Top_Long"][0]
                    support = generator.rope.table_top(layout)
                    if "folding_board" in layout["Geometry"]:
                        support += .024
                    self.assertAlmostEqual(cloth["default_pos"][2] - .08, support + .01)
                    self.assertEqual(cloth["visual"]["metallic"], 0)
                    self.assertEqual(cloth["visual"]["roughness"], .9)
                    for entries in layout["Geometry"].values():
                        for obj in entries:
                            self.assertEqual(obj["physics"]["mesh_collision_approximation"], "none")
                            self.assertAlmostEqual(obj["default_pos"][2], generator.rope.table_top(layout))

    def test_retired_task_is_not_generated_or_exported_or_registered(self):
        from scripts.export_deformable_assets import TASKS
        self.assertEqual(len(TASKS), 7)
        self.assertNotIn("drape_cloth_over_rack", TASKS)
        self.assertNotIn("drape_cloth_over_rack", generator.TASKS)
        self.assertFalse((generator.ROOT / "task/RoboDojo/config/drape_cloth_over_rack.yml").exists())
        self.assertFalse((generator.ROOT / "task/RoboDojo/tasks/drape_cloth_over_rack.py").exists())

    def test_writer_rejects_native_task_names(self):
        for task in ("fold_clothes", "fold_clothes_random", "stack_bowls"):
            with self.subTest(task=task):
                path = self.layouts / f"{task}_0.json"
                original = b'{"preserve": true}\n'
                path.write_bytes(original)

                with self.assertRaisesRegex(ValueError, "outside the custom task set"):
                    generator.save_layout(task, 0, {"replacement": True})

                self.assertEqual(path.read_bytes(), original)
