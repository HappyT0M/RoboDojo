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

        generated_names = {p.name for p in self.layouts.iterdir()} - native_files.keys()
        self.assertEqual(len(generated_names), 50)
        folding_paths = sorted(self.layouts.glob("fold_clothes_deformable_*.json"))
        self.assertEqual(len(folding_paths), 10)
        for path in folding_paths:
            layout = json.loads(path.read_text())
            self.assertIn("folding_board", layout["Geometry"])

    def test_writer_rejects_native_task_names(self):
        for task in ("fold_clothes", "fold_clothes_random", "stack_bowls"):
            with self.subTest(task=task):
                path = self.layouts / f"{task}_0.json"
                original = b'{"preserve": true}\n'
                path.write_bytes(original)

                with self.assertRaisesRegex(ValueError, "outside the custom task set"):
                    generator.save_layout(task, 0, {"replacement": True})

                self.assertEqual(path.read_bytes(), original)
