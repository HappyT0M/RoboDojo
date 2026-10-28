import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_deformable_assets import export_bundle


class DeformableAssetExportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.assets = self.root / "Assets"
        layouts = self.assets / "Eval_Layout/RoboDojo/arx_x5/0"
        layouts.mkdir(parents=True)
        self.layout = layouts / "example_0.json"
        self.layout.write_text(json.dumps({
            "Garment": {"selected": [{"category": "selected", "category_idx": 0}]},
            "Room": {"default": "Simple_Room_nolight"},
        }))
        self.model = self.assets / "Object/RoboDojo/Garment/selected/00000"
        self.model.mkdir(parents=True)
        (self.model / "metadata.json").write_text("{}")
        (self.model / "object.usd").write_text("#usda 1.0\n")
        (self.model / "texture.png").write_bytes(b"texture data")
        (self.model / ".DS_Store").write_bytes(b"exclude")
        unrelated = self.assets / "Object/RoboDojo/Garment/unrelated/00000"
        unrelated.mkdir(parents=True)
        (unrelated / "object.usd").write_text("unrelated")

    def test_exports_only_referenced_objects_with_sidecars_and_hashes(self):
        out = self.root / "export"
        manifest = export_bundle(self.assets, out, tasks=("example",), layout_count=1)
        self.assertEqual(len(manifest["files"]), 4)
        paths = [item["path"] for item in manifest["files"]]
        self.assertTrue(any(p.endswith("texture.png") for p in paths))
        self.assertFalse(any("unrelated" in p or ".DS_Store" in p for p in paths))
        for item in manifest["files"]:
            contents = (out / item["path"]).read_bytes()
            self.assertEqual(item["sha256"], hashlib.sha256(contents).hexdigest())
        self.assertTrue((self.model / "object.usd").is_file())

    def test_missing_dependency_fails_before_creating_export(self):
        (self.model / "object.usd").unlink()
        out = self.root / "export"
        with self.assertRaises(FileNotFoundError):
            export_bundle(self.assets, out, tasks=("example",), layout_count=1)
        self.assertFalse(out.exists())

    def test_existing_output_is_not_overwritten(self):
        out = self.root / "export"
        out.mkdir()
        marker = out / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            export_bundle(self.assets, out, tasks=("example",), layout_count=1)
        self.assertEqual(marker.read_text(), "keep")

    def test_rejects_dependency_symlinks_outside_assets(self):
        external = self.root / "outside.bin"
        external.write_bytes(b"not an asset")
        (self.model / "external.bin").symlink_to(external)
        with self.assertRaisesRegex(ValueError, "escapes Assets"):
            export_bundle(self.assets, self.root / "export", tasks=("example",), layout_count=1)
