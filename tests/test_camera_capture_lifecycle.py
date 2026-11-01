"""Exercise real capture lifecycle methods with fake Replicator/CUDA resources."""
import ast
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_class(relative_path, class_name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]


class FakeXFormPrim:
    def __init__(self, prim_paths_expr, name, **kwargs):
        self.prims = self.prim_paths = prim_paths_expr
        self.name = name

    def __del__(self):
        pass


class CameraCaptureLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.products = []
        self.annotators = []
        self.live = set()
        self.max_live = 0
        spec = {"rgb": {"channels": 4, "dtype": "uint8"}, "depth": {"channels": 1, "dtype": "float32"}}
        rep = SimpleNamespace(
            create=SimpleNamespace(render_product_tiled=self.create_product),
            AnnotatorRegistry=SimpleNamespace(get_annotator=self.create_annotator),
        )
        self.CameraView = load_class("env/camera_manager/capture/camera_view.py", "CameraView", {
            "XFormPrim": FakeXFormPrim, "rep": rep, "ANNOTATOR_SPEC": spec,
        })
        omega = SimpleNamespace(
            to_container=lambda config, **kw: deepcopy(vars(config) if isinstance(config, SimpleNamespace) else config),
            create=lambda config: SimpleNamespace(**config),
        )
        self.Manager = load_class("env/camera_manager/capture/tiled_capture_manager.py", "TiledCaptureManager", {
            "OmegaConf": omega, "deepcopy": deepcopy, "CameraView": self.CameraView,
        })
        camera_module = ModuleType("env.camera_manager.capture.camera_view")
        camera_module.ANNOTATOR_SPEC = spec
        warp = ModuleType("warp")
        warp.zeros = lambda *args, **kwargs: object()
        self.addCleanup(patch.stopall)
        patch.dict("sys.modules", {camera_module.__name__: camera_module, "warp": warp}).start()

    def create_product(self, **kwargs):
        path = f"/Render/product_{len(self.products)}"
        self.live.add(path)
        self.max_live = max(self.max_live, len(self.live))
        product = SimpleNamespace(path=path, destroy=Mock(side_effect=lambda: self.live.remove(path)))
        self.products.append(product)
        return product

    def create_annotator(self, *args, **kwargs):
        annotator = SimpleNamespace(attach=Mock(), detach=Mock())
        self.annotators.append(annotator)
        return annotator

    def make_manager(self):
        owner = SimpleNamespace(
            cameras=[[SimpleNamespace(prim_path=f"/World/camera_{i}", _resolution=(16, 16)) for i in range(3)]],
            camera_names=[["front", "left", "right"]],
        )
        config = {"annotator": {"common": {"enabled": True, "rgb": {"type": "rgb"}}}}
        manager = self.Manager(1, config, owner, "cpu")
        manager.initialize(object())
        return manager, owner

    def test_close_detaches_every_annotator_before_destroy_and_is_idempotent(self):
        camera = self.CameraView(["/World/camera"], output_annotators=["rgb", "depth"])
        product = camera._render_product
        events = []
        for annotator in self.annotators:
            annotator.detach.side_effect = lambda paths: events.append("detach")
        product.destroy.side_effect = lambda: events.append("destroy")
        camera.close()
        camera.close()
        self.assertEqual(events, ["detach", "detach", "destroy"])
        for annotator in self.annotators:
            annotator.detach.assert_called_once_with([product.path])
        self.assertEqual(camera._annotators, {})
        self.assertIsNone(camera._render_product)

    def test_ten_resets_release_old_resources_before_allocating_new_ones(self):
        manager, owner = self.make_manager()
        for generation in range(10):
            old_products = list(self.products)
            manager.reset()
            for product in old_products:
                product.destroy.assert_called_once()
            self.assertEqual(len(self.live), 3)
            self.assertEqual(len(manager.tiled_cameras), 3)
            self.assertEqual(len(manager.tiled_render_products), 3)
            self.assertEqual(manager.annotator_type, [["rgb"]] * 3)
            self.assertEqual(len(manager._output_buffers), 3)
            self.assertEqual(len(manager.annotator), 3)
        self.assertEqual(self.max_live, 3)
        manager.destroy()
        manager.destroy()
        self.assertFalse(self.live)
        self.assertEqual(manager._output_buffers, {})
        self.assertEqual(len(owner.cameras[0]), 3)
        self.assertEqual(owner.camera_names, [["front", "left", "right"]])
        for annotator in self.annotators:
            annotator.detach.assert_called_once()
        for product in self.products:
            product.destroy.assert_called_once()

    def test_reset_rebinds_to_camera_managers_recreated_cameras(self):
        manager, owner = self.make_manager()
        manager.reset()
        owner.cameras = [[SimpleNamespace(prim_path="/World/new_camera", _resolution=(8, 8))]]
        owner.camera_names = [["new_camera"]]
        manager.reset()
        self.assertEqual(manager.num_cams, 1)
        self.assertEqual(manager.camera_prim_paths, [["/World/new_camera"]])
        self.assertEqual(len(self.live), 1)
        manager.destroy()
        self.assertEqual(len(owner.cameras[0]), 1)

    def test_cleanup_attempts_remaining_resources_if_one_detach_fails(self):
        manager, _ = self.make_manager()
        manager.reset()
        self.annotators[0].detach.side_effect = RuntimeError("stage already removed")
        with self.assertRaisesRegex(RuntimeError, "release tiled capture"):
            manager.destroy()
        self.assertFalse(self.live)
        self.assertEqual(manager._output_buffers, {})
        for annotator in self.annotators:
            annotator.detach.assert_called_once()
        manager.destroy()


if __name__ == "__main__":
    unittest.main()
