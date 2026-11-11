"""Task semantics and scene integration without Isaac Sim or policy weights."""

import ast
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from env.reward_manager.deformable_manipulation import prepare, evaluate, _observe
from scripts import generate_additional_deformable_assets as generator
from scripts.export_deformable_assets import TASKS as EXPORT_TASKS, collect
from scripts.internal.task_inventory import build_inventory
from utils.deformable_manipulation_metrics import (
    CoverProgress, PullProgress, WipeProgress, local_points, rectangular_samples, surface_coverage,
)

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    Usd = None

ROOT = Path(__file__).resolve().parents[1]
FACES = [(0, 1, 2), (0, 2, 3)]


def square(x=0, y=0, z=.056, hx=.12, hy=.10):
    return [(x-hx, y-hy, z), (x+hx, y-hy, z), (x+hx, y+hy, z), (x-hx, y+hy, z)]


class DeformableMetricTests(unittest.TestCase):
    def test_triangle_coverage_rejects_holes_and_wrong_height(self):
        points = [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        # All corners surround the sample, but this one triangular piece misses it.
        self.assertEqual(surface_coverage(points, [(0, 1, 2)], [(-.5, .5)], .9, 1.1), set())
        self.assertEqual(surface_coverage(points, FACES, [(-.5, .5)], .9, 1.1), {0})
        self.assertEqual(surface_coverage(points, FACES, [(0, 0)], 1.2, 1.5), set())
        self.assertEqual(surface_coverage([(0, 0, 0)] * 3, [(0, 1, 2)], [(0, 0)], -1, 1), set())

    def test_cover_requires_manipulation_release_height_and_stability(self):
        config = {"target_half_extents": [.09, .07], "surface_height": .05, "stable_steps": 3}
        progress = CoverProgress(config)
        cloth = square()
        for _ in range(5):
            self.assertFalse(progress.update(cloth, FACES, []))  # Wind alone cannot succeed.
        grip = [(cloth[0], .1)]
        for _ in range(5):
            self.assertFalse(progress.update(cloth, FACES, grip))
        self.assertEqual([progress.update(cloth, FACES, []) for _ in range(3)], [False, False, True])
        self.assertFalse(progress.update(square(z=.3), FACES, []))
        self.assertFalse(progress.update(square(x=.3), FACES, []))
        self.assertFalse(progress.update([(float("nan"), 0, 0)], [], []))

    def test_local_coordinates_remove_env_origin_rotation_and_scale(self):
        q = (math.sqrt(.5), 0, 0, math.sqrt(.5))
        points = local_points([(6., 5., 3.)], (7., 3., 3.), q, (2., 1., 1.))
        for actual, expected in zip(points[0], (1., 1., 0.)):
            self.assertAlmostEqual(actual, expected)

    def test_wiping_counts_only_moving_held_cloth_near_surface(self):
        config = {"target_half_extents": [.2, .14], "surface_height": .006,
                  "minimum_coverage": .9, "minimum_wipe_distance": .15}
        progress = WipeProgress(config)
        for x in [-.2 + .02*i for i in range(21)]:
            cloth = square(x=x, z=.014, hx=.10, hy=.17)
            self.assertFalse(progress.update(cloth, FACES, []))
        self.assertFalse(progress.covered)
        for _ in range(5):
            progress.update(cloth, FACES, [(cloth[0], .1)])
        self.assertFalse(progress.covered)  # Resting on the patch is not wiping.
        for x in [.2 - .02*i for i in range(21)]:
            cloth = square(x=x, z=.2, hx=.10, hy=.17)
            self.assertFalse(progress.update(cloth, FACES, [(cloth[0], .1)]))
        self.assertFalse(progress.covered)  # Moving above the board is not contact.
        success = False
        for x in [-.2 + .02*i for i in range(21)]:
            cloth = square(x=x, z=.014, hx=.10, hy=.17)
            success = progress.update(cloth, FACES, [(cloth[0], .1)])
        self.assertTrue(success)
        self.assertGreaterEqual(progress.path_length, .15)

    def test_wind_motion_with_stationary_hand_does_not_wipe(self):
        progress = WipeProgress({"target_half_extents": [.2, .14], "surface_height": .006})
        hand = [((-.1, -.1, .014), .1)]
        for x in (0, .01, .02, .03):
            self.assertFalse(progress.update(square(x=x, z=.014), FACES, hand))
        self.assertEqual(progress.path_length, 0)
        self.assertEqual(progress.covered, set())

    def test_actual_small_wiping_cloth_can_cover_board_in_two_passes(self):
        points, faces = generator.cloth_mesh(*generator.CLOTHS['wiping_cloth'])
        progress = WipeProgress({'target_half_extents': [.2, .14], 'surface_height': .006})
        # Two overlapping horizontal passes plus the short vertical connection.
        path = [(-.13 + .01*i, -.07) for i in range(27)]
        path += [(.13, -.07 + .01*i) for i in range(1, 15)]
        path += [(.13 - .01*i, .07) for i in range(1, 27)]
        for x, y in path:
            cloth = [(px+x, py+y, .014) for px, py, _ in points]
            success = progress.update(cloth, faces, [(cloth[0], .1)])
        self.assertTrue(success)

    def pull_sequence(self, extra_hand=False, grasp=True):
        progress = PullProgress({"target_half_extents": [.09, .06], "stable_steps": 3})
        outcomes = []
        for step in range(16):
            y = .3 - step*.02
            rope = [(0, y-.5, .02), (0, y-.45, .02), (0, y-.4, .02)]
            ball = (0, y, .02)
            hands = [(rope[0], .1)] if grasp else []
            if extra_hand:
                hands.append((ball, .1))
            outcomes.append(progress.update(rope, ball, hands))
        return progress, outcomes

    def test_pull_requires_active_rope_pull_and_rejects_direct_ball_handling(self):
        progress, outcomes = self.pull_sequence()
        self.assertTrue(outcomes[-1])
        self.assertGreaterEqual(progress.pull_distance, .12)
        for kwargs in ({"extra_hand": True}, {"grasp": False}):
            progress, outcomes = self.pull_sequence(**kwargs)
            self.assertFalse(any(outcomes))
        progress = PullProgress({"target_half_extents": [.09, .06]})
        for _ in range(30):
            self.assertFalse(progress.update([(0, -.5, .02)], (0, 0, .02), []))


class DeformableAssetTests(unittest.TestCase):
    @unittest.skipIf(Usd is None, 'OpenUSD unavailable; run in the simulation environment')
    def test_new_usd_assets_compose_with_mesh_material_and_correct_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            generator.generate(assets)
            paths = sorted(assets.glob('Object/RoboDojo/*/*/00000/object.usd'))
            self.assertEqual(len(paths), 5)
            for path in paths:
                with self.subTest(asset=path):
                    stage = Usd.Stage.CreateInMemory()
                    root = stage.DefinePrim('/World/object', 'Xform')
                    root.GetReferences().AddReference(str(path))
                    meshes = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
                    self.assertEqual(len(meshes), 1)
                    mesh = meshes[0]
                    self.assertTrue(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
                    material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
                    self.assertTrue(material)
                    self.assertEqual(str(material.GetPath()), '/World/object/Looks/Surface')
                    self.assertEqual(mesh.HasAPI(UsdPhysics.CollisionAPI), 'Geometry' in path.parts)

    def test_check_mode_and_regeneration_preserve_other_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "Eval_Layout/RoboDojo/arx_x5/0/fold_clothes_0.json"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text('native content')
            generator.generate(root)
            first = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            generator.generate(root)
            generator.generate(root, check=True)
            self.assertEqual(first, {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()})
            self.assertEqual(len(first), 41)
            missing = root / "Object/RoboDojo/Garment/cover_cloth/00000/object.usd"
            missing.write_text('broken')
            with self.assertRaisesRegex(ValueError, 'Missing or changed'):
                generator.generate(root, check=True)
            self.assertEqual(missing.read_text(), 'broken')

    def test_cloth_reference_structure_and_non_degenerate_connected_meshes(self):
        for name, dims in generator.CLOTHS.items():
            points, faces = generator.cloth_mesh(*dims)
            self.assertEqual(len(points), dims[2]*dims[3])
            self.assertEqual(len(faces), 2*(dims[2]-1)*(dims[3]-1))
            self.assertEqual(set(v for f in faces for v in f), set(range(len(points))))
            for a, b, c in faces:
                self.assertGreater((points[b][0]-points[a][0])*(points[c][1]-points[a][1])
                                   - (points[b][1]-points[a][1])*(points[c][0]-points[a][0]), 0)
            path = Path('Object/RoboDojo/Garment') / name / '00000/object.usd'
            usd = generator.generated_files()[path]
            self.assertIn('def Xform "asset"', usd)
            self.assertIn('def Mesh "mesh"', usd)
            self.assertNotIn('PhysicsCollisionAPI', usd)

    def test_thirty_layouts_vary_scenes_and_start_outside_goals(self):
        for task in generator.TASKS:
            layouts = [generator.make_layout(task, i) for i in range(10)]
            self.assertEqual(len({json.dumps(scene, sort_keys=True) for scene in layouts}), 10)
            self.assertEqual(len({scene['Room']['default'] for scene in layouts}), 3)
            self.assertGreater(len({tuple(scene['Table']['scale']) for scene in layouts}), 1)
            for scene in layouts:
                top = generator.rope.table_top(scene)
                if task == 'pull_rope_to_retrieve_ball':
                    rope = scene['Articulation']['rope_chain'][0]
                    goal = scene['Geometry']['rope_retrieval_zone'][0]
                    from utils.rope_task_metrics import _rotate_by_quaternion
                    offset = _rotate_by_quaternion((.594, 0, 0), rope['default_ori'])
                    ball = [a+b for a, b in zip(rope['default_pos'], offset)]
                    local = local_points([ball], goal['default_pos'], goal['default_ori'])[0]
                    self.assertGreater(local[1], .18)
                    self.assertLess(ball[1]+.019, scene['Table']['default_pos'][1]+scene['Table']['scale'][1]/2)
                else:
                    name = 'cover_cloth' if task == 'cover_object_with_cloth' else 'wiping_cloth'
                    cloth = scene['Garment'][name][0]
                    self.assertGreater(cloth['default_pos'][2], top)
                    goal = next(v[0] for k,v in scene['Geometry'].items() if k != 'cloth_block')
                    self.assertGreater(goal['default_pos'][0]-cloth['default_pos'][0], .45)
                    self.assertIn('particle_system', cloth['physics'])

    def test_actual_assets_export_all_ten_tasks_and_all_json_configs_include_them(self):
        files, objects, _ = collect(ROOT / 'Assets')
        self.assertEqual(len(objects), 10)
        self.assertEqual(len([f for f in files if str(f).startswith('Eval_Layout/')]), 100)
        self.assertEqual(len(EXPORT_TASKS), 10)
        for path in (ROOT / 'scripts/configs').glob('pi05_deformable*.json'):
            self.assertEqual(set(json.loads(path.read_text())['tasks']), set(EXPORT_TASKS))
        inventory = build_inventory()
        self.assertTrue(all(task['runnable'] and task['dimension'] for task in inventory['tasks']))
        self.assertEqual({t['name'] for t in inventory['tasks'] if t['dimension'] == 'deformable'}, set(EXPORT_TASKS))


class RewardAdapterTests(unittest.TestCase):
    def parser_class(self):
        tree = ast.parse((ROOT / 'env/reward_manager/func_parser.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Func_Parser')
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in
                    ('__init__', 'reset', 'is_deformable_manipulation_success')]
        namespace = {}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), '<parser>', 'exec'), namespace)
        return namespace['Func_Parser']

    def test_rope_adapter_uses_grasp_point_not_wrist_and_removes_world_origin(self):
        parser = self.parser_class()(1)
        config = {'kind': 'pull', 'target_half_extents': [.09, .06], 'link_count': 12}
        parser.env = SimpleNamespace(task_env_config={'DeformableManipulation':config}, success=[True],
                                    sim=SimpleNamespace(scene=SimpleNamespace(env_origins=[[7,0,0]])))
        robot = SimpleNamespace(gripper_scale=[-.01,.044], gripper_bias=.145, gripper_move={'sign':1})
        q = math.sqrt(.5)
        parser.robot_manager = SimpleNamespace(
            robot_list=[robot],
            get_real_endpose=lambda *a, **kw: {0: [0,0,.8,q,0,0,q]},
            get_end_effector_real_val=lambda *a, **kw: {0: [0,0]},
        )
        class Layout:
            def get_instance_name(self, env_idx, label):
                return label
            def get_instance_pose(self, env_idx, label):
                return [0,.145,.765], [1,0,0,0]
            def get_scene_object(self, env_idx, inst_name):
                if inst_name == 'goal':
                    return SimpleNamespace(scale=[1,1,1])
                return SimpleNamespace(get_link_pose=lambda name: [7, .445 if name == 'ball' else .145, .8, 1,0,0,0])
        parser.layout_manager = Layout()
        _, (rope, ball, hands) = _observe(parser, 0)
        self.assertEqual(len(rope), 12)
        for a, b in zip(hands[0][0], [0,0,.035]):
            self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(hands[0][1], .01/.054)
        self.assertAlmostEqual(ball[1], .3)
        prepare(parser)
        self.assertEqual(parser.deformable_manipulation_state[0]['metric'].previous_holder, 0)
        self.assertEqual(evaluate(parser, 0), 0.)
        robot.gripper_move['sign'] = -1
        parser.robot_manager.get_end_effector_real_val = lambda *a, **kw: {0: [.044,-.044]}
        self.assertAlmostEqual(_observe(parser, 0)[1][2][0][1], 0.)

    def test_real_adapter_handles_env_offsets_topology_and_episode_reset(self):
        parser = self.parser_class()(2)
        config = {'kind':'cover', 'target_half_extents':[.09,.07], 'surface_height':.05, 'stable_steps':2}
        parser.env = SimpleNamespace(task_env_config={'DeformableManipulation':config}, success=[True,True],
                                     sim=SimpleNamespace(scene=SimpleNamespace(env_origins=[[0,0,0],[7,0,0]])))
        worlds = [square(), [(x+7,y,z) for x,y,z in square()]]
        class Prim:
            def GetAttribute(self, name):
                values = {'faceVertexCounts':[3,3], 'faceVertexIndices':[0,1,2,0,2,3]}
                return SimpleNamespace(Get=lambda:values[name])
        class Layout:
            def get_instance_name(self, env_idx, label):
                return label
            def get_instance_pose(self, env_idx, label):
                return [0,0,0],[1,0,0,0]
            def get_scene_object(self, env_idx, inst_name):
                if inst_name == 'goal':
                    return SimpleNamespace(scale=[1,1,1])
                return SimpleNamespace(prim=Prim(), sample_mesh_vertices=lambda:(worlds[env_idx],None,None,None))
        parser.layout_manager = Layout()
        parser.robot_manager = SimpleNamespace(robot_list=[])
        prepare(parser)
        self.assertEqual([s['metric'].coverage for s in parser.deformable_manipulation_state], [1.,1.])
        self.assertEqual(evaluate(parser, 0), 0.)
        parser.deformable_manipulation_state[0]['metric'].manipulated = True
        self.assertEqual(evaluate(parser, 0), 0.)
        self.assertEqual(evaluate(parser, 0), 1.)
        self.assertEqual(evaluate(parser, 1), 0.)
        parser.reset()
        self.assertEqual(parser.deformable_manipulation_state, [{},{}])
        prepare(parser)
        self.assertFalse(parser.deformable_manipulation_state[0]['metric'].manipulated)


if __name__ == '__main__':
    unittest.main()
