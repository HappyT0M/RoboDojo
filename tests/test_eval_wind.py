"""Lifecycle and substep tests without requiring Isaac Sim or model weights."""
import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from utils.eval_wind import EvalWind, EvalWindError


class EvalWindTests(unittest.TestCase):
    def setUp(self):
        self.scene = SimpleNamespace(_garment_objects=[])
        self.module = ModuleType("utils.rigid_wind")
        self.factory = Mock()
        self.module.create_rigid_wind_controller = self.factory

    def controller(self):
        return SimpleNamespace(
            bodies=[SimpleNamespace(prim_path="/World/envs/env_0/rope/ball")],
            step=Mock(), last_diagnostics={"applied_body_count": 1},
        )

    def test_zero_wind_does_not_construct_or_apply(self):
        with patch.dict("sys.modules", {"utils.rigid_wind": self.module}):
            wind = EvalWind([0, 0, 0])
            wind.start(self.scene)
            wind.step()
            self.factory.assert_not_called()
            self.assertEqual(wind.metadata()["rigid_physics_steps"], 0)

    def test_stop_then_reset_binds_new_controller_without_stacking(self):
        first, second = self.controller(), self.controller()
        self.factory.side_effect = [(first, []), (second, [])]
        with patch.dict("sys.modules", {"utils.rigid_wind": self.module}):
            wind = EvalWind([0, 0.5, 0], force_scale=100)
            wind.step()  # settling before first start
            wind.start(self.scene)
            first.step.assert_not_called()  # binding must not apply a force
            wind.step()
            wind.stop()
            wind.step()  # settling before next start
            wind.start(self.scene)
            wind.step()
            first.step.assert_called_once()
            second.step.assert_called_once()
            self.assertEqual(wind.metadata()["rigid_physics_steps"], 1)
            self.assertEqual(self.factory.call_args.kwargs["force_scale"], 100)

    def test_nonzero_wind_without_targets_is_not_silently_ignored(self):
        self.factory.return_value = (None, [{"reason": "kinematic_body"}])
        with patch.dict("sys.modules", {"utils.rigid_wind": self.module}):
            with self.assertRaisesRegex(RuntimeError, "no eligible"):
                EvalWind([1, 0, 0]).start(self.scene)

    def test_cloth_only_task_keeps_existing_wind_route(self):
        self.factory.return_value = (None, [])
        with patch.dict("sys.modules", {"utils.rigid_wind": self.module}):
            wind = EvalWind([1, 0, 0])
            wind.start(SimpleNamespace(_garment_objects=[{"cloth": object()}]))
            wind.step()
            self.assertIsNone(wind.controller)

    def test_force_failure_invalidates_trial(self):
        controller = self.controller()
        controller.step.side_effect = RuntimeError("stale view")
        self.factory.return_value = (controller, [])
        with patch.dict("sys.modules", {"utils.rigid_wind": self.module}):
            wind = EvalWind([1, 0, 0])
            wind.start(self.scene)
            with self.assertRaisesRegex(EvalWindError, "stale view"):
                wind.step()
            self.assertEqual(wind.step_count, 0)

    def test_invalid_settings_rejected(self):
        for scale in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                EvalWind([1, 0, 0], force_scale=scale)

    def test_force_runs_after_scene_write_before_every_physics_substep(self):
        path = Path(__file__).resolve().parents[1] / "env/environment/isaac/direct_rl_env.py"
        tree = ast.parse(path.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "sim_step")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        events = []
        env = SimpleNamespace(
            cfg=SimpleNamespace(decimation=3, sim=SimpleNamespace(render_interval=0), events=None),
            sim=SimpleNamespace(has_gui=lambda: False, has_rtx_sensors=lambda: False,
                                step=lambda **kw: events.append("physics")),
            scene=SimpleNamespace(write_data_to_sim=lambda: events.append("write"),
                                  update=lambda **kw: events.append("update")),
            func={"_before_physics_step": lambda: events.append("wind")},
            _sim_step_counter=0, episode_length_buf=0, common_step_counter=0,
            physics_dt=0.004, extras={},
        )
        namespace["sim_step"](env, render=False)
        self.assertEqual(events, ["write", "wind", "physics", "update"] * 3)


if __name__ == "__main__":
    unittest.main()
