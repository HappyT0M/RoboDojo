"""Preview process isolation and shutdown status tests without Isaac Sim."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from utils.preview_run_log import (
    WORKER_RESULT_ENV_VAR,
    _argument_value,
    _worker_exit_code,
    plan_preview_workers,
    run_with_timestamped_log,
    write_preview_worker_result,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    "put_rope_ball_in_basket", "pass_rope_through_ring", "route_rope_around_posts",
    "fold_clothes_deformable", "flatten_clothes", "hang_clothes",
    "put_clothes_in_bin",
]


class PreviewRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.layouts = self.root / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
        self.layouts.mkdir(parents=True)

    def add_layouts(self, tasks, ids):
        for task in tasks:
            for layout_id in ids:
                (self.layouts / f"{task}_{layout_id}.json").write_text("{}")

    def test_all_seventy_scenes_get_one_worker_with_original_options(self):
        self.add_layouts(TASKS, list(reversed(range(10))))
        options = ["--wind", "0", "0.5", "0", "--device_id=0", "--headless", "--duration_s", "8"]
        workers = plan_preview_workers(self.root, ["--tasks", ",".join(TASKS), *options])
        self.assertEqual(len(workers), 70)
        self.assertEqual(
            [(_argument_value(w, "--task_name"), _argument_value(w, "--layout_id")) for w in workers],
            [(task, str(i)) for task in TASKS for i in range(10)],
        )
        self.assertTrue(all(w[:len(options)] == options for w in workers))

    def test_explicit_layout_still_splits_multiple_tasks(self):
        for selector in (["--layout_id", "8"], ["--layout_id=8"]):
            workers = plan_preview_workers(self.root, ["--tasks", ",".join(TASKS), *selector])
            self.assertEqual(len(workers), 7)
            self.assertTrue(all(_argument_value(w, "--layout_id") == "8" for w in workers))

    def test_exact_names_gaps_and_numeric_order(self):
        self.add_layouts(["fold_clothes"], [10, 2, 0])
        self.add_layouts(["fold_clothes_deformable"], [1])
        workers = plan_preview_workers(self.root, ["--task_name=fold_clothes"])
        self.assertEqual([_argument_value(w, "--layout_id") for w in workers], ["0", "2", "10"])

    def test_help_and_missing_assets_preserve_worker_validation(self):
        self.assertEqual(plan_preview_workers(self.root, ["--help"]), [["--help"]])
        args = ["--task_name", "missing"]
        self.assertEqual(plan_preview_workers(self.root, args), [args])

    def test_result_is_required_and_native_failure_takes_precedence(self):
        result = self.root / "result.json"
        self.assertEqual(_worker_exit_code(0, result), 1)
        result.write_text("not json")
        self.assertEqual(_worker_exit_code(0, result), 1)
        with patch.dict(os.environ, {WORKER_RESULT_ENV_VAR: str(result)}):
            write_preview_worker_result(1)
            self.assertTrue(result.with_suffix(".closing").exists())
            self.assertEqual(_worker_exit_code(0, result), 1)
            write_preview_worker_result(0)
            self.assertEqual(_worker_exit_code(0, result), 0)
            self.assertEqual(_worker_exit_code(-9, result), -9)

    def run_fake_recording(self, fail_layout):
        """Execute the real main() in workers whose fake Kit close calls os._exit(0)."""
        tasks = ["fold_clothes_deformable", "flatten_clothes"]
        self.add_layouts(tasks, range(2))
        script = self.root / "scripts/record.py"
        script.parent.mkdir()
        script.write_text(textwrap.dedent(f"""\
            import ast
            import json
            import os
            from pathlib import Path
            import sys
            from types import SimpleNamespace
            sys.path.insert(0, {str(ROOT)!r})
            from utils.preview_run_log import _argument_value, write_preview_worker_result
            root = Path({str(self.root)!r})
            task = _argument_value(sys.argv[1:], '--task_name')
            layout = _argument_value(sys.argv[1:], '--layout_id')
            lock = root / 'active-worker'
            def event(kind):
                with (root / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps([kind, os.getpid(), task, layout]) + '\\n')
            def record_task(name):
                with lock.open('x'):
                    pass
                event('start')
                if name == 'fold_clothes_deformable' and layout == {fail_layout!r}:
                    raise RuntimeError('simulated recording failure')
                return name
            def close(**kwargs):
                assert kwargs == dict(wait_for_replicator=False, skip_cleanup=True)
                event('exit')
                lock.unlink()
                os._exit(0)
            simulation_app = SimpleNamespace(close=close)
            _resolve_tasks = lambda: [task]
            source = Path({str(ROOT / 'scripts/record_deformable_tasks.py')!r})
            tree = ast.parse(source.read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec'))
            main()
        """))
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "utils.preview_run_log._write_all",
            side_effect=lambda fd, data: os.write(fd, data) if fd != 1 else None,
        ):
            code = run_with_timestamped_log(str(script), ["--tasks", ",".join(tasks)], self.root / "logs")
        events = [json.loads(line) for line in (self.root / "events.jsonl").read_text().splitlines()]
        self.assertEqual([event[0] for event in events], ["start", "exit"] * 4)
        self.assertEqual(len({event[1] for event in events}), 4)
        self.assertEqual([event[2:] for event in events[::2]], [[task, str(i)] for task in tasks for i in range(2)])
        log = next((self.root / "logs").glob("*.log")).read_text()
        return code, log

    def test_workers_finish_in_order_before_next_scene_starts(self):
        code, log = self.run_fake_recording(fail_layout=None)
        self.assertEqual(code, 0)
        self.assertIn("completed_workers=4/4 failed_workers=[]", log)

    def test_failed_recording_survives_kit_exit_zero_and_next_scene_runs(self):
        code, log = self.run_fake_recording(fail_layout="1")
        self.assertEqual(code, 1)
        self.assertIn("exit_code=1 process_exit_code=0", log)
        self.assertIn("completed_workers=3/4 failed_workers=[2]", log)

    def test_shutdown_timeout_kills_only_closing_workers_and_continues(self):
        self.add_layouts(["flatten_clothes"], range(4))
        script = self.root / "scripts/record.py"
        script.parent.mkdir()
        script.write_text(textwrap.dedent(f"""\
            import os
            from pathlib import Path
            import signal
            import sys
            import time
            sys.path.insert(0, {str(ROOT)!r})
            from utils.preview_run_log import (
                _argument_value, mark_preview_worker_shutdown, write_preview_worker_result,
            )
            events = Path({str(self.root / 'pids.txt')!r})
            if events.exists():
                previous = int(events.read_text().splitlines()[-1])
                try:
                    os.kill(previous, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('Previous scene worker has not exited')
            with events.open('a') as stream:
                stream.write(str(os.getpid()) + '\\n')
            layout = _argument_value(sys.argv[1:], '--layout_id')
            if layout == '0':
                # Recording itself must not be subject to the shutdown timeout.
                time.sleep(0.7)
            elif layout in ('1', '2'):
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                if layout == '1':
                    # Simulate env.close() hanging before a final result exists,
                    # with the output pipe already closed.
                    mark_preview_worker_shutdown()
                    os.close(1)
                    os.close(2)
                else:
                    # Simulate Kit hanging after recording succeeded, while
                    # still producing output. Neither counts as a clean exit.
                    write_preview_worker_result(0)
                while True:
                    if layout == '2':
                        print('shutdown still blocked', flush=True)
                    time.sleep(0.05)
            write_preview_worker_result(0)
            os._exit(0)
        """))
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "utils.preview_run_log._write_all",
            side_effect=lambda fd, data: os.write(fd, data) if fd != 1 else None,
        ), patch("utils.preview_run_log.TERMINATE_GRACE_S", 0.05):
            code = run_with_timestamped_log(
                str(script), ["--task_name", "flatten_clothes"], self.root / "logs", shutdown_timeout_s=0.05,
            )
        self.assertEqual(code, 1)
        pids = (self.root / "pids.txt").read_text().splitlines()
        self.assertEqual(len(pids), 4)
        for pid in pids:
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid), 0)
        log = next((self.root / "logs").glob("*.log")).read_text()
        self.assertEqual(log.count("sending SIGTERM"), 2)
        self.assertEqual(log.count("sending SIGKILL"), 2)
        self.assertIn("completed_workers=2/4 failed_workers=[2, 3]", log)


if __name__ == "__main__":
    unittest.main()
