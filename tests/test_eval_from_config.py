import copy
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.eval_from_config import DEFAULT_CONFIG, ROOT, load_config, run_groups


class EvalFromConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "scripts").mkdir()
        self.config = load_config(DEFAULT_CONFIG)
        self.config["tasks"] = ["first", "second"]
        self.config["ckpt"] = "checkpoint with spaces; $(not-a-command)"

    def write_worker(self, source):
        worker = self.root / "worker.py"
        worker.write_text(source, encoding="utf-8")
        (self.root / "scripts/robodojo.sh").write_text(
            f'exec {shlex.quote(sys.executable)} {shlex.quote(str(worker))} "$@"\n',
            encoding="utf-8",
        )

    def install_recording_worker(self, fail_first_group=False):
        self.write_worker(f"""
import json, os, sys, time
from pathlib import Path
args = sys.argv
task = args[args.index('--task') + 1]
gpu = args[args.index('--policy-gpu') + 1]
i = args.index('--wind')
data = {{'start': time.time(), 'wind': args[i+1:i+4],
        'num_envs': os.environ['ROBODOJO_NUM_ENVS'],
        'env_type': os.environ['EVAL_ENV_TYPE'],
        'ckpt': args[args.index('--ckpt') + 1],
        'inherited_eval_num': os.environ.get('EVAL_NUM')}}
time.sleep(0.3)
data['end'] = time.time()
Path(f'{{gpu}}_{{task}}.json').write_text(json.dumps(data))
sys.exit(7 if {fail_first_group!r} and gpu == '0' else 0)
""")

    def result(self, gpu, task):
        return json.loads((self.root / f"{gpu}_{task}.json").read_text())

    def test_parallel_groups_keep_task_order_and_isolated_parameters(self):
        self.install_recording_worker()
        # The JSON choice must win over an unrelated shell's previous eval.
        from unittest.mock import patch
        with patch.dict(os.environ, {"EVAL_NUM": "99"}):
            self.assertEqual(run_groups(self.config, self.root), 0)
        first = self.result(0, "first")
        other = self.result(2, "first")
        self.assertLess(max(first["start"], other["start"]), min(first["end"], other["end"]))
        self.assertEqual(first["wind"], ["0", "0", "0"])
        self.assertEqual(other["wind"], ["0", "0.5", "0"])
        for gpu in (0, 2):
            self.assertGreaterEqual(self.result(gpu, "second")["start"], self.result(gpu, "first")["end"])
        self.assertEqual(first["num_envs"], "1")
        self.assertEqual(first["env_type"], "sim")
        self.assertEqual(first["ckpt"], self.config["ckpt"])
        self.assertIsNone(first["inherited_eval_num"])
        self.assertEqual(len(list((self.root / "logs").glob("*.log"))), 2)

    def test_serial_mode_finishes_one_group_before_starting_next(self):
        self.install_recording_worker()
        self.config["parallel"] = False
        self.assertEqual(run_groups(self.config, self.root), 0)
        self.assertGreaterEqual(self.result(2, "first")["start"], self.result(0, "second")["end"])

    def test_failure_stops_only_affected_group_and_returns_nonzero(self):
        self.install_recording_worker(fail_first_group=True)
        self.assertEqual(run_groups(self.config, self.root), 1)
        self.assertFalse((self.root / "0_second.json").exists())
        self.assertTrue((self.root / "2_second.json").exists())

    def test_invalid_config_is_rejected(self):
        cases = [
            ("num_envs", 0), ("eval_num", True), ("parallel", "true"),
            ("tasks", ["first", "first"]), ("rigid_force_scale", float("nan")),
            ("groups", [{"name": "bad", "wind": [0, 1], "policy_gpu": 0, "env_gpu": 1}]),
        ]
        path = self.root / "config.json"
        for key, value in cases:
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[key] = value
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_dry_run_works_without_policy_installation_or_log_creation(self):
        (self.root / "scripts/eval_from_config.py").write_text(
            (ROOT / "scripts/eval_from_config.py").read_text()
        )
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        result = subprocess.run(
            [sys.executable, str(self.root / "scripts/eval_from_config.py"),
             "--config", str(config_path), "--dry-run"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.count("[dry-run]"), 4)
        self.assertFalse((self.root / "logs").exists())

    @unittest.skipUnless(os.name == "posix", "Process groups require POSIX")
    def test_sigterm_stops_policy_and_simulator_descendants(self):
        self.config["groups"] = self.config["groups"][:1]
        child_source = """
import signal, time
from pathlib import Path
def stop(*args):
    Path('child_stopped').touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path('child_ready').touch()
while True:
    time.sleep(0.1)
"""
        self.write_worker(f"""
import signal, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', {child_source!r}])
def stop(*args):
    child.wait(timeout=3)
    Path('parent_stopped').touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path('parent_ready').touch()
while True:
    time.sleep(0.1)
""")
        runner = subprocess.Popen(
            [sys.executable, "-c",
             "from pathlib import Path; from scripts.eval_from_config import run_groups; "
             f"raise SystemExit(run_groups({self.config!r}, Path({str(self.root)!r})))"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 5
            while not all((self.root / marker).exists() for marker in ("child_ready", "parent_ready")):
                if time.monotonic() > deadline or runner.poll() is not None:
                    self.fail("fake worker did not start")
                time.sleep(0.05)
            runner.send_signal(signal.SIGTERM)
            _, stderr = runner.communicate(timeout=15)
            self.assertEqual(runner.returncode, 143, stderr)
            self.assertTrue((self.root / "child_stopped").exists())
            self.assertTrue((self.root / "parent_stopped").exists())
        finally:
            if runner.poll() is None:
                runner.terminate()
                runner.communicate(timeout=15)


class EvalNumEnvsOverrideTests(unittest.TestCase):
    def test_client_uses_override_and_preserves_yaml_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "env_cfg/sim").mkdir(parents=True)
            robot = root / "env_cfg/arx_x5.yml"
            sim = root / "env_cfg/sim/sim_config.yml"
            robot.write_text("config: {sim: sim_config}\n")
            sim.write_text("scene: {num_envs: 10}\n")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            # Stub only the external YAML reader and Isaac entrypoint; exercise
            # the real shell client's argument parsing and override logic.
            (bin_dir / "python3").write_text("""#!/bin/sh
case "$2" in
  *"['config']['sim']"*) echo sim_config ;;
  *render_interval*) echo 10 ;;
  *num_envs*) echo 10 ;;
esac
""")
            (bin_dir / "python").write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARG_OUTPUT"\n')
            for path in bin_dir.iterdir():
                path.chmod(0o755)
            env = os.environ.copy()
            env.update(PATH=f"{bin_dir}:{env['PATH']}", ARG_OUTPUT=str(root / "args.txt"))
            env.pop("ROBODOJO_NUM_ENVS", None)
            command = [
                "bash", str(ROOT / "scripts/eval_policy.sh"), "--root_dir", str(root),
                "--task_name", "example", "--env_cfg_type", "arx_x5", "--device_id", "0",
                "--policy_name", "Pi_05", "--port", "12345", "--protocol", "ws",
                # Keep both forwarding arrays nonempty for macOS Bash 3.2's
                # pre-existing empty-array + nounset behavior.
                "--wind", "0", "0", "0", "--enable_cameras",
            ]
            for override, expected in ((None, "10"), ("1", "1"), ("2", "2"), ("0", None)):
                with self.subTest(override=override):
                    if override is not None:
                        env["ROBODOJO_NUM_ENVS"] = override
                    result = subprocess.run(command, env=env, capture_output=True, text=True)
                    if expected is None:
                        self.assertEqual(result.returncode, 2)
                        self.assertIn("positive integer", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        args = (root / "args.txt").read_text().splitlines()
                        self.assertEqual(args[args.index("--num_envs") + 1], expected)
            self.assertEqual(sim.read_text(), "scene: {num_envs: 10}\n")
