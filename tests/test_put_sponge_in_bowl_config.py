import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_task_yaml_declares_one_rigid_bowl_and_one_fem_sponge_without_wind():
    config_path = ROOT / "task/RoboDojo/config/put_sponge_in_bowl.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert "wind" not in config
    assert config["Rigid"][0]["category"][0]["name"] == "bowl"
    sponge = config["Deformable"][0]["category"][0]
    assert sponge["name"] == "sponge"
    assert sponge["physics"]["type"] == "deformable"
    assert sponge["size"] == [0.06, 0.05, 0.03]


def test_task_is_registered_with_one_fixed_evaluation_layout():
    tasks = yaml.safe_load((ROOT / "task/RoboDojo/config/_task.yml").read_text(encoding="utf-8"))
    assert tasks["tasks"]["put_sponge_in_bowl"]["eval_nums"] == 1

    layout_path = ROOT / "Assets/Eval_Layout/RoboDojo/arx_x5/0/put_sponge_in_bowl_0.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    assert len(layout["Deformable"]["sponge"]) == 1
    assert len(layout["Rigid"]["bowl"]) == 1
    sponge = layout["Deformable"]["sponge"][0]
    assert sponge["physics"]["type"] == "deformable"
    assert sponge["physics"]["youngs_modulus"] > 0
    assert sponge["size"] == [0.06, 0.05, 0.03]


def test_smoke_script_has_explicit_gpu_and_reset_checks():
    script = (ROOT / "scripts/test_fem_sponge.py").read_text(encoding="utf-8")
    wrapper = (ROOT / "env/scene_manager/objects/deformable.py").read_text(encoding="utf-8")
    assert "AppLauncher" in script
    assert "get_nodal_positions_w" in script
    assert "reset" in script.lower()
    assert 'usd_path=""' in script
    assert "MeshCuboidCfg" in wrapper
