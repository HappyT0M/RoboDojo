import pytest

from utils.load_file import load_object_metadata
from utils.pipeline_utils import configure_task_sim_device


def test_missing_object_metadata_reports_the_expected_file(tmp_path):
    model_dir = tmp_path / "Deformable" / "sponge"

    with pytest.raises(FileNotFoundError, match=r"00000[/\\]metadata\.json"):
        load_object_metadata(model_dir, 0)


def test_fem_task_uses_visible_cuda_device_selected_by_device_id():
    config = {
        "sim": {"scene": {"num_envs": 1}},
        "task_env": {"Deformable": [{"category": [{"name": "sponge"}]}]},
    }

    configure_task_sim_device(config, 0)

    assert config["sim"]["device"] == "cuda:0"
    assert "device" not in config


def test_fem_task_uses_selected_visible_gpu_index():
    config = {
        "sim": {"scene": {"num_envs": 1}},
        "task_env": {"Deformable": [{"category": [{"name": "sponge"}]}]},
    }

    configure_task_sim_device(config, 1)

    assert config["sim"]["device"] == "cuda:1"


def test_fem_task_rejects_missing_gpu_id():
    config = {"task_env": {"Deformable": [{"category": [{"name": "sponge"}]}]}}

    with pytest.raises(ValueError, match="FEM deformable tasks require a CUDA simulation device"):
        configure_task_sim_device(config, None)


def test_rigid_only_task_keeps_its_existing_device_default():
    config = {"task_env": {"Rigid": [{"category": [{"name": "bowl"}]}]}}

    configure_task_sim_device(config, None)

    assert "device" not in config
