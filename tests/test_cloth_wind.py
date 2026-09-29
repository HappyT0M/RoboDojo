import pytest

from utils.cloth_wind import (
    apply_cloth_runtime_overrides,
    apply_constant_cloth_wind,
    collect_cloth_wind_readbacks,
)


def test_runtime_wind_overrides_saved_layout_for_matching_garment_category():
    task_cfg = {
        "Garment": [
            {
                "category": [{"name": "Top_Long"}],
                "physics": {
                    "particle_system": {"wind": [3.0, 0.0, 0.0]},
                    "particle_material": {"drag": 0.5},
                },
            }
        ]
    }
    layout = {
        "Garment": {
            "Top_Long": [{"physics": {"particle_system": {"wind": None}}}],
            "Towel": [{"physics": {"particle_system": {"wind": None}}}],
        }
    }

    apply_cloth_runtime_overrides(layout, task_cfg)

    assert layout["Garment"]["Top_Long"][0]["physics"]["particle_system"]["wind"] == [3.0, 0.0, 0.0]
    assert layout["Garment"]["Top_Long"][0]["physics"]["particle_material"]["drag"] == 0.5
    assert layout["Garment"]["Towel"][0]["physics"]["particle_system"]["wind"] is None


def test_apply_constant_cloth_wind_sets_vector_for_each_garment():
    task_cfg = {
        "Garment": [
            {"physics": {"particle_system": {"contact_offset": 0.01}}},
            {"physics": {"particle_system": {"wind": [1, 0, 0]}}},
        ]
    }

    apply_constant_cloth_wind(task_cfg, (0.2, -0.1, 0.0), particle_drag=0.6)

    for garment in task_cfg["Garment"]:
        assert garment["physics"]["particle_system"]["wind"] == [0.2, -0.1, 0.0]
        assert garment["physics"]["particle_material"]["drag"] == 0.6


def test_none_disables_existing_configured_wind_without_changing_drag():
    task_cfg = {
        "Garment": [
            {
                "physics": {
                    "particle_system": {"wind": [1.0, 0.0, 0.0]},
                    "particle_material": {"drag": 0.25},
                }
            }
        ]
    }

    apply_constant_cloth_wind(task_cfg, None)

    assert task_cfg["Garment"][0]["physics"]["particle_system"]["wind"] == [0.0, 0.0, 0.0]
    assert task_cfg["Garment"][0]["physics"]["particle_material"]["drag"] == 0.25


@pytest.mark.parametrize("wind", [(1.0, 2.0), (1.0, float("nan"), 0.0)])
def test_invalid_wind_vector_is_rejected(wind):
    task_cfg = {"Garment": [{"physics": {}}]}

    with pytest.raises(ValueError, match="wind"):
        apply_constant_cloth_wind(task_cfg, wind)


def test_nonzero_wind_requires_garment_object():
    with pytest.raises(ValueError, match="Garment"):
        apply_constant_cloth_wind({"Object": []}, (0.5, 0.0, 0.0))


class _FakeParticleSystem:
    def __init__(self, wind):
        self.wind = wind

    def get_wind(self):
        return self.wind


class _FakeGarment:
    def __init__(self, wind):
        self.particle_system = _FakeParticleSystem(wind)
        self.particle_system_path = "/Particle_Attribute/env_0/garment_particle_system"


def test_collect_cloth_wind_readbacks_reports_requested_and_actual_values():
    garments = [{"shirt": _FakeGarment([3.0, 0.0, 0.0])}]

    rows = collect_cloth_wind_readbacks(garments, (3.0, 0.0, 0.0))

    assert rows == [
        {
            "env_id": 0,
            "garment_name": "shirt",
            "particle_system_path": "/Particle_Attribute/env_0/garment_particle_system",
            "actual_wind": [3.0, 0.0, 0.0],
            "matches_request": True,
            "readback_error": None,
        }
    ]


def test_collect_cloth_wind_readbacks_flags_mismatch_and_readback_error():
    class _BrokenParticleSystem:
        def get_wind(self):
            raise RuntimeError("not initialized")

    mismatched = _FakeGarment([0.0, 0.0, 0.0])
    broken = _FakeGarment([0.0, 0.0, 0.0])
    broken.particle_system = _BrokenParticleSystem()

    rows = collect_cloth_wind_readbacks(
        [{"shirt": mismatched, "towel": broken}], (3.0, 0.0, 0.0)
    )

    assert rows[0]["matches_request"] is False
    assert rows[1]["actual_wind"] is None
    assert "not initialized" in rows[1]["readback_error"]
