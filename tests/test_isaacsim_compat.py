from utils.isaacsim_compat import ensure_particle_system_wind_compat


def test_aliases_single_wind_call_to_plural_setter_when_needed():
    class ParticleSystemView:
        def set_winds(self, values, indices=None):
            self.received = (values, indices)

    assert ensure_particle_system_wind_compat(ParticleSystemView) is True

    instance = ParticleSystemView()
    instance.set_wind([[3.0, 0.0, 0.0]])

    assert instance.received == ([[3.0, 0.0, 0.0]], None)


def test_does_not_replace_existing_single_wind_setter():
    class ParticleSystemView:
        def set_wind(self, value):
            self.received = value

        def set_winds(self, values, indices=None):
            raise AssertionError("plural setter should not replace the existing API")

    original_set_wind = ParticleSystemView.set_wind

    assert ensure_particle_system_wind_compat(ParticleSystemView) is False
    assert ParticleSystemView.set_wind is original_set_wind
