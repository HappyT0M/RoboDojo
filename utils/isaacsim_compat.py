"""Small compatibility shims for Isaac Sim API differences."""


def ensure_particle_system_wind_compat(particle_system_cls) -> bool:
    """Alias the legacy singular setter to the 5.x particle view setter.

    Some Isaac Sim builds pass ``winds`` to ``ParticleSystem.__init__`` but call
    ``self.set_wind(winds)`` internally, although the vectorized view exposes
    ``set_winds(values, indices=None)``. Return True only when the alias was
    installed.
    """
    if callable(getattr(particle_system_cls, "set_wind", None)):
        return False

    plural_setter = getattr(particle_system_cls, "set_winds", None)
    if not callable(plural_setter):
        raise RuntimeError(
            "Isaac Sim ParticleSystem has neither set_wind() nor set_winds(); "
            "check the installed isaacsim.core.prims extension version."
        )

    particle_system_cls.set_wind = plural_setter
    return True
