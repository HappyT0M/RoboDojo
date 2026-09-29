"""Helpers for configuring a constant wind on task cloth particle systems."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence


def _normalize_wind(wind: Sequence[float] | None) -> list[float]:
    if wind is None:
        return [0.0, 0.0, 0.0]
    if isinstance(wind, (str, bytes)) or len(wind) != 3:
        raise ValueError("wind must be a three-component finite vector")
    values = [float(value) for value in wind]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("wind must be a three-component finite vector")
    return values


def apply_constant_cloth_wind(
    task_cfg: MutableMapping,
    wind: Sequence[float] | None,
    particle_drag: float | None = None,
) -> list[float]:
    """Set wind on every Garment instance in a loaded task config.

    ``wind=None`` explicitly disables configured wind by setting a zero vector.
    ``particle_drag`` is optional so a no-wind run leaves task material settings
    unchanged.
    """
    garments = task_cfg.get("Garment", [])
    if not garments:
        if wind is not None:
            raise ValueError("task has no Garment instances; cloth wind cannot be applied")
        return [0.0, 0.0, 0.0]

    vector = _normalize_wind(wind)
    drag = None if particle_drag is None else float(particle_drag)
    if drag is not None and not math.isfinite(drag):
        raise ValueError("particle_drag must be finite")

    for garment_cfg in garments:
        physics_cfg = garment_cfg.setdefault("physics", {})
        particle_system_cfg = physics_cfg.setdefault("particle_system", {})
        particle_system_cfg["wind"] = vector.copy()
        if drag is not None:
            particle_material_cfg = physics_cfg.setdefault("particle_material", {})
            particle_material_cfg["drag"] = drag
    return vector


def apply_cloth_runtime_overrides(scene_layout, task_cfg) -> None:
    """Overlay runtime garment physics settings onto a generated scene layout.

    Evaluation episodes load object physics from pre-generated layout JSON files,
    so changing the task YAML alone does not affect those instances. This copies
    the runtime ``physics.particle_system`` and ``physics.particle_material``
    values onto matching garment categories before the layout is instantiated.
    """
    layout_garments = scene_layout.get("Garment", {})
    task_garments = task_cfg.get("Garment", [])
    if not layout_garments or not task_garments:
        return

    overrides_by_category = {}
    fallback_overrides = None
    for garment_cfg in task_garments:
        physics_cfg = garment_cfg.get("physics", {})
        overrides = {
            section: dict(physics_cfg[section])
            for section in ("particle_system", "particle_material")
            if physics_cfg.get(section)
        }
        if not overrides:
            continue
        categories = garment_cfg.get("category", [])
        category_names = [
            category.get("name")
            for category in categories
            if isinstance(category, Mapping) and category.get("name")
        ]
        for category_name in category_names:
            overrides_by_category[category_name] = overrides
        if not category_names:
            fallback_overrides = overrides

    for category_name, instances in layout_garments.items():
        overrides = overrides_by_category.get(category_name, fallback_overrides)
        if not overrides:
            continue
        for instance in instances:
            physics_cfg = instance.setdefault("physics", {})
            for section, section_overrides in overrides.items():
                physics_cfg.setdefault(section, {}).update(section_overrides)


def collect_cloth_wind_readbacks(garments_by_env, requested_wind: Sequence[float]) -> list[dict]:
    """Read back wind values from instantiated Garment particle systems.

    ``garments_by_env`` is SceneManager's ``_garment_objects`` structure. A
    failed getter is reported in the returned row instead of interrupting a
    recording run.
    """
    requested = _normalize_wind(requested_wind)
    rows = []
    for env_id, garments in enumerate(garments_by_env or []):
        for garment_name, garment in (garments or {}).items():
            particle_system = getattr(garment, "particle_system", None)
            getter = getattr(particle_system, "get_wind", None)
            actual = None
            error = None
            matches = False
            try:
                if not callable(getter):
                    raise AttributeError("particle system has no get_wind() method")
                actual = _normalize_wind(getter())
                matches = all(
                    math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-6)
                    for value, expected in zip(actual, requested)
                )
            except Exception as exc:  # diagnostics should not abort video recording
                error = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "env_id": env_id,
                    "garment_name": str(garment_name),
                    "particle_system_path": getattr(garment, "particle_system_path", None),
                    "actual_wind": actual,
                    "matches_request": matches,
                    "readback_error": error,
                }
            )
    return rows
