"""Model-independent evaluation wind lifecycle (Isaac imports are lazy)."""

import math

from utils.cloth_wind import _normalize_wind


class EvalWindError(RuntimeError):
    """A requested perturbation could not be applied; the trial is invalid."""


class EvalWind:
    """Bind fresh rigid views after reset; never apply forces during settling."""

    def __init__(self, wind=None, force_scale=1.0):
        self.wind = _normalize_wind(wind)
        self.force_scale = float(force_scale)
        if not math.isfinite(self.force_scale) or self.force_scale < 0:
            raise ValueError("rigid_force_scale must be finite and non-negative")
        self.controller = None
        self.skipped = []
        self.step_count = 0
        self.body_paths = []

    def stop(self):
        self.controller = None

    def start(self, scene_manager):
        self.stop()
        self.step_count = 0
        self.body_paths = []
        self.skipped = []
        if not any(self.wind):
            return
        if self.force_scale > 0:
            from utils.rigid_wind import create_rigid_wind_controller

            try:
                self.controller, self.skipped = create_rigid_wind_controller(
                    scene_manager, self.wind, force_scale=self.force_scale,
                )
            except Exception as exc:
                raise EvalWindError(f"Cannot initialize rigid wind: {exc}") from exc
        has_cloth = any(bool(items) for items in getattr(scene_manager, "_garment_objects", []))
        if self.controller is None and not has_cloth:
            raise EvalWindError(
                f"Wind requested but no eligible task bodies receive it; skipped={self.skipped}"
            )
        if self.controller is not None:
            self.body_paths = [body.prim_path for body in self.controller.bodies]
        print(f"[wind] world velocity={self.wind} m/s, rigid_force_scale={self.force_scale}, "
              f"rigid_bodies={len(self.body_paths)}, enabled after reset", flush=True)

    def step(self):
        if self.controller is not None:
            try:
                self.controller.step()
            except Exception as exc:
                raise EvalWindError(f"Cannot apply rigid wind: {exc}") from exc
            self.step_count += 1
            if self.step_count == 1:
                diagnostics = self.controller.last_diagnostics
                print(f"[wind] first physics step: applied={diagnostics['applied_body_count']} bodies", flush=True)
            elif self.step_count % 1000 == 0:
                diagnostics = self.controller.last_diagnostics
                max_force = max(
                    (body['force_norm_n'] for body in diagnostics.get('bodies', [])),
                    default=0.0,
                )
                print(f"[wind] physics_step={self.step_count}, velocity={self.wind} m/s, "
                      f"applied={diagnostics['applied_body_count']} bodies, "
                      f"max_force={max_force:.6g} N", flush=True)

    def metadata(self):
        return {
            "velocity_m_s": self.wind,
            "coordinate_frame": "world",
            "rigid_force_scale": self.force_scale,
            "air_density_kg_m3": 1.225,
            "drag_coefficient": 1.0,
            "area_model": "projected_local_bounding_box",
            "rigid_start_phase": "after_reset_before_policy_observation",
            "cloth_start_phase": "scene_initialization (existing behavior)",
            "rigid_body_paths": self.body_paths,
            "skipped": self.skipped,
            "rigid_physics_steps": self.step_count,
            "last_rigid_diagnostics": self.controller.last_diagnostics if self.controller else None,
        }
