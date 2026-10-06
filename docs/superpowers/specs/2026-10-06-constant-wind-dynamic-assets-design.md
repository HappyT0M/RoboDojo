# Constant Wind for Deformable and Movable Task Assets

## Goal

Extend the existing constant-wind recording workflow so one wind vector can affect cloth, the rope-and-ball articulation, and movable task rigid bodies. The feature remains a recording-time override and does not edit task YAML or generated asset files.

## Scope and exclusions

- Keep native particle-system wind for `Garment` assets.
- Apply aerodynamic drag to movable `RigidObject` assets and eligible movable task `ArticulationObject` assets, including rope links and the attached ball.
- Explicitly exclude robot articulations, fixed-base bodies, static `Geometry`, tables, rooms, and backgrounds.
- Do not add fluid wind in this first version; particle-fluid behavior needs separate API and stability validation.
- Preserve the existing `--wind WX WY WZ` and `--no-wind` controls. The vector represents ambient wind velocity in m/s, not a force value.

## Design

The recording script will inject runtime overrides into the loaded task configuration, as it already does for cloth. A recording-time wind controller will discover eligible scene objects by their registered type and task-object identity, never by applying forces to every articulation. It will keep a list of affected bodies and apply drag at every physics update while a recording episode runs. The robot and static fixture registries are not recipients.

Cloth continues to receive the native particle-system wind. Movable rigid bodies and eligible articulation links receive a force based on relative air/object velocity, `F = 0.5 * rho * Cd * A * |v_rel| * v_rel`, where `v_rel` is ambient wind minus body velocity. Cross-sectional area is estimated from runtime body bounds (or link collision dimensions when available); drag coefficient and force scale have conservative configurable defaults. The diagnostic log records requested wind, selected objects, applied-force summaries, and any bodies skipped with a reason.

Wind state is reset between runs and is zeroed when `--no-wind` is selected. If an object cannot provide the required dynamic-body view or dimensions, the script reports it clearly and continues with other eligible objects rather than silently claiming wind was applied.

## Validation

- Unit tests verify wind-vector validation, recipient filtering (including robot/static exclusions), force direction and zero-relative-wind behavior, and no-wind reset behavior without importing Isaac Sim.
- Existing cloth-wind tests continue to pass.
- An Isaac Sim smoke run records `put_rope_ball_in_basket` with wind disabled and enabled, confirms the rope/ball are selected while the robot and basket are not, and checks the per-step force diagnostics.
- A movable-rigid task smoke run confirms at least one eligible rigid body receives force. Compare against a zero-wind run to ensure the feature does not affect static fixtures or robot control.

## Risks and limits

The wind speed alone does not determine force on rigid objects. Results depend on estimated projected area, drag coefficient, body orientation, mass, and simulation substeps. The initial implementation should favor stability and expose tuning parameters instead of claiming physically exact aerodynamics. Runtime APIs differ across Isaac Sim builds, so the smoke test must run in the target simulator environment before describing the feature as validated.
