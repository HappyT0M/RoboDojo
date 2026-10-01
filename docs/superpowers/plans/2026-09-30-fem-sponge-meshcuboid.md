# FEM Sponge MeshCuboid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Create the RoboDojo FEM sponge through Isaac Lab `MeshCuboidCfg` so the spawner applies deformable schemas while preserving the PhysX FEM task route.

**Architecture:** Keep the `DeformableObject` wrapper and scene lifecycle unchanged. Build the mesh from configured dimensions and scale, attach existing FEM body/material and visual settings, and stop requiring the wrapper to open a USD mesh. Validate locally with config and static checks, then use the existing server smoke script to verify schema, nodal state, and reset under Isaac Sim.

**Tech Stack:** Python, Isaac Lab `MeshCuboidCfg`, PhysX FEM, pytest.

---

## Files and responsibilities

- Modify `env/scene_manager/objects/deformable.py`: create a procedural cuboid spawn config while retaining the constructor shape used by `SceneManager`.
- Modify `scripts/test_fem_sponge.py`: instantiate the wrapper without relying on the sponge USD asset, then check nodal state and reset as before.
- Modify `tests/test_put_sponge_in_bowl_config.py`: assert layout dimensions and that the smoke test is independent of loading the old USDA.
- Modify `docs/FEM_SPONGE_BOWL.md`: document the procedural FEM mesh and server validation limits.

## Task 1: Add a failing configuration regression check

1. Update the task config regression to assert the fixed layout carries `size == [0.06, 0.05, 0.03]` and that the smoke script passes an empty USD path. Remove the assertion that a sponge USDA file must exist for the smoke test.
2. Run `pytest tests/test_put_sponge_in_bowl_config.py -q`; expect failure until the smoke script and wrapper are updated.

## Task 2: Switch the FEM wrapper to procedural geometry

1. In `DeformableObject.__init__`, retain the `usd_path` argument for caller compatibility but remove the filesystem existence check and do not use it as spawn geometry.
2. Read `self.deformable_config["size"]` and create `sim_utils.MeshCuboidCfg(size=..., scale=self.scale, deformable_props=..., visual_material=...)`.
3. Preserve the existing FEM material values and compatibility behavior: assign `physics_material` after constructing the spawner config, then pass the config through `DeformableObjectCfg` as before.
4. Run `python -m py_compile env/scene_manager/objects/deformable.py`.

## Task 3: Make the server smoke test exercise the new path

1. Pass `usd_path=""` to `DeformableObject` so the smoke test proves no USDA is opened by the wrapper.
2. Keep checks for a finite `(N, 3)` nodal-position array, simulation advancement, and reset error at or below `1e-3 m`.
3. Run `python -m py_compile scripts/test_fem_sponge.py` and the config regression from Task 1.

## Task 4: Update task documentation and run local checks

1. Describe the sponge as a procedurally generated cuboid FEM mesh and retain the current dimensions, material parameters, estimated mass, and task success criteria.
2. Run `pytest tests/test_put_sponge_in_bowl_config.py tests/test_metadata_and_fem_device.py -q`.
3. Run `python -m py_compile env/scene_manager/objects/deformable.py scripts/test_fem_sponge.py` and `git diff --check`.
4. Report that actual schema application, soft-body contact, compression, and rebound remain to be verified on AutoDL with Isaac Sim; provide the smoke-test command.

## Scope checks

- Do not modify bowl assets, task layout, reward logic, wind configuration, robot configuration, or Isaac Sim/Isaac Lab dependencies.
- Keep the previous custom USDA and metadata in the repository unless later evidence shows the layout loader can safely remove them; this fix only stops using the USDA as the FEM spawn geometry.
