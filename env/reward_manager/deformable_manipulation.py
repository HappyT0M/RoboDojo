"""Adapter from loaded scene objects to the simulator-independent task metrics."""

from utils.deformable_manipulation_metrics import CoverProgress, PullProgress, WipeProgress, local_points
from utils.rope_task_metrics import _rotate_by_quaternion


def _flat(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [number for item in value for number in _flat(item)]


def _observe(parser, env_idx):
    config = parser.env.task_env_config["DeformableManipulation"]
    layout = parser.layout_manager

    def object_for(label):
        name = layout.get_instance_name(env_idx=env_idx, label=label)
        if name is None:
            raise ValueError(f"Missing {label!r} in deformable manipulation layout")
        return layout.get_scene_object(env_idx=env_idx, inst_name=name)

    target = object_for("goal")
    position, rotation = layout.get_instance_pose(env_idx=env_idx, label="goal")
    position, rotation = _flat(position), _flat(rotation)
    scale = _flat(target.scale)
    origin = _flat(parser.env.sim.scene.env_origins[env_idx])
    grippers = []
    for robot in parser.robot_manager.robot_list:
        if getattr(robot, "type", "target") != "target":
            continue
        pose = _flat(parser.robot_manager.get_real_endpose(robot, env_idx_list=[env_idx], is_relative=True)[env_idx])
        value = parser.robot_manager.get_end_effector_real_val(robot, env_idx_list=[env_idx])[env_idx]
        low, high = robot.gripper_scale
        values = _flat(value)
        # The first joint drives the gripper; mimic joints may have opposite signs.
        openness = (values[0] - low) / (high - low)
        if robot.gripper_move["sign"] < 0:
            openness = 1.0 - openness
        # X5's ee_link is the wrist (link6), not the fingertip contact center.
        axis = config.get("grasp_axis", [1., 0., 0.])
        offset = _rotate_by_quaternion(tuple(float(v) * robot.gripper_bias for v in axis), pose[3:7])
        grasp_position = [a + b for a, b in zip(pose[:3], offset)]
        grippers.append((local_points([grasp_position], position, rotation, scale)[0], openness))
    if config["kind"] == "pull":
        rope = object_for("rope")
        poses = [rope.get_link_pose(f"segment_{i:02d}") for i in range(int(config.get("link_count", 12)))]
        poses.append(rope.get_link_pose("ball"))
        points = [tuple(a - b for a, b in zip(_flat(pose)[:3], origin)) for pose in poses]
        local = local_points(points, position, rotation, scale)
        return config, (local[:-1], local[-1], grippers)
    cloth = object_for("cloth")
    points = [tuple(a - b for a, b in zip(_flat(point), origin)) for point in cloth.sample_mesh_vertices()[0]]
    # The new cloth assets are triangular grids; cache topology for this episode.
    state = parser.deformable_manipulation_state[env_idx]
    if "triangles" not in state:
        counts = cloth.prim.GetAttribute("faceVertexCounts").Get()
        indices = cloth.prim.GetAttribute("faceVertexIndices").Get()
        if counts is None or indices is None or any(count != 3 for count in counts):
            raise ValueError("Deformable manipulation requires a triangular cloth mesh")
        state["triangles"] = [tuple(int(v) for v in indices[i:i+3]) for i in range(0, len(indices), 3)]
    return config, (local_points(points, position, rotation, scale), state["triangles"], grippers)


def prepare(parser):
    """Capture the settled start state before the first action, once per episode."""
    for env_idx in range(parser.num_envs):
        if not parser.env.success[env_idx]:
            continue
        config, observation = _observe(parser, env_idx)
        metric = {"cover": CoverProgress, "wipe": WipeProgress, "pull": PullProgress}[config["kind"]](config)
        parser.deformable_manipulation_state[env_idx]["metric"] = metric
        metric.update(*observation)


def evaluate(parser, env_idx):
    _, observation = _observe(parser, env_idx)
    metric = parser.deformable_manipulation_state[env_idx]["metric"]
    return float(metric.update(*observation))


class DeformableManipulationCommon:
    """Task mixin; inherited task bases initialize and reset RewardManager."""

    def run_reward(self):
        prepare(self.reward_manager.func_parser)
        self.reward_manager.check([("is_deformable_manipulation_success", {})])

    def soft_reset(self, seed=None, options=None):
        # The task bases reset all metric/cumulative coverage state via RewardManager.
        return self.reset(seed=seed, options=options)
