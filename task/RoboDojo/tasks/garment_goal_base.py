"""Shared evaluation loop for scene-based garment manipulation goals."""

from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager


class GarmentGoalTask(TaskEnv):
    instruction = "Manipulate the garment to the requested target configuration."

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 900

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def run_reward(self):
        self.reward_manager.check([("is_garment_task_success", {})])

    def gen_instruction(self, env_idx):
        return [self.instruction]
