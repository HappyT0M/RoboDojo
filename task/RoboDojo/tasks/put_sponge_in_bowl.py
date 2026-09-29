"""FEM sponge manipulation task: place the sponge inside the bowl."""

from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager


class PutSpongeInBowlCommon:
    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 400
        self.containment_args = {
            "deformable_label": "sponge",
            "container_label": "bowl",
            # The inscribed square is conservative for the bowl's circular rim.
            "interior_bounds": ((-0.045, -0.045, -0.024), (0.045, 0.045, 0.028)),
            "rim_z": 0.028,
            "min_containment_fraction": 0.85,
            "stable_steps": 30,
        }

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def run_reward(self):
        self.reward_manager.check(
            [self.reward_manager.is_deformable_in_container(**self.containment_args)]
        )

    def get_score(self):
        rm = self.reward_manager
        rm.score(
            [[rm.is_deformable_in_container(**self.containment_args, update=False)]],
            [100],
            score_mode="transition",
        )

    def gen_instruction(self, env_idx):
        return ["Pick up the sponge and place it fully inside the bowl."]


class put_sponge_in_bowl(PutSpongeInBowlCommon, TaskEnv):
    pass
