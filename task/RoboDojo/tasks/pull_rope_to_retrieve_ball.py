from env.reward_manager.deformable_manipulation import DeformableManipulationCommon
from task.RoboDojo.tasks.rope_task_base import CpuRopeTask


class PullRopeToRetrieveBallCommon(DeformableManipulationCommon):
    instruction = "Pull the free end of the rope to bring its attached ball into the marked target area."


class pull_rope_to_retrieve_ball(PullRopeToRetrieveBallCommon, CpuRopeTask):
    pass
