from env.reward_manager.deformable_manipulation import DeformableManipulationCommon
from task.RoboDojo.tasks.garment_goal_base import GarmentGoalTask


class CoverObjectWithClothCommon(DeformableManipulationCommon):
    instruction = "Cover the object with the cloth so that it is fully covered from above."


class cover_object_with_cloth(CoverObjectWithClothCommon, GarmentGoalTask):
    pass
