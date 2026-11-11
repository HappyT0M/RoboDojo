from env.reward_manager.deformable_manipulation import DeformableManipulationCommon
from task.RoboDojo.tasks.garment_goal_base import GarmentGoalTask


class WipeMarkedAreaWithClothCommon(DeformableManipulationCommon):
    instruction = "Wipe the entire marked area on the table using the cloth."


class wipe_marked_area_with_cloth(WipeMarkedAreaWithClothCommon, GarmentGoalTask):
    pass
