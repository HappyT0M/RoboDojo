from task.RoboDojo.tasks.rope_task_base import CpuRopeTask


class route_rope_around_posts(CpuRopeTask):
    instruction = (
        "Grasp the rope and guide its attached ball around the right-hand side "
        "of the vase, staying close to the tabletop, then place the ball on the mat behind the vase."
    )
