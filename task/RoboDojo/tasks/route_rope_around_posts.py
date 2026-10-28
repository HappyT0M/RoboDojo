from task.RoboDojo.tasks.rope_task_base import CpuRopeTask


class route_rope_around_posts(CpuRopeTask):
    instruction = (
        "Route the rope around both posts, then place its attached ball inside the target slot beyond them."
    )
