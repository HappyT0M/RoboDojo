"""Helpers for moving object pose values into NumPy-based RoboDojo code."""

from __future__ import annotations

import numpy as np
import torch


def to_numpy_pose_component(value) -> np.ndarray:
    """Return an independent host NumPy array from a pose component.

    Isaac Lab returns rigid and articulation poses as tensors on the simulation
    device. RoboDojo's layout and reward utilities use NumPy, so CUDA tensors
    must be detached and copied to CPU before those utilities consume them.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()
