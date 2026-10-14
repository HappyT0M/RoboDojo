import numpy as np
import pytest
import torch

from utils.pose_utils import to_numpy_pose_component


def test_to_numpy_pose_component_converts_cpu_tensor_to_independent_array():
    source = torch.tensor([1.0, 2.0, 3.0])

    result = to_numpy_pose_component(source)

    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
    assert isinstance(result, np.ndarray)
    result[0] = -1.0
    assert source[0].item() == 1.0


def test_to_numpy_pose_component_converts_cuda_tensor_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this test environment")

    source = torch.tensor([1.0, 2.0, 3.0], device="cuda")

    result = to_numpy_pose_component(source)

    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
    assert isinstance(result, np.ndarray)


def test_to_numpy_pose_component_accepts_numpy_values():
    source = np.array([1.0, 2.0, 3.0])

    result = to_numpy_pose_component(source)

    np.testing.assert_array_equal(result, source)
    assert result is not source
