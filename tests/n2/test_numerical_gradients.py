"""Development-only numerical VJP checks for every differentiable N2 input."""

from __future__ import annotations

import math
import unittest
import warnings
from collections.abc import Callable, Sequence
from typing import Any

warnings.filterwarnings("ignore", message=r"Failed to initialize NumPy.*")

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

PINNED_TORCH_VERSION = "2.13.0+cpu"
FINITE_DIFFERENCE_STEP = 1.0e-6
ABSOLUTE_TOLERANCE = 2.0e-6
RELATIVE_TOLERANCE = 2.0e-5


def _exact_pin_available() -> bool:
    return torch is not None and torch.__version__ == PINNED_TORCH_VERSION


def _upstream_like(value: Any) -> Any:
    indices = torch.arange(value.numel(), dtype=torch.float64, device="cpu")
    signs = torch.where(indices.remainder(2) == 0, 1.0, -1.0)
    return (signs * (indices + 1.0) / (value.numel() + 1.0)).reshape_as(value)


def _central_vjp(function: Callable[..., Any], inputs: Sequence[Any],
                 upstream: Any, input_index: int, step: float) -> Any:
    numerical = torch.empty_like(inputs[input_index])
    for element_index in range(inputs[input_index].numel()):
        plus = [value.detach().clone() for value in inputs]
        minus = [value.detach().clone() for value in inputs]
        plus[input_index].reshape(-1)[element_index] += step
        minus[input_index].reshape(-1)[element_index] -= step
        positive = (function(*plus) * upstream).sum().item()
        negative = (function(*minus) * upstream).sum().item()
        numerical.reshape(-1)[element_index] = (positive - negative) / (2.0 * step)
    return numerical


def _check_all_vjps(test: unittest.TestCase, function: Callable[..., Any],
                     inputs: Sequence[Any], *, step: float = FINITE_DIFFERENCE_STEP,
                     absolute: float = ABSOLUTE_TOLERANCE,
                     relative: float = RELATIVE_TOLERANCE) -> None:
    variables = tuple(value.detach().clone().requires_grad_(True) for value in inputs)
    output = function(*variables)
    upstream = _upstream_like(output)
    analytical = torch.autograd.grad((output * upstream).sum(), variables)
    for input_index, expected in enumerate(analytical):
        with test.subTest(input_index=input_index):
            numerical = _central_vjp(function, inputs, upstream, input_index, step)
            torch.testing.assert_close(
                expected, numerical, atol=absolute, rtol=relative,
                msg=lambda message: f"VJP mismatch for input {input_index}: {message}",
            )


@unittest.skipUnless(_exact_pin_available(),
                     "PyTorch 2.13.0+cpu is required for numerical gradient checks")
class N2NumericalGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(1729)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

    def test_embedding_weight_vjp_with_repeated_ids(self) -> None:
        ids = torch.tensor([[0, 2, 0], [1, 2, 1]], dtype=torch.int64, device="cpu")
        weight = torch.tensor(
            [[-0.75, 0.25], [1.5, -0.5], [0.125, 0.875]],
            dtype=torch.float64, device="cpu",
        )

        def embedding(candidate: Any) -> Any:
            return torch.nn.functional.embedding(ids, candidate)

        _check_all_vjps(self, embedding, (weight,))

    def test_linear_vjps_with_and_without_bias(self) -> None:
        value = torch.tensor(
            [[[-0.75, 0.25, 1.0], [1.5, -0.5, 0.125]]],
            dtype=torch.float64, device="cpu",
        )
        weight = torch.tensor(
            [[0.5, -1.0, 0.25], [-0.75, 0.125, 1.25]],
            dtype=torch.float64, device="cpu",
        )
        bias = torch.tensor([0.375, -0.625], dtype=torch.float64, device="cpu")

        def biased(x: Any, matrix: Any, offset: Any) -> Any:
            return torch.matmul(x, matrix.transpose(0, 1)) + offset

        def unbiased(x: Any, matrix: Any) -> Any:
            return torch.matmul(x, matrix.transpose(0, 1))

        _check_all_vjps(self, biased, (value, weight, bias))
        _check_all_vjps(self, unbiased, (value, weight))

    def test_affine_biased_layer_norm_vjps(self) -> None:
        value = torch.tensor(
            [[[-1.25, 0.375, 1.75, -0.625],
              [0.25, -1.5, 0.875, 2.25]]],
            dtype=torch.float64, device="cpu",
        )
        gamma = torch.tensor([0.75, -1.125, 0.5, 1.375],
                             dtype=torch.float64, device="cpu")
        beta = torch.tensor([-0.25, 0.125, 0.375, -0.5],
                            dtype=torch.float64, device="cpu")
        epsilon = 1.0e-5

        def layer_norm(x: Any, scale: Any, offset: Any) -> Any:
            mean = x.mean(dim=-1, keepdim=True)
            variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            return (x - mean) * torch.rsqrt(variance + epsilon) * scale + offset

        _check_all_vjps(self, layer_norm, (value, gamma, beta), step=2.0e-6,
                         absolute=5.0e-6, relative=3.0e-5)

    def test_exact_gelu_and_relu_vjps(self) -> None:
        gelu_input = torch.tensor(
            [[[-2.0, -0.5, 0.25, 1.75]]], dtype=torch.float64, device="cpu")
        relu_input = torch.tensor(
            [[[-2.0, -0.25, 0.375, 1.5]]], dtype=torch.float64, device="cpu")

        def exact_gelu(value: Any) -> Any:
            return 0.5 * value * (1.0 + torch.erf(value / math.sqrt(2.0)))

        def relu(value: Any) -> Any:
            return torch.where(value > 0, value, torch.zeros_like(value))

        _check_all_vjps(self, exact_gelu, (gelu_input,))
        _check_all_vjps(self, relu, (relu_input,))

    def test_residual_vjps_for_both_operands(self) -> None:
        value = torch.tensor(
            [[[-1.0, 0.25, 1.5], [0.75, -0.5, 2.0]]],
            dtype=torch.float64, device="cpu",
        )
        branch = torch.tensor(
            [[[0.125, -0.75, 0.5], [-1.25, 0.375, 0.875]]],
            dtype=torch.float64, device="cpu",
        )

        def residual(lhs: Any, rhs: Any) -> Any:
            return lhs + rhs

        _check_all_vjps(self, residual, (value, branch))

    def test_fixed_mask_train_dropout_and_eval_vjps(self) -> None:
        value = torch.tensor(
            [[[-1.0, 0.25, 1.5, -0.75], [0.5, 2.0, -0.125, 0.875]]],
            dtype=torch.float64, device="cpu",
        )
        mask = torch.tensor(
            [[[0.0, 4.0 / 3.0, 4.0 / 3.0, 0.0],
              [4.0 / 3.0, 0.0, 4.0 / 3.0, 4.0 / 3.0]]],
            dtype=torch.float64, device="cpu",
        )

        def train(candidate: Any) -> Any:
            return candidate * mask

        def evaluate(candidate: Any) -> Any:
            return candidate.clone()

        _check_all_vjps(self, train, (value,))
        _check_all_vjps(self, evaluate, (value,))


if __name__ == "__main__":
    unittest.main()
