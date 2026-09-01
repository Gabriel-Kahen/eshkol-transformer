#!/usr/bin/env python3
"""Generate deterministic development-only O2 AdamW reference data."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import torch

from tests.q0.oracle_format import encode_fixture

PINNED_TORCH_VERSION = "2.13.0+cpu"
GENERATOR_NAME = "o2.pytorch.adamw"
GENERATOR_VERSION = 1
SEED = 24681357
ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "tests" / "q0" / "requirements-oracle.lock"


def _f32(bits: str) -> float:
    """Decode one exact O2 binary32 configuration spelling."""
    if len(bits) != 8 or bits.lower() != bits:
        raise RuntimeError(f"invalid binary32 spelling: {bits!r}")
    return struct.unpack(">f", bytes.fromhex(bits))[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f64_bits(value: float) -> str:
    return struct.pack(">d", value).hex()


def _tensor_record(
    name: str, role: str, tensor: torch.Tensor
) -> dict[str, object]:
    if tensor.device.type != "cpu":
        raise RuntimeError("O2 oracle tensors must be explicit CPU tensors")
    flattened = tensor.detach().contiguous().reshape(-1).tolist()
    if tensor.dtype == torch.float32:
        dtype = "float32"
        encoding = "ieee754-hex-be"
        data = [struct.pack(">f", float(value)).hex() for value in flattened]
    elif tensor.dtype == torch.float64:
        dtype = "float64"
        encoding = "ieee754-hex-be"
        data = [struct.pack(">d", float(value)).hex() for value in flattened]
    elif tensor.dtype == torch.int64:
        dtype = "int64"
        encoding = "twos-complement-hex-be"
        data = [int(value).to_bytes(8, "big", signed=True).hex() for value in flattened]
    else:
        raise RuntimeError(f"unsupported O2 oracle dtype: {tensor.dtype}")
    return {
        "data": data,
        "device": "cpu",
        "dtype": dtype,
        "encoding": encoding,
        "layout": "row_major",
        "name": name,
        "role": role,
        "shape": list(tensor.shape),
    }


def _case(
    name: str,
    operation: str,
    inputs: list[str],
    outputs: list[str],
    *,
    kind: str = "parity",
    exact: bool = False,
) -> dict[str, object]:
    return {
        "expectation": {"error": None, "outputs": outputs},
        "inputs": inputs,
        "kind": kind,
        "name": name,
        "operation": operation,
        "tolerance": {
            "absolute": _f64_bits(0.0 if exact else 2.0e-6),
            "equal_nan": False,
            "relative": _f64_bits(0.0 if exact else 2.0e-5),
        },
    }


def _adamw(
    groups: list[dict[str, object]],
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        groups,
        foreach=False,
        fused=False,
        capturable=False,
        differentiable=False,
        maximize=False,
        amsgrad=False,
    )


def _single_parameter_case(
    tensors: list[dict[str, object]], cases: list[dict[str, object]]
) -> None:
    initial = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    gradients = (
        torch.tensor([0.25, -0.5, 0.0], dtype=torch.float32),
        torch.tensor([-0.125, 0.25, 1.0], dtype=torch.float32),
    )
    parameter = initial.clone().requires_grad_(True)
    learning_rate = _f32("3c23d70a")
    beta1 = _f32("3f666666")
    beta2 = _f32("3f7fbe77")
    epsilon = _f32("322bcc77")
    weight_decay = _f32("3dcccccd")
    optimizer = _adamw(
        [
            {
                "params": [parameter],
                "lr": learning_rate,
                "betas": (beta1, beta2),
                "eps": epsilon,
                "weight_decay": weight_decay,
            }
        ]
    )
    tensors.extend(
        [
            _tensor_record("single.input.gradients", "input", torch.stack(gradients)),
            _tensor_record("single.input.parameter", "input", initial),
            _tensor_record(
                "single.input.group_options",
                "input",
                torch.tensor(
                    [learning_rate, beta1, beta2, epsilon, weight_decay],
                    dtype=torch.float32,
                ),
            ),
            _tensor_record(
                "single.input.update_indices",
                "input",
                torch.tensor([1, 2], dtype=torch.int64),
            ),
        ]
    )
    outputs: list[str] = []
    for index, gradient in enumerate(gradients, start=1):
        parameter.grad = gradient.clone()
        optimizer.step()
        state = optimizer.state[parameter]
        for suffix, value in (
            ("parameter", parameter),
            ("exp_avg", state["exp_avg"]),
            ("exp_avg_sq", state["exp_avg_sq"]),
        ):
            name = f"single.output.step{index}.{suffix}"
            tensors.append(_tensor_record(name, "expected", value))
            outputs.append(name)
    cases.append(
        _case(
            "adamw_bias_correction_decay_two_steps",
            "optimizer.adamw.step",
            [
                "single.input.gradients",
                "single.input.group_options",
                "single.input.parameter",
                "single.input.update_indices",
            ],
            outputs,
        )
    )


def _multiple_group_case(
    tensors: list[dict[str, object]], cases: list[dict[str, object]]
) -> None:
    first_initial = torch.tensor([2.0, -1.0], dtype=torch.float32)
    second_initial = torch.tensor([0.25, -0.75, 1.5], dtype=torch.float32)
    first_gradient = torch.tensor([0.5, -0.25], dtype=torch.float32)
    second_gradient = torch.tensor([-1.0, 0.0, 0.125], dtype=torch.float32)
    first = first_initial.clone().requires_grad_(True)
    second = second_initial.clone().requires_grad_(True)
    first_options = (
        _f32("3ca3d70a"),
        _f32("3f4ccccd"),
        _f32("3f733333"),
        _f32("358637bd"),
        _f32("00000000"),
    )
    second_options = (
        _f32("3ba3d70a"),
        _f32("3f333333"),
        _f32("3f666666"),
        _f32("3727c5ac"),
        _f32("3e4ccccd"),
    )
    optimizer = _adamw(
        [
            {
                "params": [first],
                "lr": first_options[0],
                "betas": first_options[1:3],
                "eps": first_options[3],
                "weight_decay": first_options[4],
            },
            {
                "params": [second],
                "lr": second_options[0],
                "betas": second_options[1:3],
                "eps": second_options[3],
                "weight_decay": second_options[4],
            },
        ]
    )
    first.grad = first_gradient.clone()
    second.grad = second_gradient.clone()
    optimizer.step()
    tensors.extend(
        [
            _tensor_record("groups.input.first_gradient", "input", first_gradient),
            _tensor_record(
                "groups.input.first_options",
                "input",
                torch.tensor(first_options, dtype=torch.float32),
            ),
            _tensor_record("groups.input.first_parameter", "input", first_initial),
            _tensor_record("groups.input.second_gradient", "input", second_gradient),
            _tensor_record(
                "groups.input.second_options",
                "input",
                torch.tensor(second_options, dtype=torch.float32),
            ),
            _tensor_record("groups.input.second_parameter", "input", second_initial),
            _tensor_record(
                "groups.input.assignments",
                "input",
                torch.tensor([0, 1], dtype=torch.int64),
            ),
        ]
    )
    outputs: list[str] = []
    for prefix, parameter in (("first", first), ("second", second)):
        state = optimizer.state[parameter]
        for suffix, value in (
            ("parameter", parameter),
            ("exp_avg", state["exp_avg"]),
            ("exp_avg_sq", state["exp_avg_sq"]),
        ):
            name = f"groups.output.{prefix}.{suffix}"
            tensors.append(_tensor_record(name, "expected", value))
            outputs.append(name)
    cases.append(
        _case(
            "adamw_multiple_parameter_groups",
            "optimizer.adamw.groups",
            [
                "groups.input.first_gradient",
                "groups.input.first_options",
                "groups.input.first_parameter",
                "groups.input.assignments",
                "groups.input.second_gradient",
                "groups.input.second_options",
                "groups.input.second_parameter",
            ],
            outputs,
        )
    )


def _clipping_cases(
    tensors: list[dict[str, object]], cases: list[dict[str, object]]
) -> None:
    for label, gradient in (
        ("boundary", torch.tensor([3.0, 4.0], dtype=torch.float32)),
        ("above", torch.tensor([6.0, 8.0], dtype=torch.float32)),
    ):
        maximum = torch.tensor(_f32("40a00000"), dtype=torch.float32)
        norm = torch.linalg.vector_norm(gradient, ord=2)
        clipped = gradient.clone() if norm <= maximum else gradient * (maximum / norm)
        input_name = f"clip.input.{label}.gradient"
        maximum_name = f"clip.input.{label}.maximum"
        gradient_name = f"clip.output.{label}.gradient"
        norm_name = f"clip.output.{label}.norm"
        tensors.extend(
            [
                _tensor_record(input_name, "input", gradient),
                _tensor_record(maximum_name, "input", maximum),
                _tensor_record(gradient_name, "expected", clipped),
                _tensor_record(norm_name, "expected", norm.to(torch.float32)),
            ]
        )
        cases.append(
            _case(
                f"global_l2_clip_{label}",
                "optimizer.gradient.clip",
                [input_name, maximum_name],
                [gradient_name, norm_name],
                kind="boundary",
            )
        )


def _accumulation_case(
    tensors: list[dict[str, object]], cases: list[dict[str, object]]
) -> None:
    initial = torch.tensor([0.5, -1.0], dtype=torch.float32)
    features = (
        torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32),
        torch.tensor([[0.25, -2.0], [2.0, 1.0]], dtype=torch.float32),
    )
    targets = (
        torch.tensor([0.0, 1.0], dtype=torch.float32),
        torch.tensor([-0.5, 2.0], dtype=torch.float32),
    )
    masks = (
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([1.0, 1.0], dtype=torch.float32),
    )

    numerator_gradients: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for feature, target, mask in zip(features, targets, masks, strict=True):
        local = initial.clone().requires_grad_(True)
        losses = 0.5 * (feature.mv(local) - target).square()
        weight = mask.sum()
        (losses * mask).sum().backward()
        numerator_gradients.append(local.grad.detach().clone())
        weights.append(weight.detach().clone())

    combined = initial.clone().requires_grad_(True)
    numerator = torch.zeros((), dtype=torch.float32)
    total_weight = torch.zeros((), dtype=torch.float32)
    for feature, target, mask in zip(features, targets, masks, strict=True):
        losses = 0.5 * (feature.mv(combined) - target).square()
        numerator = numerator + (losses * mask).sum()
        total_weight = total_weight + mask.sum()
    (numerator / total_weight).backward()
    expected = combined.grad.detach().clone()
    accumulated = sum(numerator_gradients) / total_weight
    if not torch.equal(accumulated, expected):
        raise RuntimeError("numerator-gradient accumulation differs from global backward")

    learning_rate = _f32("3c23d70a")
    beta1 = _f32("3f666666")
    beta2 = _f32("3f7fbe77")
    epsilon = _f32("322bcc77")
    weight_decay = _f32("3dcccccd")
    group = {
        "lr": learning_rate,
        "betas": (beta1, beta2),
        "eps": epsilon,
        "weight_decay": weight_decay,
    }
    accumulated_parameter = initial.clone().requires_grad_(True)
    reference_parameter = initial.clone().requires_grad_(True)
    accumulated_optimizer = _adamw(
        [{"params": [accumulated_parameter], **group}]
    )
    reference_optimizer = _adamw(
        [{"params": [reference_parameter], **group}]
    )
    accumulated_parameter.grad = accumulated.clone()
    reference_parameter.grad = expected.clone()
    accumulated_optimizer.step()
    reference_optimizer.step()
    accumulated_state = accumulated_optimizer.state[accumulated_parameter]
    reference_state = reference_optimizer.state[reference_parameter]
    for left, right in (
        (accumulated_parameter, reference_parameter),
        (accumulated_state["exp_avg"], reference_state["exp_avg"]),
        (accumulated_state["exp_avg_sq"], reference_state["exp_avg_sq"]),
    ):
        if not torch.equal(left, right):
            raise RuntimeError("accumulated and global-backward AdamW updates differ")

    tensors.extend(
        [
            _tensor_record(
                "accumulation.input.numerator_gradients",
                "input",
                torch.stack(numerator_gradients),
            ),
            _tensor_record(
                "accumulation.input.weights", "input", torch.stack(weights)
            ),
            _tensor_record(
                "accumulation.input.group_options",
                "input",
                torch.tensor(
                    [learning_rate, beta1, beta2, epsilon, weight_decay],
                    dtype=torch.float32,
                ),
            ),
            _tensor_record("accumulation.input.parameter", "input", initial),
            _tensor_record("accumulation.output.effective_gradient", "expected", expected),
            _tensor_record("accumulation.output.total_weight", "expected", total_weight),
            _tensor_record(
                "accumulation.output.next_parameter",
                "expected",
                accumulated_parameter,
            ),
            _tensor_record(
                "accumulation.output.exp_avg",
                "expected",
                accumulated_state["exp_avg"],
            ),
            _tensor_record(
                "accumulation.output.exp_avg_sq",
                "expected",
                accumulated_state["exp_avg_sq"],
            ),
        ]
    )
    cases.append(
        _case(
            "weighted_microbatch_accumulation",
            "optimizer.gradient.accumulate",
            [
                "accumulation.input.group_options",
                "accumulation.input.numerator_gradients",
                "accumulation.input.parameter",
                "accumulation.input.weights",
            ],
            [
                "accumulation.output.effective_gradient",
                "accumulation.output.exp_avg",
                "accumulation.output.exp_avg_sq",
                "accumulation.output.next_parameter",
                "accumulation.output.total_weight",
            ],
            kind="repeated_input",
        )
    )


def _linear_factor(n: int, warmup: int, total: int, minimum: float) -> float:
    if warmup > 0 and n <= warmup:
        return n / warmup
    if n <= total:
        return 1.0 - (1.0 - minimum) * (n - warmup) / (total - warmup)
    return minimum


def _schedule_cases(
    tensors: list[dict[str, object]], cases: list[dict[str, object]]
) -> None:
    updates = torch.arange(1, 9, dtype=torch.int64)
    constant = torch.ones(8, dtype=torch.float32)
    warmup = 2
    total = 6
    minimum = _f32("3dcccccd")
    base_learning_rate = _f32("3ca3d70a")
    linear = torch.tensor(
        [
            _linear_factor(n, warmup=warmup, total=total, minimum=minimum)
            for n in range(1, 9)
        ],
        dtype=torch.float32,
    )
    effective_lr = torch.tensor(base_learning_rate, dtype=torch.float32) * linear
    tensors.extend(
        [
            _tensor_record("schedule.input.updates", "input", updates),
            _tensor_record(
                "schedule.input.linear_definition",
                "input",
                torch.tensor([warmup, total], dtype=torch.int64),
            ),
            _tensor_record(
                "schedule.input.linear_scalars",
                "input",
                torch.tensor([minimum, base_learning_rate], dtype=torch.float32),
            ),
            _tensor_record("schedule.output.constant_factors", "expected", constant),
            _tensor_record("schedule.output.linear_factors", "expected", linear),
            _tensor_record("schedule.output.linear_learning_rates", "expected", effective_lr),
        ]
    )
    cases.extend(
        [
            _case(
                "schedule_constant_successful_updates",
                "optimizer.schedule.constant",
                ["schedule.input.updates"],
                ["schedule.output.constant_factors"],
                exact=True,
            ),
            _case(
                "schedule_linear_warmup_decay_clamp",
                "optimizer.schedule.linear",
                [
                    "schedule.input.linear_definition",
                    "schedule.input.linear_scalars",
                    "schedule.input.updates",
                ],
                [
                    "schedule.output.linear_factors",
                    "schedule.output.linear_learning_rates",
                ],
                kind="boundary",
                exact=True,
            ),
        ]
    )


def build_payload() -> dict[str, object]:
    if torch.__version__ != PINNED_TORCH_VERSION:
        raise RuntimeError(
            f"expected torch {PINNED_TORCH_VERSION}, found {torch.__version__}"
        )
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    tensors: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    _single_parameter_case(tensors, cases)
    _multiple_group_case(tensors, cases)
    _clipping_cases(tensors, cases)
    _accumulation_case(tensors, cases)
    _schedule_cases(tensors, cases)
    tensors.sort(key=lambda tensor: str(tensor["name"]))
    cases.sort(key=lambda case: str(case["name"]))

    return {
        "cases": cases,
        "generator": {
            "dependency_lock_sha256": _sha256(LOCK),
            "framework": {"name": "pytorch", "version": PINNED_TORCH_VERSION},
            "name": GENERATOR_NAME,
            "seed": SEED,
            "source_sha256": _sha256(Path(__file__).resolve()),
            "version": GENERATOR_VERSION,
        },
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.write_bytes(encode_fixture(build_payload()))


if __name__ == "__main__":
    main()
