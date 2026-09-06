#!/usr/bin/env python3
"""Generate deterministic CPU-f32 development oracles for N2 primitives."""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

warnings.filterwarnings("ignore", message=r"Failed to initialize NumPy.*")

try:
    import torch
except ModuleNotFoundError:  # Keep the independent Philox checks runnable.
    torch = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.q0.oracle_format import encode_fixture

PINNED_TORCH_VERSION = "2.13.0+cpu"
GENERATOR_NAME = "n2.pytorch.primitives"
GENERATOR_VERSION = 1
SEED = 1729

PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
U32_MASK = (1 << 32) - 1
U64_MASK = (1 << 64) - 1
U128_MASK = (1 << 128) - 1

PHILOX_KNOWN_VECTORS = (
    ((0, 0, 0, 0), (0, 0),
     (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((U32_MASK,) * 4, (U32_MASK,) * 2,
     (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
     (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
)


def _f32(value: float) -> float:
    """Round one development-oracle scalar to IEEE-754 binary32."""
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _mulhilo32(lhs: int, rhs: int) -> tuple[int, int]:
    product = (lhs & U32_MASK) * (rhs & U32_MASK)
    return (product >> 32) & U32_MASK, product & U32_MASK


def philox4x32_10(counter: Sequence[int], key: Sequence[int]) -> tuple[int, ...]:
    """Random123 Philox4x32-10 over four counter and two key words."""
    if len(counter) != 4 or len(key) != 2:
        raise ValueError("Philox requires four counter words and two key words")
    c0, c1, c2, c3 = (int(word) & U32_MASK for word in counter)
    k0, k1 = (int(word) & U32_MASK for word in key)
    for round_index in range(10):
        hi0, lo0 = _mulhilo32(PHILOX_M0, c0)
        hi1, lo1 = _mulhilo32(PHILOX_M1, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if round_index != 9:
            k0 = (k0 + PHILOX_W0) & U32_MASK
            k1 = (k1 + PHILOX_W1) & U32_MASK
    return c0, c1, c2, c3


def _u64(value: int) -> int:
    return value & U64_MASK


def _i64(value: int) -> int:
    value &= U64_MASK
    return value if value < (1 << 63) else value - (1 << 64)


def philox_lanes(state: Sequence[int], element_count: int) -> tuple[tuple[int, ...],
                                                                      tuple[int, ...]]:
    """Return lane words and the block-advanced signed-i64 state."""
    if len(state) != 4 or int(state[0]) != 1:
        raise ValueError("dropout RNG state must be i64[4] with version 1")
    if element_count < 0:
        raise ValueError("element count must be non-negative")
    key64 = _u64(int(state[1]))
    low64 = _u64(int(state[2]))
    high64 = _u64(int(state[3]))
    counter128 = low64 | (high64 << 64)
    block_count = element_count // 4 + int(element_count % 4 != 0)
    if counter128 + block_count > U128_MASK:
        raise OverflowError("Philox counter overflow")
    key = (key64 & U32_MASK, key64 >> 32)
    words: list[int] = []
    for block in range(block_count):
        current = counter128 + block
        counter = tuple((current >> shift) & U32_MASK for shift in (0, 32, 64, 96))
        words.extend(philox4x32_10(counter, key))
    next_counter = counter128 + block_count
    next_state = (1, _i64(key64), _i64(next_counter), _i64(next_counter >> 64))
    return tuple(words[:element_count]), next_state


def philox_uniforms(state: Sequence[int], element_count: int) -> tuple[tuple[float, ...],
                                                                         tuple[int, ...]]:
    words, next_state = philox_lanes(state, element_count)
    return tuple((word >> 8) * (2.0 ** -24) for word in words), next_state


def dropout_scales(state: Sequence[int], element_count: int,
                   probability: float) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Build the stateless inverted-dropout mask used by forward and backward."""
    if not math.isfinite(probability) or probability < 0.0 or probability >= 1.0:
        raise ValueError("dropout probability must satisfy 0 <= p < 1")
    if element_count < 0:
        raise ValueError("element count must be non-negative")
    if len(state) != 4 or int(state[0]) != 1:
        raise ValueError("dropout RNG state must be i64[4] with version 1")
    if probability == 0.0:
        return (1.0,) * element_count, tuple(int(word) for word in state)
    uniforms, next_state = philox_uniforms(state, element_count)
    denominator = _f32(_f32(1.0) - _f32(probability))
    scale = _f32(_f32(1.0) / denominator)
    return tuple(scale if uniform >= probability else 0.0 for uniform in uniforms), next_state


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            f"oracle generation requires PyTorch {PINNED_TORCH_VERSION}; torch is unavailable"
        )
    if torch.__version__ != PINNED_TORCH_VERSION:
        raise RuntimeError(
            f"expected torch {PINNED_TORCH_VERSION}, found {torch.__version__}"
        )
    return torch


def _tensor(name: str, role: str, value: Any) -> dict[str, object]:
    value = value.detach()
    if value.device.type != "cpu":
        raise RuntimeError("N2 oracle tensors must be explicitly on CPU")
    value = value.contiguous()
    if value.dtype == torch.float32:
        data = [struct.pack(">f", float(item)).hex() for item in value.reshape(-1)]
        dtype, encoding = "float32", "ieee754-hex-be"
    elif value.dtype == torch.int64:
        data = [struct.pack(">q", int(item)).hex() for item in value.reshape(-1)]
        dtype, encoding = "int64", "twos-complement-hex-be"
    elif value.dtype == torch.bool:
        data = ["1" if bool(item) else "0" for item in value.reshape(-1)]
        dtype, encoding = "bool", "bool01"
    else:
        raise RuntimeError(f"unsupported N2 oracle dtype: {value.dtype}")
    return {
        "data": data, "device": "cpu", "dtype": dtype, "encoding": encoding,
        "layout": "row_major", "name": name, "role": role,
        "shape": list(value.shape),
    }


def _tolerance(value: float) -> dict[str, object]:
    bits = struct.pack(">d", value).hex()
    return {"absolute": bits, "equal_nan": False, "relative": bits}


def _case(name: str, operation: str, kind: str, inputs: list[str],
          outputs: list[str], tolerance: float = 0.0) -> dict[str, object]:
    return {
        "expectation": {"error": None, "outputs": outputs},
        "inputs": inputs, "kind": kind, "name": name, "operation": operation,
        "tolerance": _tolerance(tolerance),
    }


def _append(records: list[dict[str, object]], prefix: str,
            values: Sequence[tuple[str, str, Any]]) -> None:
    for suffix, role, value in values:
        records.append(_tensor(f"{prefix}.{suffix}", role, value))


def _dropout(value: Any, probability: float, state: Sequence[int]) -> tuple[Any, Any,
                                                                            tuple[int, ...]]:
    scales, next_state = dropout_scales(state, value.numel(), probability)
    mask = torch.tensor(scales, dtype=torch.float32, device="cpu").reshape_as(value)
    return value * mask, mask, next_state


def build_payload() -> dict[str, object]:
    framework = _require_torch()
    for counter, key, expected in PHILOX_KNOWN_VECTORS:
        if philox4x32_10(counter, key) != expected:
            raise RuntimeError("independent Philox4x32-10 known-value self-check failed")
    framework.manual_seed(SEED)
    framework.use_deterministic_algorithms(True)
    framework.set_num_threads(1)

    tensors: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []

    ids = framework.tensor([[1, 3, 1], [4, 0, 3]], dtype=framework.int64)
    weight = ((framework.arange(30, dtype=framework.float32) % 11) - 5).mul(0.125)
    weight = weight.reshape(5, 6).requires_grad_()
    eup = ((framework.arange(36, dtype=framework.float32) % 7) - 3).mul(0.2)
    eup = eup.reshape(2, 3, 6)
    embedding = framework.nn.functional.embedding(ids, weight)
    (dweight,) = framework.autograd.grad((embedding * eup).sum(), (weight,))
    _append(tensors, "embedding.repeated", (
        ("input.ids", "input", ids), ("input.weight", "input", weight),
        ("input.upstream", "input", eup), ("output.forward", "expected", embedding),
        ("gradient.dweight", "analytic_gradient", dweight),
    ))
    cases += [
        _case("embedding.repeated.forward", "embedding.forward", "repeated_input",
              ["embedding.repeated.input.ids", "embedding.repeated.input.weight"],
              ["embedding.repeated.output.forward"], 2e-5),
        _case("embedding.repeated.gradient", "embedding.backward", "gradient",
              ["embedding.repeated.input.ids", "embedding.repeated.input.upstream"],
              ["embedding.repeated.gradient.dweight"], 5e-5),
    ]

    lx = ((framework.arange(24, dtype=framework.float32) % 9) - 4).mul(0.15)
    lx = lx.reshape(2, 3, 4).requires_grad_()
    lw = ((framework.arange(20, dtype=framework.float32) % 7) - 3).mul(0.1)
    lw = lw.reshape(5, 4).requires_grad_()
    lb = framework.tensor([-0.3, -0.1, 0.0, 0.2, 0.4], dtype=framework.float32,
                          requires_grad=True)
    lup = ((framework.arange(30, dtype=framework.float32) % 8) - 3).mul(0.125)
    lup = lup.reshape(2, 3, 5)
    ly = framework.matmul(lx, lw.transpose(0, 1)) + lb
    ldx, ldw, ldb = framework.autograd.grad((ly * lup).sum(), (lx, lw, lb))
    _append(tensors, "linear.bias", (
        ("input.x", "input", lx), ("input.weight", "input", lw),
        ("input.bias", "input", lb), ("input.upstream", "input", lup),
        ("output.forward", "expected", ly), ("gradient.dx", "analytic_gradient", ldx),
        ("gradient.dweight", "analytic_gradient", ldw),
        ("gradient.dbias", "analytic_gradient", ldb),
    ))
    cases += [
        _case("linear.bias.forward", "linear.forward", "parity",
              ["linear.bias.input.x", "linear.bias.input.weight", "linear.bias.input.bias"],
              ["linear.bias.output.forward"], 2e-5),
        _case("linear.bias.gradient", "linear.backward", "gradient",
              ["linear.bias.input.x", "linear.bias.input.weight", "linear.bias.input.upstream"],
              ["linear.bias.gradient.dx", "linear.bias.gradient.dweight",
               "linear.bias.gradient.dbias"], 5e-5),
    ]

    nx = lx.detach().clone().requires_grad_()
    nw = lw.detach().clone().requires_grad_()
    ny = framework.matmul(nx, nw.transpose(0, 1))
    ndx, ndw = framework.autograd.grad((ny * lup).sum(), (nx, nw))
    _append(tensors, "linear.no_bias", (
        ("input.x", "input", nx), ("input.weight", "input", nw),
        ("input.upstream", "input", lup), ("output.forward", "expected", ny),
        ("gradient.dx", "analytic_gradient", ndx),
        ("gradient.dweight", "analytic_gradient", ndw),
    ))
    cases += [
        _case("linear.no_bias.forward", "linear.forward-no-bias", "parity",
              ["linear.no_bias.input.x", "linear.no_bias.input.weight"],
              ["linear.no_bias.output.forward"], 2e-5),
        _case("linear.no_bias.gradient", "linear.backward-no-bias", "gradient",
              ["linear.no_bias.input.x", "linear.no_bias.input.weight",
               "linear.no_bias.input.upstream"],
              ["linear.no_bias.gradient.dx", "linear.no_bias.gradient.dweight"], 5e-5),
    ]

    ln_x = framework.tensor(
        [[[-1.0, 0.5, 2.0, -0.25], [3.0, 3.0, 3.0, 3.0]],
         [[0.125, -0.75, 1.25, 2.5], [-2.0, 1.0, 0.25, 0.75]]],
        dtype=framework.float32, requires_grad=True)
    gamma = framework.tensor([0.75, -1.25, 0.5, 1.5], dtype=framework.float32,
                             requires_grad=True)
    beta = framework.tensor([-0.2, 0.1, 0.3, -0.4], dtype=framework.float32,
                            requires_grad=True)
    epsilon = framework.tensor(1e-5, dtype=framework.float32)
    ln_up = ((framework.arange(16, dtype=framework.float32) % 6) - 2).mul(0.175)
    ln_up = ln_up.reshape_as(ln_x)
    mean = ln_x.mean(dim=-1, keepdim=True)
    variance = ((ln_x - mean) ** 2).mean(dim=-1, keepdim=True)
    ln_y = (ln_x - mean) * framework.rsqrt(variance + epsilon) * gamma + beta
    ln_dx, dgamma, dbeta = framework.autograd.grad(
        (ln_y * ln_up).sum(), (ln_x, gamma, beta))
    _append(tensors, "layer_norm.affine", (
        ("input.x", "input", ln_x), ("input.gamma", "input", gamma),
        ("input.beta", "input", beta), ("input.epsilon", "input", epsilon),
        ("input.upstream", "input", ln_up), ("output.forward", "expected", ln_y),
        ("gradient.dx", "analytic_gradient", ln_dx),
        ("gradient.dgamma", "analytic_gradient", dgamma),
        ("gradient.dbeta", "analytic_gradient", dbeta),
    ))
    cases += [
        _case("layer_norm.affine.forward", "layer-norm.forward", "parity",
              ["layer_norm.affine.input.x", "layer_norm.affine.input.gamma",
               "layer_norm.affine.input.beta", "layer_norm.affine.input.epsilon"],
              ["layer_norm.affine.output.forward"], 2e-5),
        _case("layer_norm.affine.gradient", "layer-norm.backward", "gradient",
              ["layer_norm.affine.input.x", "layer_norm.affine.input.gamma",
               "layer_norm.affine.input.epsilon", "layer_norm.affine.input.upstream"],
              ["layer_norm.affine.gradient.dx", "layer_norm.affine.gradient.dgamma",
               "layer_norm.affine.gradient.dbeta"], 5e-5),
    ]

    d1_x = framework.tensor([[[2.0]]], dtype=framework.float32,
                            requires_grad=True)
    d1_gamma = framework.tensor([1.25], dtype=framework.float32, requires_grad=True)
    d1_beta = framework.tensor([-0.375], dtype=framework.float32, requires_grad=True)
    d1_epsilon = framework.tensor(1e-5, dtype=framework.float32)
    d1_up = framework.tensor([[[0.5]]], dtype=framework.float32)
    d1_mean = d1_x.mean(dim=-1, keepdim=True)
    d1_variance = ((d1_x - d1_mean) ** 2).mean(dim=-1, keepdim=True)
    d1_y = (d1_x - d1_mean) * framework.rsqrt(d1_variance + d1_epsilon)
    d1_y = d1_y * d1_gamma + d1_beta
    d1_dx, d1_dgamma, d1_dbeta = framework.autograd.grad(
        (d1_y * d1_up).sum(), (d1_x, d1_gamma, d1_beta))
    _append(tensors, "layer_norm.d1", (
        ("input.x", "input", d1_x), ("input.gamma", "input", d1_gamma),
        ("input.beta", "input", d1_beta),
        ("input.epsilon", "input", d1_epsilon),
        ("input.upstream", "input", d1_up),
        ("output.forward", "expected", d1_y),
        ("gradient.dx", "analytic_gradient", d1_dx),
        ("gradient.dgamma", "analytic_gradient", d1_dgamma),
        ("gradient.dbeta", "analytic_gradient", d1_dbeta),
    ))
    cases += [
        _case("layer_norm.d1.forward", "layer-norm.forward", "boundary",
              ["layer_norm.d1.input.x", "layer_norm.d1.input.gamma",
               "layer_norm.d1.input.beta", "layer_norm.d1.input.epsilon"],
              ["layer_norm.d1.output.forward"], 2e-5),
        _case("layer_norm.d1.gradient", "layer-norm.backward", "gradient",
              ["layer_norm.d1.input.x", "layer_norm.d1.input.gamma",
               "layer_norm.d1.input.epsilon", "layer_norm.d1.input.upstream"],
              ["layer_norm.d1.gradient.dx", "layer_norm.d1.gradient.dgamma",
               "layer_norm.d1.gradient.dbeta"], 5e-5),
    ]

    near_x = framework.tensor(
        [[[1.0, 1.0000038, 0.9999962, 1.0000076],
          [-2.0, -1.9999924, -2.0000038, -1.9999962]]],
        dtype=framework.float32, requires_grad=True)
    near_gamma = framework.tensor([0.75, -0.5, 1.25, 0.25],
                                  dtype=framework.float32, requires_grad=True)
    near_beta = framework.tensor([0.1, -0.2, 0.3, -0.4],
                                 dtype=framework.float32, requires_grad=True)
    near_epsilon = framework.tensor(1e-5, dtype=framework.float32)
    near_up = framework.tensor(
        [[[0.25, -0.75, 1.5, -1.0], [-0.5, 1.25, 0.75, -1.5]]],
        dtype=framework.float32)
    near_mean = near_x.mean(dim=-1, keepdim=True)
    near_variance = ((near_x - near_mean) ** 2).mean(dim=-1, keepdim=True)
    near_y = (near_x - near_mean) * framework.rsqrt(near_variance + near_epsilon)
    near_y = near_y * near_gamma + near_beta
    near_dx, near_dgamma, near_dbeta = framework.autograd.grad(
        (near_y * near_up).sum(), (near_x, near_gamma, near_beta))
    _append(tensors, "layer_norm.near_constant", (
        ("input.x", "input", near_x), ("input.gamma", "input", near_gamma),
        ("input.beta", "input", near_beta),
        ("input.epsilon", "input", near_epsilon),
        ("input.upstream", "input", near_up),
        ("output.forward", "expected", near_y),
        ("gradient.dx", "analytic_gradient", near_dx),
        ("gradient.dgamma", "analytic_gradient", near_dgamma),
        ("gradient.dbeta", "analytic_gradient", near_dbeta),
    ))
    cases += [
        _case("layer_norm.near_constant.forward", "layer-norm.forward", "boundary",
              ["layer_norm.near_constant.input.x",
               "layer_norm.near_constant.input.gamma",
               "layer_norm.near_constant.input.beta",
               "layer_norm.near_constant.input.epsilon"],
              ["layer_norm.near_constant.output.forward"], 2e-5),
        _case("layer_norm.near_constant.gradient", "layer-norm.backward", "gradient",
              ["layer_norm.near_constant.input.x",
               "layer_norm.near_constant.input.gamma",
               "layer_norm.near_constant.input.epsilon",
               "layer_norm.near_constant.input.upstream"],
              ["layer_norm.near_constant.gradient.dx",
               "layer_norm.near_constant.gradient.dgamma",
               "layer_norm.near_constant.gradient.dbeta"], 5e-5),
    ]

    activation_x = framework.tensor(
        [[[-20.0, -3.0, -1.0, -0.0, 0.0, 0.5, 2.0, 20.0]]],
        dtype=framework.float32,
        requires_grad=True)
    activation_up = framework.tensor(
        [[[0.125, 0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -2.0]]],
        dtype=framework.float32)
    gelu = 0.5 * activation_x * (
        1.0 + framework.erf(activation_x / math.sqrt(2.0)))
    (gelu_dx,) = framework.autograd.grad((gelu * activation_up).sum(),
                                         (activation_x,), retain_graph=True)
    # The frozen N2 convention canonicalizes either signed zero to positive zero.
    relu = framework.where(activation_x > 0, activation_x,
                           framework.zeros_like(activation_x))
    (relu_dx,) = framework.autograd.grad((relu * activation_up).sum(), (activation_x,))
    _append(tensors, "activation", (
        ("input.x", "input", activation_x),
        ("input.upstream", "input", activation_up),
        ("gelu.output.forward", "expected", gelu),
        ("gelu.gradient.dx", "analytic_gradient", gelu_dx),
        ("relu.output.forward", "expected", relu),
        ("relu.gradient.dx", "analytic_gradient", relu_dx),
    ))
    cases += [
        _case("gelu.exact.forward", "gelu.forward", "parity",
              ["activation.input.x"], ["activation.gelu.output.forward"], 2e-5),
        _case("gelu.exact.gradient", "gelu.backward", "gradient",
              ["activation.input.x", "activation.input.upstream"],
              ["activation.gelu.gradient.dx"], 5e-5),
        _case("relu.zero.forward", "relu.forward", "special_value",
              ["activation.input.x"], ["activation.relu.output.forward"], 0.0),
        _case("relu.zero.gradient", "relu.backward", "gradient",
              ["activation.input.x", "activation.input.upstream"],
              ["activation.relu.gradient.dx"], 0.0),
    ]

    residual_x = ((framework.arange(12, dtype=framework.float32) % 7) - 3).mul(0.25)
    residual_x = residual_x.reshape(1, 3, 4).requires_grad_()
    branch = ((framework.arange(12, dtype=framework.float32) % 5) - 1).mul(0.125)
    branch = branch.reshape_as(residual_x).requires_grad_()
    residual_up = ((framework.arange(12, dtype=framework.float32) % 4) + 1).mul(0.1)
    residual_up = residual_up.reshape_as(residual_x)
    residual = residual_x + branch
    rdx, rdbranch = framework.autograd.grad((residual * residual_up).sum(),
                                            (residual_x, branch))
    repeated_x = residual_x.detach().clone().requires_grad_()
    repeated_branch = residual_x.detach().clone().requires_grad_()
    repeated = repeated_x + repeated_branch
    repeated_dx, repeated_dbranch = framework.autograd.grad(
        (repeated * residual_up).sum(), (repeated_x, repeated_branch))
    _append(tensors, "residual.distinct", (
        ("input.x", "input", residual_x), ("input.branch", "input", branch),
        ("input.upstream", "input", residual_up),
        ("output.forward", "expected", residual),
        ("gradient.dx", "analytic_gradient", rdx),
        ("gradient.dbranch", "analytic_gradient", rdbranch),
    ))
    _append(tensors, "residual.repeated", (
        ("input.x", "input", repeated_x),
        ("input.branch", "input", repeated_branch),
        ("input.upstream", "input", residual_up),
        ("output.forward", "expected", repeated),
        ("gradient.dx", "analytic_gradient", repeated_dx),
        ("gradient.dbranch", "analytic_gradient", repeated_dbranch),
    ))
    cases += [
        _case("residual.distinct.forward", "residual.forward", "parity",
              ["residual.distinct.input.x", "residual.distinct.input.branch"],
              ["residual.distinct.output.forward"], 0.0),
        _case("residual.distinct.gradient", "residual.backward", "gradient",
              ["residual.distinct.input.upstream"],
              ["residual.distinct.gradient.dx", "residual.distinct.gradient.dbranch"], 0.0),
        _case("residual.repeated.forward", "residual.forward", "repeated_input",
              ["residual.repeated.input.x", "residual.repeated.input.branch"],
              ["residual.repeated.output.forward"], 0.0),
        _case("residual.repeated.gradient", "residual.backward", "gradient",
              ["residual.repeated.input.upstream"],
              ["residual.repeated.gradient.dx",
               "residual.repeated.gradient.dbranch"], 0.0),
    ]

    dropout_state = (1, _i64(0x299F31D0A4093822), _i64(0x85A308D3243F6A88),
                     _i64(0x0370734413198A2E))
    dropout_x = ((framework.arange(10, dtype=framework.float32) % 7) - 3).mul(0.2)
    dropout_x = dropout_x.reshape(1, 2, 5).requires_grad_()
    dropout_up = ((framework.arange(10, dtype=framework.float32) % 4) + 1).mul(0.125)
    dropout_up = dropout_up.reshape_as(dropout_x)
    state_tensor = framework.tensor(dropout_state, dtype=framework.int64)
    for label, probability in (("p25", 0.25), ("p0", 0.0)):
        local_x = dropout_x.detach().clone().requires_grad_()
        output, _mask, next_state = _dropout(local_x, probability, dropout_state)
        (dropout_dx,) = framework.autograd.grad((output * dropout_up).sum(), (local_x,))
        p_tensor = framework.tensor(probability, dtype=framework.float32)
        next_tensor = framework.tensor(next_state, dtype=framework.int64)
        prefix = f"dropout.train.{label}"
        _append(tensors, prefix, (
            ("input.x", "input", local_x), ("input.probability", "input", p_tensor),
            ("input.state", "input", state_tensor),
            ("input.upstream", "input", dropout_up),
            ("output.forward", "expected", output),
            ("output.next_state", "expected", next_tensor),
            ("gradient.dx", "analytic_gradient", dropout_dx),
        ))
        cases += [
            _case(f"dropout.train.{label}.forward", "dropout.train.forward", "known_value",
                  [f"{prefix}.input.x", f"{prefix}.input.probability",
                   f"{prefix}.input.state"],
                  [f"{prefix}.output.forward", f"{prefix}.output.next_state"], 0.0),
            _case(f"dropout.train.{label}.gradient", "dropout.train.backward", "gradient",
                  [f"{prefix}.input.upstream", f"{prefix}.input.probability",
                   f"{prefix}.input.state"], [f"{prefix}.gradient.dx"], 0.0),
        ]
    eval_x = dropout_x.detach().clone().requires_grad_()
    eval_y = eval_x.clone()
    (eval_dx,) = framework.autograd.grad((eval_y * dropout_up).sum(), (eval_x,))
    _append(tensors, "dropout.eval", (
        ("input.x", "input", eval_x), ("input.upstream", "input", dropout_up),
        ("output.forward", "expected", eval_y),
        ("gradient.dx", "analytic_gradient", eval_dx),
    ))
    cases += [
        _case("dropout.eval.forward", "dropout.eval.forward", "parity",
              ["dropout.eval.input.x"], ["dropout.eval.output.forward"], 0.0),
        _case("dropout.eval.gradient", "dropout.eval.backward", "gradient",
              ["dropout.eval.input.upstream"], ["dropout.eval.gradient.dx"], 0.0),
    ]

    source = Path(__file__).resolve()
    lock = source.parent.parent / "n2" / "requirements-reference.lock"
    tensors.sort(key=lambda item: str(item["name"]))
    cases.sort(key=lambda item: str(item["name"]))
    return {
        "cases": cases,
        "generator": {
            "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "framework": {"name": "pytorch", "version": PINNED_TORCH_VERSION},
            "name": GENERATOR_NAME, "seed": SEED,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "version": GENERATOR_VERSION,
        },
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(encode_fixture(build_payload()))


if __name__ == "__main__":
    main()
