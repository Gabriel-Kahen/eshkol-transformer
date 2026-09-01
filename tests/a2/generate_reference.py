#!/usr/bin/env python3
"""Generate deterministic PyTorch CPU references for A2."""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r"Failed to initialize NumPy.*")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.q0.oracle_format import encode_fixture

PINNED_TORCH_VERSION = "2.13.0+cpu"
SEED = 1729
MAX_POSITION = 16_777_215


def tensor(name: str, role: str, value: torch.Tensor) -> dict[str, object]:
    value = value.detach().contiguous().cpu()
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
        raise RuntimeError(f"unsupported oracle dtype: {value.dtype}")
    return {
        "data": data, "device": "cpu", "dtype": dtype, "encoding": encoding,
        "layout": "row_major", "name": name, "role": role,
        "shape": list(value.shape),
    }


def case(name: str, operation: str, kind: str, inputs: list[str],
         outputs: list[str], tolerance: float) -> dict[str, object]:
    encoded = struct.pack(">d", tolerance).hex()
    return {
        "expectation": {"error": None, "outputs": outputs},
        "inputs": inputs, "kind": kind, "name": name, "operation": operation,
        "tolerance": {"absolute": encoded, "equal_nan": False,
                      "relative": encoded},
    }


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              qpos: torch.Tensor, kpos: torch.Tensor,
              keep: torch.Tensor) -> torch.Tensor:
    group = q.shape[1] // k.shape[1]
    expanded_k = k.repeat_interleave(group, dim=1)
    expanded_v = v.repeat_interleave(group, dim=1)
    scores = torch.matmul(q, expanded_k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    causal = kpos[:, None, None, :] <= qpos[:, None, :, None]
    admitted = keep[:, None, :, :] & causal
    any_key = admitted.any(dim=-1, keepdim=True)
    masked = scores.masked_fill(~admitted, -torch.inf)
    safe = torch.where(any_key, masked, torch.zeros_like(masked))
    probability = torch.softmax(safe, dim=-1)
    probability = torch.where(any_key, probability, torch.zeros_like(probability))
    return torch.matmul(probability, expanded_v)


def rope(x: torch.Tensor, positions: torch.Tensor,
         inv_freq: torch.Tensor) -> torch.Tensor:
    angles = positions.to(torch.float32)[..., None] * inv_freq
    cosine = torch.cos(angles)[:, None, :, :]
    sine = torch.sin(angles)[:, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine,
                        even * sine + odd * cosine), dim=-1).flatten(-2)


def build_payload() -> dict[str, object]:
    if torch.__version__ != PINNED_TORCH_VERSION:
        raise RuntimeError(
            f"expected torch {PINNED_TORCH_VERSION}, found {torch.__version__}"
        )
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    records: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []

    q = (torch.arange(24, dtype=torch.float32).remainder(7) - 3).mul(0.125)
    q = q.reshape(1, 4, 3, 2).requires_grad_()
    k = (torch.arange(12, dtype=torch.float32).remainder(5) - 2).mul(0.2)
    k = k.reshape(1, 2, 3, 2).requires_grad_()
    v = torch.arange(1, 13, dtype=torch.float32).mul(0.1)
    v = v.reshape(1, 2, 3, 2).requires_grad_()
    upstream = (torch.arange(24, dtype=torch.float32).remainder(4) + 1)
    upstream = upstream.mul(0.05).reshape_as(q)
    qpos = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    kpos = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    keep = torch.tensor([[[True, False, False], [True, True, False],
                          [False, False, False]]])
    output = attention(q, k, v, qpos, kpos, keep)
    dq, dk, dv = torch.autograd.grad((output * upstream).sum(), (q, k, v))
    for name, role, value in (
        ("gqa.input.k", "input", k),
        ("gqa.input.kpos", "input", kpos),
        ("gqa.input.mask", "input", keep),
        ("gqa.input.q", "input", q),
        ("gqa.input.qpos", "input", qpos),
        ("gqa.input.upstream", "input", upstream),
        ("gqa.input.v", "input", v),
        ("gqa.output.forward", "expected", output),
        ("gqa.gradient.dk", "analytic_gradient", dk),
        ("gqa.gradient.dq", "analytic_gradient", dq),
        ("gqa.gradient.dv", "analytic_gradient", dv),
    ):
        records.append(tensor(name, role, value))
    common = ["gqa.input.k", "gqa.input.kpos", "gqa.input.mask", "gqa.input.q",
              "gqa.input.qpos", "gqa.input.v"]
    cases.append(case("attention.gqa.forward", "causal-attention.forward", "parity",
                      common, ["gqa.output.forward"], 2.0e-5))
    cases.append(case("attention.gqa.gradient", "causal-attention.backward", "gradient",
                      common + ["gqa.input.upstream"],
                      ["gqa.gradient.dk", "gqa.gradient.dq", "gqa.gradient.dv"],
                      5.0e-5))

    rx = torch.tensor([[[[1.0, 2.0, -3.0, 4.0],
                         [0.5, -0.25, 2.0, -1.0]]]], requires_grad=True)
    rpos = torch.tensor([[0, MAX_POSITION]], dtype=torch.int64)
    rinv = torch.tensor([1.0, 0.01], dtype=torch.float32)
    rupstream = torch.tensor([[[[-0.5, 0.25, 1.5, -2.0],
                                [0.75, 1.25, -0.5, 0.125]]]])
    ry = rope(rx, rpos, rinv)
    (rdx,) = torch.autograd.grad((ry * rupstream).sum(), (rx,))
    for name, role, value in (
        ("rope.input.inv", "input", rinv),
        ("rope.input.positions", "input", rpos),
        ("rope.input.upstream", "input", rupstream),
        ("rope.input.x", "input", rx),
        ("rope.output.forward", "expected", ry),
        ("rope.gradient.dx", "analytic_gradient", rdx),
    ):
        records.append(tensor(name, role, value))
    cases.append(case("rope.boundary.forward", "rope.forward", "boundary",
                      ["rope.input.x", "rope.input.positions", "rope.input.inv"],
                      ["rope.output.forward"], 5.0e-5))
    cases.append(case("rope.boundary.gradient", "rope.backward", "gradient",
                      ["rope.input.upstream", "rope.input.positions", "rope.input.inv"],
                      ["rope.gradient.dx"], 5.0e-5))

    source = Path(__file__).resolve()
    lock = source.with_name("requirements-reference.lock")
    records.sort(key=lambda item: str(item["name"]))
    cases.sort(key=lambda item: str(item["name"]))
    return {
        "cases": cases,
        "generator": {
            "dependency_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "framework": {"name": "pytorch", "version": PINNED_TORCH_VERSION},
            "name": "a2.pytorch.attention-rope", "seed": SEED,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "version": 1,
        },
        "tensors": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(encode_fixture(build_payload()))


if __name__ == "__main__":
    main()
