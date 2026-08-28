#!/usr/bin/env python3
"""Generate the smallest deterministic PyTorch oracle fixture."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import torch

from oracle_format import encode_fixture

PINNED_TORCH_VERSION = "2.13.0+cpu"
GENERATOR_NAME = "q0.pytorch.scalar_add"
GENERATOR_VERSION = 1
SEED = 1729


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_record(name: str, role: str, tensor: torch.Tensor) -> dict[str, object]:
    if tensor.device.type != "cpu":
        raise RuntimeError("oracle generation requires an explicit CPU tensor")
    if tensor.dtype != torch.float64:
        raise RuntimeError("scalar-add fixture requires torch.float64")
    encoded = [
        struct.pack(">d", float(value)).hex()
        for value in tensor.reshape(-1).tolist()
    ]
    return {
        "data": encoded,
        "device": "cpu",
        "dtype": "float64",
        "encoding": "ieee754-hex-be",
        "layout": "row_major",
        "name": name,
        "role": role,
        "shape": list(tensor.shape),
    }


def build_payload() -> dict[str, object]:
    runtime_version = torch.__version__
    if runtime_version != PINNED_TORCH_VERSION:
        raise RuntimeError(
            f"expected torch {PINNED_TORCH_VERSION}, found {torch.__version__}"
        )
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    lhs = torch.tensor(1.25, dtype=torch.float64, device="cpu")
    rhs = torch.tensor(2.5, dtype=torch.float64, device="cpu")
    result = torch.add(lhs, rhs)

    source = Path(__file__).resolve()
    lock = source.with_name("requirements-oracle.lock")
    tensors = [
        _tensor_record("input.lhs", "input", lhs),
        _tensor_record("input.rhs", "input", rhs),
        _tensor_record("output.sum", "expected", result),
    ]
    tensors.sort(key=lambda tensor: str(tensor["name"]))
    return {
        "cases": [
            {
                "expectation": {"error": None, "outputs": ["output.sum"]},
                "inputs": ["input.lhs", "input.rhs"],
                "kind": "parity",
                "name": "scalar_add_f64",
                "operation": "numeric.add.scalar",
                "tolerance": {
                    "absolute": "0000000000000000",
                    "equal_nan": False,
                    "relative": "0000000000000000",
                },
            }
        ],
        "generator": {
            "dependency_lock_sha256": _sha256(lock),
            "framework": {"name": "pytorch", "version": PINNED_TORCH_VERSION},
            "name": GENERATOR_NAME,
            "seed": SEED,
            "source_sha256": _sha256(source),
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
