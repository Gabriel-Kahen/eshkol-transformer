"""Generate the development-only L2 PyTorch oracle fixture."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import torch
import torch.nn.functional as functional

from tests.q0.oracle_format import encode_fixture

PINNED_TORCH_VERSION = "2.13.0+cpu"
SEED = 2442


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f32(name: str, role: str, shape: tuple[int, ...], values: list[float]) -> dict[str, object]:
    return {
        "data": [struct.pack(">f", value).hex() for value in values],
        "device": "cpu",
        "dtype": "float32",
        "encoding": "ieee754-hex-be",
        "layout": "row_major",
        "name": name,
        "role": role,
        "shape": list(shape),
    }


def _i64(name: str, role: str, shape: tuple[int, ...], values: list[int]) -> dict[str, object]:
    return {
        "data": [value.to_bytes(8, "big", signed=True).hex() for value in values],
        "device": "cpu",
        "dtype": "int64",
        "encoding": "twos-complement-hex-be",
        "layout": "row_major",
        "name": name,
        "role": role,
        "shape": list(shape),
    }


def build_payload() -> dict[str, object]:
    if torch.__version__ != PINNED_TORCH_VERSION:
        raise RuntimeError(f"expected torch {PINNED_TORCH_VERSION}, found {torch.__version__}")
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    logits = torch.tensor(
        [[[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device="cpu",
        requires_grad=True,
    )
    targets = torch.tensor([[2, 0]], dtype=torch.int64, device="cpu")
    upstream = torch.tensor([[0.7, -1.25]], dtype=torch.float32, device="cpu")
    losses = functional.cross_entropy(
        logits.reshape(-1, 3), targets.reshape(-1), reduction="none"
    ).reshape(1, 2)
    (losses * upstream).sum().backward()
    assert logits.grad is not None
    source = Path(__file__).resolve()
    lock = source.with_name("requirements-oracle.lock")
    tensors = [
        _f32("input.logits", "input", (1, 2, 3), logits.detach().reshape(-1).tolist()),
        _i64("input.targets", "input", (1, 2), targets.reshape(-1).tolist()),
        _f32("input.upstream", "input", (1, 2), upstream.reshape(-1).tolist()),
        _f32("output.gradient", "analytic_gradient", (1, 2, 3), logits.grad.reshape(-1).tolist()),
        _f32("output.loss", "expected", (1, 2), losses.detach().reshape(-1).tolist()),
    ]
    tensors.sort(key=lambda tensor: str(tensor["name"]))
    return {
        "cases": [
            {
                "expectation": {"error": None, "outputs": ["output.gradient"]},
                "inputs": ["input.logits", "input.targets", "input.upstream"],
                "kind": "gradient",
                "name": "indexed_cross_entropy_backward",
                "operation": "indexed-cross-entropy.backward",
                "tolerance": {
                    "absolute": "3ec92a737110e454",
                    "equal_nan": False,
                    "relative": "3eff75104d551d69",
                },
            },
            {
                "expectation": {"error": None, "outputs": ["output.loss"]},
                "inputs": ["input.logits", "input.targets"],
                "kind": "parity",
                "name": "indexed_cross_entropy_forward",
                "operation": "indexed-cross-entropy.forward",
                "tolerance": {
                    "absolute": "3ec0c6f7a0b5ed8d",
                    "equal_nan": False,
                    "relative": "3ef4f8b588e368f1",
                },
            },
        ],
        "generator": {
            "dependency_lock_sha256": _sha256(lock),
            "framework": {"name": "pytorch", "version": PINNED_TORCH_VERSION},
            "name": "q0.pytorch.indexed_cross_entropy",
            "seed": SEED,
            "source_sha256": _sha256(source),
            "version": 1,
        },
        "tensors": tensors,
    }


def main() -> None:
    output = Path(__import__("sys").argv[1])
    output.write_bytes(encode_fixture(build_payload()))


if __name__ == "__main__":
    main()
