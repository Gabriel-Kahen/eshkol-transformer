#!/usr/bin/env python3
"""Generate the deterministic stdlib-only D2 shifted-batch fixture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tests.q0.oracle_format import encode_fixture


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _i64(name: str, role: str, shape: tuple[int, ...], values: tuple[int, ...]) -> dict[str, object]:
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


def _bool(name: str, role: str, shape: tuple[int, ...], values: tuple[bool, ...]) -> dict[str, object]:
    return {
        "data": ["1" if value else "0" for value in values],
        "device": "cpu",
        "dtype": "bool",
        "encoding": "bool01",
        "layout": "row_major",
        "name": name,
        "role": role,
        "shape": list(shape),
    }


def build_payload() -> dict[str, object]:
    source = Path(__file__).resolve()
    lock = source.with_name("requirements-oracle.lock")
    tensors = [
        _i64("input.shard0", "input", (3,), (10, 11, 12)),
        _i64("input.shard1", "input", (3,), (13, 14, 15)),
        _i64("output.inputs", "expected", (2, 4), (10, 11, 12, 13, 14, 0, 0, 0)),
        _bool("output.loss_mask", "expected", (2, 4), (True, True, True, True, True, False, False, False)),
        _i64("output.targets", "expected", (2, 4), (11, 12, 13, 14, 15, 0, 0, 0)),
    ]
    return {
        "cases": [
            {
                "expectation": {
                    "error": None,
                    "outputs": ["output.inputs", "output.loss_mask", "output.targets"],
                },
                "inputs": ["input.shard0", "input.shard1"],
                "kind": "parity",
                "name": "packed_cross_shard_shift",
                "operation": "token-batch.shifted-packed",
                "tolerance": {
                    "absolute": "0000000000000000",
                    "equal_nan": False,
                    "relative": "0000000000000000",
                },
            }
        ],
        "generator": {
            "dependency_lock_sha256": _sha256(lock),
            "framework": {"name": "python-stdlib", "version": "3"},
            "name": "d2.python-stdlib.shifted-batch",
            "seed": 0,
            "source_sha256": _sha256(source),
            "version": 1,
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
