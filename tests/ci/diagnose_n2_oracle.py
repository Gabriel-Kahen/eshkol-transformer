#!/usr/bin/env python3
"""Produce bounded evidence for an exact N2 oracle regeneration mismatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.q0.oracle_format import decode_tensor, encode_fixture, load_fixture

BYTE_DIFF_LIMIT = 128
METADATA_DIFF_LIMIT = 128
TENSOR_DIFF_LIMIT = 256
FROZEN_FIXTURE = Path(__file__).parents[1] / "q0" / "fixtures" / "n2_primitives_v1.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_diffs(left: Any, right: Any, path: str = "$") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": path, "frozen": left, "generated": right}]
    if isinstance(left, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left:
                result.append({"path": child, "frozen": None, "generated": right[key]})
            elif key not in right:
                result.append({"path": child, "frozen": left[key], "generated": None})
            else:
                result.extend(_json_diffs(left[key], right[key], child))
        return result
    if isinstance(left, list):
        result = []
        for index in range(max(len(left), len(right))):
            child = f"{path}[{index}]"
            if index >= len(left):
                result.append({"path": child, "frozen": None, "generated": right[index]})
            elif index >= len(right):
                result.append({"path": child, "frozen": left[index], "generated": None})
            else:
                result.extend(_json_diffs(left[index], right[index], child))
        return result
    return [] if left == right else [{"path": path, "frozen": left, "generated": right}]


def _value(value: Any) -> dict[str, Any]:
    if isinstance(value, float):
        if math.isnan(value):
            return {"repr": "nan", "hex": value.hex()}
        if math.isinf(value):
            return {"repr": "inf" if value > 0 else "-inf", "hex": value.hex()}
        return {"repr": repr(value), "hex": value.hex()}
    return {"repr": repr(value)}


def compare_fixtures(frozen: bytes, generated: bytes) -> dict[str, Any]:
    frozen_payload = load_fixture(frozen)
    generated_payload = load_fixture(generated)
    byte_offsets = [
        index for index in range(max(len(frozen), len(generated)))
        if (frozen[index] if index < len(frozen) else None)
        != (generated[index] if index < len(generated) else None)
    ]
    frozen_tensors = {item["name"]: item for item in frozen_payload["tensors"]}
    generated_tensors = {item["name"]: item for item in generated_payload["tensors"]}
    tensor_metadata_diffs: list[dict[str, Any]] = []
    word_diffs: list[dict[str, Any]] = []
    word_difference_count = 0
    for name in sorted(set(frozen_tensors) | set(generated_tensors)):
        if name not in frozen_tensors or name not in generated_tensors:
            tensor_metadata_diffs.append({
                "tensor": name, "frozen_present": name in frozen_tensors,
                "generated_present": name in generated_tensors,
            })
            continue
        old = frozen_tensors[name]
        new = generated_tensors[name]
        tensor_metadata_diffs.extend(_json_diffs(
            {key: value for key, value in old.items() if key != "data"},
            {key: value for key, value in new.items() if key != "data"},
            f"$.tensors[{name!r}]",
        ))
        old_values = decode_tensor(old)
        new_values = decode_tensor(new)
        for index in range(max(len(old["data"]), len(new["data"]))):
            old_word = old["data"][index] if index < len(old["data"]) else None
            new_word = new["data"][index] if index < len(new["data"]) else None
            if old_word == new_word:
                continue
            word_difference_count += 1
            if len(word_diffs) >= TENSOR_DIFF_LIMIT:
                continue
            old_value = old_values[index] if index < len(old_values) else None
            new_value = new_values[index] if index < len(new_values) else None
            entry: dict[str, Any] = {
                "tensor": name, "flat_index": index,
                "frozen_word": old_word, "generated_word": new_word,
                "frozen_value": None if old_value is None else _value(old_value),
                "generated_value": None if new_value is None else _value(new_value),
            }
            if (isinstance(old_value, (int, float)) and not isinstance(old_value, bool)
                    and isinstance(new_value, (int, float)) and not isinstance(new_value, bool)):
                entry["generated_minus_frozen"] = _value(new_value - old_value)
            word_diffs.append(entry)
    payload_metadata_diffs = _json_diffs(
        {key: value for key, value in frozen_payload.items() if key != "tensors"},
        {key: value for key, value in generated_payload.items() if key != "tensors"},
    )
    return {
        "schema": "N2_ORACLE_DIAGNOSTIC_V1", "exact_match": frozen == generated,
        "frozen": {"sha256": _sha256(frozen), "length": len(frozen)},
        "generated": {"sha256": _sha256(generated), "length": len(generated)},
        "byte_differences": {
            "count": len(byte_offsets),
            "first": [{
                "offset": offset,
                "frozen": frozen[offset] if offset < len(frozen) else None,
                "generated": generated[offset] if offset < len(generated) else None,
            } for offset in byte_offsets[:BYTE_DIFF_LIMIT]],
            "truncated": len(byte_offsets) > BYTE_DIFF_LIMIT,
        },
        "payload_metadata_differences": {
            "count": len(payload_metadata_diffs), "first": payload_metadata_diffs[:METADATA_DIFF_LIMIT],
            "truncated": len(payload_metadata_diffs) > METADATA_DIFF_LIMIT,
        },
        "tensor_metadata_differences": {
            "count": len(tensor_metadata_diffs), "first": tensor_metadata_diffs[:METADATA_DIFF_LIMIT],
            "truncated": len(tensor_metadata_diffs) > METADATA_DIFF_LIMIT,
        },
        "tensor_word_differences": {
            "count": word_difference_count, "first": word_diffs,
            "truncated": word_difference_count > TENSOR_DIFF_LIMIT,
        },
    }


def _cpu_info(path: Path = Path("/proc/cpuinfo")) -> dict[str, Any]:
    allowed = {"vendor_id", "cpu family", "model", "model name", "stepping", "flags"}
    result: dict[str, Any] = {}
    try:
        first = path.read_text(encoding="utf-8", errors="replace").split("\n\n", 1)[0]
    except OSError as error:
        return {"unavailable": type(error).__name__}
    for line in first.splitlines():
        key, separator, value = line.partition(":")
        key = key.strip()
        if separator and key in allowed:
            result[key.replace(" ", "_")] = value.strip().split() if key == "flags" else value.strip()
    return result


def host_report(torch_module: Any) -> dict[str, Any]:
    cpu_backend = getattr(getattr(torch_module, "backends", None), "cpu", None)
    mkldnn = getattr(getattr(torch_module, "backends", None), "mkldnn", None)
    return {
        "schema": "N2_ORACLE_HOST_V1",
        "python": {"version": sys.version, "implementation": platform.python_implementation()},
        "platform": {"platform": platform.platform(), "system": platform.system(),
                     "release": platform.release(), "machine": platform.machine(),
                     "libc": list(platform.libc_ver())},
        "cpu": _cpu_info(),
        "relevant_environment": {name: os.environ.get(name) for name in (
            "ATEN_CPU_CAPABILITY", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_CBWR", "MKL_ENABLE_INSTRUCTIONS")},
        "torch": {
            "version": torch_module.__version__,
            "git_version": getattr(torch_module.version, "git_version", None),
            "debug": getattr(torch_module.version, "debug", None),
            "config": torch_module.__config__.show(),
            "num_threads": torch_module.get_num_threads(),
            "num_interop_threads": torch_module.get_num_interop_threads(),
            "deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
            "cpu_capability": cpu_backend.get_cpu_capability() if cpu_backend else None,
            "mkldnn_available": mkldnn.is_available() if mkldnn else None,
            "mkldnn_enabled": mkldnn.enabled if mkldnn else None,
        },
    }


def _torch_execution_state(torch_module: Any) -> dict[str, Any]:
    cpu_backend = getattr(getattr(torch_module, "backends", None), "cpu", None)
    return {
        "num_threads": torch_module.get_num_threads(),
        "num_interop_threads": torch_module.get_num_interop_threads(),
        "deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
        "cpu_capability": cpu_backend.get_cpu_capability() if cpu_backend else None,
    }


def run(output_dir: Path, frozen_path: Path = FROZEN_FIXTURE) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    from tests.q0 import generate_n2_primitives as generator

    host = host_report(generator.torch)
    (output_dir / "host-report.json").write_text(
        json.dumps(host, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generated = encode_fixture(generator.build_payload())
    host["torch_after_generation"] = _torch_execution_state(generator.torch)
    (output_dir / "host-report.json").write_text(
        json.dumps(host, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "generated-n2-primitives-v1.json").write_bytes(generated)
    comparison = compare_fixtures(frozen_path.read_bytes(), generated)
    (output_dir / "n2-oracle-diagnostic.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"exact_match": comparison["exact_match"], "output_dir": str(output_dir)}))
    return 0 if comparison["exact_match"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    return run(parser.parse_args().output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
