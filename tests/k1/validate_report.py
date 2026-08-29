from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED_NAMES = [
    "autodiff.reverse",
    "kernel.activation",
    "kernel.causal-attention",
    "kernel.embedding-backward",
    "kernel.indexed-cross-entropy",
    "kernel.matmul",
    "kernel.norm",
    "tensor.bool",
    "tensor.contiguous",
    "tensor.f32",
    "tensor.i64",
]


def reject_float(_: str) -> None:
    raise ValueError("JSON floating-point numbers are forbidden")


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def canonical_string(value: str) -> str:
    pieces = ['"']
    for character in value:
        codepoint = ord(character)
        if character in {'"', "\\"}:
            pieces.append("\\" + character)
        elif codepoint < 0x20:
            pieces.append(f"\\u{codepoint:04x}")
        else:
            pieces.append(character)
    pieces.append('"')
    return "".join(pieces)


def canonical_json(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if value < 0:
            raise ValueError("negative JSON integer")
        return str(value)
    if isinstance(value, str):
        return canonical_string(value)
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(
            canonical_string(key) + ":" + canonical_json(value[key])
            for key in sorted(value)
        ) + "}"
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def main(path: Path) -> None:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("report must end in exactly one LF")
    text = raw.decode("utf-8")
    value = json.loads(
        text,
        parse_float=reject_float,
        parse_constant=reject_float,
        object_pairs_hook=strict_object,
    )
    canonical = (canonical_json(value) + "\n").encode("utf-8")
    if canonical != raw:
        raise ValueError("report is not canonical JSON")
    if set(value) != {
        "abi",
        "entries",
        "format",
        "process_local",
        "provider_abi",
        "version",
    }:
        raise ValueError("unexpected root fields")
    if value["format"] != "eshkol-kernel-capabilities" or value["version"] != 1:
        raise ValueError("unexpected report format/version")
    if value["abi"] != {"major": 1, "minor": 0}:
        raise ValueError("unexpected ABI version")
    if value["process_local"] is not True or value["provider_abi"] is not None:
        raise ValueError("baseline report must be process-local without a provider")
    entries = value["entries"]
    if not isinstance(entries, list) or [entry["name"] for entry in entries] != EXPECTED_NAMES:
        raise ValueError("baseline entries are missing or unsorted")
    for entry in entries:
        if entry["status"] != "unverified" or entry["deterministic"] is not False:
            raise ValueError("baseline entries must remain unverified")
        if entry["constraints"] != {
            "devices": [],
            "dtypes": [],
            "operations": [],
            "shape_ranges": [],
        }:
            raise ValueError("unverified baseline cannot advertise constraints")
    print("K1 report schema: PASS (strict parse and canonical round trip)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_report.py REPORT")
    main(Path(sys.argv[1]))
