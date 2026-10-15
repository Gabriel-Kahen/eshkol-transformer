#!/usr/bin/env python3
"""Generate the exact N5 test-only >=8192 E3 D2 preflight comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "internal/e3/lib/e3_d2_restore.esk"
SOURCE_SHA = "5f26e48410a9567f612fb5bab1fe982b6e95642d149d36f0f89278cc91d27217"
DERIVED_SHA = "10c7b775b3f31aa2125460c1b48ddd1e373171b5ae34f13e2f66d3ee1d5b825f"
OLD = b"(> remaining 8192)"
NEW = b"(>= remaining 8192)"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive(source: bytes) -> bytes:
    if digest(source) != SOURCE_SHA or source.count(OLD) != 1:
        raise ValueError("E3 D2 preflight source differs from frozen N5 parent")
    derived = source.replace(OLD, NEW)
    if digest(derived) != DERIVED_SHA:
        raise ValueError("N5 comparator mutant changed unexpectedly")
    return derived


def run(output: Path, *, check: bool) -> dict[str, str]:
    if output.name != "e3_d2_restore.esk" or not output.is_absolute():
        raise ValueError("N5 overlay needs an absolute e3_d2_restore.esk path")
    derived = derive(SOURCE.read_bytes())
    receipt = {
        "kind": "e3-n5-cap-comparator-mutant-v1",
        "source_sha256": SOURCE_SHA,
        "derived_sha256": DERIVED_SHA,
        "replacement": "> remaining 8192 => >= remaining 8192",
    }
    receipt_path = output.with_name("e3_n5_cap_mutant_receipt.json")
    encoded = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if check:
        if output.read_bytes() != derived or receipt_path.read_bytes() != encoded:
            raise ValueError("N5 generated overlay drifted")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(derived)
        receipt_path.write_bytes(encoded)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, check=args.check), sort_keys=True))


if __name__ == "__main__":
    main()
