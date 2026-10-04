#!/usr/bin/env python3
"""Exact test-only E3 finalizer mutant; never edits the production source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/e3_evaluation_metrics_provider.c"
SOURCE_SHA = "b2e6cca9acf9f39db8021ddd78be15f847fd26b39d358260928095f48635851c"
FUNCTION_SHA = "6186305f57904cc47d9862a8d733c7b0882eaef0b1ba38643865b67465c322a7"
WITH_NEWLINE_SHA = "555fd8a836817e709feb84c71f8abec7f0db2481497dcc2341774f93ec2f7bf6"
DERIVED_SHA = "47f151b382a768b6c2371a0a94f94571d301d230e24d6e25ba5e7887ec7052bb"
START = b"static int32_t evaluate_finalize("
END = b"\nstatic int32_t evaluate("
OLD = (b"  const float perplexity = expf(loss);\n"
       b"  if (!isfinite(perplexity)) return domain(e,op);\n")
NEW = (b"  float perplexity = expf(loss);\n"
       b"  if (!isfinite(perplexity)) perplexity = FLT_MAX;\n")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def derive(source: bytes) -> tuple[bytes, dict[str, object]]:
    if digest(source) != SOURCE_SHA or source.count(START) != 1 or \
            source.count(END) != 1 or source.count(OLD) != 1:
        raise ValueError("E3 production finalizer identity differs")
    start = source.index(START)
    end = source.index(END, start)
    function = source[start:end]  # Explicitly excludes the trailing newline.
    if digest(function) != FUNCTION_SHA or \
            digest(source[start:end + 1]) != WITH_NEWLINE_SHA or \
            function.count(OLD) != 1:
        raise ValueError("E3 finalizer framing or guarded anchor differs")
    offset = function.index(OLD)
    derived_function = function[:offset] + NEW + function[offset + len(OLD):]
    derived = source[:start] + derived_function + source[end:]
    if digest(derived) != DERIVED_SHA or \
            derived[:start] != source[:start] or \
            derived[start + len(derived_function):] != source[end:]:
        raise ValueError("E3 clamp overlay differs from the reviewed two-line mutant")
    receipt = {
        "kind": "e3-overflow-clamp-overlay-v1", "source_sha256": SOURCE_SHA,
        "function_sha256_excluding_trailing_newline": FUNCTION_SHA,
        "function_sha256_including_trailing_newline": WITH_NEWLINE_SHA,
        "derived_sha256": DERIVED_SHA, "function_start": start,
        "function_end_exclusive": end, "anchor_offset_in_function": offset,
    }
    return derived, receipt


def _paths(output: Path) -> tuple[Path, Path]:
    if output.name != "e3_evaluation_metrics_provider.c" or \
            output.parent.name != "native" or \
            output.parent.parent.name != ".e3-overflow-generated" or \
            not output.is_absolute() or output.parent.parent.is_symlink() or \
            output.parent.parent.resolve() != output.parent.parent:
        raise ValueError("overlay must occupy the owned overflow native stage")
    receipt = output.parent / "overlay-receipt.json"
    return output, receipt


def run(output: Path, *, check: bool = False) -> dict[str, object]:
    output, receipt = _paths(output)
    expected, metadata = derive(SOURCE.read_bytes())
    encoded = (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if check:
        if output.is_symlink() or receipt.is_symlink() or \
                output.read_bytes() != expected or receipt.read_bytes() != encoded or \
                set(output.parent.iterdir()) != {output, receipt}:
            raise ValueError("retained overlay or receipt differs")
    else:
        if output.parent.exists() or output.parent.is_symlink():
            raise ValueError("overlay native directory must be fresh")
        output.parent.mkdir(mode=0o700)
        with output.open("xb") as stream:
            stream.write(expected)
        with receipt.open("xb") as stream:
            stream.write(encoded)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, check=args.check), sort_keys=True,
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
