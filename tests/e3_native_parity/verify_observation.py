"""Validate real native E3 observations with the accepted reference toolkit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct

from tests.e3_reference.reference import ROLE_NAMES, validate_bundle
from tests.e3_reference.transcript import (
    F32_TOLERANCE,
    decode,
    encode,
    validate_mathematical_case,
    validate_observed_case,
)


def _f32(word: str) -> float:
    if len(word) != 8 or any(character not in "0123456789abcdef" for character in word):
        raise ValueError("native role observation contains a malformed binary32 word")
    value = struct.unpack(">f", bytes.fromhex(word))[0]
    if not math.isfinite(value):
        raise ValueError("native role observation contains a nonfinite value")
    return value


def _f64(word: str) -> float:
    return struct.unpack(">d", bytes.fromhex(word))[0]


def _roles(path: Path) -> list[tuple[int, str, tuple[int, ...], tuple[str, ...]]]:
    result = []
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"role observation line {line_number} is malformed")
        ordinal_text, name, shape_text, words_text = fields
        if not ordinal_text.isascii() or not ordinal_text.isdecimal():
            raise ValueError(f"role observation line {line_number} has a bad ordinal")
        shape = tuple(int(value) for value in shape_text.split(","))
        words = tuple(words_text.split(","))
        if not shape or math.prod(shape) != len(words):
            raise ValueError(f"role observation line {line_number} has a bad shape")
        for word in words:
            _f32(word)
        result.append((int(ordinal_text), name, shape, words))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    arguments = parser.parse_args()

    bundle = decode(arguments.reference.read_bytes())
    validate_bundle(bundle)
    matches = [case for case in bundle["cases"] if case["name"] == arguments.case]
    if len(matches) != 1:
        raise ValueError("requested case is absent or duplicated")
    case = matches[0]
    observed = json.loads(arguments.observed.read_bytes())
    if not isinstance(observed, dict) or observed.get("kind") != "observed-f32":
        raise ValueError("native observation is not explicitly labeled observed-f32")
    expected_batches = tuple(case["corpus"]["batches"])
    validate_observed_case(observed, expected_batches)
    measurements = validate_mathematical_case(
        observed, case["mathematical"], expected_batches,
        case["mathematical_loss_f64"],
    )

    roles = _roles(arguments.roles)
    expected_count = len(expected_batches) * len(ROLE_NAMES)
    if len(roles) != expected_count:
        raise ValueError("native observation does not contain every role for every batch")
    maximum_role_error = 0.0
    for batch_index, mathematical in enumerate(case["mathematical"]):
        reference_roles = mathematical["roles"]
        for role_index, expected in enumerate(reference_roles):
            ordinal, name, shape, words = roles[
                batch_index * len(ROLE_NAMES) + role_index
            ]
            if ordinal != batch_index or name != ROLE_NAMES[role_index]:
                raise ValueError("native role order or identity differs from the fixed schedule")
            if name != expected["name"] or list(shape) != expected["shape"]:
                raise ValueError("native role shape differs from the mathematical reference")
            reference = tuple(_f64(word) for word in expected["values_f64"])
            actual = tuple(_f32(word) for word in words)
            if len(actual) != len(reference):
                raise ValueError("native role element count differs from the reference")
            delta = max(abs(left - right) for left, right in zip(actual, reference))
            scale = max(abs(value) for value in reference)
            if delta > 1.0e-10 + 3.0e-4 * scale:
                raise ValueError(f"native role {name} exceeds the accepted M3 tolerance")
            maximum_role_error = max(maximum_role_error, delta)
            if name == "Z" and list(words) != observed["batches"][batch_index]["logits_f32"]:
                raise ValueError("native Z role and strict transcript logits differ")

    raw = encode(observed)
    if decode(raw) != observed:
        raise AssertionError("strict transcript round trip changed native observation")
    with arguments.transcript.open("xb") as output:
        output.write(raw)
    print(
        f"case={arguments.case} batches={len(expected_batches)} "
        f"maximum_role_error={maximum_role_error:.17g} "
        f"maximum_logit_error={measurements['maximum_logit_error']:.17g} "
        f"final_loss_error={measurements['error']:.17g} "
        f"final_loss_budget={measurements['budget']:.17g}"
    )


if __name__ == "__main__":
    main()
