"""Check boxed six-output and D2 cursor observations against frozen cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.d2.reference import decode_reference_cursor, encode_reference_cursor
from tests.e3_reference.corpus import CASES, validate_payload
from tests.e3_reference.transcript import decode


def verify(reference: dict[str, object], case_name: str, result: dict[str, object]) -> None:
    validate_payload(reference)
    matches = [case for case in reference["cases"] if case["name"] == case_name]
    if len(matches) != 1 or set(result) != {"outputs", "cursor_before", "cursor_after"}:
        raise ValueError("numerical result identity or fields differ")
    case = matches[0]
    before = bytes.fromhex(result["cursor_before"])
    after = bytes.fromhex(result["cursor_after"])
    if before.hex() != case["start_cursor"]:
        raise ValueError("boxed starting cursor differs from independent D2")
    fields = decode_reference_cursor(before)
    expected_after = encode_reference_cursor(
        fields.manifest_digest, fields.tokenizer_fingerprint, fields.vocab_size,
        fields.config, case["total_rows"], case["total_rows"],
    )
    if after != expected_after:
        raise ValueError("boxed end cursor differs from independent D2")
    outputs = result["outputs"]
    if not isinstance(outputs, list) or len(outputs) != 6 or any(
        type(value) is not int or not 0 <= value <= (1 << 32) - 1
        for value in outputs
    ):
        raise ValueError("boxed six outputs are malformed")


def verify_layouts(reference: dict[str, object], results: dict[str, dict[str, object]]) -> None:
    validate_payload(reference)
    cases = {case["name"]: case for case in reference["cases"]}
    if set(results) != set(cases):
        raise ValueError("missing or extra numerical layout result")
    for name, result in results.items():
        verify(reference, name, result)
    single = cases["packed-single"]["batches"]
    if single != cases["packed-sharded"]["batches"]:
        raise ValueError("independent packed D2 rows differ")
    if cases["packed-suffix"]["batches"] != single[1:]:
        raise ValueError("independent packed suffix rows differ")
    if results["packed-single"]["outputs"] != results["packed-sharded"]["outputs"]:
        raise ValueError("packed repartition outputs differ")
    if cases["unpacked-three"]["batches"] == single or \
            cases["unpacked-pairs"]["batches"] == single or \
            cases["unpacked-three"]["batches"] == cases["unpacked-pairs"]["batches"]:
        raise ValueError("independent unpacked D2 boundaries did not differ")
    if results["unpacked-three"]["outputs"] == results["packed-single"]["outputs"] or \
            results["unpacked-pairs"]["outputs"] == results["packed-single"]["outputs"]:
        raise ValueError("unpacked boundary outputs did not differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--variant", choices=("a", "b", "poison", "sanitized"),
                        required=True)
    args = parser.parse_args()
    reference = decode(args.corpus.read_bytes())
    results = {name: json.loads((args.results / f"result-{args.variant}-{name}.json").read_bytes())
               for name, *_ in CASES}
    verify_layouts(reference, results)


if __name__ == "__main__":
    main()
