"""Bind boxed private AOT results to the frozen five-case D1/D2 reference.

This checks cursor restoration, exact weight/accuracy/counters, and the
perplexity relation at the published loss. The runtime's exact loss words are
regression controls; per-row private observation is still required for the
independent N2 numerical verdict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.e3_metrics.reference_probe import exp_bits
from tests.e3_reference.corpus import CASES
from tests.e3_reference.join import validate_five_case_join
from tests.e3_reference.transcript import bits_f32, decode


def _integers(raw: str, count: int, maximum: int) -> list[int]:
    parts = raw.split(",")
    if len(parts) != count or any(
        not part or not part.isascii() or not part.isdecimal()
        or str(int(part)) != part or int(part) > maximum for part in parts
    ):
        raise ValueError("malformed private result integers")
    return [int(part) for part in parts]


def _cursor(raw: str) -> str:
    parts = raw.split(",")
    if not 168 <= len(parts) <= 512:
        raise ValueError("malformed private cursor length")
    return bytes(_integers(raw, len(parts), 255)).hex()


def parse_stdout(raw: bytes) -> dict[str, dict[str, object]]:
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("private stdout is not ASCII") from error
    if not raw.endswith(b"\n") or len(lines) != len(CASES) + 1 or \
            lines[-1] != "E3-PRIVATE-RUNTIME-PASS":
        raise ValueError("private runtime result count or completion differs")
    if raw != ("\n".join(lines) + "\n").encode("ascii"):
        raise ValueError("private stdout is not canonical LF text")
    results = {}
    for line, (name, *_rest) in zip(lines[:-1], CASES):
        fields = line.split("\t")
        if len(fields) != 5 or fields[:2] != ["E3RESULT", name]:
            raise ValueError("private result identity or ordering differs")
        values = _integers(fields[2], 6, (1 << 63) - 1)
        if any(value > (1 << 32) - 1 for value in values[:4]):
            raise ValueError("private destination word exceeds binary32")
        results[name] = {
            "outputs": values,
            "cursor_before": _cursor(fields[3]),
            "cursor_after": _cursor(fields[4]),
        }
    return results


def verify_results(joined: tuple[dict[str, object], ...],
                   results: dict[str, dict[str, object]]) -> None:
    if [case["name"] for case in joined] != [case[0] for case in CASES] or \
            set(results) != {case[0] for case in CASES}:
        raise ValueError("private result case set differs from frozen join")
    for case in joined:
        name = case["name"]
        result = results[name]
        if set(result) != {"outputs", "cursor_before", "cursor_after"}:
            raise ValueError(f"{name}: private result fields differ")
        if result["cursor_before"] != case["start_cursor"] or \
                result["cursor_after"] != case["start_cursor"]:
            raise ValueError(f"{name}: private cursor differs from restored D2 start")
        if case["end_cursor"] == case["start_cursor"]:
            raise ValueError(f"{name}: independent D2 traversal did not advance")
        outputs = result["outputs"]
        if not isinstance(outputs, list) or len(outputs) != 6 or any(
            type(value) is not int or value < 0 or value > (1 << 63) - 1
            for value in outputs
        ) or any(value > (1 << 32) - 1 for value in outputs[:4]):
            raise ValueError(f"{name}: malformed six-output publication")
        expected = case["synthetic"]["final"]
        expected_words = expected["metrics_f32"]
        expected_counts = expected["counters"]
        if outputs[1] != int(expected_words[1], 16) or \
                outputs[3] != int(expected_words[3], 16) or \
                outputs[4:] != expected_counts:
            raise ValueError(f"{name}: weight, accuracy or counters differ from D2 oracle")
        bits_f32(f"{outputs[0]:08x}", f"{name}.loss")
        if abs(outputs[2] - exp_bits(outputs[0])) > 2:
            raise ValueError(f"{name}: perplexity differs from Decimal exp of published loss")
    if results["packed-single"]["outputs"] != results["packed-sharded"]["outputs"]:
        raise ValueError("packed repartition changed six published outputs")
    if any(results[name]["outputs"] == results["packed-single"]["outputs"]
           for name in ("unpacked-three", "unpacked-pairs")):
        raise ValueError("unpacked boundary did not change published outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    args = parser.parse_args()
    corpus = decode((args.corpus_root / "corpus-reference.json").read_bytes())
    bundle = decode(args.reference.read_bytes())
    joined = validate_five_case_join(args.corpus_root, corpus, bundle)
    verify_results(joined, parse_stdout(args.stdout.read_bytes()))


if __name__ == "__main__":
    main()
