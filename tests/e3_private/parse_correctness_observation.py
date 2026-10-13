"""Bind the private compiled result cell and D2 cursors to its E3 frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from tests.e3_private.parse_numerical_observation import parse
from tests.e3_reference.transcript import canonical


RESULT = re.compile(rb"E3RESULT\t(0|[1-9][0-9]{0,9})(?:,(0|[1-9][0-9]{0,9})){5}\n\Z")
CURSOR = re.compile(rb"E3CURSOR\t([0-9a-f]{2,800})\t([0-9a-f]{2,800})\n\Z")
PASS = b"E3-NUMERICAL-RUNTIME-PASS\n"


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate correctness manifest field")
        result[key] = value
    return result


def parse_correctness(raw: bytes, manifest: dict[str, object], case_name: str):
    lines = raw.splitlines(keepends=True)
    if len(lines) < 4 or lines[-1] != PASS or not RESULT.fullmatch(lines[-3]):
        raise ValueError("compiled correctness result is absent or malformed")
    match = CURSOR.fullmatch(lines[-2])
    if match is None:
        raise ValueError("compiled correctness cursors are absent or malformed")
    if any(len(cursor) % 2 for cursor in match.groups()):
        raise ValueError("compiled correctness cursor has odd length")
    words = [int(part) for part in lines[-3].split(b"\t", 1)[1].strip().split(b",")]
    if len(words) != 6 or any(word >= 1 << 32 for word in words):
        raise ValueError("compiled correctness result is outside u32")
    cases = manifest.get("cases")
    if (not isinstance(cases, list) or len(cases) != 2
            or [case.get("name") if isinstance(case, dict) else None for case in cases]
            != ["all-correct", "all-wrong"]):
        raise ValueError("correctness case inventory differs")
    selected = [case for case in cases if case["name"] == case_name]
    if len(selected) != 1:
        raise ValueError("correctness case is absent")
    case = selected[0]
    before, after = (part.decode("ascii") for part in match.groups())
    if (before != case.get("start_cursor") or after != before):
        raise ValueError("compiled correctness caller cursor was not restored")
    observed, roles = parse(b"".join(lines[:-3]) + PASS, 3)
    final = observed["final"]
    if words[:4] != [int(word, 16) for word in final["metrics_f32"]]:
        raise ValueError("compiled six-output metrics differ from observed frame")
    if words[4:] != final["counters"]:
        raise ValueError("compiled six-output counters differ from observed frame")
    return observed, roles, {"kind": "e3-correctness-result-v1", "case": case_name,
                             "outputs_u32": words, "cursor_before": before,
                             "cursor_after": after}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("all-correct", "all-wrong"))
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw, object_pairs_hook=_unique)
    if canonical(manifest) != raw:
        raise ValueError("correctness manifest is not canonical")
    observed, roles, result = parse_correctness(args.stdout.read_bytes(), manifest, args.case)
    for path, payload in ((args.observed, canonical(observed)),
                          (args.roles, roles), (args.result, canonical(result))):
        with path.open("xb") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
