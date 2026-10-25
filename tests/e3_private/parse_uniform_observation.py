"""Require genuine zero-state admission, then parse every E3 frame record."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from tests.e3_private.parse_numerical_observation import parse
from tests.e3_reference.transcript import canonical, decode


MARKER = b"E3-UNIFORM-ZERO-STATE-PASS\n"


CURSOR = re.compile(rb"E3CURSOR\t([0-9a-f]+)\t([0-9a-f]+)\n\Z")
PASS = b"E3-NUMERICAL-RUNTIME-PASS\n"


def parse_uniform(raw: bytes, batch_count: int, expected_cursor: str):
    if not raw.startswith(MARKER) or raw.count(MARKER) != 1:
        raise ValueError("uniform model-state admission marker differs")
    if not raw.endswith(PASS):
        raise ValueError("uniform private runtime did not complete")
    body = raw[len(MARKER):-len(PASS)]
    match = CURSOR.search(body)
    if match is None or match.group(1) != match.group(2) or \
            match.group(1).decode("ascii") != expected_cursor:
        raise ValueError("uniform D2 saved cursor differs from exact reference")
    return parse(body[:match.start()] + PASS, batch_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--batch-count", required=True, type=int, choices=(3,))
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    args = parser.parse_args()
    reference = decode(args.reference.read_bytes())
    cases = reference.get("cases") if isinstance(reference, dict) else None
    if not isinstance(cases, list) or len(cases) != 1 or \
            cases[0].get("name") not in ("uniform-zero", "padding-hit"):
        raise ValueError("uniform reference case differs")
    expected_cursor = cases[0]["corpus"]["start_cursor"]
    payload, roles = parse_uniform(args.stdout.read_bytes(), args.batch_count,
                                   expected_cursor)
    args.observed.write_bytes(canonical(payload))
    args.roles.write_bytes(roles)


if __name__ == "__main__":
    main()
