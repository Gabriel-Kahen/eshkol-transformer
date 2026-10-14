"""Bind one-row numerical frame, six published words, and restored D2 cursor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from tests.e3_private.parse_numerical_observation import parse
from tests.e3_reference.n5_one_token import validate_manifest, validate_observed
from tests.e3_reference.transcript import canonical


RESULT = re.compile(rb"E3RESULT\t(0|[1-9][0-9]{0,9})(?:,(0|[1-9][0-9]{0,9})){5}\n\Z")
CURSOR = re.compile(rb"E3CURSOR\t([0-9a-f]{2,800})\t([0-9a-f]{2,800})\n\Z")
PASS = b"E3-NUMERICAL-RUNTIME-PASS\n"


def parse_one_token(raw: bytes, manifest: dict[str, object]):
    validate_manifest(manifest)
    lines = raw.splitlines(keepends=True)
    if len(lines) < 4 or lines[-1] != PASS or not RESULT.fullmatch(lines[-3]):
        raise ValueError("N5 one-token result is absent or malformed")
    cursor = CURSOR.fullmatch(lines[-2])
    if cursor is None or any(len(word) % 2 for word in cursor.groups()):
        raise ValueError("N5 one-token cursor is absent or malformed")
    words = [int(part) for part in lines[-3].split(b"\t", 1)[1].strip().split(b",")]
    if len(words) != 6 or any(word >= 1 << 32 for word in words):
        raise ValueError("N5 one-token result is outside u32")
    before, after = (part.decode("ascii") for part in cursor.groups())
    if before != after or before != manifest["start_cursor"]:
        raise ValueError("N5 one-token caller cursor was not restored")
    observed, roles = parse(b"".join(lines[:-3]) + PASS, 1)
    validate_observed(manifest, observed)
    final = observed["final"]
    if words[:4] != [int(word, 16) for word in final["metrics_f32"]]:
        raise ValueError("N5 one-token published metrics differ from frame")
    if words[4:] != final["counters"]:
        raise ValueError("N5 one-token published counters differ from frame")
    return observed, roles, {"kind": "e3-n5-one-token-result-v1",
                             "outputs_u32": words, "cursor_before": before,
                             "cursor_after": after}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    if canonical(manifest) != raw:
        raise ValueError("N5 manifest is not canonical")
    observed, roles, result = parse_one_token(args.stdout.read_bytes(), manifest)
    for path, payload in ((args.observed, canonical(observed)),
                          (args.roles, roles), (args.result, canonical(result))):
        with path.open("xb") as stream:
            stream.write(payload)


if __name__ == "__main__":
    main()
