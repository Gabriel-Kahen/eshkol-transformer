"""Parse the version-1 single-row target-offset private observation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from tests.e3_private.parse_numerical_observation import parse
from tests.e3_reference.target_offset import validate_materialized
from tests.e3_reference.transcript import canonical


PASS = b"E3-TARGET-OFFSET-RUNTIME-PASS\n"
FRAME_PASS = b"E3-NUMERICAL-RUNTIME-PASS\n"
RESULT = re.compile(
    rb"E3-TARGET-OFFSET\t(target-zero|target-seven)"
    + rb"(?:\t(0|[1-9][0-9]{0,9}))" * 6
    + rb"\t([0-9a-f]{2,800})\t([0-9a-f]{2,800})\n\Z"
)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate target manifest field")
        result[key] = value
    return result


def parse_target(raw: bytes, manifest: dict[str, object], case_name: str,
                 manifest_sha256: str) -> tuple[dict[str, object], bytes, dict[str, object]]:
    if case_name not in ("target-zero", "target-seven"):
        raise ValueError("unknown target-offset case")
    cases = manifest.get("cases")
    if (not isinstance(cases, list) or len(cases) != 2
            or [case.get("name") if isinstance(case, dict) else None for case in cases]
            != ["target-zero", "target-seven"]):
        raise ValueError("target-offset case inventory differs")
    case = cases[0 if case_name == "target-zero" else 1]
    expected = {"inputs": [3, 197],
                "targets": [197, 0 if case_name == "target-zero" else 7],
                "mask": [True, True]}
    if case.get("batches") != [expected]:
        raise ValueError("target-offset D2 row differs")
    lines = raw.splitlines(keepends=True)
    if len(lines) < 3 or lines[-1] != PASS:
        raise ValueError("target-offset runtime did not complete")
    match = RESULT.fullmatch(lines[-2])
    if match is None or match.group(1).decode("ascii") != case_name:
        raise ValueError("target-offset result is absent or mislabeled")
    outputs = [int(word) for word in match.groups()[1:7]]
    if any(word >= 1 << 32 for word in outputs):
        raise ValueError("target-offset result is outside u32")
    before, after = (word.decode("ascii") for word in match.groups()[7:])
    if (len(before) % 2 or len(after) % 2
            or before != case.get("start_cursor") or after != before):
        raise ValueError("target-offset caller cursor differs or was not restored")
    observed, roles = parse(b"".join(lines[:-2]) + FRAME_PASS, 1)
    batch = observed["batches"][0]
    if any(batch[key] != value for key, value in expected.items()):
        raise ValueError("compiled target-offset D2 row differs")
    final = observed["final"]
    if (outputs[:4] != [int(word, 16) for word in final["metrics_f32"]]
            or outputs[4:] != final["counters"]):
        raise ValueError("target-offset six outputs differ from observed frame")
    result = {
        "kind": "e3-target-offset-result-v1", "case": case_name,
        "outputs_u32": outputs, "cursor_before": before, "cursor_after": after,
        "provenance": {"corpora_sha256": manifest_sha256,
                       "stdout_sha256": hashlib.sha256(raw).hexdigest()},
    }
    return observed, roles, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=("target-zero", "target-seven"))
    parser.add_argument("--observed", required=True, type=Path)
    parser.add_argument("--roles", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    manifest_raw = (args.corpus / "corpora.json").read_bytes()
    manifest = json.loads(manifest_raw, object_pairs_hook=_unique)
    if canonical(manifest) != manifest_raw:
        raise ValueError("target-offset manifest is not canonical")
    validate_materialized(args.corpus, manifest)
    observed, roles, result = parse_target(
        args.stdout.read_bytes(), manifest, args.case,
        hashlib.sha256(manifest_raw).hexdigest(),
    )
    for path, content in ((args.observed, canonical(observed)),
                          (args.roles, roles), (args.result, canonical(result))):
        with path.open("xb") as stream:
            stream.write(content)


if __name__ == "__main__":
    main()
