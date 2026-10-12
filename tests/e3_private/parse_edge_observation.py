"""Bind a compiled N5 observation to physical D1, D2 cursor and Q0 words."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from tests.e3_private.parse_numerical_observation import parse as parse_numerical
from tests.e3_reference.edge_corpus import validate
from tests.e3_reference.transcript import canonical, validate_mathematical_case


LINE = re.compile(rb"E3EDGE\t([a-z-]+)\t(success|invalid-state)\t"
                  rb"(-?[0-9]+(?:,-?[0-9]+){5})\t"
                  rb"(-?[0-9]+(?:,-?[0-9]+){5})\t"
                  rb"([0-9a-f]{2,800})\t([0-9a-f]{2,800})\n\Z")
PASS = b"E3-EDGE-RUNTIME-PASS\n"


def _six(raw: bytes) -> list[int]:
    values = [int(value) for value in raw.split(b",")]
    if len(values) != 6 or any(not -1 <= value < 1 << 32 for value in values):
        raise ValueError("N5 six-output word is outside its type")
    return values


def parse(raw: bytes, root: Path, manifest: dict[str, object], name: str,
          oracle: dict[str, object] | None = None) -> dict[str, object]:
    validate(root, manifest)
    cases = {case["name"]: case for case in manifest["cases"]}
    if name not in cases or not raw.endswith(PASS):
        raise ValueError("N5 case or runtime completion missing")
    head = raw[:-len(PASS)]
    index = head.rfind(b"E3EDGE\t")
    if index < 0:
        raise ValueError("N5 result missing")
    match = LINE.fullmatch(head[index:])
    if match is None:
        raise ValueError("N5 result malformed")
    observed_name, status, before_raw, after_raw, cursor_raw, end_raw = match.groups()
    case = cases[name]
    if observed_name.decode() != name or status.decode() != case["outcome"]:
        raise ValueError("N5 case or typed outcome differs")
    before, after = _six(before_raw), _six(after_raw)
    cursor, end = cursor_raw.decode(), end_raw.decode()
    if len(cursor) % 2 or cursor != case["start_cursor"] or end != cursor:
        raise ValueError("N5 caller cursor changed")
    prefix = head[:index]
    if status == b"invalid-state":
        if prefix or before != after or before != [0] * 6:
            raise ValueError("N5 zero-weight failure changed caller outputs")
    else:
        if before != [0] * 6 or oracle is None:
            raise ValueError("N5 success lacks independent oracle or clean outputs")
        observed, _ = parse_numerical(prefix + b"E3-NUMERICAL-RUNTIME-PASS\n", 2)
        if (oracle.get("kind") != "e3-edge-oracle-v1"
                or oracle.get("case") != name or oracle.get("cursor") != cursor
                or oracle.get("model_seed") != 1729
                or oracle.get("python") != "3.14.6"
                or oracle.get("torch") != "2.13.0+cpu"
                or oracle.get("corpus_sha256") != hashlib.sha256(
                    (root / "edge-corpus.json").read_bytes()).hexdigest()
                or oracle.get("m3_reference_sha256") != hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / "m3/reference.py").read_bytes()).hexdigest()
                or oracle.get("q0_lock_sha256") != hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / "q0/requirements-oracle.lock").read_bytes()).hexdigest()):
            raise ValueError("N5 oracle identity differs")
        mathematical = oracle.get("mathematical")
        mathematical_loss = oracle.get("mathematical_loss_f64")
        if not isinstance(mathematical, list) or not isinstance(mathematical_loss, str):
            raise ValueError("N5 mathematical oracle schema differs")
        validate_mathematical_case(
            observed, mathematical, tuple(case["batches"]), mathematical_loss)
        expected = ([int(word, 16) for word in observed["final"]["metrics_f32"]]
                    + observed["final"]["counters"])
        if after != expected:
            raise ValueError("N5 six published outputs differ from observed frame")
    return {"kind": "e3-edge-result-v1", "case": name,
            "status": status.decode(), "before": before, "after": after,
            "cursor_before": cursor, "cursor_after": end}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--case", required=True, choices=(
        "empty", "singleton", "entry-eos", "one-valid-final"))
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest_raw = (args.corpus / "edge-corpus.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if canonical(manifest) != manifest_raw:
        raise ValueError("N5 corpus manifest is not canonical")
    oracle = None
    if args.oracle:
        oracle_raw = args.oracle.read_bytes()
        oracle = json.loads(oracle_raw)
        if canonical(oracle) != oracle_raw:
            raise ValueError("N5 oracle manifest is not canonical")
    result = parse(args.stdout.read_bytes(), args.corpus, manifest, args.case, oracle)
    with args.output.open("xb") as stream:
        stream.write(canonical(result))
