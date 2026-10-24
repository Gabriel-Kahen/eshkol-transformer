"""Independent seed-1729 M3/CE/rational-metrics oracle for the N5 one-token row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from tests.d2.reference import decode_reference_cursor
from tests.e3_reference.reference import _synthetic_case
from tests.e3_reference.transcript import F32_TOLERANCE, f32_bits
from tests.m3 import reference as m3

EXPECTED_BITS = ("40b28797", "3f800000", "43846900", "00000000")
EXPECTED_MODEL_SHA256 = "ccf18d65dcacf6982fa69631e57277e146c23b85a4a163401dc39c3d23dff0ba"
EDGE_NAMES = ("one-token", "empty", "singleton", "entry-eos")


def build(preflight: dict[str, object]) -> dict[str, object]:
    if preflight.get("kind") != "e3-n5-d2-preflight-v1":
        raise ValueError("N5 preflight kind mismatch")
    cases = {case["name"]: case for case in preflight["cases"]}
    if set(EDGE_NAMES) - cases.keys():
        raise ValueError("N5 preflight is missing edge cases")
    one = cases["one-token"]
    row = {"inputs": [3, 0], "targets": [197, 0], "mask": [True, False]}
    if (one["rows"] != [row] or one["remaining_rows"] != 1
            or one["active_tokens"] != 1):
        raise ValueError("N5 one-token D2 row differs")
    expected_cursors = {}
    for name in EDGE_NAMES:
        case = cases[name]
        cursor = bytes.fromhex(case["entry_cursor"])
        parsed = decode_reference_cursor(cursor)
        if (parsed.total_rows != case["total_rows"]
                or parsed.next_ordinal != case["skip"]
                or parsed.manifest_digest.hex() != case["manifest_digest"]):
            raise ValueError(f"{name}: D2 cursor identity differs")
        expected_cursors[name] = cursor.hex()
    m3.setup()
    parameters, successor = m3.parameters(1729)
    if tuple(successor) != (1, 1729, 290, 0):
        raise ValueError("M3 initializer successor differs")
    parameter_words = b"".join(
        bytes.fromhex(f32_bits(value))
        for name, _ in m3.PARAMETERS
        for value in parameters[name].to(m3.torch.float32).detach().reshape(-1).tolist()
    )
    if hashlib.sha256(parameter_words).hexdigest() != EXPECTED_MODEL_SHA256:
        raise ValueError("M3 parameter identity differs")
    synthetic, _, _ = _synthetic_case([row], parameters)
    bits = tuple(synthetic["final"]["metrics_f32"])
    if bits != EXPECTED_BITS or synthetic["final"]["counters"] != [1, 1]:
        raise ValueError("N5 independent one-token oracle changed")
    return {"kind": "e3-n5-one-token-oracle-v1", "metrics_f32": list(bits),
            "counters": [1, 1], "cursor_hex": expected_cursors,
            "m3_parameter_f32_sha256": EXPECTED_MODEL_SHA256}


def _float(bits: int) -> float:
    if not 0 <= bits <= 0xFFFFFFFF:
        raise ValueError("observed metric is not a binary32 word")
    value = struct.unpack(">f", bits.to_bytes(4, "big"))[0]
    if not math.isfinite(value):
        raise ValueError("observed metric is nonfinite")
    return value


def check_run(oracle: dict[str, object], output: str) -> None:
    lines = output.splitlines()
    if len(lines) != 6 or lines[-1] != "E3-N5-EDGE-PASS one=1 rejected=3 outputs=6 cursors=4":
        raise ValueError("N5 edge runtime transcript shape differs")
    words = lines[0].split()
    if len(words) != 7 or words[0] != "OUTPUTS":
        raise ValueError("N5 six-output record differs")
    values = [int(word) for word in words[1:]]
    expected = [int(word, 16) for word in oracle["metrics_f32"]]
    for index in (0, 2):
        actual_f32 = _float(values[index])
        expected_f32 = _float(expected[index])
        absolute, relative = F32_TOLERANCE
        if abs(actual_f32 - expected_f32) > absolute + relative * abs(expected_f32):
            raise ValueError(f"N5 one-token metric {index} exceeds M3/CE tolerance")
    if (values[1] != expected[1] or values[3] != expected[3]
            or values[4:] != oracle["counters"]):
        raise ValueError("N5 exact weight/accuracy/counters differ")
    for line, name in zip(lines[1:5], EDGE_NAMES):
        fields = line.split()
        if fields[:2] != ["CURSOR", name]:
            raise ValueError(f"{name}: cursor transcript label differs")
        raw = bytes(int(value) for value in fields[2:])
        if raw.hex() != oracle["cursor_hex"][name]:
            raise ValueError(f"{name}: compiled cursor differs from D2 reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--check-run", type=Path)
    args = parser.parse_args()
    if args.preflight and args.output and not args.oracle and not args.check_run:
        result = build(json.loads(args.preflight.read_text()))
        args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    elif args.oracle and args.check_run and not args.preflight and not args.output:
        check_run(json.loads(args.oracle.read_text()), args.check_run.read_text())
    else:
        parser.error("select generation or run verification")


if __name__ == "__main__":
    main()
