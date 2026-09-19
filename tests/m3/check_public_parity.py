"""Compare compiled M3 stdout with the current independent development record."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[2]
# Per-tensor maximum-norm bound; never combine small Q/K with larger parameters.
# These are mathematical-f64 versus native-f32 bounds, not a bitwise libm claim.
ABSOLUTE_BOUND = 1e-10
RELATIVE_BOUND = 3e-4
WEIGHT_BITS = (1065353216, 1056964608, 1075838976)  # 1, 1/2, 5/2; never scale VJPs.


def payload(fields, count):
    if len(fields) != count:
        raise ValueError(f"expected exactly {count} bytes, got {len(fields)}")
    values = [int(value) for value in fields]
    if any(not 0 <= value < 256 for value in values):
        raise ValueError("observed value is not a byte")
    return bytes(values)


def compare_values(label, raw, expected):
    observed = struct.unpack("<" + "f" * len(expected), raw)
    if any(not math.isfinite(value) for value in observed):
        raise ValueError(f"{label}: nonfinite result")
    maximum = max(map(abs, expected), default=0.0)
    error = max((abs(a - b) for a, b in zip(observed, expected)), default=0.0)
    bound = ABSOLUTE_BOUND + RELATIVE_BOUND * maximum
    if error > bound:
        raise ValueError(f"{label}: max error {error:.9g} exceeds per-tensor bound {bound:.9g}")
    return error, maximum


def check(reference_path, observed_path, instrumented=False):
    encoded = Path(reference_path).read_bytes()
    document = json.loads(encoded)
    manifest = json.loads(Path(__file__).with_name("reference_manifest.json").read_text())
    if manifest.get("format") != "eshkol-m3-reference-manifest" or manifest.get("version") != 1:
        raise ValueError("wrong reference manifest version")
    if hashlib.sha256(encoded).hexdigest() != manifest.get("reference_sha256"):
        raise ValueError("generated reference checksum differs from reviewed manifest")
    if document.get("format") != "eshkol-m3-mathematical-reference" or document.get("version") != 1:
        raise ValueError("wrong development reference format")
    for path in ("tests/m3/reference.py", "tests/n3k/test_initializer_reference.py", "tests/q0/requirements-oracle.lock"):
        if document.get("source_sha256", {}).get(path) != hashlib.sha256((ROOT / path).read_bytes()).hexdigest():
            raise ValueError(f"stale reference source/lock identity: {path}")
    if document.get("torch") != "2.13.0+cpu" or document.get("python") != "3.14.6":
        raise ValueError("wrong reference environment versions")
    if document.get("reference_environment") != {"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE"}:
        raise ValueError("wrong reference CPU controls")
    records = document["cases"][:3]
    if [(item["initializer_seed"], item["tokens"]) for item in records] != [(1729, [3, 197]), (0, [0, 255]), ((1 << 63) - 1, [7, 7])]:
        raise ValueError("wrong public reference case identities")
    lines = Path(observed_path).read_text().splitlines()
    if not lines or lines[-1] != "M3-PUBLIC-MODEL-PASS":
        raise ValueError("missing exact successful public completion marker")
    seen = set()
    summaries = []
    for line in lines[:-1]:
        fields = line.split()
        if len(fields) < 2 or fields[0] not in ("M3-LOGITS", "M3-GRADIENT", "M3-PARAMETER", "M3-METADATA"):
            raise ValueError("unexpected public result row")
        role, case = fields[0], int(fields[1])
        if not 0 <= case < 3:
            raise ValueError("unexpected case index")
        item = records[case]
        if role == "M3-LOGITS":
            key = (role, case)
            raw = payload(fields[2:], 2048)
            summaries.append((f"case {case} logits", *compare_values(f"case {case} logits", raw, item["logits"]["values"])))
        else:
            if not instrumented or len(fields) < 3:
                raise ValueError("test-only observation in production public result")
            index = int(fields[2])
            # JSON is canonical key-sorted; the paths themselves are canonical.
            paths = sorted(item["parameters"])
            if len(paths) != 14 or not 0 <= index < 14:
                raise ValueError("wrong unique parameter index/count")
            path = paths[index]
            key = (role, case, index)
            if role == "M3-METADATA":
                if fields[3:] != ["1", "1", str(WEIGHT_BITS[case])]:
                    raise ValueError(f"case {case} {path}: expected present/count1/exact case weight")
            else:
                expected = item["parameters" if role == "M3-PARAMETER" else "unique_vjp"][path]["values"]
                raw = payload(fields[3:], 4 * len(expected))
                if role == "M3-PARAMETER":
                    if raw != struct.pack("<" + "f" * len(expected), *expected):
                        raise ValueError(f"case {case} {path}: initializer bits differ")
                else:
                    summaries.append((f"case {case} {path}", *compare_values(f"case {case} {path}", raw, expected)))
        if key in seen:
            raise ValueError(f"duplicate result row: {key}")
        seen.add(key)
    expected_rows = {("M3-LOGITS", case) for case in range(3)}
    if instrumented:
        expected_rows |= {(role, case, index) for role in ("M3-GRADIENT", "M3-PARAMETER", "M3-METADATA") for case in range(3) for index in range(14)}
    if seen != expected_rows:
        raise ValueError("missing or unexpected result rows")
    for label, error, maximum in summaries:
        print(f"{label}: max-error={error:.9g} reference-max={maximum:.9g}")
    print("M3-INSTRUMENTED-PARITY-PASS" if instrumented else "M3-PUBLIC-LOGITS-PARITY-PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--public-output", required=True, type=Path)
    parser.add_argument("--instrumented", action="store_true")
    args = parser.parse_args()
    check(args.reference, args.public_output, args.instrumented)
