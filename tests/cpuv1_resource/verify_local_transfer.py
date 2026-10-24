"""Verify two finite c5f resource-observer holder transfers, not item-1 admission.

Reads only the sealed collector cache. No compiler, IR scan, link, or target run.
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


SOURCE = "c5f50fd8dc73603b4092757489f15906809238ab"
IR_SHA256 = "f303a35369a4ee2413c127e78d6a4613cb9c6da3ab6a27e433adc68a13e8f6c1"
HASHES = {
    "STREAM.sqlite": "0fcd911410b877bafb0bda363551a35214c8b5a1455b1ad4ff4c385329f9c47f",
    "PREFIX.bin": "373aa1a3ad16f01cbadd949ac77f4105846e8c573cc3ce81f69fe66a28f8169e",
    "FUNCTIONS.json": "391e1115936c240e8b1cdb7074ded414888e9e2f908b20fd3770cdab3c7d94ff",
    "PRIVATE-INDEX.json": "c02d5270eab07f8bf3224b644a400df9a013abde0461465884c64edb56e6ac82",
    "SUMMARY.json": "1a1b7e7659d332085a60f0c5e10d2ac42dd90513530968b7102dc4aaeeab75f9",
}

# Exact selected MAIN instructions. Equality deliberately rejects altered SSA,
# tag, source holder, closure target, capture base, or offset.
ROWS = {
    "p1": {
        25168097: "%sym3686997 = call ptr @eshkol_intern_symbol_lookup(ptr @.sym_name.101033)",
        25168098: "%8990091 = ptrtoint ptr %sym3686997 to i64",
        25168099: "%8990092 = insertvalue %eshkol_tagged_value { i8 8, i8 0, i16 0, i32 0, i64 undef }, i64 %8990091, 4",
        25168100: "store %eshkol_tagged_value %8990092, ptr %cpuv1-resource-p1-state-live_letrecstar, align 8",
        25168101: "%8990093 = load ptr, ptr @__global_arena, align 8",
        25168102: "%8990094 = call ptr @arena_allocate(ptr %8990093, i64 16), !eshkol.arena_alloc.checked !0",
        25168107: "%cpuv1-resource-p1-state-live_val = load %eshkol_tagged_value, ptr %cpuv1-resource-p1-state-live_letrecstar, align 8",
        25168108: "store %eshkol_tagged_value %cpuv1-resource-p1-state-live_val, ptr %8990094, align 8",
        25214978: "%9009453 = call ptr @arena_allocate_closure_with_header(ptr %9009452, i64 ptrtoint (ptr @tiny_lambda_473 to i64), i64 8589934595, i64 %9009451, i64 655872, ptr null), !eshkol.arena_alloc.checked !0",
        25214983: "%9009454 = getelementptr i8, ptr %9009453, i64 8",
        25214984: "%9009455 = load ptr, ptr %9009454, align 8",
        25214989: "%9009459 = ptrtoint ptr %8990094 to i64",
        25214990: "%9009460 = insertvalue %eshkol_tagged_value { i8 1, i8 16, i16 0, i32 0, i64 undef }, i64 %9009459, 4",
        25214991: "%9009461 = getelementptr i8, ptr %9009455, i64 24",
        25214992: "store %eshkol_tagged_value %9009460, ptr %9009461, align 8",
    },
    "candidate": {
        31395997: "%sym4779317 = call ptr @eshkol_intern_symbol_lookup(ptr @.sym_name.127788)",
        31395998: "%11649514 = ptrtoint ptr %sym4779317 to i64",
        31395999: "%11649515 = insertvalue %eshkol_tagged_value { i8 8, i8 0, i16 0, i32 0, i64 undef }, i64 %11649514, 4",
        31396000: "store %eshkol_tagged_value %11649515, ptr %cpuv1-resource-candidate-state-unpublished_letrecstar, align 8",
        31396001: "%11649516 = load ptr, ptr @__global_arena, align 8",
        31396002: "%11649517 = call ptr @arena_allocate(ptr %11649516, i64 16), !eshkol.arena_alloc.checked !0",
        31396007: "%cpuv1-resource-candidate-state-unpublished_val = load %eshkol_tagged_value, ptr %cpuv1-resource-candidate-state-unpublished_letrecstar, align 8",
        31396008: "store %eshkol_tagged_value %cpuv1-resource-candidate-state-unpublished_val, ptr %11649517, align 8",
        31430257: "%11663610 = call ptr @arena_allocate_closure_with_header(ptr %11663609, i64 ptrtoint (ptr @tiny_lambda_676 to i64), i64 8589934598, i64 %11663608, i64 655872, ptr null), !eshkol.arena_alloc.checked !0",
        31430262: "%11663611 = getelementptr i8, ptr %11663610, i64 8",
        31430263: "%11663612 = load ptr, ptr %11663611, align 8",
        31430268: "%11663616 = ptrtoint ptr %11649517 to i64",
        31430269: "%11663617 = insertvalue %eshkol_tagged_value { i8 1, i8 16, i16 0, i32 0, i64 undef }, i64 %11663616, 4",
        31430270: "%11663618 = getelementptr i8, ptr %11663612, i64 24",
        31430271: "store %eshkol_tagged_value %11663617, ptr %11663618, align 8",
    },
}

HEADERS = {
    "tiny_lambda_473": "ptr %captured_cpuv1-resource-p1-state-live",
    "tiny_lambda_676": "ptr %captured_cpuv1-resource-candidate-state-unpublished",
}

MISSING = [
    "The 29 namespaced/lexical private definitions absent from prefix-global definitions are bound on every required path.",
    "Initialization of all 126 private values dominates every observer call.",
    "All P1/candidate and owning observers resolve to the same canonical holders through indirect dispatch.",
    "The global arena and captured holders remain live through every observation and close, without reset or replacement.",
    "Complete generated and native calling-ABI transfer, including packet/context validity and provider execution.",
    "Recursive transitive effect closure remains FAIL; fresh-process resource runtime remains NOT_RUN.",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_rows(actual, functions):
    errors = []
    evidence = {}
    for label, expected in ROWS.items():
        evidence[label] = []
        for line, instruction in expected.items():
            row = actual.get(line)
            if row is None or row[0].strip() != instruction:
                errors.append(f"{label}: missing or changed MAIN instruction at line {line}")
                continue
            evidence[label].append({"line": line, "raw_sha256": row[1]})
    for name, capture in HEADERS.items():
        matches = [f for f in functions if f.get("name") == name]
        if len(matches) != 1 or capture not in matches[0].get("header", ""):
            errors.append(f"{name}: absent or changed typed capture header")
    return errors, evidence


def verify(cache):
    errors = []
    for name, expected in HASHES.items():
        path = cache / name
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"{name}: missing or SHA256 mismatch")
    if errors:
        return errors, {}
    summary = json.loads((cache / "SUMMARY.json").read_text())
    if summary.get("ir_sha256") != IR_SHA256 or summary.get("status") != "PASS_LEXICAL_COLLECTION_ONLY":
        return ["SUMMARY.json: wrong IR identity or collection status"], {}
    functions = json.loads((cache / "FUNCTIONS.json").read_text())
    con = sqlite3.connect(f"file:{cache / 'STREAM.sqlite'}?mode=ro&immutable=1", uri=True)
    try:
        lines = sorted(set().union(*(set(rows) for rows in ROWS.values())))
        actual = {}
        for line in lines:
            hits = con.execute(
                "SELECT raw,raw_sha256 FROM main_windows WHERE line=?", (line,)
            ).fetchall()
            if len(hits) != 1:
                errors.append(f"MAIN line {line}: expected one cached row, found {len(hits)}")
                continue
            raw, recorded = hits[0]
            if hashlib.sha256(raw).hexdigest() != recorded:
                errors.append(f"MAIN line {line}: cached row SHA256 mismatch")
            actual[line] = (raw.decode("utf-8"), recorded)
    finally:
        con.close()
    row_errors, evidence = check_rows(actual, functions)
    return errors + row_errors, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="sealed c5f collector output directory")
    args = parser.parse_args()
    errors, evidence = verify(args.cache.resolve())
    print(json.dumps({
        "source_commit": SOURCE,
        "ir_sha256": IR_SHA256,
        "local_transfer": "PASS_LOCAL_TRANSFER" if not errors else "FAIL_LOCAL_TRANSFER",
        "item1": "FAIL_ITEM1",
        "checked_rows": evidence if not errors else {},
        "errors": errors,
        "missing_premises": MISSING,
    }, indent=2, sort_keys=True))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
