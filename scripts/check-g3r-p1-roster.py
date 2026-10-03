#!/usr/bin/env python3
"""Check the private candidate roster stays outside ordinary P1 products."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "native/p1_identity.c").read_text()
header = (root / "native/p1_identity_internal.h").read_text()
test = (root / "tests/p1/test_g3r_candidate_roster.c").read_text()

def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(f"G3-R P1 ROSTER STRUCTURE FAIL: {message}")

flag = "ET_G3R_CANDIDATE_RETIRE_PRIVATE"
require(flag in source and flag in header and flag in test, "feature guard missing")
for name in (
    "et_p1_private_candidate_construction_begin_v1",
    "et_p1_private_candidate_graph_preflight_v1",
    "et_p1_private_candidate_graph_revoke_v1",
):
    require(source.count(name) == 1 and header.count(name) == 1,
            f"missing or duplicate private entry: {name}")
require("record->candidate_construction != ledger" in source and
        "record->candidate_index != i" in source and
        source.index("!p1_record_registered(record)") <
        source.index("record->candidate_construction != ledger"),
        "roster records are read before registry-first admission")
require("ledger->entries != ledger->candidate_roster_base" in source and
        "ledger->count != ledger->sealed_count" in source,
        "roster extent or identity check missing")
require("ET_G3R_CANDIDATE_RETIRE_PRIVATE" not in
        (root / "native/p1_identity_public_symbols.txt").read_text(),
        "public manifest mentions candidate feature")
for name in (
    "et_p1_private_candidate_construction_begin_v1",
    "et_p1_private_candidate_graph_preflight_v1",
    "et_p1_private_candidate_graph_revoke_v1",
):
    require(name not in
            (root / "native/p1_identity_trusted_symbols.txt").read_text(),
            "ordinary trusted manifest includes candidate symbol")
print("G3-R P1 ROSTER STRUCTURE PASS: private guard, registry-first admission")
