#!/usr/bin/env python3
"""Closed source/ownership contract for private G3-T kind-3 logits."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_full_request_preflight_source_closure.txt").read_text().splitlines()
additions = ["scripts/check-g3t-manual-logits.py", "docs/g3/G3_T_MANUAL_LOGITS_LEAF.md"]
closure = (root / "native/g3t_manual_logits_source_closure.txt").read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
header = (root / "src/eshkol_transformer/g3t_transport.h").read_text()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()
name = "et_g3t_private_logits_reserve_v1"
assert native.count(name) == header.count(name) == 1
start = native.index("void *" + name)
end = native.index("\n#endif", start)
body = native[start:end]
for phrase in (
    "g3t_active(candidate)", "c->call_kind != 0 && c->call_kind != 1",
    "c->pending_logits", "g3t_logits_allocate()",
    "et_f32_tensor_create_v1(2, shape", "g3t_registry = &logits->h",
    "c->pending_logits = logits",
):
    assert phrase in body, phrase
assert body.index("et_f32_tensor_create_v1") < body.index("g3t_registry = &logits->h")
assert "c->policy[4] != 0" not in body
assert body.index("g3t_registry = &logits->h") < body.index("c->pending_logits = logits")
assert "G3T_LOGITS = 3" in native
assert "record->kind == G3T_LOGITS" in native
borrow = native[native.index("void *et_g3t_test_logits_borrow_begin_v1"):]
borrow = borrow[:borrow.index("\n}")]
assert "candidate, G3T_LOGITS, 1" in borrow
assert "logits->h.state == G3T_PENDING" in borrow
assert "et_f32_tensor_destroy_v1(&logits->tensor" in native
assert "et_f32_tensor_scoped_begin_internal(" in native
assert "#ifdef ET_G3T_TESTING" in native
assert "-DET_G3T_MANUAL_LOGITS_PRIVATE" in runner
assert "-DET_F32_TENSOR_TESTING" in runner
assert "native/g3t_manual_logits_source_closure.txt" in runner
for witness in (
    "manual logits forged context rejected", "manual logits idle context rejected",
    "manual logits pending owned f32 [1,256]", "manual logits duplicate reserve rejected atomically",
    "manual logits borrow blocks typed release", "manual logits borrow blocks abort before pin drain",
    "manual logits abort destroys pending I2 once", "manual decode logits reserve",
    "generate kind2 does not reserve manual logits", "manual logits header cut preserves call and owners",
    "manual logits I2 cut preserves call and owners",
    "manual G1 exhausted prefill logits reserve",
    "manual G1 exhausted decode logits reserve",
):
    assert witness in test, witness
print("G3-T private kind-3 manual logits source contract: PASS")
