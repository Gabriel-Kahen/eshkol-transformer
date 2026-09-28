#!/usr/bin/env python3
"""Closed source contract for the private G3-T manual P2 transport."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_manual_p1_prefill_source_closure.txt").read_text().splitlines()
added = [
    "docs/g3/G3_T_MANUAL_P2_TRANSPORT_PROPOSAL.md",
    "scripts/check-g3t-manual-p2-prefill.py",
    "docs/g3/G3_T_MANUAL_P2_PREFILL_LEAF.md",
]
closure = (root / "native/g3t_manual_p2_prefill_source_closure.txt").read_text().splitlines()
assert closure == base + added
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))

native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
roles = (root / "src/eshkol_transformer/g3t_prefill_roles.inc").read_text()
t2 = (root / "src/eshkol_transformer/g3t_prefill2_roles.inc").read_text()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()
for phrase in (
    "ET_G3T_MANUAL_P2_PREFILL_PRIVATE",
    "et_m3_private_i64_unborrowed_v1(input->tensor",
    "et_i64_tensor_borrow_view_v1(borrow, &view",
    "view->shape[0] != 1 || view->shape[1] != 2",
    "memcmp(keys->data, c->frame.p2.kh",
    "memcmp(values->data, c->frame.p2.vh",
    "c->frame.p2.z + 256",
):
    assert phrase in native, phrase
assert "g3t_prefill2_attention(c)" in roles
assert "g3t_prefill2_plain(c, ordinal)" in roles
assert "g3t_prefill2_attention" in t2 and "g3t_prefill2_plain" in t2
for witness in (
    "manual P2 authentic I1 admission fixture",
    "manual P2 inline-only pair rejected",
    "manual P2 borrowed I1 admission rejected",
    "manual P2 malformed typed I1 rejected",
    "manual P2 inline shadow mismatch rejected",
    "manual P2 input release after copied transcript",
    "manual P2 frame both K/V rows bit-exact to M3T",
    "manual P2 A2 attention failure",
    "manual P2 synthetic provider dispatch failure",
    "manual P2 old-cache lease blocks preflight",
    "manual P2 stale binding blocks preflight",
    "manual P2 last row all 256 bits exact to M3T",
    "manual P2 committed both K/V rows bit-exact to M3T",
    "manual P2 detached logits survive generator close",
    "manual P2 repeated publication retains older logits",
    "manual P2 borrowed pending result blocks abort",
):
    assert witness in test, witness
assert '-DET_G3T_MANUAL_P2_PREFILL_PRIVATE' in runner
assert 'native/g3t_manual_p2_prefill_source_closure.txt' in runner
print("G3-T manual P2 prefill source contract: PASS")
