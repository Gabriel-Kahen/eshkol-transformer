#!/usr/bin/env python3
"""Closed native source contract for private G3-T manual decode."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3m_prefill_p2_source_closure.txt").read_text().splitlines()
added = [
    "docs/g3/G3_T_MANUAL_DECODE_TRANSPORT_PROPOSAL.md",
    "scripts/check-g3t-manual-decode.py",
    "docs/g3/G3_T_MANUAL_DECODE_LEAF.md",
]
closure = (root / "native/g3t_manual_decode_source_closure.txt").read_text().splitlines()
assert closure == base + added
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
roles = (root / "src/eshkol_transformer/g3t_prefill_roles.inc").read_text()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()
for phrase in (
    "ET_G3T_MANUAL_DECODE_PRIVATE",
    "et_m3_private_i64_unborrowed_v1(input->tensor",
    "typed->shape[0] != 1 || typed->shape[1] != 1",
    "c->frame.prefix_k",
    "c->frame.prefix_v",
    "g3t_manual_decode_preflight(c)",
    "et_a2_kv_cache_transaction_commit_v1(",
):
    assert phrase in native, phrase
assert "c->call_kind == 1" in roles
assert "g3t_prefill_attention(c)" in roles
assert "g3t_prefill_plain(c, ordinal)" in roles
for witness in (
    "manual decode no-prefix rejected",
    "manual decode f32 reservation failure",
    "manual decode dead input rejected",
    "manual decode wrong length metadata rejected",
    "manual decode malformed typed I1 rejected",
    "manual decode borrowed I1 rejected",
    "manual decode borrowed prefix rejected",
    "manual decode stale binding rejected",
    "manual decode copied input can release",
    "manual decode staged K/V position one bit-exact to M3T",
    "manual decode A2 attention failure",
    "manual decode all 256 logits bit-exact to M3T row one",
    "manual decode both committed K/V positions bit-exact to M3T",
    "manual decode at length two rejected",
    "manual decode old results survive generator close",
    "manual decode P1/G0 predecessor committed",
    "manual decode synthetic provider cut retryable",
    "manual decode staged append abort preserves prefix",
    "manual decode P1/G0 retry parity and no draw",
):
    assert witness in test, witness
assert "-DET_G3T_MANUAL_DECODE_PRIVATE" in runner
assert "native/g3t_manual_decode_source_closure.txt" in runner
assert "scripts/check-g3t-manual-decode.py" in runner
print("G3-T private manual decode native source contract: PASS")
