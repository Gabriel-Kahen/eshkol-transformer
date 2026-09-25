#!/usr/bin/env python3
"""Closed source contract for the conditional manual P1 transport witness."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_manual_logits_source_closure.txt").read_text().splitlines()
added = [
    "native/g3t_manual_p1_extension.esk",
    "scripts/check-g3t-manual-p1-prefill.py",
    "docs/g3/G3_T_MANUAL_P1_PREFILL_LEAF.md",
]
closure = (root / "native/g3t_manual_p1_prefill_source_closure.txt").read_text().splitlines()
assert closure == base + added
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))

native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
roles = (root / "src/eshkol_transformer/g3t_prefill_roles.inc").read_text()
wrapper = (root / "native/g3t_manual_p1_extension.esk").read_text()
transport = (root / "native/g3t_prefill_sample_extension.esk").read_text()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()

for stem in (
    "et_g3t_private_frame_begin_v1", "et_g3t_private_frame_prepare_v1",
    "et_g3t_private_call_prepare_end_v1", "et_g3t_private_frame_commit_v1",
    "et_g3t_private_call_finish_v1",
):
    assert stem in native
for phrase in (
    "ET_G3T_MANUAL_P1_PREFILL_PRIVATE", "c->call_kind == 0",
    "c->pending_logits->h.state != G3T_PENDING", "input->length != 1",
    "et_a2_kv_cache_create_v1", "c->frame.next_ordinal != 21",
    "et_f32_tensor_copy_bits_from_v1", "g3t_manual_p1_preflight(c)",
    "et_a2_kv_cache_transaction_view_begin_v1",
    "et_a2_kv_cache_read_borrow_begin_v1", "c->binding_ready = 1",
    "logits->h.state = G3T_LIVE", "c->pending_logits = NULL",
):
    assert phrase in native, phrase
assert "g3t_prefill_attention(c)" in roles
assert "g3t_prefill_plain(c, ordinal)" in roles
for name in ("g3t-logits-reserve", "g3t-call-prepare-end", "g3t-call-finish"):
    assert f"(define ({name} " in wrapper
assert wrapper.index("(cons entry") < wrapper.index("(g3t-native-logits-reserve")
assert "(eq? kind 'logits)" in transport
assert "(if logits-entry (vector-ref logits-entry 3) #f)" in transport
for witness in (
    "manual P1 frame KH/VH bit parity with M3T",
    "manual P1 all 256 logits bit-exact to M3T row zero",
    "manual P1 committed K/V bit-exact to M3T",
    "manual P1 result detached from parents",
    "manual P1 wrong P2 input rejected before candidate",
    "manual P1 candidate allocation failure atomic",
    "manual P1 wrong ordinal rejected without advance",
    "manual P1 wrong-kind staged result rejected",
    "manual P1 borrowed logits block prepare",
    "manual P1 borrowed old cache blocks prepare-end",
    "manual P1 stale binding blocks prepare-end",
    "manual P1 final abort rolls back staged result",
):
    assert witness in test, witness
assert '-DET_G3T_MANUAL_P1_PREFILL_PRIVATE' in runner
assert 'native/g3t_manual_p1_prefill_source_closure.txt' in runner
print("G3-T manual P1 prefill source contract: PASS")
