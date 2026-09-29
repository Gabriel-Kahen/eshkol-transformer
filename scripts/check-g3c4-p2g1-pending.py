#!/usr/bin/env python3
"""Source-private C4 P2/G1 pending-transcript closure checks."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
owner = (root / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
envelope = (root / "native/g3c4_output_envelope_extension.esk").read_text()
extension = (root / "native/g3c4_p2g1_pending_extension.esk").read_text()
fixture = (root / "tests/g3c4/test_p2_g1_pending.c").read_text()
reservation = (root / "tests/g3c4/p2_g1_pending_reservation_test.esk").read_text()
contract = (root / "docs/g3/G3_C4_P2_G1_PROPOSAL.md").read_text()

assert "#define ET_G3C4_PENDING_TOTAL_LIMIT 3" in owner
assert "#define ET_G3C4_PENDING_TOTAL_LIMIT 2" in owner
assert owner.count("ET_G3C4_PENDING_TOTAL_LIMIT") == 7
assert "ET_G3C4_P2_G1_PENDING_PRIVATE requires pending output" in owner
assert "et_i64_tensor_stride_bytes_at_v1(" in owner
assert "stride0 != (size_t)input->length * sizeof(int64_t)" in owner
assert "stride1 != sizeof(int64_t)" in owner
assert "(define g3t-p2g1-pending-gate (vector #f))" in envelope
assert "(= prompt-length 2) (= generated 1)" in envelope
assert "(vector-set! g3t-p2g1-pending-gate 0 #t)" in extension
assert "provide" not in extension
for phrase in (
    "et_g3c4_private_prompt_prefill_v1(context, input, last)",
    "et_g3c4_private_token_frame_begin_last_v1(context, last, &token)",
    "et_g3c4_private_token_forward_v1(context, token, next)",
    "et_g3c4_private_prefill3_v1(",
    "memcmp(next, reference_logits, sizeof(next))",
    "et_g3c4_private_call_prepare_end_v1(context)",
    "et_g3c4_private_call_abort_v1(context)",
    "malformed_input_view", "forged_token", "prompt_failure_cuts",
    "rejected_budget_two", "check_dead_output", "print_oracle_case",
):
    assert phrase in fixture, phrase
assert "feature-off P2/G1 remains rejected" in reservation
assert "pending output still rejects call end" in reservation
assert "P3 prefill makes no sampling draw" in contract
assert "no call finish, live output" in contract
print("G3-C4 P2/G1 pending static PASS: tuple guards=5 private reservation=1")
