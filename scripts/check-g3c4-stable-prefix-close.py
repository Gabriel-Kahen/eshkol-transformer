#!/usr/bin/env python3
"""Focused source guard for the private native stable-prefix close."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
fixture = (root / "tests/g3c4/test_p2_g2_stable_prefix_close.c").read_text()
contract = (root / "docs/g3/G3_R_C4_NATIVE_STABLE_PREFIX_CLOSE_CONTRACT.md").read_text()
assert source.count("et_g3c4_private_p2g2_stable_prefix_close_v1(") == 1
assert source.count("et_g3c4_private_p2g2_stable_prefix_check_v1(") == 1
start = source.index("int64_t et_g3c4_private_p2g2_stable_prefix_close_v1(")
end = source.index("int64_t et_g3c4_private_p2g2_stable_prefix_check_v1(")
close = source[start:end]
assert close.index("et_g3c4_p2g2_second_output_clean") < close.index("et_g3c4_output_discard_pending")
assert close.index("et_g3c4_stable_cache_capture") < close.index("et_g3c4_output_discard_pending")
assert close.index("et_g3c4_output_discard_pending") < close.index("et_g3c4_active_call_drain")
assert "if (et_g3c4_output_discard_pending(output) != 0) abort();" in close
assert "context->stable_prefix_ready = 1u;" in close
assert "context->stable_prefix_ready != 0u" in source[source.index("int64_t et_g3c4_private_call_acquire_v1("):]
for needle in (
    "held_ids", "held_cache", "owner->active == NULL", "stable_logits",
    "stable_keys", "stable_values", "fail_tail", "check_dead_output",
    "et_g3c4_private_call_acquire_v1(context, 2, 2)",
):
    assert needle in fixture, needle
assert "not a SAVE hold" in contract
assert "Eshkol" in contract
print("G3-C4 stable-prefix source guard PASS: checks=18")
