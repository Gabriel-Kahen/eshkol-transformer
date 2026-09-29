#!/usr/bin/env python3
"""Static boundary checks for the native-only C4 P2/G2 first-frame leaf."""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for predecessor in (
    "check-g3c4-output-reservation.py",
    "check-g3c4-p2g1-pending.py",
):
    subprocess.run([sys.executable, str(root / "scripts" / predecessor)],
                   cwd=root, check=True, stdout=subprocess.DEVNULL)

owner = (root / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
fixture = (root / "tests/g3c4/test_p2_g2_first_frame.c").read_text()
runner = (root / "scripts/test-g3c4-p2g2-first-frame.sh").read_text()
public = (root / "native/g3c4_public_generation_dispatch_extension.esk").read_text()
private_config = (root / "native/g3c4_call_entry_extension.esk").read_text()

for required in (
    "C4 P2/G2 first frame requires pending output and token forward",
    "#define ET_G3C4_PRIVATE_MAX_NEW 2",
    "#define ET_G3C4_PRIVATE_MAX_NEW 1",
    "#define ET_G3C4_P2_G2_TUPLE(prompt, budget)",
    "et_g3c4_p2g2_first_frame_preflight(context)",
    "context->token_frame_position != 2",
    "ET_G3C4_P2_G2_TUPLE(2, context->budget) && token_position != 2",
    "Budget two owns only a pending transcript",
    "context->call_kind != 2 || context->budget != 1 ||",
    "return et_g3c4_prefill2_impl(\n"
    "      candidate, token_ids, last_logits_output, NULL, NULL);",
    "owned_view->data != token_ids",
    "last_logits_output, input, view);",
):
    if required not in owner:
        raise SystemExit(f"P2/G2 first-frame guard missing: {required}")
if owner.count("et_g3c4_prefill2_impl(") != 3:
    raise SystemExit("P2/G2 owned-prompt prefill gained another internal caller")

for required in (
    "et_g3c4_private_prompt_prefill_preflight_v1(context, input, 2)",
    "et_g3c4_private_output_reserve_v1(context, 2)",
    "et_g3c4_private_token_frame_begin_last_v1(context, last, &token)",
    "et_g3c4_private_token_forward_v1(context, token, next)",
    "et_g3c4_private_output_prepare_v1(context, output)",
    "et_g3c4_private_token_frame_publish_v1(context, &untouched)",
    "et_g3c4_private_call_prepare_end_v1(context)",
    "et_g3c4_private_call_abort_v1(context)",
    "forged_token = 256", "malformed_input_view = mode",
    "print_oracle_case(", "reference_cache.keys", "reference_cache.values",
    "et_i64_tensor_borrow_begin_v1", "et_a2_kv_cache_read_borrow_begin_v1",
    "et_i64_tensor_test_fail_alloc_after_v1(0u)",
    "et_a2_kv_cache_test_fail_alloc_after_v1(0u)",
    "et_g3c4_private_prefill2_v1(context, prompt, logits)",
    "et_g3c4_private_prefill2_v1(context, ids, logits)",
):
    if required not in fixture:
        raise SystemExit(f"P2/G2 first-frame witness missing: {required}")

if "(= (vector-ref policy 5) 1)" not in public or \
        "(<= (vector-ref policy 5) 1)" not in private_config:
    raise SystemExit("public/private Eshkol one-token admission changed")
if "check_p2_g1_oracle.py" not in runner or \
        "cmp \"$tmp/off.symbols\" \"$tmp/on.symbols\"" not in runner:
    raise SystemExit("P2/G2 first-frame oracle or feature-off symbol gate missing")

print("G3-C4 P2/G2 first-frame static PASS: private native transcript only")
