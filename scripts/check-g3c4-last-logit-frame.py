#!/usr/bin/env python3
"""Fast structural checks for the G3-C4 Step 13A last-logit adapter."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordered(text, fragments, label):
    position = -1
    for fragment in fragments:
        position = text.find(fragment, position + 1)
        require(position >= 0, f"{label} missing or misordered: {fragment}")


def check():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-g3c4-prefill3.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_last_logit_frame.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_LAST_LOGIT_FRAME_STEP13A_CONTRACT.md").read_text()
    runner = (ROOT / "scripts/test-g3c4-last-logit-frame.sh").read_text()

    require("ET_G3C4_LAST_LOGIT_FRAME_PRIVATE requires Step 12A" in source,
            "adapter macro does not require the accepted prefill leaf")
    symbol = "et_g3c4_private_token_frame_begin_last_v1"
    require(source.count(symbol) == 1 and header.count(symbol) == 1,
            "source-private last-logit boundary changed")
    start = source.index(f"int64_t {symbol}")
    end = source.index("\n#endif", start)
    body = source[start:end]
    ordered(body, [
        "et_g3c4_range_valid(last_logits, sizeof(private_logits))",
        "et_g3c4_ranges_overlap(\n          last_logits, sizeof(private_logits),\n          speculative_token_output",
        "et_g3c4_admit_active_call(candidate)",
        "et_g3c4_prefill_binding_matches_pins(context)",
        "context->owner, sizeof(*context->owner)",
        "for (index = 0u; index < 14u; index++)",
        "et_a2_kv_cache_private_storage_overlap_v1(\n          last_logits",
        "memcpy(private_logits, last_logits, sizeof(private_logits))",
        "et_kernel_runtime_dispatch(\n              et_g3c4_sampler_runtime",
        "et_a2_kv_cache_read_borrow_begin_v1(",
        "token_position = *(const int64_t *)committed_lengths->data",
        "et_a2_kv_cache_read_borrow_end_v1(",
        "et_a2_kv_cache_transaction_begin_v1(",
        "context->token_frame_transaction = transaction",
        "context->token_frame_state = ET_G3C4_TOKEN_FRAME_SAMPLED",
        "memcpy(speculative_token_output, &token_candidate",
    ], "direct sampling and frame installation")
    require("full_logits" not in body and "+ 3u * 256u" not in body,
            "adapter reconstructs or indexes a fabricated T4 carrier")
    require("et_a2_kv_cache_transaction_commit_v1" not in body,
            "adapter publishes the pending cache transaction")

    for phrase in [
        "exact_prefill_to_committed_token", "reference + 3u * 256u",
        "categorical_legacy_parity", "failures == 8u",
        "fail_adapter = 1", "ET_G3C4_CODE_STALE_BINDING", "last[0] = NAN",
        "et_g3c4_private_call_acquire_v1(context, 2, 0)",
        "for (size_t index = 0u; index < 14u; index++)",
        "cache_input_alias", "allocation-cuts=8",
    ]:
        require(phrase in test, f"focused test omits {phrase}")
    require("--wrap=et_kernel_runtime_dispatch" in runner,
            "runner does not interpose real provider dispatch")
    for phrase in [
        "does not fabricate", "Tokenizer identity and eval admission",
        "no raw tokenizer pointer", "prompt/result owner",
        "Public generation remains blocked",
    ]:
        require(phrase in contract, f"contract omits limit: {phrase}")

    prior = (ROOT / "native/g3c4_prefill3_source_closure.txt").read_text().splitlines()
    expected = prior + [
        "native/g3c4_prefill3_source_closure.txt",
        "tests/g3c4/test_last_logit_frame.c",
        "scripts/check-g3c4-last-logit-frame.py",
        "scripts/test-g3c4-last-logit-frame.sh",
        "docs/g3/G3_C4_LAST_LOGIT_FRAME_STEP13A_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_last_logit_frame_source_closure.txt").read_text().splitlines()
    require(actual == expected,
            "last-logit closure is not the exact Step 12A successor")
    require(len(actual) == len(set(actual)), "closure has duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 last-logit static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private last-logit frame contract: PASS")
