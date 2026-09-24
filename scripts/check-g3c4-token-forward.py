#!/usr/bin/env python3
"""Fast structural checks for the G3-C4 Step 11A token-forward leaf."""

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
        [sys.executable, str(ROOT / "scripts/check-g3c4-token-frame.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_token_forward.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_TOKEN_FORWARD_STEP11A_CONTRACT.md").read_text()
    runner = (ROOT / "scripts/test-g3c4-token-forward.sh").read_text()

    require("ET_G3C4_TOKEN_FORWARD_PRIVATE requires Step 10A" in source,
            "token-forward macro does not require the accepted frame leaf")
    symbol = "et_g3c4_private_token_forward_v1"
    require(source.count(symbol) == 1 and header.count(symbol) == 1,
            "source-private token-forward boundary changed")
    start = source.index(f"int64_t {symbol}")
    end = source.index("\n#endif", start)
    body = source[start:end]

    ordered(source[source.index("int64_t et_g3c4_private_token_frame_begin_v1"):start], [
        "et_g3c4_private_sample_last_v1(",
        "et_a2_kv_cache_read_borrow_begin_v1(",
        "token_position = *(const int64_t *)committed_lengths->data",
        "et_a2_kv_cache_read_borrow_end_v1(",
        "et_a2_kv_cache_transaction_begin_v1(",
        "context->token_frame_position = token_position",
    ], "pre-transaction position snapshot")
    ordered(body, [
        "speculative_token != context->token_frame_candidate",
        "et_g3n_kernel_provider_v1()",
        "et_n2_kernel_provider_v1()",
        '"g3n.embedding-forward"',
        '"g3c4.embedding-forward"',
        "ET_G3C4_TOKEN_LINEAR(2u, scratch.n1, scratch.qt",
        "ET_G3C4_TOKEN_LINEAR(0u, scratch.n1, scratch.kt",
        "ET_G3C4_TOKEN_LINEAR(3u, scratch.n1, scratch.vt",
        '"g3n.heads.split.forward"',
        "et_g3c4_private_token_frame_stage_v1(",
        "et_a2_kv_cache_transaction_view_begin_v1(",
        "*(const int64_t *)effective_lengths->data != query_position + 1",
        '"g3c4.causal-attention.forward"',
        "et_a2_kv_cache_transaction_view_end_v1(",
        '"g3n.heads.merge.forward"',
        '"gelu.forward"',
        "ET_G3C4_TOKEN_LINEAR(10u, scratch.nf, scratch.z",
        "memcpy(logits_output, scratch.z, sizeof(scratch.z))",
    ], "one-token numerical schedule")
    ordered(body[body.index("fail:"):], [
        "et_g3c4_error_snapshot_internal()",
        "et_a2_kv_cache_transaction_view_end_v1(",
        "et_g3c4_token_frame_discard(context)",
        "et_kernel_runtime_destroy(n2_runtime)",
        "et_kernel_runtime_destroy(g3n_runtime)",
        "et_g3c4_error_restore_internal(first)",
    ], "failure cleanup")
    require("et_a2_kv_cache_transaction_commit_v1" not in body,
            "token forward publishes its pending cache transaction")
    require("et_g3c4_private_token_frame_publish_v1" not in body,
            "token forward publishes token/RNG/budget")

    for phrase in [
        "expected_capabilities[21]", "expected_operations[21]",
        "cross_provider_prefix_parity", "reference + position * 256u",
        "dispatch_failure_cuts", "fail_forward_at", "check_snapshot",
        "check_k1_error(", "ET_KERNEL_CODE_PROVIDER_REJECTED",
        "ET_KERNEL_CODE_ALLOCATION_FAILED",
        "begin_failure_cuts == 8u",
        "et_m3t_test_k1_fail_after(0u)",
        "et_a2_kv_cache_test_fail_alloc_after_v1(8u)",
        "output[index] == -123.0f", "roles=21 cuts=31 positions=2",
    ]:
        require(phrase in test, f"focused test omits {phrase}")
    require("--wrap=et_kernel_runtime_dispatch" in runner,
            "runner does not interpose actual provider dispatch")
    for provider in ["native/g3n_primitives_provider.c",
                     "native/n2_primitives_provider.c",
                     "native/g3c4_primitives_provider.c"]:
        require(provider in runner, f"runner omits accepted provider: {provider}")
    require("No single accepted provider" in contract and
            "bit-exact against all four rows" in contract and
            "adds no prompt ingestion" in contract,
            "contract omits dependency, numerical, or public boundary")

    prior = (ROOT / "native/g3c4_token_frame_source_closure.txt").read_text().splitlines()
    expected = prior + [
        "native/g3c4_token_frame_source_closure.txt",
        "tests/g3c4/test_token_forward.c",
        "scripts/check-g3c4-token-forward.py",
        "scripts/test-g3c4-token-forward.sh",
        "docs/g3/G3_C4_TOKEN_FORWARD_STEP11A_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_token_forward_source_closure.txt").read_text().splitlines()
    require(actual == expected,
            "token-forward closure is not the exact Step 10A successor")
    require(len(actual) == len(set(actual)), "closure has duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 token-forward static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private token-forward contract: PASS")
