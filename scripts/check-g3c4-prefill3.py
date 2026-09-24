#!/usr/bin/env python3
"""Fast structural checks for the G3-C4 Step 12A prefill leaf."""

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
        [sys.executable, str(ROOT / "scripts/check-g3c4-token-forward.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_prefill3.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_PREFILL3_STEP12A_CONTRACT.md").read_text()
    runner = (ROOT / "scripts/test-g3c4-prefill3.sh").read_text()

    require("ET_G3C4_PREFILL3_PRIVATE requires Step 11A" in source,
            "prefill macro does not require the accepted token-forward leaf")
    require("requires the private A2 storage query" in source,
            "prefill macro does not require complete A2 alias admission")
    symbol = "et_g3c4_private_prefill3_v1"
    require(source.count(symbol) == 1 and header.count(symbol) == 1,
            "source-private prefill boundary changed")
    start = source.index(f"int64_t {symbol}")
    end = source.index("\n#endif", start)
    body = source[start:end]
    ordered(body, [
        "et_n2_kernel_provider_v1()",
        "et_a2_kv_cache_create_v1(",
        '"g3c4.embedding.forward"',
        "ET_G3C4_PREFILL3_RESIDUAL(scratch.et, scratch.ep, scratch.x)",
        "ET_G3C4_PREFILL3_LINEAR(2u, scratch.n1, scratch.qt",
        "ET_G3C4_PREFILL3_LINEAR(0u, scratch.n1, scratch.kt",
        "ET_G3C4_PREFILL3_LINEAR(3u, scratch.n1, scratch.vt",
        "et_a2_kv_cache_transaction_begin_v1(",
        "et_a2_kv_cache_transaction_stage_layer_v1(",
        "et_a2_kv_cache_transaction_view_begin_v1(",
        '"g3c4.causal-attention.forward"',
        "et_a2_kv_cache_transaction_view_end_v1(",
        '"g3c4.gelu.forward"',
        "et_g3c4_model_pins_check_internal(",
        "et_a2_kv_cache_transaction_commit_v1(",
        "et_a2_kv_cache_destroy_v1(&old_cache",
        "context->cache = candidate_cache",
        "context->prefill_binding_ready = 1u",
        "memcpy(last_logits_output, scratch.z + 2u * 256u",
    ], "three-token numerical schedule and publication tail")
    ordered(body[body.index("fail:"):], [
        "et_g3c4_error_snapshot_internal()",
        "et_a2_kv_cache_transaction_view_end_v1(",
        "et_a2_kv_cache_transaction_abort_v1(",
        "et_a2_kv_cache_destroy_v1(&candidate_cache",
        "et_kernel_runtime_destroy(n2_runtime)",
        "et_g3c4_error_restore_internal(first)",
    ], "candidate failure cleanup")
    require("prefill_binding_values[4768]" in source,
            "exact native parameter-bit snapshot is absent")
    require("et_g3c4_prefill_binding_matches_pins(context)" in source,
            "continuation does not authenticate the snapshot")

    for phrase in [
        "expected_capabilities[21]", "expected_operations[21]",
        "reference + 2u * 256u", "reference + 3u * 256u",
        "replacement_and_dispatch_cuts", "allocation_cuts",
        "owned_alias_rejections", "failures == 11u",
        "context->pins.views[10].data", "(float *)(void *)context->cache",
        "cache_alias",
        "et_m3t_test_k1_fail_after(0u)",
        "ET_G3C4_CODE_STALE_BINDING", "check_preserved",
        "output[i] == -123.0f", "roles=21 dispatch-cuts=21 allocation-cuts=11",
    ]:
        require(phrase in test, f"focused test omits {phrase}")
    require("--wrap=et_kernel_runtime_dispatch" in runner,
            "runner does not interpose actual provider dispatch")
    for provider in ["native/n2_primitives_provider.c",
                     "native/g3c4_primitives_provider.c",
                     "native/a2_kv_cache.c"]:
        require(provider in runner, f"runner omits accepted provider: {provider}")
    for phrase in ["not the complete public binding", "tokenizer identity",
                   "cannot feed the accepted Step 9A/10A `[1024]`",
                   "no public generation loop"]:
        require(phrase in contract, f"contract omits limit: {phrase}")

    prior = (ROOT / "native/g3c4_token_forward_source_closure.txt").read_text().splitlines()
    expected = prior + [
        "native/g3c4_token_forward_source_closure.txt",
        "tests/g3c4/test_prefill3.c",
        "scripts/check-g3c4-prefill3.py",
        "scripts/test-g3c4-prefill3.sh",
        "docs/g3/G3_C4_PREFILL3_STEP12A_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_prefill3_source_closure.txt").read_text().splitlines()
    require(actual == expected,
            "prefill closure is not the exact Step 11A successor")
    require(len(actual) == len(set(actual)), "closure has duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 prefill3 static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private prefill3 contract: PASS")
