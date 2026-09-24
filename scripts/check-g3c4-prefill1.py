#!/usr/bin/env python3
"""Fast structural checks for the G3-C4 Step 16A P1 prefill leaf."""

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
    test = (ROOT / "tests/g3c4/test_prefill1.c").read_text()
    predecessor_test = (ROOT / "tests/g3c4/test_prefill3.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_PREFILL1_STEP16A_CONTRACT.md").read_text()
    runner = (ROOT / "scripts/test-g3c4-prefill1.sh").read_text()

    require("ET_G3C4_PREFILL1_PRIVATE requires Step 12A" in source,
            "P1 prefill macro does not require accepted Step 12A")
    symbol = "et_g3c4_private_prefill1_v1"
    require(source.count(symbol) == 1 and header.count(symbol) == 1,
            "source-private P1 prefill boundary changed")
    start = source.index(f"int64_t {symbol}")
    end = source.index("\n#endif\n#endif\n\n#ifdef ET_G3C4_LAST", start)
    body = source[start:end]
    ordered(body, [
        "token_id < 0 || token_id > 255",
        "et_g3c4_admit_active_call(candidate)",
        "et_g3c4_prefill1_reject_output_aliases(",
        "et_g3n_kernel_provider_v1()",
        "et_n2_kernel_provider_v1()",
        "et_a2_kv_cache_create_v1(",
        '"g3n.embedding-forward"',
        '"g3c4.embedding-forward"',
        "ET_G3C4_PREFILL1_LINEAR(2u, scratch.n1, scratch.qt",
        "ET_G3C4_PREFILL1_LINEAR(0u, scratch.n1, scratch.kt",
        "ET_G3C4_PREFILL1_LINEAR(3u, scratch.n1, scratch.vt",
        "et_a2_kv_cache_transaction_begin_v1(",
        "candidate_cache, 1u",
        "et_a2_kv_cache_transaction_stage_layer_v1(",
        "et_a2_kv_cache_transaction_view_begin_v1(",
        '"g3c4.causal-attention.forward"',
        "et_a2_kv_cache_transaction_view_end_v1(",
        '"gelu.forward"',
        "et_g3c4_model_pins_check_internal(",
        "et_a2_kv_cache_transaction_commit_v1(",
        "et_a2_kv_cache_destroy_v1(&old_cache",
        "context->cache = candidate_cache",
        "context->prefill_binding_ready = 1u",
        "memcpy(last_logits_output, scratch.z",
    ], "P1 prefill numerical and publication order")
    ordered(body[body.index("fail:"):], [
        "et_g3c4_error_snapshot_internal()",
        "et_a2_kv_cache_transaction_view_end_v1(",
        "et_a2_kv_cache_transaction_abort_v1(",
        "et_a2_kv_cache_destroy_v1(&candidate_cache",
        "et_kernel_runtime_destroy(n2_runtime)",
        "et_kernel_runtime_destroy(g3n_runtime)",
        "et_g3c4_error_restore_internal(first)",
    ], "P1 prefill rollback")
    require("generator_rng_words" not in body,
            "P1 prefill touches generator RNG")

    for phrase in [
        "expected_p1_capabilities[21]", "expected_p1_operations[21]",
        "numerical_parity_and_commit", "memcmp(actual, full",
        "prefill1_staged_keys", "dispatch_failure_cuts",
        "a2_allocation_cuts", "runtime_allocation_cuts",
        "alias_and_admission_rejections", "check_preserved",
        "roles=21 dispatch-cuts=21", "a2-allocation-cuts=11",
    ]:
        require(phrase in test or phrase in predecessor_test,
                f"focused test omits {phrase}")
    require("--wrap=et_kernel_runtime_dispatch" in runner,
            "runner does not interpose actual dispatch")
    for provider in ["native/g3n_primitives_provider.c",
                     "native/n2_primitives_provider.c",
                     "native/g3c4_primitives_provider.c"]:
        require(provider in runner, f"runner omits provider: {provider}")
    require("proves P1 numerical prefill only" in contract and
            "pre-acquire `P+G` admission" in contract and
            "adds no P2 execution" in contract,
            "contract does not preserve the bounded leaf")

    prior = (ROOT / "native/g3c4_prefill3_source_closure.txt").read_text().splitlines()
    expected = prior + [
        "native/g3c4_prefill3_source_closure.txt",
        "tests/g3c4/test_prefill1.c",
        "scripts/check-g3c4-prefill1.py",
        "scripts/test-g3c4-prefill1.sh",
        "docs/g3/G3_C4_PREFILL1_STEP16A_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_prefill1_source_closure.txt").read_text().splitlines()
    require(actual == expected,
            "P1 prefill closure is not the exact Step 12A successor")
    require(len(actual) == len(set(actual)), "closure has duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 P1 prefill static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private P1 prefill contract: PASS")
