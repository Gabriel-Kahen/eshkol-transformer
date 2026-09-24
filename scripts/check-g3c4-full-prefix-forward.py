#!/usr/bin/env python3
"""Fast structural checks for the G3-C4 Step 8A full-prefix forward leaf."""

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
        [sys.executable, str(ROOT / "scripts/check-g3c4-provider-routes.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_full_prefix_forward.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_FULL_PREFIX_FORWARD_STEP8A_CONTRACT.md").read_text()

    require("ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE requires provider routes"
            in source, "full-prefix macro is not nested under provider routes")
    symbol = "et_g3c4_private_full_prefix_forward_v1"
    require(source.count(symbol) == 1 and header.count(symbol) == 1,
            "source-private full-prefix boundary changed")
    start = source.index(f"int64_t {symbol}")
    end = source.index("\n#endif", start)
    body = source[start:end]

    for pins in [
        "inputs[1] = context->pins.views[7];\n  inputs[2] = context->pins.views[6];",
        "inputs[1] = context->pins.views[9];\n  inputs[2] = context->pins.views[8];",
        "inputs[1] = context->pins.views[12];\n  inputs[2] = context->pins.views[11];",
    ]:
        require(pins in body, f"gamma/beta pin mapping changed: {pins}")
    for fragment in [
        "UINT32_C(0x3727c5ac)",
        "token_ids[query] < 0 || token_ids[query] > 255",
        "context->call_kind != 0 && context->call_kind != 2",
        "positions[key] <= positions[query]",
        "memcpy(logits, scratch.z, sizeof(scratch.z))",
    ]:
        require(fragment in body, f"fixed forward contract omits {fragment}")
    ordered(body, [
        "et_a2_kv_cache_transaction_begin_v1(",
        "et_a2_kv_cache_transaction_stage_layer_v1(",
        "et_a2_kv_cache_transaction_view_begin_v1(",
        "et_a2_kv_cache_transaction_view_tensors_v1(",
        '"g3c4.causal-attention.forward"',
        "et_a2_kv_cache_transaction_view_end_v1(",
        "et_a2_kv_cache_transaction_abort_v1(&transaction",
        "memcpy(logits, scratch.z, sizeof(scratch.z))",
    ], "candidate-cache lifetime")
    require("et_a2_kv_cache_transaction_commit_v1" not in body,
            "full-prefix leaf publishes the candidate cache")
    require("et_g3c4_sampler_runtime" not in body and "g3s." not in body,
            "full-prefix leaf reaches the sampler")

    require("expected_step expected[21]" in test,
            "dispatch transcript is not fixed at 21 steps")
    require("--wrap=et_kernel_runtime_dispatch" in
            (ROOT / "scripts/test-g3c4-full-prefix-forward.sh").read_text(),
            "runner does not interpose K1 dispatch")
    for phrase in ["dispatch_count == 21u", "failure_cuts >= 2u",
                   "check_cache_empty", "memcmp(first, second",
                   "memcmp(first, changed"]:
        require(phrase in test, f"focused test omits {phrase}")
    require("does not construct a T1 owner" in contract,
            "contract does not state the T1 boundary")

    prior = (ROOT / "native/g3c4_provider_routes_source_closure.txt"
             ).read_text().splitlines()
    expected_closure = prior + [
        "native/g3c4_provider_routes_source_closure.txt",
        "tests/g3c4/test_full_prefix_forward.c",
        "scripts/check-g3c4-full-prefix-forward.py",
        "scripts/test-g3c4-full-prefix-forward.sh",
        "docs/g3/G3_C4_FULL_PREFIX_FORWARD_STEP8A_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_full_prefix_forward_source_closure.txt"
              ).read_text().splitlines()
    require(actual == expected_closure,
            "full-prefix closure is not the exact Step 7A successor")
    require(len(actual) == len(set(actual)), "closure has duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 full-prefix static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 fixed full-prefix static contract: PASS")
