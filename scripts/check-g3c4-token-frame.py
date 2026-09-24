#!/usr/bin/env python3
"""Fast structural checks for the G3-C4 Step 10A token-frame leaf."""

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


def body(source, symbol, next_symbol=None):
    start = source.index(f"int64_t {symbol}")
    end = source.index(
        f"\nint64_t {next_symbol}" if next_symbol else "\n#endif", start)
    return source[start:end]


def check():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-g3c4-sampler-transport.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_token_frame_commit.c").read_text()
    contract = (ROOT / "docs/g3/G3_C4_TOKEN_FRAME_STEP10A_CONTRACT.md").read_text()
    runner = (ROOT / "scripts/test-g3c4-token-frame.sh").read_text()

    require("ET_G3C4_TOKEN_FRAME_PRIVATE requires Step 9A" in source,
            "token-frame macro does not require the accepted sampler leaf")
    symbols = [
        "et_g3c4_private_token_frame_begin_v1",
        "et_g3c4_private_token_frame_stage_v1",
        "et_g3c4_private_token_frame_publish_v1",
        "et_g3c4_private_token_frame_abort_v1",
    ]
    for symbol in symbols:
        require(source.count(symbol) == 1 and header.count(symbol) == 1,
                f"source-private token-frame boundary changed: {symbol}")

    begin = body(source, symbols[0], symbols[1])
    stage = body(source, symbols[1], symbols[2])
    publish = body(source, symbols[2], symbols[3])
    abort = body(source, symbols[3])
    ordered(begin, [
        "context->call_kind != 2 || context->budget != 1",
        "et_g3c4_private_sample_last_v1(",
        "et_a2_kv_cache_transaction_begin_v1(",
        "context->token_frame_transaction = transaction",
        "context->token_frame_state = ET_G3C4_TOKEN_FRAME_SAMPLED",
        "memcpy(speculative_token_output",
    ], "sampled-frame acquisition")
    ordered(stage, [
        "speculative_token != context->token_frame_candidate",
        '"f32", 4u, kv_shape',
        "et_a2_kv_cache_transaction_stage_layer_v1(",
        "et_g3c4_token_frame_discard(context)",
        "context->token_frame_state = ET_G3C4_TOKEN_FRAME_READY",
    ], "candidate-bound K/V staging")
    ordered(publish, [
        "ET_G3C4_TOKEN_FRAME_READY",
        "token_output, sizeof(*token_output), context, sizeof(*context)",
        "et_a2_kv_cache_transaction_commit_v1(",
        "memcpy(context->generator_rng_words",
        "memcpy(token_output",
        "context->budget = 0",
        "et_g3c4_token_frame_reset(context)",
    ], "joint publication tail")
    require("et_g3c4_token_frame_discard(context)" in abort,
            "explicit frame abort does not own transaction cleanup")
    require("et_g3c4_token_frame_discard(context);" in source[source.index(
            "int64_t et_g3c4_private_call_abort_v1"):],
            "active-call abort does not own pending frame cleanup")
    require("!et_g3c4_token_frame_idle(context)" in source[source.index(
            "int64_t et_g3c4_private_call_finish_v1"):source.index(
            "int64_t et_g3c4_private_call_abort_v1")],
            "call finish does not reject a pending frame")
    require("token_frame_successor[1] ==" in source and
            "context->generator_rng_words[1]" in source,
            "frame successor is not bound to the current RNG seed")

    for forbidden in ["full_prefix_forward_v1(", "position", "result_publish",
                      "tokenizer", "public_generate"]:
        require(forbidden not in begin + stage + publish + abort,
                f"token frame invents a deferred producer/facade: {forbidden}")
    for phrase in [
        "categorical_commit", "greedy_commit", "failure_cuts", "fail_sampler",
        "et_a2_kv_cache_test_fail_alloc_after_v1", "keys[0] = NAN",
        "context->generator_rng_words[0]", "check_cache",
        "et_g3c4_private_call_finish_v1(context) != 0",
    ]:
        require(phrase in test, f"focused test omits {phrase}")
    require("--wrap=et_kernel_runtime_dispatch" in runner,
            "runner does not interpose the actual sampler dispatch")
    require("fixed at `T=4`" in contract and
            "current C4 A2 cache\ncapacity is also four" in contract,
            "contract omits the direct-composition counterexample")
    require("does not implement the missing one-token numerical forward" in contract,
            "contract overclaims numerical composition")

    prior = (ROOT / "native/g3c4_sampler_transport_source_closure.txt"
             ).read_text().splitlines()
    expected = prior + [
        "native/g3c4_sampler_transport_source_closure.txt",
        "tests/g3c4/test_token_frame_commit.c",
        "scripts/check-g3c4-token-frame.py",
        "scripts/test-g3c4-token-frame.sh",
        "docs/g3/G3_C4_TOKEN_FRAME_STEP10A_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_token_frame_source_closure.txt"
              ).read_text().splitlines()
    require(actual == expected,
            "token-frame closure is not the exact Step 9A successor")
    require(len(actual) == len(set(actual)), "closure has duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 token-frame static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private token-frame contract: PASS")
