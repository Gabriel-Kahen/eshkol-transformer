#!/usr/bin/env python3
"""Structural checks for G3-C4 Step 20A numeric output preparation."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ordered(text: str, fragments: list[str], label: str) -> None:
    position = -1
    for fragment in fragments:
        position = text.find(fragment, position + 1)
        require(position >= 0, f"{label} missing or misordered: {fragment}")


def c_function(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError(f"unterminated C function: {signature}")


def check() -> None:
    for predecessor in ("check-g3c4-output-reservation.py",
                        "check-g3c4-last-logit-frame.py",
                        "check-g3c4-token-forward.py"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / predecessor)],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_output_prepare.c").read_text()
    runner = (ROOT / "scripts/test-g3c4-output-prepare.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_OUTPUT_PREPARE_STEP20A_CONTRACT.md").read_text()

    require("ET_G3C4_OUTPUT_PREPARE_PRIVATE requires Steps 13A and 19A"
            in source, "prepare macro does not require accepted dependencies")
    symbol = "et_g3c4_private_output_prepare_v1"
    require(source.count(symbol) == 1 and header.count(symbol) == 1,
            "source-private prepare boundary changed")

    validator_start = source.index("if (header->kind == ET_G3C4_OUTPUT_KIND)")
    validator_end = source.index("return 0;\n  }", validator_start) + 13
    validator = source[validator_start:validator_end]
    ordered(validator, [
        "output->numeric_ready == 0u", "output->length == 0",
        "output->numeric_ready == 1u",
        "output->length == output->generated_length",
        "output->prompt_length + output->generated_length",
        "output->rng[0] == 1", "output->rng[1] >= 0",
    ], "pending numeric readiness states")

    cache = c_function(
        source, "static int et_g3c4_output_committed_cache_length")
    ordered(cache, [
        "et_a2_kv_cache_read_borrow_begin_v1(",
        "et_a2_kv_cache_read_borrow_layer_v1(",
        "length = *(const int64_t *)lengths->data",
        "et_a2_kv_cache_read_borrow_end_v1(&borrow, &error)",
        "*length_result = length",
    ], "scoped committed-length proof")

    prepare = c_function(source, "int64_t et_g3c4_private_output_prepare_v1")
    ordered(prepare, [
        "et_g3c4_admit_active_call(context_candidate)",
        "et_g3c4_admit_output(output_candidate, 0)",
        "et_g3c4_pending_output_lookup(context, &pending)",
        "pending != output", "output->generated_length != context->budget",
        "output->numeric_ready != 0u",
        "et_g3c4_prefill_binding_matches_pins(context)",
        "output->generated_length == 0",
        "et_g3c4_token_frame_idle(context)",
        "et_g3c4_output_committed_cache_length(",
        "committed_length != output->prompt_length",
        "context->token_frame_state != ET_G3C4_TOKEN_FRAME_READY",
        "context->token_frame_position != output->prompt_length",
        "token_candidate = context->token_frame_candidate",
        "et_i64_tensor_copy_from_v1(",
        "output->length = output->generated_length",
        "output->cache_length = committed_length",
        "memcpy(output->rng, result_rng",
        "output->numeric_ready = 1u",
    ], "G0/G1 numeric preparation")
    tail = prepare[prepare.index("et_i64_tensor_copy_from_v1("):]
    for forbidden in ("runtime_dispatch", "transaction_commit", "publish_v1",
                      "calloc(", "malloc("):
        require(forbidden not in tail,
                f"prepare tail adds publication or allocation: {forbidden}")

    for phrase in (
        "prepare_g0(owner)", "prepare_g1(owner)",
        "token_forward_dispatches == 21u", "cache_cuts == 3u",
        "g1_borrow_cut_and_admission", "ET_I64_TENSOR_CODE_ACTIVE_BORROW",
        "et_g3c4_private_call_acquire_v1(foreign_context, 2, 1)",
        "check_pending_output(foreign_output, foreign_context, 1, 1)",
        "ET_G3C4_CODE_STALE_BINDING", "saved_parameter",
        "output->length == output_length",
        "output->cache_length == output_cache_length",
        "memcmp(output->rng, output_rng", "output->ids == ids",
        "context->token_frame_transaction == transaction",
        "context->token_frame_position == position",
        "check_numeric_output", "ownership-cuts=3", "repeat-cuts=2",
        "readiness-cuts=4",
    ):
        require(phrase in test, f"focused witness omits: {phrase}")
    for phrase in (
        "compile_mode normal", "compile_mode sanitize", "detect_leaks=1",
        "runtime-repeat.stdout", "added-defined.txt", "added-undefined.txt",
        "invalid-prepare-only", "--wrap=et_kernel_runtime_dispatch",
    ):
        require(phrase in runner, f"runner omits: {phrase}")
    for phrase in (
        "not a public generation API", "exact I1 `[0]`",
        "accepted 21-dispatch token-forward route", "final recoverable operation",
        "generator RNG", "no ID decode staging", "live-output transition",
        "EOS behavior", "generation loop", "package export",
    ):
        require(phrase in contract, f"contract omits boundary: {phrase}")

    reservation = (ROOT / "native/g3c4_output_reservation_source_closure.txt").read_text().splitlines()
    last_logit = (ROOT / "native/g3c4_last_logit_frame_source_closure.txt").read_text().splitlines()
    dependencies = reservation + [path for path in last_logit if path not in reservation]
    additions = [
        "native/g3c4_output_reservation_source_closure.txt",
        "native/g3c4_last_logit_frame_source_closure.txt",
        "tests/g3c4/test_output_prepare.c",
        "scripts/check-g3c4-output-prepare.py",
        "scripts/test-g3c4-output-prepare.sh",
        "docs/g3/G3_C4_OUTPUT_PREPARE_STEP20A_CONTRACT.md",
    ]
    expected = dependencies + [path for path in additions if path not in dependencies]
    actual = (ROOT / "native/g3c4_output_prepare_source_closure.txt").read_text().splitlines()
    require(actual == expected, "output prepare closure is not the exact dependency union")
    require(len(actual) == len(set(actual)), "output prepare closure contains duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 output prepare static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private numeric output prepare contract: PASS")
