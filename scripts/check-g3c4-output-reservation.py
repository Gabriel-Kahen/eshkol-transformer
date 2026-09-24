#!/usr/bin/env python3
"""Structural checks for the G3-C4 Step 19A native output reservation."""

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
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-g3c4-prompt-prefill.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_output_reservation.c").read_text()
    runner = (ROOT / "scripts/test-g3c4-output-reservation.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_OUTPUT_RESERVATION_STEP19A_CONTRACT.md").read_text()

    require("ET_G3C4_OUTPUT_RESERVATION_PRIVATE requires Step 18A" in source,
            "reservation macro does not require accepted Step 18A")
    for symbol in ("et_g3c4_private_output_reserve_v1",
                   "et_g3c4_private_output_release_v1"):
        require(source.count(symbol) == 1 and header.count(symbol) == 1,
                f"source-private output boundary changed: {symbol}")

    record_start = source.index("typedef struct et_g3c4_output_internal")
    record_end = source.index("} et_g3c4_output_internal;", record_start)
    record = source[record_start:record_end]
    ordered(record, [
        "et_g3c4_transport_header_internal transport",
        "et_g3c4_context_internal *parent_ctx", "int64_t prompt_length",
        "int64_t generated_length", "et_i64_tensor *ids", "int64_t length",
        "int64_t cache_length", "int64_t rng[4]", "uint32_t numeric_ready",
        "uint32_t ids_copied", "uint32_t text_ready",
    ], "exact native output payload")
    require("sizeof(et_g3c4_output_internal) == 128u" in source,
            "output owner size is not sealed")

    reserve = c_function(source, "void *et_g3c4_private_output_reserve_v1")
    ordered(reserve, [
        "et_g3c4_admit_active_call(context_candidate)",
        "context->call_kind != 2", "context->budget != 0",
        "prompt_length != 1", "prompt_length + context->budget > 2",
        "et_g3c4_pending_output_lookup(context, &pending)",
        "et_g3c4_output_allocate()", "ids_shape[0] = (uint64_t)context->budget",
        "et_i64_tensor_create_v1(", "output->transport.magic",
        "output->parent_ctx = context", "output->prompt_length = prompt_length",
        "output->generated_length = context->budget",
        "et_g3c4_enroll_output(output)",
    ], "reserve-before-publish construction")
    for forbidden in ("generator_rng_words", "output->length =",
                      "output->numeric_ready =", "output->text_ready ="):
        require(forbidden not in reserve,
                f"reservation fabricates prepared state: {forbidden}")

    execute = c_function(source, "int64_t et_g3c4_private_prompt_prefill_v1")
    ordered(execute, [
        "input->length + context->budget > 2",
        "et_g3c4_pending_output_lookup(context, &pending_output)",
        "pending_output == NULL",
        "pending_output->prompt_length != input->length",
        "pending_output->generated_length != context->budget",
        "et_g3c4_prompt_prefill_borrow(input, &borrow, &view)",
    ], "reservation-gated prompt prefill")

    abort = c_function(source, "int64_t et_g3c4_private_call_abort_v1")
    ordered(abort, [
        "et_g3c4_pending_output_lookup(context, &pending_output)",
        "et_g3c4_cache_idle_preflight(context)",
        "et_g3c4_output_discard_pending(pending_output)",
        "et_g3c4_active_call_drain(context)",
    ], "abort output destruction")
    discard = c_function(source, "static int64_t et_g3c4_output_discard_pending")
    ordered(discard, [
        "et_i64_tensor_destroy_v1(&output->ids, &error)",
        "output->parent_ctx = NULL", "output->prompt_length = 0",
        "output->generated_length = 0", "memset(output->rng, 0",
        "output->numeric_ready = 0u", "output->ids_copied = 0u",
        "output->text_ready = 0u",
        "output->transport.state = ET_G3C4_CONTEXT_DEAD",
    ], "authenticated tombstone scrub")

    for phrase in (
        "reserved_prefill_abort(owner, 2, 0)",
        "reserved_prefill_abort(owner, 1, 1)", "admission_and_linkage",
        "allocation_and_borrow_cuts", "prefill1_dispatches == 21u",
        "prefill2_dispatches == 21u", "g0_cuts == 3u", "g1_cuts == 4u",
        "ET_I64_TENSOR_CODE_ACTIVE_BORROW", "check_dead_output",
        "et_a2_kv_cache_read_borrow_begin_v1", "owner-cuts=1",
        "borrow-cuts=3",
    ):
        require(phrase in test, f"focused witness omits: {phrase}")
    for phrase in (
        "compile_mode normal", "compile_mode sanitize", "detect_leaks=1",
        "runtime-repeat.stdout", "added-defined.txt", "added-undefined.txt",
        "invalid-output-only", "--wrap=et_kernel_runtime_dispatch",
    ):
        require(phrase in runner, f"runner omits: {phrase}")
    for phrase in (
        "not a public generation API", "before enrolling either object",
        "prerequisite for the first numerical write", "retains the committed prompt cache",
        "no Eshkol output shell", "numeric output preparation", "EOS behavior",
        "generation loop", "package export",
    ):
        require(phrase in contract, f"contract omits boundary: {phrase}")

    prior = (ROOT / "native/g3c4_prompt_prefill_source_closure.txt").read_text().splitlines()
    expected = prior + [
        "native/g3c4_prompt_prefill_source_closure.txt",
        "tests/g3c4/test_output_reservation.c",
        "scripts/check-g3c4-output-reservation.py",
        "scripts/test-g3c4-output-reservation.sh",
        "docs/g3/G3_C4_OUTPUT_RESERVATION_STEP19A_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_output_reservation_source_closure.txt").read_text().splitlines()
    require(actual == expected, "output reservation closure is not exact Step 18A extension")
    require(len(actual) == len(set(actual)), "output reservation closure contains duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 output reservation static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private output reservation contract: PASS")
