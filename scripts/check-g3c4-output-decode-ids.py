#!/usr/bin/env python3
"""Structural checks for G3-C4 Step 21A private output ID staging."""

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
        [sys.executable, str(ROOT / "scripts/check-g3c4-output-prepare.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_output_decode_ids.c").read_text()
    runner = (ROOT / "scripts/test-g3c4-output-decode-ids.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_OUTPUT_DECODE_IDS_STEP21A_CONTRACT.md").read_text()

    require("ET_G3C4_OUTPUT_DECODE_IDS_PRIVATE requires Step 20A and the private I1 storage query" in source,
            "decode-ID macro does not require accepted Step20A")
    i1_source = (ROOT / "native/i64_tensor.c").read_text()
    i1_header = (ROOT / "include/eshkol_transformer/i64_tensor.h").read_text()
    i1_symbol = "et_i64_tensor_private_storage_overlap_v1"
    require(i1_source.count(i1_symbol) == 1 and i1_header.count(i1_symbol) == 1,
            "source-private I1 storage query boundary changed")
    symbol = "et_g3c4_private_output_copy_decode_ids_v1"
    require(source.count(symbol) == 1 and header.count(symbol) == 1,
            "source-private output ID staging boundary changed")

    validator_start = source.index("if (header->kind == ET_G3C4_OUTPUT_KIND)")
    validator_end = source.index("return 0;\n  }", validator_start) + 13
    validator = source[validator_start:validator_end]
    ordered(validator, [
        "output->numeric_ready == 0u", "output->ids_copied == 0u",
        "output->numeric_ready == 1u",
        "output->ids_copied == 0u || output->ids_copied == 1u",
        "output->length == output->generated_length",
    ], "pending ID readiness states")

    aliases = c_function(
        source, "static int et_g3c4_output_decode_owned_alias")
    ordered(aliases, [
        "et_g3c4_transport_registry", "et_g3c4_transport_record_bytes(header)",
        "context->owner", "context->pins.views[index]",
        "et_i64_tensor_private_storage_overlap_v1(",
        "et_a2_kv_cache_private_storage_overlap_v1(",
    ], "owned carrier alias rejection")

    copy = c_function(
        source, "int64_t et_g3c4_private_output_copy_decode_ids_v1")
    ordered(copy, [
        "et_g3c4_admit_active_call(context_candidate)",
        "et_g3c4_admit_output(output_candidate, 0)",
        "et_g3c4_pending_output_lookup(context, &pending)",
        "pending != output", "output->numeric_ready != 1u",
        "output->ids_copied != 0u",
        "et_g3c4_prefill_binding_matches_pins(context)",
        "payload_bytes =", "et_g3c4_range_valid(staging_header",
        "et_g3c4_output_decode_owned_alias(",
        "et_i64_tensor_borrow_begin_v1(",
        "et_i64_tensor_borrow_view_v1(",
        "view->shape[0] !=", "view->byte_length != payload_bytes",
        "staging_header, carrier_bytes, view->data",
        "memcpy(&declared_bytes, staging_header",
        "declared_bytes < 0", "token < 0 || token > 255",
        "et_i64_tensor_borrow_end_v1(&borrow, &error)",
        "memcpy((unsigned char *)staging_header",
        "output->ids_copied = 1u",
    ], "atomic little-endian ID staging")
    tail = copy[copy.index("et_i64_tensor_borrow_end_v1(&borrow, &error)"):]
    for forbidden in ("calloc(", "malloc(", "runtime_dispatch",
                      "transaction_commit", "generator_rng_words ="):
        require(forbidden not in tail,
                f"ID staging tail adds a fallible/publication step: {forbidden}")

    for phrase in (
        "g0_route_and_aliases(owner)", "g1_failures_and_success(owner)",
        "borrow_allocation_cuts(owner)", "allocation_cuts == 1u",
        "ET_I64_TENSOR_CODE_ACTIVE_BORROW", "ET_G3C4_CODE_STALE_BINDING",
        "ET_G3C4_CODE_ALIAS", "invalid_id = 256",
        "alias_input->tensor", "alias_view->data",
        "alias_tokens[1]", "et_i64_tensor_borrow_end_v1(&alias_borrow",
        "foreign_context, prepared.output", "check_staging_unchanged",
        "UINTPTR_MAX - 7u", "context->token_frame_transaction ==",
        "context->token_frame_position == prepared->position",
        "readiness-cuts=2", "malformed-cuts=5", "repeat-cuts=1",
    ):
        require(phrase in test, f"focused witness omits: {phrase}")
    for phrase in (
        "compile_mode normal", "compile_mode sanitize", "detect_leaks=1",
        "runtime-repeat.stdout", "added-defined.txt", "added-undefined.txt",
        "invalid-decode-only", "expected-added-undefined.txt",
        "i1-added-defined.txt", "i1-added-undefined.txt",
        "--wrap=et_kernel_runtime_dispatch",
    ):
        require(phrase in runner, f"runner omits: {phrase}")
    for phrase in (
        "not T1 decoding", "exact output I1 `[G]`", "little-endian bytes",
        "before any caller byte is written", "active I1 borrow",
        "no T1 registry/core authentication", "output_accept_text",
        "EOS behavior", "package export", "public generator",
    ):
        require(phrase in contract, f"contract omits boundary: {phrase}")

    dependency = (ROOT / "native/g3c4_output_prepare_source_closure.txt").read_text().splitlines()
    additions = [
        "native/g3c4_output_prepare_source_closure.txt",
        "tests/g3c4/test_output_decode_ids.c",
        "scripts/check-g3c4-output-decode-ids.py",
        "scripts/test-g3c4-output-decode-ids.sh",
        "docs/g3/G3_C4_OUTPUT_DECODE_IDS_STEP21A_CONTRACT.md",
    ]
    expected = dependency + [path for path in additions if path not in dependency]
    actual = (ROOT / "native/g3c4_output_decode_ids_source_closure.txt").read_text().splitlines()
    require(actual == expected, "output decode-ID closure is not the exact dependency union")
    require(len(actual) == len(set(actual)), "output decode-ID closure contains duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 output decode-ID static check failed: {error}",
              file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private output ID staging contract: PASS")
