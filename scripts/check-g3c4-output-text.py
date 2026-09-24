#!/usr/bin/env python3
"""Structural checks for G3-C4 Step 21B raw decode/text readiness."""

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


def scheme_definition(text: str, signature: str) -> str:
    start = text.index(signature)
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError(f"unterminated Scheme definition: {signature}")


def check() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-g3c4-output-decode-ids.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    extension_path = ROOT / "native/g3c4_t1_output_decode_extension.esk"
    extension = extension_path.read_text()
    native_test = (ROOT / "tests/g3c4/test_output_text.c").read_text()
    t1_test = (ROOT / "tests/g3c4/t1_output_decode_test.esk").read_text()
    runner = (ROOT / "scripts/test-g3c4-output-text.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_OUTPUT_TEXT_STEP21B_CONTRACT.md").read_text()

    require("ET_G3C4_OUTPUT_TEXT_PRIVATE requires Step 21A output ID staging" in source,
            "text macro does not require Step21A")
    symbol = "et_g3c4_private_output_accept_text_v1"
    require(source.count(symbol) == 1 and header.count(symbol) == 1,
            "source-private text acceptance boundary changed")
    require("provide" not in extension and "extern" not in extension,
            "T1 extension exposes a package/native boundary")

    decoder = scheme_definition(
        extension, "(define (t1-private-g3-decode-raw-into!")
    ordered(decoder, [
        "(t1-private-entry tokenizer)", "(eq? (car entry) tokenizer)",
        "(t1-core-tokenizer? core)", "(eq? (vector-ref core 1) 'raw)",
        "(null? (vector-ref core 2))", "(null? (vector-ref core 3))",
        "(null? (vector-ref core 4))",
        "(= (t1-tokenizer-core-vocab-size core) 256)",
        "(bytevector? staging)", "(bytevector? raw)",
        "(or (= staging-length 0) (= staging-length 8))",
        "(= raw-length (quotient staging-length 8))",
        "(let validate-upper ((index 1))",
        "(bytevector-u8-ref staging index)",
        "(bytevector-u8-set!", "(bytevector-u8-ref staging 0)",
    ], "authenticated allocation-free T1 decode")
    for forbidden in ("make-bytevector", "bytevector-copy", "cons", "list ",
                      "vector ", "t1-private-register-core"):
        require(forbidden not in decoder,
                f"T1 valid decoder adds allocation/enrollment: {forbidden}")

    validator_start = source.index("if (header->kind == ET_G3C4_OUTPUT_KIND)")
    validator_end = source.index("return 0;\n  }", validator_start) + 13
    validator = source[validator_start:validator_end]
    ordered(validator, [
        "output->text_ready > 1u", "output->numeric_ready == 0u",
        "output->text_ready == 0u", "output->numeric_ready == 1u",
        "output->text_ready == 1u && output->ids_copied == 1u",
    ], "closed text readiness states")

    accept = c_function(
        source, "int64_t et_g3c4_private_output_accept_text_v1")
    ordered(accept, [
        "et_g3c4_admit_active_call(context_candidate)",
        "et_g3c4_admit_output(output_candidate, 0)",
        "et_g3c4_pending_output_lookup(context, &pending)",
        "pending != output", "output->numeric_ready != 1u",
        "output->ids_copied != 1u", "output->text_ready != 0u",
        "et_g3c4_prefill_binding_matches_pins(context)",
        "et_g3c4_range_valid(raw_header, carrier_bytes)",
        "et_g3c4_output_decode_owned_alias(",
        "memcpy(&declared_bytes, raw_header", "declared_bytes < 0",
        "et_i64_tensor_borrow_begin_v1(",
        "et_i64_tensor_borrow_view_v1(", "view->shape[0] !=",
        "view->byte_length != payload_bytes * sizeof(int64_t)",
        "et_g3c4_ranges_overlap(",
        "token < 0 || token > 255", "unsigned char *)raw_header",
        "et_i64_tensor_borrow_end_v1(&borrow, &error)",
        "output->text_ready = 1u",
    ], "atomic text acceptance")
    tail = accept[accept.index("et_i64_tensor_borrow_end_v1(&borrow, &error)"):]
    for forbidden in ("calloc(", "malloc(", "runtime_dispatch",
                      "transaction_commit", "generator_rng_words ="):
        require(forbidden not in tail,
                f"text-ready tail adds fallible/publication work: {forbidden}")

    for phrase in (
        "g0_text(owner)", "g1_text_failures_and_success(owner)",
        "raw.bytes[0] = (unsigned char)prepared.token ^ 1u",
        "prepared.context->owner", "prepared.context->pins.views[0].data",
        "cache_alias", "alias_input->tensor", "ids_alias",
        "ET_I64_TENSOR_CODE_ALLOCATION_FAILED",
        "ET_I64_TENSOR_CODE_ACTIVE_BORROW", "text_ownership_cut(owner)",
        "readiness-cuts=3", "alias-cuts=7", "mutation-cuts=1",
    ):
        require(phrase in native_test, f"native witness omits: {phrase}")
    for phrase in (
        "strict-tokenizer", "special-tokenizer", "invalid-staging",
        "registry-before", "allocation-arm 1 100", "allocation-arm 2 100",
        "public semantic decoder remains unchanged", "allocation-free=2",
    ):
        require(phrase in t1_test, f"T1 witness omits: {phrase}")
    for phrase in (
        "compile_mode normal", "compile_mode sanitize", "detect_leaks=1",
        "runtime-repeat.stdout", "added-defined.txt", "added-undefined.txt",
        "invalid-text-only", "t1-output-decode", "python-isolation.stdout",
    ):
        require(phrase in runner, f"runner omits: {phrase}")
    for phrase in (
        "actual T1", "allocates no", "complete Eshkol bytevector carrier",
        "before the first raw write", "no rooted Eshkol output entry yet",
        "no output envelope", "public generator",
    ):
        require(phrase in contract, f"contract omits boundary: {phrase}")

    dependency = (ROOT / "native/g3c4_output_decode_ids_source_closure.txt").read_text().splitlines()
    additions = [
        "native/g3c4_output_decode_ids_source_closure.txt",
        "native/g3c4_t1_output_decode_extension.esk",
        "tests/g3c4/t1_output_decode_test.esk",
        "tests/g3c4/test_output_text.c",
        "scripts/check-g3c4-output-text.py",
        "scripts/test-g3c4-output-text.sh",
        "docs/g3/G3_C4_OUTPUT_TEXT_STEP21B_CONTRACT.md",
    ]
    expected = dependency + [path for path in additions if path not in dependency]
    actual = (ROOT / "native/g3c4_output_text_source_closure.txt").read_text().splitlines()
    require(actual == expected, "output text closure is not the exact dependency union")
    require(len(actual) == len(set(actual)), "output text closure contains duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 output text static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private raw decode/text readiness contract: PASS")
