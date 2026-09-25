#!/usr/bin/env python3
"""Structural checks for G3-C4 Step 15A prompt/T1 copy ownership."""

from pathlib import Path
import re
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


def scheme_form(text: str, name: str) -> str:
    start = text.index(f"(define ({name}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError(f"unterminated Scheme form: {name}")


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
        [sys.executable, str(ROOT / "scripts/check-g3c4-t1-eval-admission.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    owner = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    eshkol = (ROOT / "native/g3c4_call_entry_extension.esk").read_text()
    test = (ROOT / "tests/g3c4/prompt_t1_borrow_test.esk").read_text()
    native_test = (ROOT / "tests/g3c4/prompt_t1_borrow_native.c").read_text()
    retention = (ROOT / "tests/g3c4/prompt_t1_borrow_retention.esk").read_text()
    allocation_test = (ROOT / "tests/g3c4/prompt_t1_borrow_allocation_test.esk").read_text()
    runner = (ROOT / "scripts/test-g3c4-prompt-t1-borrow.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_PROMPT_T1_BORROW_STEP15A_CONTRACT.md").read_text()
    t1_header = (ROOT / "native/t1_i64_shell.h").read_text()

    feature = "ET_G3C4_PROMPT_T1_BORROW_PRIVATE"
    require(feature in owner and feature in header,
            "prompt/T1 feature macro is missing")
    require("requires the private generator" in owner,
            "prompt/T1 leaf does not require the accepted generator transport")

    native_symbols = (
        "et_g3c4_private_input_from_t1_v1",
        "et_g3c4_private_tensor_release_v1",
    )
    for symbol in native_symbols:
        require(owner.count(symbol) == 1 and header.count(symbol) == 1,
                f"source-private prompt ABI changed: {symbol}")
    require("ET_G3C4_INPUT_KIND = 2" in owner and
            "ET_G3C4_RNG_KIND = 8" in owner,
            "accepted input/RNG kind assignment changed")
    require("et_i64_tensor *tensor" in owner and "int64_t length" in owner,
            "owned prompt tensor record is incomplete")
    require("G3-C4 input record must be 48 bytes" in owner,
            "prompt owner layout is not frozen")

    constructor = c_function(
        owner, "void *et_g3c4_private_input_from_t1_v1")
    ordered(constructor, [
        "et_g3c4_error_reset_internal()",
        "et_t1_i64_shell_length_v1(sealed_t1)",
        "et_t1_i64_shell_last_status_v1()",
        "length != 1 && length != 2",
        "et_t1_i64_shell_read_v1(sealed_t1",
        "words[index] < 0 || words[index] > 255",
        "et_g3c4_input_allocate()",
        "et_i64_tensor_create_v1(",
        "et_i64_tensor_copy_from_v1(",
        "ET_G3C4_INPUT_MAGIC",
        "ET_G3C4_INPUT_KIND",
        "et_g3c4_enroll_input(input)",
    ], "authenticated synchronous T1 copy")
    require("et_i64_tensor_borrow" not in constructor,
            "prompt constructor retained or created an I1 borrow")
    require("sealed_t1" not in constructor[constructor.index(
                "et_g3c4_enroll_input(input)"):],
            "prompt owner retains T1 provenance after enrollment")

    release = c_function(
        owner, "int64_t et_g3c4_private_tensor_release_v1")
    ordered(release, [
        "et_g3c4_error_reset_internal()",
        "et_g3c4_admit_input(",
        "et_i64_tensor_destroy_v1(&input->tensor",
        "input->length = 0",
        "ET_G3C4_CONTEXT_DEAD",
    ], "typed prompt release")

    # T1 remains the accepted scalar-read ABI. Step 15A may consume it but may
    # not add a raw pointer/view/borrow seam.
    t1_declarations = set(re.findall(
        r"\b(et_t1_i64_shell_[a-z0-9_]+_v1)\s*\(", t1_header))
    require(t1_declarations == {
        "et_t1_i64_shell_create_v1",
        "et_t1_i64_shell_length_v1",
        "et_t1_i64_shell_write_v1",
        "et_t1_i64_shell_read_v1",
        "et_t1_i64_shell_seal_v1",
        "et_t1_i64_shell_abort_v1",
        "et_t1_i64_shell_last_status_v1",
        "et_t1_i64_shell_test_fail_stage_v1",
        "et_t1_i64_shell_test_live_count_v1",
    }, "T1 native ABI was extended or changed")
    for forbidden in ("shell_borrow", "shell_view", "shell_data"):
        require(forbidden not in owner and forbidden not in header,
                f"invented raw T1 seam: {forbidden}")

    require("(extern ptr g3c4-native-input-from-t1 ptr" in eshkol and
            ":real et_g3c4_private_input_from_t1_v1)" in eshkol,
            "Eshkol prompt constructor extern changed")
    require("(extern i64 g3c4-native-tensor-release ptr" in eshkol and
            ":real et_g3c4_private_tensor_release_v1)" in eshkol,
            "Eshkol prompt release extern changed")
    create = scheme_form(eshkol, "generation-input-create-internal")
    require(create.startswith(
        "(define (generation-input-create-internal encoded)"),
        "prompt constructor arity changed")
    ordered(create, [
        "(t1-wave1-tensor-admitted? encoded)",
        "(vector shell 'input 'pending #f #f #f #f #f #f #f",
        "(vector-set! g3c4-registry 0 next)",
        "(g3c4-native-input-from-t1 encoded)",
        "(vector-set! canonical 3 created)",
        "(vector-set! canonical 2 'live)",
    ], "same-aggregate prompt publication")
    for forbidden in ("g3c4-generator-entry", "g3c4-baseline-tokenizer-entry",
                      "g3c4-require-model-eval"):
        require(forbidden not in create,
                f"prompt constructor invented retained provenance: {forbidden}")
    release_form = scheme_form(eshkol, "generation-tensor-release-internal!")
    require("(g3c4-dead-entry? entry 'input) #t" in release_form and
            "g3c4-native-tensor-release" in release_form and
            "g3c4-entry-dead!" in release_form,
            "Eshkol prompt release/scrub contract changed")

    for phrase in (
        "one-token prompt", "two-token prompt", "empty prompt",
        "three-token prompt", "foreign prompt shell",
        "tokenizer is not an encoded prompt", "copied input shell",
        "input retains no T1 shell or generator authority",
        "exact dead release is idempotent", "source survives input release",
    ):
        require(phrase in test, f"focused witness omits: {phrase}")
    for phrase in (
        "unsealed", "out-of-range", "wrong-kind", "allocation",
    ):
        require(phrase in native_test or phrase in test,
                f"focused native/aggregate evidence omits: {phrase}")
    for phrase in (
        "Exactly one encoded P2 source is reused", '"1024"', '"8192"',
        "peak-extra-i1", "prompt-i1-count 1", "prompt-i1-count 2",
        "prompt-transport-count 2", "native_tombstones=",
    ):
        require(phrase in retention, f"retention witness omits: {phrase}")
    for phrase in (
        "allocation-arm", "allocation-consumed", "sweep 1", "sweep 2",
        "no-pending-input?", "m3-call-state",
        "prompt/T1 allocation cuts PASS",
    ):
        require(phrase in allocation_test,
                f"Eshkol allocation witness omits: {phrase}")
    for phrase in (
        "compile_mode normal", "runtime-repeat.stdout", "compile_mode sanitize",
        "detect_leaks=1", "test-g3c4-t1-eval-admission.sh",
        "prompt_t1_borrow_native.c", "ET_I64_TENSOR_TESTING",
        "ET_T1_I64_SHELL_TESTING",
    ):
        require(phrase in runner, f"runner omits: {phrase}")
    for phrase in (
        "one or two", "no tokenizer-provenance claim", "No T1 pointer",
        "does not add to `t1_i64_shell.h`", "P3 is rejected",
        "no prompt-to-prefill adapter", "1,024 and 8,192",
    ):
        require(phrase in contract, f"contract omits boundary: {phrase}")

    prior = (ROOT / "native/g3c4_t1_eval_admission_source_closure.txt").read_text().splitlines()
    expected = prior + [
        "native/g3c4_t1_eval_admission_source_closure.txt",
        "tests/g3c4/prompt_t1_borrow_test.esk",
        "tests/g3c4/prompt_t1_borrow_native.c",
        "tests/g3c4/prompt_t1_borrow_retention.esk",
        "tests/g3c4/prompt_t1_borrow_allocation_test.esk",
        "scripts/check-g3c4-prompt-t1-borrow.py",
        "scripts/test-g3c4-prompt-t1-borrow.sh",
        "docs/g3/G3_C4_PROMPT_T1_BORROW_STEP15A_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_prompt_t1_borrow_source_closure.txt").read_text().splitlines()
    require(actual == expected,
            "prompt/T1 closure is not the exact Step 14A successor")
    require(len(actual) == len(set(actual)), "closure contains duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 prompt/T1 static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private prompt/T1 borrow contract: PASS")
