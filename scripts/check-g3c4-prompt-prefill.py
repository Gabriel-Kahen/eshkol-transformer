#!/usr/bin/env python3
"""Structural checks for the G3-C4 Step 18A prompt/prefill binding."""

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
    for predecessor in ("check-g3c4-prompt-t1-borrow.py",
                        "check-g3c4-prefill2.py"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / predecessor)],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    source = (ROOT / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
    header = (ROOT / "src/eshkol_transformer/g3c4_context_internal.h").read_text()
    test = (ROOT / "tests/g3c4/test_prompt_prefill.c").read_text()
    runner = (ROOT / "scripts/test-g3c4-prompt-prefill.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_PROMPT_PREFILL_STEP18A_CONTRACT.md").read_text()

    require("ET_G3C4_PROMPT_PREFILL_PRIVATE requires Steps 15A and 17A"
            in source, "binding macro does not require both accepted leaves")
    symbols = (
        "et_g3c4_private_prompt_prefill_preflight_v1",
        "et_g3c4_private_prompt_prefill_v1",
    )
    for symbol in symbols:
        require(source.count(symbol) == 1 and header.count(symbol) == 1,
                f"source-private binding boundary changed: {symbol}")

    preflight = c_function(
        source, "int64_t et_g3c4_private_prompt_prefill_preflight_v1")
    ordered(preflight, [
        "et_g3c4_error_reset_internal()",
        "et_g3c4_admit_idle_call(context_candidate)",
        "et_g3c4_admit_input(input_candidate, 0)",
        "input->length != 1 && input->length != 2",
        "input->length + budget > 2",
    ], "pre-acquire P+G admission")
    for forbidden in ("et_g3c4_private_call_acquire_v1", "pins_begin",
                      "prefill1_v1", "prefill2_v1", "borrow_begin_v1"):
        require(forbidden not in preflight,
                f"preflight performs work before P+G admission: {forbidden}")

    borrow = c_function(source, "static int et_g3c4_prompt_prefill_borrow")
    ordered(borrow, [
        "et_i64_tensor_borrow_begin_v1(",
        "et_i64_tensor_borrow_view_v1(",
        'strcmp(view->dtype, "i64")',
        'strcmp(view->device, "cpu")',
        "view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR",
        "view->rank != 2u",
        "view->shape[0] != 1u",
        "view->shape[1] != (uint64_t)input->length",
        "((const int64_t *)view->data)[index] < 0",
        "((const int64_t *)view->data)[index] > 255",
    ], "exact scoped I1 descriptor admission")

    execute = c_function(
        source, "int64_t et_g3c4_private_prompt_prefill_v1")
    ordered(execute, [
        "et_g3c4_admit_active_call(context_candidate)",
        "et_g3c4_admit_input(input_candidate, 0)",
        "context->call_kind != 2",
        "input->length + context->budget > 2",
        "et_g3c4_prompt_prefill_borrow(input, &borrow, &view)",
        "input->length == 1",
        "ET_G3C4_PROMPT_CAT(et_g3c4_private_pre, fill1_v1)(",
        "ET_G3C4_PROMPT_CAT(et_g3c4_private_pre, fill2_v1)(",
        "et_i64_tensor_borrow_end_v1(&borrow, &error)",
    ], "matching P1/P2 borrowed route")
    failure = execute[execute.index("fail:"):]
    ordered(failure, [
        "et_g3c4_error_snapshot_internal()",
        "et_i64_tensor_borrow_end_v1(&borrow, &error)",
        "et_g3c4_error_restore_internal(first)",
    ], "borrow cleanup and first-error restoration")
    for forbidden in ("generator_rng_words", "memcpy(token", "calloc(",
                      "malloc(", "prefill3_v1"):
        require(forbidden not in execute,
                f"binding route adds forbidden state or padding: {forbidden}")

    for phrase in (
        "route_parity", "pre_acquire_admission", "adapter_failure_cuts",
        "borrow_and_alias_failures", "prefill1_dispatches == 21u",
        "prefill2_dispatches == 21u", "input2, 1",
        "ET_I64_TENSOR_CODE_ACTIVE_BORROW",
        "ET_I64_TENSOR_CODE_ALLOCATION_FAILED", "check_preserved",
        "routes=2", "pre-acquire-cuts=5", "numerical-cuts=42",
        "borrow-cuts=3",
    ):
        require(phrase in test, f"focused witness omits: {phrase}")
    for phrase in (
        "compile_mode normal", "compile_mode sanitize",
        "detect_leaks=1", "runtime-repeat.stdout",
        "--wrap=et_kernel_runtime_dispatch", "added-defined.txt",
        "added-undefined.txt",
    ):
        require(phrase in runner, f"runner omits: {phrase}")
    for phrase in (
        "before call acquisition", "exact two-element borrowed span",
        "no P3 route", "fabricated padding token", "leave the old cache",
        "adds no Eshkol operation", "public result", "EOS behavior",
    ):
        require(phrase in contract, f"contract omits boundary: {phrase}")

    numerical = (ROOT / "native/g3c4_prefill2_source_closure.txt").read_text().splitlines()
    prompt = (ROOT / "native/g3c4_prompt_t1_borrow_source_closure.txt").read_text().splitlines()
    expected = numerical + [path for path in prompt if path not in numerical] + [
        "native/g3c4_prefill2_source_closure.txt",
        "native/g3c4_prompt_t1_borrow_source_closure.txt",
        "tests/g3c4/test_prompt_prefill.c",
        "scripts/check-g3c4-prompt-prefill.py",
        "scripts/test-g3c4-prompt-prefill.sh",
        "docs/g3/G3_C4_PROMPT_PREFILL_STEP18A_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_prompt_prefill_source_closure.txt").read_text().splitlines()
    require(actual == expected, "binding closure is not the exact dependency union")
    require(len(actual) == len(set(actual)), "binding closure contains duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 prompt/prefill static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private prompt/prefill binding contract: PASS")
