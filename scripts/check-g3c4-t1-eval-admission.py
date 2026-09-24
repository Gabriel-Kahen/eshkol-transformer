#!/usr/bin/env python3
"""Structural checks for G3-C4 Step 14A T1/eval admission."""

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


def form(text: str, name: str, kind: str = "define") -> str:
    prefix = f"({kind} ({name}" if kind == "define" else f"({kind} {name}"
    start = text.index(prefix)
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
    raise ValueError(f"unterminated form: {name}")


def check() -> None:
    for predecessor in (
        "check-g3c4-call-entry.py",
        "check-g3c4-last-logit-frame.py",
    ):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / predecessor)],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    source = (ROOT / "native/g3c4_call_entry_extension.esk").read_text()
    test = (ROOT / "tests/g3c4/t1_eval_admission_test.esk").read_text()
    runner = (ROOT / "scripts/test-g3c4-t1-eval-admission.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_T1_EVAL_ADMISSION_STEP14A_CONTRACT.md").read_text()

    fingerprint = (
        "sha256:eshkol-byte-tokenizer-v1:"
        "aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704"
    )
    require(source.count(fingerprint) == 1,
            "baseline fingerprint is not one exact private constant")
    tokenizer = form(source, "g3c4-baseline-tokenizer-entry")
    ordered(tokenizer, [
        "(t1-private-entry tokenizer)",
        '"C4 tokenizer identity is not registered"',
        "(t1-core-tokenizer? core)",
        "(eq? (vector-ref core 1) 'raw)",
        "(null? (vector-ref core 2))",
        "(null? (vector-ref core 3))",
        "(null? (vector-ref core 4))",
        "(t1-tokenizer-core-vocab-size core)",
        "g3c4-baseline-tokenizer-fingerprint",
    ], "same-aggregate baseline T1 admission")

    constructor = form(source, "g3c4-generator-create-internal")
    require(constructor.startswith(
        "(define (g3c4-generator-create-internal model tokenizer config)"),
        "generator constructor is not the accepted private arity 3")
    ordered(constructor, [
        "(g3c4-model-entry-live model operation #f)",
        "(g3c4-baseline-tokenizer-entry",
        "tokenizer operation 'invalid-argument 'unsupported",
        "(g3c4-require-model-eval",
        "(entry (vector shell 'generator 'pending #f #f model-entry",
        "(car tokenizer-entry) policy",
        "(vector-set! g3c4-registry 0 next)",
        "(g3c4-native-generator-",
    ], "atomic tokenizer/eval generator publication")

    live = form(source, "g3c4-generator-entry-live")
    require("(tokenizer (vector-ref entry 6))" in live and
            "g3c4-baseline-tokenizer-entry" not in live and
            "g3c4-require-model-eval" not in live,
            "structural close admission incorrectly gates cleanup")
    ready = form(source, "g3c4-generator-entry-call-ready")
    ordered(ready, [
        "(g3c4-generator-entry-live generator operation)",
        "(g3c4-baseline-tokenizer-entry",
        "tokenizer operation 'invalid-state 'invalid-state",
        "(g3c4-require-model-eval model-entry operation 'invalid-state)",
    ], "call-ready dependency recheck")

    macro = form(source, "g3c4-with-call-internal", "define-syntax")
    ordered(macro, [
        "(g3c4-generator-entry-call-ready generator operation)",
        "(tokenizer (vector-ref generator-entry 6))",
        "(vector shell 'call 'pending #f generator-entry model-entry",
        "tokenizer #f #f #f ledger",
        "(vector-set! g3c4-registry 0 next)",
        "(g3c4-native-call-acquire",
        "(let ((answer (begin body ...)))",
        "(g3c4-baseline-tokenizer-entry",
        "tokenizer operation 'invalid-state 'invalid-state",
        "(g3c4-require-model-eval",
        "(g3c4-native-call-prepare-end",
        "(g3c4-native-call-finish",
    ], "call publication and pre-prepare recheck")
    require("t1" not in "\n".join(
                line for line in source.splitlines() if line.startswith("(extern")),
            "T1 pointer entered the C extern boundary")

    for phrase in (
        "foreign tokenizer shell", "bare tokenizer core",
        "copied tokenizer entry", "authentic strict tokenizer",
        "authentic prefix tokenizer", "train mode construction",
        "withdrawn tokenizer call admission", "fingerprint drift call admission",
        "train drift after acquire", "fingerprint drift after acquire",
        "call inherits exact tokenizer shell", "close succeeds in train mode",
        "close succeeds after tokenizer withdrawal",
    ):
        require(phrase in test, f"focused witness omits: {phrase}")
    for phrase in (
        "compile_mode normal", "runtime-repeat.stdout", "compile_mode sanitize",
        "detect_leaks=1", "test-g3c4-call-entry.sh",
    ):
        require(phrase in runner, f"runner omits: {phrase}")
    for phrase in (
        "No T1 core", "before call-entry publication", "before native call prepare",
        "never blocked", "does not authenticate prompt ownership",
    ):
        require(phrase in contract, f"contract omits boundary: {phrase}")

    prior = (ROOT / "native/g3c4_last_logit_frame_source_closure.txt").read_text().splitlines()
    expected = prior + [
        "native/g3c4_last_logit_frame_source_closure.txt",
        "native/g3c4_call_entry_source_closure.txt",
        "native/g3c4_call_entry_extension.esk",
        "tests/g3c4/call_entry_test.esk",
        "tests/g3c4/call_entry_allocation_test.esk",
        "tests/g3c4/call_entry_publication_test.esk",
        "tests/g3c4/call_entry_retention.esk",
        "tests/g3c4/call_entry_failstop_test.esk",
        "tests/g3c4/t1_eval_admission_test.esk",
        "scripts/check-g3c4-call-entry.py",
        "scripts/test-g3c4-call-entry.sh",
        "scripts/check-g3c4-t1-eval-admission.py",
        "scripts/test-g3c4-t1-eval-admission.sh",
        "docs/g3/G3_C4_T1_EVAL_ADMISSION_STEP14A_CONTRACT.md",
    ]
    actual = (ROOT / "native/g3c4_t1_eval_admission_source_closure.txt").read_text().splitlines()
    require(actual == expected,
            "T1/eval closure is not the exact Step 13A successor")
    require(len(actual) == len(set(actual)), "closure contains duplicates")
    for path in actual:
        require((ROOT / path).is_file(), f"closure path is missing: {path}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 T1/eval static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private T1/eval admission contract: PASS")
