#!/usr/bin/env python3
"""Static contract checks for the sole G3-C4 model authority."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def definition(text, name):
    start = text.find(f"(define ({name}")
    require(start >= 0, f"missing definition: {name}")
    depth = 0
    in_string = False
    escaped = False
    comment = False
    for index in range(start, len(text)):
        char = text[index]
        if comment:
            comment = char != "\n"
        elif in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == ";":
            comment = True
        elif char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError(f"unbalanced definition: {name}")


def ordered(body, fragments, label):
    position = -1
    for fragment in fragments:
        found = body.find(fragment, position + 1)
        require(found >= 0, f"{label} missing or misordered: {fragment}")
        position = found


def check():
    subprocess.run(
        [sys.executable,
         str(ROOT / "scripts/check-g3c4-i2-prepared-route.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    source = (ROOT / "native/g3c4_model_extension.esk").read_text()
    test = (ROOT / "tests/g3c4/model_authority_test.esk").read_text()
    native = (ROOT / "tests/g3c4/model_authority_native.c").read_text()

    require(source.count(
        "(define g3c4-model-registry (vector '()))") == 1,
        "C4 model authority registry is not singular")
    require("(define g3c4-model-cleanup-root (vector #f))" in source,
            "C4 cleanup ledger is not separately rooted")
    require("(define (g3c4-model-find" not in source and
            "(define (g3c4-model-entry" not in source,
            "Step 3 adds a pending-capable model observer")
    for forbidden in [
        "(define (g3c4-public", "(extern ptr g3c4-native-generator",
        "(define (g3c4-context", "(define g3c4-context",
        "(define (g3c4-sampler", "(define (g3c4-checkpoint",
    ]:
        require(forbidden not in source,
                f"Step 3 reaches forbidden surface: {forbidden}")
    require("(m3-call operation" in
            definition(source, "g3c4-model-create-seeded-internal"),
            "seeded construction does not use the existing aggregate guard")

    require(
        "(vector #f 'c4-model 'pending #f 'seeded\n"
        "                       original successor #f handles #f #f\n"
        "                       g3c4-model-profile)" in source,
        "exact 12-slot pending model entry changed")
    require("'#((4 4) (4 4) (4 4) (4 4) (4 8) (8 4) "
            "(4) (4) (4) (4)\n"
            "     (256 4) (4) (4) (4 4))" in source,
            "C4 fourteen-parameter shape order changed")
    for path in [
        '("blocks" "0" "attention" "key" "weight")',
        '("blocks" "0" "attention" "output" "weight")',
        '("blocks" "0" "attention" "query" "weight")',
        '("blocks" "0" "attention" "value" "weight")',
        '("blocks" "0" "ffn" "down" "weight")',
        '("blocks" "0" "ffn" "up" "weight")',
        '("blocks" "0" "norm1" "bias")',
        '("blocks" "0" "norm1" "weight")',
        '("blocks" "0" "norm2" "bias")',
        '("blocks" "0" "norm2" "weight")',
        '("head" "weight")',
        '("norm_final" "bias")',
        '("norm_final" "weight")',
        '("position_embedding" "weight")',
    ]:
        require(source.count(path) >= 1, f"missing C4 path: {path}")

    create = definition(source, "g3c4-model-create-seeded-internal")
    ordered(create, [
        "(let* ((original (vector 0 0 0 0))",
        "(handles (make-vector 14 #f))",
        "(cleanup (vector #f #f 'rollback prefix-index))",
        "(entry",
        "(next (cons entry",
        "(guard (caught",
        "(vector-set! g3c4-model-registry 0 next)",
        "(vector-set! g3c4-model-cleanup-root 0 cleanup)",
        "(canonical-entry",
        "(owner (g3c4-native-owner-create-seeded seed))",
        "(i2-g3c4-construction-begin-internal owner)",
        "(vector-set! (vector-ref construction 8) 2",
        "(let register",
        "(i2-module-register-parameter-internal!",
        "(i2-module-register-tied-parameter-internal!",
        "(module-construction-parameters-internal",
        "(g3c4-model-validate-schedule",
        "(let initialize",
        "(g3c4-native-owner-initialize owner index)",
        "(let capture",
        "(i2-g3c4-construction-prepare-eval-internal!",
        "(g3c4-native-owner-prepare-seal owner)",
        "(g3c4-model-tail-preflight",
        "(vector-set! canonical-cleanup 2 'tail)",
        "(i2-g3c4-construction-seal-prepared-internal! construction)",
        "(g3c4-native-owner-commit-seal owner)",
        "(vector-set! canonical-entry 9 #f)",
        "(vector-set! canonical-entry 2 'live)",
    ], "seeded C4 transaction")

    guard_start = create.index("(guard (caught")
    guard_end = create.index(";; Canonical promotion")
    cleanup = create[guard_start:guard_end]
    ordered(cleanup, [
        "(i2-g3c4-construction-abort-internal!",
        "(g3c4-native-owner-abort",
        "(g3c4-model-dead! entry)",
    ], "C4 rollback")
    require("(if (not (= (g3c4-native-owner-commit-seal owner) 0))\n"
            "                  (exit 134))" in create,
            "native prepared commit is recoverably raised")

    dead = definition(source, "g3c4-model-dead!")
    for index in range(3, 12):
        require(f"(vector-set! entry {index} #f)" in dead,
                f"dead entry retains slot {index}")
    require("(vector-set! entry 0 " not in dead and
            "(vector-set! entry 1 " not in dead,
            "dead entry destroys identity slots")

    preflight = definition(source, "g3c4-model-tail-preflight")
    for fragment in [
        "(eq? (vector-ref entry 2) 'pending)",
        "(eq? (vector-ref construction 3) 'prepared)",
        "(eq? (vector-ref construction 9) 'g3c4)",
        "(eq? (vector-ref construction 10) owner)",
        "(vector-ref i2-construction-registry-root 1)",
        "(eq? cleanup (vector-ref g3c4-model-cleanup-root 0))",
    ]:
        require(fragment in preflight,
                f"publication preflight missing: {fragment}")

    for phrase in [
        "failed transaction leaves one dead tombstone",
        "failed transaction publishes no pending entry",
        "exact 12-slot live entry",
        "fourteen unique canonical handles",
        "exact parameter shapes and 1192 unique values",
        "canonical parameter paths",
        "exact tie group",
        "C4 position shape",
        "native owner sealed",
        "second process-lifetime model has distinct authority",
    ]:
        require(phrase in test, f"runtime regression omits: {phrase}")
    require("#define ET_G3C4_NATIVE_OWNER_TESTING 1" in native,
            "native fixture lacks owner allocation injection")
    require("et_g3c4_model_test_owner_registry_count_v1" in native and
            "et_g3c4_model_test_owner_state_v1" in native,
            "native fixture lacks authority witnesses")

    expected = (ROOT / "native/m3_package_source_closure.txt"
                ).read_text().splitlines() + [
        "native/g3c4_i2_construction_extension.esk",
        "native/g3c4_p1_construction_extension.esk",
        "native/g3c4_model_extension.esk",
        "tests/g3c4/model_authority_test.esk",
    ]
    actual = (ROOT / "native/g3c4_model_source_closure.txt"
              ).read_text().splitlines()
    require(actual == expected, "C4 model source closure changed")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 model authority static check failed: {error}",
              file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 model authority static contract: PASS")
