#!/usr/bin/env python3
"""Static contract checks for the G3-C4 route through the sole I2 ledger."""

from pathlib import Path

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
    base = (ROOT / "native/i2_wave2_extension.esk").read_text()
    root = (ROOT / "native/i2_wave2_root.esk").read_text()
    c4 = (ROOT / "native/g3c4_i2_construction_extension.esk").read_text()
    test = (ROOT / "tests/g3c4/i2_prepared_route_test.esk").read_text()

    require("\n".join(root.splitlines()[5:]) + "\n" == base,
            "I2 canonical root/extension mirror drifted")
    require(base.count("(define i2-construction-registry-root ") == 1,
            "base I2 construction registry is not singular")
    require("i2-construction-registry-root" in c4 and
            "(define i2-construction-registry-root" not in c4,
            "C4 route does not reuse the sole I2 registry")
    require("(vector 'i2-construction #f #f 'aborted (list membership) '()"
            in base and
            "(i2-carrier 'parameter '() 'parameter) 'm3t #f)" in base,
            "M3T record lacks its fixed route/null owner")
    require("'g3c4 native-owner" in c4,
            "C4 record lacks its closed route/exact owner")

    begin = definition(base, "i2-construction-begin-internal")
    require("i2-native-g3c4" not in begin and "'m3t #f" in begin,
            "M3T begin no longer has its fixed ownerless route")
    seal = definition(base, "i2-construction-seal-internal!")
    abort = definition(base, "i2-construction-abort-internal!")
    require("value 'm3t 'i2-construction-seal-internal!" in seal,
            "legacy seal does not enforce the M3T route")
    require("(vector-ref record 9) 'm3t" in abort,
            "legacy abort does not enforce the M3T route")

    registration = definition(base, "i2-module-register-parameter-internal!")
    ordered(registration, [
        "(i2-construction-preflight!",
        "(vector-set! active 5",
        "(i2-native-parameter-bind",
        "(i2-construction-bound-registration-preflight!",
    ], "registration")
    require("i2-g3c4-preflight-private" not in registration,
            "registration bypasses the closed record-route dispatcher")

    for name in [
        "i2-g3c4-construction-begin-internal",
        "i2-g3c4-construction-prepare-eval-internal!",
        "i2-g3c4-construction-seal-prepared-internal!",
        "i2-g3c4-construction-abort-internal!",
    ]:
        definition(c4, name)

    c4_begin = definition(c4, "i2-g3c4-construction-begin-internal")
    ordered(c4_begin, [
        "(i2-native-g3c4-construction-available)",
        "(vector-ref i2-construction-registry-root 1)",
        "'g3c4 native-owner",
        "(vector-set! i2-construction-registry-root 1 canonical)",
        "(module-construction-begin-internal i2-provider-name)",
    ], "C4 begin")

    prepare = definition(c4, "i2-g3c4-construction-prepare-eval-internal!")
    ordered(prepare, [
        "(i2-construction-open-route value 'g3c4 operation)",
        "(i2-construction-retained-preflight! record operation)",
        "(module-construction-prepare-eval-internal!",
        "(vector-set! record 3 'prepared)",
    ], "C4 prepare")
    require("(i2-construction-rollback-tail! record)" in prepare,
            "P1 prepare failure does not restore I2")

    prepared_seal = definition(
        c4, "i2-g3c4-construction-seal-prepared-internal!")
    ordered(prepared_seal, [
        "(vector-ref record 3) 'prepared",
        "(module-construction-seal-prepared-internal!",
        "(vector-set! record 3 'sealed)",
        "(vector-set! record 10 #f)",
        "(vector-set! i2-construction-registry-root 1 #f)",
    ], "C4 prepared seal")

    c4_abort = definition(c4, "i2-g3c4-construction-abort-internal!")
    require("(eq? (vector-ref record 3) 'open)" in c4_abort and
            "(eq? (vector-ref record 3) 'prepared)" in c4_abort,
            "C4 abort does not admit both rollback phases")
    ordered(c4_abort, [
        "(i2-construction-abort-active! record operation)",
        "(vector-set! record 1 #f)",
        "(vector-set! record 7 0)",
    ], "C4 abort")

    retained = definition(base, "i2-construction-retained-preflight!")
    require("(vector-ref record 5)" in retained and
            "(vector-ref record 8)" in retained and
            "(i2-construction-preflight!" in retained,
            "complete retained-set preflight is missing")
    abort_active = definition(base, "i2-construction-abort-active!")
    ordered(abort_active, [
        "(i2-construction-retained-preflight! record operation)",
        "(module-construction-abort-internal!",
        "(i2-construction-rollback-tail! record)",
    ], "I2 abort")

    require(c4.count(
        ":real et_i2_private_g3c4_construction_parameter_preflight_v1") == 1,
        "C4 route does not use the accepted three-argument bridge exactly once")
    require("model-registry" not in c4 and "context" not in c4 and
            "sampler" not in c4,
            "Step 2 source reaches model/context/sampler scope")

    for phrase in [
        "missing schedule aborts complete I2 route",
        "prepared abort",
        "legacy abort cannot consume C4 route",
        "prepared seal returns exact root",
        "native storage outlives I2 abort",
    ]:
        require(phrase in test, f"runtime regression omits: {phrase}")
    fixture = definition(test, "build")
    ordered(fixture, [
        "(let register",
        "(register (+ index 1)))))",
        "(let initialize",
        "(g3c4-test-owner-initialize owner index)",
    ], "C4 fixture bind-before-initialize")

    expected = (ROOT / "native/i2_wave2_source_closure.txt"
                ).read_text().splitlines() + [
        "native/g3c4_i2_construction_extension.esk",
        "native/g3c4_p1_construction_extension.esk",
        "tests/g3c4/i2_prepared_route_test.esk",
    ]
    actual = (ROOT / "native/g3c4_i2_construction_source_closure.txt"
              ).read_text().splitlines()
    require(actual == expected, "C4 I2 route source closure changed")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError) as error:
        print(f"G3-C4 I2 prepared route static check failed: {error}",
              file=__import__("sys").stderr)
        raise SystemExit(1)
    print("G3-C4 I2 prepared route static contract: PASS")
