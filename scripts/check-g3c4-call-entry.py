#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/g3c4_call_entry_extension.esk"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def form(name: str, syntax: str = "define") -> str:
    text = SOURCE.read_text()
    needle = f"({syntax} ({name}" if syntax == "define" else f"({syntax} {name}"
    start = text.find(needle)
    require(start >= 0, f"missing {syntax} {name}")
    depth = 0
    string = False
    escape = False
    comment = False
    for index in range(start, len(text)):
        char = text[index]
        if char == "\n":
            comment = False
            continue
        if comment:
            continue
        if string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                string = False
            continue
        if char == ";":
            comment = True
        elif char == '"':
            string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError(f"unbalanced {syntax} {name}")


def ordered(body: str, fragments: list[str], label: str) -> None:
    position = -1
    for fragment in fragments:
        next_position = body.find(fragment, position + 1)
        require(next_position >= 0, f"{label} missing {fragment}")
        require(next_position > position, f"{label} order differs at {fragment}")
        position = next_position


def main() -> int:
    text = SOURCE.read_text()
    closure_file = ROOT / "native/g3c4_call_entry_source_closure.txt"
    closure = closure_file.read_text().splitlines()
    require(closure == sorted(set(closure)),
            "source closure must be sorted and duplicate-free")
    require(all((ROOT / path).is_file() for path in closure),
            "source closure contains a missing file")
    for required in ("native/g3c4_call_entry_extension.esk",
                     "scripts/common.sh",
                     "tests/g3c4/call_entry_native.c",
                     "tests/g3c4/call_entry_a2_stats.c",
                     "tests/g3c4/call_entry_allocation_shim.cpp",
                     "tests/g3c4/call_entry_allocation_test.esk",
                     "tests/g3c4/call_entry_publication_test.esk",
                     "tests/g3c4/call_entry_retention.esk",
                     "tests/g3c4/call_entry_test.esk",
                     "tests/g3c4/call_entry_failstop_test.esk",
                     "scripts/test-g3c4-call-entry.sh"):
        require(required in closure, f"source closure omits {required}")
    require(text.count("(define g3c4-registry (vector '()))") == 1,
            "expected one exact g3c4-registry authority")
    require(not re.search(r"define g3c4-(generator|rng|call)-registry", text),
            "sibling transport registry is forbidden")
    for forbidden in ("generate-internal", "forward-internal", "result-entry",
                      "frame-commit", "counter-advance", "philox-round"):
        require(forbidden not in text, f"premature behavior present: {forbidden}")

    deaden = form("g3c4-entry-dead!")
    ordered(deaden,
            [f"(vector-set! entry {index} #f)" for index in range(3, 12)] +
            ["(vector-set! entry 2 'dead)"],
            "dead tombstone publication")
    dead_test = form("g3c4-dead-entry?")
    require(not re.search(r"vector-ref entry (?:[3-9]|1[01])", dead_test),
            "dead idempotence reads cleared payload slots")
    call_deaden = form("g3c4-call-dead!")
    ordered(call_deaden,
            [f"(vector-set! entry {index} #f)"
             for index in list(range(3, 10)) + [11]] +
            ["(vector-set! entry 2 'dead)",
             "(vector-set! entry 10 #f)"],
            "call tombstone/ledger publication")

    operations = (
        "g3c4-rng-create-seeded-internal",
        "g3c4-rng-clone-internal",
        "g3c4-rng-release-internal!",
        "g3c4-generator-create-internal",
        "g3c4-generator-close-internal!",
    )
    for name in operations:
        body = form(name)
        require(body.count("(m3-call operation") == 1,
                f"{name} must enter m3-call exactly once")

    normalize = form("g3c4-normalize-config")
    for key in (":profile", ":sampling", ":temperature-bits", ":top-k",
                ":top-p-bits", ":max-new-tokens", ":eos", ":seed", ":rng"):
        require(normalize.count(key) >= 1, f"normalizer omits {key}")
    for bound in ("4294967295", "2147483648", "2139095040", "1065353216"):
        require(bound in text, f"integer f32 classifier omits {bound}")

    generator = form("g3c4-generator-create-internal")
    ordered(generator, [
        "(g3c4-normalize-config config operation)",
        "(g3c4-model-entry-live model operation #f)",
        "(vector-set! g3c4-registry 0 next)",
        "(car (vector-ref g3c4-registry 0))",
        "(g3c4-native-generator-",
        "(vector-set! canonical 2 'live)",
        "(vector-set! canonical 10 #f)",
    ], "generator publication")
    require("(g3c4-native-generator-close native)" in generator and
            "(exit 134)" in generator and "(g3c4-entry-dead! entry)" in generator,
            "generator constructor cleanup is incomplete")

    close = form("g3c4-generator-close-internal!")
    ordered(close, ["(g3c4-generator-entry-live generator operation)",
                    "(g3c4-native-generator-close", "(g3c4-entry-dead! live)"],
            "generator close")

    macro = form("g3c4-with-call-internal", "define-syntax")
    require(macro.count("(m3-call operation") == 1,
            "call wrapper must enter m3-call exactly once")
    ordered(macro, [
        "(guard (caught",
        "(vector-set! g3c4-registry 0 next)",
        "(g3c4-native-call-acquire",
        "(vector-set! canonical-ledger 0 #t)",
        "(vector-set! model-entry 10 canonical)",
        "(vector-set! canonical-ledger 1 #t)",
        "(vector-set! generator-entry 9 canonical)",
        "(vector-set! canonical-ledger 2 #t)",
        "(vector-set! canonical 2 'live)",
        "(vector-set! canonical-ledger 3 #t)",
        "(vector-set! m3-call-state 0 canonical)",
        "(g3c4-native-call-prepare-end",
        "(vector-set! phase 0 'committed)",
        "(g3c4-native-call-finish",
        "(g3c4-call-success!",
    ], "call publication/commit")
    require("(if (not (= (g3c4-native-call-finish" in macro and
            macro.count("(exit 134)") >= 3,
            "postcommit finish is not fail-stop")

    rollback = form("g3c4-call-rollback!")
    ordered(rollback, ["(g3c4-native-call-abort",
                        "(vector-set! model-entry 10 #f)",
                        "(vector-set! generator-entry 9 #f)",
                        "(g3c4-call-dead! call-entry)"],
            "call rollback")
    require("(exit 134)" in rollback,
            "native abort cleanup failure is not fail-stop")

    print("G3-C4 call-entry source contract PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"G3-C4 call-entry source contract FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
