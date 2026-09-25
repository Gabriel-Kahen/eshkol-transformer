#!/usr/bin/env python3
"""Static contract checks for the append-only P1 prepared split."""

from hashlib import sha256
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


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


def check():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-e3-p1-contract.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check-tr3-p1-fixed.py")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL)

    fixed_hashes = {
        "lib/transformer/module.esk":
            "38be7a65d467753dc53abccf5195107d939d842e5988783d8f0073f4b9c9f46c",
        "tools/p1/module_surface.tsv":
            "e302fb8c7545d450b5365460393c41b61c9bba8840de27b3c3f6abfcec714104",
        "native/p1_identity_public_symbols.txt":
            "47e674a3049143cc9ee216f78d455ecb6a22fbcfc13b55aa2b7ace5587671421",
        "native/p1_package_defined_symbols.txt":
            "ea56c69b4fd42507688df052219a62dc3c8b7a9039e4aa7be9dc7bcb0febdf17",
        "native/p1_package_undefined_symbols.txt":
            # 07dbc6e admitted the reviewed indexed shell-lookup runtime calls.
            "1941268ad78dad3d432fcbe5a8cd93ee7a959d3714c23fe59e8a6f3736566173",
        "native/m3t_transport_extension.esk":
            "71247e722d72764ce1b3400cb3f818ba12c3754dd11a6c742ffcec63942d5815",
        "native/i2_wave2_root.esk":
            # c74c267 kept private rollback in call position for pinned LLVM.
            "f1314ccd4c5a382733115005c93d8712290bd63c5373eac2509093989fefad33",
    }
    for path, expected in fixed_hashes.items():
        require(digest(path) == expected, f"inherited surface drifted: {path}")

    template = (ROOT / "templates/p1/module_roots.esk.tmpl").read_text()
    source = (ROOT / "internal/p1/lib/transformer/module.esk").read_text()
    generated = template.split("@@P1_TRUSTED_BEGIN@@\n", 1)[1].split(
        "@@P1_TRUSTED_END@@", 1)[0]
    require(generated == source, "trusted P1 root is stale")
    require(sha256(definition(source, "construction-seal!").encode()).hexdigest()
            # cf5e394 compares scheduled handles by identity, avoiding a
            # structural walk over opaque mutable construction records.
            == "a74565aba7af6804de57f0074acef8d7da03acb77e7aa3a50c4e46108623c665",
            "legacy slot-61 construction seal changed")

    begin = definition(source, "construction-begin")
    require(begin.index("(guard ") < begin.index("p1-native-construction-begin"),
            "construction begin cleanup guard follows native publication")
    require("(construction-record 'module-construction-begin-internal identity)"
            in begin, "construction begin omits canonical promotion reread")

    prepare = definition(source, "construction-prepare-eval!")
    for witness in (
        "construction changed after parameter schedule",
        "construction has unattached modules",
        "construction has unattached handles",
        "(install-canonical-paths! updates)",
        "(finalize-modules! canonical-finalizers)",
        "(set-mode-modules! canonical-finalizers 'eval)",
        "(verify-finalized-mode! operation canonical-finalizers 'eval)",
        "p1-native-construction-prepare",
        "(vector-set! canonical-record 1 'prepared)",
    ):
        require(witness in prepare, f"prepare-eval omits: {witness}")
    require(prepare.index("p1-native-construction-prepare") <
            prepare.index("(vector-set! canonical-record 1 'prepared)"),
            "source PREPARED publishes before native PREPARED")

    guarded = definition(source, "construction-prepare-eval-guarded!")
    require("(p1-runtime-reserve-exception-handlers 1)" in guarded,
            "prepare cleanup handler capacity is not reserved")
    require("(guard " in guarded and guarded.count("(construction-abort! identity)") == 2,
            "prepare failure does not abort the admitted construction")
    require(guarded.index("(guard ") <
            guarded.index("(set! construction-busy? #t)") <
            guarded.index("(construction-prepare-eval! identity)"),
            "prepare cleanup guard follows authority mutation or fallible work")

    commit = definition(source, "construction-seal-prepared!")
    require("(guard " not in commit and "native-check" not in commit,
            "prepared commit tail has a recoverable handler")
    require("(if (not (= status 0)) (exit 134) #t)" in commit,
            "prepared commit status is not fail-stop")
    require(commit.index("p1-native-construction-commit-prepared") <
            commit.index("(vector-set! record 1 'sealed)"),
            "source SEALED publishes before native commit")

    abort = definition(source, "construction-abort!")
    require("p1-native-construction-abort-prepared" in abort and
            "p1-native-construction-abort" in abort,
            "source abort does not dispatch OPEN and PREPARED separately")
    shell_check = definition(source, "construction-check-shell!")
    require("(eq? state 'prepared)" in shell_check,
            "prepared shells are observable")

    closure = (ROOT / "native/g3c4_p1_construction_source_closure.txt").read_text().splitlines()
    require(closure == [
        "internal/p1/lib/transformer/module.esk",
        "native/g3c4_p1_construction_extension.esk",
        "tests/p1/providers/p1_test/tensor_provider.esk",
        "tests/p1/prepared_construction_test.esk",
    ], "prepared construction source closure changed")

    symbols = (ROOT / "native/p1_identity_trusted_symbols.txt").read_text().splitlines()
    require(symbols == sorted(symbols), "trusted P1 symbol manifest is unsorted")
    for symbol in (
        "et_p1_private_construction_prepare_v1",
        "et_p1_private_construction_commit_prepared_v1",
        "et_p1_private_construction_abort_prepared_v1",
    ):
        require(symbols.count(symbol) == 1,
                f"prepared native symbol missing or duplicated: {symbol}")

    print("P1 PREPARED STRUCTURE PASS: slots=71,72 legacy=exact public=unchanged")


if __name__ == "__main__":
    try:
        check()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"P1 PREPARED STRUCTURE FAIL: {error}", file=sys.stderr)
        sys.exit(1)
