#!/usr/bin/env python3
"""Structural gate for the rooted private G3-C4 output envelope."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)

def ordered(text: str, parts: list[str], label: str) -> None:
    at = -1
    for part in parts:
        at = text.find(part, at + 1)
        require(at >= 0, f"{label} missing or misordered: {part}")

def form(text: str, signature: str) -> str:
    start = text.index(signature)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(": depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0: return text[start:i + 1]
    raise ValueError(f"unterminated form: {signature}")

def check() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/check-g3c4-output-text.py")],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    call = (ROOT / "native/g3c4_call_entry_extension.esk").read_text()
    ext = (ROOT / "native/g3c4_output_envelope_extension.esk").read_text()
    test = (ROOT / "tests/g3c4/output_envelope_test.esk").read_text()
    alloc = (ROOT / "tests/g3c4/output_envelope_allocation_test.esk").read_text()
    runner = (ROOT / "scripts/test-g3c4-output-envelope.sh").read_text()
    contract = (ROOT / "docs/g3/G3_C4_OUTPUT_ENVELOPE_STEP22_CONTRACT.md").read_text()

    require("(vector #f #f #f #f #f)" in call, "call ledger lacks output root")
    rollback = form(call, "(define (g3c4-call-rollback!")
    ordered(rollback, ["g3c4-native-call-abort", "(vector-ref ledger 4)",
                       "(g3c4-entry-dead! child)", "(vector-set! ledger 4 #f)"],
            "rollback ownership")
    success = form(call, "(define (g3c4-call-success!")
    require("(not (vector-ref (vector-ref call-entry 10) 4))" in success,
            "success accepts a pending output")
    require("provide" not in ext, "private extension exports a public binding")
    for symbol in ("reserve", "release", "prepare", "copy_decode_ids", "accept_text"):
        require(ext.count(f"et_g3c4_private_output_{symbol}_v1") == 1,
                f"native boundary changed: {symbol}")
    reserve = form(ext, "(define (g3t-output-reserve")
    ordered(reserve, ["(make-bytevector generated 0)",
                      "(make-bytevector (* generated 8) 0)",
                      "(vector 'g3c4-output-private)", "(vector raw staging)",
                      "(vector shell 'output 'pending", "(cons entry",
                      "(vector-set! g3c4-registry 0 next)",
                      "(vector-set! ledger 4 canonical)",
                      "(g3t-native-output-reserve", "(vector-set! canonical 3 created)"],
            "rooted reservation")
    for phrase in ("g3c4-entry-dead! entry", "g3t-native-output-release native",
                   "(eq? (vector-ref ledger 4) entry)"):
        require(phrase in reserve, f"reservation cleanup omits: {phrase}")
    pending = form(ext, "(define (g3t-output-entry-pending")
    for phrase in ("(= (vector-length entry) 12)", "'output", "'pending",
                   "(eq? (vector-ref entry 4) generator-entry)",
                   "(eq? (vector-ref entry 5) model-entry)",
                   "(eq? (vector-ref entry 6) tokenizer)",
                   "(* generated 8)", "(eq? (vector-ref ledger 4) entry)"):
        require(phrase in pending, f"pending validation omits: {phrase}")
    decode = form(ext, "(define (g3t-t1-decode-output!")
    ordered(decode, ["g3t-native-output-copy-decode-ids",
                     "t1-private-g3-decode-raw-into!",
                     "g3t-native-output-accept-text"], "decode coordinator")
    for phrase in ("(exercise-route 0 31)", "(exercise-route 1 37)",
                   "real native G0/G1 preparation", "two-argument decode coordinator",
                   "rollback tombstones rooted output", "linkage-cuts=3",
                   "native-root-cut=1", "repeat-cuts=2"):
        require(phrase in test, f"integrated witness omits: {phrase}")
    for phrase in ("allocation-arm kind index", "vector-cuts", "cons-cuts",
                   "(no-pending?)"):
        require(phrase in alloc, f"allocation witness omits: {phrase}")
    for phrase in ("test-g3c4-output-text.sh", "test-g3c4-call-entry.sh",
                   "compile_mode normal", "compile_mode sanitize", "detect_leaks=1",
                   "runtime-repeat", "python-isolation.stdout"):
        require(phrase in runner, f"runner omits: {phrase}")
    for phrase in ("does not publish an output API", "exact 12-slot",
                   "before native reservation", "deliberately reaches",
                   "frame preparation, commit, publication", "matching call budget"):
        require(phrase in contract, f"contract omits: {phrase}")

    dependency = (ROOT / "native/g3c4_output_text_source_closure.txt").read_text().splitlines()
    additions = ["native/g3c4_output_text_source_closure.txt",
                 "native/g3c4_output_envelope_extension.esk",
                 "tests/g3c4/output_envelope_native.c",
                 "tests/g3c4/output_envelope_test.esk",
                 "tests/g3c4/output_envelope_allocation_test.esk",
                 "scripts/check-g3c4-output-envelope.py",
                 "scripts/test-g3c4-output-envelope.sh",
                 "docs/g3/G3_C4_OUTPUT_ENVELOPE_STEP22_CONTRACT.md"]
    expected = dependency + [p for p in additions if p not in dependency]
    actual = (ROOT / "native/g3c4_output_envelope_source_closure.txt").read_text().splitlines()
    require(actual == expected, "output envelope closure is not the exact dependency union")
    require(len(actual) == len(set(actual)), "output envelope closure has duplicates")
    for path in actual: require((ROOT / path).is_file(), f"missing closure path: {path}")

if __name__ == "__main__":
    try: check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 output envelope static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 rooted private output envelope contract: PASS")
