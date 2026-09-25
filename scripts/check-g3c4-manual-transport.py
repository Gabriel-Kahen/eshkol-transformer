#!/usr/bin/env python3
"""Static closure and ownership gate for the private C4 manual witness."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ordered(source: str, parts: list[str], label: str) -> None:
    position = -1
    for part in parts:
        position = source.find(part, position + 1)
        require(position >= 0, f"{label} missing or misordered: {part}")


def form(source: str, signature: str) -> str:
    start = source.index(signature)
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise ValueError(f"unterminated form: {signature}")


def check() -> None:
    for name in ("check-g3c4-manual-role-step.py",
                 "check-g3c4-output-envelope.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / name)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    ext = (ROOT / "native/g3c4_manual_transport_extension.esk").read_text()
    call = (ROOT / "native/g3c4_call_entry_extension.esk").read_text()
    test = (ROOT / "tests/g3c4/manual_transport_test.esk").read_text()
    runner = (ROOT / "scripts/test-g3c4-manual-transport.sh").read_text()
    require("provide" not in ext, "private extension exports a facade")
    for stem in ("logits_reserve", "frame_begin", "role_step",
                 "frame_prepare", "frame_commit"):
        require(ext.count(f"et_g3c4_private_{stem}_v1") == 1,
                f"native boundary is not exact: {stem}")
    rollback = form(call, "(define (g3c4-call-rollback!")
    require("(eq? (vector-ref child 1) 'logits)" in rollback,
            "rollback cannot tombstone pending logits")
    reserve = form(ext, "(define (g3t-logits-reserve")
    ordered(reserve, ["(vector shell 'logits 'pending", "(cons entry",
                      "(vector-set! g3c4-registry 0 next)",
                      "(vector-set! ledger 4 canonical)",
                      "(g3t-native-logits-reserve", "(vector-set! canonical 3 created)"],
            "rooted logits reservation")
    for phrase in ("g3c4-native-tensor-release native", "g3c4-entry-dead! entry",
                   "(eq? (vector-ref ledger 4) entry)"):
        require(phrase in reserve, f"reservation cleanup omits {phrase}")
    pending = form(ext, "(define (g3t-logits-entry-pending")
    for phrase in ("g3t-registry-entry", "'logits", "'pending",
                   "(vector-ref entry 4) (vector-ref call 4)",
                   "(vector-ref entry 5) (vector-ref call 5)",
                   "(vector-ref entry 6) (vector-ref call 6)",
                   "(vector-ref (vector-ref call 10) 4) entry"):
        require(phrase in pending, f"pending authentication omits {phrase}")
    commit = form(ext, "(define (g3t-manual-commit!")
    ordered(commit, ["g3t-logits-entry-pending", "(eq? answer",
                     "g3c4-baseline-tokenizer-entry", "g3c4-require-model-eval",
                     "g3c4-native-call-prepare-end", "g3t-native-frame-commit",
                     "(vector-set! phase 0 'committed)",
                     "(vector-set! pending 2 'live)",
                     "(vector-set! ledger 4 #f)", "g3c4-native-call-finish",
                     "g3c4-call-success!"], "manual publication")
    macro = form(ext, "(define-syntax g3c4-with-manual-call-internal")
    ordered(macro, ["(m3-call operation", "(guard (caught",
                    "(g3c4-native-call-acquire", "(vector-set! m3-call-state 0 canonical)",
                    "(g3t-manual-commit!"], "lexical manual cleanup")
    require(macro.count("(m3-call operation") == 1,
            "manual boundary enters shared guard more than once")
    witness = form(test, "(define (manual-frame")
    for ordinal in range(21):
        require(witness.count(f"(g3t-role-step call {ordinal})") == 1,
                f"development witness lacks explicit role {ordinal}")
    for phrase in ("run-prefill (bytevector 41)",
                   "run-prefill (bytevector 41 43)", "decode-result",
                   "out-of-order role rolls back", "wrong result shell rolls back",
                   "native reservation allocation failure"):
        require(phrase in test, f"integrated witness omits {phrase}")
    for phrase in ("compile_mode normal", "compile_mode sanitize",
                   "runtime-repeat", "detect_leaks=1", "python-isolation.stdout"):
        require(phrase in runner, f"runner omits {phrase}")
    base = (ROOT / "native/g3c4_manual_role_step_source_closure.txt").read_text().splitlines()
    other = (ROOT / "native/g3c4_output_envelope_source_closure.txt").read_text().splitlines()
    additions = ["native/g3c4_manual_role_step_source_closure.txt",
                 "native/g3c4_output_envelope_source_closure.txt",
                 "native/g3c4_manual_transport_extension.esk",
                 "tests/g3c4/manual_transport_native.c",
                 "tests/g3c4/manual_transport_test.esk",
                 "scripts/check-g3c4-manual-transport.py",
                 "scripts/test-g3c4-manual-transport.sh",
                 "docs/g3/G3_C4_MANUAL_TRANSPORT_CONTRACT.md"]
    expected = list(dict.fromkeys(base + other + additions))
    actual = (ROOT / "native/g3c4_manual_transport_source_closure.txt").read_text().splitlines()
    require(actual == expected, "manual transport closure is not the exact dependency union")
    for relative in actual:
        require((ROOT / relative).is_file(), f"missing closure file: {relative}")


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"G3-C4 manual transport static check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("G3-C4 private manual transport contract: PASS")
