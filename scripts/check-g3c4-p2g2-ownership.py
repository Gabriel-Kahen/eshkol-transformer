#!/usr/bin/env python3
"""Exact source-private P2/G2 Eshkol admission closure."""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
for predecessor in ("check-g3c4-call-entry.py",
                    "check-g3c4-output-envelope.py",
                    "check-g3c4-p2g2-carrier-bridge.py"):
    subprocess.run([sys.executable, str(root / "scripts" / predecessor)],
                   cwd=root, check=True, stdout=subprocess.DEVNULL)

call = (root / "native/g3c4_call_entry_extension.esk").read_text()
envelope = (root / "native/g3c4_output_envelope_extension.esk").read_text()
gate = (root / "native/g3c4_p2g2_ownership_extension.esk").read_text()
public = (root / "native/g3c4_public_generation_dispatch_extension.esk").read_text()
fixture = (root / "tests/g3c4/p2_g2_ownership_test.esk").read_text()
public_fixture = (root / "tests/g3c4/p2_g1_public_generation_test.esk").read_text()
runner = (root / "scripts/test-g3c4-p2g2-ownership.sh").read_text()
base_closure = (root / "native/g3c4_output_envelope_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_ownership_extension.esk",
    "native/g3c4_public_generation_dispatch_extension.esk",
    "scripts/check-g3c4-p2g2-carrier-bridge.py",
    "scripts/check-g3c4-p2g2-ownership.py",
    "scripts/test-g3c4-p2g2-carrier-bridge.sh",
    "scripts/test-g3c4-p2g2-ownership.sh",
    "tests/g3c4/p2_g1_public_generation_test.esk",
    "tests/g3c4/p2_g2_ownership_test.esk",
    "tests/q0/test_python_isolation.py",
    "docs/g3/G3_C4_P2_G2_T1_PREFIX_OWNERSHIP_CONTRACT.md",
    "docs/g3/G3_C4_P2_G2_ESHKOL_OWNERSHIP_ADMISSION_LEAF.md",
    "native/g3c4_p2g2_ownership_source_closure.txt",
]
expected_closure = base_closure + [path for path in additions
                                   if path not in base_closure]
closure = (root / "native/g3c4_p2g2_ownership_source_closure.txt").read_text().splitlines()
assert closure == expected_closure and len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure)

assert call.count("(define g3c4-p2g2-ownership-gate (vector #f))") == 1
assert call.count("(vector-ref g3c4-p2g2-ownership-gate 0)") == 2
assert "(= (vector-ref policy 5) 2)" in call
assert "(= budget 2)" in call
assert "(and (= call-kind 2) (= budget 2))" in call
assert "(define (g3c4-call-tuple?" in call
assert envelope.count("(vector-ref g3c4-p2g2-ownership-gate 0)") == 3
assert "(= prompt-length 2) (= generated 2)" in envelope
assert "(m3t-exact-i64? (vector-ref call 7))" in envelope
assert "(= (vector-ref call 7) 2)" in envelope
assert "(make-bytevector generated 0)" in envelope
assert "(make-bytevector (* generated 8) 0)" in envelope
assert gate.count("(vector-set! g3c4-p2g2-ownership-gate 0 #t)") == 1
assert "extern" not in gate and "provide" not in gate
assert "(= (vector-ref policy 5) 1)" in public
assert "public C4 still rejects budget two with private gate on" in public_fixture
for phrase in (
    "feature-off policy rejects budget two",
    "feature-off call rejects budget two",
    "terminal scope rejects budget-two call",
    "P1/G2 reservation is rejected",
    "budget-two reservation requires retained request marker",
    "native budget-two reservation failure",
    "failed native reservation leaves no child",
    "exact pending P2/G2 capacity and private child",
    "auxiliary raw length is exact",
    "auxiliary staging length is exact",
    "child ledger identity is exact",
    "pending output has no copyable token IDs",
    "pending output cannot enter T1 output decode",
    "pending output rejects changed request marker",
    "pending output has no accepted text",
    "rollback tombstones unpublished pending child",
):
    assert phrase in fixture, phrase
for phrase in ("verify_toolchain", "tests.q0.test_python_isolation",
               "compile_mode normal", "compile_mode sanitize",
               "detect_leaks=1", "cmp \"$evidence/normal-run.stdout\"",
               "ET_G3C4_P2_G2_FIRST_FRAME_PRIVATE",
               "test-g3c4-p2g1-pending.sh",
               "test-g3c4-p2g2-carrier-bridge.sh"):
    assert phrase in runner, phrase

print("G3-C4 P2/G2 ownership static PASS")
