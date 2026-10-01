#!/usr/bin/env python3
"""Closed source-private P2/G2 protected ownership dependency."""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,
                str(root / "scripts/check-g3c4-p2g2-ownership.py")],
               cwd=root, check=True, stdout=subprocess.DEVNULL)
base = (root / "native/g3c4_p2g2_ownership_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_protected_scope_extension.esk",
    "tests/g3c4/p2_g2_protected_scope_native.c",
    "tests/g3c4/p2_g2_protected_scope_test.esk",
    "tests/g3c4/p2_g2_protected_scope_feature_off_test.esk",
    "scripts/check-g3c4-p2g2-protected-scope.py",
    "scripts/test-g3c4-p2g2-protected-scope.sh",
    "docs/g3/G3_C4_P2_G2_PROTECTED_SCOPE_LEAF.md",
    "native/g3c4_p2g2_protected_scope_source_closure.txt",
]
closure = (root / "native/g3c4_p2g2_protected_scope_source_closure.txt").read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure)

scope = (root / additions[0]).read_text()
fixture = (root / additions[2]).read_text()
feature_off = (root / additions[3]).read_text()
runner = (root / "scripts/test-g3c4-p2g2-protected-scope.sh").read_text()
native = (root / "src/eshkol_transformer/g3c4_model_owner.c").read_text()
assert "(define-syntax g3c4-with-p2g2-protected-call" in scope
assert "(define (g3c4-p2g2-protected-check" in scope
assert "(define (g3c4-p2g2-protected-abort!" in scope
assert "(g3c4-native-call-acquire native 2 2)" in scope
assert "(g3c4-output-reserve call 2)" in scope
abort_scope = scope[scope.index("(define (g3c4-p2g2-protected-abort!"):]
assert abort_scope.index("(g3c4-p2g2-protected-check call operation)") < abort_scope.index(
    "(g3c4-native-call-abort (vector-ref record 5))")
assert "(vector 10 24 9 16)" in scope
assert "(eq? (vector-ref record index)" in scope
assert "(vector-ref anchor index)" in scope
assert "(define g3c4-p2g2-protected-anchors (vector '()))" in scope
assert "g3c4-native-call-finish" not in scope
assert "g3c4-native-call-prepare-end" not in scope
assert "g3c4-p2g1" not in scope
assert "g3c4-native-generation-frame-commit" not in scope
abort = native[native.index("int64_t et_g3c4_private_call_abort_v1("):]
assert abort.index("et_a2_kv_cache_transaction_view_begin_v1(") < abort.index(
    "et_g3c4_token_frame_discard(context)")
assert "defined(ET_G3C4_P2_G2_FIRST_FRAME_PRIVATE)" in abort
for phrase in ("equal-length raw replacement rejected",
               "equal-length staging replacement rejected",
               "paired auxiliary and record swap rejected",
               "prefix raw replacement rejected",
               "prefix staging replacement rejected",
               "unearned prefix phase rejected",
               "equal-shape output shell replacement rejected",
               "output ledger change rejected",
               "output native identity change rejected",
               "reservation cut keeps scope retryable",
               "model link change rejected",
               "model native identity change rejected",
               "tokenizer core identity change rejected",
               "lease rejects abort without mutation",
               "native pending I1 never published"):
    assert phrase in fixture, phrase
assert "p2_g2_protected_scope_native.c" in runner
assert "p2_g2_protected_scope_feature_off_test.esk" in runner
assert "g3c4_p2g2_ownership_extension.esk" not in feature_off
assert "protected scope rejects feature-off entry" in feature_off
assert "repeat-feature-off-run.stdout" in runner
assert "sanitize-feature-off-run.stdout" in runner
assert "compile_mode normal" in runner and "compile_mode sanitize" in runner
assert "cmp \"$evidence/normal-run.stdout\"" in runner
print("G3-C4 P2/G2 protected scope static PASS")
