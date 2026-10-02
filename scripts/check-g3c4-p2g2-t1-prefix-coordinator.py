#!/usr/bin/env python3
"""Closed source-private P2/G2 native-carrier/T1 prefix dependency."""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,
                str(root / "scripts/check-g3c4-p2g2-protected-scope.py")],
               cwd=root, check=True, stdout=subprocess.DEVNULL)
base = (root / "native/g3c4_p2g2_protected_scope_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3c4_p2g2_t1_prefix_coordinator_extension.esk",
    "tests/g3c4/p2_g2_t1_prefix_coordinator_native.c",
    "tests/g3c4/p2_g2_t1_prefix_coordinator_test.esk",
    "tests/g3c4/p2_g2_t1_prefix_coordinator_feature_off_test.esk",
    "tests/g3c4/check_p2_g2_t1_prefix_oracle.py",
    "scripts/check-g3c4-p2g2-t1-prefix-coordinator.py",
    "scripts/test-g3c4-p2g2-t1-prefix-coordinator.sh",
    "docs/g3/G3_C4_P2_G2_T1_PREFIX_COORDINATOR_LEAF.md",
    "native/g3c4_p2g2_t1_prefix_coordinator_source_closure.txt",
]
closure = (root / additions[-1]).read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure)

scope = (root / "native/g3c4_p2g2_protected_scope_extension.esk").read_text()
source = (root / additions[0]).read_text()
fixture = (root / additions[2]).read_text()
off = (root / additions[3]).read_text()
runner = (root / additions[6]).read_text()
oracle = (root / additions[4]).read_text()
native_fixture = (root / additions[1]).read_text()


def call_arity_at(text: str, start: int) -> int:
    """Count direct operands in one fixture call, ignoring nested forms."""
    depth = 1
    operands = 0
    atom = False
    quoted = False
    escaped = False
    comment = False
    for char in text[start:]:
        if comment:
            if char == "\n":
                comment = False
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == ";":
            comment = True
        elif char == '"':
            if depth == 1:
                operands += 1
            quoted = True
            atom = False
        elif char == "(":
            if depth == 1:
                operands += 1
            depth += 1
            atom = False
        elif char == ")":
            depth -= 1
            atom = False
            if depth == 0:
                return operands
        elif char.isspace():
            atom = False
        elif depth == 1 and not atom:
            operands += 1
            atom = True
    raise AssertionError("unterminated prefix probe call")


probe_name = "(g3c4-p2g2-prefix-commit-probed!"
probe_sites = []
cursor = 0
while (site := fixture.find(probe_name, cursor)) >= 0:
    probe_sites.append(call_arity_at(fixture, site + len(probe_name)))
    cursor = site + len(probe_name)
assert probe_sites == [4, 4], probe_sites
assert "(eq? (vector-ref record index)" in scope
assert "(vector-ref anchor index)" in scope
assert "'p2g2-prefix-committed" in scope
assert "(define (g3c4-p2g2-owned-input-create" in source
assert "(define-syntax g3c4-with-p2g2-prefix-call" in source
assert "(eq? (vector-ref anchor 3) native-input)" in source
assert "(eq? (vector-ref anchor 4) (vector-ref record 1))" in source
assert "et_g3c4_private_input_from_t1_p2_v1" in source
assert "et_g3c4_private_prompt_prefill_preflight_v1" in source
assert "et_g3c4_private_p2g2_first_frame_carrier_v1" in source
assert "et_g3c4_private_p2g2_prefix_commit_v1" in source
body = source[source.index("(define (g3c4-p2g2-prefix-commit-probed!"):]
assert body.index("(g3c4-p2g2-native-first-frame") < body.index(
    "(t1-private-g3-decode-raw-into!") < body.index(
    "(g3c4-p2g2-native-prefix-commit") < body.index(
    "(vector-set! (g3c4-p2g2-anchor-for call) 13")
assert body.count("(g3c4-p2g2-prefix-input-check") >= 3
assert body.index("(before-decode record)") < body.index(
    "(t1-private-g3-decode-raw-into!") < body.index(
    "(after-decode record)")
assert body.index("(g3c4-p2g2-prefix-byte-check") < body.index(
    "(g3c4-p2g2-native-prefix-commit")
assert "(g3c4-p2g2-protected-abort! call)" in fixture
for phrase in ("native prefix is committed, output stays unready",
               "held lease rejects abort in-region",
               "P3 logits and K/V bitwise direct reference",
               "rollback cache, parameters, gradients and RNG",
               "all captured carrier bytes scrubbed and child dead",
               "bridge failure never commits prefix"):
    assert phrase in fixture, phrase
assert "g3c4_p2g2_ownership_extension.esk" not in off
assert "coordinator preflight rejects feature-off entry" in off
assert "-DET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE" in runner
assert "-DET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE" in runner
assert "-DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE" in runner
assert "-DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE -DET_T1_I64_SHELL_TESTING" in runner
assert ("-DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE \\\n"
        "    -DET_I64_TENSOR_STORAGE_QUERY_PRIVATE" in runner)
assert "et_t1_i64_shell_private_c4_read_v1$'" in runner
assert "et_i64_tensor_private_t1_pair_validate_v1$'" in runner
assert "__wrap_et_i64_tensor_borrow_view_v1$'" in runner
assert "--wrap=et_i64_tensor_borrow_view_v1" in runner
assert "--wrap=et_g3c4_private_p2g2_prefix_commit_v1" in runner
assert "--wrap=et_a2_kv_cache_transaction_commit_v1" in runner
assert "check_p2_g2_t1_prefix_oracle.py" in runner
assert "COORD_ORACLE " in oracle
assert "reference.philox" in oracle and "reference.literal" in oracle
assert "coordinator_entry_binding" in native_fixture
assert "coordinator_precommit_binding" in native_fixture
assert "coordinator_prefix_binding" in native_fixture
assert "coordinator_binding_equal" in native_fixture
assert "COORD_ORACLE %s" in native_fixture
for phrase in ("impossible closed native tail fails stop",
               "different owned P2 input rejected after preflight",
               "equal-length raw byte mutation rejected",
               "equal-length staging upper byte rejected",
               "selected first-token EOS rejects before commit",
               "(probe-case 'decoder)", "(probe-case 'core)",
               "(probe-case 'registry)", "(probe-case 'model)",
               "(probe-case 'pin)", "(probe-case 'raw)",
               "(probe-case 'staging)"):
    assert phrase in fixture, phrase
for phrase in ("post-prefix wrong child ledger rejected",
               "post-prefix wrong output shell rejected",
               "post-prefix wrong output native rejected",
               "post-prefix paired equal-length auxiliary rejected"):
    assert phrase in fixture, phrase
for mode in range(1, 13):
    assert f"#f {mode} 'none)" in fixture or mode == 0
assert "compile_mode normal" in runner and "compile_mode sanitize" in runner
assert "repeat-feature-off-run.stdout" in runner
assert "sanitize-feature-off-run.stdout" in runner
assert "cmp \"$evidence/normal-run.stdout\"" in runner
assert "g3c4-native-call-finish" not in source
assert "g3c4-native-generation-frame-commit" not in source
print("G3-C4 P2/G2 T1 prefix coordinator static PASS")
