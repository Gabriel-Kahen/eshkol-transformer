#!/usr/bin/env python3
"""Closed private Eshkol wrapper contract for reviewed G3-T manual decode."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_manual_decode_source_closure.txt").read_text().splitlines()
added = [
    "docs/g3/G3_T_MANUAL_DECODE_WRAPPER_PROPOSAL.md",
    "scripts/check-g3t-manual-decode-wrapper.py",
    "docs/g3/G3_T_MANUAL_DECODE_WRAPPER_LEAF.md",
]
closure = (root / "native/g3t_manual_decode_wrapper_source_closure.txt").read_text().splitlines()
assert closure == base + added
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))

source = (root / "native/g3t_prefill_sample_extension.esk").read_text()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()
call = source[source.index("(define (g3t-call-acquire "):source.index("(define (g3t-call-abort ")]
frame = source[source.index("(define (g3t-frame-begin "):source.index("(define (g3t-role-step ")]
assert "(or (= kind 0) (= kind 1) (= kind 2))" in call
assert call.index("(eq? (vector-ref m3-call-state 0) #t)") < call.index("(m3t-exact-i64? kind)")
assert call.index("(g3t-native-call-acquire") < call.index("(vector-set! generator-entry 9 entry)")
assert "(g3t-native-fail-raw operation)" in call
assert "(g3t-call-acquire generator-entry 2)" in source
assert frame.index("(g3t-prefill-call-entry") < frame.index("(g3t-prefill-entry input 'input operation)")
assert "(and (= frame-kind 1) input-entry" in frame
assert "(and (= frame-kind 2)" in frame
assert "(or (eq? input #f)" in frame
assert "(eq? (vector-ref input-entry 2) 'live)" in frame
assert "(if input-entry (vector-ref input-entry 3) #f)" in frame
assert "(g3t-native-fail-raw operation)" in frame
assert "(provide" not in source
assert re.findall(r"\(define \((g3t-[^\s)]+)", source) == [
    "g3t-prefill-dead!", "g3t-prefill-entry", "g3t-generation-token-input-create",
    "g3t-generation-tensor-release!", "g3t-prefill-call-linked",
    "g3t-prefill-call-entry", "g3t-call-acquire", "g3t-call-abort",
    "g3t-output-reserve", "g3t-output-prepare", "g3t-frame-begin",
    "g3t-role-step", "g3t-frame-commit", "g3t-frame-prepare", "g3t-sample",
]
for witness in (
    "manual decode wrapper missing prefix rolls back",
    "manual decode wrapper F32 reservation failure rolls back",
    "manual decode wrapper generated frame rejects non-null input",
    "manual decode wrapper forged input",
    "manual decode wrapper dead input",
    "manual decode wrapper null input",
    "manual decode wrapper wrong frame kind",
    "manual decode wrapper wrong length",
    "manual decode wrapper malformed I1",
    "manual decode wrapper inline I1 mismatch",
    "manual decode wrapper borrowed I1 rejected",
    "manual decode wrapper stale binding",
    "manual decode wrapper copied owned I1 can release",
    "manual decode wrapper P1 all 256 logits bit-exact to M3T row one",
    "manual decode wrapper P1 both K/V positions bit-exact to M3T",
    "manual decode wrapper P1 A2 lengths [1]=2 mask [1,2]=[1,1]",
    "manual decode wrapper full prefix rejects append",
    "manual decode wrapper P1 detached result survives close",
    "manual decode wrapper P1/G0 provider cut and abort",
    "manual decode wrapper A2 failure maps status",
    "manual decode wrapper pending logits blocks prepare",
    "manual decode wrapper P1/G0 all 256 logits bit-exact to M3T row one",
    "manual decode wrapper P1/G0 both K/V positions bit-exact to M3T",
    "manual decode wrapper P1/G0 A2 lengths [1]=2 mask [1,2]=[1,1]",
):
    assert witness in test, witness
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
assert "lengths->rank != 1 ||" in native
assert "lengths->shape[0] != 1 ||" in native
assert "((const int64_t *)lengths->data)[0] != 2" in native
assert "mask->rank != 2 ||" in native
assert "mask->shape[1] != 2 || mask->byte_length != 2" in native
assert "((const uint8_t *)mask->data)[1] != 1" in native
assert "scripts/check-g3t-manual-decode-wrapper.py" in runner
assert "native/g3t_manual_decode_wrapper_source_closure.txt" in runner
assert "-DET_G3T_MANUAL_DECODE_PRIVATE" in runner
print("G3-T private manual decode Eshkol wrapper source contract: PASS")
