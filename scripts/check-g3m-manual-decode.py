#!/usr/bin/env python3
"""Closed source and schedule contract for private G3-M one-token decode."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_manual_decode_wrapper_source_closure.txt").read_text().splitlines()
added = [
    "native/g3m_manual_decode_extension.esk",
    "native/g3m_manual_decode_local_symbols.txt",
    "scripts/check-g3m-manual-decode.py",
    "docs/g3/G3_M_MANUAL_DECODE_COMPOSER_PROPOSAL.md",
    "docs/g3/G3_M_MANUAL_DECODE_LEAF.md",
]
closure = (root / "native/g3m_manual_decode_source_closure.txt").read_text().splitlines()
assert closure == base + added
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))

source = (root / added[0]).read_text()
symbols = (root / added[1]).read_text().splitlines()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()
assert symbols == ["g3m-decode-one!"]
assert re.findall(r"\(define \((g3m-[^\s)]+)", source) == symbols
assert source.count("(m3-call operation") == 1
assert source.index("(g3t-generator-find") < source.index("(g3t-prefill-entry input 'input operation)")
assert source.index("(guard (caught") < source.index("(g3t-call-acquire")
assert source.index("(g3t-call-acquire") < source.index("(g3t-logits-reserve")
assert source.index("(g3t-logits-reserve") < source.index("(g3t-frame-begin")
assert "(g3t-call-acquire generator-entry 1)" in source
assert "(g3t-frame-begin call input 2)" in source
assert "(eq? (vector-ref input-entry 2) 'live)" in source
roles = [int(x) for x in re.findall(r"\(g3t-role-step call (\d+)\)", source)]
assert roles == list(range(21)), roles
assert "(let roles" not in source and "(lambda " not in source
tail = source[source.index("(g3t-role-step call 20)"):]
assert re.search(
    r"\(g3t-frame-prepare call logits\)\s*"
    r"\(g3t-call-prepare-end call\)\s*"
    r"\(g3t-frame-commit call\)\s*"
    r"(?:;[^\n]*\n\s*)*\(set! committed #t\)\s*"
    r"\(g3t-call-finish call\)", tail,
)
assert "(if committed (exit 134))" in source
assert "(if call (g3t-call-abort call))" in source
assert "(provide" not in source
for witness in (
    "G3-M decode forged input rejected",
    "G3-M decode wrong-kind input rejected",
    "G3-M decode dead input rejected",
    "G3-M decode no-prefix rollback",
    "G3-M decode length-two input rejected",
    "G3-M decode malformed owned I1 rejected",
    "G3-M decode borrowed owned I1 rejected",
    "G3-M decode stale binding rejected",
    "G3-M decode busy context rejected",
    "G3-M decode F32 reservation rollback",
    "G3-M decode A2 failure rollback",
    "G3-M decode synthetic provider failure rollback",
    "G3-M decode P1 all 256 logits bit-exact to M3T",
    "G3-M decode P1 both K/V positions bit-exact to M3T",
    "G3-M decode full-cache rejected",
    "G3-M decode P1/G0 all 256 logits bit-exact to M3T",
    "G3-M decode P1/G0 both K/V positions bit-exact to M3T",
    "G3-M decode fresh-call repeat matches P1/G0",
    "G3-M decode detached logits survive parent close",
):
    assert witness in test, witness
assert '(load "g3m_manual_decode_extension.esk")' in test
assert "scripts/check-g3m-manual-decode.py" in runner
assert "native/g3m_manual_decode_source_closure.txt" in runner
assert "-DET_G3T_MANUAL_DECODE_PRIVATE" in runner
print("G3-M private one-token decode source contract: PASS")
