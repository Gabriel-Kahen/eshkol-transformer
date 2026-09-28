#!/usr/bin/env python3
"""Closed source and private schedule contract for G3-M manual P2."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_manual_p2_prefill_source_closure.txt").read_text().splitlines()
prior = [
    "native/g3m_prefill_p1_extension.esk",
    "native/g3m_prefill_p1_local_symbols.txt",
    "scripts/check-g3m-prefill-p1.py",
    "docs/g3/G3_M_PREFILL_P1_LEAF.md",
]
added = [
    "native/g3m_prefill_p2_extension.esk",
    "native/g3m_prefill_p2_local_symbols.txt",
    "scripts/check-g3m-prefill-p2.py",
    "docs/g3/G3_M_PREFILL_P2_LEAF.md",
]
closure = (root / "native/g3m_prefill_p2_source_closure.txt").read_text().splitlines()
assert closure == base + prior + added
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))
source = (root / added[0]).read_text()
symbols = (root / added[1]).read_text().splitlines()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()

assert symbols == ["g3m-prefill-p2!"]
assert re.findall(r"\(define \((g3m-[^\s)]+)", source) == symbols
assert source.count("(m3-call operation") == 1
assert source.index("(guard (caught") < source.index("(g3t-call-acquire")
assert source.index("(g3t-call-acquire") < source.index("(g3t-logits-reserve")
assert source.index("(g3t-logits-reserve") < source.index("(g3t-frame-begin")
assert "(g3t-call-acquire generator-entry 0)" in source
assert "(g3t-frame-begin call input 1)" in source
assert "(g3t-prefill-entry input 'input operation)" in source
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
for witness in (
    "G3-M P2 forged input rejected",
    "G3-M P2 wrong-kind input rejected",
    "G3-M P2 dead input rejected",
    "G3-M P2 inline-only pair rejected",
    "G3-M P2 malformed typed I1 rejected",
    "G3-M P2 borrowed I1 rejected",
    "G3-M P2 busy context rejected",
    "G3-M P2 synthetic provider cut rollback",
    "G3-M P2 old-cache preflight rolls back",
    "G3-M P2 all 256 last-row logits bit-exact to M3T",
    "G3-M P2 both K/V rows bit-exact to M3T",
    "G3-M P2 repeat is bit-exact and preserves older output",
    "G3-M P2 detached logits survive parent close",
):
    assert witness in test, witness
assert '(load "g3m_prefill_p2_extension.esk")' in test
assert "scripts/check-g3m-prefill-p2.py" in runner
assert "native/g3m_prefill_p2_source_closure.txt" in runner
print("G3-M private P2 orchestration source contract: PASS")
