#!/usr/bin/env python3
"""Exact source and local-symbol contract for private G3-M P1 orchestration."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_manual_p1_prefill_source_closure.txt").read_text().splitlines()
added = [
    "native/g3m_prefill_p1_extension.esk",
    "native/g3m_prefill_p1_local_symbols.txt",
    "scripts/check-g3m-prefill-p1.py",
    "docs/g3/G3_M_PREFILL_P1_LEAF.md",
]
closure = (root / "native/g3m_prefill_p1_source_closure.txt").read_text().splitlines()
assert closure == base + added
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))
source = (root / added[0]).read_text()
symbols = (root / added[1]).read_text().splitlines()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()

assert symbols == ["g3m-prefill-p1!"]
assert re.findall(r"\(define \((g3m-[^\s)]+)", source) == symbols
assert source.count("(m3-call operation") == 1
assert source.index("(guard (caught") < source.index("(g3t-call-acquire")
assert source.index("(g3t-call-acquire") < source.index("(g3t-logits-reserve")
assert source.index("(g3t-logits-reserve") < source.index("(g3t-frame-begin")
assert source.index("(g3t-frame-begin") < source.index("(g3t-role-step call 0)")
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
assert "(g3t-prefill-entry input 'input operation)" in source
assert "(eq? (vector-ref input-entry 2) 'live)" in source
for witness in (
    "G3-M production P1 all 256 logits bit-exact to M3T",
    "G3-M production P1 K/V bit-exact to M3T",
    "G3-M second prefill replaces old P1 cache without RNG draw",
    "G3-M both detached logits survive parent close",
    "G3-M forged input rejected", "G3-M wrong-kind input rejected",
    "G3-M dead input rejected", "G3-M P2 rejected and rolled back",
    "G3-M busy context rejected", "G3-M I2 cut preserves old cache and ownership",
    "G3-M A2 cut preserves old cache and ownership",
    "G3-M old cache borrow rollback keeps prefix",
):
    assert witness in test, witness
assert '(load "g3m_prefill_p1_extension.esk")' in test
assert 'scripts/check-g3m-prefill-p1.py' in runner
assert 'native/g3m_prefill_p1_source_closure.txt' in runner
print("G3-M private P1 production orchestration source contract: PASS")
