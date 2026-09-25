#!/usr/bin/env python3
"""Closed source inventory for the private P1/G0 leaf."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_final_publication_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3t_zero_budget_extension.esk",
    "tests/g3t/zero_budget_test.esk",
    "scripts/check-g3t-zero-budget.py",
    "scripts/test-g3t-zero-budget.sh",
    "docs/g3/G3_T_ZERO_BUDGET_LEAF.md",
]
closure = (root / "native/g3t_zero_budget_source_closure.txt").read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / name).is_file() for name in additions)
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
source = (root / additions[0]).read_text()
test = (root / additions[1]).read_text()
assert "g3t_zero_preflight" in native
assert "ET_G3T_ZERO_BUDGET_PRIVATE" in native
assert "out->h.state = G3T_LIVE" in native
assert "(define (g3t-zero-generate!" in source
assert "all 256 prefill logits equal independent M3T" in test
assert "borrow blocks final preflight" in test
assert "G3T_ONLY_NORMAL" in (root / additions[3]).read_text()
assert "(provide" not in source
assert "g3c4" not in native.lower()
print("G3-T private P1/G0 source contract: PASS")
