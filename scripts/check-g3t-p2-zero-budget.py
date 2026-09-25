#!/usr/bin/env python3
"""Closed source inventory for the private P2/G0 generation leaf."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_zero_budget_source_closure.txt").read_text().splitlines()
additions = [
    "src/eshkol_transformer/g3t_prefill2_roles.inc",
    "native/g3t_p2_zero_budget_extension.esk",
    "tests/g3t/p2_zero_budget_test.esk",
    "scripts/check-g3t-p2-zero-budget.py",
    "scripts/test-g3t-p2-zero-budget.sh",
    "docs/g3/G3_T_P2_ZERO_BUDGET_LEAF.md",
]
closure = (root / "native/g3t_p2_zero_budget_source_closure.txt").read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in additions)
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
roles = (root / additions[0]).read_text()
source = (root / additions[1]).read_text()
test = (root / additions[2]).read_text()
assert "et_g3t_private_prompt_preflight_v1" in native
assert "input->length + c->policy[4] > 2" in native
assert "g3t_prefill2_attention" in roles and "g3t_prefill2_plain" in roles
assert "et_n3k_kernel_provider_v1" in roles
assert "et_a2_kv_cache_transaction_begin_v1" in roles
assert "(define (g3t-p2-zero-generate!" in source
assert source.index("g3t-native-prompt-preflight") < source.index("g3t-call-acquire generator-entry 2")
assert "all 256 P2 logits equal independent M3T" in test
assert "P2/G1 rejected before pins" in test
assert "(provide" not in source
assert "g3c4" not in native.lower()
print("G3-T private P2/G0 source contract: PASS")
