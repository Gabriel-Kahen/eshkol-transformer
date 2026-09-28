#!/usr/bin/env python3
"""Closed source and schedule contract for private G3-M seeded P1/G1."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3m_manual_decode_source_closure.txt").read_text().splitlines()
added = [
    "native/g3m_seeded_p1_g1_extension.esk",
    "native/g3m_seeded_p1_g1_local_symbols.txt",
    "scripts/check-g3m-seeded-p1-g1.py",
    "docs/g3/G3_M_PRIVATE_SEEDED_P1_G1_PROPOSAL.md",
    "docs/g3/G3_M_PRIVATE_SEEDED_P1_G1_LEAF.md",
]
closure = (root / "native/g3m_seeded_p1_g1_source_closure.txt").read_text().splitlines()
assert closure == base + added
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))
source = (root / added[0]).read_text()
symbols = (root / added[1]).read_text().splitlines()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()
assert symbols == ["g3m-native-full-request-preflight", "g3m-generate-p1-g1!"]
assert re.findall(r"\(define \((g3m-[^\s)]+)", source) == symbols[1:]
assert source.count("(m3-call operation") == 1
assert source.index("(g3t-generator-find") < source.index("(g3t-prefill-entry input 'input operation)")
assert source.index("budget one") < source.index("(g3m-native-full-request-preflight")
assert source.index("(g3m-native-full-request-preflight") < source.index("(g3t-call-acquire")
assert source.index("(guard (caught") < source.index("(g3t-call-acquire")
assert "(g3t-call-acquire generator-entry 2)" in source
assert "(g3t-output-reserve call 1)" in source
assert "(g3t-frame-begin call input 1)" in source
assert "(g3t-frame-begin call #f 2)" in source
roles = [int(x) for x in re.findall(r"\(g3t-role-step call (\d+)\)", source)]
assert roles == list(range(21)) * 2, roles
assert "(let roles" not in source and "(lambda " not in source
sequence = [
    "(g3t-frame-prepare call #f)", "(g3t-frame-commit call)",
    "(g3t-sample call)", "(g3t-frame-begin call #f 2)",
    "(g3t-output-prepare call output)",
    "(g3t-t1-decode-output! call output)",
    "(g3t-final-commit! call output)",
]
positions = [source.index(x) for x in sequence]
assert positions == sorted(positions)
assert "(if call" in source and "(g3t-call-abort call)" in source
assert "(provide" not in source
for witness in (
    "G3-M seeded budget-zero rejected before pins",
    "G3-M seeded P2/G1 rejected before pins",
    "G3-M seeded forged input rejected",
    "G3-M seeded wrong-kind input rejected",
    "G3-M seeded wrong-kind generator rejected",
    "G3-M seeded dead input rejected",
    "G3-M seeded prefill failure rolls back all state",
    "G3-M seeded append failure retains prompt-only cache",
    "G3-M seeded A2 failure rolls back before prompt",
    "G3-M seeded exhausted categorical rejects before pins",
    "G3-M seeded genuine T1-backed P1/G1 publishes",
    "G3-M seeded borrowed I1 read-only generated route",
    "G3-M seeded borrowed I1 release blocked",
    "G3-M seeded old-cache borrow rejects",
    "G3-M seeded old-cache borrow keeps empty cache",
    "G3-M seeded old-cache borrow keeps RNG",
    "G3-M seeded old-cache borrow drains pins",
    "G3-M seeded old-cache failure same-generator retry publishes",
    "G3-M seeded stale binding after sample rejected",
    "G3-M seeded categorical sample and final publication",
    "G3-M seeded prefill all 256 logits bit-exact to M3T",
    "G3-M seeded both K/V positions bit-exact to M3T",
    "G3-M seeded append all 256 logits bit-exact to M3T",
    "G3-M seeded output text and clones",
    "G3-M seeded RNG-source continuation copies independent state",
    "G3-M seeded emitted EOS included after one append",
    "G3-M seeded unequal EOS still emits budget one",
    "G3-M seeded byte-zero prompt",
    "G3-M seeded byte-255 prompt",
    "G3-M seeded greedy emits without exhausted RNG draw",
    "G3-M seeded borrowed output release rejected",
    "G3-M seeded detached output survives generator close",
):
    assert witness in test, witness
assert '(load "g3m_seeded_p1_g1_extension.esk")' in test
assert "scripts/check-g3m-seeded-p1-g1.py" in runner
assert "native/g3m_seeded_p1_g1_source_closure.txt" in runner
assert "-DET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE" in runner
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
assert "et_g3t_test_generated_logit_bits_v1" in native
assert "#ifdef ET_G3T_TESTING\n    memcpy(c->test_generated_logits" in native
assert "et_g3t_test_flip_binding_on_sample_v1" in native
assert "#ifdef ET_G3T_TESTING\n  if (c->test_flip_binding_on_sample)" in native
print("G3-M private seeded P1/G1 source contract: PASS")
