#!/usr/bin/env python3
"""Closed source and admission-order contract for private C2 preflight."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_text_result_source_closure.txt").read_text().splitlines()
additions = [
    "scripts/check-g3t-full-request-preflight.py",
    "docs/g3/G3_T_FULL_REQUEST_PREFLIGHT_LEAF.md",
]
closure = (root / "native/g3t_full_request_preflight_source_closure.txt").read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
header = (root / "src/eshkol_transformer/g3t_transport.h").read_text()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()
name = "et_g3t_private_full_request_preflight_v1"
assert native.count(name) == header.count(name) == 1
start = native.index("int64_t " + name)
end = native.index("\n#endif", start)
body = native[start:end]
for phrase in (
    "g3t_admit(context_candidate, 0)",
    "g3t_admit_record(\n      input_candidate, G3T_INPUT, 0)",
    "input->length < 1 || input->length > 2",
    "c->policy[4] > 2 - input->length",
    "g3t_model(c->model)",
    "c->h.busy || o->active || c->pins.held_mask || !c->cache",
    "c->prefill_committed || input->h.busy",
    "g3t_bad(G3T_STATE, G3T_DRAW_EXHAUSTION)",
):
    assert phrase in body, phrase
assert body.index("g3t_admit(context_candidate, 0)") < body.index("input->length")
assert body.index("g3t_admit_record") < body.index("input->ids[i]")
for forbidden in ("g3t_registry =", "pins_begin", "transaction_begin", "calloc("):
    assert forbidden not in body, forbidden
assert native.count("int64_t et_g3t_private_prompt_preflight_v1(") == 1
assert "input->length != 2 || input->length + c->policy[4] > 2" in native
assert "G3T_DRAW_EXHAUSTION = 12" in native
assert "et_g3t_test_rng_counter_set_v1" in native
assert "#ifdef ET_G3T_TESTING" in native
assert "-DET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE" in runner
assert "native/g3t_full_request_preflight_source_closure.txt" in runner
for witness in (
    "full request P1/G0 admitted before pins",
    "full request P1/G1 admitted before pins",
    "full request P2/G0 admitted before pins",
    "full request P2/G1 rejected before pins and draw",
    "P2-only helper still rejects P1 before pins",
    "full request rejects forged generator",
    "full request rejects forged input",
    "full request rejects dead input",
    "full request rejects busy generator and counter mutation",
    "categorical G1 exhaustion is 2/12 before pins",
    "greedy G1 exhausted RNG still admitted",
):
    assert witness in test, witness
print("G3-T private C2 full-request preflight source contract: PASS")
