#!/usr/bin/env python3
"""Closed source check for private G3-T generator RNG input admission."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_rng_owner_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3t_generator_constructor_rng_source_closure.txt",
    "scripts/check-g3t-generator-rng-input.py",
    "docs/g3/G3_T_GENERATOR_RNG_INPUT_LEAF.md",
]
closure = (root / "native/g3t_generator_rng_input_source_closure.txt").read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))
extension = (root / "native/g3t_generator_constructor_extension.esk").read_text()
constructor = extension.split("(define (g3t-generator-create model tokenizer config)", 1)[1]
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
standalone = (root / "scripts/test-g3t-generator-constructor.sh").read_text()
historical = (root / "native/g3t_generator_constructor_source_closure.txt").read_text().splitlines()
current = (root / "native/g3t_generator_constructor_rng_source_closure.txt").read_text().splitlines()
assert current == historical + [
    "include/eshkol_transformer/g3n_primitives_abi.h",
    "include/eshkol_transformer/g3s_sampling_abi.h",
    "native/g3n_primitives_provider.c",
    "native/g3s_sampling_provider.c",
    "src/eshkol_transformer/g3t_prefill_roles.inc",
]
assert '"$PROJECT_ROOT/native/g3t_generator_constructor_rng_source_closure.txt"' in standalone
assert "(if (vector-ref seen 8) 'rng 'seed)" in extension
assert "(g3t-generator-find source operation)" in constructor
assert constructor.index("(g3t-generator-find source operation)") < constructor.index(
    "(vector-set! g3t-registry 0 next)"
) < constructor.index("(g3t-native-generator-rng")
assert "(g3t-native-generator-seed" in constructor
assert "(vector-ref rng-entry 3)" in constructor
for flag in (
    "ET_G3T_GENERATOR_RNG_PRIVATE",
    "ET_G3T_OUTPUT_RNG_CLONE_PRIVATE",
    "ET_G3T_FINAL_PUBLICATION_PRIVATE",
    "ET_G3T_OUTPUT_TEXT_PRIVATE",
    "ET_G3T_PREFILL_SAMPLE_PRIVATE",
):
    assert flag in standalone
for source in ("native/g3n_primitives_provider.c", "native/g3s_sampling_provider.c"):
    assert source in standalone
for witness in (
    "RNG constructor rejects forged shell",
    "RNG constructor rejects busy source",
    "RNG wrapper constructor header cut atomic",
    "RNG wrapper constructor A2 cut atomic",
    "P2/G0 Eshkol RNG constructor exact copy without draw",
    "P1/G0 Eshkol RNG constructor exact copy",
    "P1/G1 Eshkol RNG constructor copies final successor without draw",
):
    assert witness in test
print("G3-T private generator RNG input source contract: PASS")
