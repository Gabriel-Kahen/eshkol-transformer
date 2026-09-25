#!/usr/bin/env python3
"""Closed source check for the private owned scalar-token G3-T input."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_generator_rng_input_source_closure.txt").read_text().splitlines()
additions = [
    "scripts/check-g3t-owned-token-input.py",
    "docs/g3/G3_T_OWNED_TOKEN_INPUT_LEAF.md",
]
closure = (root / "native/g3t_owned_token_input_source_closure.txt").read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))

native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()
constructor = native.split("void *et_g3t_private_input_from_token_v1", 1)[1].split(
    "void *et_g3t_private_input_from_t1_v1", 1
)[0]
release = native.split("int64_t et_g3t_private_tensor_release_v1", 1)[1].split(
    "void *et_g3t_private_output_reserve_v1", 1
)[0]
wrapper = (root / "native/g3t_prefill_sample_extension.esk").read_text().split(
    "(define (g3t-generation-token-input-create token)", 1
)[1].split("(define (g3t-generation-tensor-release!", 1)[0]
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()


def balanced(source: str) -> bool:
    depth = 0
    quoted = escaped = commented = False
    for char in source:
        if char == "\n":
            commented = False
        if commented:
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == ";":
            commented = True
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not quoted


assert balanced((root / "native/g3t_prefill_sample_extension.esk").read_text())
assert balanced(test)

assert "#ifdef ET_G3T_OWNED_TOKEN_INPUT_PRIVATE" in constructor
assert "const uint64_t shape[2] = {1, 1};" in constructor
assert constructor.index("token < 0 || token > 255") < constructor.index(
    "calloc(1, sizeof(*input))"
)
assert constructor.index("et_i64_tensor_create_v1") < constructor.index(
    "et_i64_tensor_copy_from_v1"
) < constructor.index("g3t_registry = &input->h")
assert "et_i64_tensor_destroy_v1(&input->tensor, &error)" in constructor
assert "defined(ET_G3T_OWNED_TOKEN_INPUT_PRIVATE)" in release
assert release.index("et_m3_private_i64_unborrowed_v1") < release.index(
    "et_i64_tensor_destroy_v1(&input->tensor, &error)"
)
assert wrapper.index("(next (cons entry") < wrapper.index(
    "(g3t-native-input-from-token token)"
)
assert "(g3t-native-tensor-release native)" in wrapper
assert "(g3t-prefill-dead! entry)" in wrapper
assert wrapper.index("(set! native created)") < wrapper.index(
    "(vector-set! canonical 2 'live)"
)
assert runner.count("-DET_G3T_OWNED_TOKEN_INPUT_PRIVATE") == 2
assert "native/g3t_owned_token_input_source_closure.txt" in runner
for witness in (
    "scalar native rejects negative ID before allocation",
    "scalar header cut atomic",
    "scalar I1 cut atomic",
    "scalar native owns exact singleton I1",
    "scalar I1 borrow blocks typed release",
    "scalar wrapper cut tombstones without owner",
    "scalar wrapper borrow blocks release",
    "P1/G0 scalar I1 prefill commits exact cache and RNG",
    "P1/G1 scalar input owns singleton I1",
):
    assert witness in test, witness
assert "(provide" not in wrapper
print("G3-T private owned scalar token input source contract: PASS")
