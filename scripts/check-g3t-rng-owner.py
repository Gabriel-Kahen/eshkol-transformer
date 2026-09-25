#!/usr/bin/env python3
"""Closed source and transaction check for the private G3-T RNG mapping."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_t1_input_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3t_rng_owner_extension.esk",
    "scripts/check-g3t-rng-owner.py",
    "docs/g3/G3_T_RNG_OWNER_LEAF.md",
]
closure = (root / "native/g3t_rng_owner_source_closure.txt").read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))
extension = (root / additions[0]).read_text()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()


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


assert balanced(extension) and balanced(test)
assert balanced((root / "native/g3t_generator_constructor_extension.esk").read_text())
accessor = extension.split("(define (g3t-generation-output-rng output)", 1)[1].split(
    "(define (g3t-generation-rng-release! rng)", 1
)[0]
assert accessor.index("g3t-prefill-entry output 'output") < accessor.index(
    "(let* ((shell"
)
assert accessor.index("(let* ((shell") < accessor.index(
    "(next (cons entry"
) < accessor.index("(g3t-native-output-rng-clone")
assert accessor.index("(vector-set! g3t-registry 0 next)") < accessor.index(
    "(g3t-native-output-rng-clone"
)
assert accessor.index("(g3t-native-rng-release created)") < accessor.index(
    "(raise caught)"
)
assert accessor.index("(g3t-native-output-rng-clone") < accessor.index(
    "(vector-set! canonical 2 'live)"
)
assert "(provide" not in extension
assert "g3t-native-generator-rng" not in extension
assert '(load "g3t_rng_owner_extension.esk")' in test
for witness in (
    "RNG wrapper cut enrolls no live owner",
    "P2/G0 RNG shell survives parent close and output release",
    "P1/G0 RNG shell survives parent releases",
    "RNG wrapper typed release and tombstone",
):
    assert witness in test
print("G3-T private RNG owner source contract: PASS")
