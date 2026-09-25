#!/usr/bin/env python3
"""Closed source and publication-order check for private G3-T ID owners."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_owned_token_input_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3t_ids_owner_extension.esk",
    "scripts/check-g3t-ids-owner.py",
    "docs/g3/G3_T_IDS_OWNER_LEAF.md",
]
closure = (root / "native/g3t_ids_owner_source_closure.txt").read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))

extension = (root / additions[0]).read_text()
prefill = (root / "native/g3t_prefill_sample_extension.esk").read_text()
release = prefill.split("(define (g3t-generation-tensor-release! shell)", 1)[1].split(
    "(define (g3t-prefill-call-linked", 1
)[0]
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()
native = (root / "src/eshkol_transformer/g3t_transport.c").read_text()


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


assert balanced(extension) and balanced(prefill) and balanced(test)
assert extension.count("(define (g3t-generation-output-ids output)") == 1
assert extension.index("g3t-prefill-entry output 'output") < extension.index(
    "(let* ((shell"
)
assert extension.index("(shell (vector") < extension.index(
    "(next (cons entry"
) < extension.index("(result (list shell))") < extension.index(
    "(g3t-native-output-ids-clone"
)
assert extension.index("(vector-set! g3t-registry 0 next)") < extension.index(
    "(g3t-native-output-ids-clone"
)
assert extension.index("(g3t-native-tensor-release created)") < extension.index(
    "(g3t-prefill-dead! entry)"
) < extension.index("(raise caught)")
assert extension.index("(set! created native)") < extension.index(
    "(vector-set! canonical 2 'live)"
)
assert all(f"(eq? kind '{kind})" in release for kind in ("input", "ids"))
assert "(g3t-prefill-entry shell kind operation)" in release
assert "(g3t-native-tensor-release" in release
assert "et_g3t_private_output_ids_clone_v1" in extension
assert "et_g3t_private_output_ids_clone_v1" in native
assert "#ifdef ET_G3T_OUTPUT_IDS_CLONE_PRIVATE" in native
assert "(provide" not in extension
assert '(load "g3t_ids_owner_extension.esk")' in test
assert "native/g3t_ids_owner_source_closure.txt" in runner
for witness in (
    "IDs wrapper rejects forged output",
    "IDs wrapper rejects pending output",
    "IDs wrapper header cut atomic",
    "IDs wrapper I1 cut atomic",
    "P2/G0 IDs wrapper returns one rooted detached I1",
    "IDs wrapper creates a fresh list and native owner",
    "IDs wrapper borrow blocks typed release",
    "IDs wrapper borrowed output rejects clone",
    "P2/G0 IDs wrapper survives parent releases",
    "P1/G0 IDs wrapper returns one empty detached owner",
    "P1/G1 IDs wrapper returns exact one-token detached I1",
):
    assert witness in test, witness
print("G3-T private generated-ID owner source contract: PASS")
