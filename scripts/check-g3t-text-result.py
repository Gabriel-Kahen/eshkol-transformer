#!/usr/bin/env python3
"""Closed source and no-alias check for the private G3-T text result."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / "native/g3t_length_owners_source_closure.txt").read_text().splitlines()
additions = [
    "native/g3t_text_result_extension.esk",
    "scripts/check-g3t-text-result.py",
    "docs/g3/G3_T_TEXT_RESULT_LEAF.md",
]
closure = (root / "native/g3t_text_result_source_closure.txt").read_text().splitlines()
assert closure == base + additions
assert len(closure) == len(set(closure))
assert all((root / path).is_file() for path in closure if not path.startswith(".deps/"))

source = (root / additions[0]).read_text()
test = (root / "tests/g3t/p2_zero_budget_test.esk").read_text()
runner = (root / "scripts/test-g3t-p2-zero-budget.sh").read_text()
prefill = (root / "native/g3t_prefill_sample_extension.esk").read_text()
publication = (root / "native/g3t_final_publication_extension.esk").read_text()


def balanced(code: str) -> bool:
    depth = 0
    quoted = escaped = commented = False
    for char in code:
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


assert balanced(source) and balanced(test)
assert source.count("(define (g3t-generation-output-text output)") == 1
assert source.index("g3t-prefill-entry output 'output") < source.index(
    "(detached (make-bytevector length 0))"
)
assert source.index("(detached (make-bytevector length 0))") < source.index(
    "(result (list detached))"
) < source.index("(bytevector-u8-set! detached 0")
assert "(bytevector-u8-ref raw 0)" in source
assert "(= length 1)" in source and "(> length 1)" in source
assert "(vector-ref source 6)" in source
assert "(vector-set! g3t-registry" not in source
assert "(extern " not in source and "(provide " not in source
assert "(raw (make-bytevector budget 0))" in prefill
assert "(vector-set! output-entry 2 'live)" in publication
assert '(load "g3t_text_result_extension.esk")' in test
assert "scripts/check-g3t-text-result.py" in runner
assert "native/g3t_text_result_source_closure.txt" in runner
for witness in (
    "text result rejects forged output",
    "text result rejects wrong-kind input",
    "text result rejects pending output",
    "text result rejects dead output",
    "P2/G0 text results are fresh empty bytevectors in fresh lists",
    "P2/G0 text copy survives output and generator release",
    "P1/G0 text result is one detached empty bytevector",
    "P1/G0 text survives parent releases",
    "P1/G1 text results are exact detached emitted bytes",
    "P1/G1 text mutations leave source and next result unchanged",
    "P1/G1 text copy survives parent releases",
):
    assert witness in test, witness
print("G3-T private detached text result source contract: PASS")
