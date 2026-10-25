#!/usr/bin/env python3
"""Narrow source guard for I2 candidate begin recovery and seal tails."""

from pathlib import Path
import re


source = Path(__file__).resolve().parents[1] / "native/g3r_i2_candidate_construction_extension.esk"
text = source.read_text()


def section(name):
    start = text.index(f"(define ({name}")
    end = text.find("\n(define ", start + 1)
    return text[start:] if end < 0 else text[start:end]


begin = section("i2-candidate-construction-begin-internal!")
recover = begin.index("(if (vector-ref p1-cell 1)")
abort = begin.index("(module-construction-abort-internal!", recover)
rollback = begin.index("(i2-construction-rollback-tail!", recover)
assert recover < abort < rollback, "begin must abort rooted P1 identity before I2 rollback"
assert "(vector-ref canonical 1)" not in begin[recover:rollback]
assert begin.index("(vector-set! i2-construction-registry-root 0 next-records)") < begin.index(
    "(module-candidate-construction-begin-internal"
)
assert begin.index("(vector-set! i2-candidate-registry-root 0 next-candidates)") < begin.index(
    "(module-candidate-construction-begin-internal"
)

stage = section("i2-candidate-stage-head!")
assert "(vector-set! sidecar 10 (i2-modules-current))" in stage
assert "(eq? (vector-ref sidecar 10) (i2-modules-current))" in stage
assert "(i2-candidate-member-prefix!" in stage

allowed_tail = [
    ("record", "1", "#f"),
    ("record", "3", "'sealed"),
    ("record", "4", "'()"),
    ("record", "5", "'()"),
    ("record", "6", "'()"),
    ("record", "7", "0"),
    ("record", "8", "#f"),
    ("record", "10", "#f"),
    ("i2-construction-registry-root", "1", "#f"),
    ("sidecar", "3", "'sealed"),
    ("cell", "3", "count"),
    ("cell", "0", "'sealed"),
]
for name, native_call in (
    ("i2-candidate-construction-seal-internal!", "module-construction-seal-internal!"),
    (
        "i2-candidate-construction-seal-prepared-internal!",
        "module-construction-seal-prepared-internal!",
    ),
):
    body = section(name)
    assert body.index("(i2-candidate-stage-head! sidecar count operation)") < body.index(
        f"(let ((root ({native_call}"
    ), f"{name}: head must be staged before P1 seal"
    tail = body.split(";; The native P1 ", 1)[1].split("\n", 1)[1]
    tail = re.sub(r";[^\n]*", "", tail)
    assignments = re.findall(r"\(vector-set! ([\w-]+) (\d+) (#[ft]|'\(\)|'\w+|\w+)\)", tail)
    assert assignments == allowed_tail, f"{name}: postseal assignments changed: {assignments}"
    remainder = tail
    for owner, slot, value in allowed_tail:
        assignment = f"(vector-set! {owner} {slot} {value})"
        assert assignment in remainder
        remainder = remainder.replace(assignment, "", 1)
    assert re.fullmatch(r"\s*root\)\)\)\s*", remainder), (
        f"{name}: postseal tail contains more than fixed assignments"
    )

print("G3-R I2 candidate source failure-cut PASS")
