#!/usr/bin/env python3
"""Mutation gate for the unlinked original-owner joint-begin bridge.

This is source evidence only; it cannot establish held-pin P1 clone behavior.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bridge = (root / "native/g3r_joint_begin_extension.esk").read_text()
native = (root / "native/g3r_joint_hold.c").read_text()


def check(source: str) -> None:
    assert source.count("(define (g3r-joint-begin!") == 1
    assert ":real et_g3r_joint_hold_begin_v1" in source
    begin = source[source.index("(define (g3r-joint-begin!"):]
    for required in (
        "(g3t-generator-find generator operation)",
        "(g3t-model-entry-live model operation #f)",
        "(o2-require-optimizer operation optimizer-value)",
        "(eq? (vector-ref generator-entry 0) generator)",
        "(eq? (vector-ref generator-entry 2) 'live)",
        "(eq? (vector-ref generator-entry 4) #f)",
        "(eq? (vector-ref generator-entry 8) #f)",
        "(eq? (vector-ref generator-entry 9) #f)",
        "(eq? (vector-ref generator-entry 10) #f)",
        "(eq? (vector-ref generator-entry 11) 'diagnostic-c2)",
        "(eq? (vector-ref generator-entry 5) model-entry)",
        "(eq? (vector-ref generator-entry 6) tokenizer)",
        "(g3r-save-tokenizer? tokenizer)",
        "(vector-ref generator-entry 3)",
        "(vector-ref model-entry 3)",
        "(vector-ref optimizer-record 1)",
        "(if (null? joint) (g3r-joint-o2-native-fail operation))",
    ):
        assert required in begin, required
    assert begin.index("(g3r-save-tokenizer? tokenizer)") < begin.index(
        "(let ((joint (g3r-joint-begin")
    assert begin.count("(g3r-joint-begin (vector-ref generator-entry 3)") == 1
    assert "(g3r-joint-begin (vector-ref generator-entry 3)\n" in begin
    assert "                                  (vector-ref model-entry 3)\n" in begin
    assert "                                  (vector-ref optimizer-record 1)" in begin
    held = native[native.index("et_g3r_joint_hold *et_g3r_joint_hold_begin_v1("):
                  native.index("int64_t et_g3r_joint_hold_check_v1(")]
    for required in (
        "et_m3t_private_owner_save_roster_v1(",
        "et_g3r_private_g3t_save_begin_v1(",
        "et_g3r_private_o2_save_begin_v1(",
        "hold->generator = generator;",
        "hold->model = sealed_model;",
        "hold->optimizer = optimizer;",
    ):
        assert required in held, required


check(bridge)
assert "g3r_joint_begin_extension.esk" not in (root / "Makefile").read_text()
mutations = (
    bridge.replace("(g3t-generator-find generator operation)", "#f"),
    bridge.replace("(g3t-model-entry-live model operation #f)", "#f"),
    bridge.replace("(o2-require-optimizer operation optimizer-value)", "#f"),
    bridge.replace("(g3r-save-tokenizer? tokenizer)", "#t"),
    bridge.replace("(eq? (vector-ref generator-entry 5) model-entry)", "#t"),
    bridge.replace("(eq? (vector-ref generator-entry 6) tokenizer)", "#t"),
    bridge.replace("(vector-ref model-entry 3)\n", "(vector-ref generator-entry 3)\n"),
    bridge.replace("(vector-ref optimizer-record 1))))\n", "(vector-ref model-entry 3))))\n"),
    bridge.replace("(if (null? joint) (g3r-joint-o2-native-fail operation))", "#t"),
)
for index, mutation in enumerate(mutations):
    try:
        check(mutation)
    except (AssertionError, ValueError):
        continue
    raise AssertionError(f"joint begin mutation {index} escaped")
print(f"joint begin source prerequisite: pass ({len(mutations)} mutations; unlinked)")
