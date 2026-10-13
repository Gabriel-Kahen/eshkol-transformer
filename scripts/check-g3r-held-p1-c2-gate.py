#!/usr/bin/env python3
"""Bounded source closure for the unadmitted held-P1/C2 runtime fixture."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
fixture = (root / "tests/g3r/held_p1_c2_gate.esk").read_text()
runner = (root / "scripts/test-g3r-held-p1-c2-gate.sh").read_text()
leaves = (
    "g3r_held_o2_adopter_extension.esk",
    "g3r_joint_issued_o2_adopter_extension.esk",
    "g3r_joint_begin_extension.esk",
    "g3r_joint_c2_claim_extension.esk",
    "g3r_joint_original_owner_binding_extension.esk",
    "g3r_joint_p1_provenance_extension.esk",
)


def balanced(source: str) -> bool:
    depth = 0
    quoted = escaped = comment = False
    for char in source:
        if char == "\n":
            comment = False
        if comment:
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == ";":
            comment = True
        elif char == '"':
            quoted = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not quoted


def check(source: str, script: str) -> None:
    assert balanced(source)
    for leaf in leaves:
        assert source.count(f'(load "{leaf}")') == 1
        assert leaf not in (root / "Makefile").read_text()
    assert source.index('(load "c2_training_state_extension.esk")') < source.index(
        '(load "g3r_joint_issued_o2_adopter_extension.esk")')
    assert source.index('(load "g3r_joint_issued_o2_adopter_extension.esk")') < source.index(
        '(load "g3r_joint_begin_extension.esk")')
    assert source.index('(load "g3r_joint_c2_claim_extension.esk")') < source.index(
        '(load "g3r_joint_original_owner_binding_extension.esk")') < source.index(
        '(load "g3r_joint_p1_provenance_extension.esk")')
    for required in (
        "(g3r-joint-begin!", "(g3r-joint-o2-adopt!",
        "(g3r-joint-p1-provenance!", "(g3r-joint-c2-original-bound-claim!",
        "(g3r-joint-o2-drain-preclaim!", "(g3r-joint-o2-issued-match",
        "(g3r-joint-original-owner-match", "(g3r-joint-c2-copy-words",
        "(i2-checkpoint-encode", "(i2-parameter-gradient-metadata-internal",
        "(g3r-committed-save-snapshot!", "(c2-training-state-find",
        "(c2-training-state-release-internal!", "(state-dict-release! foreign-p1)",
        "(set! c2-training-state-claim-o2!", "(g3r-joint-o2-lifecycle",
        "(g3r-joint-p1-provenance-live-record",
    ):
        assert required in source, required
    assert source.index("(define before-bits") < source.index("(define held")
    assert source.index("(define before-history") < source.index("(define held")
    assert source.index("(define p1 (clone original held))") < source.index(
        "(define claimed (claim original held p1))")
    assert source.index("(define mutant-p1") < source.index(
        "(vector-set! mutant-p1 2 foreign-p1)") < source.index(
        "(g3r-joint-o2-drain-preclaim! (vector-ref mutant-held 1))")
    assert "(original-o2-claim state token)" in source
    assert script.count("-DET_G3R_SAME_OWNER_SAVE_PRIVATE") >= 4
    assert "native/g3r_joint_hold.c" in script
    assert "native/o2_optimizer.c" in script
    assert "tests/g3r/held_p1_c2_gate.esk" in script
    assert "--strict-types --optimize 0 --no-stdlib" in script


check(fixture, runner)
mutations = (
    fixture.replace('(load "g3r_joint_begin_extension.esk")', ""),
    fixture.replace('(g3r-joint-o2-adopt!', '(fake-o2-adopt!'),
    fixture.replace('(g3r-joint-p1-provenance!', '(module-state-dict'),
    fixture.replace('(g3r-joint-c2-original-bound-claim!', '(fake-claim!'),
    fixture.replace('(g3r-joint-o2-drain-preclaim!', '(fake-drain!'),
    fixture.replace('(g3r-joint-c2-copy-words', '(fake-copy-words'),
    fixture.replace('(i2-parameter-gradient-metadata-internal', '(fake-gradient'),
    fixture.replace('(original-o2-claim state token)', '#t'),
)
for index, mutation in enumerate(mutations):
    try:
        check(mutation, runner)
    except (AssertionError, ValueError):
        continue
    raise AssertionError(f"held P1/C2 mutation {index} escaped")
print(f"held P1/C2 source aggregate: pass ({len(mutations)} mutations; unadmitted)")
