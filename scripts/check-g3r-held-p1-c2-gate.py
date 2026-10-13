#!/usr/bin/env python3
"""Bounded source closure for the unadmitted held-P1/C2 runtime fixture."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
fixture = (root / "tests/g3r/held_p1_c2_gate.esk").read_text()
runner = (root / "scripts/test-g3r-held-p1-c2-gate.sh").read_text()
bit_bridge = (root / "tests/g3r/held_p1_c2_bit_bridge.c").read_text()
f32 = (root / "native/f32_tensor.c").read_text()
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


def check(source: str, script: str, bits: str) -> None:
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
    assert source.count('(load "m3_schedule.esk")') == 1
    assert source.count('(load "m3_model_extension.esk")') == 1
    assert source.index('(load "m3t_transport_extension.esk")') < source.index(
        '(load "m3_schedule.esk")') < source.index(
        '(load "m3_model_extension.esk")') < source.index(
        '(load "m3_call_adapters.esk")')
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
    assert source.index("(g3r-test-held-bits-begin native-model)") < source.index(
        "(define p1 (clone original held))") < source.index(
        "(g3r-test-held-bits-check native-model)") < source.index(
        "(g3r-test-held-bits-end native-model)") < source.index(
        "(define claimed (claim original held p1))")
    assert "(byte-list=? before-bits (model-bits (vector-ref original 5)))" in source
    assert "(equal? before-gradients\n                    (gradient-state (vector-ref original 5)))" in source
    assert source.index("(define mutant-p1") < source.index(
        "(vector-set! mutant-p1 2 foreign-p1)") < source.index(
        "(g3r-joint-o2-drain-preclaim! (vector-ref mutant-held 1))")
    assert "(original-o2-claim state token)" in source
    assert script.count("-DET_G3R_SAME_OWNER_SAVE_PRIVATE") >= 4
    assert "native/g3r_joint_hold.c" in script
    assert "native/o2_optimizer.c" in script
    assert "tests/g3r/held_p1_c2_bit_bridge.c" in script
    assert "tests/g3r/held_p1_c2_gate.esk" in script
    assert "--strict-types --optimize 0 --no-stdlib" in script
    assert "et_m3t_private_owner_save_roster_v1(model, parameters, identities)" in bits
    assert "et_f32_parameter_save_witness_begin_v1(" in bits
    assert "et_f32_parameter_save_witness_check_v1(" in bits
    assert "et_f32_parameter_save_witness_end_v1(" in bits
    assert "for (size_t index = 0; index < 14; ++index)" in bits
    witness = f32[f32.index("static int32_t save_witness_match("):f32.index(
        "int32_t et_f32_parameter_save_witness_begin_v1(")]
    assert "for (size_t k = 0; k < 2; ++k)" in witness
    assert "memcmp(t->data, w->baseline[k]->data, t->byte_length)" in witness
    assert "!tensor_exact_positive_zero(p->gradient)" in witness


check(fixture, runner, bit_bridge)
mutations = (
    (fixture.replace('(load "g3r_joint_begin_extension.esk")', ""), bit_bridge),
    (fixture.replace('(load "m3_schedule.esk")', ""), bit_bridge),
    (fixture.replace('(load "m3_model_extension.esk")', ""), bit_bridge),
    (fixture.replace('(g3r-joint-o2-adopt!', '(fake-o2-adopt!'), bit_bridge),
    (fixture.replace('(g3r-joint-p1-provenance!', '(module-state-dict'), bit_bridge),
    (fixture.replace('(g3r-joint-c2-original-bound-claim!', '(fake-claim!'), bit_bridge),
    (fixture.replace('(g3r-joint-o2-drain-preclaim!', '(fake-drain!'), bit_bridge),
    (fixture.replace('(g3r-joint-c2-copy-words', '(fake-copy-words'), bit_bridge),
    (fixture.replace('(i2-parameter-gradient-metadata-internal', '(fake-gradient'), bit_bridge),
    (fixture.replace('(g3r-test-held-bits-check native-model)', '0'), bit_bridge),
    (fixture.replace('(byte-list=? before-bits (model-bits (vector-ref original 5)))', '#t'), bit_bridge),
    (fixture.replace('(equal? before-gradients\n                    (gradient-state (vector-ref original 5)))', '#t'), bit_bridge),
    (fixture.replace('(original-o2-claim state token)', '#t'), bit_bridge),
    (fixture, bit_bridge.replace('et_f32_parameter_save_witness_check_v1(',
                                 'et_f32_parameter_save_witness_end_v1(')),
)
for index, (mutation, bridge_mutation) in enumerate(mutations):
    try:
        check(mutation, runner, bridge_mutation)
    except (AssertionError, ValueError):
        continue
    raise AssertionError(f"held P1/C2 mutation {index} escaped")
print(f"held P1/C2 source aggregate: pass ({len(mutations)} mutations; unadmitted)")
