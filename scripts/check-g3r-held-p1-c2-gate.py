#!/usr/bin/env python3
"""Bounded source closure for the unadmitted held-P1/C2 runtime fixture."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
fixture = (root / "tests/g3r/held_p1_c2_gate.esk").read_text()
bridge = (root / "native/cli3_generate_t2_bridge.esk").read_text()
runner = (root / "scripts/test-g3r-held-p1-c2-gate.sh").read_text()
symbol_probe = (root / "scripts/check-g3r-f32-allocator-symbol.sh").read_text()
bit_bridge = (root / "tests/g3r/held_p1_c2_bit_bridge.c").read_text()
f32 = (root / "native/f32_tensor.c").read_text()
m3_call_f32 = (root / "src/eshkol_transformer/m3_call_f32_integration.c").read_text()
m3t_f32 = (root / "src/eshkol_transformer/m3t_f32_integration.c").read_text()
corpus_builder = (root / "tests/d2/prepare_public_resources.py").read_text()
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


def compile_command(script: str, header: str) -> str:
    lines = script.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.rstrip() == header)
    end = start + 1
    while lines[end - 1].rstrip().endswith("\\"):
        end += 1
    return "".join(lines[start:end])


def check(source: str, script: str, bits: str) -> None:
    assert balanced(source)
    for leaf in leaves:
        assert source.count(f'(load "{leaf}")') == 1
        assert leaf not in (root / "Makefile").read_text()
    assert source.count('(load "cli3_generate_t2_bridge.esk")') == 1
    assert "(define t1-private-entry t2-private-entry)" in bridge
    assert "(t2-private-raise category operation message)" in bridge
    assert "(define t1-wave1-native-fail t2-wave2-native-fail)" in bridge
    assert source.index('(load "d2_wave2_root.esk")') < source.index(
        '(load "cli3_generate_t2_bridge.esk")') < source.index(
        '(load "g3t_generator_constructor_extension.esk")')
    assert "(t1 (t2-private-tokenizer-byte config))" in source
    assert "(t2 (t2-wave2-tokenizer-byte config))" in source
    assert "(not (eq? t1 t2))" in source
    assert ("(equal? (vector-ref (vector-ref p1 3) 0)\n"
            "                    (handle-paths (vector-ref original 5)))") in source
    assert "(define original-topology-query g3r-joint-p1-topology-query)" in source
    assert "(set! g3r-joint-p1-topology-query original-topology-query)" in source
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
    assert "output.mkdir(parents=True)" in corpus_builder
    setup = script.split('evidence="$(readlink -f "$evidence")"', 1)[0]
    assert '"$evidence/corpus"' not in setup
    assert '--output "$evidence/corpus"' in script
    provider = source[source.index("(define provider-fault (fixture corpus-directory))"):
                      source.index(";;; C2's injected O2 claim defect")]
    assert provider.index("(g3r-test-held-bits-begin provider-fault-native-model)") < provider.index(
        "(g3t-test-fail-f32-after 0)") < provider.index(
        "(clone provider-fault provider-fault-held) #f") < provider.index(
        "(g3t-test-fail-f32-after 9223372036854775807)") < provider.index(
        "(clone provider-fault provider-fault-held))")
    for required in (
        "(transformer-error-category provider-fault-error) 'internal",
        "(transformer-error-operation provider-fault-error)",
        "'module-state-dict", "(i2-owned-clone-live-count-internal)",
        "g3r-joint-p1-provenance-registry", "c2-training-state-states",
        "(g3r-test-held-bits-check provider-fault-native-model)",
        "(g3r-test-held-bits-end provider-fault-native-model)",
        "(g3r-joint-o2-issued-match", "(g3r-joint-original-owner-match",
        "(model-bits (vector-ref provider-fault 5))",
        "(gradient-state (vector-ref provider-fault 5))",
        "(g3r-joint-c2-copy-words", "(g3r-joint-p1-provenance-release! provider-fault-retry)",
        "(g3r-joint-o2-drain-preclaim!",
    ):
        assert required in provider, required
    assert script.count("-DET_G3R_SAME_OWNER_SAVE_PRIVATE") >= 4
    assert m3_call_f32.count('#include "m3t_f32_integration.c"') == 1
    assert m3t_f32.count('#include "../../native/f32_tensor.c"') == 1
    assert "compile native/f32_tensor.c f32_tensor" not in script
    for header in ("compile src/eshkol_transformer/m3_call_f32_integration.c m3_call_f32_integration \\",
                   "compile native/o2_optimizer.c o2_optimizer \\"):
        command = compile_command(script, header)
        assert "-DET_G3R_SAME_OWNER_SAVE_PRIVATE" in command
        assert "-DET_G3R_CANDIDATE_RETIRE_PRIVATE" in command
    assert '"$PROJECT_ROOT/scripts/check-g3r-f32-allocator-symbol.sh" \\' in script
    assert '"$evidence/native/m3_call_f32_integration.o"' in script
    assert '"$evidence/native/m3_call_f32_integration.defined-symbols.txt"' in script
    assert 'if ar t "$archive" | grep -Fxq f32_tensor.o; then' in script
    assert 'counts["et_f32_tensor_test_fail_alloc_after_v1"] != 1' in script
    assert 'if repeated:' in script
    assert 'LC_ALL=C nm -g --defined-only --format=posix "$object" > "$symbols"' in symbol_probe
    assert "grep -Eq '^et_f32_tensor_test_fail_alloc_after_v1 T [[:xdigit:]]+ [[:xdigit:]]+$'" in symbol_probe
    assert "-DET_F32_TENSOR_TESTING" in script
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
    assert "if (successful_allocations >= allocation_limit)" in f32
    assert "void et_f32_tensor_test_fail_alloc_after_v1(size_t allowed)" in f32


check(fixture, runner, bit_bridge)
mutations = (
    (fixture.replace('(load "cli3_generate_t2_bridge.esk")', ""), bit_bridge),
    (fixture.replace('(not (eq? t1 t2))', '#t'), bit_bridge),
    (fixture.replace('(set! g3r-joint-p1-topology-query original-topology-query)',
                     '#t'), bit_bridge),
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
    (fixture.replace('(g3t-test-fail-f32-after 0)', '(g3t-test-fail-f32-after 9223372036854775807)'), bit_bridge),
    (fixture.replace('(g3t-test-fail-f32-after 9223372036854775807)', '(g3t-test-fail-f32-after 0)'), bit_bridge),
    (fixture, bit_bridge.replace('et_f32_parameter_save_witness_check_v1(',
                                 'et_f32_parameter_save_witness_end_v1(')),
)
for index, (mutation, bridge_mutation) in enumerate(mutations):
    try:
        check(mutation, runner, bridge_mutation)
    except (AssertionError, ValueError):
        continue
    raise AssertionError(f"held P1/C2 mutation {index} escaped")
old_runner = runner.replace('"$evidence/off"',
                            '"$evidence/off" "$evidence/corpus"', 1)
assert old_runner != runner
try:
    check(fixture, old_runner, bit_bridge)
except (AssertionError, ValueError):
    pass
else:
    raise AssertionError("premature corpus creation escaped")
old_probe = runner.replace(
    '"$PROJECT_ROOT/scripts/check-g3r-f32-allocator-symbol.sh"',
    'true # skipped allocator symbol probe', 1)
assert old_probe != runner
try:
    check(fixture, old_probe, bit_bridge)
except (AssertionError, ValueError):
    pass
else:
    raise AssertionError("missing allocator symbol probe escaped")
for header in ("compile src/eshkol_transformer/m3_call_f32_integration.c m3_call_f32_integration \\",
               "compile native/o2_optimizer.c o2_optimizer \\"):
    command = compile_command(runner, header)
    old_flags = runner.replace(
        command, command.replace(" -DET_G3R_CANDIDATE_RETIRE_PRIVATE", ""), 1)
    assert old_flags != runner
    try:
        check(fixture, old_flags, bit_bridge)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError(f"missing owned-span prerequisite escaped: {header}")
duplicate_owner = runner.replace(
    "compile src/eshkol_transformer/m3_call_f32_integration.c m3_call_f32_integration \\",
    "compile native/f32_tensor.c f32_tensor -DET_F32_TENSOR_TESTING\n"
    "compile src/eshkol_transformer/m3_call_f32_integration.c m3_call_f32_integration \\", 1)
assert duplicate_owner != runner
try:
    check(fixture, duplicate_owner, bit_bridge)
except (AssertionError, ValueError):
    pass
else:
    raise AssertionError("duplicate F32 owner recipe escaped")
print(f"held P1/C2 source aggregate: pass ({len(mutations) + 5} mutations; unadmitted)")
