#!/usr/bin/env python3
"""Bounded source closure for the unadmitted held-P1/C2 runtime fixture."""
from pathlib import Path
import re
import subprocess
import sys

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


def phase_calls(source: str) -> list[int]:
    tokens = [(match.group(), match.start(), match.end()) for match in
              re.finditer(r'"(?:\\.|[^"\\])*"|;[^\n]*|[()]|[^()\s";]+', source)
              if not match.group().startswith(';')]
    phases = []
    frames = []
    fixture_definitions = 0

    def token_at(index: int) -> str:
        return tokens[index][0] if index < len(tokens) else ''

    for index, (token, start, _) in enumerate(tokens):
        if token == ')':
            assert frames
            frames.pop()
            continue
        if token != '(':
            continue
        if not frames and tuple(token_at(index + offset) for offset in range(1, 4)) == (
                'define', '(', 'fixture'):
            assert tuple(token_at(index + offset) for offset in range(1, 6)) == (
                'define', '(', 'fixture', 'directory', ')')
            fixture_definitions += 1
        if token_at(index + 1) != 'g3r-test-phase':
            frames.append(index)
            continue
        assert index + 3 < len(tokens)
        value, _, _ = tokens[index + 2]
        closing, _, end = tokens[index + 3]
        assert value.isascii() and value.isdecimal() and closing == ')'
        phase = int(value)
        if 100 <= phase <= 103:
            assert frames
            outer = frames[0]
            assert tuple(token_at(outer + offset) for offset in range(1, 6)) == (
                'define', '(', 'fixture', 'directory', ')')
            expected_frames = {100: 1, 101: 2, 102: 2, 103: 3}
            assert len(frames) == expected_frames[phase]
            assert all(token_at(frame + 1) == 'let*' for frame in frames[1:])
        else:
            assert not frames
        line_start = source.rfind('\n', 0, start) + 1
        line_end = source.find('\n', end)
        if line_end == -1:
            line_end = len(source)
        assert not source[line_start:start].strip()
        assert not source[end:line_end].strip()
        phases.append(phase)
        frames.append(index)
    assert not frames
    assert fixture_definitions == 1
    return phases


def check(source: str, script: str, bits: str) -> None:
    assert balanced(source)
    assert '(extern i64 g3r-test-phase i64 :real et_g3r_test_phase_v1)' in source
    source_phases = phase_calls(source)
    assert source_phases == [
        *range(30), *range(100, 104), *range(200, 207),
        300, 301, 400, 401, 500, 501, 502, 600, 999,
    ]
    loads = [line for line in source.splitlines() if line.startswith('(load "')]
    assert len(loads) == 29
    for index, load in enumerate(loads, 1):
        assert load + f'\n(g3r-test-phase {index})' in source
    assert source.index('(g3r-test-phase 999)') < source.index(
        '(display "G3-R held P1/C2 source aggregate PASS: checks=")')
    assert "int64_t et_g3r_test_phase_v1(int64_t phase)" in bits
    assert 'fprintf(stderr, "G3R_PHASE %lld\\n"' in bits
    assert 'fflush(stderr) != 0' in bits
    assert 're.fullmatch(r"G3R_PHASE ([0-9]+)", line)' in script
    assert '''fixture_phases = [100, 101, 102, 103]
expected_phases = [
    *range(30), *fixture_phases, *range(200, 207),
    *fixture_phases, 300, 301,
    *fixture_phases, 400, 401,
    *fixture_phases, 500, 501, 502,
    *fixture_phases, 600, 999,
]
if phases != expected_phases:''' in script
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


def swap_phase_calls(source: str, first: int, second: int) -> str:
    first_call = f'(g3r-test-phase {first})'
    second_call = f'(g3r-test-phase {second})'
    return (source.replace(first_call, '(g3r-test-phase swapped)', 1)
            .replace(second_call, first_call, 1)
            .replace('(g3r-test-phase swapped)', second_call, 1))


check(fixture, runner, bit_bridge)
check(fixture + '\n; (g3r-test-phase 203)\n'
      '(define phase-text "(g3r-test-phase 203)")\n', runner, bit_bridge)
mutations = (
    (fixture.replace('(g3r-test-phase 0)', ''), bit_bridge),
    # Keep the phase numbers in order while moving receipts into dead helpers.
    (fixture.replace('(define (fixture directory)\n  (g3r-test-phase 100)',
                     '(define (unused-phase)\n  (g3r-test-phase 100))\n'
                     '(define (fixture directory)'), bit_bridge),
    (fixture.replace('    (g3r-test-phase 101)',
                     '    (define (unused-phase)\n'
                     '      (g3r-test-phase 101))'), bit_bridge),
    (fixture.replace('    (g3r-test-phase 102)',
                     '    (begin\n      (g3r-test-phase 102))'), bit_bridge),
    (fixture.replace('      (g3r-test-phase 103)',
                     '      (define (unused-phase)\n'
                     '        (g3r-test-phase 103))'), bit_bridge),
    (fixture.replace('      (g3r-test-phase 103)', '')
            .replace('(define (controls tuple)',
                     '(define (unused-phase)\n  (g3r-test-phase 103))\n'
                     '(define (controls tuple)'), bit_bridge),
    (fixture.replace('(define (controls tuple)',
                     '(define (fixture directory) #f)\n'
                     '(define (controls tuple)'), bit_bridge),
    (fixture.replace('(g3r-test-phase 203)', ''), bit_bridge),
    (fixture.replace('(g3r-test-phase 203)',
                     '(g3r-test-phase 203)\n(g3r-test-phase 203)'), bit_bridge),
    (fixture.replace('(define corpus-directory (car (cdr (command-line))))',
                     '(define corpus-directory (car (cdr (command-line)))) '
                     '(g3r-test-phase 203)'), bit_bridge),
    (fixture.replace('(g3r-test-phase 203)',
                     '(g3r-test-phase 203) (g3r-test-phase 203)'), bit_bridge),
    (fixture.replace('(define held (begin-held original))\n(g3r-test-phase 203)',
                     '(define held (begin-held original)) (g3r-test-phase 203)'), bit_bridge),
    (fixture.replace('(define held (begin-held original))\n(g3r-test-phase 203)',
                     '(define held (begin-held original))\n(begin\n'
                     '(g3r-test-phase 203))'), bit_bridge),
    (swap_phase_calls(fixture, 203, 204), bit_bridge),
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
    assert mutation != fixture or bridge_mutation != bit_bridge
    try:
        check(mutation, runner, bridge_mutation)
    except (AssertionError, ValueError):
        continue
    raise AssertionError(f"held P1/C2 mutation {index} escaped")
for index, (mutation, _) in enumerate(mutations[1:7], 1):
    assert balanced(mutation)
    try:
        phase_calls(mutation)
    except AssertionError:
        continue
    raise AssertionError(f"phase scope mutation {index} escaped scanner")

# Exercise the embedded runtime parser with a complete trace and three interior
# counterexamples. The fixture function emits 100-103 on each of five calls.
phase_parser = runner.split("<<'PY_PHASES'\n", 1)[1].split("\nPY_PHASES", 1)[0]
read_phases = 'Path(sys.argv[1]).read_text()'
assert read_phases in phase_parser
phase_parser = phase_parser.replace(read_phases, 'sys.stdin.read()', 1)
runtime_phases = [
    *range(30), *range(100, 104), *range(200, 207),
    *range(100, 104), 300, 301,
    *range(100, 104), 400, 401,
    *range(100, 104), 500, 501, 502,
    *range(100, 104), 600, 999,
]


def parser_accepts(phases: list[int]) -> bool:
    result = subprocess.run(
        [sys.executable, '-c', phase_parser],
        input=''.join(f'G3R_PHASE {phase}\n' for phase in phases),
        text=True, capture_output=True, check=False)
    return result.returncode == 0


assert parser_accepts(runtime_phases)
interior = runtime_phases.index(203)
assert not parser_accepts(runtime_phases[:interior] + runtime_phases[interior + 1:])
assert not parser_accepts(runtime_phases[:interior] + [203] + runtime_phases[interior:])
reordered = runtime_phases.copy()
reordered[interior], reordered[interior + 1] = reordered[interior + 1], reordered[interior]
assert not parser_accepts(reordered)

runner_mutations = (
    runner.replace('if phases != expected_phases:',
                   'if phases[:30] != list(range(30)) or phases[-1] != 999:', 1),
    runner.replace('*range(200, 207)', '200, 201, 202, 204, 205, 206', 1),
    runner.replace('*fixture_phases, 300, 301', '*fixture_phases, 300, 300, 301', 1),
    runner.replace('*fixture_phases, 400, 401', '*fixture_phases, 401, 400', 1),
)
for index, mutation in enumerate(runner_mutations):
    assert mutation != runner
    try:
        check(fixture, mutation, bit_bridge)
    except (AssertionError, ValueError):
        continue
    raise AssertionError(f"runtime phase predicate mutation {index} escaped")
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
print(f"held P1/C2 source aggregate: pass ({len(mutations) + len(runner_mutations) + 5} mutations; unadmitted)")
