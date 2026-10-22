"""Source inventory only; compiler depfiles and symbol discovery remain required."""
import hashlib
import json
from pathlib import Path
import re
from tests.cpu4_trainer_control.test_source_contract import parse

ROOT = Path(__file__).resolve().parents[2]
ROSTER = 'native/cpu4_v1_private_inputs.json'
ENTRY = 'native/cpu4_v1_private_root.esk'


def source_closure(repo, roster, overrides=None):
    overrides = overrides or {}
    seen, order = set(), []
    def visit(path):
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        relative = path.relative_to(repo).as_posix()
        text = overrides.get(relative, path.read_text())
        forms = parse(text)
        order.append((relative, forms))
        for form in forms:
            if form and form[0] in ('load', 'require'):
                assert len(form) == 2 and isinstance(form[1], str), 'nonliteral source edge'
                name = form[1] if form[0] == 'load' else form[1].replace('.', '/') + '.esk'
                # Eshkol resolves a load beside its source, then trusted -I roots.
                candidates = [path.parent / name] + [repo / base / name for base in roster['source_include_dirs']]
                target = next((candidate for candidate in candidates if candidate.is_file()), None)
                assert target is not None, f'unresolved source edge {relative}: {name}'
                visit(target)
    visit(repo / ENTRY)
    return order


def native_closure(repo, roster):
    """Follow actual .c inclusions; helper-only guards exclude boxed bridges."""
    owned = {}
    def visit(path, owner):
        relative = path.relative_to(repo).as_posix()
        assert relative not in owned, f'duplicate native identity {relative}: {owned.get(relative)}, {owner}'
        owned[relative] = owner
        text = path.read_text()
        for include in re.findall(r'^\s*#include\s+"([^"\n]+\.c)"', text, re.M):
            if relative == 'native/i2_wave2_package_bridge.c':
                assert include == 't1_wave1_package_bridge.c', 'unexpected I2 inclusion'
                assert '#ifndef ET_I2_NATIVE_HELPERS_ONLY\n#include "e1b_error_consumer_bridge.h"\n#include "t1_wave1_package_bridge.c"\n#endif' in text
                assert 'ET_I2_NATIVE_HELPERS_ONLY' in features(roster, owner)
                continue
            if relative == 'native/o2_wave2_package_bridge.c':
                assert include in {'i2_wave2_package_bridge.c', 'o2_wave2_public_bridge_extension.c'}, 'unexpected O2 inclusion'
                assert '#ifndef ET_O2_NATIVE_HELPERS_ONLY\n#include "e1b_error_consumer_bridge.h"\n#include "i2_wave2_package_bridge.c"\n#endif' in text
                assert '#ifndef ET_O2_NATIVE_HELPERS_ONLY\n#include "o2_wave2_public_bridge_extension.c"\n#endif' in text
                assert 'ET_O2_NATIVE_HELPERS_ONLY' in features(roster, owner)
                continue
            visit((path.parent / include).resolve(), owner)
    for unit in roster['translation_units']:
        visit((repo / unit['source']).resolve(), unit['source'])
    return owned


def features(roster, source):
    local = next(unit['defines'] for unit in roster['translation_units'] if unit['source'] == source)
    return set(roster['common_defines'] + local)


def manifest(repo, roster):
    paths = {path for path, _ in source_closure(repo, roster)} | set(native_closure(repo, roster))
    # Preserve the current IR transformation/proof pair, without running either.
    paths |= local_native_inputs(repo, roster)
    paths |= {ROSTER, 'native/cpu4_training_native_sources.txt', 'native/e3_cpu4_consumer_native_sources.txt',
              'scripts/build-e1b-consumer.sh', 'scripts/g3g-manual-native-inputs.sh',
              'tests/e3_private/asan_ir_sources.sha256', 'tests/shared/prepare_eshkol_asan_ir.py',
              'tests/e3_private/check_asan_ir.py'}
    return ''.join(hashlib.sha256((repo / path).read_bytes()).hexdigest() + '  ' + path + '\n'
                   for path in sorted(paths))


def load_roster():
    return json.loads((ROOT / ROSTER).read_text())


COMMON = {'ET_G3C4_NATIVE_OWNER_PRIVATE', 'ET_G3C4_NATIVE_PINS_PRIVATE',
          'ET_CPU4_TRAINING_PRIVATE', 'ET_CPU4_OBJECTIVE_PRIVATE',
          'ET_CPU4_CANDIDATE_PRIVATE', 'ET_G3R_CANDIDATE_RETIRE_PRIVATE'}
FORBIDDEN_SOURCE = {'native/i2_wave2_extension.esk', 'native/e3_cpu4_boundary.esk',
                    'native/e3_cpu4_private_root.esk', 'native/d2_wave2_root.esk',
                    'native/cpu4_checkpoint_root.esk', 'native/c2_wave2_root.esk',
                    'native/k2_wave2_root.esk'}


def validate(repo, roster, overrides=None):
    assert roster['candidate_mode'] == 'on', 'candidate mode must be ON'
    assert set(roster['common_defines']) == COMMON, 'shared feature mismatch'
    units = [unit['source'] for unit in roster['translation_units']]
    assert len(units) == len(set(units)), 'duplicate translation unit'
    for unit in roster['translation_units']:
        defs = features(roster, unit['source'])
        assert not any('ET_M3T_PACKAGE_BUILD' in item or item.endswith('=0') or
                       'TESTING' in item or 'REUSE_E3_M3_OWNER' in item for item in defs), 'forbidden native feature'
    required = {
        'native/k2_capabilities.c': {'ET_C2_CARRIER_FACTORIES'},
        'native/i2_wave2_package_bridge.c': {'ET_I2_NATIVE_HELPERS_ONLY', 'ET_G3C4_I2_CONSTRUCTION_PRIVATE', 'ET_I2_PRIVATE_OWNED_CLONE_MATCH', 'ET_TR3_C_I2_RESTORE_PRIVATE', 'ET_C2_I2_MODEL_COPY'},
        'native/o2_wave2_package_bridge.c': {'ET_O2_NATIVE_HELPERS_ONLY', 'ET_TR3_O2_STEP_CLEAR_BRIDGE', 'ET_C2_O2_RECONSTRUCT_BRIDGE'},
        'native/o2_optimizer.c': {'ET_TR3_O2_STEP_CLEAR_NATIVE', 'ET_TR3_C_O2_RESTORE_NATIVE', 'ET_I2_PRIVATE_OWNED_CLONE_MATCH'},
        'native/tr3_c_restore_bindings.c': {'ET_TR3_C_RESTORE_BINDINGS', 'ET_TR3_C_I2_RESTORE_PRIVATE', 'ET_TR3_C_O2_RESTORE_NATIVE', 'ET_I2_PRIVATE_OWNED_CLONE_MATCH'},
        'native/p1_identity.c': {'ET_P1_TRUSTED_BUILD=1'},
        'native/e3_d2_native.c': {'ET_TR3_C_D2_RESTORE'},
        'src/eshkol_transformer/m3_call_f32_integration.c': {'ET_F32_TENSOR_STORAGE_QUERY_PRIVATE', 'ET_I2_PRIVATE_OWNED_CLONE_MATCH'},
        'src/eshkol_transformer/m3_i64_integration.c': {'ET_I64_TENSOR_STORAGE_QUERY_PRIVATE', 'ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE'},
        'native/t1_i64_shell.c': {'ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE'},
        'native/a2_kv_cache.c': {'ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE', 'ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE'},
    }
    for source, needed in required.items():
        assert needed <= features(roster, source), f'missing feature for {source}'
    # Union actual production component lists, replacing enclosing TUs once.
    expected = set()
    for source in ('native/cpu4_training_native_sources.txt', 'native/e3_cpu4_consumer_native_sources.txt'):
        expected.update((repo / source).read_text().splitlines())
    expected -= {'native/d2_native.c', 'src/eshkol_transformer/m3t_f32_integration.c', 'src/eshkol_transformer/m3_model.c'}
    expected |= {'src/eshkol_transformer/g3t_transport.c', 'native/a2_kv_cache.c',
                 'native/o2_optimizer.c', 'native/o2_wave2_package_bridge.c',
                 'native/tr3_c_restore_bindings.c', 'native/c2_x1_canonical.c'}
    expected |= {'native/' + name + '_provider.c' for name in ('g3n_primitives', 'g3s_sampling', 'g3c4_primitives')}
    expected |= {'native/' + name + '.c' for name in (
        'k2_capabilities', 'c2_checkpoint_codec', 'c2_checkpoint_format',
        'c2_checkpoint_save_bridge', 'c2_checkpoint_reader', 'c2_checkpoint_core',
        'c2_checkpoint_load_bridge', 'c2_checkpoint_inspect_bridge')}
    assert set(units) == expected, 'native union missing/extra production TU'
    # Read all generation feature operands from the existing reviewed builder.
    builder = (repo / 'scripts/build-e1b-consumer.sh').read_text()
    for source, first, last in (
        ('src/eshkol_transformer/g3t_transport.c',
         'package_source_flags=(\n        -DET_G3T_PREFILL_SAMPLE_PRIVATE',
         '    elif [[ "${package_policy}" == g3g-manual-public-aggregate &&\n            "${package_native_source}" == "${PROJECT_ROOT}/src/eshkol_transformer/m3_call_f32_integration.c"'),
        ('src/eshkol_transformer/g3c4_model_owner.c',
         'package_source_flags=(\n        -DET_G3C4_CONTEXT_PRIVATE',
         '    fi\n    "${e1b_clean_toolchain_env[@]}"')):
        section = builder[builder.index(first):builder.index(last, builder.index(first))]
        assert set(re.findall(r'-D([A-Z0-9_]+(?:=\d+)?)', section)) <= features(roster, source), 'generation feature omitted'
    closure = source_closure(repo, roster, overrides)
    paths = {path for path, _ in closure}
    assert not paths & FORBIDDEN_SOURCE, 'copied authority/root entered source union'
    names = {}
    for path, forms in closure:
        for form in forms:
            if len(form) > 2 and form[0] in ('define', 'define-syntax'):
                name = form[1][0] if isinstance(form[1], list) else form[1]
                assert name not in names, f'duplicate definition {name}: {names.get(name)}, {path}'
                names[name] = path
    for prefix, source in (('t1', 'internal/t1/lib/transformer/tokenizer_internal.esk'),
                           ('t2', 'internal/t2/lib/transformer/tokenizer_internal.esk')):
        for suffix in ('entry', 'tokenizer-byte', 'tokenizer-parse', 'tokenizer-payload-size',
                       'tokenizer-serialize', 'tokenizer-encode-staging', 'tokenizer-decode-staging',
                       'tokenizer-vocab-size', 'tokenizer-fingerprint', 'tokenizer-special-token-id'):
            assert names.get(prefix + '-private-' + suffix) == source, 'original tokenizer authority absent/replaced'
    assert names['t1-wave1-tensor-admitted?'] == 'native/t1_wave1_root.esk', 'T1 predicate replaced'
    assert names['i2-module-registry-root'] == 'native/i2_wave2_root.esk', 'I2 root replaced'
    assert names['m3-call-state'] == 'native/m3_model_extension.esk', 'M3 call state replaced'
    for needed in ('cpu4-training-step-internal!', 'cpu4-trainer-restore-internal!',
                   'cpu4-trainer-detach-internal!', 'cpu4-owned-candidate-create-internal',
                   'e3-evaluate-into!', 'g3c4-public-generator-create'):
        assert needed in names, f'required operation missing: {needed}'
    # Actual factory source dependencies must resolve to these original leaves.
    for name, source in FACTORY_PROVIDERS.items():
        assert names.get(name) == source, f'factory prerequisite absent/replaced: {name}'
    for name, source in HANDOFF_PROVIDERS.items():
        assert names.get(name) == source, f'handoff prerequisite absent/replaced: {name}'
    tables = {path: {f[1][0]: f for f in forms
                    if len(f) > 2 and f[0] == 'define' and isinstance(f[1], list)}
              for path, forms in closure}
    checkpoint = tables['native/cpu4_checkpoint_extension.esk']
    factory = tables['native/cpu4_checkpoint_factory_extension.esk']
    handoff = tables['native/cpu4_generator_handoff_extension.esk']
    for table, entry in ((checkpoint, 'cpu4-trainer-snapshot-internal'),
                         (checkpoint, 'cpu4-trainer-restore-internal!'),
                         (factory, 'cpu4-checkpoint-factory-create-internal!'),
                         (factory, 'cpu4-checkpoint-factory-retire-internal!'),
                         (handoff, 'cpu4-owned-generator-handoff-internal!'),
                         (handoff, 'cpu4-checkpoint-factory-generator-handoff-internal!'),
                         (handoff, 'cpu4-generator-handoff-retry-internal!')):
        assert table[entry][2][0] == 'cpu4-checkpoint-require-nonregional!', 'nonregional guard must be first'
    checkpoint_forms = next(forms for path, forms in closure if path == 'native/cpu4_checkpoint_extension.esk')
    assert ['extern', 'ptr', 'cpu4-checkpoint-native-current-region', ':real', 'region_current'] in checkpoint_forms, 'actual runtime region query required'
    public_forms = next(forms for path, forms in closure if path == 'native/c2_public_extension.esk')
    assert public_forms[0] == ['vector-set!', 't2-wave2-policy-projector', '0', 'c2-policy-t2-entry-internal'], 'original T2 policy installation changed'
    # Initialization order follows the actual leaves' declared prerequisites.
    order = {path: i for i, (path, _) in enumerate(closure)}
    for before, after in (
        ('g3c4_model_extension', 'cpu4_candidate_extension'),
        ('cpu4_candidate_extension', 'cpu4_candidate_children_extension'),
        ('cpu4_model_binding_extension', 'cpu4_trainer_control_extension'),
        ('cpu4_trainer_control_extension', 'cpu4_training_step_extension'),
        ('cpu4_training_schedule_extension', 'cpu4_training_step_extension'),
        ('tr3_o2_step_clear_extension', 'cpu4_training_step_extension'),
        ('tr3_c_restore_bindings_extension', 'cpu4_checkpoint_extension'),
        ('cpu4_trainer_detach_extension', 'cpu4_candidate_children_extension'),
        ('cpu4_candidate_children_extension', 'cpu4_checkpoint_factory_extension'),
        ('cpu4_checkpoint_extension', 'cpu4_checkpoint_factory_extension'),
        ('k2_wave2_extension', 'c2_public_extension'),
        ('c2_model_encode_extension', 'c2_checkpoint_save_extension'),
        ('c2_o2_encode_extension', 'c2_checkpoint_save_extension'),
        ('c2_persistence_policy_extension', 'c2_public_extension'),
        ('c2_checkpoint_inspect_extension', 'c2_public_extension'),
        ('c2_checkpoint_save_extension', 'c2_public_extension'),
        ('c2_checkpoint_load_extension', 'c2_public_extension'),
        ('c2_public_extension', 'cpu4_checkpoint_factory_extension'),
        ('cpu4_checkpoint_factory_extension', 'cpu4_generator_handoff_extension'),
        ('g3c4_public_generation_dispatch_extension', 'cpu4_generator_handoff_extension'),
        ('e3_p1_modes_extension', 'e3_cpu4_private_extension'),
        ('e3_errors_extension', 'e3_cpu4_private_extension')):
        left, right = 'native/' + before + '.esk', 'native/' + after + '.esk'
        assert left in order and right in order, 'leaf prerequisite absent'
        assert order[left] < order[right], 'leaf prerequisite order'
    # Root defines only the already-existing checkpoint fail-stop adapter.
    forms = next(forms for path, forms in closure if path == ENTRY)
    assert forms[:2] == [
        ['load', '../internal/t1/lib/transformer/tokenizer_internal.esk'],
        ['load', '../internal/t2/lib/transformer/tokenizer_internal.esk']], 'both exact tokenizer leaves must precede roots'
    assert [f for f in forms if f[0] != 'load'] == [
        ['define', ['tr3-lease-fail-stop', 'status'], ['cpu4-binding-fail-stop', 'status']]], 'aggregate grew orchestration/alias'
    native = native_closure(repo, roster)
    for identity in ('native/f32_tensor.c', 'native/i64_tensor.c', 'native/d2_native.c',
                     'native/i2_wave2_package_bridge.c', 'native/o2_optimizer.c',
                     'src/eshkol_transformer/m3_model.c', 'src/eshkol_transformer/m3t_transport.c'):
        assert identity in native, f'native identity absent: {identity}'
    return closure, native


def local_native_inputs(repo, roster):
    """Conservative local header/inc inventory, not compiler-observed depfiles."""
    seen = set()
    def visit(path):
        relative = path.relative_to(repo).as_posix()
        if relative in seen:
            return
        seen.add(relative)
        text = path.read_text()
        for include in re.findall(r'^\s*#include\s+["<]([^">\n]+)[">]', text, re.M):
            if include.endswith('.c'):
                # The precise selected enclosing-TU graph is checked separately.
                continue
            candidates = [path.parent / include] + [repo / base / include for base in roster['native_include_dirs']]
            target = next((p.resolve() for p in candidates if p.is_file()), None)
            if target is not None:
                visit(target)
    for path in native_closure(repo, roster):
        visit(repo / path)
    return seen


FACTORY_PROVIDERS = {
    **{name: 'native/c2_checkpoint_load_extension.esk' for name in (
        'c2-checkpoint-load-stage-internal', 'c2-checkpoint-load-capability-descriptors-internal',
        'c2-checkpoint-load-reconstruct-internal', 'c2-checkpoint-load-staging-release-internal!',
        'c2-checkpoint-load-staging-dead-internal?')},
    'c2-public-capability-admit-all': 'native/c2_public_extension.esk',
    'k2-public-capability-discover': 'native/k2_wave2_extension.esk',
    'k2-public-capability-require': 'native/k2_wave2_extension.esk',
    'cpu4-checkpoint-require-nonregional!': 'native/cpu4_checkpoint_extension.esk',
    'cpu4-owned-candidate-create-internal': 'native/cpu4_candidate_children_extension.esk',
    'cpu4-checkpoint-factory-create-internal!': 'native/cpu4_checkpoint_factory_extension.esk',
    'cpu4-checkpoint-factory-retire-internal!': 'native/cpu4_checkpoint_factory_extension.esk',
}

HANDOFF_PROVIDERS = {
    **{name: 'native/cpu4_generator_handoff_extension.esk' for name in (
        'cpu4-owned-generator-handoff-internal!',
        'cpu4-checkpoint-factory-generator-handoff-internal!',
        'cpu4-generator-handoff-retry-internal!')},
    'cpu4-candidate-adopt-public-g2-entry-internal!': 'native/cpu4_candidate_extension.esk',
    'g3c4-public-g2-profile-live': 'native/g3c4_public_generation_dispatch_extension.esk',
    'g3c4-public-generator-create': 'native/g3c4_public_generation_dispatch_extension.esk',
    'g3c4-generator-entry-call-ready': 'native/g3c4_call_entry_extension.esk',
}
