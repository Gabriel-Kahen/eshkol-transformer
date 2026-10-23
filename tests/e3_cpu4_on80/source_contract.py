"""Exact consumer projection and existing candidate seed/lifetime prerequisites."""
from tests.e3_cpu4_consumer.test_source import forms

OLD_LOAD = '(load "e3_cpu4_private_root.esk")'
NEW_LOAD = '(load "cpu4_v1_private_root.esk")'
OLD_CREATE = '(define model (g3c4-model-create-seeded-internal 1729))'
NEW_CREATE = ('(define candidate (cpu4-candidate-model-create-internal! resolved))\n'
              '(define model (cpu4-candidate-model-internal candidate))')
SUCCESS = '(display "CPU4-CONSUMER-PASS checks=")'
RETIRE = (';;; Both frames unbound, dataset closed, binding/lease released, images and\n'
          ';;; destinations destroyed above. Existing strict candidate retirement proves\n'
          ';;; no live context/lease/children or optimizer survived; no new hook.\n'
          '(cpu4-candidate-retire-internal! candidate)\n')


def projected(original):
    for anchor in (OLD_LOAD, OLD_CREATE, SUCCESS):
        assert original.count(anchor) == 1, 'original projection anchor differs'
    return original.replace(OLD_LOAD, NEW_LOAD).replace(OLD_CREATE, NEW_CREATE).replace(SUCCESS, RETIRE + SUCCESS)


def validate_text(original, current):
    assert current == projected(original), 'consumer differs outside exact root/producer/retire projection'
    original_forms = forms(original)
    actual_forms = forms(current)
    expected_forms = forms(projected(original))
    assert actual_forms == expected_forms and len(actual_forms) == len(original_forms) + 2
    assert current.count(NEW_LOAD) == 1 and OLD_LOAD not in current
    order = ('(e3-frame-unbind! frame)', '(e3-frame-unbind! handoff)',
             '(d2-token-dataset-close! dataset)',
             "(check 'release (cpu4-model-binding-release-internal! binding))",
             '(image-destroy image)', '(cpu4-candidate-retire-internal! candidate)', SUCCESS)
    positions = [current.index(item) for item in order]
    assert positions == sorted(positions), 'retire precedes original drainage'
    assert original.count('(check ') == current.count('(check '), 'assertions changed'


def validate_variant(repo):
    original = (repo / 'tests/e3_cpu4_consumer/runtime.esk').read_text()
    current = (repo / 'tests/e3_cpu4_on80/runtime.esk').read_text()
    validate_text(original, current)
    candidate = (repo / 'native/cpu4_candidate_extension.esk').read_text()
    model = (repo / 'native/g3c4_model_extension.esk').read_text()
    native = (repo / 'src/eshkol_transformer/cpu4_candidate.inc').read_text()
    # Existing candidate resolved-seed and the original producer enter the same
    # real native seeded constructor/14 initializers; no seed reconstruction.
    assert '(x1-private-config-ref resolved \'run.seed)' in candidate
    assert '(seed (resolved-seed resolved))' in candidate
    assert '(capability (native-begin seed))' in candidate
    assert '(owner-initialize owner index)' in candidate
    assert '(owner (g3c4-native-owner-create-seeded seed))' in model
    assert '(g3c4-native-owner-initialize owner index)' in model
    assert 'et_g3c4_private_model_owner_create_seeded_v1(seed)' in native
    assert '(vector-set! row 1 \'unpublished)' in candidate
    assert '(define (model candidate)\n      (let ((row (unpublished candidate' in candidate
    assert '(define (retire candidate) (retire-owned-optimizer candidate \'()))' in candidate
    assert '(native-prepare (vector-ref row 2) optimizer \'())' in candidate
    assert '!owner->successor_ready || owner->training_lease != NULL ||' in native
    assert 'if (owner->live_contexts == 0u) return 0;' in native
    assert 'static int64_t cpu4_candidate_retire_preflight(' in native
    assert 'status = cpu4_candidate_generator_preflight(candidate, generator);' in native
    assert '(module-train! model)' in current
    assert '\\"run.seed\\":1729' in current
    assert '(cpu4-model-binding-create-internal model resolved)' in current
    return True
