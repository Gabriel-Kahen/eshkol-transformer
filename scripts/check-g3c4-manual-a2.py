#!/usr/bin/env python3
"""Structural gate for private manual ordinal-10 A2 seam."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def check():
    subprocess.run([sys.executable, str(ROOT / 'scripts/check-g3c4-manual-pre-a2.py')],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / 'src/eshkol_transformer/g3c4_model_owner.c').read_text()
    test = (ROOT / 'tests/g3c4/test_manual_a2.c').read_text()
    predecessor = (ROOT / 'tests/g3c4/test_manual_pre_a2.c').read_text()
    included = (ROOT / 'tests/g3c4/test_manual_a2_predecessor.inc').read_text()
    expected_included = predecessor.replace(
        'int32_t __wrap_et_kernel_runtime_dispatch(\n    const et_kernel_runtime *runtime',
        'int32_t et_g3c4_manual_a2_base_dispatch(\n    const et_kernel_runtime *runtime'
    ).replace('int main(void) {\n  et_g3c4_model_owner_internal *owner',
              'int et_g3c4_manual_pre_a2_predecessor_main(void) {\n  et_g3c4_model_owner_internal *owner')
    assert included == expected_included, 'inherited pre-A2 witness drifted'
    runner = (ROOT / 'scripts/test-g3c4-manual-a2.sh').read_text()
    contract = (ROOT / 'docs/g3/G3_C4_MANUAL_A2_ORDINAL10_CONTRACT.md').read_text()
    role = source[source.index('static inline __attribute__((unused)) int64_t et_g3c4_manual_a2_run'):]
    assert '(frame->a2_candidate == NULL || frame->a2_transaction == NULL)' in source
    for fragment in ('et_a2_kv_cache_create_v1', 'et_a2_kv_cache_read_borrow_layer_v1',
                     'et_a2_kv_cache_transaction_stage_layer_v1',
                     'et_a2_kv_cache_transaction_view_begin_v1',
                     'kernel.causal-attention', 'causal-attention.forward',
                     'et_a2_kv_cache_transaction_view_end_v1',
                     'frame->a2_candidate = candidate', 'frame->next_ordinal = 11'):
        assert fragment in role, fragment
    assert role.index('et_a2_kv_cache_transaction_view_end_v1') < role.index('frame->next_ordinal = 11')
    assert 'et_a2_kv_cache_transaction_abort_v1' in source[source.index('int64_t et_g3c4_private_call_abort_v1'):]
    assert 'et_g3c4_private_role_step_v1' not in source
    for fragment in ('a2_case(owner, 1, 1)', 'a2_case(owner, 1, 2)',
                     'a2_case(owner, 2, 1)', 'fail_a2', 'logits_borrow',
                     'frame->a2_transaction = NULL', 'frame->a2_candidate = NULL',
                     'et_g3c4_private_call_abort_v1(context) == ET_G3C4_INTERNAL'):
        assert fragment in test, fragment
    for fragment in ('compile_mode normal', 'compile_mode sanitize',
                     'test-g3c4-manual-pre-a2.sh', 'added-defined.txt'):
        assert fragment in runner, fragment
    assert 'No Eshkol entry' in contract and 'ordinal 10' in contract
    previous = (ROOT / 'native/g3c4_manual_pre_a2_source_closure.txt').read_text().splitlines()
    additions = ['native/g3c4_manual_pre_a2_source_closure.txt',
                 'include/eshkol_transformer/a2_attention_abi.h',
                 'native/a2_attention_provider.c',
                 'tests/g3c4/test_manual_a2_predecessor.inc',
                 'tests/g3c4/test_manual_a2.c', 'scripts/check-g3c4-manual-a2.py',
                 'scripts/test-g3c4-manual-a2.sh',
                 'docs/g3/G3_C4_MANUAL_A2_ORDINAL10_CONTRACT.md']
    expected = previous + [x for x in additions if x not in previous]
    actual = (ROOT / 'native/g3c4_manual_a2_source_closure.txt').read_text().splitlines()
    assert actual == expected and len(actual) == len(set(actual))
    for path in actual: assert (ROOT / path).is_file(), path

if __name__ == '__main__':
    try: check()
    except (OSError, AssertionError, subprocess.CalledProcessError) as error:
        print(f'G3-C4 manual A2 static check failed: {error}', file=sys.stderr)
        raise SystemExit(1)
    print('G3-C4 internal manual A2 contract: PASS')
