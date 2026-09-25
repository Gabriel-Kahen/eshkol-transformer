#!/usr/bin/env python3
"""Structural gate for private manual numerical ordinals 14..20."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check() -> None:
    subprocess.run([sys.executable, str(ROOT / 'scripts/check-g3c4-manual-r.py')],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / 'src/eshkol_transformer/g3c4_model_owner.c').read_text()
    test = (ROOT / 'tests/g3c4/test_manual_tail.c').read_text()
    prior = (ROOT / 'tests/g3c4/test_manual_r.c').read_text()
    inherited = (ROOT / 'tests/g3c4/test_manual_tail_predecessor.inc').read_text()
    expected_inherited = prior.replace(
        'int32_t __wrap_et_kernel_runtime_dispatch(\n    const et_kernel_runtime *runtime',
        'int32_t et_g3c4_manual_tail_base_dispatch(\n    const et_kernel_runtime *runtime'
    ).replace('int main(void) {\n  et_g3c4_model_owner_internal *owner',
              'int et_g3c4_manual_r_predecessor_main(void) {\n  et_g3c4_model_owner_internal *owner')
    assert inherited == expected_inherited, 'inherited R witness drifted'
    signature = 'static inline __attribute__((unused)) int64_t et_g3c4_manual_tail_run'
    role = source[source.index(signature):source.index('\n#endif\n#endif', source.index(signature))]
    sequence = ['ordinal < 14 || ordinal > 20', 'et_g3c4_admit_active_call',
                'frame->next_ordinal != ordinal', 'et_g3c4_pending_logits_lookup',
                'et_g3c4_cache_idle_preflight', 'switch (ordinal)',
                'et_g3c4_token_runtime_discover', 'et_g3c4_token_dispatch',
                'memcpy(destination, result', 'frame->next_ordinal = ordinal + 1']
    at = -1
    for part in sequence:
        at = role.find(part, at + 1)
        assert at >= 0, f'missing or misordered: {part}'
    for part in ('case 14: case 19:', 'case 15: case 17: case 20:',
                 'case 16:', 'case 18:',
                 'context->pins.views[ordinal == 14 ? 9u : 12u]',
                 'context->pins.views[ordinal == 14 ? 8u : 11u]',
                 '(ordinal == 17 ? 4u : 10u)',
                 'UINT32_C(0x3727c5ac)', 'kernel.norm', 'layer-norm.forward',
                 'g3n.layer-norm.forward', 'g3n.linear.forward-no-bias',
                 'n3k.linear.forward-no-bias', 'kernel.activation', 'gelu.forward',
                 'n3k.gelu.forward', 'g3n.residual.forward',
                 'n3k.residual.forward', 'float epsilon, result[512] = {0}'):
        assert part in role, part
    assert 'et_g3c4_private_role_step_v1' not in source
    assert '(frame->next_ordinal < 0 || frame->next_ordinal > 21)' in source
    for part in ('tail_case(owner, 1, 1)', 'tail_case(owner, 1, 2)',
                 'tail_case(owner, 2, 1)', 'whole_reference(',
                 'whole_dispatches == 21u', 'fail_tail', 'snapshot_at_candidate',
                 'logits_borrow', 'memcmp(slot, whole_outputs[ordinal - 14]'):
        assert part in test, part
    runner = (ROOT / 'scripts/test-g3c4-manual-tail.sh').read_text()
    for part in ('compile_mode normal', 'compile_mode sanitize',
                 'test-g3c4-manual-r.sh', 'added-defined.txt',
                 'predecessor-hashes.stdout'):
        assert part in runner, part
    contract = (ROOT / 'docs/g3/G3_C4_MANUAL_TAIL_ORDINALS14_20_CONTRACT.md').read_text()
    assert 'No Eshkol entry' in contract and 'ordinals 14–20' in contract
    previous = (ROOT / 'native/g3c4_manual_r_source_closure.txt').read_text().splitlines()
    additions = ['native/g3c4_manual_r_source_closure.txt',
                 'tests/g3c4/test_manual_tail_predecessor.inc',
                 'tests/g3c4/test_manual_tail.c',
                 'scripts/check-g3c4-manual-tail.py',
                 'scripts/test-g3c4-manual-tail.sh',
                 'docs/g3/G3_C4_MANUAL_TAIL_ORDINALS14_20_CONTRACT.md']
    expected = previous + [path for path in additions if path not in previous]
    actual = (ROOT / 'native/g3c4_manual_tail_source_closure.txt').read_text().splitlines()
    assert actual == expected and len(actual) == len(set(actual))
    for path in actual:
        assert (ROOT / path).is_file(), path


if __name__ == '__main__':
    try:
        check()
    except (OSError, AssertionError, ValueError, subprocess.CalledProcessError) as error:
        print(f'G3-C4 manual tail static check failed: {error}', file=sys.stderr)
        raise SystemExit(1)
    print('G3-C4 internal manual tail contract: PASS')
