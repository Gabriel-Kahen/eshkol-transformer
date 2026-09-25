#!/usr/bin/env python3
"""Structural gate for the private manual ordinal-12 attention-out projection."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check() -> None:
    subprocess.run([sys.executable, str(ROOT / 'scripts/check-g3c4-manual-at.py')],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / 'src/eshkol_transformer/g3c4_model_owner.c').read_text()
    test = (ROOT / 'tests/g3c4/test_manual_ao.c').read_text()
    prior = (ROOT / 'tests/g3c4/test_manual_at.c').read_text()
    inherited = (ROOT / 'tests/g3c4/test_manual_ao_predecessor.inc').read_text()
    expected_inherited = prior.replace(
        'int32_t __wrap_et_kernel_runtime_dispatch(\n    const et_kernel_runtime *runtime',
        'int32_t et_g3c4_manual_ao_base_dispatch(\n    const et_kernel_runtime *runtime'
    ).replace('int main(void) {\n  et_g3c4_model_owner_internal *owner',
              'int et_g3c4_manual_at_predecessor_main(void) {\n  et_g3c4_model_owner_internal *owner')
    assert inherited == expected_inherited, 'inherited AT witness drifted'
    signature = 'static inline __attribute__((unused)) int64_t et_g3c4_manual_ao_run'
    role = source[source.index(signature):source.index('\n#endif\n#endif', source.index(signature))]
    sequence = ['et_g3c4_admit_active_call', 'frame->next_ordinal != 12',
                'et_g3c4_pending_logits_lookup', 'et_g3c4_cache_idle_preflight',
                'inputs[0] = et_g3c4_view(frame->at', 'inputs[1] = context->pins.views[1]',
                'et_g3c4_token_runtime_discover', 'et_g3c4_token_dispatch',
                'memcpy(frame->ao, ao_candidate', 'frame->next_ordinal = 13']
    at = -1
    for part in sequence:
        at = role.find(part, at + 1)
        assert at >= 0, f'missing or misordered: {part}'
    for part in ('g3n.linear-forward', 'g3n.linear.forward-no-bias',
                 'n3k.linear', 'n3k.linear.forward-no-bias',
                 'uint64_t row[4] = {1u, 0u, 4u, 4u}'):
        assert part in role, part
    assert 'et_g3c4_private_role_step_v1' not in source.split('#ifdef ET_G3C4_MANUAL_ROLE_STEP_PRIVATE\n')[0]
    assert '(frame->next_ordinal < 0 || frame->next_ordinal > 13)' in source
    for part in ('ao_case(owner, 1, 1)', 'ao_case(owner, 1, 2)',
                 'ao_case(owner, 2, 1)', 'fail_ao', 'snapshot_at_candidate',
                 'logits_borrow', 'ao_context->pins.views[1].data'):
        assert part in test, part
    runner = (ROOT / 'scripts/test-g3c4-manual-ao.sh').read_text()
    for part in ('compile_mode normal', 'compile_mode sanitize',
                 'test-g3c4-manual-at.sh', 'added-defined.txt',
                 'predecessor-hashes.stdout'):
        assert part in runner, part
    contract = (ROOT / 'docs/g3/G3_C4_MANUAL_AO_ORDINAL12_CONTRACT.md').read_text()
    assert 'No Eshkol entry' in contract and 'ordinal 12' in contract
    previous = (ROOT / 'native/g3c4_manual_at_source_closure.txt').read_text().splitlines()
    additions = ['native/g3c4_manual_at_source_closure.txt',
                 'tests/g3c4/test_manual_ao_predecessor.inc',
                 'tests/g3c4/test_manual_ao.c',
                 'scripts/check-g3c4-manual-ao.py',
                 'scripts/test-g3c4-manual-ao.sh',
                 'docs/g3/G3_C4_MANUAL_AO_ORDINAL12_CONTRACT.md']
    expected = previous + [path for path in additions if path not in previous]
    actual = (ROOT / 'native/g3c4_manual_ao_source_closure.txt').read_text().splitlines()
    assert actual == expected and len(actual) == len(set(actual))
    for path in actual:
        assert (ROOT / path).is_file(), path


if __name__ == '__main__':
    try:
        check()
    except (OSError, AssertionError, ValueError, subprocess.CalledProcessError) as error:
        print(f'G3-C4 manual AO static check failed: {error}', file=sys.stderr)
        raise SystemExit(1)
    print('G3-C4 internal manual AO contract: PASS')
