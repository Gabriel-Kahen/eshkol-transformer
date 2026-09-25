#!/usr/bin/env python3
"""Structural gate for the private manual ordinal-13 attention residual."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check() -> None:
    subprocess.run([sys.executable, str(ROOT / 'scripts/check-g3c4-manual-ao.py')],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / 'src/eshkol_transformer/g3c4_model_owner.c').read_text()
    test = (ROOT / 'tests/g3c4/test_manual_r.c').read_text()
    prior = (ROOT / 'tests/g3c4/test_manual_ao.c').read_text()
    inherited = (ROOT / 'tests/g3c4/test_manual_r_predecessor.inc').read_text()
    expected_inherited = prior.replace(
        'int32_t __wrap_et_kernel_runtime_dispatch(\n    const et_kernel_runtime *runtime',
        'int32_t et_g3c4_manual_r_base_dispatch(\n    const et_kernel_runtime *runtime'
    ).replace('int main(void) {\n  et_g3c4_model_owner_internal *owner',
              'int et_g3c4_manual_ao_predecessor_main(void) {\n  et_g3c4_model_owner_internal *owner')
    assert inherited == expected_inherited, 'inherited AO witness drifted'
    signature = 'static inline __attribute__((unused)) int64_t et_g3c4_manual_r_run'
    role = source[source.index(signature):source.index('\n#endif\n#endif', source.index(signature))]
    sequence = ['et_g3c4_admit_active_call', 'frame->next_ordinal != 13',
                'et_g3c4_pending_logits_lookup', 'et_g3c4_cache_idle_preflight',
                'inputs[0] = et_g3c4_view(frame->x',
                'inputs[1] = et_g3c4_view(frame->ao',
                'et_g3c4_token_runtime_discover', 'et_g3c4_token_dispatch',
                'memcpy(frame->r, r_candidate', 'frame->next_ordinal = 14']
    at = -1
    for part in sequence:
        at = role.find(part, at + 1)
        assert at >= 0, f'missing or misordered: {part}'
    for part in ('g3n.residual-forward', 'g3n.residual.forward',
                 'n3k.residual', 'n3k.residual.forward',
                 'uint64_t shape[3] = {1u, 0u, 4u}'):
        assert part in role, part
    assert 'et_g3c4_private_role_step_v1' not in source
    assert '(frame->next_ordinal < 0 || frame->next_ordinal > 14)' in source
    for part in ('r_case(owner, 1, 1)', 'r_case(owner, 1, 2)',
                 'r_case(owner, 2, 1)', 'fail_r', 'snapshot_at_candidate',
                 'logits_borrow', 'inputs[0].data == frame->x',
                 'inputs[1].data == frame->ao'):
        assert part in test, part
    runner = (ROOT / 'scripts/test-g3c4-manual-r.sh').read_text()
    for part in ('compile_mode normal', 'compile_mode sanitize',
                 'test-g3c4-manual-ao.sh', 'added-defined.txt',
                 'predecessor-hashes.stdout'):
        assert part in runner, part
    contract = (ROOT / 'docs/g3/G3_C4_MANUAL_R_ORDINAL13_CONTRACT.md').read_text()
    assert 'No Eshkol entry' in contract and 'ordinal 13' in contract
    previous = (ROOT / 'native/g3c4_manual_ao_source_closure.txt').read_text().splitlines()
    additions = ['native/g3c4_manual_ao_source_closure.txt',
                 'tests/g3c4/test_manual_r_predecessor.inc',
                 'tests/g3c4/test_manual_r.c',
                 'scripts/check-g3c4-manual-r.py',
                 'scripts/test-g3c4-manual-r.sh',
                 'docs/g3/G3_C4_MANUAL_R_ORDINAL13_CONTRACT.md']
    expected = previous + [path for path in additions if path not in previous]
    actual = (ROOT / 'native/g3c4_manual_r_source_closure.txt').read_text().splitlines()
    assert actual == expected and len(actual) == len(set(actual))
    for path in actual:
        assert (ROOT / path).is_file(), path


if __name__ == '__main__':
    try:
        check()
    except (OSError, AssertionError, ValueError, subprocess.CalledProcessError) as error:
        print(f'G3-C4 manual R static check failed: {error}', file=sys.stderr)
        raise SystemExit(1)
    print('G3-C4 internal manual R contract: PASS')
