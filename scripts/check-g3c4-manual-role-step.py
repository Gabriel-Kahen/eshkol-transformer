#!/usr/bin/env python3
"""Structural gate for the closed manual role-step dispatcher."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check():
    subprocess.run([sys.executable, str(ROOT / 'scripts/check-g3c4-manual-frame-commit.py')],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / 'src/eshkol_transformer/g3c4_model_owner.c').read_text()
    header = (ROOT / 'src/eshkol_transformer/g3c4_context_internal.h').read_text()
    test = (ROOT / 'tests/g3c4/test_manual_role_step.c').read_text()
    runner = (ROOT / 'scripts/test-g3c4-manual-role-step.sh').read_text()
    symbol = 'et_g3c4_private_role_step_v1'
    assert source.count(symbol) == 1 and header.count(symbol) == 1
    assert 'ET_G3C4_MANUAL_ROLE_STEP_PRIVATE requires manual frame publication' in source
    body = source[source.index('int64_t et_g3c4_private_role_step_v1'):]
    body = body[:body.index('\n#endif')]
    sequence = ['et_g3c4_admit_active_call', 'ordinal < 0 || ordinal > 20',
                'frame->publication_state != 0u', 'frame->next_ordinal != ordinal',
                'et_g3c4_manual_role0_run', 'et_g3c4_manual_pre_a2_run',
                'et_g3c4_manual_a2_run', 'et_g3c4_manual_at_run',
                'et_g3c4_manual_ao_run', 'et_g3c4_manual_r_run',
                'et_g3c4_manual_tail_run']
    at = -1
    for phrase in sequence:
        at = body.find(phrase, at + 1)
        assert at >= 0, phrase
    assert 'frame->next_ordinal =' not in body
    assert 'et_kernel_runtime_dispatch' not in body
    for phrase in ('role_step_case(owner, 1, 1)', 'role_step_case(owner, 1, 2)',
                   'role_step_case(owner, 2, 1)', 'provider_retries == 21u',
                   'et_a2_kv_cache_test_fail_alloc_after_v1(0u)',
                   'ET_KERNEL_CODE_PROVIDER_REJECTED', 'whole_outputs[6]',
                   'et_g3c4_private_frame_commit_v1'):
        assert phrase in test, phrase
    for phrase in ('compile_mode normal', 'compile_mode sanitize',
                   'test-g3c4-manual-frame-commit.sh', 'added-defined.txt',
                   'predecessor-hashes.stdout', 'result-seal.sha256'):
        assert phrase in runner, phrase
    old = (ROOT / 'native/g3c4_manual_frame_commit_source_closure.txt').read_text().splitlines()
    additions = ['native/g3c4_manual_frame_commit_source_closure.txt',
                 'tests/g3c4/test_manual_role_step.c',
                 'scripts/check-g3c4-manual-role-step.py',
                 'scripts/test-g3c4-manual-role-step.sh',
                 'docs/g3/G3_C4_MANUAL_ROLE_STEP_CONTRACT.md']
    actual = (ROOT / 'native/g3c4_manual_role_step_source_closure.txt').read_text().splitlines()
    assert actual == old + [p for p in additions if p not in old]
    assert len(actual) == len(set(actual))
    for path in actual:
        assert (ROOT / path).is_file(), path


if __name__ == '__main__':
    try:
        check()
    except (OSError, AssertionError, ValueError, subprocess.CalledProcessError) as error:
        print(f'G3-C4 manual role step static check failed: {error}', file=sys.stderr)
        raise SystemExit(1)
    print('G3-C4 private manual role step contract: PASS')
