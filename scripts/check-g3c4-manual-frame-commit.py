#!/usr/bin/env python3
"""Structural gate for the manual frame publication seam."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check():
    subprocess.run([sys.executable, str(ROOT / 'scripts/check-g3c4-manual-tail.py')],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    source = (ROOT / 'src/eshkol_transformer/g3c4_model_owner.c').read_text()
    header = (ROOT / 'src/eshkol_transformer/g3c4_context_internal.h').read_text()
    test = (ROOT / 'tests/g3c4/test_manual_frame_commit.c').read_text()
    runner = (ROOT / 'scripts/test-g3c4-manual-frame-commit.sh').read_text()
    for symbol in ('et_g3c4_private_frame_prepare_v1',
                   'et_g3c4_private_frame_commit_v1'):
        assert source.count(symbol) == 1 and header.count(symbol) == 1, symbol
    prepare = source[source.index('int64_t et_g3c4_private_frame_prepare_v1'):]
    commit = source[source.index('int64_t et_g3c4_private_frame_commit_v1'):]
    assert prepare.index('et_g3c4_manual_publication_preflight') < prepare.index(
        'et_f32_tensor_copy_bits_from_v1') < prepare.index(
        'frame->publication_state = 1u')
    assert commit.index('et_g3c4_manual_publication_preflight') < commit.index(
        'et_a2_kv_cache_transaction_commit_v1') < commit.index(
        'et_a2_kv_cache_destroy_v1') < commit.index('context->cache =') < commit.index(
        'pending->transport.state = ET_G3C4_LOGITS_PUBLISHED')
    assert 'frame->publication_state = 3u' in commit
    for phrase in ('publication_case(owner, 1, 1, 0)',
                   'publication_case(owner, 1, 2, 0)',
                   'publication_case(owner, 2, 1, 0)',
                   'publication_case(owner, 1, 1, 1)',
                   'test_fail_alloc_after_v1(0u)',
                   'transaction_view_begin_v1', 'check_cache_snapshot_equal',
                   'et_g3c4_private_call_finish_v1'):
        assert phrase in test, phrase
    for phrase in ('compile_mode normal', 'compile_mode sanitize',
                   'test-g3c4-manual-tail.sh', 'added-defined.txt',
                   'predecessor-hashes.stdout'):
        assert phrase in runner, phrase
    previous = (ROOT / 'native/g3c4_manual_tail_source_closure.txt').read_text().splitlines()
    additions = ['native/g3c4_manual_tail_source_closure.txt',
                 'tests/g3c4/test_manual_frame_commit.c',
                 'scripts/check-g3c4-manual-frame-commit.py',
                 'scripts/test-g3c4-manual-frame-commit.sh',
                 'docs/g3/G3_C4_MANUAL_FRAME_COMMIT_CONTRACT.md']
    actual = (ROOT / 'native/g3c4_manual_frame_commit_source_closure.txt').read_text().splitlines()
    assert actual == previous + [p for p in additions if p not in previous]
    assert len(actual) == len(set(actual))
    for path in actual:
        assert (ROOT / path).is_file(), path


if __name__ == '__main__':
    try:
        check()
    except (OSError, AssertionError, ValueError, subprocess.CalledProcessError) as error:
        print(f'G3-C4 manual frame commit static check failed: {error}', file=sys.stderr)
        raise SystemExit(1)
    print('G3-C4 private manual frame commit contract: PASS')
