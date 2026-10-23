"""Strict composite development checker; synthetic tests are not learning proof."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.cpu4_resume.check_resume import check as check_resume, integers

MODES = ('two-token', 'first-eos')
EXPECTED_CHECKS = 19
HANDOFF = '14 4768 1 28 28'
RESULT = {'two-token': '255 2 4 3 41 3 41', 'first-eos': '3 1 3 3 3'}


def check(text, mode):
    if mode not in MODES:
        raise ValueError('closed generation mode required')
    lines = text.splitlines()
    if len(lines) != 11 or text != '\n'.join(lines) + '\n':
        raise ValueError('expected exactly eleven LF-terminated composite records')
    result = check_resume('\n'.join(lines[:6] + [lines[9]]) + '\n')
    if lines[6] != f'CPU4-TRAINED-G2-HANDOFF {mode} {HANDOFF}':
        raise ValueError('wrong same-model handoff/retirement evidence')
    if lines[7] != f'CPU4-TRAINED-G2-RESULT {mode} {RESULT[mode]}':
        raise ValueError('frozen learned generation expectation differs')
    parts = lines[8].split(' ')
    if parts[0] != 'CPU4-TRAINED-G2-RNG' or len(parts) != 5:
        raise ValueError('missing genuine detached RNG evidence')
    words = integers(parts[1:])
    if words[0] != 1 or any(word > 9223372036854775807 for word in words):
        raise ValueError('wrong RNG version or non-int64 words')
    if lines[10] != (f'CPU4-TRAINED-G2-PASS {mode} {EXPECTED_CHECKS} 375 14 28 0 0 '
                     'retained-model-and-T1 partial-bytes=13424 census=NOT-RUN'):
        raise ValueError('wrong assertion/cleanup/explicit-retention boundary')
    return result


if __name__ == '__main__':
    check(Path(sys.argv[1]).read_text(), sys.argv[2])
    print('CPU4-TRAINED-G2-CHECK-PASS mode=' + sys.argv[2] + ' scope=tiny-synthetic-trained-FILE-resume')
