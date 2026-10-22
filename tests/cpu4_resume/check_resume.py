"""Development checker for genuine emitted FILE-resume evidence; no model math."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/cpu4_heldout'))
from check_heldout import check as check_heldout

EXPECTED_CHECKS = 770
STATE = {
    'saved125': (500, 125, 124, 4768, 9536, 125, 1065353216),
    'restored125': (500, 125, 124, 4768, 9536, 125, 1065353216),
    'equal250': (1000, 250, 249, 4768, 9536, 250, 1065353216),
}


def integers(parts):
    if any(re.fullmatch(r'0|[1-9][0-9]*', part) is None for part in parts):
        raise ValueError('expected canonical unsigned decimal integers')
    return tuple(map(int, parts))


def check(text):
    lines = text.splitlines()
    if len(lines) != 7 or text != '\n'.join(lines) + '\n':
        raise ValueError('expected exactly seven LF-terminated caller records')
    expected = ('CPU4-HELDOUT-OBS baseline', 'CPU4-RESUME-STATE saved125',
                'CPU4-HELDOUT-OBS post', 'CPU4-RESUME-STATE restored125',
                'CPU4-RESUME-STATE equal250', 'CPU4-RESUME-HELDOUT', 'CPU4-RESUME-PASS')
    for line, prefix in zip(lines, expected):
        if not line.startswith(prefix + ' '):
            raise ValueError('missing, duplicate, extra or reordered caller record')
    baseline, post = check_heldout(lines[0] + '\n' + lines[2] + '\n')
    for index, phase in ((1, 'saved125'), (3, 'restored125'), (4, 'equal250')):
        if integers(lines[index].split(' ')[2:]) != STATE[phase]:
            raise ValueError('wrong actual native/trainer state, payload size or scheduler bits')
    # All four actual metrics plus physical/batch/trainer counts match exactly.
    restored_report = integers(lines[5].split(' ')[1:])
    if restored_report != integers(lines[2].split(' ')[2:]):
        raise ValueError('fresh restored E3 frame differs from uninterrupted post report')
    if integers(lines[6].split(' ')[1:]) != (EXPECTED_CHECKS, 250, 125, 375, 1192, 2384, 0):
        raise ValueError('wrong assertion/actual-step budget or incomplete owner cleanup')
    return baseline, post


def main():
    before, after = check(Path(sys.argv[1]).read_text())
    print(f'CPU4-RESUME-CHECK-PASS baseline-bits={before} post-bits={after} '
          'actual-steps=375 A=250 B=125 restored-U=125 final-U=250 '
          'values=1192 moment-words=2384 scope=tiny-synthetic-FILE-resume')


if __name__ == '__main__':
    main()
