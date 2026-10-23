"""Synthetic strict-format negatives, never actual model output."""
import unittest
from tests.cpu4_resume.test_driver import synthetic_stdout
from .check_output import check, MODES, HANDOFF, RESULT, EXPECTED_CHECKS


def records(mode):
    plain=synthetic_stdout().decode().splitlines()
    return plain[:6]+[f'CPU4-TRAINED-G2-HANDOFF {mode} {HANDOFF}',
        f'CPU4-TRAINED-G2-RESULT {mode} {RESULT[mode]}','CPU4-TRAINED-G2-RNG 1 41 0 0',plain[6],
        f'CPU4-TRAINED-G2-PASS {mode} {EXPECTED_CHECKS} 375 14 28 0 0 retained-model-and-T1 partial-bytes=13424 census=NOT-RUN']


def text(lines):return '\n'.join(lines)+'\n'


class Checker(unittest.TestCase):
    def test_both_frozen_modes(self):
        for mode in MODES:check(text(records(mode)),mode)

    def test_every_record_and_numeric_generation_field_rejects_mutation(self):
        for mode in MODES:
            for index,line in enumerate(records(mode)):
                bad=records(mode);bad[index]=line+' extra'
                with self.subTest(mode=mode,index=index),self.assertRaises(ValueError):check(text(bad),mode)
            for index in (6,7,10):
                for slot,part in enumerate(records(mode)[index].split(' ')):
                    if part.isdecimal():
                        bad=records(mode);parts=bad[index].split(' ');parts[slot]=str(int(part)+1);bad[index]=' '.join(parts)
                        with self.subTest(mode=mode,index=index,slot=slot),self.assertRaises(ValueError):check(text(bad),mode)

    def test_wrong_mode_learning_relaxation_reordering_and_retention_claim(self):
        for mode in MODES:
            base=records(mode)
            cases=[base+[base[0]],base[1:],list(reversed(base)),[line.replace('census=NOT-RUN','census=PASS') for line in base],
                   [line.replace('partial-bytes=13424','total-bytes=13424') for line in base],
                   [line.replace('CPU4-TRAINED-G2-RNG 1','CPU4-TRAINED-G2-RNG 0') for line in base]]
            for bad in cases:
                with self.assertRaises(ValueError):check(text(bad),mode)
            with self.assertRaises(ValueError):check(text(base),'unknown')
            with self.assertRaises(ValueError):check(text(base),MODES[1-MODES.index(mode)])
            with self.assertRaises(ValueError):check(text(base).rstrip('\n'),mode)
            bad=base[:];bad[2]=bad[2].replace('1056964608','1084227584');bad[5]=bad[5].replace('1056964608','1084227584')
            with self.assertRaises(ValueError):check(text(bad),mode)


if __name__=='__main__':unittest.main()
