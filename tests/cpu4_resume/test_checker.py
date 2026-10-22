"""Synthetic format/acceptance negatives, never runtime or learning evidence."""
import struct
import unittest
from check_resume import check, EXPECTED_CHECKS


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def records(before=5., after=.5):
    tail = f'{bits(after)} 1082130432 {bits(2.)} {bits(.5)} 4 1 1000 250 249'
    return [f'CPU4-HELDOUT-OBS baseline {bits(before)} 1082130432 {bits(150.)} {bits(.25)} 4 1 0 0 0',
            'CPU4-RESUME-STATE saved125 500 125 124 4768 9536 125 1065353216',
            'CPU4-HELDOUT-OBS post ' + tail,
            'CPU4-RESUME-STATE restored125 500 125 124 4768 9536 125 1065353216',
            'CPU4-RESUME-STATE equal250 1000 250 249 4768 9536 250 1065353216',
            'CPU4-RESUME-HELDOUT ' + tail,
            f'CPU4-RESUME-PASS {EXPECTED_CHECKS} 250 125 375 1192 2384 0']


def text(lines):
    return '\n'.join(lines) + '\n'


class Checker(unittest.TestCase):
    def test_valid_and_exact_boundary(self):
        self.assertEqual(check(text(records())), (bits(5.), bits(.5)))
        check(text(records(before=10., after=9.)))

    def test_frozen_ten_percent_and_metric_rejections(self):
        for before, after in ((5., 5.), (10., 9.01), (0., 0.), (5., float('nan'))):
            with self.subTest(before=before, after=after), self.assertRaises(ValueError):
                check(text(records(before, after)))

    def test_native_count_schedule_and_complete_byte_sizes(self):
        for index in (1, 3, 4):
            for slot in range(2, 9):
                bad = records()
                parts = bad[index].split(' ')
                parts[slot] = str(int(parts[slot]) + 1)
                bad[index] = ' '.join(parts)
                with self.subTest(index=index, slot=slot), self.assertRaises(ValueError):
                    check(text(bad))

    def test_each_resumed_metric_and_count_must_equal_actual_post(self):
        for slot in range(1, 10):
            bad = records()
            parts = bad[5].split(' ')
            parts[slot] = str(int(parts[slot]) + 1)
            bad[5] = ' '.join(parts)
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                check(text(bad))

    def test_no_restored_counter_substitutes_for_actual_continuation(self):
        for slot, value in ((1, EXPECTED_CHECKS-1), (2, 125), (3, 0), (4, 500), (5, 4768), (6, 2383), (7, 1)):
            bad = records()
            parts = bad[6].split(' ')
            parts[slot] = str(value)
            bad[6] = ' '.join(parts)
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                check(text(bad))

    def test_canonical_order_and_exact_region(self):
        for bad in (records()[1:], records()+[records()[0]], list(reversed(records())),
                    [line.replace(' 500 ', ' 0500 ') for line in records()]):
            with self.assertRaises(ValueError):
                check(text(bad))
        with self.assertRaises(ValueError):
            check(text(records()).rstrip('\n'))


if __name__ == '__main__':
    unittest.main()
