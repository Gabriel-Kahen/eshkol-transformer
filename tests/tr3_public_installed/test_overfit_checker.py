"""Negative controls for the predeclared one-batch loss criterion."""

import unittest

from tests.tr3_public_installed.check_one_batch_overfit import check


GOOD = [
    "TR3-OVERFIT-OBS before 1084227584 1 1",  # 5.0f
    "TR3-OVERFIT-OBS repeat 1084227584 1 1",
    "TR3-OVERFIT-TRAIN 32 32",
    "TR3-OVERFIT-OBS after 1080033280 1 1",  # 3.5f
]


class OverfitCheckerTests(unittest.TestCase):
    def test_synthetic_valid_control(self) -> None:
        self.assertEqual(check(GOOD), (5.0, 3.5))

    def test_rejects_no_or_insufficient_improvement(self) -> None:
        for word in (1084227584, 1082130432, 0x7F800000):
            with self.subTest(word=word), self.assertRaises(AssertionError):
                check(GOOD[:-1] + [f"TR3-OVERFIT-OBS after {word} 1 1"])

    def test_rejects_false_counts_and_unstable_baseline(self) -> None:
        for lines in (
            GOOD[:2] + ["TR3-OVERFIT-TRAIN 31 31"] + GOOD[3:],
            GOOD[:3] + ["TR3-OVERFIT-OBS after 1080033280 2 1"],
            GOOD[:1] + ["TR3-OVERFIT-OBS repeat 1080033280 1 1"] + GOOD[2:],
            GOOD + ["TR3-OVERFIT-OBS extra 1 1 1"],
        ):
            with self.subTest(lines=lines), self.assertRaises(AssertionError):
                check(lines)


if __name__ == "__main__":
    unittest.main()
