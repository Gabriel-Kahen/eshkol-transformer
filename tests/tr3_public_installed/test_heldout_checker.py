"""Negative controls for the held-out acceptance checker."""

import unittest

from tests.tr3_public_installed.check_heldout import check


GOOD = ("TR3-HELDOUT-OBS 0 1085366897 1077936128 3 2\n"
        "TR3-HELDOUT-TRAIN 32 64 31\n"
        "TR3-HELDOUT-OBS 1 1084227584 1077936128 3 2\n"
        "TR3-HELDOUT-OBS 0 1085366897 1077936128 3 2\n"
        "TR3-HELDOUT-AOT-PASS\n")


class HeldoutCheckerTests(unittest.TestCase):
    def test_valid_synthetic_checker_control(self) -> None:
        self.assertEqual(check(GOOD), (1085366897, 1084227584))

    def test_rejects_false_improvement(self) -> None:
        for after in (1085366897, 1085000000, 1086000000, 0x7F800000):
            with self.subTest(after=after), self.assertRaises(ValueError):
                check(GOOD.replace("OBS 1 1084227584", f"OBS 1 {after}"))

    def test_rejects_wrong_counts_and_retained_metrics(self) -> None:
        for corrupt in (
            GOOD.replace("TRAIN 32 64 31", "TRAIN 31 62 30"),
            GOOD.replace("1077936128 3 2", "1077936128 2 2", 1),
            GOOD.replace("1077936128 3 2", "1073741824 3 2", 1),
            GOOD.replace("OBS 0 1085366897", "OBS 0 1084227584", 1),
            GOOD + "TR3-HELDOUT-OBS 2 1 1 1 1\n",
        ):
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                check(corrupt)


if __name__ == "__main__":
    unittest.main()
