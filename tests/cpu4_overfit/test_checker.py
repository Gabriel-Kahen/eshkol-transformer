"""Independent synthetic negative controls for the predeclared acceptance gate."""
import struct
import unittest
from check_overfit import check, SAMPLES


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def log(initial=5.0, final=0.5):
    observations = [f"CPU4-OVERFIT-OBS {n} {n-1} {bits(4*(initial if n == 1 else final))} "
                    f"1082130432 {bits(initial if n == 1 else final)} 4 {n} {n-1}"
                    for n in SAMPLES]
    return observations + ["CPU4-OVERFIT-TRAIN 250 1000 249", "CPU4-OVERFIT-PARAM 10 1192 4768",
        "CPU4-OVERFIT-RETENTION 250 56 1000 1000000 0 250 0 0 26000", "CPU4-OVERFIT-RUNTIME-PASS 2010"]


class Checker(unittest.TestCase):
    def test_predeclared_valid_and_exact_zero_loss(self):
        self.assertEqual(check(log()), (5.0, 0.5))
        self.assertEqual(check(log(final=0.0)), (5.0, 0.0))

    def test_rejects_noop_and_independent_relative_absolute_failures(self):
        for initial, final in ((5., 5.), (3., 0.9), (5., 1.1)):
            with self.subTest(initial=initial, final=final), self.assertRaises(AssertionError):
                check(log(initial, final))

    def test_rejects_nonfinite_negative_and_bad_initial(self):
        for word in (0x7F800000, 0x7FC00000, 0xBF000000):
            value = log()
            parts = value[0].split()
            parts[5] = str(word)
            value[0] = " ".join(parts)
            with self.subTest(word=word), self.assertRaises(AssertionError):
                check(value)
        with self.assertRaises(AssertionError):
            check(log(initial=0.0))

    def test_rejects_false_counts_loss_and_parameter_noop(self):
        for index, replacement in (
            (0, "CPU4-OVERFIT-OBS 1 1 1101004800 1082130432 1084227584 4 1 0"),
            (0, "CPU4-OVERFIT-OBS 1 0 1101004800 1077936128 1084227584 4 1 0"),
            (0, "CPU4-OVERFIT-OBS 1 0 1101004800 1082130432 1073741824 4 1 0"),
            (9, "CPU4-OVERFIT-TRAIN 249 996 248"),
            (10, "CPU4-OVERFIT-PARAM 0 1192 4768"),
            (10, "CPU4-OVERFIT-PARAM 1 1192 1192"),
            (12, "CPU4-OVERFIT-RUNTIME-PASS 2009"),
        ):
            value = log()
            value[index] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(AssertionError):
                check(value)

    def test_rejects_live_payload_plans_and_retention_overflow(self):
        for replacement in (
            "CPU4-OVERFIT-RETENTION 250 57 0 10 0 250 0 0 26000",
            "CPU4-OVERFIT-RETENTION 250 56 0 10 1 250 0 0 26000",
            "CPU4-OVERFIT-RETENTION 250 56 0 10 0 249 1 0 26000",
            "CPU4-OVERFIT-RETENTION 250 56 0 10 0 250 0 1 26000",
            "CPU4-OVERFIT-RETENTION 250 56 0 67108865 0 250 0 0 26000",
            "CPU4-OVERFIT-RETENTION 250 56 0 10 0 250 0 0 262145",
        ):
            value = log()
            value[11] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(AssertionError):
                check(value)

    def test_rejects_missing_extra_reordered_or_malformed_output(self):
        for value in (log()[:-1], log()+["extra"], [log()[1], log()[0]]+log()[2:], ["bad"]+log()[1:]):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                check(value)


if __name__ == "__main__":
    unittest.main()
