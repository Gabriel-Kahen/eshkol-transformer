"""Synthetic parser tests only; these are not actual held-out observations."""
import struct
import unittest

from check_heldout import check


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def line(phase, mean, *, weight=4.0, ppl=2.0, accuracy=0.5, physical=4, batches=1,
         counters=None):
    if counters is None:
        counters = (0, 0, 0) if phase == "baseline" else (1000, 250, 249)
    return " ".join(map(str, ("CPU4-HELDOUT-OBS", phase, bits(mean), bits(weight),
        bits(ppl), bits(accuracy), physical, batches, *counters))) + "\n"


class HeldoutChecker(unittest.TestCase):
    def test_fixed_threshold_including_exact_boundary(self):
        for post in (0.0, 8.0, 9.0):
            self.assertEqual(check(line("baseline", 10.0) + line("post", post)),
                             (bits(10.0), bits(post)))

    def test_mean_failure_and_nonfinite_metrics(self):
        for phase, field, value in (("post", "mean", 9.00001), ("baseline", "mean", 0.0),
                ("post", "mean", -1.0), ("baseline", "mean", float("nan")),
                ("post", "mean", float("inf")), ("baseline", "ppl", float("inf")),
                ("post", "accuracy", float("nan")), ("post", "ppl", 0.5),
                ("post", "accuracy", -0.1), ("post", "accuracy", 1.1)):
            args = {"mean": 10.0 if phase == "baseline" else 8.0, field: value}
            rows = {"baseline": line("baseline", 10.0), "post": line("post", 8.0)}
            rows[phase] = line(phase, **args)
            with self.subTest(phase=phase, field=field, value=value), self.assertRaises(ValueError):
                check(rows["baseline"] + rows["post"])

    def test_count_and_counter_drift(self):
        for phase in ("baseline", "post"):
            for change in ({"weight": 3.0}, {"physical": 5}, {"batches": 2},
                           {"counters": (1, 0, 0)}, {"counters": (1000, 249, 249)},
                           {"counters": (1000, 250, 250)}):
                rows = {"baseline": line("baseline", 10.0), "post": line("post", 8.0)}
                rows[phase] = line(phase, 10.0 if phase == "baseline" else 8.0, **change)
                with self.subTest(phase=phase, change=change), self.assertRaises(ValueError):
                    check(rows["baseline"] + rows["post"])

    def test_record_custody_and_strict_syntax(self):
        before, after = line("baseline", 10.0), line("post", 8.0)
        good = before + after
        bad = ("", before, after + before, before + before, good + after, good + "\n",
               good.rstrip("\n"), good.replace(" ", "  ", 1), good.replace("\n", "\r\n"),
               good.replace(str(bits(10.0)), "4294967296", 1),
               good.replace(str(bits(10.0)), "-1", 1),
               good.replace(str(bits(10.0)), "+123", 1),
               good.replace(str(bits(10.0)), "0123", 1),
               good.replace(str(bits(10.0)), "1e3", 1))
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                check(text)


if __name__ == "__main__":
    unittest.main()
