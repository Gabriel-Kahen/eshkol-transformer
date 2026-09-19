"""Reject incomplete or malformed evidence before reporting retention slopes."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.m3.check_m3_retention import check, MODES


class RetentionReport(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "retention.tsv"
        self.header = "mode\thorizon\tarena_bytes\tpeak_rss_kib\n"
        self.rows = [f"{mode}\t{horizon}\t{horizon * 128}\t4096\n"
                     for mode in sorted(MODES) for horizon in (1024, 8192)]

    def validate(self, rows, header=None):
        self.path.write_text((self.header if header is None else header) + "".join(rows))
        with redirect_stdout(StringIO()) as output:
            check(self.path)
        return output.getvalue()

    def test_complete_report_separates_rss_and_cumulative_arena(self):
        result = self.validate(self.rows)
        self.assertEqual(result.count("bytes_per_additional_iteration=128.000000"), 5)
        self.assertIn("peak_rss_kib8192=4096", result)
        self.assertIn("no flat training-memory claim", result)

    def test_missing_duplicate_malformed_and_decreasing_measurements(self):
        replacements = ("unknown\t1024\t131072\t4096\n",
                        "failure\t1024\t131072\n",
                        "failure\t1024\t131072\t4096\textra\n",
                        "failure\t1024\t-1\t4096\n",
                        "failure\t1024\t0\t4096\n",
                        "failure\t1024\t1e6\t4096\n",
                        "failure\t512\t131072\t4096\n")
        cases = [self.rows[:-1], self.rows + self.rows[:1]]
        cases += [[replacement] + self.rows[1:] for replacement in replacements]
        cases += [[self.rows[0], "failure\t8192\t1\t4096\n"] + self.rows[2:]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.validate(rows)
        with self.assertRaises(ValueError):
            self.validate(self.rows, "mode\thorizon\tpeak_rss_kib\tarena_bytes\n")


if __name__ == "__main__":
    unittest.main()
