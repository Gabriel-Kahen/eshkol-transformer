"""Synthetic rejection tests for the native evidence parser, not memory evidence."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.m3.check_native_retention import MODES, HORIZONS, check, expected, parse


def report(mode="reset", horizon=1024, fields=None):
    fields = expected(mode, horizon) if fields is None else fields
    return (f"M3-ARENA-RETENTION-PASS {mode} {horizon}\nM3 native "
            + " ".join(f"{k}={v}" for k, v in fields.items()) + "\n")


class NativeRetention(unittest.TestCase):
    def test_all_cases_and_exact_independent_slopes(self):
        with TemporaryDirectory() as root:
            for mode in MODES:
                for horizon in HORIZONS:
                    Path(root, f"arena-{mode}-{horizon}.stdout").write_text(report(mode, horizon))
            with redirect_stdout(StringIO()) as out:
                check(root)
            for mode, slope in (("forward", 1768), ("logits", 1888), ("vjp", 48), ("reset", 128), ("failure", 0)):
                self.assertIn(f"retention {mode}: scoped_control_bytes_per_iteration={slope}", out.getvalue())
            Path(root, "arena-reset-1024.stdout").unlink()
            with self.assertRaises(FileNotFoundError):
                check(root)

    def test_every_field_is_checked(self):
        fields = expected("reset", 1024)
        for key in fields:
            changed = dict(fields)
            changed[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse(report(fields=changed), "reset", 1024)
        # Updating a count and its matching byte total still violates the independently
        # derived fixture plan count; arithmetic self-consistency is insufficient.
        changed = dict(fields, **{"i2-retired-reset-plans": 1025, "i2-retired": fields["i2-retired"] + 40})
        with self.assertRaises(ValueError):
            parse(report(fields=changed), "reset", 1024)

    def test_malformed_duplicate_and_incomplete_report(self):
        valid = report()
        cases = [valid.replace("graphs=0 ", ""), valid.replace("graphs=0", "graphs=-1"),
                 valid.replace("graphs=0", "graphs=00"), valid.replace("graphs=0", "graphs=1e2"),
                 valid.replace("graphs=0", "graphs=0 graphs=0"), valid.replace("graphs=0", "extra=0 graphs=0"),
                 valid + valid, valid.splitlines()[1] + "\n", valid.replace("reset 1024", "vjp 1024"),
                 "\n".join(reversed(valid.splitlines())), "noise\n" + valid, valid + "noise\n"]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value, "reset", 1024)


if __name__ == "__main__":
    unittest.main()
