"""Reject records that would make a partial or inconsistent R2 run look complete."""

from pathlib import Path
import tempfile
import unittest

from tests.e3_private.verify_retention_evidence import (
    FIELDS, GRADIENTS, HORIZONS, PROFILES, verify,
)


def fixture(directory: Path) -> None:
    for horizon in HORIZONS:
        for gradient in GRADIENTS:
            begins = horizon + 1
            values = {
                "arena_used": 512 + (gradient == "present") * 64,
                "arena_total": 1024 + (gradient == "present") * 64,
                "arena_blocks": 1,
                "caller_region_used": 128,
                "failure_region_used": 256,
                "retry_region_used": 128,
                "borrow_begins": begins,
                "borrow_views": begins * 2,
                "retired_borrows": begins,
                "retired_control_bytes": begins * 32,
                "d2_live_allocations": 1,
                "fd_peak_growth": 1,
            }
            line = (f"E3-COMPLETE-CALL-RETENTION-PASS horizon={horizon} "
                    f"gradient={gradient} "
                    + " ".join(f"{key}={values[key]}" for key in FIELDS) + "\n")
            for profile in PROFILES:
                stem = directory / f"{horizon}-{gradient}-{profile}"
                Path(f"{stem}.stdout").write_text(line)
                Path(f"{stem}.stderr").write_text("")
                Path(f"{stem}.rss").write_text(
                    "rss_before_kib=900 rss_after_kib=1000\n")
                Path(f"{stem}.time").write_text(
                    "elapsed_seconds=0.10 peak_rss_kib=1100\n")


class RetentionEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        fixture(self.directory)

    def test_complete_records(self) -> None:
        self.assertEqual(verify(self.directory), {
            "runs": 20, "retired_control_bytes_each": 32,
            "maximum_peak_rss_kib": 1100,
        })

    def test_missing_run_is_not_a_pass(self) -> None:
        (self.directory / "8192-present-sanitized.stdout").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            verify(self.directory)

    def test_duplicate_field_is_not_a_pass(self) -> None:
        path = self.directory / "1024-absent-normal-a.stdout"
        path.write_text(path.read_text().replace(
            " arena_used=512", " arena_used=512 arena_used=512"))
        with self.assertRaisesRegex(ValueError, "malformed"):
            verify(self.directory)

    def test_horizon_growth_is_not_flat_retention(self) -> None:
        for profile in PROFILES:
            path = self.directory / f"8192-absent-{profile}.stdout"
            path.write_text(path.read_text().replace(
                " arena_used=512", " arena_used=513"))
        with self.assertRaisesRegex(ValueError, "baseline changes"):
            verify(self.directory)

    def test_borrow_accounting_must_reconcile(self) -> None:
        for profile in PROFILES:
            path = self.directory / f"1024-absent-{profile}.stdout"
            path.write_text(path.read_text().replace(
                " retired_borrows=1025", " retired_borrows=1024"))
        with self.assertRaisesRegex(ValueError, "borrow controls"):
            verify(self.directory)

    def test_peak_must_cover_samples(self) -> None:
        path = self.directory / "1024-absent-normal-a.time"
        path.write_text("elapsed_seconds=0.10 peak_rss_kib=999\n")
        with self.assertRaisesRegex(ValueError, "resource measurement"):
            verify(self.directory)


if __name__ == "__main__":
    unittest.main()
