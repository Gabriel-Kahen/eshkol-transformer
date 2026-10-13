"""Causal regression for the compiled held-holder gate's nm assertion."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-g3c4-p2g2-held-holder-symbols.py"
PREFIX = "et_g3c4_private_p2g2_holder_"
NAMES = (
    "new", "generation", "bind", "begin_capture", "witness",
    "normal_close", "abort_attempt", "end_after_busy", "dispose",
)


class HeldHolderSymbolGate(unittest.TestCase):
    def run_checker(self, mode, names):
        with tempfile.TemporaryDirectory() as directory:
            listing = Path(directory) / "symbols.txt"
            listing.write_text("".join(
                f"00000000 T {name}\n" for name in names))
            return subprocess.run(
                [sys.executable, str(CHECKER), mode, str(listing)],
                capture_output=True, text=True, check=False)

    def test_wrappers_do_not_count_as_holder_definitions(self):
        intended = [f"{PREFIX}{name}_v1" for name in NAMES]
        wrappers = [f"__wrap_{PREFIX}{name}_v1" for name in (
            "begin_capture", "witness", "abort_attempt", "end_after_busy")]
        result = self.run_checker("on", intended + wrappers)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [
            f"00000000 T {name}" for name in sorted(intended)])
        off = self.run_checker("off", wrappers)
        self.assertEqual(off.returncode, 0, off.stderr)
        self.assertFalse(off.stdout)

    def test_missing_extra_and_feature_off_exports_fail(self):
        intended = [f"{PREFIX}{name}_v1" for name in NAMES]
        missing = self.run_checker("on", intended[:-1])
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn(f"missing=['{intended[-1]}']", missing.stderr)
        extra_name = f"{PREFIX}unexpected_v1"
        extra = self.run_checker("on", intended + [extra_name])
        self.assertNotEqual(extra.returncode, 0)
        self.assertIn(f"extra=['{extra_name}']", extra.stderr)
        off = self.run_checker("off", [intended[0]])
        self.assertNotEqual(off.returncode, 0)
        self.assertIn("holder off ABI mismatch", off.stderr)


if __name__ == "__main__":
    unittest.main()
