"""Reject malformed D2 phase requests before toolchain or build work."""

from pathlib import Path
import subprocess
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "test-d2.sh"


class D2ResourcePhaseTests(unittest.TestCase):
    def test_invalid_phase_requests_fail_early(self):
        for args in (("--phase",), ("--phase", "semantics"),
                     ("--phase", "resources", "extra"), ("--unknown", "resources")):
            with self.subTest(args=args):
                result = subprocess.run(
                    ["/usr/bin/bash", str(SCRIPT), *args],
                    capture_output=True, text=True, timeout=5,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("usage:", result.stderr)
                self.assertNotIn("toolchain", result.stderr)


if __name__ == "__main__":
    unittest.main()
