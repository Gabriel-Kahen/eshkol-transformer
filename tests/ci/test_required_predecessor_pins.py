"""Check required suite source pins before building the suites."""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = (
    # native-numerics
    "native/l3s_predecessor_sources.sha256",
    "tests/e3_metrics/expected/predecessor_sources.sha256",
    "tests/n3k/expected/predecessor_sources.sha256",
    # g3n-forward
    "tests/g3n/expected/predecessor_sources.sha256",
    "tests/g3c4/expected/predecessor_sources.sha256",
    # g3s-sampling
    "tests/g3s/expected/predecessor_sources.sha256",
    # model-composition
    "tests/m3cg/predecessor_sources.sha256",
)


class RequiredPredecessorPinsTest(unittest.TestCase):
    def test_required_predecessor_source_bytes(self):
        for manifest in MANIFESTS:
            with self.subTest(manifest=manifest):
                result = subprocess.run(
                    ["sha256sum", "--quiet", "-c", manifest],
                    cwd=ROOT, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
