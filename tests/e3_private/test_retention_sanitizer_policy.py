"""Reject drift in the actual retention tuple's sanitizer import inventory."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "tests/e3_private/retention_package"


class RetentionSanitizerPolicyTests(unittest.TestCase):
    def run_policy(self, kind=None, mutation=None):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            inputs = []
            for name in ("undefined_symbols", "sanitized_noninstrumentation_undefined_symbols",
                         "sanitized_instrumentation_undefined_symbols"):
                target = out / name
                target.write_bytes(Path(str(PREFIX) + "_" + name + ".txt").read_bytes())
                inputs.append(target)
            if kind is not None:
                rows = inputs[kind].read_text().splitlines()
                if mutation == "missing":
                    rows.pop()
                elif mutation == "extra":
                    rows.append("zz_unreviewed_import")
                elif mutation == "same_count":
                    rows[-1] = "zz_unreviewed_import"
                inputs[kind].write_text("\n".join(sorted(rows)) + "\n")
            return subprocess.run(
                ["bash", "-c", 'set -euo pipefail; die() { echo "$*" >&2; exit 1; }; '
                 'PROJECT_ROOT=$2; source "$1"; '
                 'e3_retention_verify_sanitizer_imports "$3" "$4" "$5"',
                 "policy", str(ROOT / "scripts/e3-retention-sanitizer-import-policy.sh"),
                 str(ROOT), *(str(p) for p in inputs)], capture_output=True)

    def test_exact_emitted_inventory(self):
        result = self.run_policy()
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_noninstrumentation_drift(self):
        for mutation in ("missing", "extra", "same_count"):
            with self.subTest(mutation=mutation):
                self.assertNotEqual(self.run_policy(1, mutation).returncode, 0)

    def test_instrumentation_drift(self):
        for mutation in ("missing", "extra", "same_count"):
            with self.subTest(mutation=mutation):
                self.assertNotEqual(self.run_policy(2, mutation).returncode, 0)

    def test_normal_inventory_drift(self):
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                self.assertNotEqual(self.run_policy(0, mutation).returncode, 0)

    def test_only_sanitized_retention_branch_admits_delta(self):
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        sanitized = builder.split('if [[ "${e3_private_test_sanitize}" == 1 ]]; then')[-1]
        self.assertIn('if [[ "${e3_tuple_kind:-}" == retention ]]; then', sanitized)
        self.assertIn('die "E3 sanitized private test object adds noninstrumentation imports"', sanitized)
        self.assertIn('e1b_compare_exact_undefined_allowlist', sanitized)


if __name__ == "__main__":
    unittest.main()
