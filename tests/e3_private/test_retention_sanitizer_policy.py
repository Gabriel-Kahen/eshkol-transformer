"""Reject drift in the actual retention tuple's sanitizer import inventory."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "tests/e3_private/retention_package"
FAULT_PREFIX = ROOT / "tests/e3_private/fault_package"
MUTANT_PREFIX = ROOT / "tests/e3_private/retention_first_eval_mutant_package"


class RetentionSanitizerPolicyTests(unittest.TestCase):
    def run_policy(self, kind=None, mutation=None, tuple_kind="retention"):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            inputs = []
            prefix = {"fault": FAULT_PREFIX, "retention": PREFIX,
                      "retention-first-eval-mutant": MUTANT_PREFIX}[tuple_kind]
            for name in ("undefined_symbols", "sanitized_noninstrumentation_undefined_symbols",
                         "sanitized_instrumentation_undefined_symbols"):
                target = out / name
                target.write_bytes(Path(str(prefix) + "_" + name + ".txt").read_bytes())
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
                 'e3_tuple_kind=$1; PROJECT_ROOT=$3; source "$2"; '
                 'e3_retention_verify_sanitizer_imports "$4" "$5" "$6"',
                 "policy", tuple_kind,
                 str(ROOT / "scripts/e3-retention-sanitizer-import-policy.sh"),
                 str(ROOT), *(str(p) for p in inputs)], capture_output=True)

    def test_exact_emitted_inventory(self):
        for tuple_kind in ("retention", "retention-first-eval-mutant"):
            with self.subTest(tuple_kind=tuple_kind):
                result = self.run_policy(tuple_kind=tuple_kind)
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

    def test_fault_stage3_policy_rejects_drift(self):
        self.assertEqual(self.run_policy(tuple_kind="fault").returncode, 0)
        for kind in (1, 2):
            self.assertNotEqual(
                self.run_policy(kind, "same_count", tuple_kind="fault").returncode, 0)

    def test_verified_split_inventories_reach_published_evidence(self):
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        start = builder.index('if [[ "${e3_retention_sanitizer_inventory_verified}" == 1 ]]; then')
        end = builder.index('\nfi\n', start) + len('\nfi\n')
        publication = builder[start:end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch = root / "scratch"
            scratch.mkdir()
            expected = {}
            for target, suffix in (
                ("noninstrumentation-undefined.txt", "sanitized_noninstrumentation_undefined_symbols"),
                ("sanitizer-undefined.txt", "sanitized_instrumentation_undefined_symbols"),
            ):
                expected[target] = Path(str(PREFIX) + "_" + suffix + ".txt").read_bytes()
                (scratch / target).write_bytes(expected[target])
            for verified in ("1", "0"):
                evidence = root / ("evidence" + verified)
                result = subprocess.run(
                    ["bash", "-c", 'set -euo pipefail; e1b_tmp=$1; evidence_dir=$2; '
                     'package_policy=e3-private-aggregate; e3_tuple_kind=retention; '
                     'e3_retention_sanitizer_inventory_verified=$3; '
                     'mkdir "${evidence_dir}.tmp.$$"; ' + publication +
                     'mv "${evidence_dir}.tmp.$$" "$evidence_dir"',
                     "publication", str(scratch), str(evidence), verified], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                if verified == "1":
                    for name, contents in expected.items():
                        self.assertEqual((evidence / name).read_bytes(), contents)
                else:
                    self.assertEqual(list(evidence.iterdir()), [])

    def test_only_sanitized_retention_branch_admits_delta(self):
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        start = builder.index('e3_retention_sanitizer_inventory_verified=0\n'
                              'if [[ "${e3_private_test_sanitize}" == 1 ]]; then')
        end = builder.index('if grep -E ', start)
        sanitized = builder[start:end]
        self.assertIn('if [[ "${e3_tuple_kind:-}" == fault ||\n'
                      '        "${e3_tuple_kind:-}" == retention ]]; then', sanitized)
        self.assertIn('e3_retention_verify_sanitizer_imports', sanitized)
        self.assertIn('die "E3 sanitized private test object adds noninstrumentation imports"', sanitized)
        self.assertIn('e1b_compare_exact_undefined_allowlist', sanitized)


if __name__ == "__main__":
    unittest.main()
