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
                elif mutation == "foreign_same_count":
                    removed, added = (("puts", "getrandom") if kind == 1 else
                                      ("__asan_load1", "__asan_report_load16"))
                    rows.remove(removed)
                    rows.append(added)
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

    def test_instrumented_ir_import_signature(self):
        normal = set(Path(str(PREFIX) + "_undefined_symbols.txt").read_text().splitlines())
        non = set(Path(str(PREFIX) + "_sanitized_noninstrumentation_undefined_symbols.txt").read_text().splitlines())
        instrumentation = set(Path(str(PREFIX) + "_sanitized_instrumentation_undefined_symbols.txt").read_text().splitlines())
        self.assertEqual((len(non), len(instrumentation)), (172, 59))
        self.assertEqual(non - normal, {"bcmp", "puts"})
        self.assertEqual(normal - non, {
            "eshkol_ad_seed_flag", "eshkol_runtime_fatal", "eshkol_taylor_lift_ad_node",
            "memcmp", "memcpy", "memmove", "memset", "printf",
        })
        self.assertTrue({
            "__asan_alloca_poison", "__asan_allocas_unpoison", "__asan_load1",
            "__asan_load8", "__asan_store1", "__asan_store2", "__asan_store4",
            "__asan_store8",
        } <= instrumentation)

    def test_unrelated_same_count_imports_rejected(self):
        for kind in (1, 2):
            with self.subTest(kind=kind):
                self.assertNotEqual(self.run_policy(kind, "foreign_same_count").returncode, 0)

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

    def test_verified_split_inventories_reach_published_evidence(self):
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        start = builder.index('if [[ "${e3_retention_sanitizer_inventory_verified}" == 1 ]]; then')
        end = builder.index('if [[ -n "${package_source_closure}" ]]; then', start)
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
        self.assertIn('if [[ "${e3_tuple_kind:-}" == retention ]]; then', sanitized)
        self.assertIn('e3_retention_verify_sanitizer_imports', sanitized)
        self.assertIn('die "E3 sanitized private test object adds noninstrumentation imports"', sanitized)
        self.assertIn('e1b_compare_exact_undefined_allowlist', sanitized)


if __name__ == "__main__":
    unittest.main()
