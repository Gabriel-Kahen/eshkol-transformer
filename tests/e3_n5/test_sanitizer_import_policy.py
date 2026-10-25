"""Reject import drift in the measured noninstalled N5 sanitizer tuple."""

from pathlib import Path
import hashlib
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PREFIX = ROOT / "tests/e3_n5/package"
POLICY = ROOT / "scripts/e3-n5-sanitizer-import-policy.sh"


class N5SanitizerImportPolicyTests(unittest.TestCase):
    def run_policy(self, change=None, kind="n5"):
        with tempfile.TemporaryDirectory(prefix="e3-n5-imports-") as directory:
            normal = Path(str(PREFIX) + "_undefined_symbols.txt")
            non = Path(str(PREFIX) + "_sanitized_noninstrumentation_undefined_symbols.txt")
            instrumentation = Path(str(PREFIX) + "_sanitized_instrumentation_undefined_symbols.txt")
            originals = (normal, non, instrumentation, non, instrumentation)
            files = []
            for index, original in enumerate(originals):
                target = Path(directory) / f"inventory-{index}.txt"
                target.write_bytes(original.read_bytes())
                files.append(target)
            if change is not None:
                index, old, new = change
                rows = files[index].read_text().splitlines()
                if old is None:
                    rows.append(new)
                elif new is None:
                    rows.remove(old)
                else:
                    rows[rows.index(old)] = new
                files[index].write_text("\n".join(sorted(rows)) + "\n")
            script = ('set -euo pipefail; die() { echo "$*" >&2; exit 77; }; '
                      'e3_tuple_kind=$1; source "$2"; '
                      'e3_n5_verify_sanitizer_imports "$3" "$4" "$5" "$6" "$7"')
            return subprocess.run(
                ["bash", "-c", script, "policy", kind, str(POLICY),
                 *(str(path) for path in files)],
                capture_output=True, text=True, check=False,
            )

    def test_exact_measured_split_and_delta(self):
        self.assertEqual(self.run_policy().returncode, 0)
        normal = set(Path(str(PREFIX) + "_undefined_symbols.txt").read_text().splitlines())
        non = set(Path(str(PREFIX) + "_sanitized_noninstrumentation_undefined_symbols.txt")
                  .read_text().splitlines())
        instrumentation = set(Path(str(PREFIX) + "_sanitized_instrumentation_undefined_symbols.txt")
                              .read_text().splitlines())
        self.assertEqual((len(normal), len(non), len(instrumentation)), (176, 170, 57))
        for suffix, digest in (
            ("undefined_symbols", "59b619891b121be4e872840a8ba4be3150e3041b3e4569447293409f4ce8f708"),
            ("sanitized_noninstrumentation_undefined_symbols", "ce882fcb0794927c83225366cfa22e1e76bfe2de32e089f7ad87a58d6c742ed4"),
            ("sanitized_instrumentation_undefined_symbols", "e500dfb82c45fcd94c2d2cef383e4868c73088f892fd5c42b95d9c49dcd94d7b"),
        ):
            path = Path(str(PREFIX) + f"_{suffix}.txt")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(non - normal, {"bcmp", "puts"})
        self.assertEqual(normal - non, {
            "eshkol_ad_seed_flag", "eshkol_runtime_fatal", "eshkol_taylor_lift_ad_node",
            "memcmp", "memcpy", "memmove", "memset", "printf",
        })

    def test_rejects_unrelated_tuple(self):
        self.assertEqual(self.run_policy(kind="numerical").returncode, 77)

    def test_rejects_actual_import_drift(self):
        for change in ((1, "bcmp", None), (1, None, "zz_unreviewed_import"),
                       (1, "puts", "zz_unreviewed_import"),
                       *((1, symbol, None) for symbol in
                         ("eshkol_display_value", "eshkol_runtime_current_output_fp",
                          "fflush", "fputc")),
                       (2, "__asan_load8", None),
                       (2, None, "__asan_unreviewed")):
            with self.subTest(change=change):
                self.assertEqual(self.run_policy(change).returncode, 77)

    def test_rejects_manifest_and_normal_drift(self):
        for change in ((0, "printf", None), (0, None, "zz_unreviewed_import"),
                       (0, "fflush", "zz_unreviewed_import"),
                       (3, "bcmp", "zz_unreviewed_import"),
                       (3, "fputc", "zz_unreviewed_import"),
                       (4, "__asan_load8", "__asan_unreviewed")):
            with self.subTest(change=change):
                self.assertEqual(self.run_policy(change).returncode, 77)

    def test_builder_uses_n5_only_policy(self):
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        branch = builder.split('elif [[ "${e3_tuple_kind:-}" == n5 ]]; then', 1)[1].split(
            'elif [[ "${e3_tuple_kind:-}" == numerical', 1)[0]
        self.assertIn('source "${PROJECT_ROOT}/scripts/e3-n5-sanitizer-import-policy.sh"', branch)
        self.assertIn('e3_n5_verify_sanitizer_imports', branch)
        self.assertIn('e3_retention_sanitizer_inventory_verified=1', branch)


if __name__ == "__main__":
    unittest.main()
