"""Exact tuple and installed-closure policy for the bounded E3 facade."""
from pathlib import Path
import itertools
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native"


def lines(name: str) -> list[str]:
    return (NATIVE / name).read_text().splitlines()


class PublicPackagePolicy(unittest.TestCase):
    def tuple(self, kind: str) -> list[Path]:
        if kind == "base":
            stem, prefix = "e3_private", "e3_private_package"
        elif kind == "diagnostic":
            stem, prefix = "e3_diagnostic_private", "e3_diagnostic_private_package"
        else:
            stem, prefix = "e3_diagnostic_public", "e3_diagnostic_public_package"
        return [
            NATIVE / f"{stem}_root.esk" if kind == "public" else NATIVE / f"{stem}_driver_root.esk",
            NATIVE / f"{stem}_bridge.c",
            NATIVE / f"{prefix}_private_renames.txt",
            NATIVE / f"{prefix}_public_exports.txt",
        ]

    def run_policy(self, inputs: list[Path]) -> subprocess.CompletedProcess[str]:
        script = r'''
set -euo pipefail
die() { printf '%s\n' "$1" >&2; exit 1; }
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=(
  "${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib"
  "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src"
  "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib"
)
source "${PROJECT_ROOT}/scripts/e3-private-package-policy.sh"
printf 'kind=%s\nprefix=%s\ndiagnostic=%s\npublic=%s\n' \
  "${e3_tuple_kind}" "${e3_prefix}" "${e3_diagnostic_tuple}" "${e3_public_tuple}"
'''
        return subprocess.run(
            ["bash", "-c", script, "policy", str(ROOT), *(str(v) for v in inputs)],
            check=False, text=True, capture_output=True,
        )

    def test_three_exact_tuples_select_without_changing_predecessors(self) -> None:
        for kind, diagnostic, public in (
            ("base", "0", "0"), ("diagnostic", "1", "0"),
            ("public", "0", "1"),
        ):
            result = self.run_policy(self.tuple(kind))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"kind={kind}\n", result.stdout)
            self.assertIn(f"diagnostic={diagnostic}\n", result.stdout)
            self.assertIn(f"public={public}\n", result.stdout)

    def test_every_mixed_or_partial_tuple_rejects(self) -> None:
        tuples = {kind: self.tuple(kind) for kind in ("base", "diagnostic", "public")}
        for selectors in itertools.product(tuples, repeat=4):
            if len(set(selectors)) == 1:
                continue
            values = [tuples[kind][index] for index, kind in enumerate(selectors)]
            result = self.run_policy(values)
            self.assertNotEqual(result.returncode, 0, selectors)
            self.assertIn("E3 policy rejects mixed", result.stderr, selectors)
        unrelated = NATIVE / "m3_package_bridge.c"
        public = self.tuple("public")
        for index in range(4):
            values = list(public)
            values[index] = unrelated
            result = self.run_policy(values)
            self.assertNotEqual(result.returncode, 0, index)
            self.assertIn("requires the exact lexical repository tuple", result.stderr)

    def test_public_exports_are_exact_existing_d2_plus_three_e3(self) -> None:
        base = set(lines("e3_private_package_public_exports.txt"))
        d2 = set(lines("d2_wave2_public_exports.txt"))
        public = set(lines("e3_diagnostic_public_package_public_exports.txt"))
        expected = (base - {"et_e3_test_run_v1"}) | d2 | {
            "et_e1b_public_e3_diagnostic_evaluate_fixed_v1",
            "et_e1b_public_e3_diagnostic_evaluation_f32_bits_v1",
            "et_e1b_public_e3_diagnostic_evaluation_count_v1",
        }
        self.assertEqual(public, expected)
        self.assertFalse(any("_test_" in value or value.startswith("et_e3_")
                             for value in public))
        self.assertEqual(lines("e3_diagnostic_public_package_public_exports.txt"),
                         sorted(public))

    def test_public_manifests_and_closures_are_exact(self) -> None:
        prefix = "e3_diagnostic_public_package"
        exports = set(lines(f"{prefix}_public_exports.txt"))
        errors = set(lines("e3_diagnostic_private_package_defined_symbols.txt")) - set(
            lines("e3_diagnostic_private_package_public_exports.txt"))
        self.assertEqual(set(lines(f"{prefix}_defined_symbols.txt")), exports | errors)
        self.assertEqual(lines(f"{prefix}_public_strings.txt"),
                         lines(f"{prefix}_defined_symbols.txt"))
        self.assertEqual(lines(f"{prefix}_undefined_symbols.txt"),
                         lines("e3_diagnostic_private_package_undefined_symbols.txt"))
        self.assertEqual(lines(f"{prefix}_native_objects.txt"),
                         lines("e3_diagnostic_private_package_native_objects.txt"))
        source = lines(f"{prefix}_source_closure.txt")
        self.assertEqual(source[0], "native/e3_diagnostic_public_root.esk")
        self.assertEqual(source[-1], "native/e3_diagnostic_private_extension.esk")
        self.assertFalse(any(value.startswith("tests/") for value in source))
        native = lines(f"{prefix}_native_source_closure.txt")
        self.assertIn("native/e3_diagnostic_public_bridge.c", native)
        self.assertIn("native/e3_d2_public_wrappers.inc", native)
        self.assertNotIn("native/e3_private_bridge.c", native)
        self.assertNotIn("native/e3_diagnostic_private_bridge.c", native)
        self.assertEqual(
            (NATIVE / "e3_d2_public_wrappers.inc").read_text(),
            "".join((NATIVE / "d2_wave2_package_bridge.c").read_text().splitlines(True)[7:]),
        )

    def test_installed_facades_and_builder_branch_are_bounded(self) -> None:
        self.assertEqual(lines("e3_diagnostic_public_package_facades.txt"), [
            "transformer/config.esk", "transformer/data.esk",
            "transformer/diagnostic_transport.esk", "transformer/error_consumer.esk",
            "transformer/error_public.esk", "transformer/evaluation.esk",
            "transformer/model.esk", "transformer/module.esk",
            "transformer/tokenizer.esk",
        ])
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('"${e3_diagnostic_tuple}" == 1 || "${e3_public_tuple}" == 1', builder)
        self.assertIn('if [[ "${e3_public_tuple:-0}" != 1 ]]; then\n      package_native_define=-DET_E3_TESTING', builder)
        self.assertGreaterEqual(builder.count('"${e3_public_tuple:-0}" != 1'), 4)


if __name__ == "__main__":
    unittest.main()
