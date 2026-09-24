"""Lightweight exact-tuple checks for the E3 diagnostic package successor."""

from pathlib import Path
import itertools
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "native"


def lines(name: str) -> list[str]:
    return (NATIVE / name).read_text().splitlines()


class DiagnosticPackagePolicy(unittest.TestCase):
    def run_policy(self, inputs: list[Path]) -> subprocess.CompletedProcess[str]:
        quoted = [str(path) for path in inputs]
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
printf 'kind=%s\nprefix=%s\ndiagnostic=%s\n' \
  "${e3_tuple_kind}" "${e3_prefix}" "${e3_diagnostic_tuple}"
printf 'input=%s\n' "${e3_inputs[@]}"
'''
        return subprocess.run(
            ["bash", "-c", script, "policy", str(ROOT), *quoted],
            check=False,
            text=True,
            capture_output=True,
        )

    def tuple(self, diagnostic: bool) -> list[Path]:
        prefix = "e3_diagnostic_private_package" if diagnostic else "e3_private_package"
        stem = "e3_diagnostic_private" if diagnostic else "e3_private"
        return [
            NATIVE / f"{stem}_driver_root.esk",
            NATIVE / f"{stem}_bridge.c",
            NATIVE / f"{prefix}_private_renames.txt",
            NATIVE / f"{prefix}_public_exports.txt",
        ]

    def test_base_tuple_selection_is_unchanged(self) -> None:
        base = self.tuple(False)
        result = self.run_policy(base)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = (
            "kind=base\n"
            f"prefix={NATIVE / 'e3_private_package'}\n"
            "diagnostic=0\n"
            + "".join(f"input={value}\n" for value in base)
        )
        self.assertEqual(result.stdout, expected)

    def test_exact_diagnostic_tuple_is_selected(self) -> None:
        result = self.run_policy(self.tuple(True))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kind=diagnostic\n", result.stdout)
        self.assertIn("prefix=" + str(NATIVE / "e3_diagnostic_private_package") + "\n",
                      result.stdout)
        self.assertIn("diagnostic=1\n", result.stdout)
        self.assertEqual(result.stdout.count("input="), 4)

    def test_mixed_and_partial_tuples_reject(self) -> None:
        base = self.tuple(False)
        diagnostic = self.tuple(True)
        for selectors in itertools.product((False, True), repeat=4):
            if all(selectors) or not any(selectors):
                continue
            values = [diagnostic[index] if selected else base[index]
                      for index, selected in enumerate(selectors)]
            mixed = self.run_policy(values)
            self.assertNotEqual(mixed.returncode, 0, selectors)
            self.assertIn("rejects mixed base and diagnostic tuples",
                          mixed.stderr, selectors)

        unrelated = ROOT / "native/m3_package_bridge.c"
        for candidate in (base, diagnostic):
            for index in range(4):
                values = list(candidate)
                values[index] = unrelated
                partial = self.run_policy(values)
                self.assertNotEqual(partial.returncode, 0, (candidate, index))
                self.assertIn("requires the exact lexical repository tuple",
                              partial.stderr, (candidate, index))

    def test_manifest_delta_is_exact(self) -> None:
        base_prefix = "e3_private_package"
        diagnostic_prefix = "e3_diagnostic_private_package"
        additions = {
            "et_e3_diagnostic_test_failure_v1",
            "et_e3_diagnostic_test_run_v1",
        }
        for suffix in ("public_exports", "public_strings", "defined_symbols"):
            base = set(lines(f"{base_prefix}_{suffix}.txt"))
            diagnostic = set(lines(f"{diagnostic_prefix}_{suffix}.txt"))
            self.assertEqual(diagnostic - base, additions)
            self.assertEqual(base - diagnostic, set())
        base_undefined = set(lines(f"{base_prefix}_undefined_symbols.txt"))
        diagnostic_undefined = set(
            lines(f"{diagnostic_prefix}_undefined_symbols.txt")
        )
        self.assertEqual(
            diagnostic_undefined - base_undefined,
            {"eshkol_builtin_arena_used"},
        )
        self.assertEqual(base_undefined - diagnostic_undefined, set())
        self.assertEqual(
            lines(f"{diagnostic_prefix}_undefined_symbols.txt"),
            sorted(diagnostic_undefined),
        )
        self.assertEqual(lines(f"{base_prefix}_native_objects.txt"),
                         lines(f"{diagnostic_prefix}_native_objects.txt"))

        source = lines(f"{diagnostic_prefix}_source_closure.txt")
        self.assertEqual(source[0], "native/e3_diagnostic_private_driver_root.esk")
        self.assertEqual(source[1:-2], lines(f"{base_prefix}_source_closure.txt"))
        self.assertEqual(source[-2:], [
            "native/e3_diagnostic_private_extension.esk",
            "tests/e3_diagnostic_private/driver.esk",
        ])

        native = lines(f"{diagnostic_prefix}_native_source_closure.txt")
        stripped = [value for value in native if value not in {
            "native/e3_diagnostic_private_bridge.c",
            "native/e3_diagnostic_destinations.c",
        }]
        self.assertEqual(stripped, lines(f"{base_prefix}_native_source_closure.txt"))

        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        start = builder.index("public_string_pattern='")
        end = builder.index(
            'LC_ALL=C grep -E "${public_string_pattern}"', start
        )
        selection = builder[start:end]
        candidates = [
            "et_e1b_error_category_v1",
            "et_e1b_public_m3_model_forward_v1",
            "et_e3_test_run_v1",
            "et_e3_diagnostic_test_failure_v1",
            "et_e3_diagnostic_test_run_v1",
            "et_e3_diagnostic_test_other_v1",
        ]
        script = (
            "set -euo pipefail\n"
            "package_policy=e3-private-aggregate\n"
            "e3_diagnostic_tuple=$1\n"
            + selection
            + "printf '%s\\n' "
            + " ".join(candidates)
            + ' | LC_ALL=C grep -E "${public_string_pattern}"\n'
        )
        base_strings = subprocess.run(
            ["bash", "-c", script, "extract", "0"], check=True,
            text=True, capture_output=True,
        ).stdout.splitlines()
        diagnostic_strings = subprocess.run(
            ["bash", "-c", script, "extract", "1"], check=True,
            text=True, capture_output=True,
        ).stdout.splitlines()
        self.assertEqual(base_strings, candidates[:3])
        self.assertEqual(diagnostic_strings, candidates[:5])

    def test_no_provisional_public_surface_is_exported(self) -> None:
        exported = "\n".join(lines("e3_diagnostic_private_package_public_exports.txt"))
        for name in (
            "diagnostic_evaluate_fixed",
            "diagnostic_evaluation_f32_bits",
            "diagnostic_evaluation_count",
        ):
            self.assertNotIn(name, exported)


if __name__ == "__main__":
    unittest.main()
