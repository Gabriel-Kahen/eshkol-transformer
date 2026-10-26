"""The one supplemental retention import cannot widen E3 private authority."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "scripts/e3-private-unresolved-policy.sh"
OBSERVER = "et_e3_retention_region_used_v1"


class RetentionUnresolvedPolicyTest(unittest.TestCase):
    def test_builder_passes_derived_exact_tuple_and_inventory(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        self.assertIn('source "${PROJECT_ROOT}/scripts/e3-private-package-policy.sh"',
                      builder)
        self.assertIn('source "${PROJECT_ROOT}/scripts/e3-private-unresolved-policy.sh"',
                      builder)
        self.assertIn('e3_reject_unresolved_private "${e3_tuple_kind}"', builder)
        self.assertIn('"${e1b_tmp}/undefined.txt" "${e1b_tmp}/expected-undefined.txt"', builder)
        self.assertLess(builder.index('e1b_compare_exact_undefined_allowlist',
                                      builder.index('e3_retention_sanitizer_inventory_verified=0')),
                        builder.index('e3_reject_unresolved_private'))

    def check_policy(self, kind: str, actual: list[str],
                     expected: list[str], accepted: bool,
                     *, boundary: bool = False) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            actual_path = directory / "actual.txt"
            expected_path = directory / "expected.txt"
            actual_path.write_text("\n".join(actual) + "\n")
            expected_path.write_text("\n".join(expected) + "\n")
            command = '''
set -euo pipefail
die() { printf '%s\\n' "$*" >&2; exit 77; }
source "$1"
e3_retention_boundary_diagnostic="$5"
e3_reject_unresolved_private "$2" "$3" "$4"
'''
            result = subprocess.run(
                ["bash", "-c", command, "bash", str(POLICY), kind,
                 str(actual_path), str(expected_path), "1" if boundary else "0"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0 if accepted else 77,
                             result.stderr)
            if not accepted:
                self.assertIn("unresolved private native authority",
                              result.stderr)

    def test_exact_declared_retention_supplement_only(self) -> None:
        runtime = ["eshkol_builtin_arena_used", "fputc"]
        for kind in ("retention", "retention-overflow"):
            self.check_policy(kind, runtime + [OBSERVER],
                              runtime + [OBSERVER], True)
            self.check_policy(kind, runtime + [OBSERVER], runtime, False)
        for kind in ("base", "diagnostic", "public", "numerical", "uniform",
                     "mode-graph", "retention-overflows", ""):
            self.check_policy(kind, runtime + [OBSERVER],
                              runtime + [OBSERVER], False)

    def test_other_private_imports_remain_forbidden(self) -> None:
        for forbidden in ("et_e3_private_acquire_v1",
                          "et_d2_private_cursor_v1",
                          "et_m3_private_graph_capture_v1"):
            for kind in ("retention", "retention-overflow"):
                self.check_policy(kind, [OBSERVER, forbidden],
                                  [OBSERVER, forbidden], False)

    def test_boundary_hooks_require_exact_diagnostic_authority(self) -> None:
        hooks = ["et_e3_retention_boundary_mark_v1",
                 "et_e3_retention_boundary_read_v1"]
        self.check_policy("retention", hooks, hooks, True, boundary=True)
        self.check_policy("retention", hooks, hooks, False)
        self.check_policy("fault", hooks, hooks, False, boundary=True)
        self.check_policy("retention", hooks, hooks[:1], False, boundary=True)
        self.check_policy("retention", hooks + ["et_e3_private_acquire_v1"],
                          hooks + ["et_e3_private_acquire_v1"], False, boundary=True)

    def test_overflow_declares_and_links_the_same_observer(self) -> None:
        manifest = (ROOT / "tests/e3_private/overflow_package_undefined_symbols.txt")
        self.assertEqual(manifest.read_text().splitlines().count(OBSERVER), 1)
        runner = (ROOT / "scripts/test-e3-finite-overflow.sh").read_text()
        self.assertIn('tests/e3_private/retention_arena.cpp', runner)
        self.assertIn('"$out/$mode/arena.o"', runner)
        supplement = (ROOT / "tests/e3_private/retention_arena.cpp").read_text()
        self.assertIn('extern "C" int64_t ' + OBSERVER + '()', supplement)


if __name__ == "__main__":
    unittest.main()
