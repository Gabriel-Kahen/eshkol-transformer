"""Exact source composition, tuple isolation, and observation negatives."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.e3_mode_graph.verify_observation import verify


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate-e3-mode-graph-source.py"
SPEC = importlib.util.spec_from_file_location("e3_mode_graph_generator", GENERATOR)
assert SPEC and SPEC.loader
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)
BASE = ROOT / "native/e3_private_package"
MODE = ROOT / "tests/e3_mode_graph/package"


def lines(prefix: Path, suffix: str) -> list[str]:
    return Path(f"{prefix}_{suffix}.txt").read_text().splitlines()


class ModeGraphTupleTest(unittest.TestCase):
    def test_exact_generated_sources_and_mutations(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e3-mode-graph-") as temporary:
            for mode in generator.OUTPUT_DIGESTS:
                output = Path(temporary) / mode
                for extra in ([], ["--check"]):
                    result = subprocess.run(
                        [sys.executable, "-B", str(GENERATOR),
                         "--output-dir", str(output), "--mode", mode, *extra],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                p1 = (output / "transformer/module.esk").read_text()
                extension = (output / "e3_mode_graph_extension.esk").read_text()
                self.assertEqual(generator.digest(p1),
                                 generator.OUTPUT_DIGESTS[mode][0])
                self.assertEqual(generator.digest(extension),
                                 generator.OUTPUT_DIGESTS[mode][1])
                self.assertEqual(extension.count(
                    "\n                                (e3-mode-graph-note! entry)"), 1)
                self.assertEqual(extension.count(
                    "\n                                         (e3-mode-graph-note! entry)"), 1)
                self.assertEqual(extension.count(
                    "\n                     (e3-mode-graph-preacquire! entry)"), 1)
                self.assertLess(extension.index(
                    "\n                     (e3-mode-graph-preacquire! entry)"),
                                extension.index("(e3-native-acquire"))
                self.assertLess(extension.index(
                    "\n                                (e3-mode-graph-note! entry)"),
                                extension.index("(e3-forward-schedule-internal frame)"))
                if mode == "graph-create":
                    for operation in ("m3t-public-workspace-begin!",
                                      "m3-forward-schedule-internal",
                                      "m3-native-graph-capture",
                                      "m3-reset!", "m3-native-graph-release"):
                        self.assertIn(operation, extension)
                    self.assertNotIn("g3t-checked-m3t-workspace-begin!",
                                     extension)
                else:
                    self.assertNotIn("m3-native-graph-capture", extension)
            normal = (Path(temporary) / "normal/transformer/module.esk").read_text()
            missed = (Path(temporary) / "missed-child/transformer/module.esk").read_text()
            enter_only = (Path(temporary) / "enter-write-only/transformer/module.esk").read_text()
            old = "(vector-set! (vector-ref nodes 2) 1 e3-mode-eval)"
            self.assertEqual(enter_only, normal.replace(old, "#t", 1))
            self.assertEqual(missed.count("(if (= index 2) e3-mode-train e3-mode-eval)"), 1)
            self.assertNotIn("(if (= index 2) e3-mode-train e3-mode-eval)", enter_only)
            poisoned = Path(temporary) / "normal/e3_mode_graph_extension.esk"
            poisoned.write_text(poisoned.read_text() + "; poisoned\n")
            result = subprocess.run(
                [sys.executable, "-B", str(GENERATOR), "--output-dir",
                 str(Path(temporary) / "normal"), "--mode", "normal", "--check"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("bytes differ", result.stderr)

    def test_exact_noninstalled_manifests(self) -> None:
        for suffix in ("native_objects",):
            self.assertEqual(lines(MODE, suffix), lines(BASE, suffix))
        for suffix in ("defined_symbols", "public_exports", "public_strings"):
            self.assertEqual(lines(MODE, suffix),
                             sorted(lines(BASE, suffix) +
                                    ["et_e3_mode_graph_run_v1"]))
        self.assertEqual(lines(MODE, "private_renames"),
                         lines(BASE, "private_renames") + [
                             "e3-mode-graph-dispatch "
                             "et_e3_mode_graph_dispatch_cabi_v1"])
        closure = lines(MODE, "source_closure")
        self.assertEqual(closure[:2], ["tests/e3_mode_graph/driver_root.esk",
                                       "tests/e3_mode_graph/private_root.esk"])
        self.assertEqual(closure.count("@BUILD@/source/transformer/module.esk"), 1)
        self.assertEqual(closure.count("@BUILD@/source/e3_mode_graph_extension.esk"), 1)
        self.assertNotIn("internal/p1/lib/transformer/module.esk", closure)
        self.assertNotIn("native/e3_private_extension.esk", closure)
        self.assertEqual(closure[-2:], ["tests/e3_private/driver.esk",
                                        "tests/e3_mode_graph/driver.esk"])
        native = lines(MODE, "native_source_closure")
        self.assertEqual(native.count("tests/e3_mode_graph/bridge.c"), 1)
        self.assertEqual(lines(MODE, "archive_members"),
                         ["e3_mode_graph_package.o"])
        for tuple_root in ("native/e3_private_package",
                           "native/e3_diagnostic_private_package",
                           "native/e3_diagnostic_public_package",
                           "tests/e3_private/numerical_package"):
            source = lines(ROOT / tuple_root, "source_closure")
            self.assertNotIn("@BUILD@/source/transformer/module.esk", source)
            self.assertFalse(any("mode_graph" in item for item in source))
        # Retention deliberately opts into only the accepted P1 mode seam.
        retention = lines(ROOT / "tests/e3_private/retention_package", "source_closure")
        self.assertEqual(retention.count("@BUILD@/source/transformer/module.esk"), 1)
        self.assertNotIn("internal/p1/lib/transformer/module.esk", retention)
        self.assertIn("native/e3_private_extension.esk", retention)
        self.assertFalse(any("mode_graph" in item for item in retention))

    def test_exact_tuple_admission_and_near_match_rejection(self) -> None:
        policy = ROOT / "scripts/e3-private-package-policy.sh"
        root = ROOT / "tests/e3_mode_graph/driver_root.esk"
        bridge = ROOT / "tests/e3_mode_graph/bridge.c"
        script = '''
PROJECT_ROOT=$1
raw_private_root=$2
raw_package_bridge=$3
raw_package_renames=$4
raw_public_exports=$5
raw_include_dirs=("${PROJECT_ROOT}/internal/p1/lib" "${PROJECT_ROOT}/internal/c1/lib" "${PROJECT_ROOT}/internal/t1/lib" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/internal/d2/lib" "${PROJECT_ROOT}/internal/e3/lib")
die() { printf '%s\\n' "$*" >&2; exit 77; }
source "$6"
printf '%s\\n' "$e3_tuple_kind"
'''
        values = [str(ROOT), str(root), str(bridge),
                  str(Path(f"{MODE}_private_renames.txt")),
                  str(Path(f"{MODE}_public_exports.txt")), str(policy)]
        exact = subprocess.run(["bash", "-c", script, "bash", *values],
                               capture_output=True, text=True)
        self.assertEqual((exact.returncode, exact.stdout), (0, "mode-graph\n"),
                         exact.stderr)
        values[2] = str(ROOT / "native/e3_private_bridge.c")
        mixed = subprocess.run(["bash", "-c", script, "bash", *values],
                               capture_output=True, text=True)
        self.assertEqual(mixed.returncode, 77)
        self.assertIn("mixed", mixed.stderr)

    def test_compiled_runner_and_observer_controls(self) -> None:
        builder = (ROOT / "scripts/build-e1b-consumer.sh").read_text()
        frame = (ROOT / "src/eshkol_transformer/m3_model.c").read_text()
        runner = (ROOT / "scripts/test-e3-mode-graph.sh").read_text()
        driver = (ROOT / "tests/e3_mode_graph/driver.esk").read_text()
        self.assertIn('"${e3_tuple_kind}" == mode-graph', builder)
        self.assertIn("-DET_E3_MODE_GRAPH_TESTING", builder)
        self.assertIn("e3_private_test_reloc_flags=(-Wl,-d)", builder)
        self.assertIn("#ifdef ET_E3_MODE_GRAPH_TESTING", frame)
        self.assertIn("enroll(&g->r, M3_GRAPH);", frame)
        self.assertIn("++e3_graph_enrollments", frame)
        self.assertIn("++e3_graph_root_releases", frame)
        self.assertIn('cmp "${out}/package-a/e3_mode_graph_package.o"', runner)
        self.assertIn('ASAN_OPTIONS=detect_leaks=1:halt_on_error=1', runner)
        self.assertIn('kill-missed-child-${gradient}', runner)
        self.assertIn('kill-graph-create-${gradient}', runner)
        self.assertIn('status}" == 134', runner)
        self.assertIn(':real et_e3_test_retention_probe_v1', driver)
        self.assertIn("(>= index 17) (<= index 26)", driver)
        self.assertIn("(e3-mg-owner-controls-restored? probes)", driver)

    def test_parser_rejects_missing_or_mismatched_in_call_data(self) -> None:
        def transcript(mask: int, graph: int) -> str:
            result: list[str] = []
            for label, count in (("success", 4), ("later-failure", 1),
                                 ("retry", 4)):
                for ordinal in range(count):
                    result.append(f"E3MG-NOTE {label} {ordinal} {mask} "
                                  f"{graph} {graph} {graph} 0")
                result.append(f"E3MG-END {label} {count} "
                              f"{graph} {graph} {graph} 0")
            result.append("E3-MODE-GRAPH-RUNTIME-PASS")
            return "\n".join(result) + "\n"
        verify(transcript((1 << 17) - 1, 0), "normal")
        verify(transcript(((1 << 17) - 1) ^ 4, 0), "missed-child")
        verify(transcript((1 << 17) - 1, 1), "graph-create")
        with self.assertRaisesRegex(ValueError, "in-call mode-mask"):
            verify(transcript(((1 << 17) - 1) ^ 4, 0), "normal")
        with self.assertRaisesRegex(ValueError, "in-call graph-event"):
            verify(transcript((1 << 17) - 1, 1), "normal")
        with self.assertRaisesRegex(ValueError, "did not complete"):
            verify(transcript((1 << 17) - 1, 0).rsplit("\n", 2)[0], "normal")


if __name__ == "__main__":
    unittest.main()
