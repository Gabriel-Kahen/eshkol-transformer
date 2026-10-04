"""Source identity and fault-site checks for the disposable terminal overlays."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.g3c4 import check_p2_g2_terminal_compiler_evidence as compiler_evidence
from tests.shared.prepare_eshkol_asan_ir import prepare, sha


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tests/g3c4/generate_p2_g2_terminal_fault_overlays.py"
SPEC = importlib.util.spec_from_file_location("terminal_overlays", GENERATOR)
OVERLAYS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OVERLAYS)


class TerminalOverlayTests(unittest.TestCase):
    def test_exact_sources_and_fault_sites(self) -> None:
        t1 = OVERLAYS.T1.read_bytes()
        composer = OVERLAYS.COMPOSER.read_bytes()
        self.assertEqual(OVERLAYS.digest(t1), OVERLAYS.T1_SHA)
        self.assertEqual(OVERLAYS.digest(composer), OVERLAYS.COMPOSER_SHA)
        t1_text = t1.decode("utf-8")
        composer_text = composer.decode("utf-8")
        self.assertEqual(t1_text.count(OVERLAYS.T1_MARKER), 1)
        self.assertEqual(composer_text.count(OVERLAYS.COMPOSER_MARKER), 1)
        self.assertLess(
            composer_text.index("(g3c4-p2g2-native-terminal"),
            composer_text.index(OVERLAYS.COMPOSER_MARKER),
        )
        self.assertLess(
            composer_text.index(OVERLAYS.COMPOSER_MARKER),
            composer_text.index("(vector-set! output 1 'terminal-output)"),
        )

    def test_generated_overlays_remove_to_exact_originals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "overlays"
            result = subprocess.run(
                [sys.executable, "-B", str(GENERATOR), str(destination)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            identity = subprocess.run(
                ["sha256sum", "-c", "SHA256SUMS"], cwd=destination,
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(identity.returncode, 0, identity.stderr)
            self.assertEqual(len(identity.stdout.splitlines()), 9)
            t1_overlay = (destination / "t1-decode-overlay.esk").read_text()
            composer_overlay = (destination / "composer-postzero-overlay.esk").read_text()
            self.assertEqual(t1_overlay.replace(OVERLAYS.T1_INSERT, "", 1),
                             OVERLAYS.T1.read_text())
            self.assertEqual(composer_overlay.replace(
                OVERLAYS.COMPOSER_INSERT, "", 1), OVERLAYS.COMPOSER.read_text())
            self.assertIn(OVERLAYS.COMPOSER_MARKER + OVERLAYS.COMPOSER_INSERT,
                          composer_overlay)
            self.assertIn(str(destination / "composer-postzero-overlay.esk"),
                          (destination / "postzero.esk").read_text())

    def test_mutated_source_with_same_marker_is_rejected_before_output(self) -> None:
        for source_name in ("T1", "COMPOSER"):
            with self.subTest(source=source_name), tempfile.TemporaryDirectory() as tmp:
                original = getattr(OVERLAYS, source_name)
                changed = Path(tmp) / original.name
                changed.write_bytes(original.read_bytes() + b"\n; mutated fixture\n")
                destination = Path(tmp) / "overlays"
                setattr(OVERLAYS, source_name, changed)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(
                            ValueError, "immutable original source identity changed"
                        ):
                            OVERLAYS.main(destination)
                finally:
                    setattr(OVERLAYS, source_name, original)
                self.assertFalse(destination.exists())

    def test_missing_and_duplicate_markers_are_rejected(self) -> None:
        for source, marker, insertion, after in (
            (OVERLAYS.T1.read_text(), OVERLAYS.T1_MARKER,
             OVERLAYS.T1_INSERT, False),
            (OVERLAYS.COMPOSER.read_text(), OVERLAYS.COMPOSER_MARKER,
             OVERLAYS.COMPOSER_INSERT, True),
        ):
            with self.subTest(marker=marker[:24]):
                for changed in (source + marker,
                                source.replace(marker, "removed marker", 1)):
                    with self.assertRaisesRegex(
                        ValueError, "source marker must occur exactly once"
                    ):
                        OVERLAYS.insert_exact(changed, marker, insertion, after)

    def test_asan_ir_copy_is_reversible_and_definitions_are_counted(self) -> None:
        raw = (b'define i32 @load(ptr %p) {\n'
               b'entry:\n %v = load i32, ptr %p\n ret i32 %v\n}\n'
               b'define i32 @copy(ptr %p) #0 {\n ret i32 0\n}\n'
               b'attributes #0 = { nounwind }\n')
        derived, definitions = prepare(raw)
        self.assertEqual(len(definitions), 2)
        self.assertEqual(derived.count(b" sanitize_address {\n"), 2)
        self.assertEqual(derived.replace(b" sanitize_address {\n", b" {\n"), raw)
        self.assertEqual(definitions[0], sha(b'define i32 @load(ptr %p) {\n'))

    def test_asan_ir_unsupported_syntax_fails_closed(self) -> None:
        for raw in (
            b'declare i32 @load(ptr)\n',
            b'define i32 @load(ptr %p) sanitize_address {\n ret i32 0\n}\n',
            b'define i32 @load(ptr %p)\n{\n ret i32 0\n}\n',
            b'define i32 @load(ptr %p) { nounwind } {\n ret i32 0\n}\n',
            b'define i32 @load(ptr %p) {\r\n ret i32 0\r\n}\r\n',
            b' define i32 @load(ptr %p) {\n ret i32 0\n}\n',
            b'define i32 @load(ptr %p) #0 {\n ret i32 0\n}\n'
            b'attributes #0 = { sanitize_address }\n',
        ):
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                prepare(raw)

    def test_compiler_artifact_set_rejects_missing_extra_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "raw.ll").write_text("define x\n")
            (directory / "raw.d").write_text("target: source\n")
            compiler_evidence.require_files(directory, {"raw.ll", "raw.d"})
            for missing in ({"raw.ll"}, {"raw.ll", "raw.d", "extra"}):
                with self.assertRaises(ValueError):
                    compiler_evidence.require_files(directory, missing)
            (directory / "raw.d").unlink()
            (directory / "raw.d").symlink_to(directory / "raw.ll")
            with self.assertRaises(ValueError):
                compiler_evidence.require_files(directory, {"raw.ll", "raw.d"})

    def test_compiler_dep_target_and_source_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = directory / "target.o"
            source = directory / "fixture.esk"
            dep = directory / "fixture.d"
            dep.write_text(f"{target}: \\\n  {source}\n")
            compiler_evidence.dep_target(dep, target, source)
            for wrong_target, wrong_source in (
                (directory / "other.o", source),
                (target, directory / "other.esk"),
            ):
                with self.assertRaises(ValueError):
                    compiler_evidence.dep_target(dep, wrong_target, wrong_source)
            dep.write_text(f"{target}: {source}.extra\n")
            with self.assertRaises(ValueError):
                compiler_evidence.dep_target(dep, target, source)

    def test_compiler_source_identity_rejects_changed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source = directory / "fixture.esk"
            source.write_text("original")
            identity = f"{sha(source.read_bytes())}  {source}\n"
            (directory / "case-source.before").write_text(identity)
            (directory / "case-source.after").write_text(identity)
            self.assertEqual(compiler_evidence.source_identity(directory, "case"),
                             source)
            source.write_text("changed")
            with self.assertRaises(ValueError):
                compiler_evidence.source_identity(directory, "case")

    def test_compiler_recorded_exit_and_streams_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for suffix, value in (("command", "clang -c source\n"),
                                  ("stdout", ""), ("stderr", ""),
                                  ("exit", "0\n")):
                (directory / f"case.{suffix}").write_text(value)
            compiler_evidence.recorded(directory, "case", 0)
            with self.assertRaises(ValueError):
                compiler_evidence.recorded(directory, "case", 134)
            (directory / "case.stderr").unlink()
            with self.assertRaises(ValueError):
                compiler_evidence.recorded(directory, "case", 0)

    def test_asan_symbol_gate_rejects_module_initializer_only(self) -> None:
        self.assertIsNone(compiler_evidence.REPORT.search("U __asan_init\n"))
        self.assertIsNotNone(compiler_evidence.REPORT.search(
            "U __asan_report_load4\n"))


if __name__ == "__main__":
    unittest.main()
