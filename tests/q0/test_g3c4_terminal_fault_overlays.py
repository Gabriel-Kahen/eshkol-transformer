"""Source identity and fault-site checks for the disposable terminal overlays."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.g3c4 import check_p2_g2_terminal_compiler_evidence as compiler_evidence
from tests.shared.prepare_eshkol_asan_ir import main as prepare_file
from tests.shared.prepare_eshkol_asan_ir import prepare, scan_pair, sha


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tests/g3c4/generate_p2_g2_terminal_fault_overlays.py"
SPEC = importlib.util.spec_from_file_location("terminal_overlays", GENERATOR)
OVERLAYS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OVERLAYS)


class TerminalOverlayTests(unittest.TestCase):
    def test_compiler_verifier_direct_invocation_is_cwd_independent(self) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-I", "-B",
                 str(ROOT / "tests/g3c4/check_p2_g2_terminal_compiler_evidence.py")],
                cwd=tmp, env=environment, capture_output=True, text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.startswith("usage: "), result.stderr)
        self.assertNotIn("Traceback", result.stderr)

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
            b'define i32 @good(ptr %p) {\n ret i32 0\n}\n'
            b'define\ti32 @unmarked(ptr %p) {\n %x = load i32, ptr %p\n ret i32 %x\n}\n',
            b'define i32 @load(ptr %p) #0 {\n ret i32 0\n}\n'
            b'attributes #0 = { sanitize_address }\n',
        ):
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                prepare(raw)

    def test_asan_file_streaming_and_unsafe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            raw = directory / "raw.ll"
            derived = directory / "derived.ll"
            receipt = directory / "receipt.json"
            raw.write_bytes(b'define i32 @good(ptr %p) {\n ret i32 0\n}\n' * 4096)
            prepare_file(raw, derived, receipt)
            self.assertEqual(scan_pair(raw, derived),
                             json.loads(receipt.read_text()))
            with self.assertRaises(ValueError):
                prepare_file(raw, derived, directory / "other.json")
            derived.write_bytes(derived.read_bytes().replace(b"@good", b"@evil", 1))
            with self.assertRaisesRegex(ValueError, "derived IR differs"):
                scan_pair(raw, derived)
            derived.unlink()
            derived.symlink_to(directory / "absent.ll")
            with self.assertRaisesRegex(ValueError, "output already exists"):
                prepare_file(raw, derived, directory / "other.json")
            derived.unlink()
            receipt.unlink()
            receipt.symlink_to(directory / "absent.json")
            with self.assertRaisesRegex(ValueError, "output already exists"):
                prepare_file(raw, derived, receipt)
            receipt.unlink()
            raw.unlink()
            raw.symlink_to(directory / "absent.ll")
            with self.assertRaises((ValueError, FileNotFoundError)):
                prepare_file(raw, derived, receipt)

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
            external = directory / "external"
            external.mkdir()
            (external / "raw.ll").write_text("define x\n")
            (external / "raw.d").write_text("target: source\n")
            linked = directory / "linked"
            linked.symlink_to(external, target_is_directory=True)
            with self.assertRaises(ValueError):
                compiler_evidence.require_files(linked, {"raw.ll", "raw.d"})

    def test_compiler_dep_target_and_source_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = directory / "target.o"
            source = directory / "fixture.esk"
            source.write_text("source\n")
            dep = directory / "fixture.d"
            dep.write_text(f"{target}: \\\n  {source}\n")
            compiler_evidence.dep_target(dep, target, {source})
            for wrong_target, wrong_source in (
                (directory / "other.o", source),
                (target, directory / "other.esk"),
            ):
                with self.assertRaises(ValueError):
                    compiler_evidence.dep_target(dep, wrong_target, {wrong_source})
            dep.write_text(f"{target}: {source}.extra\n")
            with self.assertRaises(ValueError):
                compiler_evidence.dep_target(dep, target, {source})
            extra = directory / "extra.esk"
            extra.write_text("extra\n")
            dep.write_text(f"{target}: {source} \\\n  {extra}\n")
            compiler_evidence.dep_target(dep, target, {source, extra})
            for bad in (
                f"{target}: /tmp/unreviewed.esk\nphony: {source}\n",
                f"{target}: {source} {extra}\n",
                f"{target}:\n",
                f"{target}: {source} {source}\n",
                f"{target}: {source}\\x\n",
                f"{target}: {source}\\\n  {extra}\n",
                f"{target}: {str(source)[:-4]}\\\n  .esk\n",
                f"{str(target)[:-2]}\\\n  .o: {source}\n",
            ):
                dep.write_text(bad)
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    compiler_evidence.dep_target(dep, target, {source})
            spaced = directory / "space #$.esk"
            spaced.write_text("source\n")
            dep.write_text(f"{target}: {str(spaced).replace(' ', chr(92)+' ').replace('#', chr(92)+'#').replace('$', '$$')}\n")
            compiler_evidence.dep_target(dep, target, {spaced})
            dep.write_text(f"{target}: {spaced}\n")
            with self.assertRaises(ValueError):
                compiler_evidence.dep_target(dep, target, {spaced})
            linked = directory / "linked.esk"
            linked.symlink_to(source)
            dep.write_text(f"{target}: {linked}\n")
            with self.assertRaisesRegex(ValueError, "symlink dependency"):
                compiler_evidence.dep_target(dep, target, {source})
            parent_link = directory / "alias"
            parent_link.symlink_to(directory, target_is_directory=True)
            dep.write_text(f"{target}: {parent_link / source.name}\n")
            with self.assertRaisesRegex(ValueError, "symlink dependency"):
                compiler_evidence.dep_target(dep, target, {source})

    def test_frozen_eshkol_dependency_inventory_covers_each_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mode = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                OVERLAYS.main(mode / "overlays")
            for fixture in compiler_evidence.FIXTURES:
                with self.subTest(fixture=fixture):
                    expected = compiler_evidence.closed_dependencies(
                        mode, "eshkol", fixture)
                    source = {
                        "terminal": ROOT / "tests/g3c4/p2_g2_terminal_composer_test.esk",
                        "feature-off": ROOT / "tests/g3c4/p2_g2_terminal_composer_feature_off_test.esk",
                    }.get(fixture, mode / "overlays" / f"{fixture}.esk")
                    self.assertIn(source.resolve(), expected)
                    self.assertGreater(len(expected), 15)

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
