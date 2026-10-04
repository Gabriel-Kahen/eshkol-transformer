"""Source identity and fault-site checks for the disposable terminal overlays."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
