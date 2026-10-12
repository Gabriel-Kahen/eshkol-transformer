"""The installed CLI3 app may import D1 validation, never D1 write or D2."""

from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tests" / "cli3" / "stage-cli3-d1-readonly.py"
SOURCE = ROOT / "lib" / "transformer" / "data.esk"
EXPORTS = {
    name for name in (ROOT / "native" / "d1_e1b_public_exports.txt").read_text().splitlines()
    if name != "et_e1b_public_d1_token_corpus_write_v1"
}


class ReadonlyFacadeTests(unittest.TestCase):
    def stage(self, source: str) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.esk"
            output_path = Path(directory) / "data.esk"
            source_path.write_text(source)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(source_path), str(output_path)],
                text=True, capture_output=True, check=False,
            )
            output = output_path.read_text() if output_path.exists() else ""
            return result, output

    def test_exact_public_readonly_projection(self) -> None:
        result, output = self.stage(SOURCE.read_text())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("(require transformer.error_consumer)", output)
        self.assertEqual(set(re.findall(r":real (et_e1b_public_d1_[a-z0-9_]+)", output)),
                         EXPORTS)
        self.assertEqual(len(EXPORTS), 7)
        for forbidden in ("token-corpus-write!", "et_e1b_public_d2_",
                          "token-dataset-open", "token-batch-validate"):
            self.assertNotIn(forbidden, output)

    def test_changed_marker_or_writer_boundary_fails_closed(self) -> None:
        source = SOURCE.read_text()
        for changed in (
            source.replace(";;; D1 facade component begin.", "", 1),
            source.replace("(define e1b-safe-d1-token-corpus-write-call\n",
                           "(define e1b-safe-d1-writer-call\n", 1),
            source.replace("(define (token-corpus-validate ",
                           "(define (different-validate ", 1),
            source.replace(":real et_e1b_public_d1_token_corpus_validate_v1",
                           ":real et_e1b_public_d2_token_dataset_open_v1", 1),
        ):
            with self.subTest(changed=changed[:100]):
                result, output = self.stage(changed)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
