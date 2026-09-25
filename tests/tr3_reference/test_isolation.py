from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TR3ReferenceIsolationTests(unittest.TestCase):
    def test_reference_is_not_registered_in_production_or_ci(self) -> None:
        for relative in ("Makefile", "tests/ci/topology.py", "tests/ci/test_topology.py"):
            self.assertNotIn("tr3_reference", (ROOT / relative).read_text())
        for directory in ("src", "native"):
            if (ROOT / directory).exists():
                for path in (ROOT / directory).rglob("*"):
                    if path.is_file():
                        self.assertNotIn(b"tr3_reference", path.read_bytes())

    def test_no_generated_bulk_artifact_is_checked_in(self) -> None:
        forbidden = {"corpora.json", "summary.json", "trajectory.json", "tokenizer.etok"}
        present = {path.name for path in (ROOT / "tests/tr3_reference").iterdir()}
        self.assertFalse(forbidden & present)


if __name__ == "__main__":
    unittest.main()
