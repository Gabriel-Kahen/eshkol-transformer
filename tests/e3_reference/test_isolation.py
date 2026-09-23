from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DevelopmentIsolationTests(unittest.TestCase):
    def test_no_runtime_or_ci_registration(self) -> None:
        for relative in (
            "Makefile", "scripts/ci-build-prerequisites.sh", ".github/workflows/ci.yml",
            "tests/ci/topology.py",
        ):
            self.assertNotIn("e3_reference", (ROOT / relative).read_text())

    def test_directory_contains_source_and_documentation_only(self) -> None:
        allowed = {".md", ".py"}
        files = [path for path in Path(__file__).parent.iterdir() if path.is_file()]
        self.assertTrue(files)
        self.assertTrue(all(path.suffix in allowed for path in files))


if __name__ == "__main__":
    unittest.main()
