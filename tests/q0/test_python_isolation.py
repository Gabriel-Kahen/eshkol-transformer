from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q0 = ROOT / "tests" / "q0"


class PythonIsolationTests(unittest.TestCase):
    def test_python_and_torch_are_test_only(self) -> None:
        python_files = [
            path
            for path in ROOT.rglob("*.py")
            if ".git" not in path.parts and ".tmp" not in path.parts
        ]
        for path in python_files:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(
                    path.is_relative_to(ROOT / "tests"),
                    f"Python entered non-test path {path.relative_to(ROOT)}",
                )
        manifest = (ROOT / "eshkol.toml").read_text(encoding="utf-8")
        self.assertNotIn(".py", manifest)
        self.assertNotIn("torch", manifest.lower())
        self.assertTrue((Q0 / "generate_scalar_add.py").is_file())
        self.assertTrue((Q0 / "requirements-oracle.lock").is_file())

    def test_fixture_reader_has_no_executable_deserialization(self) -> None:
        source = (Q0 / "oracle_format.py").read_text(encoding="utf-8").lower()
        for forbidden in ("pickle", "torch.load", "eval(", "exec(", "importlib"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
