from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q0 = ROOT / "tests" / "q0"


def development_python(path: Path) -> bool:
    # Accepted E3-D2 build-only source preparation, never a runtime dependency.
    return path.is_relative_to(ROOT / "tests") or path == (
        ROOT / "scripts" / "generate-e3-d2-source.py"
    )


class PythonIsolationTests(unittest.TestCase):
    def test_python_and_torch_are_test_only(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        python_files = [
            ROOT / item.decode("utf-8")
            for item in tracked.split(b"\0")
            if item
        ]
        for path in python_files:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(
                    development_python(path),
                    f"Python entered an unapproved path {path.relative_to(ROOT)}",
                )
        manifest = (ROOT / "eshkol.toml").read_text(encoding="utf-8")
        self.assertNotIn(".py", manifest)
        self.assertNotIn("torch", manifest.lower())
        self.assertTrue((Q0 / "generate_scalar_add.py").is_file())
        self.assertTrue((Q0 / "requirements-oracle.lock").is_file())

    def test_only_exact_build_generator_is_admitted_outside_tests(self) -> None:
        self.assertTrue(development_python(ROOT / "scripts/generate-e3-d2-source.py"))
        for relative in (
            "scripts/generate-other-source.py", "scripts/runtime.py",
            "scripts/generate-e3-d2-source.py.extra.py",
            "scripts/nested/generate-e3-d2-source.py",
            "src/generate-e3-d2-source.py", "lib/runtime.py",
            "internal/e3/runtime.py", "native/runtime.py",
        ):
            with self.subTest(path=relative):
                self.assertFalse(development_python(ROOT / relative))

    def test_fixture_reader_has_no_executable_deserialization(self) -> None:
        source = (Q0 / "oracle_format.py").read_text(encoding="utf-8").lower()
        for forbidden in ("pickle", "torch.load", "eval(", "exec(", "importlib"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
