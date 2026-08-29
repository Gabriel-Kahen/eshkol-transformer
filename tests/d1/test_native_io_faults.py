from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class NativeIoFaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        normal = os.environ.get("D1_CORPUS_TOOL")
        faults = os.environ.get("D1_FAULT_CORPUS_TOOL")
        if not normal or not faults:
            raise RuntimeError("D1 corpus tools must be provided by the gate")
        cls.normal_tool = Path(normal).resolve()
        cls.fault_tool = Path(faults).resolve()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="d1-native-io-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_writer(
        self,
        tool: Path,
        directory: Path,
        *,
        fault: str | None = None,
        fail_call: int = 1,
        expected: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        environment = {"PATH": "/definitely-not-a-real-path"}
        if fault is not None:
            environment["ET_D1_TEST_FAULT"] = fault
            environment["ET_D1_TEST_FAIL_CALL"] = str(fail_call)
        completed = subprocess.run(
            [str(tool), "write", str(directory), "standard"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env=environment,
        )
        self.assertEqual(completed.returncode, expected,
                         completed.stdout + completed.stderr)
        return completed

    @staticmethod
    def corpus_bytes(directory: Path) -> dict[str, bytes]:
        return {
            path.name: path.read_bytes()
            for path in sorted(directory.iterdir())
            if path.is_file()
        }

    def test_partial_writes_complete_byte_identically(self) -> None:
        expected = self.root / "expected"
        expected.mkdir()
        self.run_writer(self.normal_tool, expected)
        expected_bytes = self.corpus_bytes(expected)

        for invocation in (1, 4):
            with self.subTest(invocation=invocation):
                actual = self.root / f"short-{invocation}"
                actual.mkdir()
                self.run_writer(
                    self.fault_tool,
                    actual,
                    fault="short-write",
                    fail_call=invocation,
                )
                self.assertEqual(self.corpus_bytes(actual), expected_bytes)

    def test_write_and_close_failures_cleanup_without_manifest(self) -> None:
        for fault in ("write-enospc", "write-eio", "close-eio"):
            for invocation in (1, 4):
                with self.subTest(fault=fault, invocation=invocation):
                    directory = self.root / f"{fault}-{invocation}"
                    directory.mkdir()
                    unrelated = directory / "unrelated.bin"
                    unrelated.write_bytes(b"preserve")
                    result = self.run_writer(
                        self.fault_tool,
                        directory,
                        fault=fault,
                        fail_call=invocation,
                        expected=2,
                    )
                    self.assertIn("ERROR category=io", result.stdout)
                    self.assertIn("operation=token-corpus-write!", result.stdout)
                    self.assertIn("source-domain=d1-native-io", result.stdout)
                    self.assertIn("cause=#f", result.stdout)
                    self.assertFalse((directory / "manifest.etm").exists())
                    self.assertFalse((directory / "manifest.etm.tmp").exists())
                    self.assertFalse((directory / ".d1-writer-lock").exists())
                    self.assertEqual(
                        sorted(path.name for path in directory.iterdir()),
                        ["unrelated.bin"],
                    )
                    self.assertEqual(unrelated.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
