from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts" / "check-d2-development-dependency.sh"


class DevelopmentDependencyGuardTests(unittest.TestCase):
    def run_checker(self, artifact: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(CHECKER), str(artifact)],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def compile(self, source: Path, output: Path, *arguments: str) -> None:
        compiler = os.environ.get("CC", "cc")
        subprocess.run(
            [compiler, *arguments, str(source), "-o", str(output)],
            check=True,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_incidental_raw_short_substrings_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="d2-dependency-clean-") as raw:
            artifact = Path(raw) / "incidental.bin"
            artifact.write_bytes(b"torchbearer\0!Py_noise\0py_identifier\0")
            result = self.run_checker(artifact)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")

    def test_bounded_raw_library_marker_reports_artifact_and_channel(self) -> None:
        with tempfile.TemporaryDirectory(prefix="d2-dependency-strings-") as raw:
            artifact = Path(raw) / "linked.bin"
            artifact.write_bytes(b"prefix\0/usr/lib/libpython3.11.so.1.0\0suffix")
            result = self.run_checker(artifact)
            self.assertEqual(result.returncode, 1)
            self.assertIn(f"artifact={artifact} channel=strings", result.stderr)
            self.assertIn("libpython3.11.so.1.0", result.stderr)

    def test_python_c_api_symbol_is_rejected_by_nm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="d2-dependency-nm-") as raw:
            directory = Path(raw)
            source = directory / "python_api.c"
            artifact = directory / "python_api.o"
            source.write_text(
                "extern void Py_Initialize(void);\n"
                "void (*dependency_probe)(void) = Py_Initialize;\n",
                encoding="utf-8",
            )
            self.compile(source, artifact, "-std=c11", "-c")
            result = self.run_checker(artifact)
            self.assertEqual(result.returncode, 1)
            self.assertIn(f"artifact={artifact} channel=nm", result.stderr)
            self.assertIn("Py_Initialize", result.stderr)

    def test_dynamic_python_named_dependency_is_rejected_by_ldd(self) -> None:
        with tempfile.TemporaryDirectory(prefix="d2-dependency-ldd-") as raw:
            directory = Path(raw)
            library_source = directory / "library.c"
            program_source = directory / "program.c"
            library = directory / "pythonproof.so"
            program = directory / "program"
            library_source.write_text(
                "int dependency_probe(void) { return 0; }\n", encoding="utf-8"
            )
            program_source.write_text(
                "extern int dependency_probe(void);\n"
                "int main(void) { return dependency_probe(); }\n",
                encoding="utf-8",
            )
            self.compile(library_source, library, "-std=c11", "-shared", "-fPIC")
            self.compile(
                program_source,
                program,
                "-std=c11",
                f"-L{directory}",
                "-Wl,-rpath,$ORIGIN",
                "-Wl,--no-as-needed",
                "-l:pythonproof.so",
            )
            result = self.run_checker(program)
            self.assertEqual(result.returncode, 1)
            self.assertIn(f"artifact={program} channel=ldd", result.stderr)
            self.assertIn("pythonproof.so", result.stderr)


if __name__ == "__main__":
    unittest.main()
