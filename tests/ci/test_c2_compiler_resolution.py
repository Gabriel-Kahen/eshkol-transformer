from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON = PROJECT_ROOT / "scripts" / "common.sh"
LOCK = PROJECT_ROOT / "toolchain" / "eshkol.lock"


class C2CompilerResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.build = self.root / "build"
        self.bin.mkdir()
        self.build.mkdir()
        lock = dict(
            line.split("\t", 1)
            for line in LOCK.read_text(encoding="utf-8").splitlines()
            if "\t" in line
        )
        self.version = lock["clang_version"]
        self.cc = self._write_compiler("clang-21")
        self.cxx = self._write_compiler("clang++-21")
        (self.build / "eshkol-transformer-provenance.tsv").write_text(
            f"cc_path\t{self.cc}\n"
            f"cc_version\t{self.version}\n"
            f"cxx_path\t{self.cxx}\n"
            f"cxx_version\t{self.version}\n",
            encoding="utf-8",
        )

    def _write_compiler(self, name: str) -> Path:
        path = self.bin / name
        path.write_text(
            "#!/usr/bin/env bash\n"
            f"printf 'clang version {self.version}\\n'\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path.resolve()

    def _resolve(
        self, cc: str = "clang-21", cxx: str = "clang++-21"
    ) -> subprocess.CompletedProcess[str]:
        command = f"""
source {COMMON!s}
cc=
cxx=
resolve_provenance_compilers cc cxx "$CC" "$CXX"
printf '%s\\n%s\\n' "$cc" "$cxx"
"""
        environment = {
            **os.environ,
            "CC": cc,
            "CXX": cxx,
            "ESHKOL_BUILD_DIR": str(self.build),
            "PATH": f"{self.bin}:/usr/bin:/bin",
        }
        environment.pop("ESHKOL_ALLOW_UNSUPPORTED_HOST", None)
        return subprocess.run(
            ["/usr/bin/bash", "-c", command],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_resolves_versioned_path_only_compilers(self) -> None:
        result = self._resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, f"{self.cc}\n{self.cxx}\n")
        self.assertEqual(result.stderr, "")

    def test_accepts_absolute_provenance_compiler_paths(self) -> None:
        result = self._resolve(str(self.cc), str(self.cxx))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, f"{self.cc}\n{self.cxx}\n")

    def test_rejects_same_version_compiler_outside_provenance(self) -> None:
        other = self._write_compiler("other-clang-21")
        result = self._resolve(other.name)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match Eshkol provenance", result.stderr)


if __name__ == "__main__":
    unittest.main()
