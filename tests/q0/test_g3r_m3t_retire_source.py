#!/usr/bin/env python3
"""Lightweight source gate; the supported native runtime gate remains separate."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASE = "ca2a62b35789850c55f4adf87c8a240215f2d275"
SOURCES = ("src/eshkol_transformer/m3t_transport.c", "native/f32_tensor.c")
HEADERS = ("src/eshkol_transformer/m3t_transport.h", "native/f32_parameter_internal.h")


def run(*argv: str) -> bytes:
    return subprocess.check_output(argv, cwd=ROOT, stderr=subprocess.PIPE)


def preprocess(base: Path, source: str) -> bytes:
    return run("cc", "-E", "-P", "-std=c11", "-I", str(base / "include"),
               "-I", str(base / "native"), str(base / source))


class NativeRetireSource(unittest.TestCase):
    def test_feature_off_preprocessing_matches_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name in SOURCES + HEADERS:
                target = base / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(run("git", "show", f"{BASE}:{name}"))
            (base / "include").symlink_to(ROOT / "include")
            for name in ("src/eshkol_transformer/m3t_f32_scoped.h",
                         "native/kernel_abi.h", "native/t1_i64_shell.h"):
                (base / name).symlink_to(ROOT / name)
            for name in SOURCES:
                with self.subTest(name=name):
                    self.assertEqual(preprocess(base, name), preprocess(ROOT, name))

    def test_feature_on_source_is_syntactically_valid(self) -> None:
        flags = ("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic",
                 "-Wno-misleading-indentation", "-DET_G3R_CANDIDATE_RETIRE_PRIVATE",
                 "-DET_F32_TENSOR_TESTING", "-DET_I64_TENSOR_TESTING",
                 "-I", str(ROOT / "include"), "-I", str(ROOT / "native"), "-fsyntax-only")
        for source in ("tests/m3t/test_candidate_retire.c",
                       "src/eshkol_transformer/m3t_f32_integration.c"):
            with self.subTest(source=source):
                run(*flags, str(ROOT / source))

    def test_private_api_is_not_in_installed_header(self) -> None:
        installed = (ROOT / "include/eshkol_transformer/f32_tensor.h").read_text()
        self.assertNotIn("et_f32_parameter_private_destroy_ready_v1", installed)
        self.assertNotIn("et_m3t_private_candidate_retire_commit_v1", installed)


if __name__ == "__main__":
    unittest.main()
