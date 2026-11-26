#!/usr/bin/env python3
"""Parity and invalid-boundary checks for the generated private SHA entry."""

import ctypes
import hashlib
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "tests/c2/c2_native_sha_feasibility.py"


class NativeShaFeasibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix="c2-native-sha-")
        cls.directory = Path(cls.work.name)
        source = cls.directory / "format.c"
        library = cls.directory / "format.so"
        subprocess.run([sys.executable, str(GENERATOR), "native", str(source)], check=True)
        subprocess.run([
            os.environ.get("CC", "clang"), "-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
            "-I", str(ROOT / "native"), "-I", str(ROOT / "include"),
            str(source), str(ROOT / "native/c2_x1_canonical.c"), "-o", str(library),
        ], check=True)
        cls.function = ctypes.CDLL(str(library)).et_c2_development_sha256_whole_v1
        cls.function.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p)
        cls.function.restype = ctypes.c_int64

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def digest(self, data):
        source = ctypes.create_string_buffer(struct.pack("=q", len(data)) + data)
        result = ctypes.create_string_buffer(struct.pack("=q", 32) + b"\0" * 32)
        self.assertEqual(self.function(source, len(data), result), 0)
        return result.raw[8:40]

    def test_known_vectors_and_padding_edges(self):
        for data in (b"", b"abc", b"a" * 1000000):
            with self.subTest(length=len(data)):
                self.assertEqual(self.digest(data), hashlib.sha256(data).digest())
        for length in (1, 3, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129):
            data = bytes((17 * index + 91) % 256 for index in range(length))
            with self.subTest(length=length):
                self.assertEqual(self.digest(data), hashlib.sha256(data).digest())

    def test_exact_joint_fixture_bytes(self):
        fixtures = self.directory / "fixtures"
        subprocess.run([sys.executable, str(ROOT / "tests/c2/prepare_checkpoint_operational_fixtures.py"),
                        str(fixtures)], check=True)
        data = (fixtures / "joint-exact.c2").read_bytes()
        self.assertEqual(len(data), 16777216)
        self.assertEqual(self.digest(data), hashlib.sha256(data).digest())

    def test_invalid_boundaries_preserve_output(self):
        source = ctypes.create_string_buffer(struct.pack("=q", 1) + b"x")
        wrong_source = ctypes.create_string_buffer(struct.pack("=q", 2) + b"x")
        result = ctypes.create_string_buffer(struct.pack("=q", 32) + b"!" * 32)
        wrong_result = ctypes.create_string_buffer(struct.pack("=q", 31) + b"!" * 32)
        for pointer, length, output in (
            (None, 1, result), (source, -1, result),
            (source, 2305843009213693952, result), (source, 1, None),
            (wrong_source, 1, result), (source, 1, wrong_result),
        ):
            with self.subTest(length=length, null_input=pointer is None, null_output=output is None):
                self.assertEqual(self.function(pointer, length, output), -1)
                self.assertEqual(result.raw[8:40], b"!" * 32)
                self.assertEqual(wrong_result.raw[8:40], b"!" * 32)
        empty = ctypes.create_string_buffer(struct.pack("=q", 0))
        self.assertEqual(self.function(empty, 0, result), 0)
        self.assertEqual(result.raw[8:40], hashlib.sha256(b"").digest())

    def test_generated_c1_overlay_is_private_and_single_substitution(self):
        runtime = self.directory / "joint.esk"
        subprocess.run([sys.executable, str(ROOT / "tests/c2/c2_joint_seam_timing.py"),
                        "generate", str(runtime)], check=True)
        overlay = self.directory / "overlay/transformer/checkpoint_internal.esk"
        original = (ROOT / "internal/c1/lib/transformer/checkpoint_internal.esk").read_bytes()
        subprocess.run([sys.executable, str(GENERATOR), "overlay", str(overlay)], check=True)
        modified = overlay.read_text()
        self.assertEqual(modified.count("(digest (c2-development-native-sha256 input))"), 1)
        self.assertEqual(modified.count("(digest (c1-sha256 input))"), 0)
        self.assertEqual(modified.count("(c2-development-sha256-whole\n"), 1)
        self.assertEqual((ROOT / "internal/c1/lib/transformer/checkpoint_internal.esk").read_bytes(),
                         original)


if __name__ == "__main__":
    unittest.main()
