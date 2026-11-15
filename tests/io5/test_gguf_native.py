"""Optional local witness; never downloads or builds the native GGML dependency."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/io5"))
import gguf_interop as gg
import safetensors_interop as c1


@unittest.skipUnless(os.environ.get("IO5_GGUF_NATIVE_PREFIX"),
                     "local native GGML prefix not configured; native execution unproved here")
class NativeGGUFTests(unittest.TestCase):
    def test_existing_native_reader_and_writer(self):
        prefix = Path(os.environ["IO5_GGUF_NATIVE_PREFIX"]).resolve(strict=True)
        headers = [prefix / "include" / name for name in ("gguf.h", "ggml.h")]
        library = (prefix / "lib/libggml-base.so").resolve(strict=True)
        for path in headers + [library]:
            self.assertTrue(path.is_file(), str(path))
            print(f"IO5_NATIVE_ID {path.name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
        compiler = shutil.which(os.environ.get("IO5_GGUF_NATIVE_CC", "cc"))
        self.assertIsNotNone(compiler, "configured native compiler unavailable")
        environment = dict(os.environ, LD_LIBRARY_PATH=str(prefix / "lib"))
        environment.pop("LD_PRELOAD", None)
        environment.pop("LD_AUDIT", None)
        version = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                                 check=True, timeout=10).stdout.splitlines()[0]
        print(f"IO5_NATIVE_COMPILER {version}")
        scratch = ROOT / ".tmp/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="io5-native-", dir=scratch) as directory:
            work = Path(directory)
            executable = work / "gguf-native-roundtrip"
            subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                            "-I", str(prefix / "include"),
                            str(ROOT / "tests/io5/gguf_native_roundtrip.c"),
                            "-L", str(prefix / "lib"), "-Wl,-rpath," + str(prefix / "lib"),
                            "-lggml-base", "-o", str(executable)],
                           check=True, capture_output=True, env=environment, timeout=30)
            linked = subprocess.run(["ldd", str(executable)], check=True, capture_output=True,
                                    text=True, env=environment, timeout=10).stdout
            self.assertTrue(any("libggml-base" in line and str(prefix / "lib") in line
                                for line in linked.splitlines()), linked)
            bits = (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001,
                    0x7F800001, 1, 0x007FFFFF, 0x00800000, 0x3F800000,
                    0xFFFFFFFF, 0x7F7FFFFF)
            shapes = ((12,), (3, 4), (2, 3, 4), (2, 3, 4, 5), (3, 4))
            tensors = []
            for index, shape in enumerate(shapes):
                count = c1.elements(shape)
                data = b"".join(struct.pack("<I", bits[i % len(bits)]) for i in range(count))
                tensors.append(c1.Tensor((f"path.{index}", "λ"),
                                         "buffer" if index == 2 else "parameter", shape, data))
            original = c1.encode_c1(c1.State("native-witness", tuple(tensors), ((1, 4),)))
            source, repacked, observations = (work / name for name in
                                               ("source.gguf", "repacked.gguf", "observations.bin"))
            source.write_bytes(gg.encode_gguf(original))
            native = subprocess.run([str(executable), str(source), str(repacked), str(observations)],
                                    capture_output=True, env=environment, timeout=30)
            self.assertEqual(native.returncode, 0, native.stderr.decode(errors="replace"))
            raw = observations.read_bytes()
            self.assertEqual(raw[:8], b"IO5GGUF1")
            at = 8

            def number():
                nonlocal at
                value = struct.unpack_from("<Q", raw, at)[0]
                at += 8
                return value

            self.assertEqual(number(), len(tensors))
            for index, tensor in enumerate(tensors):
                length = number()
                self.assertEqual(raw[at:at + length], f"t{index:06d}".encode())
                at += length
                self.assertEqual(tuple(number() for _ in range(4)),
                                 tuple(reversed(tensor.shape)) + (1,) * (4 - len(tensor.shape)))
                length = number()
                self.assertEqual(length, len(tensor.data))
                self.assertEqual(raw[at:at + length], tensor.data)
                at += length
            self.assertEqual(at, len(raw))
            self.assertEqual(gg.decode_gguf(repacked.read_bytes()), original)
            self.assertEqual(c1.decode_c1(original).aliases, ((1, 4),))


if __name__ == "__main__":
    unittest.main()
