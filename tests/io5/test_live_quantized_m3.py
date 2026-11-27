"""Opt-in compiled M3 evaluation after mapped narrow-weight IO5 ingress."""

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/io5"))
import safetensors_interop as safe
from test_live_c1_interop import first_safetensors_payload, flip
from test_model_mapping import M3_ROWS, PROVIDER
from test_gguf_interop import unpack_export, wire as gguf_wire

TOOL = ROOT / "tools/io5/quantized_interchange.py"
BRIDGE_ENV = "IO5_LIVE_QUANTIZED_BRIDGE"
PROFILE = "eshkol-m3-v1"


def reference_narrow(dtype, f32):
    if dtype == "F16":
        return b"".join(struct.pack("<e", value) for value, in struct.iter_unpack("<f", f32))
    # Round to nearest BF16, using the retained low bit to break ties to even.
    return b"".join(struct.pack("<H", (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)
                    for bits, in struct.iter_unpack("<I", f32))


def reference_widen(dtype, narrow):
    if dtype == "F16":
        return b"".join(struct.pack("<f", value) for value, in struct.iter_unpack("<e", narrow))
    return b"".join(struct.pack("<I", bits << 16) for bits, in struct.iter_unpack("<H", narrow))


class LiveQuantizedM3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bridge = os.environ.get(BRIDGE_ENV)
        if not bridge:
            if __name__ == "__main__":
                raise RuntimeError(f"Set {BRIDGE_ENV} to the compiled io5-quantized-m3 binary")
            raise unittest.SkipTest("requires scripts/test-io5-live-quantized.sh")
        cls.bridge = Path(bridge).resolve()
        if not (cls.bridge.is_file() and os.access(cls.bridge, os.X_OK)):
            raise RuntimeError(f"Compiled M3 bridge is not executable: {cls.bridge}")

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.original = self.directory / "original.etcp"
        self.baseline = self.logits(self.command(self.bridge, "write", self.original).stdout)
        self.original_bytes = self.original.read_bytes()

    def command(self, *args, status=0):
        result = subprocess.run(list(map(str, args)), cwd=self.directory, text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, status, f"{args}\n{result.stdout}{result.stderr}")
        if status == 0:
            self.assertEqual(result.stderr, "")
        return result

    def cli(self, *args, status=0):
        return self.command(sys.executable, TOOL, *args, status=status)

    def logits(self, stdout):
        tag, *bits = stdout.split()
        self.assertEqual(tag, "M3-LOGITS")
        self.assertEqual(len(bits), 2048)
        raw = bytes(map(int, bits))
        self.assertTrue(all(math.isfinite(value) for value, in struct.iter_unpack("<f", raw)))
        return raw

    def export(self, dtype):
        wire = self.directory / f"{dtype}.safetensors"
        self.cli("c1-to-safetensors", self.original, wire,
                 "--dtype", dtype, "--profile", PROFILE, "--scheme", "huggingface")
        return wire

    def assert_m3_state(self, state):
        self.assertEqual(state.provider, PROVIDER)
        self.assertEqual([(t.path, t.shape, t.kind) for t in state.tensors],
                         [(tuple(path.split(".")), shape, "parameter")
                          for path, _, _, shape in M3_ROWS])
        self.assertEqual(state.aliases, ((10, 14),))
        self.assertEqual(state.tensors[10].data, state.tensors[14].data)

    def evaluate(self, path, readback):
        logits = self.logits(self.command(self.bridge, "evaluate", path, readback).stdout)
        self.assertEqual(readback.read_bytes(), path.read_bytes())
        return logits

    def test_mapped_safetensors_model_ingress(self):
        original = safe.decode_c1(self.original_bytes)
        self.assert_m3_state(original)
        self.assertEqual(self.evaluate(self.original, self.directory / "original-readback.etcp"),
                         self.baseline)
        source_digest = hashlib.sha256(self.original_bytes).hexdigest()
        reference_logits = struct.unpack("<512f", self.baseline)
        print(f"M3 F32 source C1 SHA-256={source_digest}", flush=True)
        for dtype, weight_bound, logit_bound in (("F16", 8.0e-6, 5.0e-5),
                                                ("BF16", 6.25e-5, 4.0e-4)):
            with self.subTest(dtype=dtype):
                wire = self.export(dtype)
                raw = wire.read_bytes()
                payload_offset = first_safetensors_payload(raw)
                header = json.loads(raw[8:payload_offset])
                metadata = header.pop("__metadata__")
                self.assertEqual(metadata["eshkol.io5.profile"], PROFILE)
                self.assertEqual(metadata["eshkol.io5.lossy"], "true")
                self.assertEqual(metadata["eshkol.io5.source_c1_sha256"], source_digest)
                self.assertEqual(set(header), {name for _, name, _, _ in M3_ROWS})
                payload_bytes = len(raw) - payload_offset
                self.assertEqual(payload_bytes * 2, sum(len(t.data) for t in original.tensors))

                restored = self.directory / f"{dtype}.etcp"
                self.cli("safetensors-to-c1", wire, restored)
                restored_bytes = restored.read_bytes()
                restored_state = safe.decode_c1(restored_bytes)
                self.assert_m3_state(restored_state)
                weight_delta = 0
                cursor = 0
                for (_, name, _, shape), source, decoded in zip(
                    M3_ROWS, original.tensors, restored_state.tensors, strict=True,
                ):
                    end = cursor + len(source.data) // 2
                    self.assertEqual(header[name], {"dtype": dtype, "shape": list(shape),
                                                    "data_offsets": [cursor, end]}, name)
                    payload = raw[payload_offset + cursor:payload_offset + end]
                    self.assertEqual(payload, reference_narrow(dtype, source.data), name)
                    self.assertEqual(decoded.data, reference_widen(dtype, payload), name)
                    source_values = struct.iter_unpack("<f", source.data)
                    decoded_values = struct.iter_unpack("<f", decoded.data)
                    weight_delta = max(weight_delta, max(
                        abs(a - b) for (a,), (b,) in zip(source_values, decoded_values, strict=True)))
                    cursor = end
                self.assertEqual(cursor, payload_bytes)
                observed = self.evaluate(restored, self.directory / f"{dtype}-readback.etcp")
                observed_logits = struct.unpack("<512f", observed)
                logit_delta = max(abs(a - b) for a, b in zip(reference_logits, observed_logits, strict=True))
                self.assertGreater(weight_delta, 0)
                self.assertLessEqual(weight_delta, weight_bound)
                self.assertGreater(logit_delta, 0)
                self.assertLessEqual(logit_delta, logit_bound)
                print(f"M3 {dtype}: {payload_bytes} wire payload bytes, max |Δweight|={weight_delta:.9g}, "
                      f"max |Δlogit|={logit_delta:.9g}, restored C1 SHA-256="
                      f"{hashlib.sha256(restored_bytes).hexdigest()}", flush=True)

    def export_gguf(self):
        wire = self.directory / "F16.gguf"
        self.cli("c1-to-gguf", self.original, wire, "--dtype", "F16",
                 "--profile", PROFILE, "--scheme", "gguf")
        return wire

    def test_mapped_gguf_f16_model_ingress(self):
        original = safe.decode_c1(self.original_bytes)
        self.assert_m3_state(original)
        wire = self.export_gguf()
        raw = wire.read_bytes()
        metadata, tensors, _, _ = unpack_export(raw, element_size=2)
        fields = {name: (kind, value) for name, kind, value in metadata}
        self.assertEqual(len(fields), len(metadata))
        self.assertEqual(fields["general.architecture"], (8, "eshkol"))
        self.assertEqual(fields["general.alignment"], (4, 32))
        self.assertEqual(fields["general.file_type"], (4, 1))
        self.assertEqual(fields["eshkol.io5.profile"], (8, PROFILE))
        self.assertEqual(fields["eshkol.io5.scheme"], (8, "gguf"))
        self.assertEqual(fields["eshkol.io5.storage_dtype"], (8, "F16"))
        self.assertEqual(fields["eshkol.io5.lossy"], (8, "true"))
        self.assertEqual(fields["eshkol.io5.source_c1_sha256"],
                         (8, hashlib.sha256(self.original_bytes).hexdigest()))

        expected_tensors, expected_state = [], []
        cursor = 0
        weight_delta = 0
        for (_, _, name, shape), source in zip(M3_ROWS, original.tensors, strict=True):
            count = len(source.data) // 4
            values = struct.unpack(f"<{count}f", source.data)
            payload = struct.pack(f"<{count}e", *values)
            widened = struct.unpack(f"<{count}e", payload)
            expected_tensors.append((name, shape[::-1], 1, cursor, payload))
            expected_state.append(replace(source, data=struct.pack(f"<{count}f", *widened)))
            cursor += (len(payload) + 31) // 32 * 32
            weight_delta = max(weight_delta, max(abs(a - b) for a, b in zip(values, widened, strict=True)))
        self.assertEqual(tensors, expected_tensors)
        self.assertEqual(raw, gguf_wire(metadata, expected_tensors))
        self.assertEqual(fields["eshkol.io5.tensor_dtypes"][0], 8)
        self.assertEqual(json.loads(fields["eshkol.io5.tensor_dtypes"][1]),
                         {name: "F16" for _, _, name, _ in M3_ROWS})

        restored = self.directory / "gguf-f16.etcp"
        self.cli("gguf-to-c1", wire, restored)
        restored_bytes = restored.read_bytes()
        self.assert_m3_state(safe.decode_c1(restored_bytes))
        self.assertEqual(restored_bytes, safe.encode_c1(replace(original, tensors=tuple(expected_state))))
        observed = self.evaluate(restored, self.directory / "gguf-f16-readback.etcp")
        logit_delta = max(abs(a - b) for a, b in zip(struct.unpack("<512f", self.baseline),
                                                   struct.unpack("<512f", observed), strict=True))
        self.assertGreater(weight_delta, 0)
        self.assertLessEqual(weight_delta, 8.0e-6)
        self.assertGreater(logit_delta, 0)
        self.assertLessEqual(logit_delta, 5.0e-5)
        payload_bytes = sum(len(tensor[-1]) for tensor in tensors)
        self.assertEqual(payload_bytes * 2, sum(len(t.data) for t in original.tensors))
        print(f"M3 GGUF F16: {len(raw)} file bytes, {payload_bytes} tensor bytes, "
              f"max |Δweight|={weight_delta:.9g}, max |Δlogit|={logit_delta:.9g}, restored C1 SHA-256="
              f"{hashlib.sha256(restored_bytes).hexdigest()}", flush=True)

    def test_gguf_wire_corruption_does_not_publish_c1(self):
        raw = self.export_gguf().read_bytes()
        _, tensors, _, start = unpack_export(raw, element_size=2)
        padding_at = next(start + offset + len(data) for _, _, _, offset, data in tensors if len(data) % 32)
        self.assertEqual(raw[padding_at], 0)
        for label, at, diagnostic in (("tensor-padding", padding_at, "GGUF nonzero tensor padding"),
                                      ("payload", start, "payload SHA-256 mismatch")):
            with self.subTest(case=label):
                wire = self.directory / f"{label}.gguf"
                wire.write_bytes(flip(raw, at))
                destination = self.directory / f"{label}-rejected.etcp"
                result = self.cli("gguf-to-c1", wire, destination, status=2)
                self.assertIn(diagnostic, result.stderr)
                self.assertFalse(destination.exists())
                destination.write_bytes(self.original_bytes)
                result = self.cli("gguf-to-c1", wire, destination, "--overwrite", status=2)
                self.assertIn(diagnostic, result.stderr)
                self.assertEqual(destination.read_bytes(), self.original_bytes)

    def test_wire_corruption_does_not_publish_c1(self):
        for dtype in ("F16", "BF16"):
            with self.subTest(dtype=dtype):
                wire = self.export(dtype)
                raw = wire.read_bytes()
                wire.write_bytes(flip(raw, first_safetensors_payload(raw)))
                destination = self.directory / f"{dtype}-rejected.etcp"
                result = self.cli("safetensors-to-c1", wire, destination, status=2)
                self.assertIn("payload SHA-256 mismatch", result.stderr)
                self.assertFalse(destination.exists())
                destination.write_bytes(self.original_bytes)
                result = self.cli("safetensors-to-c1", wire, destination, "--overwrite", status=2)
                self.assertIn("payload SHA-256 mismatch", result.stderr)
                self.assertEqual(destination.read_bytes(), self.original_bytes)

    def test_live_c1_rejections_do_not_publish_readback_or_logits(self):
        state = safe.decode_c1(self.original_bytes)
        payload_offset = struct.unpack_from("<Q", self.original_bytes, 64)[0]
        # Same element count and valid C1 checksums exercise the module's shape guard.
        wrong_shape = replace(state.tensors[0], shape=(2, 8))
        for label, raw, diagnostic in (
            ("payload", flip(self.original_bytes, payload_offset), "checksum"),
            ("missing-tie", safe.encode_c1(replace(state, aliases=())),
             "state alias graph does not match destination module"),
            ("shape", safe.encode_c1(replace(state, tensors=(wrong_shape, *state.tensors[1:]))),
             "state tensor shape mismatch"),
        ):
            with self.subTest(case=label):
                path = self.directory / f"{label}.etcp"
                path.write_bytes(raw)
                readback = self.directory / f"{label}-rejected.etcp"
                result = self.command(self.bridge, "evaluate", path, readback, status=2)
                self.assertIn("IO5 QUANTIZED M3 ERROR", result.stdout)
                self.assertIn(diagnostic, result.stdout)
                self.assertNotIn("M3-LOGITS", result.stdout)
                self.assertFalse(readback.exists())


if __name__ == "__main__":
    unittest.main()
