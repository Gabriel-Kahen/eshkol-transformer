"""Exercise explicitly mapped external narrow weights in the compiled M3 gate."""

import hashlib
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/c1"))
from test_checkpoint_format import build_container, parse_golden
from test_model_mapping import M3_ROWS, PROVIDER
from test_quantized_interchange import build_raw_gguf
from test_safetensors_interop import safe as safetensors_wire

TOOL = ROOT / "tools/io5/quantized_interchange.py"
PROFILE = "eshkol-m3-v1"


def narrow(source, dtype):
    """Return independently rounded wire and widened F32 bytes."""
    count = len(source) // 4
    if dtype == "F16":
        wire = struct.pack(f"<{count}e", *struct.unpack(f"<{count}f", source))
        return wire, struct.pack(f"<{count}f", *struct.unpack(f"<{count}e", wire))
    # Add half an ULP minus one, then the retained low bit for ties to even.
    bits = [(value + 0x7FFF + ((value >> 16) & 1)) >> 16
            for value, in struct.iter_unpack("<I", source)]
    return struct.pack(f"<{count}H", *bits), struct.pack(f"<{count}I", *(value << 16 for value in bits))


def external_gguf(rows):
    descriptors = []
    offset = 0
    for name, shape, payload in rows:
        descriptors.append((name, shape[::-1], 1, offset, payload))  # GGML F16 type
        offset += len(payload) + (-len(payload) % 32)
    return build_raw_gguf([
        ("general.architecture", 8, "externalfixture"),  # STRING
        ("general.file_type", 4, 1),  # UINT32 MOSTLY_F16
    ], descriptors)


def external_safetensors(rows):
    header = {}
    data = bytearray()
    for name, shape, payload in rows:
        start = len(data)
        data.extend(payload)
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [start, len(data)]}
    return safetensors_wire(header, bytes(data))


FORMATS = (
    ("gguf", "gguf", "F16", external_gguf, 8.0e-6, 5.0e-5),
    ("safetensors", "huggingface", "BF16", external_safetensors, 6.25e-5, 4.0e-4),
)


class ExternalQuantizedM3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = Path(os.environ["IO5_LIVE_QUANTIZED_BRIDGE"]).resolve()
        if not (cls.bridge.is_file() and os.access(cls.bridge, os.X_OK)):
            raise RuntimeError(f"Compiled M3 bridge is not executable: {cls.bridge}")

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.original = self.directory / "original.etcp"
        self.reference = self.logits(self.bridge, "write", self.original)
        raw = self.original.read_bytes()
        state = parse_golden(raw, expected_provider=PROVIDER.encode())
        self.assertEqual(tuple(group.members for group in state.aliases), ((10, 14),))
        self.source = []
        for (path, _, _, shape), record in zip(M3_ROWS, state.records, strict=True):
            self.assertEqual((record.path, record.kind, record.dtype), (tuple(path.encode().split(b".")), 1, 3))
            self.assertEqual(struct.unpack_from(f"<{record.rank}Q", raw, record.extent_start), shape)
            self.source.append(raw[record.payload_start:record.payload_start + record.payload_bytes])

    def command(self, *args, status=0):
        result = subprocess.run(list(map(str, args)), cwd=self.directory, text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, status, f"{args}\n{result.stdout}{result.stderr}")
        if status == 0:
            self.assertEqual(result.stderr, "")
        return result

    def logits(self, *args):
        tag, *bits = self.command(*args).stdout.split()
        self.assertEqual(tag, "M3-LOGITS")
        self.assertEqual(len(bits), 4 * 512)
        values = struct.unpack("<512f", bytes(map(int, bits)))
        self.assertTrue(all(map(math.isfinite, values)))
        return values

    def fixture(self, suffix, dtype):
        rows, records = [], []
        weight_delta = 0.0
        for (path, hf_name, gguf_name, shape), source in zip(M3_ROWS, self.source, strict=True):
            wire, widened = narrow(source, dtype)
            name = gguf_name if suffix == "gguf" else hf_name
            rows.append((name, shape, wire))
            records.append((tuple(path.encode().split(b".")), 1, 3, shape, widened))
            weight_delta = max(weight_delta, max(abs(a - b) for (a,), (b,) in zip(
                struct.iter_unpack("<f", source), struct.iter_unpack("<f", widened), strict=True)))
        expected_c1 = build_container(tuple(records), ((10, 14),), PROVIDER.encode())
        return rows, expected_c1, weight_delta

    def import_command(self, suffix, wire, destination, *flags, status=0):
        return self.command(sys.executable, TOOL, f"{suffix}-to-c1", wire, destination, *flags, status=status)

    def assert_import_rejected(self, suffix, wire, destination, diagnostic, *flags):
        result = self.import_command(suffix, wire, destination, *flags, status=2)
        self.assertIn(diagnostic, result.stderr)
        self.assertFalse(destination.exists())
        previous = self.original.read_bytes()
        destination.write_bytes(previous)
        result = self.import_command(suffix, wire, destination, *flags, "--overwrite", status=2)
        self.assertIn(diagnostic, result.stderr)
        self.assertEqual(destination.read_bytes(), previous)
        destination.unlink()

    def test_external_weights_load_and_evaluate(self):
        for suffix, scheme, dtype, build_wire, weight_bound, logit_bound in FORMATS:
            with self.subTest(format=suffix, dtype=dtype):
                rows, expected_c1, weight_delta = self.fixture(suffix, dtype)
                wire = self.directory / f"external.{suffix}"
                wire.write_bytes(build_wire(rows))
                restored = self.directory / f"{suffix}.etcp"
                readback = self.directory / f"{suffix}-readback.etcp"
                self.import_command(suffix, wire, restored, "--external", "--scheme", scheme,
                                    "--profile", PROFILE, "--provider", PROVIDER, "--tie-head-to-embedding")
                self.assertEqual(restored.read_bytes(), expected_c1)
                observed = self.logits(self.bridge, "evaluate", restored, readback)
                self.assertEqual(readback.read_bytes(), expected_c1)
                logit_delta = max(abs(a - b) for a, b in zip(self.reference, observed, strict=True))
                self.assertGreater(weight_delta, 0)
                self.assertLessEqual(weight_delta, weight_bound)
                self.assertGreater(logit_delta, 0)
                self.assertLessEqual(logit_delta, logit_bound)
                print(f"M3 external {suffix} {dtype}: 512 logits, max |Δweight|={weight_delta:.9g}, "
                      f"max |Δlogit|={logit_delta:.9g}, tied F32 C1 SHA-256="
                      f"{hashlib.sha256(expected_c1).hexdigest()}", flush=True)

    def test_missing_mapping_arguments_and_unequal_tie_reject_before_publication(self):
        for suffix, scheme, dtype, build_wire, _, _ in FORMATS:
            with self.subTest(format=suffix, dtype=dtype):
                rows, _, _ = self.fixture(suffix, dtype)
                wire = self.directory / f"external.{suffix}"
                wire.write_bytes(build_wire(rows))
                destination = self.directory / f"{suffix}-rejected.etcp"
                common = ("--external", "--scheme", scheme, "--tie-head-to-embedding")
                profile, provider = ("--profile", PROFILE), ("--provider", PROVIDER)
                self.assert_import_rejected(suffix, wire, destination, "explicit supported profile",
                                            *common, *provider)
                self.assert_import_rejected(suffix, wire, destination, "explicit external provider",
                                            *common, *profile)

                name, shape, payload = rows[10]
                first_bits, = struct.unpack_from("<H", payload)
                sign_flip = struct.pack("<H", first_bits ^ 0x8000)  # distinct finite head value
                rows[10] = (name, shape, sign_flip + payload[2:])
                wire.write_bytes(build_wire(rows))
                self.assert_import_rejected(suffix, wire, destination, "head/embedding tie requires equal",
                                            *common, *profile, *provider)


if __name__ == "__main__":
    unittest.main()
