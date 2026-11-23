"""Run the compiled C1 bridge across the five IO5 command-line routes."""

from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from test_gguf_interop import unpack_export

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools/io5"
sys.path.insert(0, str(TOOLS))
import safetensors_interop as safe

INDEX = "model.safetensors.index.json"
M3 = ("--profile", "eshkol-m3-v1")
BRIDGE_ENV = "IO5_LIVE_C1_BRIDGE"
ROUTES = (
    ("safetensors_interop.py", "safetensors", (), "safetensors"),
    ("gguf_interop.py", "gguf", (), "gguf"),
    ("model_mapping.py", "mapped-safetensors", M3, "safetensors"),
    ("model_mapping.py", "mapped-gguf", M3, "gguf"),
    ("model_mapping.py", "sharded-safetensors",
     (*M3, "--scheme", "huggingface", "--max-shard-bytes", "4200"), "shards"),
)


def flip(raw, offset):
    return raw[:offset] + bytes([raw[offset] ^ 1]) + raw[offset + 1:]


def first_safetensors_payload(raw):
    return 8 + struct.unpack_from("<Q", raw)[0]


class LiveC1InteropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bridge = os.environ.get(BRIDGE_ENV)
        if not bridge:
            if __name__ == "__main__":
                raise RuntimeError(f"Set {BRIDGE_ENV} to the compiled live-c1-io5 binary")
            raise unittest.SkipTest("requires scripts/test-io5-live-c1.sh")
        cls.bridge = Path(bridge).resolve()
        if not (cls.bridge.is_file() and os.access(cls.bridge, os.X_OK)):
            raise RuntimeError(f"Compiled live C1 bridge is not executable: {cls.bridge}")

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.original = self.root / "original.etcp"
        result = self.command(self.bridge, "write", self.original)
        self.assertEqual(result.stdout, "WROTE M3 C1\n")
        self.original_bytes = self.original.read_bytes()

    def command(self, *args, status=0):
        result = subprocess.run(list(map(str, args)), text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, status, f"{args}\n{result.stdout}{result.stderr}")
        return result

    def verify(self, path):
        result = self.command(self.bridge, "verify", path)
        self.assertEqual(result.stdout, "VERIFIED M3 C1\n")

    def test_five_byte_identical_round_trips_with_live_reload(self):
        self.verify(self.original)
        for tool, route, options, wire in ROUTES:
            with self.subTest(route=route):
                directory = self.root / route
                directory.mkdir()
                external = directory / (INDEX if wire == "shards" else f"wire.{wire}")
                restored = directory / "restored.etcp"
                self.command(sys.executable, TOOLS / tool, f"c1-to-{route}", self.original, external, *options)
                self.command(sys.executable, TOOLS / tool, f"{route}-to-c1", external, restored)
                self.assertEqual(restored.read_bytes(), self.original_bytes)
                self.verify(restored)
                if wire == "shards":
                    index = json.loads(external.read_bytes())
                    shards = sorted(set(index["weight_map"].values()))
                    self.assertGreater(len(shards), 1)
                    for basename in shards:
                        self.assertLessEqual((directory / basename).stat().st_size, 4200)
                    tamper_target = directory / shards[0]
                    offset = first_safetensors_payload(tamper_target.read_bytes())
                    diagnostic = "shard SHA-256 mismatch"
                else:
                    tamper_target = external
                    raw = external.read_bytes()
                    offset = first_safetensors_payload(raw) if wire == "safetensors" else unpack_export(raw)[3]
                    diagnostic = "checksum"
                tamper_target.write_bytes(flip(tamper_target.read_bytes(), offset))
                rejected = directory / "rejected.etcp"
                result = self.command(sys.executable, TOOLS / tool, f"{route}-to-c1", external, rejected, status=2)
                self.assertIn(diagnostic, result.stderr)
                self.assertFalse(rejected.exists())

    def test_c1_payload_corruption_fails_live_load(self):
        altered = self.root / "corrupt.etcp"
        payload_offset = struct.unpack_from("<Q", self.original_bytes, 64)[0]
        altered.write_bytes(flip(self.original_bytes, payload_offset))
        result = self.command(self.bridge, "verify", altered, status=2)
        self.assertIn("corrupt-data", result.stdout)
        self.assertIn("checksum", result.stdout)

    def test_valid_c1_checksums_do_not_bypass_fixture_verification(self):
        state = safe.decode_c1(self.original_bytes)
        first = state.tensors[0]
        for changed, diagnostic in (
            (replace(state, provider="changed-provider"), "provider identities differ"),
            (replace(state, aliases=()), "unexpected head/embedding alias group"),
            (replace(state, tensors=(replace(first, path=("blocks", "0", "attention", "extra", "weight")),)
                                    + state.tensors[1:]), "unexpected M3 tensor paths"),
            (replace(state, tensors=(replace(first, kind="buffer"),) + state.tensors[1:]),
             "unexpected tensor metadata"),
            (replace(state, tensors=(replace(first, shape=(2, 8)),) + state.tensors[1:]),
             "unexpected tensor metadata"),
            (replace(state, tensors=(replace(first, data=flip(first.data, 0)),) + state.tensors[1:]),
             "unexpected F32 payload bytes"),
        ):
            with self.subTest(diagnostic=diagnostic):
                altered = self.root / "valid-but-different.etcp"
                altered.write_bytes(safe.encode_c1(changed))
                result = self.command(self.bridge, "verify", altered, status=2)
                self.assertIn(diagnostic, result.stdout)


if __name__ == "__main__":
    unittest.main()
