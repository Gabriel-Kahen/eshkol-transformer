from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "io5"))
sys.path.insert(0, str(ROOT / "tests" / "c1"))
import safetensors_interop as io
from test_checkpoint_format import build_container, parse_golden


def safe(header, data=b""):
    encoded = io.json_bytes(header)
    return struct.pack("<Q", len(encoded)) + encoded + data


def unpack(raw):
    length = struct.unpack_from("<Q", raw)[0]
    return json.loads(raw[8:8 + length]), raw[8 + length:]


def fixture():
    # Preserve every interesting f32 representation as bits, with no float conversion.
    bits = (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001,
            1, 0x007FFFFF, 0x00800000, 0x3F800000, 0xFFFFFFFF, 0x7F7FFFFF)
    data = struct.pack("<12I", *bits)
    return io.State("i2-dense-cpu-f32-v1", (
        io.Tensor(("a.b", "λ/\x00"), "parameter", (3, 4), data),
        io.Tensor(("empty",), "buffer", (2, 0, 3), b""),
        io.Tensor(("scalar",), "buffer", (), struct.pack("<I", 0x80000000)),
        io.Tensor(("tied",), "parameter", (3, 4), data),
    ), ((0, 3),))


class InteropTests(unittest.TestCase):
    def test_bitwise_roundtrip_and_independent_c1_parser(self):
        state = fixture()
        original = io.encode_c1(state)
        parsed = parse_golden(original, expected_provider=b"i2-dense-cpu-f32-v1")
        self.assertEqual(len(parsed.records), 4)
        independent = build_container(tuple((tuple(p.encode("utf-8") for p in tensor.path),
                                             1 if tensor.kind == "parameter" else 2,
                                             3, tensor.shape, tensor.data)
                                            for tensor in state.tensors),
                                      state.aliases, state.provider.encode("utf-8"))
        self.assertEqual(original, independent)
        self.assertEqual(io.decode_c1(original), state)
        exported = io.encode_safetensors(original)
        self.assertEqual(exported, io.encode_safetensors(original))
        self.assertEqual(io.decode_safetensors(exported), original)

    def test_m3_shape_roster_and_tied_head(self):
        manifest = json.loads((ROOT / "tests/m3/reference_manifest.json").read_text())
        entries = {}
        for index, (name, shape) in enumerate(sorted(manifest["parameter_shapes"].items())):
            entries[name] = io.Tensor(tuple(name.split("/")), "parameter", tuple(shape),
                                      struct.pack("<I", 0x3F000000 + index) * io.elements(tuple(shape)))
        head = entries["head/weight"]
        entries["token_embedding/weight"] = io.Tensor(("token_embedding", "weight"), "parameter",
                                                       head.shape, head.data)
        names = sorted(entries)
        state = io.State("i2-dense-cpu-f32-v1", tuple(entries[n] for n in names),
                         ((names.index("head/weight"), names.index("token_embedding/weight")),))
        original = io.encode_c1(state)
        self.assertEqual(io.decode_safetensors(io.encode_safetensors(original)), original)
        self.assertEqual(len(state.tensors), 15)

    def test_empty_state_and_scalars(self):
        for state in (io.State("inert", ()), io.State("inert", (io.Tensor(("x",), "parameter", (), bytes(4)),))):
            with self.subTest(state=state):
                original = io.encode_c1(state)
                self.assertEqual(io.decode_safetensors(io.encode_safetensors(original)), original)

    def test_plain_import_is_explicit_and_lossless_tensor_bytes(self):
        raw = safe({"dot.name": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                    "zero": {"dtype": "F32", "shape": [0], "data_offsets": [4, 4]},
                    "__metadata__": {"description": "ignored explicitly"}}, b"\x01\x00\xc0\x7f")
        with self.assertRaisesRegex(io.InteropError, "plain safetensors requires"):
            io.decode_safetensors(raw)
        result = io.decode_c1(io.decode_safetensors(raw, plain_provider="inert"))
        self.assertEqual(result.tensors[0].path, ("dot.name",))
        self.assertEqual(result.tensors[0].kind, "parameter")
        self.assertEqual(result.tensors[0].data, b"\x01\x00\xc0\x7f")
        self.assertEqual(result.aliases, ())

    def test_bad_safetensors_structures(self):
        descriptor = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        cases = [b"", bytes(8), struct.pack("<Q", 2**64 - 1) + b"{}",
                 struct.pack("<Q", 3) + b" {}", safe([], b""),
                 safe({"x": descriptor}, bytes(5)), safe({"x": descriptor}, bytes(3)),
                 safe({"x": descriptor, "y": descriptor}, bytes(4)),
                 safe({"__metadata__": {"bad": 3}}),
                 safe({"x": descriptor, "__metadata__": {"ignored": "\ud800"}}, bytes(4)),
                 struct.pack("<Q", 13) + b'{"x":0,"x":0}',
                 struct.pack("<Q", 17) + b'{"x":' + b"[" * 17 + b"]" * 17 + b"}"]
        for field, value in (("dtype", "F16"), ("shape", [True]), ("shape", [1.0]),
                             ("shape", [-1]), ("shape", [2**64]), ("shape", [2**63, 3]),
                             ("shape", [1] * 65), ("shape", [float("nan")]),
                             ("data_offsets", [-1, 3]), ("data_offsets", [4, 0]),
                             ("data_offsets", [True, 4]), ("data_offsets", [0, 0]),
                             ("data_offsets", [1, 5]), ("data_offsets", [0, 2**64])):
            changed = dict(descriptor, **{field: value})
            cases.append(safe({"x": changed}, bytes(4)))
        cases += [safe({"x": dict(descriptor, extra=0)}, bytes(4)),
                  safe({"x": {"shape": [1], "data_offsets": [0, 4]}}, bytes(4)),
                  safe({"x": {"dtype": "F32", "shape": [0], "data_offsets": [1, 1]}}, b"x"),
                  safe({"\ud800": descriptor}, bytes(4))]
        for index, raw in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(io.InteropError):
                io.decode_safetensors(raw, plain_provider="inert")
        with self.assertRaisesRegex(io.InteropError, "duplicate JSON"):
            io.parse_json(b'{"shape":[],"shape":[]}', 100)
        with self.assertRaisesRegex(io.InteropError, "nesting limit"):
            io.parse_json(b"[" * 17 + b"]" * 17, 100)
        for header in (b'{"x":{"dtype":"F32","shape":[-0],"data_offsets":[0,0]}}',
                       b'{"x":{"dtype":"F32","shape":[0],"data_offsets":[-0,0]}}'):
            with self.subTest(header=header), self.assertRaisesRegex(io.InteropError, "unsigned integer"):
                io.decode_safetensors(struct.pack("<Q", len(header)) + header, plain_provider="inert")

    def test_io5_manifest_rejects_loss_and_corruption(self):
        raw = io.encode_safetensors(io.encode_c1(fixture()))
        header, data = unpack(raw)
        mutations = []
        for key, value in ((io.VERSION_KEY, "2.0"), (io.DIGEST_KEY, "0" * 64),
                           ("eshkol.io5.extra", "x"), (io.MANIFEST_KEY, "[]"),
                           (io.MANIFEST_KEY, "{\"provider\":1,\"provider\":2}"),
                           (io.MANIFEST_KEY, "\ud800")):
            changed = copy.deepcopy(header)
            changed["__metadata__"][key] = value
            mutations.append(safe(changed, data))
        for mutate in (lambda m: m["entries"].pop(),
                       lambda m: m["entries"][0].update(path=["z"]),
                       lambda m: m["entries"][0].update(kind="other"),
                       lambda m: m["entries"][1].update(name=m["entries"][0]["name"]),
                       lambda m: m.update(provider=""),
                       lambda m: m.update(aliases=[[0, 99]]),
                       lambda m: m.update(aliases=[[0, 0]]),
                       lambda m: m.update(aliases=[[3, 0]]),
                       lambda m: m.update(aliases=[[0, 3], [0, 3]]),
                       lambda m: m.update(aliases=[[0, 1]])):
            changed = copy.deepcopy(header)
            manifest = json.loads(changed["__metadata__"][io.MANIFEST_KEY])
            mutate(manifest)
            changed["__metadata__"][io.MANIFEST_KEY] = io.json_bytes(manifest).decode()
            mutations.append(safe(changed, data))
        mutations.append(raw[:-1] + bytes([raw[-1] ^ 1]))
        changed = copy.deepcopy(header)
        changed["__metadata__"][io.MANIFEST_KEY] = changed["__metadata__"][io.MANIFEST_KEY].replace(
            '"aliases":[[0,3]]', '"aliases":[[-0,3]]')
        self.assertIn('"aliases":[[-0,3]]', changed["__metadata__"][io.MANIFEST_KEY])
        mutations.append(safe(changed, data))
        for index, changed in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(io.InteropError):
                io.decode_safetensors(changed)
        with self.assertRaisesRegex(io.InteropError, "cannot override"):
            io.decode_safetensors(raw, plain_provider="other")

    def test_c1_corruption_and_unsupported(self):
        original = io.encode_c1(fixture())
        cases = [original[:-1], original + b"\x00", bytes(len(original))]
        for offset in (16, 32, 40, 48, 56, 64, 72, 80, 84, 88, 92, 96, 100,
                       128 + len(fixture().provider) + 34):
            changed = bytearray(original)
            changed[offset] ^= 1
            changed[-32:] = io.digest(io.WHOLE_DOMAIN, changed[:-32])
            cases.append(bytes(changed))
        # A whole-container-valid file with a broken record digest must still fail.
        changed = bytearray(original)
        changed[-33] ^= 1
        changed[-32:] = io.digest(io.WHOLE_DOMAIN, changed[:-32])
        cases.append(bytes(changed))
        # Independently built and checksummed malformed logical states reach deep validation.
        basic = ((b"a",), 1, 3, (1,), bytes(4))
        cases += [build_container((((b"bool",), 2, 1, (1,), b"\x00"),)),
                  build_container((((b"i64",), 2, 2, (1,), bytes(8)),)),
                  build_container((basic, basic)),
                  build_container((((b"\xff",), 1, 3, (1,), bytes(4)),)),
                  build_container((basic, ((b"b",), 1, 3, (), bytes(4))), ((0, 1),)),
                  build_container((basic, ((b"b",), 2, 3, (1,), bytes(4))), ((0, 1),)),
                  build_container((basic, ((b"b",), 1, 3, (1,), b"\x01" * 4)), ((0, 1),)),
                  build_container((basic,), ((0, 99),))]
        for index, changed in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(io.InteropError):
                io.encode_safetensors(changed)
        first, *rest = fixture().tensors
        for tensor in (io.Tensor(first.path, first.kind, (4, 3), first.data),
                       io.Tensor(first.path, first.kind, first.shape, bytes(len(first.data)))):
            bad = io.State("inert", tuple([first] + rest[:-1] + [io.Tensor(("tied",), tensor.kind, tensor.shape, tensor.data)]), ((0, 3),))
            with self.assertRaisesRegex(io.InteropError, "alias shape/value conflict"):
                io.encode_c1(bad)

    def test_lowered_policy_exact_and_one_over(self):
        original = io.encode_c1(fixture())
        exported = io.encode_safetensors(original)
        header, _ = unpack(exported)
        length = struct.unpack_from("<Q", exported)[0]
        self.assertEqual(io.decode_safetensors(exported, io.Limits(file=len(exported), metadata=length, tensor=48, entries=4)), original)
        for limits in (io.Limits(file=len(exported) - 1), io.Limits(metadata=length - 1),
                       io.Limits(tensor=47), io.Limits(entries=3)):
            with self.subTest(limits=limits), self.assertRaises(io.InteropError):
                io.decode_safetensors(exported, limits)
        for kwargs in ({"file": 2**64}, {"tensor": -1}, {"entries": True}):
            with self.assertRaises(io.InteropError):
                io.Limits(**kwargs)

    def test_seeded_mutation_smoke(self):
        rng = random.Random(1729)
        for original, decode in ((io.encode_c1(fixture()), io.decode_c1),
                                 (io.encode_safetensors(io.encode_c1(fixture())), io.decode_safetensors)):
            for _ in range(128):
                changed = bytearray(original)
                offset = rng.randrange(len(changed))
                changed[offset] ^= 1 << rng.randrange(8)
                try:
                    decode(bytes(changed))
                except io.InteropError:
                    pass  # Mutated optional text/whitespace may remain valid.

    def test_cli_atomic_no_clobber_and_failed_import(self):
        command = [sys.executable, str(ROOT / "tools/io5/safetensors_interop.py")]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, target, restored = (directory / name for name in ("source", "safe", "restored"))
            original = io.encode_c1(fixture())
            source.write_bytes(original)
            result = subprocess.run(command + ["c1-to-safetensors", str(source), str(target)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
            result = subprocess.run(command + ["safetensors-to-c1", str(target), str(restored)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(restored.read_bytes(), original)
            before = target.read_bytes()
            result = subprocess.run(command + ["c1-to-safetensors", str(source), str(target)], capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(target.read_bytes(), before)
            source.write_bytes(b"bad")
            result = subprocess.run(command + ["c1-to-safetensors", str(source), str(target), "--overwrite"], capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(target.read_bytes(), before)
            self.assertFalse(list(directory.glob(".io5-*")))
            link = directory / "link"
            link.symlink_to(target)
            with self.assertRaises(OSError):
                io.read_file(link, io.Limits())
            with self.assertRaises(io.InteropError):
                io.read_file(target, io.Limits(file=1))
            # A failed replacement or file fsync leaves the old destination intact.
            for operation in ("os.replace", "os.fsync"):
                with patch.object(io.os, operation.split(".")[1], side_effect=OSError("injected")):
                    with self.assertRaises(OSError):
                        io.publish(target, b"new", True)
                self.assertEqual(target.read_bytes(), before)
                self.assertFalse(list(directory.glob(".io5-*")))
            # The post-publication directory fsync may fail after the new bytes are visible.
            with patch.object(io.os, "fsync", side_effect=[None, OSError("directory fsync")]):
                with self.assertRaises(OSError):
                    io.publish(target, b"published", True)
            self.assertEqual(target.read_bytes(), b"published")

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "optional independent safetensors package not installed")
    def test_independent_safetensors_package(self):
        import safetensors
        self.assertEqual(safetensors.__version__, "0.6.2", "reference version requires an explicit contract update")
        original = io.encode_c1(fixture())
        decoded = dict(safetensors.deserialize(io.encode_safetensors(original)))
        for index, tensor in enumerate(fixture().tensors):
            actual = decoded[f"t{index:06d}"]
            self.assertEqual(actual["shape"], list(tensor.shape))
            self.assertEqual(bytes(actual["data"]), tensor.data)
        tensors = {"external": {"dtype": "float32", "shape": [2], "data": struct.pack("<II", 0x7F800001, 0x80000000)},
                   "scalar": {"dtype": "float32", "shape": [], "data": bytes(4)},
                   "empty": {"dtype": "float32", "shape": [0, 2], "data": b""}}
        raw = safetensors.serialize(tensors)
        state = io.decode_c1(io.decode_safetensors(raw, plain_provider="inert"))
        for tensor in state.tensors:
            self.assertEqual(tensor.data, tensors[tensor.path[0]]["data"])
            self.assertEqual(tensor.shape, tuple(tensors[tensor.path[0]]["shape"]))


if __name__ == "__main__":
    unittest.main()
