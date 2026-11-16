from __future__ import annotations

import copy
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/io5"))
sys.path.insert(0, str(ROOT / "tests/c1"))
import gguf_interop as gg
import safetensors_interop as c1
from test_checkpoint_format import build_container, parse_golden


def fixture():
    bits = (0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001,
            1, 0x007FFFFF, 0x00800000, 0x3F800000, 0xFFFFFFFF, 0x7F7FFFFF)
    data = struct.pack("<12I", *bits)
    return c1.State("i2-dense-cpu-f32-v1", (
        c1.Tensor(("a.b", "λ/\x00"), "parameter", (3, 4), data),
        c1.Tensor(("b",), "buffer", (12,), data),
        c1.Tensor(("c",), "parameter", (2, 3, 4), struct.pack("<24I", *range(24))),
        c1.Tensor(("d",), "buffer", (2, 3, 4, 5), struct.pack("<120I", *range(120))),
        c1.Tensor(("tied",), "parameter", (3, 4), data),
    ), ((0, 4),))


def wire(metadata=None, tensors=None, *, trailing=b""):
    """Independent compact wire builder, including deliberately malformed fixtures."""
    if metadata is None:
        metadata = [("general.architecture", 8, "test")]
    if tensors is None:
        tensors = [("x", (3, 2), 0, 0, bytes(24))]  # Wire GGML shape, not C order.
    def string(value):
        encoded = value.encode("utf-8") if type(value) is str else value
        return struct.pack("<Q", len(encoded)) + encoded
    prefix = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata)))
    for key, kind, value in metadata:
        prefix += string(key) + struct.pack("<I", kind)
        prefix += string(value) if kind == 8 else struct.pack("<I", value)
    for name, dims, kind, offset, data in tensors:
        prefix += string(name) + struct.pack("<I", len(dims))
        prefix += struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", kind, offset)
    prefix += bytes((-len(prefix)) % 32)
    for name, dims, kind, offset, data in tensors:
        prefix += data + bytes((-len(data)) % 32)
    return bytes(prefix) + trailing


def unpack_export(raw):
    reader = gg.Reader(raw, c1.Limits())
    metadata = []
    count = struct.unpack_from("<Q", raw, 16)[0]
    for _ in range(count):
        key, kind = reader.text(128), reader.number("I")
        value = reader.text(1024 * 1024) if kind == 8 else reader.number("I")
        metadata.append((key, kind, value))
    infos = []
    for _ in range(struct.unpack_from("<Q", raw, 8)[0]):
        name, rank = reader.text(63), reader.number("I")
        dims = tuple(reader.number("Q") for _ in range(rank))
        kind, offset = reader.number("I"), reader.number("Q")
        infos.append((name, dims, kind, offset))
    start = (reader.at + 31) // 32 * 32
    tensors = []
    for name, dims, kind, offset in infos:
        size = 4
        for extent in dims:
            size *= extent
        tensors.append((name, dims, kind, offset, raw[start + offset:start + offset + size]))
    return metadata, tensors, reader.at, start


class GGUFTests(unittest.TestCase):
    def test_roundtrip_independent_c1_and_wire(self):
        state = fixture()
        raw = c1.encode_c1(state)
        independent = build_container(tuple((tuple(p.encode("utf-8") for p in t.path),
                                             1 if t.kind == "parameter" else 2,
                                             3, t.shape, t.data) for t in state.tensors),
                                      state.aliases, state.provider.encode())
        self.assertEqual(raw, independent)
        parse_golden(raw, expected_provider=state.provider.encode())
        exported = gg.encode_gguf(raw)
        self.assertEqual(exported, gg.encode_gguf(raw))
        self.assertEqual(gg.decode_gguf(exported), raw)
        metadata, tensors, _, _ = unpack_export(exported)
        self.assertEqual(wire(metadata, tensors), exported)
        self.assertEqual([t[1] for t in tensors], [tuple(reversed(t.shape)) for t in state.tensors])
        empty = c1.encode_c1(c1.State("inert", ()))
        self.assertEqual(gg.decode_gguf(gg.encode_gguf(empty)), empty)

    def test_plain_import_explicit_shape_order_names_and_alignment(self):
        raw = wire(tensors=[("dots.slash/λ", (3, 2), 0, 0, struct.pack("<6I", *range(6)))])
        with self.assertRaisesRegex(c1.InteropError, "explicit"):
            gg.decode_gguf(raw)
        state = c1.decode_c1(gg.decode_gguf(raw, plain_provider="inert"))
        self.assertEqual(state.tensors[0].path, ("dots.slash/λ",))
        self.assertEqual(state.tensors[0].kind, "parameter")
        self.assertEqual(state.tensors[0].shape, (2, 3))
        self.assertEqual(state.tensors[0].data, struct.pack("<6I", *range(6)))
        self.assertEqual(state.aliases, ())
        for alignment in ([], [("general.alignment", 4, 32)]):
            raw = wire([("general.architecture", 8, "test"), *alignment])
            gg.decode_gguf(raw, plain_provider="inert")
        metadata = [("general.architecture", 8, "test")] + [(f"extra.k{n}", 8, "") for n in range(127)]
        raw = wire(metadata, [("x" * 63, (1,), 0, 0, bytes(4))])
        state = c1.decode_c1(gg.decode_gguf(raw, plain_provider="inert"))
        self.assertEqual(state.tensors[0].path, ("x" * 63,))

    def test_unsupported_c1_domains(self):
        for shape, data in (((), bytes(4)), ((0,), b""), ((1, 1, 1, 1, 1), bytes(4))):
            raw = c1.encode_c1(c1.State("inert", (c1.Tensor(("x",), "parameter", shape, data),)))
            with self.subTest(shape=shape), self.assertRaises(c1.InteropError):
                gg.encode_gguf(raw)
        for raw in (build_container((((b"b",), 2, 1, (1,), b"\x00"),)),
                    build_container((((b"i",), 2, 2, (1,), bytes(8)),))):
            with self.assertRaisesRegex(c1.InteropError, "only F32"):
                gg.encode_gguf(raw)

    def test_header_metadata_and_string_negatives(self):
        good = wire()
        cases = [b"", good[:23], good[:-1], good + b"x", b"FUGG" + good[4:]]
        for offset, fmt, value in ((4, "I", 2), (4, ">I", 3), (8, "Q", 4097),
                                   (8, "Q", 2**64 - 1), (16, "Q", 129), (24, "Q", 2**64 - 1)):
            changed = bytearray(good)
            struct.pack_into(fmt if fmt.startswith(">") else "<" + fmt, changed, offset, value)
            cases.append(bytes(changed))
        for metadata in ([], [("general.architecture", 4, 0)], [("general.architecture", 8, "bad/name")],
                         [("general.architecture", 8, "test"), ("general.architecture", 8, "test")],
                         [("Invalid.Key", 8, "x")], [("a..b", 8, "x")], [("x" * 129, 8, "x")],
                         [(b"\xff", 8, "x")], [("general.architecture", 8, b"\xff")],
                         [("split.count", 4, 1)], [("general.quantization_version", 4, 2)]):
            cases.append(wire(metadata))
        for kind in (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 0xFFFFFFFF):
            cases.append(wire([("general.architecture", 8, "test"), ("extra", kind, 0)]))
        for kind, alignment in ((4, 0), (4, 8), (4, 24), (4, 64), (4, 2**32 - 1), (8, "32")):
            cases.append(wire([("general.architecture", 8, "test"), ("general.alignment", kind, alignment)]))
        for index, raw in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(c1.InteropError):
                gg.decode_gguf(raw, plain_provider="inert")

    def test_tensor_bounds_padding_order_and_shape_negatives(self):
        cases = []
        for name, dims, kind, offset, data in (
                ("", (1,), 0, 0, bytes(4)), ("x" * 64, (1,), 0, 0, bytes(4)),
                ("x\0y", (1,), 0, 0, bytes(4)), (b"\xff", (1,), 0, 0, bytes(4)),
                ("x", (), 0, 0, bytes(4)), ("x", (1,) * 5, 0, 0, bytes(4)),
                ("x", (0,), 0, 0, b""), ("x", (2**63,), 0, 0, bytes(4)),
                ("x", (2**32, 2**32), 0, 0, bytes(4)),
                ("x", (2**63 - 1,), 0, 0, bytes(4)),
                ("x", (1,), 1, 0, bytes(4)), ("x", (1,), 2, 0, bytes(4)),
                ("x", (1,), 0xFFFFFFFF, 0, bytes(4)),
                ("x", (1,), 0, 1, bytes(4)), ("x", (1,), 0, 32, bytes(4)),
                ("x", (1,), 0, 2**64 - 32, bytes(4))):
            cases.append(wire(tensors=[(name, dims, kind, offset, data)]))
        for tensors in ([ ("x", (1,), 0, 0, bytes(4)), ("x", (1,), 0, 32, bytes(4)) ],
                        [ ("x", (1,), 0, 0, bytes(4)), ("y", (1,), 0, 0, bytes(4)) ],
                        [ ("x", (1,), 0, 32, bytes(4)), ("y", (1,), 0, 0, bytes(4)) ],
                        [ ("x", (1,), 0, 0, bytes(4)), ("y", (1,), 0, 64, bytes(4)) ]):
            cases.append(wire(tensors=tensors))
        exported = gg.encode_gguf(c1.encode_c1(fixture()))
        _, _, end, start = unpack_export(exported)
        self.assertLess(end, start)
        for offset in (end, start + 48):
            changed = bytearray(exported)
            changed[offset] = 1
            with self.subTest(padding_offset=offset), self.assertRaisesRegex(c1.InteropError, "nonzero .*padding"):
                gg.decode_gguf(bytes(changed))
        for index, raw in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(c1.InteropError):
                gg.decode_gguf(raw, plain_provider="inert")

    def test_manifest_semantics_checksum_and_corruption(self):
        original = gg.encode_gguf(c1.encode_c1(fixture()))
        metadata, tensors, _, _ = unpack_export(original)
        meta = {key: (kind, value) for key, kind, value in metadata}
        mutations = []
        for key, value in ((gg.VERSION_KEY, "2.0"), (gg.DIGEST_KEY, "0" * 64),
                           ("eshkol.io5.extra", "x"), (gg.MANIFEST_KEY, "[]"),
                           (gg.MANIFEST_KEY, '{"provider":1,"provider":2}'),
                           ("general.architecture", "llama")):
            changed = dict(meta, **{key: (8, value)})
            mutations.append(wire([(k, t, v) for k, (t, v) in changed.items()], tensors))
        for mutate in (lambda m: m["entries"].pop(),
                       lambda m: m["entries"][0].update(path=["z"]),
                       lambda m: m["entries"][1].update(name="t000000"),
                       lambda m: m["entries"][0].update(kind="buffer"),
                       lambda m: m.update(provider=""),
                       lambda m: m.update(aliases=[[0, 99]]),
                       lambda m: m.update(aliases=[[0, 0]]),
                       lambda m: m.update(aliases=[[4, 0]]),
                       lambda m: m.update(aliases=[[0, 4], [0, 4]]),
                       lambda m: m.update(aliases=[[0, 1]]),
                       lambda m: m["entries"][0].update(path=["\ud800"])):
            changed = copy.deepcopy(meta)
            manifest = json.loads(changed[gg.MANIFEST_KEY][1])
            mutate(manifest)
            changed[gg.MANIFEST_KEY] = (8, c1.json_bytes(manifest).decode())
            mutations.append(wire([(k, t, v) for k, (t, v) in changed.items()], tensors))
        changed = copy.deepcopy(tensors)
        name, dims, kind, offset, data = changed[0]
        changed[0] = name, dims, kind, offset, bytes([data[0] ^ 1]) + data[1:]
        mutations.append(wire(metadata, changed))
        changed_meta = dict(meta)
        manifest = changed_meta[gg.MANIFEST_KEY][1].replace('"aliases":[[0,4]]', '"aliases":[[-0,4]]')
        self.assertIn('"aliases":[[-0,4]]', manifest)
        changed_meta[gg.MANIFEST_KEY] = (8, manifest)
        mutations.append(wire([(k, t, v) for k, (t, v) in changed_meta.items()], tensors))
        for index, raw in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(c1.InteropError):
                gg.decode_gguf(raw)
        with self.assertRaisesRegex(c1.InteropError, "cannot override"):
            gg.decode_gguf(original, plain_provider="other")

    def test_exact_and_one_over_lowering_policies(self):
        raw_c1 = c1.encode_c1(fixture())
        raw = gg.encode_gguf(raw_c1)
        _, _, _, start = unpack_export(raw)
        exact = c1.Limits(file=len(raw), metadata=start, tensor=480, entries=5)
        self.assertEqual(gg.decode_gguf(raw, exact), raw_c1)
        self.assertEqual(gg.encode_gguf(raw_c1, exact), raw)
        for limits in (c1.Limits(file=len(raw) - 1), c1.Limits(metadata=start - 1),
                       c1.Limits(tensor=479), c1.Limits(entries=4)):
            for convert, source in ((gg.decode_gguf, raw), (gg.encode_gguf, raw_c1)):
                with self.subTest(limits=limits, convert=convert), self.assertRaises(c1.InteropError):
                    convert(source, limits)
        self.assertEqual(gg.aligned(0), 0)
        with self.assertRaises(c1.InteropError):
            gg.aligned(2**64 - 1)

    def test_cli_no_clobber_and_malformed_input(self):
        command = [sys.executable, str(ROOT / "tools/io5/gguf_interop.py")]
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("source", "gguf", "restored")]
            source, target, restored = paths
            original = c1.encode_c1(fixture())
            source.write_bytes(original)
            for direction, first, second in (("c1-to-gguf", source, target), ("gguf-to-c1", target, restored)):
                result = subprocess.run(command + [direction, str(first), str(second)], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(restored.read_bytes(), original)
            before = target.read_bytes()
            result = subprocess.run(command + ["c1-to-gguf", str(source), str(target)], capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(target.read_bytes(), before)
            source.write_bytes(b"bad")
            result = subprocess.run(command + ["c1-to-gguf", str(source), str(target), "--overwrite"], capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(target.read_bytes(), before)
            self.assertFalse(list(Path(directory).glob(".io5-*")))

    def test_seeded_mutation_smoke(self):
        rng = random.Random(1729)
        raw = gg.encode_gguf(c1.encode_c1(fixture()))
        for _ in range(256):
            changed = bytearray(raw)
            offset = rng.randrange(len(changed))
            changed[offset] ^= 1 << rng.randrange(8)
            try:
                gg.decode_gguf(bytes(changed))
            except c1.InteropError:
                pass  # Ignored string metadata edits can remain valid.

    @unittest.skipUnless(importlib.util.find_spec("gguf"), "optional independent gguf package absent")
    def test_independent_upstream_reader_writer(self):
        import gguf
        import numpy as np
        self.assertEqual(importlib.metadata.version("gguf"), "0.19.0")
        self.assertEqual(np.__version__, "2.5.4")
        original = c1.encode_c1(fixture())
        exported = gg.encode_gguf(original)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, rewritten, plain = (directory / name for name in ("source.gguf", "rewritten.gguf", "plain.gguf"))
            source.write_bytes(exported)
            reader = gguf.GGUFReader(source)
            for expected, actual in zip(fixture().tensors, reader.tensors, strict=True):
                self.assertEqual(tuple(actual.shape), tuple(reversed(expected.shape)))
                self.assertEqual(actual.data.shape, expected.shape)
                self.assertEqual(actual.data.view(np.uint32).tobytes(), expected.data)
            writer = gguf.GGUFWriter(rewritten, "eshkol")
            for key, field in reader.fields.items():
                if key.startswith("eshkol.io5."):
                    writer.add_string(key, field.contents())
            writer.add_uint32("general.alignment", 32)
            for actual in reader.tensors:
                writer.add_tensor(actual.name, actual.data)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            self.assertEqual(gg.decode_gguf(rewritten.read_bytes()), original)
            writer = gguf.GGUFWriter(plain, "test")
            expected = {}
            for rank, shape in enumerate(((7,), (2, 3), (2, 3, 4), (2, 3, 4, 5)), 1):
                size = c1.elements(shape)
                array = (np.arange(size, dtype=np.uint32) + np.uint32(0x7F800001)).view(np.float32).reshape(shape)
                expected[f"rank{rank}"] = array
                writer.add_tensor(f"rank{rank}", array)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            state = c1.decode_c1(gg.decode_gguf(plain.read_bytes(), plain_provider="inert"))
            for tensor in state.tensors:
                array = expected[tensor.path[0]]
                self.assertEqual(tensor.shape, array.shape)
                self.assertEqual(tensor.data, array.view(np.uint32).tobytes())


if __name__ == "__main__":
    unittest.main()
