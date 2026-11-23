from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / part) for part in ("tools/io5", "tests/io5", "tests/c1")]
import model_mapping as mapping
import safetensors_interop as safe
from test_checkpoint_format import build_container
from test_gguf_interop import unpack_export, wire
from test_safetensors_interop import safe as safetensors_wire, unpack

TOOL = ROOT / "tools/io5/model_mapping.py"
PROVIDER = "i2-dense-cpu-f32-v1"
BITS = (0x80000000, 1, 0x007FFFFF, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F800001)
# Independent fixed-profile roster and expected external spelling.
M3_ROWS = (
    ("blocks.0.attention.key.weight", "model.layers.0.self_attn.k_proj.weight", "blk.0.attn_k.weight", (4, 4)),
    ("blocks.0.attention.output.weight", "model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight", (4, 4)),
    ("blocks.0.attention.query.weight", "model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight", (4, 4)),
    ("blocks.0.attention.value.weight", "model.layers.0.self_attn.v_proj.weight", "blk.0.attn_v.weight", (4, 4)),
    ("blocks.0.ffn.down.weight", "model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight", (4, 8)),
    ("blocks.0.ffn.up.weight", "model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight", (8, 4)),
    ("blocks.0.norm1.bias", "model.layers.0.input_layernorm.bias", "blk.0.attn_norm.bias", (4,)),
    ("blocks.0.norm1.weight", "model.layers.0.input_layernorm.weight", "blk.0.attn_norm.weight", (4,)),
    ("blocks.0.norm2.bias", "model.layers.0.post_attention_layernorm.bias", "blk.0.ffn_norm.bias", (4,)),
    ("blocks.0.norm2.weight", "model.layers.0.post_attention_layernorm.weight", "blk.0.ffn_norm.weight", (4,)),
    ("head.weight", "lm_head.weight", "output.weight", (256, 4)),
    ("norm_final.bias", "model.norm.bias", "output_norm.bias", (4,)),
    ("norm_final.weight", "model.norm.weight", "output_norm.weight", (4,)),
    ("position_embedding.weight", "model.embed_positions.weight", "position_embd.weight", (2, 4)),
    ("token_embedding.weight", "model.embed_tokens.weight", "token_embd.weight", (256, 4)),
)
MOD5_ROOT_ROWS = (
    ("head.weight", "lm_head.weight", "output.weight", (17, 8)),
    ("norm_final.weight", "model.norm.weight", "output_norm.weight", (8,)),
    ("token_embedding.weight", "model.embed_tokens.weight", "token_embd.weight", (17, 8)),
)
MOD5_BLOCK_ROWS = (
    ("attention.key.weight", "self_attn.k_proj.weight", "attn_k.weight", (0, 8)),
    ("attention.output.weight", "self_attn.o_proj.weight", "attn_output.weight", (8, 8)),
    ("attention.query.weight", "self_attn.q_proj.weight", "attn_q.weight", (8, 8)),
    ("attention.value.weight", "self_attn.v_proj.weight", "attn_v.weight", (0, 8)),
    ("ffn.down.weight", "mlp.down_proj.weight", "ffn_down.weight", (8, 12)),
    ("ffn.gate.weight", "mlp.gate_proj.weight", "ffn_gate.weight", (12, 8)),
    ("ffn.up.weight", "mlp.up_proj.weight", "ffn_up.weight", (12, 8)),
    ("norm1.weight", "input_layernorm.weight", "attn_norm.weight", (8,)),
    ("norm2.weight", "post_attention_layernorm.weight", "ffn_norm.weight", (8,)),
)


def tensor(name, shape, seed=0):
    count = safe.elements(shape)
    bits = (BITS[(i + seed) % len(BITS)] for i in range(count))
    return safe.Tensor(tuple(name.split(".")), "parameter", shape, struct.pack(f"<{count}I", *bits))


def m3_fixture():
    tensors = [tensor(name, shape, index) for index, (name, _, _, shape) in enumerate(M3_ROWS)]
    tensors[-1] = replace(tensors[-1], data=tensors[10].data)
    return safe.State(PROVIDER, tuple(tensors), ((10, 14),))


def decoder_fixture(*, tied=False):
    tensors = [tensor("token_embedding.weight", (17, 6)), tensor("head.weight", (17, 6)),
               tensor("norm_final.weight", (6,))]
    for i in range(3):
        shapes = {"norm1.weight": (6,), "norm2.weight": (6,), "attention.query.weight": (6, 6),
                  "attention.key.weight": (2, 6), "attention.value.weight": (2, 6),
                  "attention.output.weight": (6, 6), "ffn.up.weight": (10, 6),
                  "ffn.down.weight": (6, 10), "ffn.gate.weight": (10, 6),
                  "attention.key.bias": (2,), "ffn.gate.bias": (10,)}
        tensors.extend(tensor(f"blocks.{i}.{part}", shape, i) for part, shape in shapes.items())
    tensors.sort(key=lambda t: t.path)
    paths = [t.path for t in tensors]
    groups = (tuple(paths.index(p) for p in (("head", "weight"), ("token_embedding", "weight"))),) if tied else ()
    return safe.State(PROVIDER, tuple(tensors), groups)


def mod5_fixture(kv_width, *, tied):
    rows = [
        (f"blocks.{i}.{path}", f"model.layers.{i}.{hf}", f"blk.{i}.{gguf}",
         shape if shape[0] else (kv_width, shape[1]))
        for i in range(2)
        for path, hf, gguf, shape in MOD5_BLOCK_ROWS
    ] + list(MOD5_ROOT_ROWS)
    tensors = [tensor(path, shape, index) for index, (path, _, _, shape) in enumerate(rows)]
    head, embedding = len(rows) - 3, len(rows) - 1
    if tied:
        tensors[embedding] = replace(tensors[embedding], data=tensors[head].data)
    return safe.State(PROVIDER, tuple(tensors), ((head, embedding),) if tied else ()), rows


def c1_bytes(state):
    return build_container(
        tuple((tuple(p.encode() for p in t.path), 1 if t.kind == "parameter" else 2, 3, t.shape, t.data)
              for t in state.tensors), state.aliases, state.provider.encode())


def gguf_plain(encoded, *, mismatch=None):
    _, rows, _, _ = unpack_export(encoded)
    return wire(tensors=[(name, shape, kind, offset, mismatch(data) if name == "output.weight" and mismatch else data)
                         for name, shape, kind, offset, data in rows])


def safetensors_plain(encoded, *, mismatch=None):
    header, data = unpack(encoded)
    header.pop("__metadata__")
    if mismatch:
        start, end = header["lm_head.weight"]["data_offsets"]
        data = data[:start] + mismatch(data[start:end]) + data[end:]
    return safetensors_wire(header, data)


def flip(raw):
    return bytes([raw[0] ^ 1]) + raw[1:]


class ModelMappingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = self.root / mapping.INDEX_NAME
        self.original = c1_bytes(m3_fixture())

    def cli(self, *args, success=True):
        result = subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True)
        self.assertEqual(result.returncode, 0 if success else 2, result.stderr.decode())
        return result

    def build(self, raw=None, **options):
        return mapping.build_sharded(
            self.original if raw is None else raw, options.pop("limits", safe.Limits()),
            **({"profile": mapping.M3, "scheme": "huggingface", "max_shard_bytes": 4200,
                "max_shards": mapping.MAX_SHARDS, "allow_oversize": False} | options))

    def install(self, files):
        for name, data in files.items():
            (self.root / name).write_bytes(data)

    def test_m3_exact_roster_and_all_bitwise_round_trips(self):
        manifest = json.loads((ROOT / "tests/m3/reference_manifest.json").read_text(encoding="utf-8"))
        manifest_shapes = {
            tuple(path.split("/")): tuple(shape)
            for path, shape in manifest["parameter_shapes"].items()
        }
        manifest_shapes[("token_embedding", "weight")] = manifest_shapes[("head", "weight")]
        self.assertEqual(
            {tuple(name.split(".")): shape for name, _, _, shape in M3_ROWS},
            manifest_shapes,
        )
        self.assertEqual(len(m3_fixture().tensors), 15)
        self.assertEqual(len(m3_fixture().tensors) - len(m3_fixture().aliases), 14)
        for format_name, scheme, column in (("safetensors", "huggingface", 1), ("gguf", "gguf", 2)):
            with self.subTest(scheme=scheme):
                encoded = mapping.encode_mapped(self.original, format_name, mapping.M3, scheme)
                if format_name == "safetensors":
                    header, _ = unpack(encoded)
                    metadata = header.pop("__metadata__")
                    names = set(header)
                else:
                    fields, rows, _, _ = unpack_export(encoded)
                    metadata = {key: value for key, _, value in fields}
                    names = {row[0] for row in rows}
                self.assertEqual(names, {row[column] for row in M3_ROWS})
                self.assertEqual(metadata[mapping.MAPPING_VERSION], "1.0")
                self.assertEqual(metadata[mapping.PROFILE], mapping.M3)
                self.assertEqual(metadata[mapping.SCHEME], scheme)
                self.assertEqual(mapping.decode_mapped(encoded, format_name), self.original)
        files = self.build()
        self.assertGreater(len(files), 2)
        self.assertTrue(all(len(data) <= 4200 for name, data in files.items() if name != mapping.INDEX_NAME))
        self.assertEqual(files, self.build())
        index = json.loads(files[mapping.INDEX_NAME])
        self.assertEqual(index["metadata"]["total_size"], (1184 + 1024) * 4)
        self.assertEqual(index["weight_map"].keys(), {row[1] for row in M3_ROWS})
        self.install(files)
        self.assertEqual(mapping.read_sharded(self.index), self.original)

    def test_decoder_multilayer_swiglu_rmsnorm_round_trips(self):
        state = decoder_fixture(tied=True)
        original = c1_bytes(state)
        for format_name, scheme, names in (
            ("safetensors", "huggingface", ("model.layers.2.self_attn.k_proj.bias", "model.layers.1.mlp.gate_proj.weight")),
            ("gguf", "gguf", ("blk.2.attn_k.bias", "blk.1.ffn_gate.weight")),
        ):
            with self.subTest(scheme=scheme):
                encoded = mapping.encode_mapped(original, format_name, mapping.DECODER, scheme)
                for name in names:
                    self.assertIn(name.encode(), encoded)
                self.assertEqual(mapping.decode_mapped(encoded, format_name, profile=mapping.DECODER), original)
        files = self.build(original, profile=mapping.DECODER, scheme="gguf", max_shard_bytes=900)
        self.install(files)
        self.assertEqual(mapping.read_sharded(self.index, scheme="gguf"), original)

    def test_mod5_multilayer_mqa_gqa_mha_mapped_and_sharded_round_trips(self):
        for kv_width in (2, 4, 8):
            for tied in (False, True):
                state, rows = mod5_fixture(kv_width, tied=tied)
                original = c1_bytes(state)
                for format_name, scheme, name_column in (
                    ("safetensors", "huggingface", 1), ("gguf", "gguf", 2),
                ):
                    with self.subTest(kv_width=kv_width, tied=tied, scheme=scheme):
                        encoded = mapping.encode_mapped(original, format_name, mapping.MOD5, scheme)
                        if format_name == "safetensors":
                            header, _ = unpack(encoded)
                            metadata = header.pop("__metadata__")
                            actual = {name: tuple(entry["shape"]) for name, entry in header.items()}
                        else:
                            fields, infos, _, _ = unpack_export(encoded)
                            metadata = {key: value for key, _, value in fields}
                            actual = {name: dims[::-1] for name, dims, _, _, _ in infos}
                        self.assertEqual(actual, {row[name_column]: row[3] for row in rows})
                        self.assertEqual(metadata[mapping.PROFILE], mapping.MOD5)
                        self.assertEqual(mapping.decode_mapped(encoded, format_name), original)
                        files = self.build(original, profile=mapping.MOD5, scheme=scheme, max_shard_bytes=900)
                        self.assertGreater(len(files), 2)
                        self.install(files)
                        self.assertEqual(mapping.read_sharded(self.index, profile=mapping.MOD5,
                                                              scheme=scheme), original)

    def test_mod5_plain_external_import_and_exact_tie(self):
        for tied in (False, True):
            state, rows = mod5_fixture(4, tied=tied)
            for format_name, scheme, name_column, plain in (
                ("safetensors", "huggingface", 1, safetensors_plain),
                ("gguf", "gguf", 2, gguf_plain),
            ):
                with self.subTest(tied=tied, scheme=scheme):
                    options = dict(external=True, profile=mapping.MOD5, scheme=scheme, provider=PROVIDER)
                    original = c1_bytes(state)
                    encoded = mapping.encode_mapped(original, format_name, mapping.MOD5, scheme)
                    imported = mapping.decode_mapped(plain(encoded), format_name, tie=tied, **options)
                    self.assertEqual(imported, original)
                    if tied:
                        self.assertEqual(mapping.decode_mapped(plain(encoded), format_name, **options),
                                         c1_bytes(replace(state, aliases=())))
                        with self.assertRaisesRegex(safe.InteropError, "equal shapes and exact bits"):
                            mapping.decode_mapped(plain(encoded, mismatch=flip), format_name, tie=True, **options)
                        shape_mismatch = replace(state.tensors[-3], shape=(8, 17))
                        unexpected = replace(state, tensors=state.tensors[:-3] + (shape_mismatch,) + state.tensors[-2:],
                                             aliases=())
                        external = mapping.encode_mapped(c1_bytes(unexpected), format_name, mapping.MOD5, scheme)
                        with self.assertRaisesRegex(safe.InteropError, "equal shapes and exact bits"):
                            mapping.decode_mapped(plain(external), format_name, tie=True, **options)

    def test_mod5_rejects_incompatible_rosters_on_export_and_external_import(self):
        fixture, _ = mod5_fixture(2, tied=False)
        tensors = fixture.tensors
        for label, invalid in (
            ("final norm bias", tensors + (tensor("norm_final.bias", (8,)),)),
            ("norm1 bias", tensors + (tensor("blocks.0.norm1.bias", (8,)),)),
            ("norm2 bias", tensors + (tensor("blocks.0.norm2.bias", (8,)),)),
            ("attention bias", tensors + (tensor("blocks.0.attention.query.bias", (8,)),)),
            ("SwiGLU bias", tensors + (tensor("blocks.0.ffn.gate.bias", (12,)),)),
            ("learned positions", tensors + (tensor("position_embedding.weight", (2, 8)),)),
            ("missing gate", tuple(t for t in tensors if t.path != ("blocks", "1", "ffn", "gate", "weight"))),
            ("no layers", tuple(t for t in tensors if t.path[0] != "blocks")),
            ("missing layer 0", tuple(t for t in tensors if t.path[:2] != ("blocks", "0"))),
            ("missing root", tuple(t for t in tensors if t.path[0] != "norm_final")),
            ("wrong norm rank", tuple(replace(t, shape=(1, 8)) if t.path == ("norm_final", "weight") else t
                                     for t in tensors)),
        ):
            raw = c1_bytes(replace(fixture, tensors=tuple(sorted(invalid, key=lambda t: t.path))))
            for format_name, scheme, plain in (
                ("safetensors", "huggingface", safetensors_plain), ("gguf", "gguf", gguf_plain),
            ):
                with self.subTest(case=label, scheme=scheme):
                    with self.assertRaises(safe.InteropError):
                        mapping.encode_mapped(raw, format_name, mapping.MOD5, scheme)
                    if label in ("missing layer 0", "missing root", "wrong norm rank"):
                        continue  # These rosters also fail legacy decoder export.
                    encoded = mapping.encode_mapped(raw, format_name, mapping.DECODER, scheme)
                    with self.assertRaises(safe.InteropError):
                        mapping.decode_mapped(plain(encoded), format_name, external=True,
                                              profile=mapping.MOD5, scheme=scheme, provider=PROVIDER)

    def test_mod5_import_rejects_missing_extra_biased_wrong_rank_and_corrupted(self):
        state, _ = mod5_fixture(2, tied=False)
        original = c1_bytes(state)

        def rebuild_safetensors(header, data):
            entries = {k: v for k, v in header.items() if k != "__metadata__"}
            payload = bytearray()
            new_header = {}
            if "__metadata__" in header:
                new_header["__metadata__"] = dict(header["__metadata__"])
            for name in sorted(entries):
                entry = entries[name]
                start, end = entry["data_offsets"]
                chunk = data[start:end]
                offset = len(payload)
                payload.extend(chunk)
                new_header[name] = {
                    "dtype": entry["dtype"],
                    "shape": list(entry["shape"]),
                    "data_offsets": [offset, len(payload)],
                }
            return safetensors_wire(new_header, bytes(payload))

        def mutate_safetensors(raw_bytes, case):
            header, data = unpack(raw_bytes)
            if case == "missing_root":
                header.pop("model.norm.weight")
            elif case == "missing_block":
                header.pop("model.layers.1.mlp.gate_proj.weight")
            elif case == "extra":
                end = len(data)
                data += struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
                header["model.extra.weight"] = {"dtype": "F32", "shape": [4], "data_offsets": [end, len(data)]}
            elif case == "biased":
                end = len(data)
                data += struct.pack("<8f", *([0.5] * 8))
                header["model.norm.bias"] = {"dtype": "F32", "shape": [8], "data_offsets": [end, len(data)]}
            elif case == "wrong_rank":
                header["model.norm.weight"]["shape"] = [1, 8]
            elif case == "corrupted":
                return raw_bytes[:-1] + bytes([raw_bytes[-1] ^ 0xFF])
            return rebuild_safetensors(header, data)

        def mutate_gguf(raw_bytes, case):
            if case == "corrupted":
                return raw_bytes[:-4] + bytes([raw_bytes[-4] ^ 0xFF]) + raw_bytes[-3:]
            fields, rows, _, _ = unpack_export(raw_bytes)
            mutated = list(rows)
            if case == "missing_root":
                mutated = [row for row in mutated if row[0] != "output_norm.weight"]
            elif case == "missing_block":
                mutated = [row for row in mutated if row[0] != "blk.1.ffn_gate.weight"]
            elif case == "extra":
                mutated.append(("extra.weight", (4,), 0, 0, struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)))
            elif case == "biased":
                mutated.append(("output_norm.bias", (8,), 0, 0, struct.pack("<8f", *([0.5] * 8))))
            elif case == "wrong_rank":
                mutated = [
                    (name, (8, 1), kind, offset, payload) if name == "output_norm.weight"
                    else (name, shape, kind, offset, payload)
                    for name, shape, kind, offset, payload in mutated
                ]
            repacked = []
            offset = 0
            for name, shape, kind, _, payload in mutated:
                repacked.append((name, shape, kind, offset, payload))
                offset += len(payload) + ((-len(payload)) % 32)
            return wire(metadata=fields, tensors=repacked)

        def mutate_sharded(files, case):
            mutated = dict(files)
            index = json.loads(mutated[mapping.INDEX_NAME])
            weight_map = dict(index["weight_map"])
            checksums = dict(index["metadata"][mapping.SHARD_SHA256])
            if case == "corrupted":
                shard_name = next(iter(checksums))
                raw = mutated[shard_name]
                mutated[shard_name] = raw[:-1] + bytes([raw[-1] ^ 0xFF])
                return mutated
            if case == "missing_root":
                target = "model.norm.weight"
            elif case == "missing_block":
                target = "model.layers.1.mlp.gate_proj.weight"
            elif case == "wrong_rank":
                target = "model.norm.weight"
            else:
                target = next(iter(weight_map))
            shard_name = weight_map[target]
            header, data = unpack(mutated[shard_name])
            if case in ("missing_root", "missing_block"):
                entry = header.pop(target)
                weight_map.pop(target)
                start, end = entry["data_offsets"]
                index["metadata"]["total_size"] -= end - start
            elif case == "extra":
                end = len(data)
                data += struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
                header["model.extra.weight"] = {"dtype": "F32", "shape": [4], "data_offsets": [end, len(data)]}
                weight_map["model.extra.weight"] = shard_name
                index["metadata"]["total_size"] += 16
            elif case == "biased":
                end = len(data)
                data += struct.pack("<8f", *([0.5] * 8))
                header["model.norm.bias"] = {"dtype": "F32", "shape": [8], "data_offsets": [end, len(data)]}
                weight_map["model.norm.bias"] = shard_name
                index["metadata"]["total_size"] += 32
            elif case == "wrong_rank":
                header[target]["shape"] = [1, 8]
            if any(k != "__metadata__" for k in header):
                rebuilt = rebuild_safetensors(header, data)
                mutated[shard_name] = rebuilt
                checksums[shard_name] = hashlib.sha256(rebuilt).hexdigest()
            else:
                mutated.pop(shard_name)
                checksums.pop(shard_name)
            index["weight_map"] = weight_map
            index["metadata"][mapping.SHARD_SHA256] = checksums
            mutated[mapping.INDEX_NAME] = json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
            return mutated

        def signed_invalid_sharded(files, case):
            mutated = mutate_sharded(files, case)
            index = json.loads(mutated[mapping.INDEX_NAME])
            base_tensors = list(state.tensors)
            if case == "missing_root":
                bad_tensors = [t for t in base_tensors if t.path != ("norm_final", "weight")]
            elif case == "missing_block":
                bad_tensors = [t for t in base_tensors if t.path != ("blocks", "1", "ffn", "gate", "weight")]
            elif case == "extra":
                bad_tensors = sorted(
                    base_tensors + [safe.Tensor(("extra", "weight"), "parameter", (4,),
                                                struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))],
                    key=lambda t: t.path,
                )
            elif case == "biased":
                bad_tensors = sorted(
                    base_tensors + [safe.Tensor(("norm_final", "bias"), "parameter", (8,),
                                                struct.pack("<8f", *([0.5] * 8)))],
                    key=lambda t: t.path,
                )
            else:
                bad_tensors = [replace(t, shape=(1, 8)) if t.path == ("norm_final", "weight") else t
                               for t in base_tensors]
            bad_state = replace(state, tensors=tuple(bad_tensors), aliases=())
            bad_c1 = c1_bytes(bad_state)
            path_to_ext = {
                tuple(row[0].split(".")): row[1]
                for row in mod5_fixture(2, tied=False)[1]
            }
            path_to_ext[("extra", "weight")] = "model.extra.weight"
            path_to_ext[("norm_final", "bias")] = "model.norm.bias"
            ext_names = tuple(path_to_ext[t.path] for t in bad_state.tensors)
            index["metadata"][safe.MANIFEST_KEY] = safe.make_manifest(bad_state, ext_names)
            index["metadata"][safe.DIGEST_KEY] = hashlib.sha256(bad_c1).hexdigest()
            mutated[mapping.INDEX_NAME] = json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
            return mutated

        expected_plain_regex = {
            "missing_root": "missing MOD5 root tensor",
            "missing_block": "missing MOD5 block tensor",
            "extra": "unmapped external tensor name",
            "biased": "unmapped external tensor name",
            "wrong_rank": "decoder tensor rank/extent mismatch",
        }
        expected_roster_regex = {
            "missing_root": "missing MOD5 root tensor",
            "missing_block": "missing MOD5 block tensor",
            "extra": "unknown canonical path",
            "biased": "unknown canonical path",
            "wrong_rank": "decoder tensor rank/extent mismatch",
        }
        cases = ("missing_root", "missing_block", "extra", "biased", "wrong_rank", "corrupted")
        st_encoded = mapping.encode_mapped(original, "safetensors", mapping.MOD5, "huggingface")
        gguf_encoded = mapping.encode_mapped(original, "gguf", mapping.MOD5, "gguf")
        sharded_files = self.build(original, profile=mapping.MOD5, scheme="huggingface", max_shard_bytes=900)

        for case in cases:
            with self.subTest(route="safetensors-mapped", case=case):
                mutated = mutate_safetensors(st_encoded, case)
                with self.assertRaises(safe.InteropError) as ctx:
                    mapping.decode_mapped(mutated, "safetensors", profile=mapping.MOD5, scheme="huggingface")
                if case != "corrupted":
                    self.assertNotIn("noncontiguous", str(ctx.exception))
                    with self.assertRaisesRegex(safe.InteropError, expected_plain_regex[case]):
                        mapping.decode_mapped(safetensors_plain(mutated), "safetensors", external=True,
                                              profile=mapping.MOD5, scheme="huggingface", provider=PROVIDER)
            with self.subTest(route="gguf-mapped", case=case):
                mutated = mutate_gguf(gguf_encoded, case)
                with self.assertRaises(safe.InteropError) as ctx:
                    mapping.decode_mapped(mutated, "gguf", profile=mapping.MOD5, scheme="gguf")
                self.assertNotIn("noncontiguous", str(ctx.exception))
                if case != "corrupted":
                    with self.assertRaisesRegex(safe.InteropError, expected_plain_regex[case]):
                        mapping.decode_mapped(gguf_plain(mutated), "gguf", external=True,
                                              profile=mapping.MOD5, scheme="gguf", provider=PROVIDER)
            with self.subTest(route="sharded-safetensors", case=case):
                mutated_dir = self.root / f"mod5_bad_{case}"
                mutated_dir.mkdir()
                for name, blob in mutate_sharded(sharded_files, case).items():
                    (mutated_dir / name).write_bytes(blob)
                with self.assertRaises(safe.InteropError) as ctx:
                    mapping.read_sharded(mutated_dir / mapping.INDEX_NAME, profile=mapping.MOD5, scheme="huggingface")
                if case != "corrupted":
                    self.assertNotIn("coverage", str(ctx.exception))
                    self.assertNotIn("checksum", str(ctx.exception))
                    signed_dir = self.root / f"mod5_signed_bad_{case}"
                    signed_dir.mkdir()
                    for name, blob in signed_invalid_sharded(sharded_files, case).items():
                        (signed_dir / name).write_bytes(blob)
                    with self.assertRaisesRegex(safe.InteropError, expected_roster_regex[case]):
                        mapping.read_sharded(signed_dir / mapping.INDEX_NAME, profile=mapping.MOD5, scheme="huggingface")

    def test_unmapped_shards_preserve_arbitrary_paths_kinds_and_empty_state(self):
        for state in (safe.State("utf8 λ", ()),
                      safe.State("utf8 λ", (safe.Tensor(("a.b", "λ"), "buffer", (0,), b""),
                                            tensor("c", (1,))))):
            raw = c1_bytes(state)
            bundle = self.build(raw, profile=None, scheme=None)
            index = json.loads(bundle[mapping.INDEX_NAME])
            self.assertNotIn(mapping.MAPPING_VERSION, index["metadata"])
            self.assertEqual(list(index["weight_map"]), [f"t{i:06d}" for i in range(len(state.tensors))])
            self.install(bundle)
            self.assertEqual(mapping.read_sharded(self.index), raw)

    def test_external_formats_require_explicit_selection_and_validate_ties(self):
        untied, tied = (c1_bytes(decoder_fixture(tied=choice)) for choice in (False, True))
        for format_name, scheme, plain in (("safetensors", "huggingface", safetensors_plain),
                                            ("gguf", "gguf", gguf_plain)):
            with self.subTest(scheme=scheme):
                mapped = mapping.encode_mapped(untied, format_name, mapping.DECODER, scheme)
                options = dict(external=True, provider=PROVIDER, profile=mapping.DECODER, scheme=scheme)
                self.assertEqual(mapping.decode_mapped(plain(mapped), format_name, **options), untied)
                self.assertEqual(mapping.decode_mapped(plain(mapped), format_name, **options, tie=True), tied)
                with self.assertRaisesRegex(safe.InteropError, "exact bits"):
                    mapping.decode_mapped(plain(mapped, mismatch=flip), format_name, **options, tie=True)
                with self.assertRaisesRegex(safe.InteropError, "external import cannot override"):
                    mapping.decode_mapped(mapped, format_name, **options)
                for key in ("profile", "scheme", "provider"):
                    with self.assertRaises(safe.InteropError):
                        mapping.decode_mapped(plain(mapped), format_name, **(options | {key: None}))
                with self.assertRaisesRegex(safe.InteropError, "metadata fields"):
                    mapping.decode_mapped(plain(mapped), format_name)
                m3_mapped = mapping.encode_mapped(self.original, format_name, mapping.M3, scheme)
                m3_options = options | {"profile": mapping.M3}
                self.assertEqual(mapping.decode_mapped(plain(m3_mapped), format_name, **m3_options), self.original)
                with self.assertRaisesRegex(safe.InteropError, "exact bits"):
                    mapping.decode_mapped(plain(m3_mapped, mismatch=flip), format_name, **m3_options)

    def test_invalid_profiles_paths_shapes_and_layer_rosters(self):
        m3 = m3_fixture()
        decoder = decoder_fixture()
        bad_states = (
            (replace(m3, tensors=(tensor("blocks.0.attention.key.weight", (2, 8)),) + m3.tensors[1:]), mapping.M3),
            (replace(m3, tensors=m3.tensors[:-2] + m3.tensors[-1:], aliases=((10, 13),)), mapping.M3),
            (replace(m3, aliases=()), mapping.M3),
            (replace(m3, tensors=(replace(m3.tensors[0], kind="buffer"),) + m3.tensors[1:]), mapping.M3),
            (replace(decoder, tensors=decoder.tensors[11:]), mapping.DECODER),
            (replace(decoder, tensors=tuple(t for t in decoder.tensors if t.path[-2:] != ("gate", "weight"))),
             mapping.DECODER),
            (replace(decoder, tensors=tuple(t for t in decoder.tensors if t.path != ("blocks", "1", "norm1", "weight"))),
             mapping.DECODER),
            (replace(decoder, tensors=(tensor("!", (2,)),) + decoder.tensors), mapping.DECODER),
            (replace(decoder, tensors=(replace(decoder.tensors[0], path=("blocks", "00") + decoder.tensors[0].path[2:]),)
                     + decoder.tensors[1:]), mapping.DECODER),
        )
        for index, (state, profile) in enumerate(bad_states):
            with self.subTest(case=index):
                with self.assertRaises(safe.InteropError):
                    mapping.encode_mapped(c1_bytes(state), "safetensors", profile, "huggingface")

    def test_metadata_names_are_bound_to_paths_and_reject_coincidences(self):
        mapped = mapping.encode_mapped(self.original, "safetensors", mapping.M3, "huggingface")
        header, data = unpack(mapped)
        metadata = header["__metadata__"]
        manifest = json.loads(metadata[safe.MANIFEST_KEY])
        manifest["entries"][0]["name"], manifest["entries"][1]["name"] = (
            manifest["entries"][1]["name"], manifest["entries"][0]["name"])
        metadata[safe.MANIFEST_KEY] = safe.json_bytes(manifest).decode()
        with self.assertRaisesRegex(safe.InteropError, "names do not match"):
            mapping.decode_mapped(safetensors_wire(header, data), "safetensors")
        manifest["entries"][0]["name"] = manifest["entries"][1]["name"]
        metadata[safe.MANIFEST_KEY] = safe.json_bytes(manifest).decode()
        with self.assertRaisesRegex(safe.InteropError, "IO5 tensor name"):
            mapping.decode_mapped(safetensors_wire(header, data), "safetensors")

        header, data = unpack(safetensors_plain(mapped))
        name = "model.layers.0.self_attn.k_proj.weight"
        header["model.layers.00.self_attn.k_proj.weight"] = header.pop(name)
        options = dict(external=True, provider=PROVIDER, profile=mapping.M3, scheme="huggingface")
        with self.assertRaisesRegex(safe.InteropError, "layer index spelling"):
            mapping.decode_mapped(safetensors_wire(header, data), "safetensors", **options)
        header, data = unpack(safetensors_plain(mapped))
        header["unmapped"] = header.pop(name)
        with self.assertRaisesRegex(safe.InteropError, "unmapped external"):
            mapping.decode_mapped(safetensors_wire(header, data), "safetensors", **options)
        repeated = b'{"duplicate":0,"duplicate":1}'
        with self.assertRaisesRegex(safe.InteropError, "duplicate JSON key"):
            mapping.decode_mapped(struct.pack("<Q", len(repeated)) + repeated, "safetensors", **options)
        for bad_format in ("", "pickle", "GGUF", "safetensor"):
            with self.subTest(bad_format=bad_format):
                with self.assertRaisesRegex(safe.InteropError, "unsupported wire format"):
                    mapping.encode_mapped(self.original, bad_format, mapping.M3, "huggingface")
                with self.assertRaisesRegex(safe.InteropError, "unsupported wire format"):
                    mapping.decode_mapped(mapped, bad_format)

    def test_invalid_index_schema_json_paths_and_checksums(self):
        files = self.build()
        self.install(files)
        for corrupted in (b"{", b'{"metadata":{},"metadata":{},"weight_map":{}}',
                          b'{"metadata":{},"weight_map":[],"extra":0}', b"[" * 17 + b"]" * 17,
                          b'{"metadata":{"duplicate":0,"duplicate":1},"weight_map":{}}'):
            self.index.write_bytes(corrupted)
            with self.assertRaises(safe.InteropError):
                mapping.read_sharded(self.index)
        for invalid in ("../evil.safetensors", "/abs/path", "foo/bar.safetensors", "a..b.safetensors",
                        "foo\\bar.safetensors", "bad\0.safetensors"):
            with self.subTest(shard=invalid):
                index = json.loads(files[mapping.INDEX_NAME])
                hashes = index["metadata"][mapping.SHARD_SHA256]
                shard = next(iter(hashes))
                hashes[invalid] = hashes.pop(shard)
                index["weight_map"] = {k: invalid if v == shard else v for k, v in index["weight_map"].items()}
                self.index.write_bytes(safe.json_bytes(index))
                with self.assertRaisesRegex(safe.InteropError, "invalid shard basename"):
                    mapping.read_sharded(self.index)
                with self.assertRaisesRegex(safe.InteropError, "invalid shard basename"):
                    mapping.publish_sharded(self.index, {invalid: b"x", mapping.INDEX_NAME: b"{}"}, overwrite=False)
        with self.assertRaisesRegex(safe.InteropError, "published last"):
            mapping.publish_sharded(self.index, {}, overwrite=False)
        for field, bad, expected in ((safe.DIGEST_KEY, "0" * 64, "C1 checksum mismatch"),
                                     (safe.DIGEST_KEY, "A" * 64, "invalid IO5 C1 checksum"),
                                     ("total_size", True, "u64"),
                                     ("total_size", 0, "total_size mismatch"),
                                     (mapping.SHARDED_VERSION, "2.0", "unsupported IO5 sharded version"),
                                     (mapping.SCHEME, "gguf", "tensor names do not match")):
            with self.subTest(field=field, bad=bad):
                index = json.loads(files[mapping.INDEX_NAME])
                index["metadata"][field] = bad
                self.index.write_bytes(safe.json_bytes(index))
                with self.assertRaisesRegex(safe.InteropError, expected):
                    mapping.read_sharded(self.index)

    def test_shard_checksums_symlinks_missing_files_and_disjoint_coverage(self):
        files = self.build()
        self.install(files)
        shard = next(iter(files))
        path = self.root / shard
        path.write_bytes(files[shard][:-1] + flip(files[shard][-1:]))
        with self.assertRaisesRegex(safe.InteropError, "shard SHA-256 mismatch"):
            mapping.read_sharded(self.index)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            mapping.read_sharded(self.index)
        target = self.root / "original"
        target.write_bytes(files[shard])
        path.symlink_to(target)
        with self.assertRaises(OSError):
            mapping.read_sharded(self.index)
        with self.assertRaisesRegex(safe.InteropError, "nonsymlink"):
            mapping.publish_sharded(self.index, files, overwrite=True)
        path.unlink()
        path.write_bytes(files[shard])
        index = json.loads(files[mapping.INDEX_NAME])
        index["metadata"][mapping.SHARD_SHA256][shard] = "0" * 64
        self.index.write_bytes(safe.json_bytes(index))
        with self.assertRaisesRegex(safe.InteropError, "shard SHA-256 mismatch"):
            mapping.read_sharded(self.index)
        for operation in ("extra", "missing", "duplicate"):
            with self.subTest(operation=operation):
                header, data = unpack(files[shard])
                name, desc = next(iter(header.items()))
                if operation == "extra":
                    header["extra"] = {"dtype": "F32", "shape": [1], "data_offsets": [len(data), len(data) + 4]}
                    changed = safetensors_wire(header, data + bytes(4))
                elif operation == "missing":
                    changed = safetensors_wire({})
                else:
                    key = next(k for k, v in index["weight_map"].items() if v != shard)
                    header[key] = header.pop(name)
                    changed = safetensors_wire(header, data)
                path.write_bytes(changed)
                index["metadata"][mapping.SHARD_SHA256][shard] = hashlib.sha256(changed).hexdigest()
                self.index.write_bytes(safe.json_bytes(index))
                with self.assertRaisesRegex(safe.InteropError, "weight_map coverage mismatch"):
                    mapping.read_sharded(self.index)
        self.install(files)
        target_index = self.root / "saved-index"
        self.index.rename(target_index)
        self.index.symlink_to(target_index)
        with self.assertRaises(OSError):
            mapping.read_sharded(self.index)

    def test_exact_shard_budget_count_and_oversize_singletons(self):
        raw = c1_bytes(safe.State("inert", (tensor("a", (128,)),)))
        bundle = self.build(raw, profile=None, scheme=None)
        shard, shard_bytes = next(iter(bundle.items()))
        count = len(shard_bytes)
        self.assertEqual(bundle, self.build(raw, profile=None, scheme=None, max_shard_bytes=count, max_shards=1))
        with self.assertRaisesRegex(safe.InteropError, "single tensor exceeds"):
            self.build(raw, profile=None, scheme=None, max_shard_bytes=count - 1)
        self.assertEqual(bundle, self.build(raw, profile=None, scheme=None, max_shard_bytes=count - 1,
                                           allow_oversize=True))
        for options in ({"max_shards": 0}, {"max_shards": mapping.MAX_SHARDS + 1},
                        {"max_shard_bytes": 0}, {"max_shard_bytes": safe.Limits().file + 1},
                        {"limits": safe.Limits(tensor=511), "allow_oversize": True}):
            with self.subTest(options=options), self.assertRaises(safe.InteropError):
                self.build(raw, profile=None, scheme=None, **options)
        self.install(bundle)
        with self.assertRaisesRegex(safe.InteropError, "shard count limit"):
            mapping.read_sharded(self.index, max_shards=0)
        # A peer can repad files and update hashes. The aggregate cap still applies.
        small = c1_bytes(safe.State("inert", tuple(tensor(f"a{i}", (1,)) for i in range(8))))
        small_bundle = self.build(small, profile=None, scheme=None, max_shard_bytes=80)
        index = json.loads(small_bundle.pop(mapping.INDEX_NAME))
        for name, data in small_bundle.items():
            length = struct.unpack_from("<Q", data)[0]
            pad = 800
            changed = struct.pack("<Q", length + pad) + data[8:8 + length] + b" " * pad + data[8 + length:]
            small_bundle[name] = changed
            index["metadata"][mapping.SHARD_SHA256][name] = hashlib.sha256(changed).hexdigest()
        small_bundle[mapping.INDEX_NAME] = safe.json_bytes(index)
        limits = safe.Limits(file=3000, metadata=2500)
        self.assertLess(len(small_bundle[mapping.INDEX_NAME]), limits.metadata)
        self.assertTrue(all(len(data) < limits.file for data in small_bundle.values()))
        self.assertGreater(sum(len(data) for name, data in small_bundle.items() if name != mapping.INDEX_NAME),
                           limits.file + limits.metadata)
        self.install(small_bundle)
        with self.assertRaisesRegex(safe.InteropError, "size policy"):
            mapping.read_sharded(self.index, limits)

    def test_cli_atomic_publication_modes_and_failed_overwrite(self):
        source, result = self.root / "input.c1", self.root / "restored.c1"
        source.write_bytes(self.original)
        options = ("--profile", mapping.M3, "--scheme", "huggingface")
        inspect = self.cli("inspect-mapping", source, *options)
        self.assertEqual(len(json.loads(inspect.stdout)[safe.MANIFEST_KEY]["entries"]), 15)
        for format_name, scheme in (("safetensors", "huggingface"), ("gguf", "gguf")):
            exported = self.root / f"output.{format_name}"
            self.cli(f"c1-to-mapped-{format_name}", source, exported, "--profile", mapping.M3)
            self.cli(f"mapped-{format_name}-to-c1", exported, result, "--scheme", scheme)
            self.assertEqual(result.read_bytes(), self.original)
            self.cli(f"mapped-{format_name}-to-c1", exported, result, success=False)
            result.unlink()
            plain = safetensors_plain if format_name == "safetensors" else gguf_plain
            exported.write_bytes(plain(exported.read_bytes()))
            self.cli(f"mapped-{format_name}-to-c1", exported, result, "--external", "--provider", PROVIDER,
                     "--tie-head-to-embedding", "--scheme", scheme, "--profile", mapping.M3)
            self.assertEqual(result.read_bytes(), self.original)
            result.unlink()

        command = ("c1-to-sharded-safetensors", source, self.index, *options, "--max-shard-bytes", 4200)
        shard_name = next(iter(self.build()))
        blocker = self.root / shard_name
        blocker.write_bytes(b"existing")
        self.cli(*command, success=False)
        self.assertFalse(self.index.exists())
        self.assertEqual(blocker.read_bytes(), b"existing")
        blocker.unlink()
        self.cli(*command)
        self.cli(*command, success=False)
        for path in (self.index, *self.root.glob("*.safetensors")):
            if path.name.startswith("model"):
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.cli("sharded-safetensors-to-c1", self.index, result)
        self.assertEqual(result.read_bytes(), self.original)
        self.assertEqual(result.stat().st_mode & 0o777, 0o600)
        original_files = {path.name: path.read_bytes() for path in self.root.glob("model*")}
        state = m3_fixture()
        updated = c1_bytes(replace(state, tensors=(replace(state.tensors[0], data=flip(state.tensors[0].data)),)
                                   + state.tensors[1:]))
        incoming = self.build(updated)
        self.assertTrue(any(incoming[name] != original_files[name]
                            for name in incoming if name != mapping.INDEX_NAME))
        real_replace = mapping.os.replace

        def fail_index(current, name, **kwargs):
            if name == mapping.INDEX_NAME:
                raise OSError("injected index replacement")
            real_replace(current, name, **kwargs)

        with patch.object(mapping.os, "replace", side_effect=fail_index):
            with self.assertRaisesRegex(OSError, "injected"):
                mapping.publish_sharded(self.index, incoming, overwrite=True)
        self.assertEqual({p.name: p.read_bytes() for p in self.root.glob("model*")}, original_files)
        self.assertFalse(list(self.root.glob(".io5-*")))
        source.write_bytes(updated)
        self.cli(*command, "--overwrite")
        self.assertEqual(mapping.read_sharded(self.index), updated)
        self.cli("sharded-safetensors-to-c1", self.index, result, success=False)
        self.assertEqual(result.read_bytes(), self.original)
        self.index.write_bytes(b"broken")
        self.cli("sharded-safetensors-to-c1", self.index, result, "--overwrite", success=False)
        self.assertEqual(result.read_bytes(), self.original)

    @unittest.skipUnless(importlib.util.find_spec("safetensors"), "optional independent safetensors package absent")
    def test_independent_safetensors_reference(self):
        import safetensors

        self.assertEqual(safetensors.__version__, "0.6.2")
        expected = {row[1]: tensor for row, tensor in zip(M3_ROWS, m3_fixture().tensors, strict=True)}
        encoded = mapping.encode_mapped(self.original, "safetensors", mapping.M3, "huggingface")
        tensors = dict(safetensors.deserialize(encoded))
        self.assertEqual(tensors.keys(), expected.keys())
        for name, description in tensors.items():
            self.assertEqual(description["shape"], list(expected[name].shape))
            self.assertEqual(bytes(description["data"]), expected[name].data)
        plain = safetensors.serialize({name: {"dtype": "float32", "shape": entry["shape"],
                                              "data": bytes(entry["data"])}
                                       for name, entry in tensors.items()})
        self.assertEqual(mapping.decode_mapped(plain, "safetensors", external=True, provider=PROVIDER,
                                               profile=mapping.M3, scheme="huggingface"), self.original)
        files = self.build()
        index = json.loads(files.pop(mapping.INDEX_NAME))
        for name, raw in files.items():
            self.assertEqual(dict(safetensors.deserialize(raw)).keys(),
                             {key for key, shard in index["weight_map"].items() if shard == name})

    @unittest.skipUnless(importlib.util.find_spec("gguf"), "optional independent gguf package absent")
    def test_independent_gguf_reference(self):
        import gguf
        import numpy as np

        self.assertEqual(importlib.metadata.version("gguf"), "0.19.0")
        self.assertEqual(np.__version__, "2.5.4")
        expected = {row[2]: tensor for row, tensor in zip(M3_ROWS, m3_fixture().tensors, strict=True)}
        encoded = mapping.encode_mapped(self.original, "gguf", mapping.M3, "gguf")
        source, output = self.root / "mapped.gguf", self.root / "external.gguf"
        source.write_bytes(encoded)
        reader = gguf.GGUFReader(source)
        self.assertEqual({item.name for item in reader.tensors}, expected.keys())
        writer = gguf.GGUFWriter(output, "external")
        for item in reader.tensors:
            self.assertEqual(item.data.shape, expected[item.name].shape)
            self.assertEqual(item.data.view(np.uint32).tobytes(), expected[item.name].data)
            writer.add_tensor(item.name, item.data)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        self.assertEqual(mapping.decode_mapped(output.read_bytes(), "gguf", external=True, provider=PROVIDER,
                                               profile=mapping.M3, scheme="gguf"), self.original)


if __name__ == "__main__":
    unittest.main()
