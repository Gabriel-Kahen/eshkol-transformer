from __future__ import annotations

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
sys.path[:0] = [str(ROOT / part) for part in ("tools/io5", "tests/io5", "tests/c1")]

import gguf_interop as gg  # noqa: E402
import model_mapping as mapping  # noqa: E402
import quantized_interchange as qi  # noqa: E402
import safetensors_interop as safe  # noqa: E402
from test_checkpoint_format import build_container, parse_golden  # noqa: E402
from test_model_mapping import M3_ROWS, MOD5_BLOCK_ROWS, MOD5_ROOT_ROWS, PROVIDER  # noqa: E402
from test_safetensors_interop import safe as safetensors_wire, unpack as unpack_safetensors  # noqa: E402

TOOL = ROOT / "tools/io5/quantized_interchange.py"


def deterministic_f32_bytes(count: int, seed: int, scale: float = 1.75) -> bytes:
    """Generate deterministic non-dyadic finite f32 weights with varied dynamic range."""
    vals = []
    for i in range(count):
        # Deterministic trigonometric mix so F16/BF16/Q8_0 exhibit real lossy rounding.
        angle = (i + 1) * 0.37129 + (seed + 1) * 1.137
        v = math.sin(angle) * scale + 0.13 * math.cos(angle * 2.3)
        vals.append(v)
    return struct.pack(f"<{count}f", *vals) if count else b""


def finite_tensor(name: str, shape: tuple[int, ...], seed: int, scale: float = 1.75) -> safe.Tensor:
    count = safe.elements(shape)
    return safe.Tensor(tuple(name.split(".")), "parameter", shape, deterministic_f32_bytes(count, seed, scale))


def m3_finite_state() -> safe.State:
    tensors = [
        finite_tensor(name, shape, idx, scale=1.5 if len(shape) == 2 else 0.75)
        for idx, (name, _, _, shape) in enumerate(M3_ROWS)
    ]
    # Preserve required head.weight (10) <-> token_embedding.weight (14) alias.
    tensors[14] = replace(tensors[14], data=tensors[10].data)
    return safe.State(PROVIDER, tuple(tensors), ((10, 14),))


def mod5_finite_state(*, hidden: int = 32, kv_width: int = 16, inter: int = 64,
                      vocab: int = 32, layers: int = 2, tied: bool = True) -> safe.State:
    """MOD5 fixture where all 2D weight tensors have GGML ne[0] (C1 shape[-1]) divisible by 32."""
    tensors: list[safe.Tensor] = []
    seed = 100
    for layer in range(layers):
        layer_shapes = {
            "norm1.weight": (hidden,),
            "norm2.weight": (hidden,),
            "attention.query.weight": (hidden, hidden),
            "attention.key.weight": (kv_width, hidden),
            "attention.value.weight": (kv_width, hidden),
            "attention.output.weight": (hidden, hidden),
            "ffn.up.weight": (inter, hidden),
            "ffn.down.weight": (hidden, inter),
            "ffn.gate.weight": (inter, hidden),
        }
        for suffix, shape in layer_shapes.items():
            tensors.append(finite_tensor(f"blocks.{layer}.{suffix}", shape, seed))
            seed += 1
    tensors.append(finite_tensor("token_embedding.weight", (vocab, hidden), seed))
    seed += 1
    tensors.append(finite_tensor("norm_final.weight", (hidden,), seed))
    seed += 1
    tensors.append(finite_tensor("head.weight", (vocab, hidden), seed))
    tensors.sort(key=lambda t: t.path)
    paths = [t.path for t in tensors]
    aliases: tuple[tuple[int, ...], ...] = ()
    if tied:
        h_idx = paths.index(("head", "weight"))
        e_idx = paths.index(("token_embedding", "weight"))
        tensors[e_idx] = replace(tensors[e_idx], data=tensors[h_idx].data)
        aliases = ((min(h_idx, e_idx), max(h_idx, e_idx)),)
    return safe.State(PROVIDER, tuple(tensors), aliases)


def decoder_finite_state() -> safe.State:
    """Decoder fixture mixing ne[0] % 32 == 0 (Q8_0) and prod % 32 == 0 with ne[0] % 32 != 0 (ffn.down (32, 48))."""
    tensors = [
        finite_tensor("token_embedding.weight", (32, 32), 1),
        finite_tensor("head.weight", (32, 32), 2),
        finite_tensor("norm_final.weight", (32,), 3),
        finite_tensor("norm_final.bias", (32,), 4),
        finite_tensor("blocks.0.norm1.weight", (32,), 5),
        finite_tensor("blocks.0.norm2.weight", (32,), 6),
        finite_tensor("blocks.0.attention.query.weight", (32, 32), 7),
        finite_tensor("blocks.0.attention.key.weight", (16, 32), 8),
        finite_tensor("blocks.0.attention.value.weight", (16, 32), 9),
        finite_tensor("blocks.0.attention.output.weight", (32, 32), 10),
        finite_tensor("blocks.0.ffn.up.weight", (48, 32), 11),
        finite_tensor("blocks.0.ffn.down.weight", (32, 48), 12),
        finite_tensor("blocks.0.ffn.gate.weight", (48, 32), 13),
        finite_tensor("blocks.0.ffn.gate.bias", (48,), 14),
    ]
    tensors.sort(key=lambda t: t.path)
    return safe.State(PROVIDER, tuple(tensors), ())


def to_c1(state: safe.State) -> bytes:
    raw = build_container(
        tuple((tuple(p.encode("utf-8") for p in t.path), 1 if t.kind == "parameter" else 2, 3, t.shape, t.data)
              for t in state.tensors),
        state.aliases,
        state.provider.encode("utf-8"),
    )
    parse_golden(raw, expected_provider=state.provider.encode("utf-8"))
    return raw


def unpack_quant_gguf(raw: bytes):
    """Independent GGUF v3 binary inspector supporting F32, F16, and Q8_0."""
    magic, version, tensor_count, meta_count = struct.unpack_from("<4sIQQ", raw, 0)
    assert magic == b"GGUF" and version == 3
    reader = gg.Reader(raw, safe.Limits())
    metadata = {}
    for _ in range(meta_count):
        key = reader.text(128, nonempty=True)
        kind = reader.number("I")
        value = reader.text(1024 * 1024) if kind == gg.STRING else reader.number("I")
        metadata[key] = (kind, value)
    infos = []
    for _ in range(tensor_count):
        name = reader.text(63, nonempty=True)
        rank = reader.number("I")
        dims = tuple(reader.number("Q") for _ in range(rank))
        ggml_type = reader.number("I")
        offset = reader.number("Q")
        infos.append((name, dims, ggml_type, offset))
    data_at = gg.aligned(reader.at)
    rows = []
    for name, dims, ggml_type, offset in infos:
        count = 1
        for d in dims:
            count *= d
        if ggml_type == qi.GGUF_TYPE_F32:
            size = count * 4
        elif ggml_type == qi.GGUF_TYPE_F16:
            size = count * 2
        elif ggml_type == qi.GGUF_TYPE_Q8_0:
            size = (count // 32) * 34
        else:
            raise AssertionError(f"unexpected type {ggml_type}")
        rows.append((name, dims, ggml_type, offset, raw[data_at + offset:data_at + offset + size]))
    return metadata, rows, reader.at, data_at


def build_raw_gguf(
    metadata: list[tuple[str, int, str | int]],
    tensors: list[tuple[str, tuple[int, ...], int, int, bytes]],
    *,
    version: int = 3,
    prefix_pad: bytes | None = None,
    tensor_pads: list[bytes] | None = None,
    trailing: bytes = b"",
) -> bytes:
    """Independent wire builder for GGUF positive and negative test fixtures."""
    prefix = bytearray(struct.pack("<4sIQQ", b"GGUF", version, len(tensors), len(metadata)))
    for key, kind, value in metadata:
        prefix += gg.string(key) + struct.pack("<I", kind)
        prefix += gg.string(value) if kind == gg.STRING else struct.pack("<I", value)
    for name, dims, ggml_type, offset, _ in tensors:
        prefix += gg.string(name) + struct.pack("<I", len(dims))
        prefix += struct.pack("<" + "Q" * len(dims), *dims)
        prefix += struct.pack("<IQ", ggml_type, offset)
    if prefix_pad is None:
        prefix += bytes((-len(prefix)) % 32)
    else:
        prefix += prefix_pad
    body = bytearray()
    for idx, (_, _, _, _, data) in enumerate(tensors):
        body += data
        if tensor_pads is not None:
            body += tensor_pads[idx]
        else:
            body += bytes((-len(data)) % 32)
    return bytes(prefix) + bytes(body) + trailing


class QuantizedInterchangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lib_env = os.environ.get("ESHKOL_IO5_QUANT_LIB")
        lib_path = Path(lib_env) if lib_env else qi.default_native_lib_path()
        if not lib_path.is_file():
            if os.environ.get("IO5_QUANT_REQUIRED") == "1" or lib_env or __name__ == "__main__":
                raise RuntimeError(
                    f"native IO5 quant codec missing: {lib_path}; build with scripts/build-io5-quant.sh"
                )
            raise unittest.SkipTest("requires scripts/test-io5-quant.sh (native IO5 quant codec not built)")

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.m3_c1 = to_c1(m3_finite_state())
        self.mod5_c1 = to_c1(mod5_finite_state(tied=True))
        self.mod5_untied_c1 = to_c1(mod5_finite_state(tied=False))
        self.decoder_c1 = to_c1(decoder_finite_state())

    def test_safetensors_f16_bf16_roundtrips_and_binary_layout(self) -> None:
        cases = [
            ("m3", self.m3_c1, mapping.M3, "huggingface"),
            ("mod5-tied", self.mod5_c1, mapping.MOD5, "huggingface"),
            ("mod5-untied", self.mod5_untied_c1, mapping.MOD5, "gguf"),
            ("decoder", self.decoder_c1, mapping.DECODER, "huggingface"),
            ("unmapped", self.m3_c1, None, None),
        ]
        for label, src_c1, profile, scheme in cases:
            for storage_dtype, max_abs_bound, max_rel_bound in (
                ("F16", 1.0e-3, 6.0e-4),
                ("BF16", 1.0e-2, 5.0e-3),
            ):
                with self.subTest(case=label, dtype=storage_dtype):
                    encoded = qi.encode_quantized_safetensors(
                        src_c1, storage_dtype, profile=profile, scheme=scheme
                    )
                    # Determinism check.
                    self.assertEqual(
                        encoded,
                        qi.encode_quantized_safetensors(src_c1, storage_dtype, profile=profile, scheme=scheme),
                    )
                    header, data = unpack_safetensors(encoded)
                    meta = header.pop("__metadata__")
                    self.assertEqual(meta[qi.QUANT_VERSION_KEY], "1.0")
                    self.assertEqual(meta[qi.STORAGE_DTYPE_KEY], storage_dtype)
                    self.assertEqual(meta[qi.LOSSY_KEY], "true")
                    self.assertEqual(meta[qi.SOURCE_DIGEST_KEY], hashlib.sha256(src_c1).hexdigest())
                    self.assertNotIn(safe.DIGEST_KEY, meta)
                    self.assertNotIn(gg.DIGEST_KEY, meta)
                    if profile is not None:
                        self.assertEqual(meta[mapping.PROFILE], profile)
                        self.assertEqual(meta[mapping.SCHEME], scheme)

                    # Verify every descriptor uses exact narrow dtype and 2:1 payload ratio.
                    src_state = safe.decode_c1(src_c1)
                    f32_payload_bytes = sum(len(t.data) for t in src_state.tensors)
                    self.assertEqual(len(data) * 2, f32_payload_bytes)
                    for desc in header.values():
                        self.assertEqual(desc["dtype"], storage_dtype)

                    restored = qi.decode_quantized_safetensors(encoded, profile=profile, scheme=scheme)
                    parse_golden(restored, expected_provider=PROVIDER.encode("utf-8"))
                    # Explicitly verify lossy round-trip does NOT falsely equal source F32 bytes.
                    self.assertNotEqual(restored, src_c1)

                    metrics = qi.measure_reconstruction_error(src_c1, restored)
                    self.assertLessEqual(metrics["__overall__"]["max_abs"], max_abs_bound)
                    self.assertLessEqual(metrics["__overall__"]["max_rel"], max_rel_bound)

                    # Second round-trip from already quantized values must be bit-exact!
                    re_encoded = qi.encode_quantized_safetensors(
                        restored, storage_dtype, profile=profile, scheme=scheme
                    )
                    re_restored = qi.decode_quantized_safetensors(re_encoded, profile=profile, scheme=scheme)
                    self.assertEqual(re_restored, restored)

    def test_gguf_f16_and_q8_0_roundtrips_and_exact_block_layout(self) -> None:
        # 1. GGUF F16 on M3 and MOD5.
        for label, src_c1, profile in (
            ("m3", self.m3_c1, mapping.M3),
            ("mod5", self.mod5_c1, mapping.MOD5),
        ):
            with self.subTest(format="gguf-f16", case=label):
                encoded = qi.encode_quantized_gguf(src_c1, "F16", profile=profile, scheme="gguf")
                metadata, rows, prefix_end, data_at = unpack_quant_gguf(encoded)
                self.assertEqual(data_at % 32, 0)
                self.assertEqual(encoded[prefix_end:data_at], bytes(data_at - prefix_end))
                self.assertEqual(metadata["general.architecture"], (gg.STRING, "eshkol"))
                self.assertEqual(metadata["general.alignment"], (gg.UINT32, 32))
                self.assertEqual(metadata["general.file_type"], (gg.UINT32, qi.GGUF_FTYPE_MOSTLY_F16))
                self.assertNotIn("general.quantization_version", metadata)
                self.assertEqual(metadata[qi.LOSSY_KEY], (gg.STRING, "true"))
                self.assertEqual(metadata[qi.STORAGE_DTYPE_KEY], (gg.STRING, "F16"))
                self.assertNotIn(gg.DIGEST_KEY, metadata)
                for _, _, ggml_type, offset, _ in rows:
                    self.assertEqual(ggml_type, qi.GGUF_TYPE_F16)
                    self.assertEqual(offset % 32, 0)
                restored = qi.decode_quantized_gguf(encoded, profile=profile, scheme="gguf")
                parse_golden(restored, expected_provider=PROVIDER.encode("utf-8"))
                self.assertNotEqual(restored, src_c1)
                metrics = qi.measure_reconstruction_error(src_c1, restored)
                self.assertLessEqual(metrics["__overall__"]["max_abs"], 1.0e-3)
                self.assertLessEqual(metrics["__overall__"]["max_rel"], 6.0e-4)

        # 2. GGUF Q8_0 on M3 (mixed multiple-of-32 and non-multiple-of-32 + 1D norms/biases)
        #    and MOD5 (all 2D projections Q8_0, all 1D norms in F32 or F16 fallback).
        for fallback in ("F32", "F16"):
            for label, src_c1, profile in (
                ("m3", self.m3_c1, mapping.M3),
                ("mod5", self.mod5_c1, mapping.MOD5),
                ("decoder", self.decoder_c1, mapping.DECODER),
            ):
                with self.subTest(format="gguf-q8_0", case=label, fallback=fallback):
                    encoded = qi.encode_quantized_gguf(
                        src_c1, "Q8_0", q8_fallback=fallback, profile=profile, scheme="gguf"
                    )
                    metadata, rows, prefix_end, data_at = unpack_quant_gguf(encoded)
                    self.assertEqual(metadata["general.file_type"], (gg.UINT32, qi.GGUF_FTYPE_MOSTLY_Q8_0))
                    self.assertEqual(metadata["general.quantization_version"], (gg.UINT32, 2))
                    self.assertEqual(metadata[qi.Q8_FALLBACK_KEY], (gg.STRING, fallback))
                    dtypes_map = json.loads(metadata[qi.TENSOR_DTYPES_KEY][1])

                    q8_count = 0
                    fallback_count = 0
                    for name, dims, ggml_type, offset, payload in rows:
                        self.assertEqual(offset % 32, 0)
                        c_shape = tuple(reversed(dims))
                        elems = safe.elements(c_shape)
                        expected_dtype = qi.select_gguf_tensor_dtype(c_shape, "Q8_0", fallback)
                        self.assertEqual(dtypes_map[name], expected_dtype)
                        self.assertEqual(qi.GGUF_TYPE_TO_NAME[ggml_type], expected_dtype)
                        if expected_dtype == "Q8_0":
                            q8_count += 1
                            self.assertEqual(len(c_shape), 2)
                            self.assertEqual(dims[0] % 32, 0)
                            self.assertEqual(c_shape[-1] % 32, 0)
                            self.assertEqual(elems % 32, 0)
                            self.assertEqual(len(payload), (elems // 32) * 34)
                            # Inspect every 34-byte Q8_0 block: finite f16 scale + i8 in [-127, 127].
                            for blk_idx in range(elems // 32):
                                blk = payload[blk_idx * 34:(blk_idx + 1) * 34]
                                d_bits = struct.unpack_from("<H", blk, 0)[0]
                                self.assertNotEqual(d_bits & 0x7C00, 0x7C00)
                                qs = struct.unpack_from("<32b", blk, 2)
                                self.assertTrue(all(-127 <= q <= 127 for q in qs))
                        else:
                            fallback_count += 1
                            # Either 1D norm/bias OR GGML ne[0] (C1 shape[-1]) not divisible by 32 (no silent padding!).
                            self.assertTrue(len(c_shape) == 1 or (c_shape[-1] % 32 != 0))
                    self.assertGreater(fallback_count, 0)
                    if label == "m3":
                        # M3 tiny shapes have ne[0] in {4, 8}, so all 15 tensors stay in q8_fallback (no silent padding).
                        self.assertEqual(q8_count, 0)
                        self.assertEqual(fallback_count, 15)
                    elif label == "mod5":
                        # MOD5 has 2*7 + 2 = 16 Q8_0 2D tensors (all ne[0] in {32, 64}) and 5 1D norms in fallback.
                        self.assertEqual(q8_count, 16)
                        self.assertEqual(fallback_count, 5)
                    elif label == "decoder":
                        # Decoder has 8 Q8_0 2D tensors (ne[0]=32), 1 2D fallback (ffn.down (32,48) with prod%32==0 but ne[0]=48),
                        # plus 5 1D norms/biases in fallback = 6 fallback total.
                        self.assertEqual(q8_count, 8)
                        self.assertEqual(fallback_count, 6)
                        self.assertEqual(dtypes_map["blk.0.ffn_down.weight"], fallback)

                    restored = qi.decode_quantized_gguf(encoded, profile=profile, scheme="gguf")
                    parse_golden(restored, expected_provider=PROVIDER.encode("utf-8"))
                    if label == "m3" and fallback == "F32":
                        # Every M3 tensor stayed in F32 fallback because ne[0] < 32.
                        self.assertEqual(restored, src_c1)
                    else:
                        self.assertNotEqual(restored, src_c1)
                    metrics = qi.measure_reconstruction_error(src_c1, restored)
                    # Q8_0 quantization step is ~amax/127 -> half-step error <= ~0.0085 for scale <= 1.88.
                    self.assertLessEqual(metrics["__overall__"]["max_abs"], 0.015)
                    for path_key, m in metrics.items():
                        if path_key == "__overall__":
                            continue
                        self.assertLessEqual(m["rel_l2"], 0.01)
                    # With F32 fallback, 1D norms/biases (and ne[0]%32!=0 tensors) must have exact 0.0 error!
                    if fallback == "F32":
                        self.assertEqual(metrics["norm_final.weight"]["max_abs"], 0.0)
                        self.assertEqual(metrics["blocks.0.norm1.weight"]["max_abs"], 0.0)
                        if label == "decoder":
                            self.assertEqual(metrics["blocks.0.ffn.down.weight"]["max_abs"], 0.0)

        # 3. Explicit (32, 16) test: prod(shape)=512 is divisible by 32, but GGML ne[0]=16 is NOT.
        #    Export MUST fall back (never emit GGML-invalid Q8_0); import MUST reject if declared Q8_0.
        non_row_aligned_c1 = to_c1(safe.State(PROVIDER, (finite_tensor("w", (32, 16), 99),)))
        for fb in ("F32", "F16"):
            enc_fb = qi.encode_quantized_gguf(non_row_aligned_c1, "Q8_0", q8_fallback=fb)
            meta_fb, rows_fb, _, _ = unpack_quant_gguf(enc_fb)
            self.assertEqual(len(rows_fb), 1)
            self.assertEqual(rows_fb[0][1], (16, 32))  # GGML dims = reversed(C1 shape) -> ne[0] = 16
            self.assertEqual(qi.GGUF_TYPE_TO_NAME[rows_fb[0][2]], fb)
            self.assertEqual(json.loads(meta_fb[qi.TENSOR_DTYPES_KEY][1]), {"t000000": fb})
            restored_fb = qi.decode_quantized_gguf(enc_fb)
            if fb == "F32":
                self.assertEqual(restored_fb, non_row_aligned_c1)

    def test_empty_safetensors_tensor_and_external_imports(self) -> None:
        # Empty (zero-extent) tensor allowed in unmapped safetensors F16/BF16.
        empty_state = safe.State(PROVIDER, (safe.Tensor(("empty",), "parameter", (0, 4), b""),))
        empty_c1 = to_c1(empty_state)
        for dt in ("F16", "BF16"):
            st = qi.encode_quantized_safetensors(empty_c1, dt)
            restored = qi.decode_quantized_safetensors(st)
            self.assertEqual(restored, empty_c1)

        # Plain external safetensors F16 import (mapped with --external and unmapped with --plain-provider).
        encoded_st = qi.encode_quantized_safetensors(self.mod5_c1, "F16", profile=mapping.MOD5, scheme="huggingface")
        header, data = unpack_safetensors(encoded_st)
        header.pop("__metadata__")
        plain_st = safetensors_wire(header, data)
        restored_ext = qi.decode_quantized_safetensors(
            plain_st, profile=mapping.MOD5, scheme="huggingface", external=True, provider=PROVIDER, tie=True
        )
        self.assertEqual(restored_ext, qi.decode_quantized_safetensors(encoded_st))
        restored_plain = qi.decode_quantized_safetensors(plain_st, plain_provider=PROVIDER)
        parse_golden(restored_plain, expected_provider=PROVIDER.encode("utf-8"))

        # Plain external GGUF Q8_0 import.
        encoded_gg = qi.encode_quantized_gguf(self.mod5_c1, "Q8_0", profile=mapping.MOD5, scheme="gguf")
        _, rows, _, _ = unpack_quant_gguf(encoded_gg)
        plain_gg = build_raw_gguf(
            [("general.architecture", gg.STRING, "llama"), ("general.alignment", gg.UINT32, 32)],
            rows,
        )
        restored_gg_ext = qi.decode_quantized_gguf(
            plain_gg, profile=mapping.MOD5, scheme="gguf", external=True, provider=PROVIDER, tie=True
        )
        self.assertEqual(restored_gg_ext, qi.decode_quantized_gguf(encoded_gg))
        restored_gg_plain = qi.decode_quantized_gguf(plain_gg, plain_provider=PROVIDER)
        parse_golden(restored_gg_plain, expected_provider=PROVIDER.encode("utf-8"))

    def test_cli_roundtrips_and_no_clobber(self) -> None:
        src_path = self.root / "mod5.c1"
        src_path.write_bytes(self.mod5_c1)
        st_path = self.root / "mod5_bf16.safetensors"
        st_out = self.root / "mod5_from_bf16.c1"
        gg_path = self.root / "mod5_q8.gguf"
        gg_out = self.root / "mod5_from_q8.c1"

        for cmd in (
            ["c1-to-safetensors", src_path, st_path, "--dtype", "BF16", "--profile", mapping.MOD5],
            ["safetensors-to-c1", st_path, st_out, "--profile", mapping.MOD5],
            ["c1-to-gguf", src_path, gg_path, "--dtype", "Q8_0", "--q8-fallback", "F16", "--profile", mapping.MOD5],
            ["gguf-to-c1", gg_path, gg_out, "--profile", mapping.MOD5],
        ):
            res = subprocess.run([sys.executable, str(TOOL), *map(str, cmd)], capture_output=True)
            self.assertEqual(res.returncode, 0, res.stderr.decode())

        parse_golden(st_out.read_bytes(), expected_provider=PROVIDER.encode("utf-8"))
        parse_golden(gg_out.read_bytes(), expected_provider=PROVIDER.encode("utf-8"))

        # Default no-clobber rejects overwrite; --overwrite succeeds.
        res_fail = subprocess.run(
            [sys.executable, str(TOOL), "c1-to-gguf", str(src_path), str(gg_path), "--dtype", "F16"],
            capture_output=True,
        )
        self.assertEqual(res_fail.returncode, 2)
        res_ok = subprocess.run(
            [sys.executable, str(TOOL), "c1-to-gguf", str(src_path), str(gg_path), "--dtype", "F16", "--overwrite"],
            capture_output=True,
        )
        self.assertEqual(res_ok.returncode, 0, res_ok.stderr.decode())

    def test_negative_missing_native_lib_fails_closed(self) -> None:
        with self.assertRaisesRegex(safe.InteropError, "native IO5 quant codec library missing"):
            qi.encode_quantized_safetensors(self.m3_c1, "F16", lib_path=self.root / "nonexistent.so")

        # An unrelated shared object missing the IO5 quant ABI symbols must raise InteropError
        # (and fail closed with CLI exit code 2, never uncaught AttributeError exit code 1).
        import _ctypes
        unrelated_so = Path(_ctypes.__file__)
        with self.assertRaisesRegex(safe.InteropError, "missing required symbol"):
            qi.encode_quantized_safetensors(self.m3_c1, "F16", lib_path=unrelated_so)

        src_path = self.root / "badlib_src.c1"
        dst_path = self.root / "badlib_dst.safetensors"
        src_path.write_bytes(self.m3_c1)
        env = dict(os.environ, ESHKOL_IO5_QUANT_LIB=str(unrelated_so))
        res = subprocess.run(
            [sys.executable, str(TOOL), "c1-to-safetensors", str(src_path), str(dst_path), "--dtype", "F16"],
            capture_output=True,
            env=env,
        )
        self.assertEqual(res.returncode, 2)
        self.assertIn("missing required symbol", res.stderr.decode())

    def test_negative_overflow_and_nonfinite_inputs(self) -> None:
        # 1. F16 max boundary (65504.0 accepted, nextafter(65504.0)=0x477fe001 and 65520.0 rejected).
        ok_state = safe.State(PROVIDER, (safe.Tensor(("w",), "parameter", (2,), struct.pack("<2f", 65504.0, -65504.0)),))
        ok_c1 = to_c1(ok_state)
        qi.encode_quantized_safetensors(ok_c1, "F16")
        qi.encode_quantized_gguf(ok_c1, "F16")

        for bad_bits, label in (
            (0x477FE001, "nextafter_65504"),
            (0xC77FE001, "neg_nextafter_65504"),
            (struct.unpack("<I", struct.pack("<f", 65520.0))[0], "65520"),
            (0x7F800000, "pos_inf"),
            (0xFF800000, "neg_inf"),
            (0x7FC00000, "qnan"),
            (0x7F800001, "snan"),
        ):
            bad_state = safe.State(PROVIDER, (safe.Tensor(("w",), "parameter", (1,), struct.pack("<I", bad_bits)),))
            bad_c1 = to_c1(bad_state)
            with self.subTest(f16_bad=label):
                with self.assertRaises(safe.InteropError):
                    qi.encode_quantized_safetensors(bad_c1, "F16")
                with self.assertRaises(safe.InteropError):
                    qi.encode_quantized_gguf(bad_c1, "F16")

        # 2. BF16 overflow (0x7f7f8000 -> rounds to Inf) and nonfinite.
        for bad_bits in (0x7F7F8000, 0xFF7F8000, 0x7F800000, 0x7FC00001):
            bad_c1 = to_c1(safe.State(PROVIDER, (safe.Tensor(("w",), "parameter", (1,), struct.pack("<I", bad_bits)),)))
            with self.assertRaises(safe.InteropError):
                qi.encode_quantized_safetensors(bad_c1, "BF16")

        # 3. Q8_0 scale overflow (amax > 127 * 65504 = 8319008.0) and non-finite input.
        for bad_val in (8319010.0, float("inf"), float("nan")):
            arr = [0.0] * 32
            arr[17] = bad_val
            bad_c1 = to_c1(safe.State(PROVIDER, (safe.Tensor(("w",), "parameter", (1, 32), struct.pack("<32f", *arr)),)))
            with self.assertRaises(safe.InteropError):
                qi.encode_quantized_gguf(bad_c1, "Q8_0")

        # 4. Non-finite F32 fallback tensor in Q8_0 GGUF rejected.
        bad_norm = to_c1(safe.State(PROVIDER, (safe.Tensor(("n",), "parameter", (4,), struct.pack("<4f", 1.0, float("nan"), 0.0, 1.0)),)))
        with self.assertRaises(safe.InteropError):
            qi.encode_quantized_gguf(bad_norm, "Q8_0", q8_fallback="F32")

        # 5. Non-finite F16 and BF16 payloads rejected on decode (both safetensors and GGUF).
        for dtype, bad_words in (
            ("F16", (0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFFFF)),
            ("BF16", (0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0xFFFF)),
        ):
            for word in bad_words:
                raw_st = safetensors_wire({"w": {"dtype": dtype, "shape": [2], "data_offsets": [0, 4]}},
                                          struct.pack("<2H", 0x3C00, word))
                with self.assertRaisesRegex(safe.InteropError, "non-finite"):
                    qi.decode_quantized_safetensors(raw_st, plain_provider=PROVIDER)
                if dtype == "F16":
                    raw_gg = build_raw_gguf(
                        [("general.architecture", gg.STRING, "test")],
                        [("w", (2,), qi.GGUF_TYPE_F16, 0, struct.pack("<2H", 0x3C00, word))],
                    )
                    with self.assertRaisesRegex(safe.InteropError, "non-finite"):
                        qi.decode_quantized_gguf(raw_gg, plain_provider=PROVIDER)

    def test_negative_q8_0_wire_truncation_scales_alignment_and_types(self) -> None:
        good_q8_block = struct.pack("<H32b", 0x3C00, *([1] * 32))

        # 1. NaN / Inf f16 block scales in Q8_0 GGUF.
        for bad_scale in (0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFFFF):
            bad_blk = struct.pack("<H32b", bad_scale, *([1] * 32))
            raw = build_raw_gguf(
                [("general.architecture", gg.STRING, "test")],
                [("w", (32, 1), qi.GGUF_TYPE_Q8_0, 0, bad_blk)],
            )
            with self.subTest(bad_scale=hex(bad_scale)):
                with self.assertRaisesRegex(safe.InteropError, "non-finite"):
                    qi.decode_quantized_gguf(raw, plain_provider=PROVIDER)

        # 2. Truncated Q8_0 block (file truncated mid-block and mid-padding).
        good_raw = build_raw_gguf(
            [("general.architecture", gg.STRING, "test")],
            [("w", (32, 1), qi.GGUF_TYPE_Q8_0, 0, good_q8_block)],
        )
        for cut in (1, 10, 31, 33, 34):
            with self.assertRaises(safe.InteropError):
                qi.decode_quantized_gguf(good_raw[:-cut], plain_provider=PROVIDER)

        # 3. Non-multiple-of-32 innermost dimension (ne[0]) declared as Q8_0 in GGUF:
        #    both prod%32!=0 (dims=(16,1)) AND prod%32==0 with ne[0]%32!=0 (dims=(16,32) -> C1 (32,16)).
        raw_bad_count = build_raw_gguf(
            [("general.architecture", gg.STRING, "test")],
            [("w", (16, 1), qi.GGUF_TYPE_Q8_0, 0, good_q8_block)],
        )
        with self.assertRaisesRegex(safe.InteropError, "divisible by 32"):
            qi.decode_quantized_gguf(raw_bad_count, plain_provider=PROVIDER)

        raw_bad_ne0 = build_raw_gguf(
            [("general.architecture", gg.STRING, "test")],
            [("w", (16, 32), qi.GGUF_TYPE_Q8_0, 0, good_q8_block * 16)],
        )
        with self.assertRaisesRegex(safe.InteropError, "ne\\[0\\].*divisible by 32"):
            qi.decode_quantized_gguf(raw_bad_ne0, plain_provider=PROVIDER)

        # 4. Unsupported GGUF tensor types (Q4_0=2, Q4_1=3, I8=24, F64=28, BF16=30, 0xFFFFFFFF).
        for bad_type in (2, 3, 24, 28, 30, 0xFFFFFFFF):
            raw_type = build_raw_gguf(
                [("general.architecture", gg.STRING, "test")],
                [("w", (32,), bad_type, 0, bytes(32))],
            )
            with self.assertRaisesRegex(safe.InteropError, "unsupported GGUF tensor type"):
                qi.decode_quantized_gguf(raw_type, plain_provider=PROVIDER)

        # 5. Unsupported safetensors dtypes (F32, F64, I32, U8).
        for bad_dtype, size in (("F32", 4), ("F64", 8), ("I32", 4), ("U8", 1)):
            raw_st = safetensors_wire({"w": {"dtype": bad_dtype, "shape": [1], "data_offsets": [0, size]}}, bytes(size))
            with self.assertRaisesRegex(safe.InteropError, "unsupported safetensors quantized dtype"):
                qi.decode_quantized_safetensors(raw_st, plain_provider=PROVIDER)

        # 6. Bad GGUF tensor alignment, nonzero padding, trailing bytes, split.* metadata.
        bad_ggufs = [
            build_raw_gguf([("general.architecture", gg.STRING, "test"), ("general.alignment", gg.UINT32, 16)],
                           [("w", (32,), qi.GGUF_TYPE_Q8_0, 0, good_q8_block)]),
            build_raw_gguf([("general.architecture", gg.STRING, "test"), ("general.alignment", gg.UINT32, 64)],
                           [("w", (32,), qi.GGUF_TYPE_Q8_0, 0, good_q8_block)]),
            build_raw_gguf([("general.architecture", gg.STRING, "test")],
                           [("w", (32,), qi.GGUF_TYPE_Q8_0, 16, good_q8_block)]),
            build_raw_gguf([("general.architecture", gg.STRING, "test")],
                           [("w", (32,), qi.GGUF_TYPE_Q8_0, 0, good_q8_block)],
                           tensor_pads=[b"\x01" + bytes(29)]),
            build_raw_gguf([("general.architecture", gg.STRING, "test")],
                           [("w", (32,), qi.GGUF_TYPE_Q8_0, 0, good_q8_block)],
                           trailing=b"\x00"),
            build_raw_gguf([("general.architecture", gg.STRING, "test"), ("split.count", gg.UINT32, 2)],
                           [("w", (32,), qi.GGUF_TYPE_Q8_0, 0, good_q8_block)]),
        ]
        for idx, raw_bad in enumerate(bad_ggufs):
            with self.subTest(bad_gguf=idx):
                with self.assertRaises(safe.InteropError):
                    qi.decode_quantized_gguf(raw_bad, plain_provider=PROVIDER)

    def test_negative_metadata_tamper_and_invalid_profile_rosters(self) -> None:
        # 1. Payload byte tamper caught by payload_sha256 in both Safetensors and GGUF.
        st_good = qi.encode_quantized_safetensors(self.mod5_c1, "F16", profile=mapping.MOD5, scheme="huggingface")
        st_tampered = bytearray(st_good)
        st_tampered[-1] ^= 0x01
        with self.assertRaisesRegex(safe.InteropError, "payload SHA-256 mismatch"):
            qi.decode_quantized_safetensors(bytes(st_tampered))

        gg_good = qi.encode_quantized_gguf(self.mod5_c1, "Q8_0", profile=mapping.MOD5, scheme="gguf")
        meta, rows, _, _ = unpack_quant_gguf(gg_good)
        # Flip one quant byte in the first Q8_0 tensor.
        tampered_rows = []
        flipped = False
        for name, dims, gtype, off, data in rows:
            if not flipped and gtype == qi.GGUF_TYPE_Q8_0:
                b = bytearray(data)
                b[2] ^= 0x01
                data = bytes(b)
                flipped = True
            tampered_rows.append((name, dims, gtype, off, data))
        gg_tampered = build_raw_gguf([(k, kind, val) for k, (kind, val) in sorted(meta.items())], tampered_rows)
        with self.assertRaisesRegex(safe.InteropError, "payload SHA-256 mismatch"):
            qi.decode_quantized_gguf(gg_tampered)

        # 1b. Framed payload_sha256 rejects same-byte-size tensor reshape and manifest tampering.
        header_reshape, data_reshape = unpack_safetensors(st_good)
        header_reshaped = json.loads(json.dumps(header_reshape))
        # Reshape (32, 32) -> (16, 64) preserving exact byte count.
        self.assertEqual(header_reshaped["model.layers.0.self_attn.q_proj.weight"]["shape"], [32, 32])
        header_reshaped["model.layers.0.self_attn.q_proj.weight"]["shape"] = [16, 64]
        with self.assertRaisesRegex(safe.InteropError, "payload SHA-256 mismatch"):
            qi.decode_quantized_safetensors(safetensors_wire(header_reshaped, data_reshape))

        header_manifest_tamper = json.loads(json.dumps(header_reshape))
        m_obj = json.loads(header_manifest_tamper["__metadata__"][safe.MANIFEST_KEY])
        m_obj["provider"] = "tampered-provider"
        header_manifest_tamper["__metadata__"][safe.MANIFEST_KEY] = json.dumps(
            m_obj, separators=(",", ":"), ensure_ascii=True
        )
        with self.assertRaisesRegex(safe.InteropError, "payload SHA-256 mismatch"):
            qi.decode_quantized_safetensors(safetensors_wire(header_manifest_tamper, data_reshape))

        # GGUF reshape preserving exact byte count: (32, 32) -> (64, 16) (GGML dims (64, 16) -> C1 (16, 64), ne[0]=64).
        reshaped_rows = []
        did_reshape = False
        for name, dims, gtype, off, data in rows:
            if not did_reshape and name == "blk.0.attn_q.weight":
                self.assertEqual(dims, (32, 32))
                dims = (64, 16)
                did_reshape = True
            reshaped_rows.append((name, dims, gtype, off, data))
        self.assertTrue(did_reshape)
        gg_reshaped = build_raw_gguf([(k, kind, val) for k, (kind, val) in sorted(meta.items())], reshaped_rows)
        with self.assertRaisesRegex(safe.InteropError, "payload SHA-256 mismatch"):
            qi.decode_quantized_gguf(gg_reshaped)

        # 2. Lossy file carrying lossless c1-sha256 key or lossy != "true" rejected.
        header, data = unpack_safetensors(st_good)
        header_lossless = json.loads(json.dumps(header))
        header_lossless["__metadata__"][safe.DIGEST_KEY] = "0" * 64
        with self.assertRaisesRegex(safe.InteropError, "lossless F32"):
            qi.decode_quantized_safetensors(safetensors_wire(header_lossless, data))

        header_not_lossy = json.loads(json.dumps(header))
        header_not_lossy["__metadata__"][qi.LOSSY_KEY] = "false"
        with self.assertRaisesRegex(safe.InteropError, "lossy flag"):
            qi.decode_quantized_safetensors(safetensors_wire(header_not_lossy, data))

        # 3. Profile mismatch assertion on decode.
        with self.assertRaisesRegex(safe.InteropError, "profile mismatch"):
            qi.decode_quantized_safetensors(st_good, profile=mapping.M3)
        with self.assertRaisesRegex(safe.InteropError, "profile mismatch"):
            qi.decode_quantized_gguf(gg_good, profile=mapping.M3)

        # 4. Invalid profile rosters rejected on export.
        #    - M3 state exported as MOD5 fails (MOD5 rejects biases & position_embedding).
        with self.assertRaises(safe.InteropError):
            qi.encode_quantized_safetensors(self.m3_c1, "F16", profile=mapping.MOD5, scheme="huggingface")
        with self.assertRaises(safe.InteropError):
            qi.encode_quantized_gguf(self.m3_c1, "Q8_0", profile=mapping.MOD5, scheme="gguf")
        #    - MOD5 state exported as M3 fails (wrong shapes & missing biases).
        with self.assertRaises(safe.InteropError):
            qi.encode_quantized_gguf(self.mod5_c1, "Q8_0", profile=mapping.M3, scheme="gguf")
        #    - Untied M3 fails M3 profile requirement.
        untied_m3 = replace(m3_finite_state(), aliases=())
        with self.assertRaisesRegex(safe.InteropError, "head/embedding tie"):
            qi.encode_quantized_safetensors(to_c1(untied_m3), "BF16", profile=mapping.M3, scheme="huggingface")


if __name__ == "__main__":
    unittest.main()
