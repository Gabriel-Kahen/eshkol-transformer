#!/usr/bin/env python3
"""Bounded non-executable IO5 mixed-precision (F16/BF16) and quantized (Q8_0) interchange."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import sys

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import gguf_interop as gg  # noqa: E402
import model_mapping as mapping  # noqa: E402
import safetensors_interop as safe  # noqa: E402
from safetensors_interop import InteropError, Limits, State, Tensor, require  # noqa: E402

QUANT_ABI_VERSION_V1_0 = (1 << 16) | 0
ET_IO5_QUANT_OK = 0
ET_IO5_QUANT_ERR_INVALID_ARG = 1
ET_IO5_QUANT_ERR_INVALID_LENGTH = 2
ET_IO5_QUANT_ERR_ALIAS = 3
ET_IO5_QUANT_ERR_NONFINITE = 4
ET_IO5_QUANT_ERR_OVERFLOW = 5

QK8_0 = 32
Q8_0_BLOCK_BYTES = 34

GGUF_TYPE_F32 = 0
GGUF_TYPE_F16 = 1
GGUF_TYPE_Q8_0 = 8
GGUF_TYPE_TO_NAME = {
    GGUF_TYPE_F32: "F32",
    GGUF_TYPE_F16: "F16",
    GGUF_TYPE_Q8_0: "Q8_0",
}
NAME_TO_GGUF_TYPE = {name: code for code, name in GGUF_TYPE_TO_NAME.items()}

GGUF_FTYPE_MOSTLY_F16 = 1
GGUF_FTYPE_MOSTLY_Q8_0 = 7
GGUF_QUANTIZATION_VERSION = 2

SAFETENSORS_STORAGE_DTYPES = ("F16", "BF16")
GGUF_STORAGE_DTYPES = ("F16", "Q8_0")
Q8_FALLBACK_DTYPES = ("F32", "F16")
QUANTIZED_PROFILES = (mapping.M3, mapping.DECODER)

QUANT_VERSION_KEY = "eshkol.io5.quant_version"
STORAGE_DTYPE_KEY = "eshkol.io5.storage_dtype"
LOSSY_KEY = "eshkol.io5.lossy"
SOURCE_DIGEST_KEY = "eshkol.io5.source_c1_sha256"
PAYLOAD_DIGEST_KEY = "eshkol.io5.payload_sha256"
TENSOR_DTYPES_KEY = "eshkol.io5.tensor_dtypes"
Q8_FALLBACK_KEY = "eshkol.io5.q8_fallback_dtype"

SAFETENSORS_BASE_KEYS = frozenset({
    QUANT_VERSION_KEY,
    STORAGE_DTYPE_KEY,
    LOSSY_KEY,
    SOURCE_DIGEST_KEY,
    PAYLOAD_DIGEST_KEY,
    safe.MANIFEST_KEY,
})
GGUF_BASE_KEYS = frozenset({
    QUANT_VERSION_KEY,
    STORAGE_DTYPE_KEY,
    LOSSY_KEY,
    TENSOR_DTYPES_KEY,
    SOURCE_DIGEST_KEY,
    PAYLOAD_DIGEST_KEY,
    safe.MANIFEST_KEY,
})
LOSSLESS_FORBIDDEN_KEYS = frozenset({safe.DIGEST_KEY, gg.DIGEST_KEY, safe.VERSION_KEY, gg.VERSION_KEY})

SHA256_RE = re.compile(r"[0-9a-f]{64}")

_NATIVE_LIB_CACHE: dict[str, ctypes.CDLL] = {}

_STATUS_MESSAGES = {
    ET_IO5_QUANT_ERR_INVALID_ARG: "native IO5 quant codec invalid argument",
    ET_IO5_QUANT_ERR_INVALID_LENGTH: "native IO5 quant codec invalid length or non-multiple-of-32 count",
    ET_IO5_QUANT_ERR_ALIAS: "native IO5 quant codec aliased buffers",
    ET_IO5_QUANT_ERR_NONFINITE: "native IO5 quant codec rejected non-finite value or scale",
    ET_IO5_QUANT_ERR_OVERFLOW: "native IO5 quant codec rejected value exceeding finite target range",
}


def default_native_lib_path() -> Path:
    return Path(__file__).resolve().parents[2] / "build" / "io5_quant" / "libeshkol_transformer_io5_quant.so"


def load_native_codec(lib_path: str | Path | None = None) -> ctypes.CDLL:
    if lib_path is None:
        env_path = os.environ.get("ESHKOL_IO5_QUANT_LIB")
        resolved = Path(env_path) if env_path else default_native_lib_path()
    else:
        resolved = Path(lib_path)
    key = str(resolved)
    cached = _NATIVE_LIB_CACHE.get(key)
    if cached is not None:
        return cached
    require(resolved.is_file(), f"native IO5 quant codec library missing: {resolved}")
    try:
        lib = ctypes.CDLL(str(resolved))
    except OSError as error:
        raise InteropError(f"failed to load native IO5 quant codec library {resolved}: {error}") from error

    try:
        lib.et_io5_quant_abi_version.argtypes = []
        lib.et_io5_quant_abi_version.restype = ctypes.c_uint32
        abi = lib.et_io5_quant_abi_version()
        require(abi == QUANT_ABI_VERSION_V1_0, f"unexpected native IO5 quant ABI version: {abi:#x}")

        lib.et_io5_validate_f32_finite.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        lib.et_io5_validate_f32_finite.restype = ctypes.c_uint32

        bulk_args = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        for fn_name in (
            "et_io5_encode_f16",
            "et_io5_decode_f16",
            "et_io5_encode_bf16",
            "et_io5_decode_bf16",
            "et_io5_quantize_q8_0",
            "et_io5_dequantize_q8_0",
        ):
            fn = getattr(lib, fn_name)
            fn.argtypes = bulk_args
            fn.restype = ctypes.c_uint32
    except AttributeError as error:
        raise InteropError(
            f"native IO5 quant codec library {resolved} missing required symbol: {error}"
        ) from error

    _NATIVE_LIB_CACHE[key] = lib
    return lib


def _check_status(status: int) -> None:
    if status != ET_IO5_QUANT_OK:
        raise InteropError(_STATUS_MESSAGES.get(status, f"native IO5 quant codec error code {status}"))


def native_validate_f32_finite(raw_f32: bytes, count: int, *, lib_path: str | Path | None = None) -> None:
    require(type(raw_f32) is bytes and len(raw_f32) == count * 4, "f32 validation length mismatch")
    lib = load_native_codec(lib_path)
    src = ctypes.create_string_buffer(raw_f32, max(len(raw_f32), 1))
    _check_status(lib.et_io5_validate_f32_finite(ctypes.addressof(src), len(raw_f32), count))


def _call_bulk(fn_name: str, src_bytes: bytes, dst_len: int, count: int, *,
               lib_path: str | Path | None = None) -> bytes:
    lib = load_native_codec(lib_path)
    fn = getattr(lib, fn_name)
    src_buf = ctypes.create_string_buffer(src_bytes, max(len(src_bytes), 1))
    dst_buf = ctypes.create_string_buffer(max(dst_len, 1))
    status = fn(ctypes.addressof(src_buf), len(src_bytes),
                ctypes.addressof(dst_buf), dst_len, count)
    _check_status(status)
    return dst_buf.raw[:dst_len]


def native_encode_f16(raw_f32: bytes, count: int, *, lib_path: str | Path | None = None) -> bytes:
    require(len(raw_f32) == count * 4, "f16 encode input byte length mismatch")
    return _call_bulk("et_io5_encode_f16", raw_f32, count * 2, count, lib_path=lib_path)


def native_decode_f16(raw_f16: bytes, count: int, *, lib_path: str | Path | None = None) -> bytes:
    require(len(raw_f16) == count * 2, "f16 decode input byte length mismatch")
    return _call_bulk("et_io5_decode_f16", raw_f16, count * 4, count, lib_path=lib_path)


def native_encode_bf16(raw_f32: bytes, count: int, *, lib_path: str | Path | None = None) -> bytes:
    require(len(raw_f32) == count * 4, "bf16 encode input byte length mismatch")
    return _call_bulk("et_io5_encode_bf16", raw_f32, count * 2, count, lib_path=lib_path)


def native_decode_bf16(raw_bf16: bytes, count: int, *, lib_path: str | Path | None = None) -> bytes:
    require(len(raw_bf16) == count * 2, "bf16 decode input byte length mismatch")
    return _call_bulk("et_io5_decode_bf16", raw_bf16, count * 4, count, lib_path=lib_path)


def native_quantize_q8_0(raw_f32: bytes, count: int, *, lib_path: str | Path | None = None) -> bytes:
    require(count > 0 and count % QK8_0 == 0, "Q8_0 requires positive element count divisible by 32")
    require(len(raw_f32) == count * 4, "Q8_0 quantize input byte length mismatch")
    dst_len = (count // QK8_0) * Q8_0_BLOCK_BYTES
    return _call_bulk("et_io5_quantize_q8_0", raw_f32, dst_len, count, lib_path=lib_path)


def native_dequantize_q8_0(raw_q8: bytes, count: int, *, lib_path: str | Path | None = None) -> bytes:
    require(count > 0 and count % QK8_0 == 0, "Q8_0 requires positive element count divisible by 32")
    expected_len = (count // QK8_0) * Q8_0_BLOCK_BYTES
    require(len(raw_q8) == expected_len, "Q8_0 dequantize input byte length mismatch (truncated or oversize)")
    return _call_bulk("et_io5_dequantize_q8_0", raw_q8, count * 4, count, lib_path=lib_path)


def select_gguf_tensor_dtype(shape: tuple[int, ...], storage_dtype: str, q8_fallback: str = "F32") -> str:
    require(storage_dtype in GGUF_STORAGE_DTYPES, "unsupported GGUF quantized storage dtype")
    require(q8_fallback in Q8_FALLBACK_DTYPES, "unsupported Q8_0 fallback dtype")
    if storage_dtype == "F16":
        return "F16"
    # Explicit GGML Q8_0 policy: 1D norms/biases and tensors whose innermost
    # row dimension (C1 shape[-1] == GGML ne[0]) is not divisible by 32 stay
    # in q8_fallback (F32 or F16); never silently pad.
    if len(shape) == 1:
        return q8_fallback
    return "Q8_0" if (len(shape) >= 2 and shape[-1] > 0 and shape[-1] % QK8_0 == 0) else q8_fallback


def _resolve_names(state: State, profile: str | None, scheme: str | None, default_scheme: str) -> tuple[tuple[str, ...], str | None]:
    if profile is None:
        require(scheme is None, "--scheme requires --profile")
        return mapping.synthetic_names(state), None
    _require_quantized_profile(profile)
    chosen_scheme = default_scheme if scheme is None else scheme
    return mapping.mapped_names(state, profile, chosen_scheme), chosen_scheme


def _require_quantized_profile(profile: str | None) -> None:
    require(profile in QUANTIZED_PROFILES, "unsupported quantized profile")


def _payload_digest(
    manifest: str,
    entries: list[tuple[str, str, tuple[int, ...], bytes]],
) -> str:
    h = hashlib.sha256()
    manifest_bytes = manifest.encode("utf-8")
    h.update(b"eshkol.io5.quant.payload.v1\0")
    h.update(struct.pack("<Q", len(manifest_bytes)))
    h.update(manifest_bytes)
    h.update(struct.pack("<Q", len(entries)))
    for name, dtype, shape, payload in entries:
        name_bytes = name.encode("utf-8")
        dtype_bytes = dtype.encode("ascii")
        h.update(struct.pack("<QQQ", len(name_bytes), len(dtype_bytes), len(shape)))
        h.update(name_bytes)
        h.update(dtype_bytes)
        if shape:
            h.update(struct.pack("<" + "Q" * len(shape), *shape))
        h.update(struct.pack("<Q", len(payload)))
        h.update(payload)
    return h.hexdigest()


def encode_quantized_safetensors(
    raw_c1: bytes,
    storage_dtype: str,
    limits: Limits = Limits(),
    *,
    profile: str | None = None,
    scheme: str | None = None,
    lib_path: str | Path | None = None,
) -> bytes:
    require(storage_dtype in SAFETENSORS_STORAGE_DTYPES, "unsupported safetensors quantized dtype: require F16 or BF16")
    state = safe.decode_c1(raw_c1, limits)
    load_native_codec(lib_path)
    names, chosen_scheme = _resolve_names(state, profile, scheme, "huggingface")
    encoder = native_encode_f16 if storage_dtype == "F16" else native_encode_bf16

    header: dict[str, object] = {}
    payloads: list[bytes] = []
    digest_entries: list[tuple[str, str, tuple[int, ...], bytes]] = []
    offset = 0
    for name, tensor in zip(names, state.tensors, strict=True):
        count = safe.elements(tensor.shape)
        encoded = encoder(tensor.data, count, lib_path=lib_path)
        require(len(encoded) <= limits.tensor, "safetensors tensor byte limit")
        header[name] = {
            "dtype": storage_dtype,
            "shape": tensor.shape,
            "data_offsets": (offset, offset + len(encoded)),
        }
        payloads.append(encoded)
        digest_entries.append((name, storage_dtype, tensor.shape, encoded))
        offset += len(encoded)

    manifest = safe.make_manifest(state, names)
    metadata: dict[str, str] = {
        QUANT_VERSION_KEY: "1.0",
        STORAGE_DTYPE_KEY: storage_dtype,
        LOSSY_KEY: "true",
        SOURCE_DIGEST_KEY: hashlib.sha256(raw_c1).hexdigest(),
        PAYLOAD_DIGEST_KEY: _payload_digest(manifest, digest_entries),
        safe.MANIFEST_KEY: manifest,
    }
    if profile is not None:
        assert chosen_scheme is not None
        metadata.update(mapping.mapping_fields(profile, chosen_scheme))
    header["__metadata__"] = metadata

    encoded_header = safe.json_bytes(header)
    encoded_header += b" " * (safe.padded_safetensors_header_size(len(encoded_header)) - len(encoded_header))
    require(len(encoded_header) <= limits.metadata, "safetensors header limit")
    require(8 + len(encoded_header) + offset <= limits.file, "safetensors file limit")
    return struct.pack("<Q", len(encoded_header)) + encoded_header + b"".join(payloads)


def _decode_quantized_safetensors_wire(
    raw: bytes, limits: Limits
) -> tuple[dict[str, tuple[str, tuple[int, ...], bytes]], dict[str, str]]:
    require(8 <= len(raw) <= limits.file, "safetensors file size limit")
    header_len = struct.unpack_from("<Q", raw)[0]
    require(2 <= header_len <= limits.metadata and header_len <= len(raw) - 8, "safetensors header span")
    require(raw[8:9] == b"{", "safetensors header must start with '{'")
    header = safe.parse_json(raw[8:8 + header_len], limits.metadata)
    require(type(header) is dict, "safetensors header object")
    metadata = header.pop("__metadata__", {})
    require(type(metadata) is dict and all(type(v) is str for v in metadata.values()), "safetensors string metadata")
    require(len(header) <= limits.entries, "tensor count limit")
    start = 8 + header_len
    descriptors: dict[str, tuple[str, tuple[int, ...], int, int]] = {}
    spans: list[tuple[int, int]] = []
    for name, descriptor in header.items():
        safe.text_bytes(name, 65536, "tensor name")
        require(type(descriptor) is dict and set(descriptor) == {"dtype", "shape", "data_offsets"},
                "tensor descriptor fields")
        dtype = descriptor["dtype"]
        require(dtype in SAFETENSORS_STORAGE_DTYPES, f"unsupported safetensors quantized dtype: {dtype}")
        shape, offsets = descriptor["shape"], descriptor["data_offsets"]
        require(type(shape) is list and type(offsets) is list and len(offsets) == 2,
                "tensor shape/offset arrays")
        shape_tuple = tuple(shape)
        count = safe.elements(shape_tuple)
        wire_size = count * 2
        restored_f32_size = count * 4
        begin, end = map(safe.uint, offsets)
        require(begin <= end and end <= len(raw) - start and end - begin == wire_size
                and wire_size <= limits.tensor and restored_f32_size <= limits.tensor,
                "tensor offset/shape span")
        descriptors[name] = (dtype, shape_tuple, begin, end)
        spans.append((begin, end))
    cursor = 0
    for begin, end in sorted(spans):
        require(begin == cursor, "noncontiguous tensor offsets")
        cursor = end
    require(cursor == len(raw) - start, "unindexed safetensors data")
    wire_tensors = {
        name: (dtype, shape_tuple, raw[start + begin:start + end])
        for name, (dtype, shape_tuple, begin, end) in descriptors.items()
    }
    return wire_tensors, metadata


def decode_quantized_safetensors(
    raw: bytes,
    limits: Limits = Limits(),
    *,
    profile: str | None = None,
    scheme: str | None = None,
    external: bool = False,
    provider: str | None = None,
    tie: bool = False,
    plain_provider: str | None = None,
    lib_path: str | Path | None = None,
) -> bytes:
    wire_tensors, metadata = _decode_quantized_safetensors_wire(raw, limits)
    load_native_codec(lib_path)
    reserved = mapping.io5_fields(metadata)
    if reserved:
        require(not external and plain_provider is None and provider is None and not tie,
                "external/plain import cannot override IO5 metadata")
        require(not (reserved & LOSSLESS_FORBIDDEN_KEYS),
                "lossy quantized safetensors must not carry lossless F32 IO5 keys")
        require(reserved in (SAFETENSORS_BASE_KEYS, SAFETENSORS_BASE_KEYS | mapping.MAPPING_FIELDS),
                "unsupported IO5 quantized safetensors metadata fields")
        require(metadata[QUANT_VERSION_KEY] == "1.0", "unsupported IO5 quant version")
        storage_dtype = metadata[STORAGE_DTYPE_KEY]
        require(storage_dtype in SAFETENSORS_STORAGE_DTYPES, "unsupported IO5 storage dtype")
        require(metadata[LOSSY_KEY] == "true", "lossy flag must be 'true' for quantized safetensors")
        require(SHA256_RE.fullmatch(metadata[SOURCE_DIGEST_KEY]) is not None, "invalid source C1 checksum")
        require(SHA256_RE.fullmatch(metadata[PAYLOAD_DIGEST_KEY]) is not None, "invalid payload checksum")

        selected_profile: str | None = None
        selected_scheme: str | None = None
        if mapping.MAPPING_VERSION in metadata:
            selected_profile, selected_scheme = mapping.selection_from_metadata(metadata, profile, scheme)
            _require_quantized_profile(selected_profile)
        else:
            require(profile is None and scheme is None, "unmapped quantized safetensors has no profile/scheme")

        # Validate every wire tensor dtype matches recorded storage_dtype before decoding.
        for name, (dtype, _, _) in wire_tensors.items():
            require(dtype == storage_dtype, f"tensor {name} dtype {dtype} disagrees with {storage_dtype}")

        # Parse manifest entry names to verify payload checksum in canonical entry order.
        manifest_obj = safe.parse_json(
            safe.text_bytes(metadata[safe.MANIFEST_KEY], limits.metadata, "IO5 manifest"),
            limits.metadata,
        )
        require(type(manifest_obj) is dict and set(manifest_obj) == {"provider", "entries", "aliases"},
                "IO5 manifest fields")
        entries = manifest_obj["entries"]
        require(type(entries) is list and len(entries) == len(wire_tensors), "IO5 manifest counts")
        ordered_digest_entries: list[tuple[str, str, tuple[int, ...], bytes]] = []
        seen_names: set[str] = set()
        for entry in entries:
            require(type(entry) is dict and "name" in entry and type(entry["name"]) is str, "IO5 entry fields")
            name = entry["name"]
            require(name in wire_tensors and name not in seen_names, "IO5 tensor name")
            seen_names.add(name)
            dtype_w, shape_w, payload_w = wire_tensors[name]
            ordered_digest_entries.append((name, dtype_w, shape_w, payload_w))
        require(_payload_digest(metadata[safe.MANIFEST_KEY], ordered_digest_entries) == metadata[PAYLOAD_DIGEST_KEY],
                "IO5 quantized safetensors payload SHA-256 mismatch")

        decoder = native_decode_f16 if storage_dtype == "F16" else native_decode_bf16
        named_f32: dict[str, Tensor] = {}
        for name, (_, shape_tuple, payload) in wire_tensors.items():
            f32_bytes = decoder(payload, safe.elements(shape_tuple), lib_path=lib_path)
            named_f32[name] = Tensor((name,), "parameter", shape_tuple, f32_bytes)

        state, names = safe.state_from_manifest(metadata[safe.MANIFEST_KEY], named_f32, limits)
        encoded = safe.encode_c1(state, limits)
        expected_names = (
            mapping.synthetic_names(state)
            if selected_profile is None
            else mapping.mapped_names(state, selected_profile, selected_scheme)
        )
        require(names == expected_names, "IO5 manifest tensor names do not match selected mapping")
        return encoded

    # Non-IO5 external/plain import.
    named_f32 = {}
    for name, (dtype, shape_tuple, payload) in wire_tensors.items():
        decoder = native_decode_f16 if dtype == "F16" else native_decode_bf16
        f32_bytes = decoder(payload, safe.elements(shape_tuple), lib_path=lib_path)
        named_f32[name] = Tensor((name,), "parameter", shape_tuple, f32_bytes)

    if external:
        require(plain_provider is None, "cannot combine --external with --plain-provider")
        _require_quantized_profile(profile)
        mapping.check_selection(profile, scheme)
        assert profile is not None and scheme is not None
        safe.text_bytes(provider, 127, "explicit external provider")
        assert provider is not None
        state = mapping.plain_state(named_f32, profile, scheme, provider, tie)
        encoded = safe.encode_c1(state, limits)
        mapping.mapped_names(state, profile, scheme)
        return encoded

    require(plain_provider is not None and profile is None and scheme is None and provider is None and not tie,
            "plain quantized safetensors requires explicit --plain-provider or --external")
    safe.text_bytes(plain_provider, 127, "provider")
    tensors = tuple(named_f32[name] for name in sorted(named_f32, key=lambda v: v.encode("utf-8")))
    return safe.encode_c1(State(plain_provider, tensors), limits)


def _gguf_validate_shape(shape: tuple[int, ...], limits: Limits) -> int:
    require(1 <= len(shape) <= 4, "unsupported GGUF rank: require 1..4")
    require(all(type(extent) is int and 0 < extent <= (1 << 63) - 1 for extent in shape),
            "unsupported GGUF extents: require positive signed-i64")
    count = safe.elements(shape)
    require(count * 4 <= limits.tensor, "GGUF restored F32 tensor byte limit")
    return count


def _gguf_wire_size(ggml_type: int, shape: tuple[int, ...], count: int, limits: Limits) -> int:
    if ggml_type == GGUF_TYPE_F32:
        size = count * 4
    elif ggml_type == GGUF_TYPE_F16:
        size = count * 2
    elif ggml_type == GGUF_TYPE_Q8_0:
        require(
            len(shape) >= 1 and shape[-1] > 0 and shape[-1] % QK8_0 == 0 and count > 0 and count % QK8_0 == 0,
            "GGUF Q8_0 tensor innermost dimension (ne[0] / shape[-1]) must be divisible by 32",
        )
        size = (count // QK8_0) * Q8_0_BLOCK_BYTES
    else:
        raise InteropError(f"unsupported GGUF tensor type: {ggml_type}")
    require(size <= limits.tensor, "GGUF wire tensor byte limit")
    return size


def encode_quantized_gguf(
    raw_c1: bytes,
    storage_dtype: str,
    limits: Limits = Limits(),
    *,
    q8_fallback: str = "F32",
    profile: str | None = None,
    scheme: str | None = None,
    lib_path: str | Path | None = None,
) -> bytes:
    require(storage_dtype in GGUF_STORAGE_DTYPES, "unsupported GGUF quantized dtype: require F16 or Q8_0")
    require(q8_fallback in Q8_FALLBACK_DTYPES, "unsupported Q8_0 fallback dtype: require F32 or F16")
    state = safe.decode_c1(raw_c1, limits)
    load_native_codec(lib_path)
    names, chosen_scheme = _resolve_names(state, profile, scheme, "gguf")

    infos: list[bytes] = []
    padded_payloads: list[bytes] = []
    digest_entries: list[tuple[str, str, tuple[int, ...], bytes]] = []
    tensor_dtypes: dict[str, str] = {}
    offset = 0

    for name, tensor in zip(names, state.tensors, strict=True):
        count = _gguf_validate_shape(tensor.shape, limits)
        safe.text_bytes(name, 63, "GGUF tensor name")
        require("\0" not in name, "GGUF tensor name")
        dtype_name = select_gguf_tensor_dtype(tensor.shape, storage_dtype, q8_fallback)
        ggml_type = NAME_TO_GGUF_TYPE[dtype_name]
        if dtype_name == "F32":
            native_validate_f32_finite(tensor.data, count, lib_path=lib_path)
            wire_data = tensor.data
        elif dtype_name == "F16":
            wire_data = native_encode_f16(tensor.data, count, lib_path=lib_path)
        else:
            wire_data = native_quantize_q8_0(tensor.data, count, lib_path=lib_path)
        expected_size = _gguf_wire_size(ggml_type, tensor.shape, count, limits)
        require(len(wire_data) == expected_size, "GGUF encoded payload size mismatch")

        tensor_dtypes[name] = dtype_name
        digest_entries.append((name, dtype_name, tensor.shape, wire_data))
        infos.append(
            gg.string(name)
            + struct.pack("<I", len(tensor.shape))
            + struct.pack("<" + "Q" * len(tensor.shape), *reversed(tensor.shape))
            + struct.pack("<IQ", ggml_type, offset)
        )
        padded = gg.aligned(len(wire_data))
        require(offset + padded <= limits.file, "GGUF payload/file limit")
        padded_payloads.extend((wire_data, bytes(padded - len(wire_data))))
        offset += padded

    manifest = safe.make_manifest(state, names)
    require(len(manifest) <= limits.metadata, "GGUF manifest limit")
    dtypes_json = safe.json_bytes(tensor_dtypes).decode("ascii")
    require(len(dtypes_json) <= limits.metadata, "GGUF tensor_dtypes limit")

    metadata: dict[str, tuple[int, str | int]] = {
        "general.architecture": (gg.STRING, "eshkol"),
        "general.alignment": (gg.UINT32, gg.ALIGNMENT),
        "general.file_type": (
            gg.UINT32,
            GGUF_FTYPE_MOSTLY_F16 if storage_dtype == "F16" else GGUF_FTYPE_MOSTLY_Q8_0,
        ),
        QUANT_VERSION_KEY: (gg.STRING, "1.0"),
        STORAGE_DTYPE_KEY: (gg.STRING, storage_dtype),
        LOSSY_KEY: (gg.STRING, "true"),
        TENSOR_DTYPES_KEY: (gg.STRING, dtypes_json),
        SOURCE_DIGEST_KEY: (gg.STRING, hashlib.sha256(raw_c1).hexdigest()),
        PAYLOAD_DIGEST_KEY: (gg.STRING, _payload_digest(manifest, digest_entries)),
        safe.MANIFEST_KEY: (gg.STRING, manifest),
    }
    if storage_dtype == "Q8_0":
        metadata["general.quantization_version"] = (gg.UINT32, GGUF_QUANTIZATION_VERSION)
        metadata[Q8_FALLBACK_KEY] = (gg.STRING, q8_fallback)
    if profile is not None:
        assert chosen_scheme is not None
        for key, value in mapping.mapping_fields(profile, chosen_scheme).items():
            metadata[key] = (gg.STRING, value)

    require(len(metadata) <= gg.MAX_METADATA_PAIRS, "GGUF metadata count limit")
    prefix = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(state.tensors), len(metadata)))
    for key, (kind, value) in sorted(metadata.items()):
        encoded = gg.string(key) + struct.pack("<I", kind)
        encoded += gg.string(value) if kind == gg.STRING else struct.pack("<I", value)
        require(len(prefix) + len(encoded) <= limits.metadata, "GGUF metadata/info limit")
        prefix += encoded
    for info in infos:
        require(len(prefix) + len(info) <= limits.metadata, "GGUF metadata/info limit")
        prefix += info
    data_at = gg.aligned(len(prefix))
    require(data_at <= limits.metadata and data_at + offset <= limits.file, "GGUF prefix/file limit")
    prefix += bytes(data_at - len(prefix))
    return bytes(prefix) + b"".join(padded_payloads)


def _decode_quantized_gguf_wire(
    raw: bytes, limits: Limits
) -> tuple[dict[str, tuple[int, tuple[int, ...], bytes]], dict[str, str | int]]:
    require(24 <= len(raw) <= limits.file, "GGUF file size limit")
    magic, version, tensor_count, metadata_count = struct.unpack_from("<4sIQQ", raw)
    require(magic == b"GGUF", "GGUF magic")
    require(version == 3, "unsupported GGUF version/endian: require v3 little endian")
    require(tensor_count <= limits.entries and metadata_count <= gg.MAX_METADATA_PAIRS, "GGUF count limit")

    reader = gg.Reader(raw, limits)
    metadata: dict[str, str | int] = {}
    for _ in range(metadata_count):
        key = reader.text(128, nonempty=True)
        require(gg.KEY.fullmatch(key) is not None and key not in metadata, "GGUF metadata key/duplicate")
        require(not key.startswith("split."), "unsupported GGUF shards")
        kind = reader.number("I")
        if key == "general.alignment":
            require(kind == gg.UINT32 and reader.number("I") == gg.ALIGNMENT,
                    "unsupported GGUF alignment: require UINT32 32")
            metadata[key] = gg.ALIGNMENT
        elif key == "general.file_type":
            require(kind == gg.UINT32, "unsupported GGUF general.file_type metadata type")
            ftype = reader.number("I")
            require(ftype in (GGUF_FTYPE_MOSTLY_F16, GGUF_FTYPE_MOSTLY_Q8_0),
                    f"unsupported GGUF general.file_type: {ftype}")
            metadata[key] = ftype
        elif key == "general.quantization_version":
            require(kind == gg.UINT32 and reader.number("I") == GGUF_QUANTIZATION_VERSION,
                    "unsupported GGUF general.quantization_version: require UINT32 2")
            metadata[key] = GGUF_QUANTIZATION_VERSION
        else:
            require(kind == gg.STRING, "unsupported GGUF metadata type: require STRING")
            metadata[key] = reader.text(limits.metadata)

    architecture = metadata.get("general.architecture")
    require(type(architecture) is str and re.fullmatch(r"[a-z0-9]+", architecture, re.ASCII) is not None,
            "GGUF architecture string required")

    descriptors: dict[str, tuple[int, tuple[int, ...], int, int]] = {}
    for _ in range(tensor_count):
        name = reader.text(63, nonempty=True)
        require("\0" not in name and name not in descriptors, "GGUF tensor name/duplicate")
        rank = reader.number("I")
        require(1 <= rank <= 4, "unsupported GGUF rank: require 1..4")
        shape = tuple(reversed(tuple(reader.number("Q") for _ in range(rank))))
        count = _gguf_validate_shape(shape, limits)
        ggml_type = reader.number("I")
        require(ggml_type in GGUF_TYPE_TO_NAME, f"unsupported GGUF tensor type: {ggml_type}")
        wire_size = _gguf_wire_size(ggml_type, shape, count, limits)
        offset = reader.number("Q")
        require(offset % gg.ALIGNMENT == 0, "GGUF tensor offset alignment")
        descriptors[name] = (ggml_type, shape, offset, wire_size)

    data_at = gg.aligned(reader.at)
    require(data_at <= len(raw) and data_at <= limits.metadata, "GGUF data start span/limit")
    require(not any(raw[reader.at:data_at]), "GGUF nonzero prefix padding")

    cursor = 0
    for _, _, offset, wire_size in descriptors.values():
        require(offset == cursor, "GGUF noncontiguous tensor offsets")
        padded = gg.aligned(wire_size)
        require(offset <= len(raw) - data_at and padded <= len(raw) - data_at - offset,
                "GGUF tensor data/padding span")
        end = offset + wire_size
        cursor = offset + padded
        require(not any(raw[data_at + end:data_at + cursor]), "GGUF nonzero tensor padding")
    require(data_at + cursor == len(raw), "GGUF exact EOF")

    wire_tensors = {
        name: (ggml_type, shape, raw[data_at + offset:data_at + offset + wire_size])
        for name, (ggml_type, shape, offset, wire_size) in descriptors.items()
    }
    return wire_tensors, metadata


def _decode_gguf_payload(ggml_type: int, shape: tuple[int, ...], payload: bytes, *,
                         lib_path: str | Path | None = None) -> bytes:
    count = safe.elements(shape)
    if ggml_type == GGUF_TYPE_F32:
        native_validate_f32_finite(payload, count, lib_path=lib_path)
        return payload
    if ggml_type == GGUF_TYPE_F16:
        return native_decode_f16(payload, count, lib_path=lib_path)
    if ggml_type == GGUF_TYPE_Q8_0:
        require(
            len(shape) >= 1 and shape[-1] > 0 and shape[-1] % QK8_0 == 0,
            "GGUF Q8_0 tensor innermost dimension (ne[0] / shape[-1]) must be divisible by 32",
        )
        return native_dequantize_q8_0(payload, count, lib_path=lib_path)
    raise InteropError(f"unsupported GGUF tensor type: {ggml_type}")


def decode_quantized_gguf(
    raw: bytes,
    limits: Limits = Limits(),
    *,
    profile: str | None = None,
    scheme: str | None = None,
    external: bool = False,
    provider: str | None = None,
    tie: bool = False,
    plain_provider: str | None = None,
    lib_path: str | Path | None = None,
) -> bytes:
    wire_tensors, metadata = _decode_quantized_gguf_wire(raw, limits)
    load_native_codec(lib_path)
    reserved = mapping.io5_fields(metadata)
    if reserved:
        require(not external and plain_provider is None and provider is None and not tie,
                "external/plain import cannot override IO5 metadata")
        require(not (reserved & LOSSLESS_FORBIDDEN_KEYS),
                "lossy quantized GGUF must not carry lossless F32 IO5 keys")
        require(metadata["general.architecture"] == "eshkol", "IO5 GGUF architecture profile")
        storage_dtype = metadata.get(STORAGE_DTYPE_KEY)
        require(storage_dtype in GGUF_STORAGE_DTYPES, "unsupported IO5 GGUF storage dtype")
        expected_base = set(GGUF_BASE_KEYS)
        if storage_dtype == "Q8_0":
            expected_base.add(Q8_FALLBACK_KEY)
        require(reserved in (expected_base, expected_base | mapping.MAPPING_FIELDS),
                "unsupported IO5 quantized GGUF metadata fields")
        require(metadata[QUANT_VERSION_KEY] == "1.0", "unsupported IO5 quant version")
        require(metadata[LOSSY_KEY] == "true", "lossy flag must be 'true' for quantized GGUF")
        require(type(metadata[SOURCE_DIGEST_KEY]) is str and SHA256_RE.fullmatch(metadata[SOURCE_DIGEST_KEY]) is not None,
                "invalid source C1 checksum")
        require(type(metadata[PAYLOAD_DIGEST_KEY]) is str and SHA256_RE.fullmatch(metadata[PAYLOAD_DIGEST_KEY]) is not None,
                "invalid payload checksum")

        q8_fallback = "F32"
        if storage_dtype == "Q8_0":
            q8_fallback_val = metadata[Q8_FALLBACK_KEY]
            require(type(q8_fallback_val) is str and q8_fallback_val in Q8_FALLBACK_DTYPES,
                    "invalid Q8_0 fallback dtype metadata")
            q8_fallback = q8_fallback_val
            require(metadata.get("general.file_type") == GGUF_FTYPE_MOSTLY_Q8_0,
                    "IO5 Q8_0 GGUF requires general.file_type = 7")
            require(metadata.get("general.quantization_version") == GGUF_QUANTIZATION_VERSION,
                    "IO5 Q8_0 GGUF requires general.quantization_version = 2")
        else:
            require(metadata.get("general.file_type") == GGUF_FTYPE_MOSTLY_F16,
                    "IO5 F16 GGUF requires general.file_type = 1")
            require("general.quantization_version" not in metadata,
                    "IO5 F16 GGUF must not set general.quantization_version")

        selected_profile: str | None = None
        selected_scheme: str | None = None
        if mapping.MAPPING_VERSION in metadata:
            selected_profile, selected_scheme = mapping.selection_from_metadata(metadata, profile, scheme)
            _require_quantized_profile(selected_profile)
        else:
            require(profile is None and scheme is None, "unmapped quantized GGUF has no profile/scheme")

        dtypes_text = metadata[TENSOR_DTYPES_KEY]
        require(type(dtypes_text) is str, "tensor_dtypes metadata must be string")
        dtypes_map = safe.parse_json(safe.text_bytes(dtypes_text, limits.metadata, "tensor_dtypes"), limits.metadata)
        require(type(dtypes_map) is dict and set(dtypes_map) == set(wire_tensors),
                "tensor_dtypes map does not match GGUF tensor roster")
        for name, (ggml_type, shape, _) in wire_tensors.items():
            recorded = dtypes_map[name]
            expected_policy_dtype = select_gguf_tensor_dtype(shape, storage_dtype, q8_fallback)
            require(recorded == GGUF_TYPE_TO_NAME[ggml_type],
                    f"tensor {name} recorded dtype {recorded} disagrees with wire type {ggml_type}")
            require(recorded == expected_policy_dtype,
                    f"tensor {name} dtype {recorded} violates explicit {storage_dtype}/{q8_fallback} policy")

        manifest_text = metadata[safe.MANIFEST_KEY]
        require(type(manifest_text) is str, "IO5 manifest must be string")
        manifest_obj = safe.parse_json(
            safe.text_bytes(manifest_text, limits.metadata, "IO5 manifest"),
            limits.metadata,
        )
        require(type(manifest_obj) is dict and set(manifest_obj) == {"provider", "entries", "aliases"},
                "IO5 manifest fields")
        entries = manifest_obj["entries"]
        require(type(entries) is list and len(entries) == len(wire_tensors), "IO5 manifest counts")
        ordered_digest_entries: list[tuple[str, str, tuple[int, ...], bytes]] = []
        seen_names: set[str] = set()
        for entry in entries:
            require(type(entry) is dict and "name" in entry and type(entry["name"]) is str, "IO5 entry fields")
            name = entry["name"]
            require(name in wire_tensors and name not in seen_names, "IO5 tensor name")
            seen_names.add(name)
            ggml_type_w, shape_w, payload_w = wire_tensors[name]
            ordered_digest_entries.append((name, GGUF_TYPE_TO_NAME[ggml_type_w], shape_w, payload_w))
        require(_payload_digest(manifest_text, ordered_digest_entries) == metadata[PAYLOAD_DIGEST_KEY],
                "IO5 quantized GGUF payload SHA-256 mismatch")

        named_f32: dict[str, Tensor] = {}
        for name, (ggml_type, shape, payload) in wire_tensors.items():
            f32_bytes = _decode_gguf_payload(ggml_type, shape, payload, lib_path=lib_path)
            named_f32[name] = Tensor((name,), "parameter", shape, f32_bytes)

        state, names = safe.state_from_manifest(manifest_text, named_f32, limits)
        encoded = safe.encode_c1(state, limits)
        expected_names = (
            mapping.synthetic_names(state)
            if selected_profile is None
            else mapping.mapped_names(state, selected_profile, selected_scheme)
        )
        require(names == expected_names, "IO5 manifest tensor names do not match selected mapping")
        return encoded

    # Non-IO5 external/plain import.
    named_f32 = {}
    for name, (ggml_type, shape, payload) in wire_tensors.items():
        f32_bytes = _decode_gguf_payload(ggml_type, shape, payload, lib_path=lib_path)
        named_f32[name] = Tensor((name,), "parameter", shape, f32_bytes)

    if external:
        require(plain_provider is None, "cannot combine --external with --plain-provider")
        _require_quantized_profile(profile)
        mapping.check_selection(profile, scheme)
        assert profile is not None and scheme is not None
        safe.text_bytes(provider, 127, "explicit external provider")
        assert provider is not None
        state = mapping.plain_state(named_f32, profile, scheme, provider, tie)
        encoded = safe.encode_c1(state, limits)
        mapping.mapped_names(state, profile, scheme)
        return encoded

    require(plain_provider is not None and profile is None and scheme is None and provider is None and not tie,
            "plain quantized GGUF requires explicit --plain-provider or --external")
    safe.text_bytes(plain_provider, 127, "provider")
    tensors = tuple(named_f32[name] for name in sorted(named_f32, key=lambda v: v.encode("utf-8")))
    return safe.encode_c1(State(plain_provider, tensors), limits)


def measure_reconstruction_error(
    original_c1: bytes, restored_c1: bytes, limits: Limits = Limits()
) -> dict[str, dict[str, float]]:
    orig_state = safe.decode_c1(original_c1, limits)
    rest_state = safe.decode_c1(restored_c1, limits)
    require(len(orig_state.tensors) == len(rest_state.tensors), "tensor count mismatch")
    require(orig_state.aliases == rest_state.aliases, "alias topology mismatch")

    metrics: dict[str, dict[str, float]] = {}
    overall_max_abs = 0.0
    overall_max_rel = 0.0
    for orig, rest in zip(orig_state.tensors, rest_state.tensors, strict=True):
        require(orig.path == rest.path and orig.kind == rest.kind and orig.shape == rest.shape,
                "tensor metadata mismatch")
        count = safe.elements(orig.shape)
        orig_vals = struct.unpack(f"<{count}f", orig.data) if count else ()
        rest_vals = struct.unpack(f"<{count}f", rest.data) if count else ()
        max_abs = 0.0
        max_rel = 0.0
        sum_sq_err = 0.0
        sum_sq_ref = 0.0
        for a, b in zip(orig_vals, rest_vals, strict=True):
            require(math.isfinite(a) and math.isfinite(b), "non-finite value in error measurement")
            err = abs(a - b)
            if err > max_abs:
                max_abs = err
            if abs(a) > 0.0:
                rel = err / abs(a)
                if rel > max_rel:
                    max_rel = rel
            sum_sq_err += err * err
            sum_sq_ref += a * a
        rel_l2 = math.sqrt(sum_sq_err / sum_sq_ref) if sum_sq_ref > 0.0 else 0.0
        key = ".".join(orig.path)
        metrics[key] = {"max_abs": max_abs, "max_rel": max_rel, "rel_l2": rel_l2}
        overall_max_abs = max(overall_max_abs, max_abs)
        overall_max_rel = max(overall_max_rel, max_rel)
    metrics["__overall__"] = {"max_abs": overall_max_abs, "max_rel": overall_max_rel}
    return metrics


def _add_limits(parser: argparse.ArgumentParser) -> None:
    for name, value in vars(Limits()).items():
        parser.add_argument("--max-" + name, type=int, default=value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_st_exp = sub.add_parser("c1-to-safetensors")
    p_st_exp.add_argument("source", type=Path)
    p_st_exp.add_argument("destination", type=Path)
    p_st_exp.add_argument("--dtype", choices=SAFETENSORS_STORAGE_DTYPES, required=True)
    p_st_exp.add_argument("--profile", choices=QUANTIZED_PROFILES)
    p_st_exp.add_argument("--scheme", choices=mapping.SCHEMES)
    p_st_exp.add_argument("--overwrite", action="store_true")
    _add_limits(p_st_exp)

    p_st_imp = sub.add_parser("safetensors-to-c1")
    p_st_imp.add_argument("source", type=Path)
    p_st_imp.add_argument("destination", type=Path)
    p_st_imp.add_argument("--profile", choices=QUANTIZED_PROFILES)
    p_st_imp.add_argument("--scheme", choices=mapping.SCHEMES)
    p_st_imp.add_argument("--external", action="store_true")
    p_st_imp.add_argument("--provider")
    p_st_imp.add_argument("--tie-head-to-embedding", action="store_true")
    p_st_imp.add_argument("--plain-provider")
    p_st_imp.add_argument("--overwrite", action="store_true")
    _add_limits(p_st_imp)

    p_gg_exp = sub.add_parser("c1-to-gguf")
    p_gg_exp.add_argument("source", type=Path)
    p_gg_exp.add_argument("destination", type=Path)
    p_gg_exp.add_argument("--dtype", choices=GGUF_STORAGE_DTYPES, required=True)
    p_gg_exp.add_argument("--q8-fallback", choices=Q8_FALLBACK_DTYPES, default="F32")
    p_gg_exp.add_argument("--profile", choices=QUANTIZED_PROFILES)
    p_gg_exp.add_argument("--scheme", choices=mapping.SCHEMES)
    p_gg_exp.add_argument("--overwrite", action="store_true")
    _add_limits(p_gg_exp)

    p_gg_imp = sub.add_parser("gguf-to-c1")
    p_gg_imp.add_argument("source", type=Path)
    p_gg_imp.add_argument("destination", type=Path)
    p_gg_imp.add_argument("--profile", choices=QUANTIZED_PROFILES)
    p_gg_imp.add_argument("--scheme", choices=mapping.SCHEMES)
    p_gg_imp.add_argument("--external", action="store_true")
    p_gg_imp.add_argument("--provider")
    p_gg_imp.add_argument("--tie-head-to-embedding", action="store_true")
    p_gg_imp.add_argument("--plain-provider")
    p_gg_imp.add_argument("--overwrite", action="store_true")
    _add_limits(p_gg_imp)

    args = parser.parse_args()
    try:
        limits = Limits(**{name: getattr(args, "max_" + name) for name in vars(Limits())})
        raw = safe.read_file(args.source, limits)
        if args.command == "c1-to-safetensors":
            result = encode_quantized_safetensors(
                raw, args.dtype, limits, profile=args.profile, scheme=args.scheme
            )
        elif args.command == "safetensors-to-c1":
            result = decode_quantized_safetensors(
                raw, limits, profile=args.profile, scheme=args.scheme,
                external=args.external, provider=args.provider,
                tie=args.tie_head_to_embedding, plain_provider=args.plain_provider,
            )
        elif args.command == "c1-to-gguf":
            result = encode_quantized_gguf(
                raw, args.dtype, limits, q8_fallback=args.q8_fallback,
                profile=args.profile, scheme=args.scheme,
            )
        else:
            result = decode_quantized_gguf(
                raw, limits, profile=args.profile, scheme=args.scheme,
                external=args.external, provider=args.provider,
                tie=args.tie_head_to_embedding, plain_provider=args.plain_provider,
            )
        safe.publish(args.destination, result, args.overwrite)
        return 0
    except (InteropError, OSError) as error:
        print(f"IO5 quantized interchange error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
