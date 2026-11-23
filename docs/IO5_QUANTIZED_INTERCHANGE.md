# IO5 mixed-precision and quantized weight interchange 1.0

This IO5 extension provides bounded, non-executable weight interchange between
canonical [C1 1.0](CHECKPOINT_FORMAT.md) CPU dense contiguous F32 checkpoints
and lossy narrow/quantized wire formats:

- **Safetensors** `F16` and `BF16` (2 bytes/element, 2:1 payload ratio vs F32),
  preserving `eshkol-m3-v1`, `eshkol-decoder-v1`, `eshkol-mod5-v1`, and unmapped
  C1 checkpoints.
- **GGUF v3** `F16` (GGML type `1`) and `Q8_0` (GGML type `8`, 34 bytes per 32
  elements), with an explicit non-padding policy for 1D norms/biases and
  non-multiple-of-32 tensors.

All hot numerical conversions run in reviewed native C11
(`native/io5_quant_codec.c`, ABI header `include/eshkol_transformer/io5_quant_abi.h`)
compiled into deterministic `libeshkol_transformer_io5_quant.a` and
`libeshkol_transformer_io5_quant.so`. The Python interchange tool
(`tools/io5/quantized_interchange.py`) loads the native shared library directly
and fails closed if it is absent; there is no silent pure-Python fallback.

## Native C11 codec ABI (`io5_quant_abi.h`)

The native codec is C11 and C++17 compatible (`extern "C"`), performs zero heap
allocations (verified with symbol audit and `--wrap=malloc/calloc/realloc/free`),
operates on little-endian byte buffers with zero misaligned-pointer UB, and
validates every element/block before writing any destination bytes.

- **`f32 <-> f16`** (`et_io5_encode_f16`, `et_io5_decode_f16`, scalar bit helpers):
  IEEE-754 binary16 round-to-nearest-even (RNE) matching [MP4](MP4_MIXED_PRECISION.md)
  bit semantics. Preserves signed zero and subnormals. Rejects all non-finite
  `f32` inputs, all finite `f32` inputs with `|x| > 65504.0` (`0x477fe000`), and
  all 2,048 non-finite `f16` bit patterns on decode.
- **`f32 <-> bf16`** (`et_io5_encode_bf16`, `et_io5_decode_bf16`, scalar bit helpers):
  IEEE-754 bfloat16 RNE matching MP4 bit semantics. Rejects all non-finite `f32`
  inputs, finite inputs whose rounded encoding is bfloat16 infinity
  (`|bits| >= 0x7f7f8000`), and all 256 non-finite `bf16` bit patterns on decode.
- **GGML `Q8_0`** (`et_io5_quantize_q8_0`, `et_io5_dequantize_q8_0`,
  `et_io5_q8_0_byte_size`):
  - Block geometry: `ET_IO5_QK8_0 = 32`, `ET_IO5_Q8_0_BLOCK_BYTES = 34`
    (2-byte little-endian IEEE-754 `f16` scale `d` + 32 signed `int8_t` quants
    `qs[32]`).
  - Quantization: `d = f16_rne(max(|x|) / 127.0)`. Zero blocks emit `d = 0x0000`
    and `qs[0..31] = 0`. Blocks with `max(|x|) / 127.0 > 65504.0` reject with
    `ET_IO5_QUANT_ERR_OVERFLOW`. Non-zero finite scales compute
    `q = clamp(rne(x / f16_to_f32(d)), -127, 127)`.
  - Dequantization: validates every block's `f16` scale is finite before writing
    outputs, then computes `x_hat = f16_to_f32(d) * (float)qs[i]`.
  - Rejects `element_count == 0`, non-multiple-of-32 `element_count`, mismatched
    byte lengths, `NULL` buffers, and aliased/overlapping buffers.
- **Finite F32 validator** (`et_io5_validate_f32_finite`): rejects NaN/Inf in
  `F32` fallback tensors retained in `Q8_0` GGUF files.

## Explicit lossy-dtype policy and wire metadata

Lossy storage formats (`F16`, `BF16`, `Q8_0`) **never** claim lossless C1 byte
equality. Lossless IO5 checksum keys (`eshkol.io5.c1-sha256`,
`eshkol.io5.c1_sha256`, `eshkol.io5.version`, `eshkol.io5.gguf_version`) are
explicitly forbidden in quantized/mixed-precision artifacts.

### Safetensors (`F16` / `BF16`)

| Metadata key | Value |
|---|---|
| `eshkol.io5.quant_version` | `1.0` |
| `eshkol.io5.storage_dtype` | `F16` or `BF16` |
| `eshkol.io5.lossy` | `true` |
| `eshkol.io5.source_c1_sha256` | Lowercase hex SHA-256 of the source canonical F32 C1 file (provenance only) |
| `eshkol.io5.payload_sha256` | Lowercase hex SHA-256 of serialized narrow payloads in manifest order |
| `eshkol.io5.manifest` | JSON object with `provider`, `entries`, `aliases` |
| `eshkol.io5.mapping_version`, `eshkol.io5.profile`, `eshkol.io5.scheme` | Present together when mapped with a profile |

### GGUF v3 (`F16` / `Q8_0`)

- Little-endian GGUF v3, `general.architecture = "eshkol"`,
  `general.alignment = UINT32(32)`, contiguous 32-byte aligned tensor offsets,
  zero prefix/inter-tensor/final padding, reversed GGML dimension order.
- `general.file_type = UINT32(1)` (`MOSTLY_F16`) for `F16` exports;
  `general.file_type = UINT32(7)` (`MOSTLY_Q8_0`) and
  `general.quantization_version = UINT32(2)` for `Q8_0` exports.
- **Explicit `Q8_0` tensor selection policy (zero silent padding)**:
  - 1D tensors (norms and biases) stay in the explicit `--q8-fallback` dtype
    (`F32` default, or `F16`).
  - Rank 2–4 tensors whose element count is divisible by 32 are quantized to
    `Q8_0` (type `8`).
  - Rank 2–4 tensors whose element count is not divisible by 32 stay in
    `--q8-fallback` (`F32` or `F16`) and are never silently padded.
  - `eshkol.io5.tensor_dtypes` records the exact wire dtype (`F32`, `F16`, or
    `Q8_0`) of every tensor; import verifies every wire descriptor against both
    `eshkol.io5.tensor_dtypes` and the declared policy.

## Rejection rules

Import and export fail closed (CLI exit code `2`) on:

- Missing native C11 shared library or ABI version mismatch.
- Truncated or oversize `Q8_0` blocks, non-multiple-of-32 `Q8_0` element counts,
  and non-finite (`±Inf`, `qNaN`, `sNaN`) `f16` block scales.
- Non-finite `F32`/`F16`/`BF16` values, `F16` overflow (`|x| > 65504.0`), `BF16`
  infinity-rounding overflow, and `Q8_0` scale overflow (`|x| > 8319008.0`).
- Unsupported GGUF tensor types (anything outside `0=F32`, `1=F16`, `8=Q8_0`),
  unsupported safetensors dtypes (anything outside `F16`, `BF16`), non-32
  alignment, unaligned tensor offsets, nonzero padding, trailing bytes, and
  `split.*` metadata.
- Payload SHA-256 mismatch, `eshkol.io5.lossy != "true"`, injected lossless F32
  IO5 keys, invalid/incomplete profile rosters, and mismatched aliased tensors.

## Commands and validation

```sh
bash scripts/build-io5-quant.sh
bash scripts/test-io5-quant.sh

python3 tools/io5/quantized_interchange.py c1-to-safetensors model.c1 model_bf16.safetensors \
  --dtype BF16 --profile eshkol-mod5-v1
python3 tools/io5/quantized_interchange.py safetensors-to-c1 model_bf16.safetensors restored.c1
python3 tools/io5/quantized_interchange.py c1-to-gguf model.c1 model_q8.gguf \
  --dtype Q8_0 --q8-fallback F32 --profile eshkol-mod5-v1
python3 tools/io5/quantized_interchange.py gguf-to-c1 model_q8.gguf restored_q8.c1
```

## Measured validation evidence

Validated in `eshkol-r0-debug:20260928` (Ubuntu 22.04.5 x86-64, Clang 21.1.8) via
`CC=clang-21 CXX=clang++-21 bash scripts/test-io5-quant.sh`:

- Deterministic native artifacts (reproduced byte-identically across two builds,
  zero allocator/math/threading imports):
  - `libeshkol_transformer_io5_quant.a` SHA-256:
    `f288e25b379dec6508ec5d7fa4bdf95113ef82b30a9abf5e38fe711f96f8bb65`
  - `libeshkol_transformer_io5_quant.so` SHA-256:
    `9a712c3e4f01bf79180603664ce1f0c556a0468d022140380923c5bbdab609e6`
- C++17 header/linkage check (`tests/io5/header_quant_cpp.cpp`): PASS.
- C11 exhaustive native suite (`tests/io5/test_io5_quant_native.c`: all 65,536
  `f16` and `bf16` bit patterns, unaligned buffers, Q8_0 round-trip & edge
  bounds, aliasing/length/overflow/non-finite rejections): PASS in normal mode
  (with `--wrap=malloc/calloc/realloc/free` enforcing zero heap allocations) and
  under ASan/UBSan/LSan (`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`,
  `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`).
- Python quantized interchange suite (`tests/io5/test_quantized_interchange.py`,
  8 test groups): PASS.
- Q0 Python isolation (`tests/q0/test_python_isolation.py`) and CI topology
  (`scripts/check-ci-topology.sh`): PASS.

Measured round-trip reconstruction errors on deterministic non-dyadic finite
checkpoints (`tests/io5/test_quantized_interchange.py`):

| Profile | Format / Dtype | Source C1 (bytes) | Wire (bytes) | Max Abs Error | Max Rel Error |
|---|---|---:|---:|---:|---:|
| `eshkol-m3-v1` | Safetensors `F16` | 10,941 | 8,088 | `4.874e-04` | `4.729e-04` |
| `eshkol-m3-v1` | Safetensors `BF16` | 10,941 | 8,104 | `3.862e-03` | `3.760e-03` |
| `eshkol-m3-v1` | GGUF v3 `F16` | 10,941 | 8,000 | `4.874e-04` | `4.729e-04` |
| `eshkol-m3-v1` | GGUF v3 `Q8_0` (`F32` fallback) | 10,941 | 6,304 | `6.412e-03` | `1.000e+00` (near-zero `q=0`; block `rel_l2 < 1e-2`) |
| `eshkol-m3-v1` | GGUF v3 `Q8_0` (`F16` fallback) | 10,941 | 6,176 | `6.412e-03` | `1.000e+00` (near-zero `q=0`; block `rel_l2 < 1e-2`) |
| `eshkol-mod5-v1` (2L) | Safetensors `F16` | 22,764 | 15,104 | `4.883e-04` | `4.774e-04` |
| `eshkol-mod5-v1` (2L) | Safetensors `BF16` | 22,764 | 15,128 | `3.906e-03` | `3.731e-03` |
| `eshkol-mod5-v1` (2L) | GGUF v3 `F16` | 22,764 | 14,496 | `4.883e-04` | `4.774e-04` |
| `eshkol-mod5-v1` (2L) | GGUF v3 `Q8_0` (`F32` fallback) | 22,764 | 10,400 | `7.384e-03` | `1.000e+00` (near-zero `q=0`; block `rel_l2 < 1e-2`) |
| `eshkol-mod5-v1` (2L) | GGUF v3 `Q8_0` (`F16` fallback) | 22,764 | 10,240 | `7.384e-03` | `1.000e+00` (near-zero `q=0`; block `rel_l2 < 1e-2`) |

## Limitations and scope

- Bounded non-streaming interchange (reuses IO5 64 MiB file / 32 MiB tensor /
  1 MiB metadata limits); peak memory and throughput are unmeasured.
- Only `F16` and `BF16` safetensors and `F32`/`F16`/`Q8_0` GGUF v3 are admitted;
  4-bit/sub-byte GGML quantizers (`Q4_0`, `Q4_K`, etc.), sharded quantized
  indexes, live runtime dequantized matmul kernels, and GPU execution are
  outside this slice.
