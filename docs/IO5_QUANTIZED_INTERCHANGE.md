# IO5 mixed-precision and quantized weight interchange 1.0

This IO5 extension provides bounded, non-executable weight interchange between
canonical [C1 1.0](CHECKPOINT_FORMAT.md) CPU dense contiguous F32 checkpoints
and lossy narrow/quantized wire formats:

- **Safetensors** `F16` and `BF16` (2 bytes/element, 2:1 payload ratio vs F32),
  with the merged `eshkol-m3-v1` and `eshkol-decoder-v1` mappings, or
  unmapped C1 state.
- **GGUF v3** `F16` (GGML type `1`) and `Q8_0` (GGML type `8`, 34 bytes per 32
  elements), with an explicit non-padding policy for 1D norms/biases and
  tensors whose innermost GGML row dimension (`ne[0]` / C1 `shape[-1]`) is not
  divisible by 32.

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
  IEEE-754 binary16 round-to-nearest-even (RNE). Preserves signed zero and
  subnormals. Rejects all non-finite
  `f32` inputs, all finite `f32` inputs with `|x| > 65504.0` (`0x477fe000`), and
  all 2,048 non-finite `f16` bit patterns on decode.
- **`f32 <-> bf16`** (`et_io5_encode_bf16`, `et_io5_decode_bf16`, scalar bit helpers):
  IEEE-754 bfloat16 RNE. Rejects all non-finite `f32`
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
    Exact F16 midpoint and integer-quant thresholds make the wire result
    independent of the caller's floating-point rounding mode.
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
| `eshkol.io5.payload_sha256` | Lowercase hex SHA-256 over domain-separated length-framed `eshkol.io5.manifest` and per-tensor `(name, dtype, shape, payload)` entries in manifest order |
| `eshkol.io5.manifest` | JSON object with `provider`, `entries`, `aliases` |
| `eshkol.io5.mapping_version`, `eshkol.io5.profile`, `eshkol.io5.scheme` | Present together when mapped with a profile |

### GGUF v3 (`F16` / `Q8_0`)

- Little-endian GGUF v3, `general.architecture = "eshkol"`,
  `general.alignment = UINT32(32)`, contiguous 32-byte aligned tensor offsets,
  zero prefix/inter-tensor/final padding, reversed GGML dimension order.
- `general.file_type = UINT32(1)` (`MOSTLY_F16`) for `F16` exports;
  `general.file_type = UINT32(7)` (`MOSTLY_Q8_0`) and
  `general.quantization_version = UINT32(2)` for `Q8_0` exports.
- **Explicit GGML `Q8_0` tensor selection policy (zero silent padding)**:
  - 1D tensors (norms and biases) stay in the explicit `--q8-fallback` dtype
    (`F32` default, or `F16`).
  - Rank 2–4 tensors whose innermost row dimension (`ne[0]` in GGML order,
    matching C1 `shape[-1]`) is divisible by 32 are quantized to `Q8_0`
    (type `8`), ensuring GGML row-aligned blocks (`nb[1] = (ne[0] / 32) * 34`)
    never cross row boundaries.
  - Rank 2–4 tensors whose innermost dimension (`ne[0]` / `shape[-1]`) is not
    divisible by 32 — even when total `prod(shape)` is divisible by 32, e.g.,
    `(32, 16)` or `(32, 48)` — stay in `--q8-fallback` (`F32` or `F16`) and are
    never silently padded.
  - `eshkol.io5.tensor_dtypes` records the exact wire dtype (`F32`, `F16`, or
    `Q8_0`) of every tensor; import verifies every wire descriptor against both
    `eshkol.io5.tensor_dtypes` and the declared policy, and rejects any GGUF
    `Q8_0` tensor (IO5 or external) whose `ne[0] % 32 != 0`.

## Rejection rules

Import and export fail closed (CLI exit code `2`) on:

- Missing native C11 shared library, missing required ABI symbols, or ABI
  version mismatch.
- Truncated or oversize `Q8_0` blocks, non-multiple-of-32 `Q8_0` innermost
  dimensions (`ne[0]` / `shape[-1]`) or element counts, and non-finite (`±Inf`,
  `qNaN`, `sNaN`) `f16` block scales.
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
  --dtype BF16 --profile eshkol-decoder-v1
python3 tools/io5/quantized_interchange.py safetensors-to-c1 model_bf16.safetensors restored.c1
python3 tools/io5/quantized_interchange.py c1-to-gguf model.c1 model_q8.gguf \
  --dtype Q8_0 --q8-fallback F32 --profile eshkol-decoder-v1
python3 tools/io5/quantized_interchange.py gguf-to-c1 model_q8.gguf restored_q8.c1
```

## Focused validation evidence

The stacked candidate was tested locally with Clang 22.1.8 and a separate
Ubuntu 22.04 LLVM 21.1.8 container for the native and GGML checks:

- `CC=clang CXX=clang++ bash scripts/test-io5-quant.sh`: deterministic static
  and shared library rebuilds, export/import symbol audit, C++17 header/linkage,
  allocation-guarded C11 numerical tests, ASan/UBSan/LSan, and eight Python
  interchange groups passed. The static and shared library SHA-256 values were
  `df68378a7a040bc01a56866b684e6f2d702c0fd36610c128a8e7b85fbbbe5f7f`
  and `ee06b71662f5e6efbfe248a6030568784b0019353d84f7cc12e86f2c9db59153`
  for the final Clang 22 tree. The LLVM 21 archive/shared hashes were
  `7e1fcfe52fcf4affc9b1c7562fd56e7321c14004b9c759d4586c1366c0b254bc`
  and `771ab9265bf2cc5cb56822e59876561e61077ffee52f437ce201fd5567cae90f`.
- `scripts/test-io5-q8-ggml-decoder.sh` passed against the source-pinned GGML
  decoder: all 192 F32 output bits across six raw Q8_0 blocks agreed. The
  witness checks native GGUF type, shape and 204 payload bytes. It does not
  claim encoder bit identity.
- `IO5_GGUF_REFERENCE_PYTHON=... IO5_GGUF_NATIVE_PREFIX=...
  IO5_GGML_SOURCE_ARCHIVE=... bash scripts/test-io5-gguf.sh` passed all 42
  IO5 and five Python-isolation groups
  with pinned `gguf==0.19.0`, `numpy==2.5.4`, `safetensors==0.6.2`, and the
  selected GGML native reader/writer. These checks include the existing
  lossless IO5 regression suite.
- The tests cover the merged M3 and decoder mappings, a bare state, framed
  payload checksums, reshape/manifest/byte corruption, invalid scale and
  geometry, explicit Q8_0 fallback, unavailable native library, and an
  MOD5 profile requests that fail closed despite the separate lossless mapping.

Measured examples from deterministic finite C1 fixtures with the default F32
Q8_0 fallback:

| State | Format / dtype | C1 bytes | Wire bytes | Max absolute error |
|---|---|---:|---:|---:|
| M3 | safetensors F16 | 10,941 | 8,088 | 4.874e-4 |
| M3 | safetensors BF16 | 10,941 | 8,104 | 3.862e-3 |
| M3 | GGUF F16 | 10,941 | 8,000 | 4.874e-4 |
| M3 | GGUF Q8_0 (all tensors fall back to F32) | 10,941 | 12,448 | 0 |
| decoder, two layers | safetensors F16 | 85,548 | 46,544 | 4.883e-4 |
| decoder, two layers | safetensors BF16 | 85,548 | 46,568 | 3.906e-3 |
| decoder, two layers | GGUF F16 | 85,548 | 45,888 | 4.883e-4 |
| decoder, two layers | GGUF Q8_0 | 85,548 | 27,136 | 7.389e-3 |

## Opt-in M3 model evaluation

Run `make test-io5-live-quantized` with the pinned Eshkol toolchain.
The gate saves a finite, seed-1729 M3 model through the Eshkol C1 writer, then
uses the IO5 CLI to export and import mapped Hugging Face safetensors in F16
and BF16. It loads each restored C1 checkpoint into a fresh seed-0 M3 model
through the C1 reader and P1 module loader. The Eshkol M3 forward schedule
evaluates the same `[1,2]` input `(3, 197)` in each model and returns all 512
CPU F32 logits. The test measures finite, nonzero changes in weights and
logits from the original F32 evaluation. An independent standard-library
reference checks every mapped wire tensor's offsets, shape and narrowed
values against the source C1 tensors, then widens the wire payload and
compares every restored F32 weight bit. Both dtypes use 4,416 wire payload
bytes, half the F32 payload size. On Ubuntu 22.04 with LLVM and Clang 21.1.8,
the gate measured these fixture-specific maximum absolute deltas.

| Wire dtype | Weight delta | Logit delta | Test limits (weight, logit) |
|---|---:|---:|---:|
| F16 | 7.626e-6 | 2.932e-5 | 8.00e-6, 5.00e-5 |
| BF16 | 6.096e-5 | 2.450e-4 | 6.25e-5, 4.00e-4 |

The post-evaluation model snapshot must reserialize to the input C1 bytes,
including the tied head and token embedding. A flipped safetensors payload
byte fails import before creating or replacing a destination. Independently,
a stale C1 payload checksum, a validly checksummed C1 file without the tie,
and a validly checksummed C1 file with a wrong shape and unchanged element
count fail the live load before logits or a model-state readback are published.

## Limitations and scope

- Bounded non-streaming interchange (reuses IO5 64 MiB file / 32 MiB tensor /
  1 MiB metadata limits); peak memory and throughput are unmeasured.
- Only `F16` and `BF16` safetensors and `F32`/`F16`/`Q8_0` GGUF v3 are admitted;
  4-bit/sub-byte GGML quantizers (`Q4_0`, `Q4_K`, etc.), sharded quantized
  indexes, live runtime dequantized matmul kernels, and GPU execution are
  outside this slice.
- This quantized tool supports only M3 and decoder profiles. The separate
  lossless `eshkol-mod5-v1` mapping does not enable quantized MOD5 export,
  tagged import, or plain external import. The opt-in M3 evaluation covers
  safetensors converted to F32 C1 before load. Direct safetensors loading,
  live GGUF evaluation, CLI3 C2 ingress, and training resume are outside this
  gate.
