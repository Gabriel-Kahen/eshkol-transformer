# W4A32 INT4 Group-Wise Weight Quantization & Fused Dequant-Matmul Engine

## Overview

This standalone module implements an Eshkol-native **W4A32** (4-bit quantized weights, IEEE-754 binary32 activations and outputs, binary64 inner accumulation) quantization and fused dequantization-matmul engine:

- Public C11/C++17 ABI: `include/eshkol_transformer/w4a32_quant_abi.h`
- Native C11 implementation: `native/w4a32_quant.c` (`build/w4a32_quant/libeshkol_transformer_w4a32_quant.a`)
- Eshkol AOT bridge: `native/w4a32_quant_eshkol_bridge.{h,c}` (`build/w4a32_quant/libeshkol_transformer_w4a32_quant_aot.a`)
- Compiled Eshkol AOT driver: `tests/w4a32_quant/test_w4a32_aot.esk`
- Native C11 / C++17 / Python oracle tests: `tests/w4a32_quant/{header_w4a32_cpp.cpp,test_w4a32_native.c,test_w4a32_oracle.py}`
- Build & verification scripts: `scripts/build-w4a32-quant.sh`, `scripts/test-w4a32-quant.sh`

All hot kernels (`quantize`, `dequantize`, fused `gemm`, reference `f32_gemm`, and buffer-level container `serialize`/`parse`) perform **zero heap allocations** (enforced at link time via `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free`), use **byte-wise little-endian loads/stores** with zero misaligned-pointer undefined behavior, and execute **two-pass validate-before-write** so output buffers remain completely untouched on any error.

---

## Quantized Weight Formats & Exact Bit Layouts

All weight matrices `W` have logical shape `[N, K]` (row-major by output feature `n in [0, N-1]`). Quantization blocks are strictly **row-aligned** and **never cross rows**: `K` must be a positive multiple of `group_size`, and `1 <= M, N, K <= 4096`.

### 1. GGML `Q4_0` (`ET_W4A32_FORMAT_Q4_0 = 1`, `group_size = 32`, `18 bytes/block`)

Matches GGML `block_q4_0` byte-for-byte:

| Byte Offset | Size | Field | Encoding |
|---:|---:|---|---|
| `0..1` | 2 B | `d` | Little-endian IEEE-754 `binary16` scale (`d = signed_max / -8.0`) |
| `2..17` | 16 B | `qs[16]` | Split-half 4-bit unsigned nibbles (`q in [0, 15]`) |

- **Split-half nibble packing** (for `j = 0..15`):
  - Low nibble `qs[j] & 0x0F` encodes element `j` (`0..15`).
  - High nibble `(qs[j] >> 4) & 0x0F` encodes element `j + 16` (`16..31`).
- **Reconstruction**: `w_hat = d * ((int32_t)q - 8)` (`q - 8 in [-8, +7]`).
- **Scale quantization**: `signed_max` is the first element in the 32-element block achieving maximum `|w[j]|`, and `d = f16_rne((float)((double)signed_max / -8.0))`. All-zero blocks store `d = 0x0000` (`+0.0`) and `qs[0..15] = 0x88` (`q = 8`).
- **Rejections**: Non-finite weights (`ET_W4A32_ERR_NONFINITE`), `binary16` scale overflow (`|signed_max / -8.0| >= 65520.0`, `ET_W4A32_ERR_OVERFLOW`), non-finite `f16` scales (`Inf`/`NaN`) during dequantize/GEMM/container I/O (`ET_W4A32_ERR_NONFINITE`), and `K % 32 != 0` (`ET_W4A32_ERR_SHAPE`).

### 2. Asymmetric Group-Wise `Q4_ASYM` (`ET_W4A32_FORMAT_Q4_ASYM = 2`, `group_size in {32, 64, 128}`)

| Byte Offset | Size | Field | Encoding |
|---:|---:|---|---|
| `0..1` | 2 B | `d` | Little-endian IEEE-754 `binary16` scale (`d = (max - m_q) / 15.0 >= 0`) |
| `2..3` | 2 B | `m` | Little-endian IEEE-754 `binary16` minimum/offset (`m_q = f16_rne(min)`) |
| `4 .. 4 + half - 1` | `G / 2` B | `qs[half]` | Split-half 4-bit unsigned nibbles (`half = group_size / 2`, `q in [0, 15]`) |

- **Split-half nibble packing** (for `j = 0 .. half - 1`):
  - Low nibble `qs[j] & 0x0F` encodes element `j`.
  - High nibble `(qs[j] >> 4) & 0x0F` encodes element `j + half`.
- **Reconstruction**: `w_hat = d * (float)q + m`, `q in [0, 15]`.
- Eliminates wasted negative/positive half-range on skewed or shifted weight distributions.

### Compression Ratios vs `F32` (32 bits/weight)

| Format | Group Size `G` | Block Bytes | Effective Bits / Weight | Compression Ratio vs `F32` |
|---|---:|---:|---:|---:|
| `F32` (baseline) | — | — | `32.00` | `1.000x` |
| `Q4_0` | `32` | `18 B` | `4.50` | **`7.111x`** (`65,536 B -> 9,216 B`) |
| `Q4_ASYM` | `32` | `20 B` | `5.00` | `6.400x` |
| `Q4_ASYM` | `64` | `36 B` | `4.50` | **`7.111x`** (`65,536 B -> 9,216 B`) |
| `Q4_ASYM` | `128` | `68 B` | `4.25` | `7.529x` |

---

## Fused Dequant-Matmul Kernels

- `et_w4a32_q4_0_gemm_v1(m, n, k, x, w_q4_0, w_q4_0_bytes, bias, y)`
- `et_w4a32_asym_gemm_v1(m, n, k, group_size, x, w_asym, w_asym_bytes, bias, y)`

Computes `Y[M, N] = X[M, K] @ dequant(W_q[N, K])^T + (bias ? bias[N] : 0)` directly from packed 4-bit blocks without materializing an `[N, K]` `f32` weight matrix:

- **Q4_0 inner accumulation** (binary64): `acc += d * sum_{j=0..15} (x[j]*(q0 - 8) + x[j+16]*(q1 - 8))`
- **Q4_ASYM inner accumulation** (binary64): `acc += d * sum(x_i * q_i) + m * sum(x_i)`
- Supports decode (`M = 1` GEMV) and prefill (`M > 1` GEMM), optional `bias` (`NULL` allowed).
- Validates FP environment (`FE_TONEAREST`, default x87/MXCSR with FTZ/DAZ disabled), shapes, 4-byte float alignment, pairwise buffer disjointness, finite `X`/`bias`/`f16` block scales, and dry-runs overflow detection whenever `K * max|X| * max|W_hat| + max|bias| > FLT_MAX` so `Y` is never modified on error.

---

## Checksummed Quantized Weight Container (`ETQ41000` v1.0)

Deterministic, non-executable (`0600`), versioned binary container:

| Byte Offset | Size | Field | Encoding |
|---:|---:|---|---|
| `0..7` | 8 B | Magic | ASCII `"ETQ41000"` |
| `8..11` | 4 B | `version_major` | `u32 LE` (`1`) |
| `12..15` | 4 B | `version_minor` | `u32 LE` (`0`) |
| `16..19` | 4 B | `format` | `u32 LE` (`1 = Q4_0`, `2 = Q4_ASYM`) |
| `20..23` | 4 B | `group_size` | `u32 LE` (`32`, `64`, or `128`) |
| `24..27` | 4 B | `rows_n` | `u32 LE` (`N`) |
| `28..31` | 4 B | `cols_k` | `u32 LE` (`K`) |
| `32..39` | 8 B | `payload_bytes` | `u64 LE` (exact packed weight byte count) |
| `40..47` | 8 B | `reserved_flags` | `u64 LE` (`0`; non-zero rejected) |
| `48 .. 48 + payload_bytes - 1` | `payload_bytes` | Packed blocks | Row-major packed `Q4_0` or `Q4_ASYM` blocks |
| `48 + payload_bytes .. +31` | 32 B | `sha256` | SHA-256 digest of bytes `[0 .. 48 + payload_bytes - 1]` |

- `et_w4a32_container_save` writes via `mkstemp` (mode `0600`) + `write` + `fsync` + `close` + atomic `rename`.
- `et_w4a32_container_load` opens with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC`, verifies `S_ISREG`, validates SHA-256 and all embedded `f16` scales/mins, checks caller-expected format/shape, and commits `out_packed` atomically.

---

## Verification & Measured Results

Verified inside container `eshkol-r0-debug:20260928` (`clang-21` / `clang++-21` 21.1.8, pinned `/tc/build/eshkol-run`):

| Check | Result |
|---|---|
| Deterministic static archives (`ar rcsD`, dual fresh builds) | **PASS** — core `2fafd2281e0f48648ec8b8d27446495760a5c9bdc149d60c6b3f1cff34c58003`, bridge `d905e07af026f78954d0db2be5025f34370b1fa91f40a7cecdabb650e8ad4f2a` |
| C++17 header contract & linkage (`header_w4a32_cpp.cpp`) | **PASS** (normal `-O2` and ASan/UBSan/LSan) |
| Native C11 suite (`test_w4a32_native.c`, `--wrap=malloc/calloc/realloc/free`) | **PASS** — zero heap allocations on all hot kernels, fused GEMV/GEMM parity `max_abs_err = 0.000e+00`, odd-offset packed byte buffers clean under UBSan |
| Independent Python oracle (`test_w4a32_oracle.py`) | **PASS** — 8 cases, `4,168` packed bytes bit-exact, `270` GEMV/GEMM outputs `max_deq_err = 0.000e+00`, `max_gemm_err = 0.000e+00`, `max_q4_recon = 0.2177`, `max_asym_recon = 0.0993`, `ETQ41000` SHA-256 + 7 negative status codes verified |
| ASan / UBSan / LSan (`detect_leaks=1:halt_on_error=1`) | **PASS** (C11 wrapped + C++17) |
| Compiled Eshkol AOT 2-layer MLP (`64 -> 128 -> 64`, `--strict-types --optimize 0 --no-stdlib`) | **PASS** — dual fresh ELFs byte-identical, SHA-256 `322216beea656f828d24ac0c9c95e028475b3bb3ad28b7b395f0ee2bb23773be` |
| AOT compression & skewed reconstruction (`W2 in [0.80, 1.55]`) | `F32 = 65,536 B`, `Q4_0 (G=32) = 9,216 B` (`7.111x`), `Q4_ASYM (G=64) = 9,216 B` (`7.111x`); skewed max error `Q4_0 = 0.094043` vs `Q4_ASYM = 0.025006` (**3.76x reduction**) |
| AOT fused GEMV (`M=1`) & GEMM (`M=8`) parity & 2-process `ETQ41000` reload | `q4_gemv = 0`, `q4_gemm = 0`, `asym_gemv = 0`, `asym_gemm = 0`; bitwise-identical packed & output SHA-256 across fresh OS processes + 12/12 negative rejections |
| Python isolation & CI topology gates | **PASS** (`tests/q0/test_python_isolation.py` 5/5 OK; `scripts/check-ci-topology.sh` 23 suites PASS) |

---

## Measured Limitations & Unsupported Cases

- **CPU-only serial reference execution**: Executed on x86-64 CPU in IEEE-754 binary32/binary64 (`-ffp-contract=off -fno-fast-math`). No GPU, SIMD intrinsics, or multi-threaded dispatch is claimed.
- **Weight-only quantization (`W4A32`)**: Activations `X`, biases, and outputs `Y` are `binary32`; 4-bit activation quantization (`W4A4` / `W4A8`) and backward training through quantized weights (QAT / STE) are outside this v1.0 contract.
- **Dimension bounds**: `1 <= M, N, K <= 4096`, `K % group_size == 0` with `group_size == 32` (`Q4_0`) or `group_size in {32, 64, 128}` (`Q4_ASYM`).
- **Floating-point environment**: Requires round-to-nearest-even (`FE_TONEAREST`) and default masked exceptions with FTZ/DAZ disabled; non-default rounding modes are explicitly rejected (`ET_W4A32_ERR_FP_ENV`).
- **Cold file I/O paths**: `et_w4a32_container_save` and `et_w4a32_container_load` use bounded heap allocations for the temporary path / file read buffer; all numerical kernels and buffer-level container serialize/parse perform zero heap allocations.
