# Quantized KV-Cache Engine (`FP32` / `INT8_SYM` / `FP8_E4M3FN`) & Fused GQA Decode

Standalone, Eshkol-native implementation of a unified quantized KV-cache engine
supporting baseline `FP32`, per-token/per-head symmetric `INT8`, and scaled OCP
FP8 `E4M3FN` storage, fused zero-allocation online-softmax GQA causal and
sliding-window decode, versioned SHA-256 checksummed snapshots (`ETKQ1000`
v1.0), and a compiled Eshkol AOT driver.

## Artifacts and Public Contracts

- Public C11/C++17 ABI header: `include/eshkol_transformer/kv_cache_quant_abi.h`
- Core implementation: `native/kv_cache_quant.c` -> deterministic archive
  `build/kv_cache_quant/libeshkol_transformer_kv_cache_quant.a`
  - SHA-256 (Clang 21.1.8, Ubuntu 22.04, `-O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math`):
    `3cc781ac549eed13d70d4a2d4085a9eb73e381d2a34d5aa4ab60309b9d9213d6`
- Eshkol AOT bridge: `native/kv_cache_quant_eshkol_bridge.h`,
  `native/kv_cache_quant_eshkol_bridge.c` -> deterministic archive
  `build/kv_cache_quant/libeshkol_transformer_kv_cache_quant_aot.a`
  - SHA-256:
    `2242f03e0f0ccca218fd953a49f77611edb786024bbc63ae20e832b7551f9137`
- Compiled Eshkol AOT driver: `tests/kv_cache_quant/test_kv_cache_quant_aot.esk`
  - Compiled with `--strict-types --optimize 0 --no-stdlib` from two fresh
    directories; byte-identical ELF SHA-256:
    `155debe1e37db2681b018ca72f1589ef9ea1d38bd28955818c7620333757af02`
- Native C11 / C++17 / Python oracle test suite:
  - `tests/kv_cache_quant/header_kvq_cpp.cpp`
  - `tests/kv_cache_quant/test_kv_cache_quant_native.c`
  - `tests/kv_cache_quant/test_kv_cache_quant_oracle.py`
- Build & verification scripts:
  - `scripts/build-kv-cache-quant.sh`
  - `scripts/test-kv-cache-quant.sh`

## Quantization Formulas

Every stored vector corresponds to one `(layer, kv_side in {K=0, V=1}, pos, kv_head)`
slice of length `D_h = head_dim`.

### 1. Baseline `ET_KVQ_FMT_FP32` (`format = 1`)
- Stores raw IEEE-754 `binary32` values (`4 * D_h` bytes per vector).
- Dequantization is the identity map.

### 2. Symmetric INT8 `ET_KVQ_FMT_INT8_SYM` (`format = 2`)
- `amax = max_{0 <= d < D_h} |x[d]|`
- Zero-vector rule: if `amax == 0.0f`, then `scale = 0.0f` and `q[d] = 0` for all `d`.
- Non-zero rule:
  - `scale = (float)((double)amax / 127.0)` (if subnormal `amax` underflows
    `scale` to `0.0f`, `scale` clamps to `FLT_TRUE_MIN = 2^-149` so
    `x[d] / scale` remains an exact integer in `[-63, 63]`; if `amax` near
    `FLT_MAX` causes `scale` to round up such that `(float)(127.0 * (double)scale)`
    overflows `binary32`, `scale` steps down by 1 ULP to `0x7c010203` so all
    dequantized values in `[-127, 127]` stay strictly finite).
  - `q[d] = rne_clamp_i8((double)x[d] / (double)scale) in [-127, 127]` using
    exact IEEE-754 round-to-nearest-even (`-128` is forbidden and rejected as
    `ET_KVQ_ERR_CORRUPT` on dequantize/decode/load; negative or `-0.0f` scales
    are rejected as `ET_KVQ_ERR_NONFINITE`, and `scale == 0.0f` with non-zero
    quantized payloads is rejected as `ET_KVQ_ERR_CORRUPT` in both `INT8_SYM`
    and `FP8_E4M3FN`).
- Dequantization: `x_deq[d] = (float)((double)q[d] * (double)scale)`.
- Storage per vector: `D_h` int8 bytes + `4` bytes scale (`D_h + 4` bytes).

### 3. Scaled OCP FP8 E4M3FN `ET_KVQ_FMT_FP8_E4M3FN` (`format = 3`)
- OCP FP8 `E4M3FN` (1 sign, 4 exponent bias 7, 3 mantissa, no Inf,
  `max_finite = 448.0`, `min_subnormal = 2^-9`, NaN `0x7F`/`0xFF` rejected with
  `ET_KVQ_ERR_NONFINITE`).
- Zero-vector rule: if `amax == 0.0f`, then `scale = 0.0f` and `q[d] = 0x00`.
- Non-zero rule:
  - `scale = (float)((double)amax / 448.0)` (subnormal fallback to `2^-149`).
  - `scaled = (float)((double)x[d] / (double)scale)`.
  - `q[d] = encode_e4m3fn_rne(scaled)` (pure integer IEEE-754 RNE encoder,
    saturating `|scaled| > 464.0` to `0x7E`/`0xFE`, never emitting NaN).
- Dequantization:
  `x_deq[d] = (float)((double)decode_e4m3fn(q[d]) * (double)scale)`.
- Storage per vector: `D_h` uint8 bytes + `4` bytes scale (`D_h + 4` bytes).

## Memory Footprint and Compression Table

Backing storage packs all `float` scales first (`total_vectors * 4` bytes,
naturally 4-byte aligned) followed by quantized bytes (`total_vectors * D_h`),
achieving exact `D_h + 4` bytes per vector with zero padding waste.

| Head Dim `D_h` | FP32 Bytes / Vec | Quantized Bytes / Vec (`INT8`/`FP8`) | Compression Ratio | `L=2, H_kv=2, T=32` Active FP32 | `L=2, H_kv=2, T=32` Active Quant |
|---:|---:|---:|---:|---:|---:|
| `16` | `64 B` | `20 B` | `3.2000x` | `16,384 B` | `5,120 B` |
| `32` | `128 B` | `36 B` | `3.5556x` | `32,768 B` | `9,216 B` |
| `64` | `256 B` | `68 B` | `3.7647x` (`64/17`) | `65,536 B` | `17,408 B` |
| `128` | `512 B` | `132 B` | `3.8788x` | `131,072 B` | `33,792 B` |
| `256` | `1024 B` | `260 B` | `3.9385x` | `262,144 B` | `66,560 B` |

## Fused Quantized Online-Softmax GQA Decode

`et_kvq_decode_gqa_v1` performs single-pass online-softmax GQA attention
(`num_q_heads % num_kv_heads == 0`) over `t in [t_start, seq_len - 1]`, where
`t_start = 0` for `window_size == 0` (full causal) or
`max(0, seq_len - window_size)` for sliding-window attention:

- Dequantizes `K` and `V` on the fly in registers without materializing an FP32
  KV cache.
- Accumulates dot products, running max, running denominator, and weighted `V`
  vectors in IEEE-754 `binary64` under verified `FE_TONEAREST` +
  `#pragma STDC FP_CONTRACT OFF` and x86_64 x87/MXCSR FTZ/DAZ guards.
- Stages outputs in a bounded stack buffer (`ET_KVQ_MAX_Q_HEADS * ET_KVQ_MAX_HEAD_DIM`)
  for strict failure atomicity and **zero heap allocations** on the hot path
  (enforced via `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free`).

## Snapshot Wire Format (`ETKQ1000` v1.0)

Little-endian, non-executable, SHA-256 checksummed binary format:
1. **64-byte header**: magic `"ETKQ1000"` (`8 B`), version `major=1, minor=0`
   (`2+2 B`), `format` (`4 B`), `num_layers` (`4 B`), `num_kv_heads` (`4 B`),
   `head_dim` (`4 B`), `max_seq_len` (`4 B`), `min_seq_len` (`4 B`),
   `reserved=0` (`4 B`), `payload_bytes` (`8 B`), `total_active_slices` (`8 B`),
   `reserved2=0` (`8 B`).
2. **Payload (`payload_bytes`)**: `layer_seq_len[num_layers]` (`4 * L` bytes),
   followed by active `[layer][side][pos][kv_head]` entries (f32 words for
   `FP32`; f32 scale + `head_dim` raw bytes for `INT8_SYM` / `FP8_E4M3FN`).
3. **32-byte trailer**: SHA-256 over bytes `[0 .. 64 + payload_bytes - 1]`.

Save writes atomically via a `0600` random `.etkq-*.tmp` file in the parent
directory (`O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC`), `fchmod(0600)`,
`fsync`, `renameat`, and parent `fsync`, rejecting existing symlink/non-regular
targets. Load opens with `O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK`,
verifies `S_ISREG`, exact length, header fields, constant-time SHA-256, format
and geometry compatibility, and all payload values before mutating the target
cache.

## Measured Verification Evidence (`eshkol-r0-debug:20260928`)

Executed with `CC=clang-21`, `CXX=clang++-21`, `--network none`:

- **Deterministic Archives & Symbol Audit**: PASS (`cmp` identical; exact
  global exports and restricted libc/POSIX imports verified).
- **C++17 Header & Linkage**: PASS (`standard_layout` + `trivially_copyable`).
- **Native C11 (`-O2` + `--wrap` hot-path zero-alloc trap) & ASan/UBSan/LSan
  (`detect_leaks=1:halt_on_error=1`)**: PASS.
- **Python Exact-Fraction & binary64 Oracle**:
  `256 FP8 codes + 21 ties + 5 vec pairs 0.0 ULP + 6 GQA cases / 768 vals max_abs_err=0.000e+00`.
- **Compiled Eshkol AOT Driver (`L=2, H_q=8, H_kv=2, D_h=64, T=32`)**:
  - Compression: `65536 B -> 17408 B` (`ratio = 3.764705882352941`, exact `64/17`)
  - Fused vs explicit dequantized decode: `0` mismatched bits (`0.0` error) for
    both `INT8_SYM` and `FP8_E4M3FN` (full causal and `window_size=8`)
  - Accuracy vs FP32 baseline:
    - Normal heads (`qh 0..3`): `INT8 RMSE = 1.556e-4`, `FP8 RMSE = 2.714e-3`
    - Outlier heads (`qh 4..7`, `d=0` outlier `±12.0`, `d=1..63` small
      activations): `INT8 RMSE = 4.924e-3` vs `FP8 RMSE = 3.716e-4`
      (`13.25x` lower RMSE for `FP8_E4M3FN`)
  - Multi-step top-1 logit argmax: `4/4` checkpoints (`t = 8, 16, 24, 32`)
    match FP32 exactly for both `INT8_SYM` and `FP8_E4M3FN`.
  - Truncate (`32 -> 20`) + re-append (`20..31`): `0` mismatched bits.
  - `ETKQ1000` snapshot round-trip: `0` mismatched bits; format mismatch and
    payload tamper rejected (`ET_KVQ_ERR_FORMAT_MISMATCH`, `ET_KVQ_ERR_CHECKSUM`).
  - Dual fresh AOT ELFs & stdout: byte-identical (`cmp` PASS).
- **Repository Gates**: `tests/q0/test_python_isolation.py` (5/5 OK) and
  `scripts/check-ci-topology.sh` (PASS).

## Explicit Limitations and Unsupported Cases

- CPU-only IEEE-754 `binary32`/`binary64` execution; no GPU kernel or silent
  fallback.
- Bounded compile-time limits: `num_layers <= 64`, `num_kv_heads <= 64`,
  `num_q_heads <= 64`, `head_dim <= 256`, `max_seq_len <= 65536`, backing
  storage `<= 256 MiB`.
- Non-finite inputs (`NaN`, `±Inf`), non-`FE_TONEAREST` rounding modes, x86_64
  FTZ/DAZ MXCSR modes, aliased buffers, and `INT8 = -128` or `FP8 = 0x7F/0xFF`
  bit patterns are explicitly rejected without mutating state.
- Snapshot persistence (`et_kvq_cache_save_snapshot` / `load_snapshot`) is a
  cold I/O path requiring Linux POSIX `openat`/`fstatat`/`renameat`/`getrandom`
  and allocates a temporary serialization buffer; hot append/dequant/decode
  kernels perform zero heap allocations and zero syscalls.
