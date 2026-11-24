# OCP FP8 (`E4M3FN` & `E5M2`) Codec, Delayed-Scaling Manager & Block-Scaled `W8A8` GEMM

Standalone, Eshkol-native implementation of bit-exact OCP FP8 (`E4M3FN` and
`E5M2`) IEEE-754 round-to-nearest-even (RNE) codecs, a delayed-scaling `amax`
history ring-buffer manager, DeepSeek-V3 style fine-grained block-scaled
`W8A8` `E4M3FN` GEMM (forward and Straight-Through Estimator backward), and a
zero-allocation Eshkol AOT bridge.

## Artifacts and Public Contracts

- Public C11/C++17 ABI header: `include/eshkol_transformer/fp8_gemm_abi.h`
- Core implementation: `native/fp8_gemm.c` -> deterministic archive
  `build/fp8_gemm/libeshkol_transformer_fp8_gemm.a`
  - SHA-256 (Clang 21.1.8, Ubuntu 22.04, `-O2 -ffp-contract=off -fno-fast-math`):
    `f1bae814df8dddd657b6a4d483f360a12f347fdb7b976cd9b06101abffcfd0e6`
  - Zero heap allocations: `malloc`, `calloc`, `realloc`, and `free` are absent
    from undefined symbols (audited via `nm -u`) and trapped at link time via
    `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free`.
  - Byte-wise alignment-safe `memcpy` loads/stores for all `float` buffers.
  - Two-pass validate-before-write semantics guaranteeing caller output buffers
    and state structs remain byte-for-byte untouched on any error.
- Eshkol AOT bridge: `native/fp8_gemm_eshkol_bridge.h`,
  `native/fp8_gemm_eshkol_bridge.c` -> deterministic archive
  `build/fp8_gemm/libeshkol_transformer_fp8_gemm_aot.a`
  - SHA-256:
    `3eb25dc9ae858a1c655f78203dc686b5f08437de7b434da5a6dcbfb1ed528b9f`
  - Zero heap allocations; accepts untyped `void *` pointers for safe FFI
    marshaling from compiled Eshkol AOT binaries.
- Compiled Eshkol AOT driver: `tests/fp8_gemm/test_fp8_gemm_aot.esk`
- Native C11 / C++17 / Python oracle tests:
  - `tests/fp8_gemm/header_fp8_cpp.cpp`
  - `tests/fp8_gemm/test_fp8_native.c`
  - `tests/fp8_gemm/test_fp8_oracle.py`
- Build and verification scripts:
  - `scripts/build-fp8-gemm.sh`
  - `scripts/test-fp8-gemm.sh`

## OCP FP8 Bit Layout Tables (v1.0)

### `E4M3FN` (`S.EEEE.MMM`, exponent bias = 7, NO Infinity)

| Category | Sign (`b7`) | Exponent (`b6..b3`) | Mantissa (`b2..b0`) | Codes (hex) | Exact Value / Semantics |
|---|---:|---:|---:|---|---|
| Signed zero | `0` / `1` | `0000` (`0`) | `000` (`0`) | `0x00` / `0x80` | `+0.0` / `-0.0` |
| Subnormal | `s` | `0000` (`0`) | `001..111` (`1..7`) | `0x01..0x07`, `0x81..0x87` | `(-1)^s * mant * 2^-9` (`min = 2^-9 = 0.001953125`) |
| Normal (`exp 1..14`) | `s` | `0001..1110` (`1..14`) | `000..111` (`0..7`) | `0x08..0x77`, `0x88..0xF7` | `(-1)^s * 2^(exp - 7) * (1 + mant / 8)` |
| Max-exponent finite | `s` | `1111` (`15`) | `000..110` (`0..6`) | `0x78..0x7E`, `0xF8..0xFE` | `(-1)^s * 2^8 * (1 + mant / 8)` (`max = 448.0` at `0x7E`) |
| NaN (rejected on decode) | `0` / `1` | `1111` (`15`) | `111` (`7`) | `0x7F` / `0xFF` | Non-finite (`ET_FP8_ERR_NONFINITE = -2`) |

- Total codepoints: 256 (`254` finite, `2` NaN).
- IEEE-754 RNE overflow boundary: `|x| == 464.0` (`0x43E80000`) is the exact
  midpoint between `448.0` (`0x7E`, even LSB) and the unbounded next grid point
  `480.0` (`0x7F`, odd LSB), so `464.0` rounds to `0x7E` (`448.0`); any finite
  `|x| > 464.0` (`>= 0x43E80001`) saturates to `0x7E`/`0xFE` in
  `ET_FP8_OVERFLOW_SATURATE_TO_MAX` mode or fails with `ET_FP8_ERR_OVERFLOW`
  (`-3`) in `ET_FP8_OVERFLOW_REJECT` mode.
- Non-finite `f32` inputs (`+Inf`, `-Inf`, `qNaN`, `sNaN`) are **always**
  rejected with `ET_FP8_ERR_NONFINITE` (`-2`) in both overflow modes.

### `E5M2` (`S.EEEEE.MM`, exponent bias = 15, IEEE-754 Inf/NaN)

| Category | Sign (`b7`) | Exponent (`b6..b2`) | Mantissa (`b1..b0`) | Codes (hex) | Exact Value / Semantics |
|---|---:|---:|---:|---|---|
| Signed zero | `0` / `1` | `00000` (`0`) | `00` (`0`) | `0x00` / `0x80` | `+0.0` / `-0.0` |
| Subnormal | `s` | `00000` (`0`) | `01..11` (`1..3`) | `0x01..0x03`, `0x81..0x83` | `(-1)^s * mant * 2^-16` (`min = 2^-16`) |
| Normal (`exp 1..30`) | `s` | `00001..11110` (`1..30`) | `00..11` (`0..3`) | `0x04..0x7B`, `0x84..0xFB` | `(-1)^s * 2^(exp - 15) * (1 + mant / 4)` (`max = 57344.0` at `0x7B`) |
| Infinity (rejected on decode) | `0` / `1` | `11111` (`31`) | `00` (`0`) | `0x7C` / `0xFC` | Non-finite (`ET_FP8_ERR_NONFINITE = -2`) |
| NaN (rejected on decode) | `0` / `1` | `11111` (`31`) | `01..11` (`1..3`) | `0x7D..0x7F`, `0xFD..0xFF` | Non-finite (`ET_FP8_ERR_NONFINITE = -2`) |

- Total codepoints: 256 (`248` finite, `8` Inf/NaN).
- IEEE-754 RNE overflow boundary: `|x| == 61440.0` (`0x47700000`) is the exact
  midpoint between `57344.0` (`0x7B`, odd LSB) and `65536.0` (`0x7C`, even LSB,
  Inf slot), so `|x| >= 61440.0` overflows (`0x7B`/`0xFB` in saturate mode,
  `ET_FP8_ERR_OVERFLOW` in reject mode), while `nextafterf(61440.0f, 0.0f)`
  (`0x476FFFFF`) rounds to `0x7B`.

## Delayed-Scaling Amax History Manager

`et_fp8_delayed_scale_state_v1` maintains a deterministic ring buffer of length
`history_len in [1, 64]` recording per-tensor `amax` values. At each step:

```text
history_amax = max_{0 <= i < count} history[i]   (0.0 when count == 0)
scale_raw    = (fp8_max / max(history_amax, eps)) * 2^(-margin)    (binary64)
scale        = clamp((float)scale_raw, min_scale, max_scale)
```

- `fp8_max` is `448.0` for `ET_FP8_FORMAT_E4M3FN` and `57344.0` for
  `ET_FP8_FORMAT_E5M2`.
- `margin in [0, 16]` uses exact binary64 scaling `1.0 / (double)(1u << margin)`.
- Zero-`amax` safety is guaranteed by `max(history_amax, eps)` (`eps > 0`) and
  `[min_scale, max_scale]` clamping.
- Negative or non-finite `observed_amax` inputs are rejected without mutating
  ring-buffer state.

## Block-Scaled `W8A8` `E4M3FN` GEMM & STE Backward

DeepSeek-V3 style fine-grained block scaling:
- Activations `X[M, K]` use `1 x block_k` dequantization scales `S_x[M, K / block_k]`.
- Weights `W[N, K]` use `block_n x block_k` dequantization scales
  `S_w[N / block_n, K / block_k]` with `block_k in {16, 32}`,
  `block_n in {1, 16, 32}`, `K % block_k == 0`, and `N % block_n == 0`.
- Block quantization computes:

```text
amax_b  = max_{(r, c) in block b} |src[r, c]|
scale_b = (amax_b == 0) ? 1.0f : max(FLT_MIN, (float)((double)amax_b / 448.0))
q[r, c] = encode_e4m3_rne_saturate((float)((double)src[r, c] / (double)scale_b))
```

- Block-scaled `W8A8` GEMM forward (`et_fp8_block_gemm_e4m3_v1`):

```text
Y[m, n] = (float)(
    sum_{kb = 0}^{K/block_k - 1}
      ( sum_{kk = 0}^{block_k - 1}
          (double)decode_e4m3(X_q[m, kb*block_k + kk])
        * (double)decode_e4m3(W_q[n, kb*block_k + kk])
      ) * ((double)S_x[m, kb] * (double)S_w[n / block_n, kb])
    + (bias ? (double)bias[n] : 0.0)
)
```

- Straight-Through Estimator backward (`et_fp8_block_gemm_e4m3_backward_v1`)
  computes exact gradients of `Y = dequant(X_q, S_x) @ dequant(W_q, S_w)^T + bias`
  with respect to `X_deq`, `W_deq`, and `bias` in `binary64` before casting to
  `binary32`.

## Verification Results

Verified inside the pinned `eshkol-r0-debug:20260928` container
(`CC=clang-21 CXX=clang++-21`, Ubuntu 22.04, LLVM/Clang 21.1.8, pinned Eshkol
toolchain, `--network none`):

- Deterministic archive & symbol audit (`scripts/build-fp8-gemm.sh`): PASS
  - Exact defined export sets verified; zero allocator/IO/threading imports.
- C++17 header and linkage check (`tests/fp8_gemm/header_fp8_cpp.cpp`): PASS
- Native C11 suite (normal `-O2` and ASan/UBSan/LSan with
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`
  and `--wrap=malloc/calloc/realloc/free`): PASS
- Independent exact-rational (`fractions.Fraction`) + `binary64` Python oracle
  (`tests/fp8_gemm/test_fp8_oracle.py`):
  - `512` FP8 codepoints (`256 E4M3FN + 256 E5M2`), `57` RNE tie-to-even &
    overflow boundaries, `20` delayed-scaling steps, `4` block-GEMM cases
    (`4,904` tensor values):
  - `gemm_max_err = 0.000e+00`, `ste_bwd_max_err = 0.000e+00` (bit-exact `f32`
    parity with independent `binary64` dequantized reference).

### Measured Outlier-Heavy Accuracy: Block-Scaled FP8 vs Per-Tensor FP8

Test tensor: `X[8, 64]` with a single channel outlier `X[0, 0] = 224.0` and
small normal activations `~0.0003..0.0048`, `W[16, 64]` with normal weights
`~0.125..2.0` (`W[n, 0] = +/-0.03125`), `bias[16]`, `block_k = 16`:

| Quantization Mode | Dequant GEMM Parity `max_abs_err` | `max_abs_err` vs `f32` `X @ W^T + b` | RMSE vs `f32` `X @ W^T + b` | STE Central-FD `max_abs_err` |
|---|---:|---:|---:|---:|
| Per-tensor `E4M3FN` baseline | — | `0.00638` | `0.00284` | — |
| Block-scaled `1x16` / `1x16` (`bn=1`) | `4.768e-07` (`<= 1e-6`) | `0.00310` (**2.06x** lower) | `0.00092` (**3.09x** lower) | `1.218e-05` |
| Block-scaled `1x16` / `16x16` (`bn=16`) | `4.768e-07` (`<= 1e-6`) | `0.00310` (**2.06x** lower) | `0.00098` (**2.90x** lower) | `9.849e-06` |

### Compiled Eshkol AOT Verification (`tests/fp8_gemm/test_fp8_gemm_aot.esk`)

Compiled twice from clean temporary directories via
`eshkol-run --strict-types --optimize 0 --no-stdlib`:
- `FP8 AOT CODEC e4m3_finite=254 e4m3_reject=2 e5m2_finite=248 e5m2_reject=8 PASS`
- `FP8 AOT DELAYED_SCALE steps=10 history=4 margin=1 final_amax=56 final_scale=4 PASS`
- `FP8 AOT GEMM parity_max_err=4.842342766764318e-07 block_vs_f32=0.003104269504547119 pertensor_vs_f32=0.006375565079451917 ste_fd_checked=48 ste_fd_max_err=4.5634806045136145e-08 PASS`
- `FP8 AOT NEGATIVE PASS`
- Byte-identical dual ELF SHA-256:
  `1b590cbc9754b57311905db07668150fe85b6f9750b876ced7dab216d313c299`

### Repository Gates
- `python3 -m unittest -v tests/q0/test_python_isolation.py`: 5/5 OK
- `bash scripts/check-ci-topology.sh`: PASS (23 shared suites, 38 commands)

## Measured Limitations and Unsupported Cases

- CPU reference execution only (deterministic serial C11); no GPU/Tensor-Core
  FP8 hardware kernel is claimed.
- Block GEMM currently supports `E4M3FN` weights and activations (`W8A8`) with
  `block_k in {16, 32}` and `block_n in {1, 16, 32}`, requiring `K % block_k == 0`
  and `N % block_n == 0`. Other block sizes or non-multiple dimensions fail
  explicitly with `ET_FP8_ERR_UNSUPPORTED_BLOCK` (`-5`) or `ET_FP8_ERR_SHAPE` (`-6`).
- Non-standard IEEE-754 floating-point environments (non-`FE_TONEAREST` rounding
  mode, or x86-64 MXCSR FTZ/DAZ enabled) are detected and explicitly rejected
  with `ET_FP8_ERR_ENVIRONMENT` (`-8`).
- Zero-length arrays (`count == 0`, `M == 0`, `N == 0`, `K == 0`), overlapping
  output/input buffers, non-positive scales, and non-finite FP8/f32 values are
  explicitly rejected with untouched output buffers.
