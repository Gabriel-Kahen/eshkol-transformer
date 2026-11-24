# MOD5 RoPE Provider and 2-Layer Llama-Style AOT Decoder

MOD5 RoPE provides deterministic Rotary Position Embeddings (`kernel.rope`,
operations `rope.forward` and `rope.backward`) through K1 ABI 1.0 and direct
C11 helpers, plus an Eshkol AOT bridge that composes with MOD5 RMSNorm, SwiGLU,
and GQA attention to train a 2-layer Llama-style decoder.

## Artifacts and public contracts

- Header: `include/eshkol_transformer/mod5_rope_abi.h` (C11 and C++17 compatible)
- Provider archive: `build/mod5_rope/libeshkol_transformer_mod5_rope.a`
  - Exports only `et_mod5_rope_kernel_provider_v1`, `et_mod5_rope_forward_f32`,
    and `et_mod5_rope_backward_f32`.
  - Zero allocator imports (`malloc`, `calloc`, `realloc`, `free` absent from
    undefined symbols and trapped during dispatch tests).
- Eshkol AOT bridge: `native/mod5_rope_eshkol_bridge.h`,
  `native/mod5_rope_eshkol_bridge.c`,
  `build/mod5_rope/libeshkol_transformer_mod5_rope_bridge.a`
  - Exports only `et_mod5_rope_aot_{open,close,forward,backward,store_f32,load_f32,store_i64,load_i64}`.
  - Accepts untyped `void *` tensor pointers so raw Eshkol FFI pointers
    (including intentionally misaligned `ptr-add` addresses) are validated
    without misaligned-pointer UB.

## Mathematical specification

For input tensor `x` of dense row-major shape `[N, H, T, Dh]` (`f32`), integer
positions `positions` of shape `[N, T]` (`i64`, `0 <= pos <= 16777215`), base
frequency `theta > 1.0` (default `10000.0`), and position scale `scale > 0.0`
(default `1.0`, enabling linear/NTK position scaling):

Adjacent channel pairs `(x[2k], x[2k + 1])` for `k = 0 .. Dh/2 - 1` rotate by:

```text
inv_freq[k] = pow(theta, (-2.0 * k) / Dh)       (binary64)
phi         = ((double)pos / scale) * inv_freq[k] (binary64)
c           = (float)cos(phi)                   (binary64 -> binary32)
s           = (float)sin(phi)                   (binary64 -> binary32)
```

Forward (`rope.forward`):

```text
y0 = x0 * c - x1 * s
y1 = x0 * s + x1 * c
```

Exact orthogonal transpose backward (`rope.backward`):

```text
dx0 =  dy0 * c + dy1 * s
dx1 = -dy0 * s + dy1 * c
```

The rotation matrix is orthogonal (`y0^2 + y1^2 == x0^2 + x1^2`,
`backward(forward(x)) == x`), and the Jacobian is independent of `x`.

## K1 ABI 1.0 contract and validation

- Provider / implementation: `mod5.rope.cpu-f32.serial`, version `1.0`,
  evidence `MOD5:cpu-f32-rope-v1`, `VERIFIED`, `deterministic = 1`, compute
  dtype `f32`, device `cpu`.
- Admitted request shapes: `[N, H, T, Dh]` with `N in 1..2`, `H in 1..4`,
  `T in 1..64`, and even `Dh in {2, 4, 6, 8, 10, 12, 14, 16}` (8 verified K1
  shape range rows).

| Operation | Inputs in order | Output |
|---|---|---|
| `rope.forward` | `x f32[N,H,T,Dh]`, `positions i64[N,T]`, `theta f32[]`, `scale f32[]` | `y f32[N,H,T,Dh]` |
| `rope.backward` | `dy f32[N,H,T,Dh]`, `positions i64[N,T]`, `theta f32[]`, `scale f32[]` | `dx f32[N,H,T,Dh]` |

Validation enforces:
- Odd `Dh` and out-of-range `N, H, T, Dh` reject with `UNSUPPORTED`.
- NULL or misaligned pointers (`_Alignof(float)` / `_Alignof(int64_t)`),
  input-input overlap, output aliasing inputs/K1 metadata/`error`, negative or
  `> 16777215` positions, `theta <= 1.0`, `scale <= 0.0`, non-finite inputs,
  and non-finite binary64/binary32 intermediates reject with
  `INVALID_ARGUMENT`.
- Non-standard x87/MXCSR FP environments (non-RN rounding, unmasked exceptions,
  FTZ, DAZ) reject with `UNSUPPORTED`.
- Dry-run preflight restores `fenv_t` and `errno` and guarantees caller output
  buffers remain byte-for-byte untouched on any error.

## Validation results

Run `bash scripts/test-mod5-rope.sh` in the pinned Ubuntu 22.04 / Clang 21.1.8
/ Eshkol toolchain environment:

- Deterministic archive & symbol allowlist audit (`mod5_rope_provider.o`,
  `mod5_rope_eshkol_bridge.o`): PASS.
- C++17 header and linkage check (`clang++-21 -std=c++17 -Wall -Wextra -Werror -Wpedantic`): PASS.
- Native C11 suite (normal `-O2` and ASan/UBSan/LSan with
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1`
  and `--wrap=malloc/calloc/realloc/free` trapping allocations during dispatch):
  - 9 shape/scaling cases up to `[2, 4, 64, 16]` (`theta = 10000.0` and
    `500000.0`, `scale = 0.5, 1.0, 2.0`, positions up to `16777215`).
  - Maximum orthogonality error: `1.304e-07`.
  - Maximum inverse error (`backward(forward(x))` and `forward(backward(x))`):
    `2.384e-07`.
  - 379 central finite-difference checks: maximum absolute error `7.808e-06`.
  - Full negative suite verified byte-for-byte untouched output sentinels.
- Independent binary64 Python oracle (`tests/mod5_rope/test_mod5_rope_oracle.py`):
  - 9 cases, 43,716 forward & backward values, maximum absolute error
    `1.538e-07` (`<= 1e-6`).

## Compiled Eshkol AOT 2-layer Llama-style decoder

`tests/mod5_rope/test_mod5_rope_aot.esk` compiles with
`eshkol-run --strict-types --optimize 0 --no-stdlib` linking
`libeshkol_transformer_mod5_rope.a`, `libeshkol_transformer_mod5.a`,
`libeshkol_transformer_k1.a`, `mod5_rope_eshkol_bridge.o`, and
`mod5_eshkol_bridge.o`.

Architecture (`N=1, T=8, D=16, F=32, Hq=4, Hkv=2, Dh=4, V=8`):
- 2 pre-norm Llama-style decoder layers:
  `RMSNorm -> (Wq, Wk, Wv) -> RoPE(Q, K) -> causal GQA (ALiBi=0) -> Wo -> +res -> RMSNorm -> (Wgate, Wup) -> SwiGLU -> Wdown -> +res`
- Final `RMSNorm -> W_lm -> Cross-Entropy loss` over 8 target tokens.
- Malformed FFI checks (odd `Dh=3`, misaligned raw pointers via `ptr-add`,
  non-finite input, invalid `theta`/`scale`, negative position): PASS with
  untouched output sentinels.
- Central finite-difference verification across all 20 parameter tensors and
  input `x0` (42 checks): `max_abs_err = 1.1863484360219445e-05`, PASS.
- Two fresh compilations produce byte-identical ELFs and byte-identical stdout:
  - ELF SHA-256: `84f6c4f91491b20eef3f6143547368354046484dadddc658534287a8c9174b6c`

| SGD Step | Cross-Entropy Loss |
|---:|---:|
| 0 | 2.7692083156846783 |
| 1 | 1.914849891737687 |
| 2 | 1.405782779661751 |
| 3 | 1.0484409783680018 |
| 4 | 0.7776871269267092 |
| 5 | 0.5722118842850066 |
| 6 | 0.42404034034959587 |
| 7 | 0.3222136205274513 |
| 8 | 0.2527173000164875 |
| 9 | 0.20423311314487522 |
| 10 | 0.1693674245345082 |
| 11 | 0.14353964186829057 |
| 12 | 0.12388576726268735 |

Final weighted parameter checksum: `1178.4679830291452`.

## Limitations and scope

- Bounded to serial CPU binary32 evaluation on the reviewed x86-64 FP lane for
  `N in 1..2`, `H in 1..4`, `T in 1..64`, and even `Dh in 2..16`.
- K1 scalar views encode `theta` and `scale` as binary32 (`f32[]`), while
  direct C11 helpers accept binary64 `theta` and `scale`.
- GPU execution, mixed precision, arbitrary shapes, and performance claims are
  outside this scope.
