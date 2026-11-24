# MOD5 Multi-Head Latent Attention (MLA) and Compressed Latent KV-Cache

MOD5 MLA provides DeepSeek-style Multi-Head Latent Attention (`kernel.mla`,
operations `mla.forward` and `mla.backward`) through K1 ABI 1.0, direct C11
helpers, a zero-allocation compressed latent KV-cache (`C_kv`-only storage),
and an Eshkol AOT bridge verified with compiled Eshkol AOT training and
incremental decode.

## Artifacts and public contracts

- Header: `include/eshkol_transformer/mod5_mla_abi.h` (C11 and C++17 compatible)
- Provider archive: `build/mod5_mla/libeshkol_transformer_mod5_mla.a`
  - Reproducible archive SHA-256:
    `2c94ffeba653d11d2d1cf64065602c6492c363c5f2914119c4723612fb48a17e`
  - Exports only:
    - `et_mod5_mla_kernel_provider_v1`
    - `et_mod5_mla_forward_f32`, `et_mod5_mla_backward_f32`
    - `et_mod5_mla_latent_kv_bytes`, `et_mod5_mla_full_kv_bytes`,
      `et_mod5_mla_kv_compression_ratio`
    - `et_mod5_mla_cache_init`, `et_mod5_mla_cache_reset`,
      `et_mod5_mla_cache_prefill_f32`, `et_mod5_mla_cache_step_f32`
  - Zero allocator imports (`malloc`, `calloc`, `realloc`, `free` absent from
    undefined symbols and trapped via `--wrap` during dispatch and cache tests).
- Eshkol AOT bridge: `native/mod5_mla_eshkol_bridge.h`,
  `native/mod5_mla_eshkol_bridge.c`,
  `build/mod5_mla/libeshkol_transformer_mod5_mla_bridge.a`
  - Exports only `et_mod5_mla_aot_*` transport, telemetry, KV-cache, and scalar
    marshaling symbols with zero allocator imports.
  - Accepts untyped `void *` pointers so raw Eshkol FFI addresses (including
    intentionally misaligned `ptr-add` addresses) are validated without
    misaligned-pointer UB.

## Mathematical specification

Admitted domain: `N in 1..2`, `T in 1..32`, `D in 4..32`, `H in 1..4`,
`Dh in 2..16`, and `dc in 1..16` with strict KV compression `dc < 2 * H * Dh`.
Let `HD = H * Dh` and `scale = 1.0 / sqrt((double)Dh)`.

### Forward pass (`mla.forward`)

All internal projections, attention scores, log-sum-exp softmax, and output
accumulations execute in IEEE-754 binary64 (`double`) with contraction and fast
math disabled, rounding outputs to finite binary32 (`float`):

```text
C_kv = X @ W_dkv^T          [N, T, dc]      (W_dkv: [dc, D])
Q    = X @ W_q^T            [N, T, HD]      (W_q:   [HD, D])
K    = C_kv @ W_uk^T        [N, T, HD]      (W_uk:  [HD, dc])
V    = C_kv @ W_uv^T        [N, T, HD]      (W_uv:  [HD, dc])
```

For each batch `n`, head `h in 0..H-1`, and query position `tq in 0..T-1`,
causal keys `tk in 0..tq` are scored and normalized via exact binary64
log-sum-exp:

```text
score[tk]     = (sum_{k=0}^{Dh-1} Q[n, tq, h*Dh + k] * K[n, tk, h*Dh + k]) * scale
m             = max_{0 <= tk <= tq} score[tk]
lse           = m + log(sum_{tk=0}^{tq} exp(score[tk] - m))
p[tk]         = exp(score[tk] - lse)
Attn[n,tq,h]  = sum_{tk=0}^{tq} p[tk] * V[n, tk, h]
O             = Attn @ W_o^T                [N, T, D]  (W_o: [D, HD])
```

### Exact analytic backward (`mla.backward`)

Given upstream gradient `dO [N, T, D]`, exact analytic gradients are accumulated
in binary64 and rounded to finite binary32:

```text
dW_o   = sum_n (dO_n^T @ Attn_n)                           [D, HD]
dAttn  = dO @ W_o                                          [N, T, HD]

Per (n, h, tq):
  dp[tk]  = dot(dAttn[n, tq, h], V[n, tk, h])
  mean_dp = sum_{tk=0}^{tq} p[tk] * dp[tk]
  ds[tk]  = p[tk] * (dp[tk] - mean_dp)
  ddot    = ds[tk] * scale
  dQ[n, tq, h] += ddot * K[n, tk, h]
  dK[n, tk, h] += ddot * Q[n, tq, h]
  dV[n, tk, h] += p[tk] * dAttn[n, tq, h]

dW_uk  = sum_n (dK_n^T @ C_kv_n)                           [HD, dc]
dW_uv  = sum_n (dV_n^T @ C_kv_n)                           [HD, dc]
dC_kv  = dK @ W_uk + dV @ W_uv                             [N, T, dc]

dW_q   = sum_n (dQ_n^T @ X_n)                              [HD, D]
dW_dkv = sum_n (dC_kv_n^T @ X_n)                           [dc, D]
dX     = dQ @ W_q + dC_kv @ W_dkv                          [N, T, D]
```

## Compressed latent KV-cache memory savings

Standard Multi-Head Attention caches full `K` and `V` tensors per token:

```text
full_mha_kv_bytes = N * max_tokens * (2 * H * Dh) * sizeof(float)
```

`et_mod5_mla_kv_cache` stores **only** the compressed latent `C_kv` tensor in
caller-owned dense `f32[N, max_tokens, dc]` storage and reconstructs `K, V` on
the fly via `W_uk` and `W_uv` during `et_mod5_mla_cache_step_f32`:

```text
latent_kv_bytes   = N * max_tokens * dc * sizeof(float)
compression_ratio = (2 * H * Dh) / dc
```

| Configuration `[N, T, D, H, Dh, dc]` | `latent_kv_bytes` | `full_mha_kv_bytes` | Compression ratio |
|---|---:|---:|---:|
| `[1, 8, 16, 4, 8, 8]` (AOT) | 256 | 2,048 | **8.0x** |
| `[1, 16, 16, 4, 8, 8]` | 512 | 4,096 | **8.0x** |
| `[1, 16, 16, 4, 16, 8]` | 512 | 8,192 | **16.0x** |
| `[2, 32, 32, 4, 16, 16]` | 4,096 | 32,768 | **8.0x** |

## K1 ABI 1.0 contract and failure-atomic validation

- Provider / implementation: `mod5.mla.cpu-f32.serial`, version `1.0`, evidence
  `MOD5:cpu-f32-mla-v1`, `VERIFIED`, `deterministic = 1`, dtype `f32`, device
  `cpu`.
- Request shape `[N, T, D, H, Dh, dc]` is published as 16 pairwise-disjoint K1
  shape range rows covering the exact `dc < 2 * H * Dh` domain with zero
  over-admission.
- K1 tensor schemas:
  - `mla.forward`: 6 inputs (`X, W_q, W_dkv, W_uk, W_uv, W_o`), 1 output (`O`).
  - `mla.backward`: 7 inputs (`X, W_q, W_dkv, W_uk, W_uv, W_o, dO`), 6 outputs
    (`dX, dW_q, dW_dkv, dW_uk, dW_uv, dW_o`).
- Validation enforces:
  - Out-of-domain extents or `dc >= 2 * H * Dh` reject with `UNSUPPORTED`.
  - Null or misaligned pointers, overlapping inputs, outputs aliasing inputs,
    other outputs, K1 metadata, cache descriptors, or `error`, non-finite
    binary32 operands/intermediates/outputs, and KV-cache overflow reject with
    `INVALID_ARGUMENT` (with `ET_KERNEL_CODE_BUFFER_TOO_SMALL` on cache
    overflow).
  - Non-standard x86-64 x87/MXCSR FP environments (non-RN rounding, unmasked
    exceptions, FTZ, DAZ) reject with `UNSUPPORTED`.
  - Dry-run preflight restores `fenv_t` and `errno` and leaves all caller
    output buffers and cache state byte-for-byte untouched on any error.

## Verification results

Run `bash scripts/test-mod5-mla.sh` in the pinned `eshkol-r0-debug:20260928`
container (Ubuntu 22.04 x86-64, Clang/Clang++ 21.1.8, pinned Eshkol toolchain):

- Deterministic archives & symbol allowlist audit (`mod5_mla_provider.o`,
  `mod5_mla_eshkol_bridge.o`): **PASS** (byte-identical archives across two
  fresh build directories).
- C++17 header and linkage check (`clang++-21 -std=c++17 -Wall -Wextra -Werror
  -Wpedantic`): **PASS**.
- Native C11 suite (normal `-O2` and ASan/UBSan/LSan with
  `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1`
  and `--wrap=malloc/calloc/realloc/free` trapping allocations during dispatch):
  - 8 shape cases spanning `[1, 1, 4, 1, 2, 1]` through `[2, 32, 32, 4, 16, 16]`,
    including `H=4, Dh=8, dc=8` (`8.0x`) and `H=4, Dh=16, dc=8` (`16.0x`).
  - 412 Richardson-extrapolated central finite-difference checks across all 6
    gradients (`dX, dW_q, dW_dkv, dW_uk, dW_uv, dW_o`):
    `max_fd_err = 6.179e-06`.
  - Compressed latent KV-cache prefill+decode vs full-context MLA forward parity
    across all 8 cases: `max_cache_diff = 5.960e-08` (`<= 1e-6`), and bitwise
    identical (`0.0`) between partial prefill+step, token-by-token decode, and
    full prefill.
  - Full negative suite (invalid dimensions, `dc >= 2*H*Dh`, null/misaligned
    pointers, aliasing, NaN/Inf, finite-input overflow, cache overflow, corrupted
    cached `C_kv`, x87/MXCSR FP modes) verified byte-for-byte untouched output
    sentinels and cache state.
- Independent binary64 Python oracle (`tests/mod5_mla/test_mod5_mla_oracle.py`):
  - 8 cases, 25,180 verified values across `O`, `dX`, `dW_q`, `dW_dkv`, `dW_uk`,
    `dW_uv`, `dW_o`, and `O_cached`:
    `max_abs_diff = 6.144e-08`, `max_cached_diff = 6.144e-08` (`<= 1e-5`).
- Python isolation (`tests/q0/test_python_isolation.py`): 5/5 tests **PASS**.
- CI topology (`scripts/check-ci-topology.sh`): **PASS**.

## Compiled Eshkol AOT verification

`tests/mod5_mla/test_mod5_mla_aot.esk` compiles with
`eshkol-run --strict-types --optimize 0 --no-stdlib` linking
`libeshkol_transformer_mod5_mla.a`, `libeshkol_transformer_k1.a`, and
`mod5_mla_eshkol_bridge.o` at `N=1, T=8, D=16, H=4, Dh=8, dc=8`:

- Malformed FFI checks (`dc == 2*H*Dh`, `T=33`, misaligned `ptr-add` input,
  non-finite `+Inf` input, aliased output, and cache overflow) with untouched
  output sentinels: **PASS**.
- Central finite-difference checks across all 6 tensors (`18` checks):
  `max_abs_err = 9.25854877209531e-07`, **PASS**.
- Compressed latent KV-cache prefill (`3` tokens) + incremental decode (`5`
  tokens) using the trained weights vs full-context MLA forward:
  - `latent_bytes = 256`, `full_mha_kv_bytes = 2048`, `compression_ratio = 8`
  - `max_abs_diff = 5.9604644775390625e-08` (`<= 1e-6`), **PASS**.
- Two fresh compilations produced byte-identical ELFs and byte-identical stdout:
  - ELF SHA-256: `4d49db997d188760797f0ccb062fafb10515a00de80f01a48a9f0eaf5859ab7c`

| SGD Step | MSE Loss |
|---:|---:|
| 0 | 0.420970881660562 |
| 1 | 0.32859342636820543 |
| 2 | 0.25310815933052494 |
| 3 | 0.22267274262972112 |
| 4 | 0.20641499106919964 |
| 5 | 0.19589241352187653 |
| 6 | 0.18814229125612464 |
| 7 | 0.18207207810421627 |
| 8 | 0.17713056416142875 |
| 9 | 0.17299199381047228 |
| 10 | 0.1694471198751924 |
| 11 | 0.16635409067066986 |
| 12 | 0.16361287404814034 |

Final weighted parameter checksum: `-5.034568538374837`.

## Limitations and scope

- Bounded to serial CPU binary32 storage with binary64 internal arithmetic on
  the reviewed x86-64 FP lane for `N in 1..2`, `T in 1..32`, `D in 4..32`,
  `H in 1..4`, `Dh in 2..16`, `dc in 1..16` with `dc < 2 * H * Dh`.
- Decoupled RoPE query/key projections, absorbed matrix-absorption fast paths
  (`W_q^T W_uk`), GPU execution, mixed precision, arbitrary extents, and
  throughput benchmarks are outside this scope.
