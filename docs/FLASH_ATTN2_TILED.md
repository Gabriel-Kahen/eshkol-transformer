# FlashAttention-2 Exact Tiled Online-Softmax Attention (Forward + `O(T)` LSE Recomputation Backward)

This module implements a standalone, Eshkol-native **FlashAttention-2** exact
tiled online-softmax attention engine (Dao 2023) supporting Multi-Head Attention
(MHA), Grouped-Query Attention (GQA), and Multi-Query Attention (MQA), causal
masking, sliding-window attention, Gemma-2 logit softcapping, `O(T)`
log-sum-exp recomputation backward without ever materializing a `T_q x T_k`
attention matrix, zero heap allocations, failure-atomic staged writes, and a
compiled Eshkol AOT driver.

## Artifacts and public contracts

- Public C11/C++17 header: `include/eshkol_transformer/flash_attn2_abi.h`
  (`extern "C"`, standard-layout structs, ABI `1.0`)
- Core archive: `build/flash_attn2/libeshkol_transformer_flash_attn2.a`
  (from `native/flash_attn2.c`)
  - Deterministic archive SHA-256:
    `c75340d048569f0a5a6615bd1d2a32ea879a2cb58bd6e93e8775df0c4f84b430`
  - Exported symbols:
    - `et_fa2_tile_scratch_bytes_v1`
    - `et_fa2_validate_config_v1`
    - `et_fa2_forward_v1`
    - `et_fa2_backward_v1`
  - Zero dynamic allocator imports (`nm -u` verified: no `malloc`, `calloc`,
    `realloc`, or `free`; runtime dispatch verified under
    `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free`).
- Eshkol AOT bridge archive:
  `build/flash_attn2/libeshkol_transformer_flash_attn2_aot.a`
  (from `native/flash_attn2_eshkol_bridge.{h,c}`)
  - Deterministic archive SHA-256:
    `d74118a06b497113341372b7dd38ce54faa132cf09de5cd8624f22ea1117f9f4`
  - Exports only `et_fa2_aot_*` transport, stats accessor, FP-env negative test
    helper, and scalar marshaling symbols with zero allocator imports.
- Compiled Eshkol AOT driver: `tests/flash_attn2/test_flash_attn2_aot.esk`
  - Compiled via `eshkol-run --strict-types --optimize 0 --no-stdlib`
  - Byte-identical dual fresh compilation ELF SHA-256:
    `822f86fef12732c58718dd9e127ad21d56789b0b23b9563e62b30a51fa41d6cd`
- Build & verification scripts:
  - `scripts/build-flash-attn2.sh`
  - `scripts/test-flash-attn2.sh`
- Test suites:
  - `tests/flash_attn2/header_fa2_cpp.cpp`
  - `tests/flash_attn2/test_flash_attn2_native.c`
  - `tests/flash_attn2/test_flash_attn2_oracle.py`

## Tensor shape, domain & precision contract

Admitted bounded domain (`et_fa2_config_v1`):

- `batch` (`B`): `1 .. 4`
- `num_heads` (`H_q`): `1 .. 16`
- `num_kv_heads` (`H_kv`): `1 .. H_q` with `H_q % H_kv == 0`
  (MHA `H_kv = H_q`, GQA `1 < H_kv < H_q`, MQA `H_kv = 1`;
  head mapping `h_kv = h / (H_q / H_kv)`)
- `seq_q` (`T_q`), `seq_k` (`T_k`): `1 .. 256` (square or non-square)
- `head_dim` (`D_h`): `1 .. 64`
- `tile_r` (`B_r`), `tile_c` (`B_c`): each in `{8, 16, 32}`
- `causal`: `0` (disabled) or `1` (enabled)
- `sliding_window` (`W`): `0` (disabled) or `>= 1`
- `reserved`: must be `0`
- `scale`: `0.0` selects default `1.0 / sqrt((double)D_h)`; otherwise finite `> 0.0`
- `logit_softcap` (`c`): `0.0` (disabled); otherwise finite `> 0.0`

Dense row-major layouts (BHTD / BNSD):

| Tensor | Shape | Storage Type |
|---|---|---|
| `Q`, `O`, `dO`, `dQ` | `[B, H_q, T_q, D_h]` | IEEE-754 `binary32` (`float`) |
| `K`, `V`, `dK`, `dV` | `[B, H_kv, T_k, D_h]` | IEEE-754 `binary32` (`float`) |
| `LSE` | `[B, H_q, T_q]` | IEEE-754 `binary64` (`double`) |

All dot products, softcapping, online softmax accumulators `(m, l, O_acc)`,
backward row projection `D_i`, and tile gradient accumulators execute in
IEEE-754 `binary64` (`double`) under `FE_TONEAREST` with `#pragma STDC
FENV_ACCESS ON`, `#pragma STDC FP_CONTRACT OFF`, `-ffp-contract=off
-fexcess-precision=standard -fno-fast-math`, and x87/MXCSR FP-environment
guards (rejecting non-nearest rounding, FTZ/DAZ, or unmasked exceptions with
`ET_FA2_ERR_FP_ENV`). All outputs remain byte-for-byte untouched on any error
via input validation plus non-committing preflight staging outside the
analytically overflow-free range.

## Exact masking, sliding window & softcapping semantics

### Exact pair predicate `pair_allowed(q_pos, k_pos)`

For `q_pos in [0, T_q - 1]` and `k_pos in [0, T_k - 1]`:

- **`causal == 1`, `W == 0`**: `k_pos <= q_pos`
- **`causal == 1`, `W >= 1`**: `0 <= (q_pos - k_pos) < W`
- **`causal == 0`, `W == 0`**: all `(q_pos, k_pos)`
- **`causal == 0`, `W >= 1`**: `|q_pos - k_pos| < W`

Every query row must admit at least one unmasked key, which holds if and only
if `W == 0 || T_q <= T_k || (T_q - T_k) < W`. Configurations violating this
invariant fail explicitly with `ET_FA2_ERR_SHAPE_MISMATCH`.

### Exact necessary-and-sufficient tile skipping

For query tile `[q_start, q_end)` (`q_last = q_end - 1`) and key tile
`[k_start, k_end)` (`k_last = k_end - 1`), the tile is completely masked and
skipped (`skipped_tiles++`) if and only if:

- `causal == 1`: `k_start > q_last`, OR (`W >= 1 && q_start - k_last >= W`)
- `causal == 0`: `W >= 1 && (k_start - q_last >= W || q_start - k_last >= W)`

Otherwise the tile is evaluated (`evaluated_tiles++`). Both forward and backward
satisfy `evaluated_tiles + skipped_tiles == total_tiles == B * H_q *
ceil(T_q / B_r) * ceil(T_k / B_c)`.

### Gemma-2 logit softcapping

When `logit_softcap = c > 0`:

```text
S_raw = (Q[b, h, i] . K[b, h_kv, j]) * scale
t     = tanh(S_raw / c)
S     = c * t
```

Exact chain-rule derivative in backward (free of `cosh` overflow):

```text
dS_raw = dS_capped * (1.0 - t * t)
```

## Tiled FlashAttention-2 algorithms & `O(B_r * B_c)` scratch

### Forward pass (`et_fa2_forward_v1`)

Iterates Q tiles of height `B_r` (outer) and KV tiles of width `B_c` (inner):

```text
m[r] = -inf,   l[r] = 0.0,   O_acc[r, 0..D_h-1] = 0.0
for each non-masked KV tile:
  S_tile[r, c] = softcap( (Q[r] . K[c]) * scale )   for allowed (q_pos, k_pos)
  m_tile       = max_c S_tile[r, c]
  m_new        = (l[r] == 0) ? m_tile : max(m[r], m_tile)
  alpha        = (l[r] == 0) ? 0.0    : exp(m[r] - m_new)
  P_tile[r, c] = exp(S_tile[r, c] - m_new)
  l[r]         = alpha * l[r] + sum_c P_tile[r, c]
  O_acc[r]     = alpha * O_acc[r] + sum_c P_tile[r, c] * V[c]
  m[r]         = m_new
O[r]   = (float)(O_acc[r] / l[r])
LSE[r] = m[r] + log(l[r])
```

No `T_q x T_k` score or probability matrix is ever allocated or materialized.
`et_fa2_tile_scratch_bytes_v1(B_r, B_c) = 2 * B_r * B_c * sizeof(double)` is
strictly `O(B_r * B_c)` and constant across sequence lengths (e.g., identically
`4096` bytes for `(B_r, B_c) = (16, 16)` at `T = 32, 64, 128, 256`).

### Recomputation backward pass (`et_fa2_backward_v1`)

For each `(b, h_kv)` and each query head `h` in its GQA group:

1. Precompute `D[i] = sum_d dO[b, h, i, d] * O[b, h, i, d]` (`O(T_q)`).
2. Iterate KV tiles of size `B_c` (outer) and Q tiles of size `B_r` (inner),
   skipping completely masked tiles.
3. Recompute `S_raw`, `S_eff`, `P_tile = exp(S_eff - LSE[i])`, `dP_tile = dO . V`,
   `dS_capped = P_tile * (dP_tile - D[i])`, and `dS_raw = dS_capped * (1 - t^2)`
   strictly inside `O(B_r * B_c)` tile buffers.
4. Accumulate `dQ` per query head and fold `dK_tile`, `dV_tile` across all query
   heads in the GQA group into `dK[b, h_kv]` and `dV[b, h_kv]`.

## Measured verification results

Verified inside `eshkol-r0-debug:20260928` (`clang-21`, `clang++-21`,
`eshkol-run --strict-types --optimize 0 --no-stdlib`):

| Gate | Measured Result |
|---|---|
| Deterministic archives (`ar rcsD`) | Byte-identical across two fresh builds; strict `nm` export/import audit PASS (zero allocator imports) |
| C++17 header & linkage (`-std=c++17`) | PASS (normal `-O2` and ASan/UBSan/LSan) |
| Constant `O(B_r * B_c)` scratch (`T = 32, 64, 128, 256`) | `4096` bytes identically at `(16, 16)` across all four `T` values; `skipped_tiles > 0` verified |
| Native parity & tile-size invariance (9 configs x `(8,8),(16,16),(32,16)`: MHA/GQA/MQA, causal/non-causal, window, softcap, non-square, `T=256`, `D_h=64`) | vs dense ref: `O = 0.000e+00`, `LSE = 8.882e-16`, `dQ = 2.980e-08`, `dK = 1.192e-07`, `dV = 0.000e+00`; tile invariance: `O = 0.000e+00`, `LSE = 8.882e-16`, `grad = 0.000e+00` |
| Native central FD gradients (`dQ/dK/dV`, GQA + Causal + Window + Softcap) | `checked = 300`, `max_abs_err = 6.811e-06` (`< 1e-5`) |
| Independent Python `binary64` oracle (`test_flash_attn2_oracle.py`, 9 cases, 11,124 values) | `max_err`: `O = 3.276e-08`, `LSE = 4.441e-16`, `dQ = 2.392e-08`, `dK = 5.652e-08`, `dV = 1.186e-07`, `py_fd_err = 5.396e-09` |
| Negative, aliasing, failure-atomic overflow & FP-env checks | All fail closed with byte-for-byte untouched sentinels; zero heap allocations enforced via `--wrap=malloc/calloc/realloc/free` |
| ASan + UBSan + LSan (`detect_leaks=1:halt_on_error=1`) | PASS (native C11 + C++17) |
| Compiled Eshkol AOT (`test_flash_attn2_aot.esk`) | Dual fresh compiles produce byte-identical ELF (`822f86fe...a41d6cd`); tile invariance `(8,8)/(16,16)/(32,16)` `skip8=28/64`, `err_o=0`, `err_lse32=4.441e-16`; constant scratch `4096` across `T=32,64,128,256`; FD `checked=24`, `max_abs_err=2.867e-06`; 12-step monotonic SGD `1.211444 -> 0.173508` (`checksum=766.280643`) |
| `tests/q0/test_python_isolation.py` & `scripts/check-ci-topology.sh` | PASS (5 isolation tests OK; 23 shared CI suites / 38 commands) |

## Measured limitations and unsupported cases

- **CPU-only execution**: executes on x86-64 CPU in IEEE-754 `binary64` internal
  accumulation with `binary32` tensor I/O; no GPU/CUDA/ROCm kernel is claimed.
- **Bounded domain**: `B <= 4`, `H_q <= 16`, `T_q, T_k <= 256`, `D_h <= 64`,
  and tile dimensions `B_r, B_c in {8, 16, 32}`; out-of-range dimensions fail
  explicitly with `ET_FA2_ERR_UNSUPPORTED`.
- **No attention dropout or arbitrary 2D boolean mask**: supports causal,
  sliding-window, and Gemma-2 softcapping predicates only.
- **Non-empty row invariant**: configurations where any query row has zero
  unmasked keys (`sliding_window > 0 && T_q >= T_k + sliding_window`) fail
  explicitly with `ET_FA2_ERR_SHAPE_MISMATCH`.
- **Strict x86-64 IEEE-754 FP environment**: requires `FE_TONEAREST`, masked
  exceptions, and FTZ/DAZ disabled; non-default FP environments fail explicitly
  with `ET_FA2_ERR_FP_ENV`.
