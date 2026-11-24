# Multi-LoRA Batched Serving (S-LoRA / Punica SGMV) & Multi-Adapter Training

## Overview

The standalone Eshkol-native Multi-LoRA SGMV subsystem implements batched serving and simultaneous multi-adapter training for heterogeneous-rank LoRA adapters sharing a single frozen base linear projection (`W_base[out_dim, in_dim]` plus optional `bias[out_dim]`), modeled on S-LoRA / Punica Segmented Gather-Matrix-Multiply (SGMV).

### Delivered Files

- Public C11/C++17 ABI: `include/eshkol_transformer/multi_lora_sgmv_abi.h`
- Core engine: `native/multi_lora_sgmv.c`
- Eshkol AOT bridge: `native/multi_lora_sgmv_eshkol_bridge.h`, `native/multi_lora_sgmv_eshkol_bridge.c`
- C++17 linkage & layout test: `tests/multi_lora_sgmv/header_mlora_cpp.cpp`
- Native C11 & zero-alloc audit test: `tests/multi_lora_sgmv/test_multi_lora_sgmv_native.c`
- Independent Python `binary64` oracle: `tests/multi_lora_sgmv/test_multi_lora_sgmv_oracle.py`
- Compiled Eshkol AOT driver: `tests/multi_lora_sgmv/test_multi_lora_sgmv_aot.esk`
- Build & verification scripts: `scripts/build-multi-lora-sgmv.sh`, `scripts/test-multi-lora-sgmv.sh`

## Mathematical Contract & SGMV Formulation

Given frozen base weight `W_base[out_dim, in_dim]`, optional `bias[out_dim]`, batch input `X[M, in_dim]`, and arbitrary unsorted per-row adapter routing `row_adapter_ids[M]` where each entry is either `ET_MLORA_BASE_ONLY_ID (-1)` or an active registered `adapter_id >= 0` with rank `r_a in [1, max_rank]`, scaling `alpha_a > 0`, and weights `A_a[r_a, in_dim]`, `B_a[out_dim, r_a]`:

1. **Segmented Dispatch**:
   - Rows are partitioned in a single zero-allocation pass into base-only rows (`-1`) and contiguous per-adapter index segments `seg_rows[seg_offsets[s] .. seg_offsets[s] + seg_counts[s])`, preserving ascending batch row order within every segment.
2. **SGMV Forward (`et_mlora_sgmv_forward_v1`)**:
   - All inner products accumulate in IEEE-754 `binary64` under `FE_TONEAREST` with `#pragma STDC FP_CONTRACT OFF` and x86-64 x87/MXCSR environment verification (`ET_MLORA_ERR_FP_ENV` on violation):
     - Base-only rows (`adapter_id == -1`):
       `Y[i, o] = (float)(sum_k W_base[o, k] * X[i, k] + bias[o])`
     - Adapted rows (`adapter_id == a`, `scale_a = (double)alpha_a / (double)r_a`):
       - **SGMV Shrink**: `H_seg[p, r] = sum_k A_a[r, k] * X[seg_rows[p], k]` (`binary64`)
       - **SGMV Expand + Base**: `Y[i, o] = (float)(sum_k W_base[o, k] * X[i, k] + bias[o] + scale_a * sum_r B_a[o, r] * H_seg[p, r])`
   - Because each row's `binary64` reduction is invariant to batch composition and ordering, batched SGMV forward is **bitwise identical (`0.000e+00`, `memcmp == 0`)** to per-row single-adapter execution (`et_mlora_single_adapter_forward_v1`).
3. **Multi-Adapter Analytic Backward (`et_mlora_sgmv_backward_v1`)**:
   - Computes `dX[M, in_dim]` and per-adapter `dA_a[r_a, in_dim]`, `dB_a[out_dim, r_a]` in `binary64` staging buffers before atomic commit:
     - `dH_seg[p, r] = scale_a * sum_o dY[i, o] * B_a[o, r]`
     - `dB_a[o, r]   = (float)(scale_a * sum_{p in seg(a)} dY[i, o] * H_seg[p, r])`
     - `dA_a[r, k]   = (float)(sum_{p in seg(a)} dH_seg[p, r] * X[i, k])`
     - `dX[i, k]     = (float)(sum_o dY[i, o] * W_base[o, k] + sum_r dH_seg[p, r] * A_a[r, k])`
   - **Zero cross-adapter gradient bleed**: every active registered adapter absent from `row_adapter_ids` receives exact `+0.0f` (`0x00000000`) gradients; every active participating adapter receives bitwise-identical gradients to a compact single-adapter sub-batch of its own rows.
   - Frozen `W_base` and `bias` remain strictly immutable (verified by live SHA-256 digest `et_mlora_get_base_sha256_v1`).
4. **Per-Adapter AdamW (`et_mlora_adamw_step_adapter_v1`)**:
   - Maintains independent `(m_A, v_A, m_B, v_B, step)` per adapter slot, updated atomically in `binary64` with decoupled weight decay and deterministic integer-exponent bias correction.

## Adapter Registry Lifecycle & Hot-Swap

- Fixed-slot pool (`max_adapters <= 16`) allocated once in `et_mlora_registry_create_v1`.
- **Zero heap allocations** in all hot-path operations (`register`, `unregister`, `pin`, `unpin`, `sgmv_forward`, `single_adapter_forward`, `sgmv_backward`, `adamw_step_adapter`), audited with linker `--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free`.
- `et_mlora_pin_adapter_v1` / `et_mlora_unpin_adapter_v1` track per-slot `pin_count` (double-unpin rejected with `ET_MLORA_ERR_UNDERFLOW`).
- `et_mlora_unregister_adapter_v1` rejects pinned adapters with `ET_MLORA_ERR_BUSY`. Reusing a freed slot monotonically increments `generation`.

## Versioned Multi-Adapter Bundle (`ETML1000` v1.0)

Non-executable little-endian binary format persisting frozen-base binding + all registered adapters (sorted by ascending `adapter_id`) + optional per-adapter AdamW optimizer state:

- **Header (72 bytes)**: magic `"ETML1000"` (8 B), `version_major=1` (4 B), `version_minor=0` (4 B), `in_dim` (4 B), `out_dim` (4 B), `max_rank` (4 B), `adapter_count` (4 B), `flags` (4 B: bit 0 `HAS_BIAS`, bit 1 `HAS_ADAMW`), `total_bytes` (4 B), `base_sha256` (32 B over `(in_dim, out_dim, has_bias, W_base, bias)`).
- **Adapter Records**: 24-byte header (`adapter_id`, `rank`, `alpha` bits, `adamw_step`, `generation`, `reserved=0`) + finite `A`, `B` (and optional finite `m_A`, non-negative `v_A`, `m_B`, non-negative `v_B`).
- **Trailer (32 bytes)**: SHA-256 over all preceding bytes.
- **Persistence safety**: `et_mlora_bundle_save_v1` uses `mkstemp` + `fchmod(0600)` + `write` + `fsync` + atomic `rename`; `et_mlora_bundle_load_v1` opens with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC` (rejecting symlinks with `ET_MLORA_ERR_SYMLINK`) and commits atomically into the registry (rejecting pinned registries with `ET_MLORA_ERR_BUSY`).

## Measured Verification Evidence (`eshkol-r0-debug:20260928`)

Verified inside pinned container `eshkol-r0-debug:20260928` (`CC=clang-21`, `CXX=clang++-21`, `--network none`):

- **Deterministic Archives (`scripts/build-multi-lora-sgmv.sh`, dual-build `cmp` PASS)**:
  - `build/multi_lora_sgmv/libeshkol_transformer_multi_lora_sgmv.a`: `7faaafcad686f37dde47c11d7f5ac127ce2d84566f9a1c837220990a5edfa11f`
  - `build/multi_lora_sgmv/libeshkol_transformer_multi_lora_sgmv_aot.a`: `3a64f4c4cbf5ad43e24411a2ef97dd37caa276b2bb671fa9f9c6473732bc7dcd`
- **Compiled Eshkol AOT ELF (`--strict-types --optimize 0 --no-stdlib`, dual fresh builds `cmp` PASS)**:
  - `test_multi_lora_sgmv_aot` SHA-256: `c55ce447ae3803e17ca5318a923720769af681f42138879f2d60abbcf9132f9a`
- **Native C11 (`-O2` and ASan/UBSan/LSan `detect_leaks=1:halt_on_error=1`)**:
  - Heterogeneous ranks `r = 0 (base-only -1), 1, 2, 4, 8, 16`: bitwise parity `0.000e+00`, central FD `max_fd_err=0.000e+00`, idle adapter bleed `0.000e+00`, hot-path heap allocations `0`.
- **Independent Python `binary64` Oracle (`tests/multi_lora_sgmv/test_multi_lora_sgmv_oracle.py`)**:
  - 3 heterogeneous SGMV forward/backward cases + 12-step multi-adapter AdamW trajectory (`3,305` values): `sgmv_max_err=0.000e+00`, `adamw_max_err=0.000e+00`.
- **Compiled Eshkol AOT Driver (`tests/multi_lora_sgmv/test_multi_lora_sgmv_aot.esk`)**:
  - Mixed unsorted `M=12` (`base_rows=3`, `adapted_rows=9`, `segments=3`, ranks `2, 4, 8`): `bitwise_diff=0 PASS`.
  - Step-1 MSE central FD checks (`10` checks across `dX`, `dA`, `dB` for all 3 ranks): `max_err=2.3773482382694056e-06`, `idle_bleed=0 PASS`.
  - 12-step heterogeneous multi-adapter AdamW training (all 3 losses strictly decreasing; frozen base SHA-256 invariant):
    - Adapter 101 (`r=2`): `0.11798611108834661 -> 0.05599620298183078`
    - Adapter 202 (`r=4`): `0.16289999657968848 -> 0.08825447135219455`
    - Adapter 303 (`r=8`): `0.16455832998272513 -> 0.04799863617034929`
  - Hot-swap (`pin` -> `ET_MLORA_ERR_BUSY` -> `unpin` -> `unregister` -> hot-load adapter `505`): `old_gen=1 new_gen=2 busy_reject=PASS`.
  - Fresh OS process `ETML1000` reload: exact `Y_SHA256` and `BASE_SHA256` match across all 4 adapters + all tamper/boundary/FP-env negatives PASS.
- **Repository Gates**:
  - `tests/q0/test_python_isolation.py`: `5/5 OK`
  - `scripts/check-ci-topology.sh`: `CI TOPOLOGY PASS`

## Measured Limitations & Unsupported Cases

- Bounded CPU reference dimensions: `max_adapters <= 16`, `M <= 64`, `in_dim <= 64`, `out_dim <= 64`, `max_rank <= min(16, in_dim, out_dim)`. Larger production dimensions require extending the fixed-slot pool capacities or an explicit arena allocator.
- CPU-only `binary32` storage with IEEE-754 `binary64` inner accumulation; no GPU/HIP/CUDA kernel dispatch or quantized/mixed-precision adapter weights are claimed here.
- Single-threaded synchronous API; concurrent host-thread access to the same `et_mlora_registry` requires external synchronization.
- Non-IEEE rounding modes (`fegetround() != FE_TONEAREST`) or x86-64 FTZ/DAZ/unmasked MXCSR/x87 states are explicitly rejected with `ET_MLORA_ERR_FP_ENV` rather than silently accepted.
