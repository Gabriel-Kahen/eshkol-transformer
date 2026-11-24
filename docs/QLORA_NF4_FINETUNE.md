# QLoRA NF4 Double-Quantized Fine-Tuning Engine

This document specifies the standalone Eshkol-native **QLoRA (NormalFloat4 Double-Quantized Frozen Base + Trainable LoRA Fine-Tuning)** engine, exact Dettmers et al. (2023) NF4 codebook and packing layout, single- and double-quantized scale codecs, fused frozen-NF4 + LoRA linear forward/backward kernels, decoupled AdamW 2-layer MLP trainer, versioned `ETQL1000` v1.0 checkpoint contract, and compiled Eshkol AOT verification.

## Artifacts and public headers

- Public C11/C++17 ABI: `include/eshkol_transformer/qlora_nf4_abi.h`
- Core engine: `native/qlora_nf4.c` -> `build/qlora_nf4/libeshkol_transformer_qlora_nf4.a`
- Eshkol AOT bridge: `native/qlora_nf4_eshkol_bridge.h`, `native/qlora_nf4_eshkol_bridge.c` -> `build/qlora_nf4/libeshkol_transformer_qlora_nf4_aot.a`
- Native C11 & C++17 tests: `tests/qlora_nf4/test_qlora_nf4_native.c`, `tests/qlora_nf4/header_qlora_cpp.cpp`
- Development oracle: `tests/qlora_nf4/test_qlora_nf4_oracle.py`
- Compiled Eshkol AOT driver: `tests/qlora_nf4/test_qlora_nf4_aot.esk`
- Build and test scripts: `scripts/build-qlora-nf4.sh`, `scripts/test-qlora-nf4.sh`

## Mathematical contract

### 1. Exact NF4 quantile codebook & block packing

`ET_NF4_CODEBOOK[16]` implements the exact Dettmers et al. (2023) 16-entry NormalFloat4 quantile table:

```
[-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
 -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
  0.07958029955625534,  0.16093020141124725,  0.24611230194568634, 0.33791524171829224,
  0.44070982933044434,  0.5626170039176941,   0.7229568362236023,  1.0]
```

- **Block geometry**: Row-aligned blocks with `block_size` in `{32, 64, 128}` (default `64`), `cols_in % block_size == 0` (blocks never cross rows).
- **Deterministic nearest-codebook rounding**: Each block computes `absmax = max_j |w[j]|`. All-zero blocks emit `absmax = 0.0f` and nibble `7` (`0x77` packed). Otherwise each element normalizes in `binary64` (`v = (double)w[j] / (double)absmax`) and selects the largest index `q in [0, 15]` satisfying `v >= mid[q-1]`, where `mid[i] = 0.5 * ((double)ET_NF4_CODEBOOK[i] + (double)ET_NF4_CODEBOOK[i+1])` (strict midpoint tie-break to larger index, matching the Python oracle bit-for-bit).
- **Nibble packing**: Two 4-bit nibbles per byte, low-nibble-first: element `2*p` in `(byte & 0x0F)`, element `2*p + 1` in `((byte >> 4) & 0x0F)`.

### 2. Single Quantization (`NF4_SQ`) vs Double Quantization (`NF4_DQ`)

For `num_params = rows_out * cols_in` and `num_blocks = num_params / block_size`:

| Mode | Flat packed layout | Total bytes | Bits/param (`block_size=64`) |
|---|---|---|---|
| `ET_QLORA_QUANT_NF4_SQ` (`1`) | `nibbles[num_params/2]` + `absmax_f32_le[num_blocks]` | `num_params/2 + 4 * num_blocks` | `4.500` (`7.111x` vs FP32) |
| `ET_QLORA_QUANT_NF4_DQ` (`2`) | `nibbles[num_params/2]` + `q_scales_u8[num_blocks]` + `(offset_f32, scale_f32)[num_dq_blocks]` | `num_params/2 + num_blocks + 8 * ceil(num_blocks / dq_block_size)` | `4.129` at `dq_block_size=256` (`7.750x` vs FP32) |

For each second-level DQ block (`dq_block_size` in `{16, 32, 64, 256}`):
- `offset_f32 = min(absmax_group)`, `scale_f32 = (float)(((double)max - (double)offset_f32) / 255.0)`.
- `q_s[b] = clamp(floor(((double)absmax[b] - (double)offset_f32) / (double)scale_f32 + 0.5), 0, 255)`.
- Deterministic scale reconstruction: `absmax_hat[b] = (float)((double)offset_f32 + (double)scale_f32 * (double)q_s[b])`.
- Weight dequantization: `w_hat = (float)((double)absmax_hat * (double)ET_NF4_CODEBOOK[q])`.

### 3. Fused Frozen-NF4 + LoRA Linear Kernels & 2-Layer Trainer

All hot numerical kernels (`quantize`, `dequantize`, `linear_forward`, `linear_backward`, `merge_to_f32`, `f32_linear_forward`, `adamw_step`, `mlp_forward`, `mlp_train_step`, `checkpoint_serialize`, `checkpoint_parse`) execute with **zero heap allocations** (enforced via `-Wl,--wrap=malloc,calloc,realloc,free`), `binary64` inner dot accumulations, strict IEEE-754 rounding checks (`FE_TONEAREST`, x86-64 x87 CW & MXCSR), pairwise disjointness checks, finite-only operand/scale/gradient validation, and failure atomicity.

- **Forward**: `Y[m, out] = sum_k dequant(W_nf4[out, k]) * X[m, k] + bias[out] + (alpha / r) * ((X @ A^T) @ B^T)[m, out]` with `A: [r, cols_in]`, `B: [rows_out, r]`. Zero-initialized `B` matches frozen NF4 base-only forward bitwise (`memcmp == 0`).
- **Analytic backward** (`W_nf4` strictly `const uint8_t *`, never mutated):
  - `H = X @ A^T`, `dH = (alpha / r) * dY @ B`
  - `dB = (alpha / r) * dY^T @ H`
  - `dA = dH^T @ X`
  - `dX = dY @ dequant(W_nf4) + dH @ A`
  - Optional `dbias = sum_m dY[m, :]`
- **Merge helper (`et_qlora_merge_to_f32`)**: `W_merged[out, k] = dequant(W_nf4[out, k]) + (alpha / r) * (B @ A)[out, k]`.
- **2-layer QLoRA MLP trainer**: `Z1 = Linear1(X)` -> `H1 = SiLU(Z1)` -> `Y = Linear2(H1)`, MSE loss, decoupled AdamW updating LoRA `(A1, B1, A2, B2)` only while keeping `(W1_nf4, W2_nf4)` strictly frozen.

### 4. Versioned Checksummed Checkpoint Contract (`ETQL1000` v1.0)

| Offset | Size (bytes) | Field | Validation |
|---:|---:|---|---|
| `0` | `8` | Magic `"ETQL1000"` | Exact match (`ET_QLORA_ERR_FORMAT`) |
| `8` | `8` | `version_major=1`, `version_minor=0` (`u32` LE) | Exact match (`ET_QLORA_ERR_VERSION`) |
| `16` | `32` | `d_in, d_hidden, d_out, r, block_size, dq_block_size, quant_mode, alpha` | Matches expected model config (`ET_QLORA_ERR_SHAPE`) |
| `48` | `28` | `adamw_step` (`u64` LE), `lr, beta1, beta2, eps, weight_decay` (`f32` LE) | Valid finite optimizer state |
| `76` | `4` | `flags` (`u32` LE) | Must be `0` (`ET_QLORA_ERR_FORMAT`) |
| `80` | `32` | `base_nf4_sha256` | Matches canonical length-prefixed SHA-256 of `(W1_nf4, W2_nf4)` (`ET_QLORA_ERR_BASE_HASH`) |
| `112` | `16` | Reserved zeros | Must be all `0` (`ET_QLORA_ERR_FORMAT`) |
| `128` | `12 * lora_params` | `A1, B1, A2, B2, m_*, v_*` (`f32` LE) | All finite; all `v_* >= 0.0f` (`ET_QLORA_ERR_NONFINITE`) |
| Trailer | `32` | `checkpoint_sha256` | SHA-256 over bytes `[0, size - 32)` (`ET_QLORA_ERR_CHECKSUM`) |

- Atomic save (`et_qlora_checkpoint_save`): sibling `mkstemp` (`0600` via `fchmod`) + full write + `fsync` + `close` + `rename`.
- Checked load (`et_qlora_checkpoint_load`): `O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC` (rejects symlinks with `ET_QLORA_ERR_SYMLINK`), verifies `S_ISREG` and exact size before parsing.

## Measured verification results

Verified inside the pinned Ubuntu 22.04 container (`eshkol-r0-debug:20260928`, Clang 21.1.8, pinned Eshkol `fe9dfd52`) via `bash scripts/test-qlora-nf4.sh`, `python3 -m unittest -v tests/q0/test_python_isolation.py`, and `bash scripts/check-ci-topology.sh`:

| Check | Measured result |
|---|---|
| Core deterministic archive (`libeshkol_transformer_qlora_nf4.a`) | SHA-256 `d6c71ccccf4efa3b6a21cf01fdb3a749db1d5eee5ab105d14a84207cee9947ef` |
| Bridge deterministic archive (`libeshkol_transformer_qlora_nf4_aot.a`) | SHA-256 `bc62e5239948624531290903e4f5c7bdb2f52c9ee3f94ac2037764a2bbec011c` |
| Compiled Eshkol AOT ELF (`tests/qlora_nf4/test_qlora_nf4_aot.esk`) | Byte-identical across 2 fresh builds; SHA-256 `574ca34ac0f068f92f47b8d3d00243d9b1c42440a5ed8d438fbb22d3e2c34786` |
| C++17 ABI header & archive linkage | PASS (`-O2` + ASan/UBSan/LSan) |
| Native `128x128` Gaussian weight footprint & RMSE (`block=64, dq=256`) | `4.1289` bits/param (`7.750x`), `nf4_sq=2.289048e-02`, `nf4_dq=2.289057e-02` < `unif_int4=2.693712e-02` |
| Native central FD gradient checks (`dX, dA, dB, dbias`) | 128 checks, `max_fd_err = 1.455e-06` (`< 1e-5`), `max_merge_diff = 2.384e-07` |
| Native 12-step AdamW & 6+6 `ETQL1000` resume parity | `init=2.956661e-02 -> final=7.653659e-03`, bitwise resume (`memcmp == 0`), `0600` mode |
| Python binary64 oracle (`tests/qlora_nf4/test_qlora_nf4_oracle.py`) | 4 linear exact-packed cases (`max_err = 0.000e+00`) + 12-step AdamW MLP (`max_err = 5.551e-17`) |
| Sanitizers (`ASan`, `UBSan`, `LSan detect_leaks=1`) + zero-alloc trap | PASS (C11 + C++17 suites) |
| AOT multi-layer (`128->128->64`) footprint (`block=64, dq=256`) | FP32 `98304` B vs SQ `13824` B (`4.5` bpp) vs DQ `12688` B (`4.1302` bpp, `7.748x`) |
| AOT Gaussian RMSE comparison | `nf4_sq=0.022976` & `nf4_dq=0.022982` < `unif_int4=0.027021` |
| AOT central FD checks (`dX, dA, dB`) | 12 checks, `max_err = 4.028e-06` (`< 1e-5`) |
| AOT 12-step AdamW & 6+6 fresh-process resume parity | `0.025593 -> 0.007859`, identical `BASE_SHA256`, `Y_SHA256`, `STATE_SHA256`, `max_merge_diff = 4.768e-07` |
| AOT negative & tamper suite | 13/13 PASS (invalid block/DQ sizes, invalid ranks, non-finite inputs, capacity mismatch, base-hash mismatch, corrupt checksum, truncated, bad magic, bad version, symlink) |
| Python isolation & CI topology gates | PASS (5/5 unittest OK; 23 shared suites / 38 commands) |

### Compiled Eshkol AOT 12-step AdamW trace (`128->128->64`, `r=4`, `alpha=8.0`, `lr=0.003`)

| Step | Uninterrupted 12-step loss | 6+6 fresh-process resumed loss |
|---:|---:|---:|
| 0 | `0.025593319066957117` | — |
| 1 | `0.024038123977261832` | — |
| 2 | `0.022173440981680382` | — |
| 3 | `0.02009080734838838` | — |
| 4 | `0.01798743707164252` | — |
| 5 | `0.016049315297688344` | — |
| 6 | `0.01435784931725001` | saved (`ETQL1000` v1.0) |
| 7 | `0.012879230798827366` | `0.012879230798827366` |
| 8 | `0.011554950682972815` | `0.011554950682972815` |
| 9 | `0.010376536286527776` | `0.010376536286527776` |
| 10 | `0.00937326939399382` | `0.00937326939399382` |
| 11 | `0.008550453675959857` | `0.008550453675959857` |
| 12 | `0.007858855973242599` | `0.007858855973242599` |

## Limitations and scope

- CPU serial deterministic engine bounded to `M <= 128`, `cols_in <= 512`, `rows_out <= 512`, `r <= 32`, `block_size in {32, 64, 128}`, and `dq_block_size in {16, 32, 64, 256}`.
- Does not implement GPU CUDA/HIP kernels, paged optimizer offloading, or bfloat16/fp16 compute activations; inner dot accumulations run in IEEE-754 `binary64` with `binary32` tensor storage.
- `tests/qlora_nf4/test_qlora_nf4_oracle.py` is a development-only verification oracle; all delivered quantization, training, checkpointing, merge, and AOT execution paths are Eshkol/C-native.
