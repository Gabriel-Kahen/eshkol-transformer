# LoRA Adapter Fine-Tuning Engine

This document specifies the Eshkol-native Low-Rank Adaptation (LoRA) linear
projection engine, frozen-base guarantees, lossless weight merge/unmerge,
versioned binary adapter checkpoint contract (`ETLR1000` v1.0), and compiled
Eshkol AOT verification.

## Artifacts and public headers

- Public C11/C++17 ABI: `include/eshkol_transformer/lora_adapter_abi.h`
- Core implementation: `native/lora_adapter.c` -> `build/lora/libeshkol_transformer_lora.a`
- Eshkol AOT bridge: `native/lora_eshkol_bridge.h`, `native/lora_eshkol_bridge.c` -> `build/lora/libeshkol_transformer_lora_aot.a`
- Native & C++17 tests: `tests/lora/test_lora_native.c`, `tests/lora/header_lora_cpp.cpp`
- Development oracle: `tests/lora/test_lora_oracle.py`
- Compiled Eshkol AOT verifier: `tests/lora/test_lora_aot.esk`
- Build and test scripts: `scripts/build-lora.sh`, `scripts/test-lora.sh`

## Mathematical contract

Given dense row-major binary32 tensors with dimensions `M` in `[1, 64]`,
`K` in `[1, 64]`, `N` in `[1, 64]`, rank `r` in `[1, 16]` with
`r <= min(K, N)`, and finite scaling parameter `alpha > 0.0f` (`scale = alpha / (float)r`):

| Tensor | Shape | Role | Mutability |
|---|---|---|---|
| `X` | `[M, K]` | Input activations | Read-only (`const float *`) |
| `W` | `[N, K]` | Base projection weights | Strictly frozen (`const float *`) |
| `A` | `[r, K]` | Down-projection adapter | Trainable (`et_lora_init_adapter` deterministic non-zero seed) |
| `B` | `[N, r]` | Up-projection adapter | Trainable (`et_lora_init_adapter` exact `+0.0f`) |
| `Y`, `dY` | `[M, N]` | Output / upstream gradient | Output / read-only |
| `dX` | `[M, K]` | Input gradient | Output |
| `dA` | `[r, K]` | Down-projection gradient | Output |
| `dB` | `[N, r]` | Up-projection gradient | Output |

### Serial binary32 evaluation order

All kernels evaluate in serial IEEE-754 binary32 with `#pragma STDC FENV_ACCESS ON`,
`#pragma STDC FP_CONTRACT OFF`, `-ffp-contract=off -fexcess-precision=standard -fno-fast-math`,
and zero heap allocation during forward, backward, merge, and unmerge:

- **Base forward**: `Y_base[m, n] = sum_{k=0..K-1} X[m, k] * W[n, k]`
- **Adapted forward**:
  - `H[m, i] = sum_{k=0..K-1} X[m, k] * A[i, k]`
  - `Y[m, n] = (sum_{k=0..K-1} X[m, k] * W[n, k]) + scale * (sum_{i=0..r-1} H[m, i] * B[n, i])`
  - Because `lora_acc` initializes at `+0.0f` and `scale > 0.0f`, zero-initialized `B` (`+0.0f`) produces a **bitwise identical** output to `et_lora_base_forward` under round-to-nearest-even.
- **Analytic backward** (no `dW`; `W` is strictly read-only):
  - Recompute `H = X @ A^T` (`[M, r]`)
  - `dB[n, i] = scale * sum_{m=0..M-1} dY[m, n] * H[m, i]` (`[N, r]`)
  - `dH[m, i] = scale * sum_{n=0..N-1} dY[m, n] * B[n, i]` (`[M, r]`)
  - `dA[i, k] = sum_{m=0..M-1} dH[m, i] * X[m, k]` (`[r, K]`)
  - `dX[m, k] = (sum_{n=0..N-1} dY[m, n] * W[n, k]) + (sum_{i=0..r-1} dH[m, i] * A[i, k])` (`[M, K]`)
- **Weight merge and unmerge**:
  - `delta[n, k] = scale * sum_{i=0..r-1} B[n, i] * A[i, k]`
  - `W_merged[n, k] = W_base[n, k] + delta[n, k]`
  - `W_restored[n, k] = W_merged[n, k] - delta[n, k]`
  - For dyadic-aligned operands (and whenever `B = 0`), merge followed by unmerge is **bitwise exact**; for general normalized binary32 operands, forward and restoration parity hold within `<= 1e-6`.

### Validation, safety, and failure atomicity

- **FP environment**: On x86-64, both x87 (`fnstcw`, bits `0x0c3f == 0x003f`) and MXCSR (`_mm_getcsr`, bits `0xffc0 == 0x1f80`) plus `fegetround() == FE_TONEAREST` are enforced (round-to-nearest, all exceptions masked, FTZ/DAZ disabled). Non-compliant environments return `ET_LORA_ERR_FP_ENV`.
- **Alignment and disjointness**: All float buffers must be 4-byte aligned, have non-overflowing address spans, and be pairwise disjoint (including disjoint from `cfg`/`out_meta`). Raw `void *` entry points (`et_lora_*_raw`) enable misaligned-pointer boundary checks without casting misaligned pointers in C11/C++17.
- **Finite-only binary32**: Every input operand, intermediate accumulator, and output value must be finite; NaN/Inf inputs or intermediate overflow return `ET_LORA_ERR_NONFINITE`.
- **Failure atomicity**: Outputs are staged into bounded stack buffers and committed only after all checks and computations succeed; any error leaves caller outputs untouched.

## Binary adapter checkpoint contract (`ETLR1000` v1.0)

Adapter checkpoints are non-executable (`0600`), versioned, little-endian binary
files with SHA-256 base-weight binding and full-file integrity verification:

| Byte offset | Size (bytes) | Field | Validation |
|---:|---:|---|---|
| `0` | `8` | Magic `"ETLR1000"` | Exact match (`ET_LORA_ERR_FORMAT`) |
| `8` | `4` | `version_major` (`uint32_t` LE) | Must be `1` (`ET_LORA_ERR_VERSION`) |
| `12` | `4` | `version_minor` (`uint32_t` LE) | Must be `0` (`ET_LORA_ERR_VERSION`) |
| `16` | `4` | `K` (`uint32_t` LE) | `1..64`, matches caller `expected_k` (`ET_LORA_ERR_SHAPE`) |
| `20` | `4` | `N` (`uint32_t` LE) | `1..64`, matches caller `expected_n` (`ET_LORA_ERR_SHAPE`) |
| `24` | `4` | `r` (`uint32_t` LE) | `1..16`, `r <= min(K, N)`, matches caller `expected_r` (`ET_LORA_ERR_SHAPE`) |
| `28` | `4` | `alpha` (IEEE-754 f32 LE) | Finite, `> 0.0f`, finite `alpha / r` (`ET_LORA_ERR_NONFINITE` / `ET_LORA_ERR_FORMAT`) |
| `32` | `32` | `base_weight_sha256` | Canonical LE binary32 SHA-256 of `W_base[N, K]` (`ET_LORA_ERR_BASE_HASH`) |
| `64` | `4 * r * K` | `A[r, K]` payload (f32 LE) | Every element finite (`ET_LORA_ERR_NONFINITE`) |
| `64 + 4*r*K` | `4 * N * r` | `B[N, r]` payload (f32 LE) | Every element finite (`ET_LORA_ERR_NONFINITE`) |
| Trailer | `32` | `checkpoint_sha256` | SHA-256 over bytes `[0, size - 32)` (`ET_LORA_ERR_CHECKSUM`) |

- **Atomic write** (`et_lora_checkpoint_save`): serializes into a bounded buffer, writes to a sibling `mkstemp` (`0600`) temporary file, `fsync`s and `close`s the descriptor, and atomically `rename`s onto the target path (unlinking the temporary file on any failure).
- **Checked read** (`et_lora_checkpoint_load`): opens with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC` (rejecting symlinks with `ET_LORA_ERR_SYMLINK`), verifies `S_ISREG`, exact length (`ET_LORA_ERR_TRUNCATED` / `ET_LORA_ERR_FORMAT`), magic, version, SHA-256 checksum, finite payloads, caller `(K, N, r)` match, and base-weight SHA-256 binding before committing outputs.

## Measured verification results

Verified inside the pinned Ubuntu 22.04 container (`eshkol-r0-debug:20260928`,
Clang 21.1.8, pinned Eshkol `fe9dfd52`) via `bash scripts/test-lora.sh`,
`python3 -m unittest tests/q0/test_python_isolation.py`, and
`bash scripts/check-ci-topology.sh`:

| Check | Measured result |
|---|---|
| Core deterministic archive (`libeshkol_transformer_lora.a`) | SHA-256 `4e99c47b6e88230748fca7b548a6d76117146a0d121497ecae5df4677ee9c5ec` |
| Bridge deterministic archive (`libeshkol_transformer_lora_aot.a`) | SHA-256 `b9a73a1a093224b4dffe05a6c957d3e62306f56b2f029be22da6bc4e55b31afa` |
| C++17 header & archive linkage (`clang++-21 -std=c++17`) | PASS (normal + ASan/UBSan/LSan) |
| Zero-init `B` parity with base forward | Bitwise identical (`memcmp == 0`) across 5 shapes up to `[64, 64, 64, r=16]` |
| Central finite-difference gradients (`dX`, `dA`, `dB`) | 211 checks, `max_abs_err = 9.515e-06` |
| Native merge forward parity (`Y_adapter` vs `X @ W_merged^T`) | `max_fwd_diff = 7.451e-08` (`<= 1e-6`) |
| Native unmerge weight restoration (`W_base` vs `W_restored`) | `max_unmerge_diff = 1.490e-08` (`<= 1e-6`; bitwise exact on dyadic weights) |
| Python double-precision oracle (`tests/lora/test_lora_oracle.py`) | 8 cases, 23,570 values, `max_abs_err = 4.818e-06` |
| Sanitizers (`ASan`, `UBSan`, `LSan detect_leaks=1`) | PASS (C11 + C++17 suites) |
| Compiled Eshkol AOT (`tests/lora/test_lora_aot.esk`) | Byte-identical ELF & stdout across 2 fresh builds; ELF SHA-256 `10bbbb47381ec9ba69b8fa49f34e3830ab0ebbd4a04ad68c1d041649a6c46ce5` |
| AOT fresh-process reload output parity | Bitwise identical `Y_SHA256` & `BASE_SHA256`; `max_merge_diff = 2.384e-07`, `max_unmerge_diff = 1.490e-08` |
| Negative & tamper suite (native + AOT) | PASS (truncated, corrupt checksum, bad magic, version mismatch, base-hash mismatch, rank/shape mismatch, non-finite payload/alpha, symlink, misalignment, aliasing, FP-env) |

### Compiled Eshkol AOT 12-step fine-tuning trace

Configuration: `M=8, K=16, N=12, r=4, alpha=8.0, lr=0.25, seed=4242`
(`--strict-types --optimize 0 --no-stdlib`), 9 central finite-difference checks
across `dA` and `dB`, frozen base weight verified after every step:

| Step | MSE loss |
|---:|---:|
| 0 | `1.1559755990407452` |
| 1 | `1.155232491187637` |
| 2 | `1.1544806229159321` |
| 3 | `1.1536969411548244` |
| 4 | `1.1528581782084886` |
| 5 | `1.1519400688340644` |
| 6 | `1.1509165232421334` |
| 7 | `1.1497591353667096` |
| 8 | `1.1484364043129094` |
| 9 | `1.1469133049788942` |
| 10 | `1.1451506402559068` |
| 11 | `1.1431048295017079` |
| 12 | `1.1407276997910758` |

## Limitations and scope

- Bounded CPU serial binary32 implementation for `M, K, N <= 64` and `r <= 16`.
- GPU acceleration, mixed-precision adapters (QLoRA / fp16 / bf16), multi-layer
  adapter registries, and performance benchmarking are outside this scope.
- `tests/lora/test_lora_oracle.py` is a development-only verification oracle; the
  delivered training, checkpointing, merge/unmerge, and AOT execution paths are
  fully Eshkol/C-native.
