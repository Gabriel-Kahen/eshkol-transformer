# O2 Muon (Newton-Schulz5) + Hybrid AdamW + WSD LR Scheduler

## Scope and Dependency Boundary

This module delivers a standalone, deterministic, failure-atomic CPU binary32/binary64
native implementation and Eshkol AOT bridge for:

1. **Newton-Schulz 5-step quintic matrix orthogonalization** (`et_muon_newton_schulz5_f32` / `_raw`),
2. **Hybrid Muon (2D matrix) + AdamW (1D vector/norm/embedding) parameter-group optimization**
   with explicit per-group `apply_weight_decay` masking (`et_muon_hybrid_step` / `et_muon_hybrid_step_wsd`), and
3. **WSD (Warmup-Stable-Decay) learning-rate scheduling** (`et_wsd_lr`, `et_wsd_validate_config`).

Public C11/C++17 ABI: [o2_muon_abi.h](../include/eshkol_transformer/o2_muon_abi.h)  
Core native implementation: [o2_muon_optimizer.c](../native/o2_muon_optimizer.c)  
Eshkol AOT bridge: [o2_muon_eshkol_bridge.h](../native/o2_muon_eshkol_bridge.h), [o2_muon_eshkol_bridge.c](../native/o2_muon_eshkol_bridge.c)  
Build & test gates: [build-o2-muon.sh](../scripts/build-o2-muon.sh), [test-o2-muon.sh](../scripts/test-o2-muon.sh)

No existing O2 files, `Makefile`, or `docs/ROADMAP.md` are modified.

## Numerical Environment and Safety Contract

- Compiled with `-std=c11 -Wall -Wextra -Werror -Wpedantic -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math -fstack-protector-all -fPIC` (`#pragma STDC FENV_ACCESS ON`, `#pragma STDC FP_CONTRACT OFF`).
- Every public entry point verifies IEEE-754 round-to-nearest-even and, on `x86_64`, default x87 control word (`(x87 & 0x0c3f) == 0x003f`) and MXCSR (`(mxcsr & 0xffc0) == 0x1f80`, masked exceptions, FTZ/DAZ disabled). Any other FP environment returns `ET_MUON_ERR_FP_ENV` before mutation.
- All `_raw` entry points and `void *` group fields validate alignment via `uintptr_t` before casting to `float *` (zero misaligned-pointer UB).
- All buffer spans within an operation—including across all parameter groups in a hybrid step—must be pairwise disjoint; overlapping buffers return `ET_MUON_ERR_ALIAS`.
- Every input, hyperparameter, intermediate, and narrowed binary32 output is checked for finiteness (`ET_MUON_ERR_NONFINITE`). No CPU/scalar/approximate/precision fallback is hidden.
- All mutating calls are **failure-atomic**: updated parameters and optimizer state buffers for every group are staged in temporary storage and verified finite before any `memcpy` commits to caller buffers. On any error, every caller buffer is preserved bit-for-bit.

## Newton-Schulz Quintic Orthogonalization

Given a dense row-major binary32 matrix $G \in \mathbb{R}^{m \times n}$ with $m, n \in [1, 64]$,
`ns_steps` $\in [1, 16]$ (default `5`), and finite $\varepsilon > 0$:

1. **Binary64 Frobenius normalization**:
   $$\|G\|_F = \sqrt{\sum_{i,j} G_{i,j}^2}, \qquad d = \|G\|_F + \varepsilon$$
   All-zero matrices ($G = 0$) normalize cleanly to $X_0 = 0$ and output exact finite $U = 0$.
2. **Tall-vs-wide transpose**:
   Let $r = \min(m, n)$, $c = \max(m, n)$. If $m > n$, transpose into $X_0 = G^\top / d \in \mathbb{R}^{r \times c}$; otherwise $X_0 = G / d \in \mathbb{R}^{r \times c}$.
3. **Quintic iteration** ($k = 0, \dots, \text{ns\_steps} - 1$) in binary64 with $(a, b, c) = (3.4445, -4.7750, 2.0315)$:
   $$A_k = X_k X_k^\top \in \mathbb{R}^{r \times r}, \qquad B_k = b A_k + c A_k^2, \qquad X_{k+1} = a X_k + B_k X_k$$
4. **Transpose back (if $m > n$) and narrow to finite binary32** into disjoint output $U \in \mathbb{R}^{m \times n}$.
5. **Orthogonality error diagnostic** (`et_muon_orthogonality_error_f32`):
   $$e(U) = \begin{cases} \|U U^\top - I_m\|_F & \text{if } m \le n \\ \|U^\top U - I_n\|_F & \text{if } m > n \end{cases}$$

## Hybrid Muon + AdamW Parameter-Group Contract

`et_muon_hybrid_step` accepts `1..64` heterogeneous parameter groups:

- **2D Matrix Groups (`ET_MUON_GROUP_MUON_2D`)**:
  - Shape: `rows, cols` $\in [1, 64]$, `variance_buf == NULL`, `apply_weight_decay` $\in \{0, 1\}$.
  - Momentum buffer update: $B \leftarrow \beta B + G$ ($\beta \in [0, 1)$).
  - Orthogonalization input: $M = G + \beta B$ if `muon_nesterov == 1`, else $M = B$.
  - $U = \mathrm{NS}_k(M, \varepsilon)$ via the binary64 Newton-Schulz core.
  - Aspect-ratio scaling: $s = \sqrt{\max(1, \text{rows} / \text{cols})}$.
  - Parameter update:
    $$W \leftarrow \begin{cases} W - \eta \cdot (\lambda W + s U) & \text{if } \text{apply\_weight\_decay} = 1 \\ W - \eta \cdot (s U) & \text{if } \text{apply\_weight\_decay} = 0 \end{cases}$$

- **1D Vector / Norm / Embedding Groups (`ET_MUON_GROUP_ADAMW_1D`)**:
  - Shape: `rows` $\in [1, 4096]$, `cols == 1`, `variance_buf != NULL` (non-negative finite), `apply_weight_decay` $\in \{0, 1\}$.
  - Effective LR: $\eta_{\text{adamw}} = \eta \cdot \text{adamw\_lr\_scale}$.
  - Bias-corrected deterministic AdamW at 1-based `step_index` $t \ge 1$ using binary64 exponentiation by squaring for $\beta_1^t, \beta_2^t$:
    $$m \leftarrow \beta_1 m + (1 - \beta_1) g, \qquad v \leftarrow \beta_2 v + (1 - \beta_2) g^2$$
    $$\hat{m} = \frac{m}{1 - \beta_1^t}, \qquad \hat{v} = \frac{v}{1 - \beta_2^t}$$
    $$W_{\text{base}} = \begin{cases} W \cdot (1 - \eta_{\text{adamw}} \lambda_{\text{adamw}}) & \text{if } \text{apply\_weight\_decay} = 1 \\ W & \text{if } \text{apply\_weight\_decay} = 0 \end{cases}$$
    $$W \leftarrow W_{\text{base}} - \eta_{\text{adamw}} \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon_{\text{adamw}}}$$
  - When `apply_weight_decay == 0`, decoupled weight decay is strictly omitted; on zero gradients with zero moments, norm/embedding weights remain bit-identical even if `adamw_weight_decay > 0`.

## WSD (Warmup-Stable-Decay) LR Schedule

Configuration (`et_wsd_config`) requires `total_steps >= 1`, `warmup_steps + stable_steps <= total_steps` (checked without `uint64_t` overflow), `0 <= min_lr <= max_lr`, `max_lr > 0`, `decay_kind` $\in \{\text{LINEAR}, \text{COSINE}\}$, and `reserved == 0`.

For 0-based `step` with `stable_end = warmup_steps + stable_steps` and `decay_steps = total_steps - stable_end`:

- **Warmup** `[0, warmup_steps)`: `phase = 0` (`ET_WSD_PHASE_WARMUP`), $\text{lr} = \text{max\_lr} \cdot \frac{\text{step} + 1}{\text{warmup\_steps}}$.
- **Stable** `[warmup_steps, stable_end)`: `phase = 1` (`ET_WSD_PHASE_STABLE`), $\text{lr} = \text{max\_lr}$.
- **Decay** `[stable_end, total_steps]` (clamped at `min_lr` for `step >= total_steps` or `decay_steps == 0`): `phase = 2` (`ET_WSD_PHASE_DECAY`), exact `max_lr` at `step == stable_end`, exact `min_lr` at `step >= total_steps`, and for interior $t = \frac{\text{step} - \text{stable\_end}}{\text{decay\_steps}} \in (0, 1)$:
  - Linear: $\text{max\_lr} - (\text{max\_lr} - \text{min\_lr}) t$
  - Cosine: $\text{min\_lr} + \frac{1}{2}(\text{max\_lr} - \text{min\_lr})(1 + \cos(\pi t))$

## Measured Verification Evidence

Verified inside pinned container `eshkol-r0-debug:20260928` (`--network none`, `clang-21`/`clang++-21`, `eshkol-run` pinned at `/tc/build`):

- **Deterministic archives & symbol audit**: PASS
  - `libeshkol_transformer_o2_muon.a` SHA-256: `071843e3119641ed76ab526d94498151c1afbad9435fd450a487d7794fa57959`
  - `libeshkol_transformer_o2_muon_aot.a` SHA-256: `9f09f105df1ef18f893ba0c9d5cfd0a6b6ec9c2fc089b92c40ac34a4cd53ff07`
- **C++17 header & linkage**: PASS (`clang++-21 -std=c++17`)
- **Native C11 normal (`-O2`) + ASan/UBSan/LSan**: PASS (zero leaks, zero UB)
- **Independent Python binary64 oracle** (`tests/o2_muon/test_o2_muon_oracle.py`, 1,410 scalar comparisons across 8 NS5 cases, 5 WSD schedules, and 2 multi-step hybrid Muon+AdamW trajectories):
  - `max_ns_err = 3.297e-08` (threshold `<= 1e-5`)
  - `max_hybrid_err = 1.469e-07` (threshold `<= 1e-5`)
  - `max_wsd_err = 0.000e+00` (threshold `<= 1e-12`)
- **Compiled Eshkol AOT training** (`tests/o2_muon/test_o2_muon_aot.esk`, `--strict-types --optimize 0 --no-stdlib`, 2-layer MLP + RMSNorm, `B=16, D=8, H=12`, wide `W1 [8,12]` + tall `W2 [12,8]` Muon `wd=0.01`, `gamma1 [8]` + `gamma2 [12]` AdamW `wd=0.0`, 6 central finite-difference gradient checks, negative/mask suite):
  - Two fresh compilations produced byte-identical ELFs and identical stdout.
  - **AOT ELF SHA-256**: `7d95d4473193b77a46974a89b32b9c28510cd462c65c87f9a3ba57f7713c6836`
  - **Max orthogonality error across 15 steps**: `1.7822233969142451` (strictly `< sqrt(8) = 2.8284`)

### 15-Step Compiled Eshkol AOT Loss & WSD Trace

Initial loss (step 0 before first update): `1.0113014369061777`

| Step | WSD Phase | Effective LR | `ortho(dW1)` `[8,12]` | `ortho(dW2)` `[12,8]` | Post-Step Loss |
|-----:|-----------|-------------:|----------------------:|----------------------:|---------------:|
| 0 | Warmup (0) | 0.0062500000 | 1.234217 | 1.782223 | 0.9723369313 |
| 1 | Warmup (0) | 0.0125000000 | 1.247170 | 1.774517 | 0.8976160922 |
| 2 | Warmup (0) | 0.0187500000 | 1.276788 | 1.764756 | 0.7935155865 |
| 3 | Warmup (0) | 0.0250000000 | 1.333813 | 1.739480 | 0.6691638135 |
| 4 | Stable (1) | 0.0250000000 | 1.417389 | 1.554017 | 0.5603878956 |
| 5 | Stable (1) | 0.0250000000 | 1.375762 | 1.387313 | 0.4658163625 |
| 6 | Stable (1) | 0.0250000000 | 1.274031 | 1.298259 | 0.3840601140 |
| 7 | Stable (1) | 0.0250000000 | 1.324900 | 1.234626 | 0.3140749328 |
| 8 | Stable (1) | 0.0250000000 | 1.463114 | 1.227551 | 0.2552189236 |
| 9 | Decay (2) | 0.0250000000 | 1.335676 | 1.276193 | 0.2067921102 |
| 10 | Decay (2) | 0.0228514412 | 1.449200 | 1.377441 | 0.1708024408 |
| 11 | Decay (2) | 0.0172264412 | 1.361039 | 1.266327 | 0.1484322084 |
| 12 | Decay (2) | 0.0102735588 | 1.470454 | 1.414285 | 0.1367813427 |
| 13 | Decay (2) | 0.0046485588 | 1.429991 | 1.502515 | 0.1318441647 |
| 14 | Decay (2) | 0.0025000000 | 1.439395 | 1.537873 | 0.1292607859 |

## Explicit Limitations

- Dense, contiguous, row-major CPU binary32 storage only (no GPU, mixed-precision fp16/bf16, sparse gradients, or multi-threaded execution claims).
- Dimension bounds are fixed at `rows, cols in [1, 64]` for 2D Muon matrices, `length in [1, 4096]` for 1D AdamW groups, `num_groups in [1, 64]`, and `ns_steps in [1, 16]`.
- Exact bitwise repeatability assumes IEEE-754 binary32/binary64 round-to-nearest-even with `-ffp-contract=off` and default x87/MXCSR state.
