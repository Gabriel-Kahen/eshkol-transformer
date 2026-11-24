# GRPO / RLVR Alignment Engine & Compiled Eshkol AOT Driver

Standalone, Eshkol-native **Group Relative Policy Optimization (GRPO)** with **Reinforcement Learning with Verifiable Rewards (RLVR)** implemented in strict C11 with zero heap allocations on all hot paths, paired with a compiled Eshkol AOT bridge and driver.

## Artifacts & Source Files

- Public C11 / C++17 ABI: `include/eshkol_transformer/grpo_abi.h`
- Core C11 engine: `native/grpo.c` -> `build/grpo/libeshkol_transformer_grpo.a`
- Eshkol AOT bridge: `native/grpo_eshkol_bridge.h`, `native/grpo_eshkol_bridge.c` -> `build/grpo/libeshkol_transformer_grpo_aot.a`
- Tests:
  - C++17 ABI & linkage: `tests/grpo/header_grpo_cpp.cpp`
  - Native C11 unit, FD gradient, & negative tests: `tests/grpo/test_grpo_native.c`
  - Independent binary64 Python oracle & f64 FD checker: `tests/grpo/test_grpo_oracle.py`
  - Compiled Eshkol AOT training & verification driver: `tests/grpo/test_grpo_aot.esk`
- Build & verification scripts: `scripts/build-grpo.sh`, `scripts/test-grpo.sh`

## Mathematical Specification

All internal reductions (mean, variance, log-sum-exp, importance ratios, Schulman k3 KL, and analytic gradients) accumulate in IEEE-754 binary64 (`double`) under `FE_TONEAREST` with `#pragma STDC FP_CONTRACT OFF` and round to IEEE-754 binary32 (`float`) on output.

### 1. Group-Relative Advantage Normalization (`et_grpo_group_advantages_v1`)

Given $B$ prompt groups and $G \ge 2$ completions per group ($N = B \cdot G$), with finite scalar verifier rewards $r \in \mathbb{R}^{B \times G}$:

$$\mu_b = \frac{1}{G} \sum_{g=0}^{G-1} r_{b,g}, \qquad \sigma_b^2 = \frac{1}{G} \sum_{g=0}^{G-1} (r_{b,g} - \mu_b)^2 \quad (\text{population variance})$$

- **Standard GRPO (`ET_GRPO_ADV_STANDARD`, requires $\varepsilon > 0$):**
  $$A_{b,g} = \frac{r_{b,g} - \mu_b}{\sqrt{\sigma_b^2} + \varepsilon}$$
  When all rewards in group $b$ are identical, $\sigma_b^2 = 0$ and $A_{b,g} = 0$ exactly.
- **Dr. GRPO (`ET_GRPO_ADV_DR_GRPO`, requires $\varepsilon \ge 0$):**
  $$A_{b,g} = r_{b,g} - \mu_b$$

### 2. Per-Token & Sequence Log-Probabilities (`et_grpo_token_logprobs_v1` / backward)

For logits $z \in \mathbb{R}^{N \times T \times V}$, targets $y \in \{0, \dots, V-1\}^{N \times T}$, and binary response mask $m \in \{0, 1\}^{N \times T}$:

- When $m_{n,t} = 0$: $\ell_{\pi,n,t} = 0$ and $\frac{\partial L}{\partial z_{n,t,v}} = 0$.
- When $m_{n,t} = 1$: with $c_{n,t} = \max_v z_{n,t,v}$,
  $$\mathrm{lse}_{n,t} = c_{n,t} + \log \sum_{v=0}^{V-1} \exp(z_{n,t,v} - c_{n,t}), \qquad \ell_{\pi,n,t} = z_{n,t,y_{n,t}} - \mathrm{lse}_{n,t}$$
  $$S_n = \sum_{t=0}^{T-1} m_{n,t} \ell_{\pi,n,t}$$
- Exact analytic Jacobian pullback given upstream $g_{n,t} = \frac{\partial L}{\partial \ell_{\pi,n,t}}$:
  $$\frac{\partial L}{\partial z_{n,t,v}} = m_{n,t} \cdot g_{n,t} \cdot \left(\mathbf{1}[v = y_{n,t}] - \exp(z_{n,t,v} - \mathrm{lse}_{n,t})\right)$$

### 3. GRPO Clipped Surrogate + Unbiased Schulman k3 KL (`et_grpo_loss_forward_v1` / backward)

Given hyperparams $\epsilon_{\text{low}} \in (0, 1]$, $\epsilon_{\text{high}} > 0$, $\beta \ge 0$, $c_{\text{low}} = 1 - \epsilon_{\text{low}}$, $c_{\text{high}} = 1 + \epsilon_{\text{high}}$:

For each active token $(n, t)$ with $m_{n,t} = 1$, $\Delta_{\text{old}} = \ell_{\pi,n,t} - \ell_{\text{old},n,t}$, and $\Delta_{\text{ref}} = \ell_{\text{ref},n,t} - \ell_{\pi,n,t}$:

- **Explicit Overflow Guard:** If $\Delta_{\text{old}} > 50.0$ or $\Delta_{\text{ref}} > 50.0$, fail closed with `ET_GRPO_ERR_OVERFLOW` before mutating any output.
- **Importance Ratio & Clipped Surrogate:**
  $$\rho = \exp(\Delta_{\text{old}}), \qquad \rho_{\text{clip}} = \mathrm{clip}(\rho, c_{\text{low}}, c_{\text{high}})$$
  $$\mathrm{pg\_obj} = \min(\rho A_n, \rho_{\text{clip}} A_n)$$
  Clipping is active iff $(A_n > 0 \land \rho > c_{\text{high}}) \lor (A_n < 0 \land \rho < c_{\text{low}})$:
  $$\frac{\partial\,\mathrm{pg\_obj}}{\partial \ell_{\pi,n,t}} = \begin{cases} 0 & \text{if clipped}, \\ \rho A_n & \text{otherwise}. \end{cases}$$
- **Schulman k3 Non-Negative KL Estimator:**
  $$\mathrm{kl} = \mathrm{expm1}(\Delta_{\text{ref}}) - \Delta_{\text{ref}} = \exp(\Delta_{\text{ref}}) - \Delta_{\text{ref}} - 1 \ge 0$$
  $$\frac{\partial\,\mathrm{kl}}{\partial \ell_{\pi,n,t}} = 1 - \exp(\Delta_{\text{ref}}) = -\mathrm{expm1}(\Delta_{\text{ref}})$$
- **Reduction Modes** (let $M_n = \sum_t m_{n,t}$, $M_{\text{total}} = \sum_n M_n > 0$, $N_{\text{active}} = |\{n : M_n > 0\}|$):
  - `ET_GRPO_NORM_GLOBAL_TOKEN_MEAN`: $w_{n,t} = 1 / M_{\text{total}}$
  - `ET_GRPO_NORM_PER_SEQ_MEAN`: $w_{n,t} = 1 / (N_{\text{active}} \cdot M_n)$
  In both modes $\sum_{m_{n,t}=1} w_{n,t} = 1$, and:
  $$L = \underbrace{\sum_{m_{n,t}=1} w_{n,t} (-\mathrm{pg\_obj}_{n,t})}_{\text{pg\_loss}} + \beta \underbrace{\sum_{m_{n,t}=1} w_{n,t}\,\mathrm{kl}_{n,t}}_{\text{mean\_kl}}$$
  $$\frac{\partial L}{\partial \ell_{\pi,n,t}} = m_{n,t} \cdot w_{n,t} \cdot \left(-\frac{\partial\,\mathrm{pg\_obj}}{\partial \ell_{\pi,n,t}} + \beta \cdot (-\mathrm{expm1}(\Delta_{\text{ref}}))\right)$$
  The fused backward chains $\frac{\partial L}{\partial \ell_{\pi,n,t}}$ directly through the log-softmax Jacobian to `out_grad_logits[N, T, V]`.
  `clip_fraction` is reported as `clipped_tokens / active_tokens`.

### 4. Deterministic RLVR Verifiers

- `et_grpo_verify_exact_match_v1`: binary-safe byte comparison with optional ASCII whitespace trimming (`ET_GRPO_VERIFY_TRIM_WS`), returning `1.0f` on exact match and `0.0f` otherwise.
- `et_grpo_verify_format_and_answer_v1`: strict non-nested `<think>...</think><answer>...</answer>` format verification (rejecting empty reasoning/answer spans, nested tags, and trailing garbage) combined with exact answer extraction and configurable `(format_weight, answer_weight)`.

## Tensor Shape & Safety Contracts

| Tensor / Buffer | Shape | Dtype / Alignment | Validation |
|---|---|---|---|
| `rewards` | `[B, G]` or `[N]` | `float` (4-byte) | Finite; `1 <= B <= 256`, `2 <= G <= 64`, `N <= 1024` |
| `advantages` | `[B, G]` or `[N]` | `float` (4-byte) | Finite; pairwise disjoint from outputs |
| `pi_logits`, `grad_logits` | `[N, T, V]` | `float` (4-byte) | Finite; `T <= 256`, `V <= 4096`, `N*T*V <= 4194304` |
| `targets` | `[N, T]` | `int32_t` (4-byte) | `0 <= targets[n,t] < V` for all `(n,t)` |
| `mask` | `[N, T]` | `uint8_t` (1-byte) | Strictly in `{0, 1}`; loss ops require `active_tokens >= 1` |
| `log_pi`, `log_old`, `log_ref` | `[N, T]` | `float` (4-byte) | Finite; read-only inputs may alias (`log_old == log_ref`) |

- **Zero heap allocation in core engine:** `nm -u` on `grpo.o` is audited in `scripts/build-grpo.sh` and permits only math/memory/FP primitives (`bcmp`, `exp`, `expm1`, `fegetround`, `log`, `memcmp`, `memcpy`, `memset`, `sqrt`) — zero `malloc`/`calloc`/`realloc`/`free`, zero I/O, zero threads.
- **Byte-for-byte output preservation on error:** All core functions execute a full validation pass before writing any output buffer; negative tests verify `0xA5` sentinel preservation across every failure path.

## Measured Verification Evidence

Verified inside container `eshkol-r0-debug:20260928` (`CC=clang-21`, `CXX=clang++-21`, pinned Eshkol toolchain `/tc/build`):

- **Deterministic static archives:**
  - `libeshkol_transformer_grpo.a` SHA-256: `1d219f955d3e23943f581267e0048876e0e2d0a363ed1b525577526e3fe8a420`
  - `libeshkol_transformer_grpo_aot.a` SHA-256: `25b983dd27ea95fce43dbcfdf5137f3ee36e3b3a37bb84647f79e14a1cbabe58`
- **Numerical gradient checks (central finite differences, unclipped & clipped regimes):**
  - C11 binary32 central FD (`h = 1/256`):
    - `et_grpo_token_logprobs_backward_v1`: `max_rel_err = 9.478e-05`
    - `et_grpo_logprob_loss_backward_v1`: `max_rel_err = 6.170e-05`
    - `et_grpo_loss_backward_v1`: `max_rel_err = 1.040e-04`
  - Python binary64 reference oracle + f64 central FD (`h = 1e-6`, 8 cases, 1,342 comparisons):
    - Oracle value parity: `max_abs_err = 2.533e-06`
    - F64 FD vs native `grad_logits`: `max_abs_err = 6.353e-09`, `max_rel_err = 2.785e-07`
- **Sanitizers:** C11 + C++17 suites pass under `-fsanitize=address,undefined` with `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and `UBSAN_OPTIONS=halt_on_error=1`.
- **Compiled Eshkol AOT driver (`tests/grpo/test_grpo_aot.esk`, `--strict-types --optimize 0 --no-stdlib`):**
  - Dual fresh compilations produce byte-identical ELFs and byte-identical stdout.
  - AOT ELF SHA-256: `67993957b9a8bf3d7a5354bad29d0047a6bf4408fdee23be51f83e08646860ce`
  - Multi-step alignment loop (10 outer steps × 3 inner off-policy epochs, $B=2, G=4, T=6, V=8$):
    - Policy expected RLVR verifier reward strictly increases every step: `0.286991 -> 0.922678`
    - Final greedy-decoded verifier reward: `1.0` (100% format + exact answer)
    - Schulman k3 KL remains non-negative and bounded: `2.04e-15 -> 0.489625` (`max_kl = 0.513691`)
    - Both unclipped (`epoch 0` on all 10 steps) and clipped (`total_clipped = 34` tokens) regimes verified
    - 8 in-Eshkol FD gradient checks + 9 fail-closed negative checks PASS

## Measured Limitations & Unsupported Cases

- **CPU-only binary32/binary64 execution:** No GPU / HIP / CUDA offload is claimed or implemented.
- **Fixed compile-time dimension bounds:** `ET_GRPO_MAX_GROUPS = 256`, `ET_GRPO_MAX_GROUP_SIZE = 64`, `ET_GRPO_MAX_BATCH = 1024`, `ET_GRPO_MAX_SEQ_LEN = 256`, `ET_GRPO_MAX_VOCAB = 4096`, `ET_GRPO_MAX_LOGITS_ELEMS = 4194304`, `ET_GRPO_MAX_VERIFY_BYTES = 16384`. Larger shapes fail closed with `ET_GRPO_ERR_SHAPE`.
- **Log-ratio overflow cap:** Log-probability differences exceeding `ET_GRPO_MAX_LOG_DIFF = 50.0f` fail closed with `ET_GRPO_ERR_OVERFLOW` rather than saturating or producing non-finite ratios.
- **Non-differentiable clip kink points:** At exact boundaries $\rho = 1 - \epsilon_{\text{low}}$ ($A < 0$) and $\rho = 1 + \epsilon_{\text{high}}$ ($A > 0$), the unclipped subgradient ($\rho A$) is selected consistently with `clipped` requiring strict inequality (`rho > c_high` or `rho < c_low`).
- **AOT bridge context allocation:** While `native/grpo.c` performs zero heap allocations, `et_grpo_aot_create` / `et_grpo_aot_destroy` in `native/grpo_eshkol_bridge.c` allocate/free the single opaque AOT driver context once at setup/teardown.
- **Standalone integration:** Not yet wired into the root `Makefile` or `docs/ROADMAP.md` (disjoint branch deliverable).
