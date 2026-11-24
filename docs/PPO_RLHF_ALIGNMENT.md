# RLHF PPO-Clip + GAE + Value Critic Alignment Engine & Eshkol AOT Driver

Standalone, Eshkol-native **Proximal Policy Optimization (PPO-Clip)** alignment engine with **KL-shaped rewards**, **Generalized Advantage Estimation (GAE)**, **exact masked advantage whitening**, **clipped value critic + entropy regularization**, **multi-epoch Actor-Critic training with `target_kl` early stopping**, **frozen reference policy SHA-256 verification**, and **versioned SHA-256 checksummed `ETPP1000` v1.0 checkpoints**, paired with a compiled Eshkol AOT bridge and driver.

## Artifacts & Source Files

- Public C11 / C++17 ABI: `include/eshkol_transformer/ppo_rlhf_abi.h`
- Core C11 engine (zero heap allocation): `native/ppo_rlhf.c` -> `build/ppo_rlhf/libeshkol_transformer_ppo_rlhf.a`
- Eshkol AOT bridge: `native/ppo_rlhf_eshkol_bridge.h`, `native/ppo_rlhf_eshkol_bridge.c` -> `build/ppo_rlhf/libeshkol_transformer_ppo_rlhf_aot.a`
- Tests:
  - C++17 ABI & linkage: `tests/ppo_rlhf/header_ppo_cpp.cpp`
  - Native C11 unit, FD gradient, fresh-process `ETPP1000` resume & negative tests: `tests/ppo_rlhf/test_ppo_rlhf_native.c`
  - Independent binary64 Python reference oracle, f64 FD checker & multi-iteration trajectory verifier: `tests/ppo_rlhf/test_ppo_rlhf_oracle.py`
  - Compiled Eshkol AOT alignment & fresh-process resume driver: `tests/ppo_rlhf/test_ppo_rlhf_aot.esk`
- Build & verification scripts: `scripts/build-ppo-rlhf.sh`, `scripts/test-ppo-rlhf.sh`

## Mathematical Specification

All internal reductions, log-softmax, GAE recurrences, whitening moments, clipped surrogates, analytic gradients, and AdamW updates accumulate in IEEE-754 `binary64` (`double`) under `FE_TONEAREST` with `#pragma STDC FENV_ACCESS ON`, `#pragma STDC FP_CONTRACT OFF`, and runtime x87/MXCSR FTZ/DAZ/rounding checks. Results round to IEEE-754 `binary32` (`float`) on output.

### 1. Per-Token KL-Shaped Rewards (`et_ppo_kl_shaped_rewards_v1`)

For environment rewards $r_{\text{env}}[b, t]$, rollout log-probabilities $\ell_{\text{old}}[b, t]$, frozen reference log-probabilities $\ell_{\text{ref}}[b, t]$, binary mask $m[b, t] \in \{0, 1\}$, and $\beta_{\text{kl}} \ge 0$:

$$r_{\text{shaped}}[b, t] = \begin{cases} r_{\text{env}}[b, t] - \beta_{\text{kl}} \cdot (\ell_{\text{old}}[b, t] - \ell_{\text{ref}}[b, t]) & m[b, t] = 1, \\ 0 & m[b, t] = 0. \end{cases}$$

### 2. Generalized Advantage Estimation (`et_ppo_gae_v1`) & Exact Whitening (`et_ppo_whiten_advantages_v1`)

For each sequence $b \in [0, B)$, reverse-time recurrence $t = T - 1, \dots, 0$ initialized with $v_{\text{next}} = 0, A_{\text{next}} = 0$ ($\gamma, \lambda \in [0, 1]$, active count $M \ge 1$):

- If $m[b, t] = 0$: $A[b, t] = 0$, $R[b, t] = 0$, and reset $v_{\text{next}} = 0, A_{\text{next}} = 0$ (strictly isolating prompt/padding boundaries and terminal states).
- If $m[b, t] = 1$:
  $$\delta_t = r_{\text{shaped}}[b, t] + \gamma v_{\text{next}} - v[b, t], \qquad A[b, t] = \delta_t + \gamma \lambda A_{\text{next}}, \qquad R[b, t] = A[b, t] + v[b, t],$$
  followed by $v_{\text{next}} \leftarrow v[b, t]$, $A_{\text{next}} \leftarrow A[b, t]$.

Masked advantage whitening over active tokens ($m[b, t] = 1$, count $M \ge 1$, $\varepsilon > 0$) uses **direct binary64 division** (avoiding reciprocal double-rounding) and population variance:

$$\mu = \frac{\sum_{m=1} A_i}{M}, \qquad \sigma^2 = \frac{\sum_{m=1} (A_i - \mu)^2}{M}, \qquad \hat{A}_i = \frac{A_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} \quad (0 \text{ when } m_i = 0).$$

Because every `float` has a 24-bit significand and $M \le 65536 = 2^{16}$, $M \cdot c$ is exact in `binary64` ($24 + 16 = 40 \le 53$), so all-equal advantages yield exact $+0.0\text{f}$ (`0x00000000`) for every count $M \in [1, 65536]$.

### 3. PPO-Clip Policy + Clipped Value Critic + Entropy Forward & Analytic Backward (`et_ppo_loss_forward_v1`, `et_ppo_loss_backward_v1`)

For each active token $i = (b, t)$ with logits $z_i \in \mathbb{R}^V$, action $a_i \in \{0, \dots, V-1\}$, value $v_i$, old logprob $\ell_{\text{old}, i}$, old value $v_{\text{old}, i}$, advantage $A_i$, and return $R_i$:

- **Log-softmax & Entropy:** With $c_i = \max_v z_{i, v}$, $\mathrm{lse}_i = c_i + \log \sum_v \exp(z_{i, v} - c_i)$, $\log p_{i, v} = z_{i, v} - \mathrm{lse}_i$, $p_{i, v} = \exp(\log p_{i, v})$:
  $$\ell_{\pi, i} = \log p_{i, a_i}, \qquad H_i = -\sum_{v=0}^{V-1} p_{i, v} \log p_{i, v}.$$
- **Overflow Guard:** $\Delta_{\ell, i} = \ell_{\pi, i} - \ell_{\text{old}, i} > 50.0$ fails closed with `ET_PPO_ERR_OVERFLOW`.
- **PPO-Clip Policy Surrogate (strict-inequality tie convention):**
  With $\rho_i = \exp(\Delta_{\ell, i})$, $c_{\text{low}} = 1 - \epsilon_{\text{low}}$, $c_{\text{high}} = 1 + \epsilon_{\text{high}}$:
  $$\text{clipped}_i \iff (A_i > 0 \land \rho_i > c_{\text{high}}) \lor (A_i < 0 \land \rho_i < c_{\text{low}}),$$
  $$\mathrm{pg}_i = \begin{cases} \mathrm{clip}(\rho_i, c_{\text{low}}, c_{\text{high}}) A_i & \text{clipped}_i, \\ \rho_i A_i & \text{otherwise}, \end{cases} \qquad \frac{\partial\,\mathrm{pg}_i}{\partial \ell_{\pi, i}} = \begin{cases} 0 & \text{clipped}_i, \\ \rho_i A_i & \text{otherwise}. \end{cases}$$
  At exact kink ties ($\rho_i = c_{\text{high}}$ with $A_i > 0$, or $\rho_i = c_{\text{low}}$ with $A_i < 0$), the unclipped branch and subgradient $\rho_i A_i$ are selected consistently.
- **Clipped Value Critic Loss (strict-inequality & clamped-region convention):**
  Let $u_i = (v_i - R_i)^2$. When `use_value_clip == 1` and $|v_i - v_{\text{old}, i}| > \epsilon_v$, let $v_{\text{clip}, i} = v_{\text{old}, i} + \mathrm{sign}(v_i - v_{\text{old}, i}) \epsilon_v$ and $c_i = (v_{\text{clip}, i} - R_i)^2$:
  $$\text{v\_clipped}_i \iff (\text{use\_value\_clip} = 1) \land (|v_i - v_{\text{old}, i}| > \epsilon_v) \land (c_i > u_i),$$
  $$L_{v, i} = \tfrac{1}{2} \max(u_i, c_i), \qquad \frac{\partial L_{v, i}}{\partial v_i} = \begin{cases} 0 & \text{v\_clipped}_i, \\ v_i - R_i & \text{otherwise}. \end{cases}$$
  Evaluating clipping only when $|v_i - v_{\text{old}, i}| > \epsilon_v$ prevents spurious 1-ULP value clipping from $v_{\text{old}} + (v - v_{\text{old}})$ inside the unclipped trust box.
- **Total Loss, Diagnostics & Exact Analytic Pullbacks:**
  $$L_{\text{policy}} = -\frac{1}{M}\sum_{m=1} \mathrm{pg}_i, \qquad L_{\text{value}} = \frac{1}{M}\sum_{m=1} L_{v, i}, \qquad \bar{H} = \frac{1}{M}\sum_{m=1} H_i,$$
  $$L_{\text{total}} = L_{\text{policy}} + c_{\text{vf}} L_{\text{value}} - c_{\text{ent}} \bar{H},$$
  $$\mathrm{approx\_kl} = \frac{1}{M}\sum_{m=1} \frac{1}{2} (\ell_{\pi, i} - \ell_{\text{old}, i})^2, \qquad \mathrm{EV} = \begin{cases} 0 & \mathrm{Var}(R) = 0, \\ 1 - \frac{\mathrm{Var}(R - v)}{\mathrm{Var}(R)} & \mathrm{Var}(R) > 0, \end{cases}$$
  $$\frac{\partial L_{\text{total}}}{\partial z_{i, v}} = \frac{m_i}{M} \left[ -\frac{\partial\,\mathrm{pg}_i}{\partial \ell_{\pi, i}} (\mathbf{1}[v = a_i] - p_{i, v}) + c_{\text{ent}}\, p_{i, v} (\log p_{i, v} + H_i) \right], \qquad \frac{\partial L_{\text{total}}}{\partial v_i} = \frac{m_i}{M} c_{\text{vf}} \frac{\partial L_{v, i}}{\partial v_i}.$$

### 4. Multi-Epoch Actor-Critic Trainer, KL Early Stopping & `ETPP1000` v1.0 Checkpoints

- **Model layout:** Trainable parameter vector `params[P]` (`P = V*D + V + D + 1`: `W_pi[V, D]`, `b_pi[V]`, `W_v[D]`, `b_v[1]`) and strictly read-only frozen reference policy `ref_params[V*D + V]` (`W_ref[V, D]`, `b_ref[V]`), with canonical little-endian IEEE-754 float32 SHA-256 verification (`et_ppo_weight_sha256_v1`).
- **Rollout update (`et_ppo_train_rollout_v1`):** Snapshots `old_logprobs`, `old_values`, and `ref_logprobs`, computes KL-shaped rewards, GAE, optional whitening, and runs up to `ppo_epochs` updates using decoupled AdamW (`et_ppo_adamw_step_v1`, binary64 with optional global L2 gradient-norm clipping and exact integer-power bias correction). If `target_kl > 0` and an epoch's `approx_kl > target_kl`, the epoch aborts before stepping (`early_stopped = 1`, `stopped_epoch = epoch`). All mutations are staged so any error leaves caller state unchanged.
- **Checkpoint (`ETPP1000` v1.0):** 184-byte little-endian header (`"ETPP1000"`, major/minor `1.0`, `feat_dim`, `vocab_size`, `param_count`, `flags=0`, `step`, full `et_ppo_trainer_config`, 32-byte `ref_sha256`, `payload_bytes`) + `3 * P * 4` bytes (`params`, `adam_m`, `adam_v`, all finite, `adam_v >= 0`) + 32-byte SHA-256 trailer over header + payload.
  - `et_ppo_checkpoint_save_v1`: atomic same-directory temp file (`O_CREAT | O_EXCL | O_NOFOLLOW`, `0600`), `fsync(fd)`, `rename`, parent directory `fsync`.
  - `et_ppo_checkpoint_load_v1`: `open(..., O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)`, `S_ISREG` check, exact size & SHA-256 verification, frozen `ref_sha256` match, staged commit on success only.

## Measured Verification Evidence

Verified inside pinned container `eshkol-r0-debug:20260928` (`--rm --network none`, `CC=clang-21`, `CXX=clang++-21`, `/tc/build`):

- **Deterministic static archives & zero-heap core audit:**
  - `build/ppo_rlhf/libeshkol_transformer_ppo_rlhf.a` SHA-256: `9664d03db661774a132bbcd9805f6c78c3e3fad13efda65641b82c912040797a` (`nm -u ppo_rlhf.o` audited: zero `malloc`/`calloc`/`realloc`/`free`)
  - `build/ppo_rlhf/libeshkol_transformer_ppo_rlhf_aot.a` SHA-256: `c1c8224fc0106132adad1e416ff71dff17ac03982224f6a0d3a3c596de8aa265`
- **C++17 standard-layout ABI & linkage:** PASS
- **Native C11 central FD gradient checks & whitening:**
  - All-equal advantage whitening across `N = 2..64` (6 tricky binary32 constants): exact `+0.0f` (`0x00000000`)
  - `use_value_clip=0`: `max_d_logits_err = 1.583e-06`, `max_d_values_err = 2.570e-07`
  - `use_value_clip=1`: `max_d_logits_err = 8.837e-07`, `max_d_values_err = 1.434e-07`
  - Negative & `0xA5` sentinel preservation tests (including `FE_TOWARDZERO` and MXCSR FTZ): PASS
- **Independent binary64 Python reference oracle + f64 FD + multi-iteration trajectory:**
  - 6 cases, 442 values: `max_abs_err = 2.352e-07`
  - Independent f64 central FD (`h = 1e-6`): `max_f64_fd_logits = 9.038e-09`, `max_f64_fd_values = 5.996e-09`
  - 3-iteration × 3-epoch Actor-Critic PPO trajectory parity: `traj_max_param_err = 0.000e+00` (exact bit-for-bit match)
- **Sanitizers:** C11 & C++17 suites PASS under `-fsanitize=address,undefined` with `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.
- **Compiled Eshkol AOT driver (`tests/ppo_rlhf/test_ppo_rlhf_aot.esk`, `--strict-types --optimize 0 --no-stdlib`):**
  - Dual fresh compilations produce byte-identical ELFs and byte-identical combined train+reload stdout.
  - AOT ELF SHA-256: `fcbacd38c7062f5f8d034e50ea6fe089bfa8ba2227c963b883e62eb7ceb3abeb`
  - In-Eshkol central FD (30 checks): `max_logit_err = 1.572e-06`, `max_value_err = 9.313e-08` (`< 1e-5`)
  - 16 PPO alignment iterations (`B=16, T=4, D=8, V=6`, 4 epochs/iter, 64 AdamW steps): expected environment reward strictly increases every iteration `-0.134908 -> +0.420646` while frozen reference SHA-256 remains invariant (`ref_frozen=PASS`)
  - Tight `target_kl = 1e-4` early stopping: `triggered=1`, `epochs=1`, `approx_kl=0.009264`
  - Fresh OS process `ETPP1000` checkpoint reload at iteration 8 & resume through iteration 16: exact bit-for-bit parity (`final_reward = 0.42064561951729607`, `final_step = 64`, `RELOAD_REF_SHA256 == TRAIN_REF_SHA256`, `RELOAD_PARAM_SHA256 == TRAIN_PARAM_SHA256`)
  - 17 fail-closed negative checks (tamper, truncation, symlink, ref mismatch, NaN/Inf, invalid `gamma`/`lam`/`eps`/`clip`, empty mask, OOB actions, FP-env trap, overflow, aliasing): PASS
- **Repository policy guards:** `tests/q0/test_python_isolation.py` (5/5 OK) and `scripts/check-ci-topology.sh` (PASS).

## Measured Limitations & Unsupported Cases

- **CPU-only binary32/binary64 execution:** No GPU / HIP / CUDA execution is claimed or implemented.
- **Bounded compile-time dimensions:** Core loss ops enforce `B <= 256`, `T <= 256`, `V <= 4096`, `B*T*V <= 4194304`; the built-in allocation-free Actor-Critic trainer & `ETPP1000` v1.0 format enforce `feat_dim <= 64`, `vocab_size <= 64`, `B*T <= 1024`, `ppo_epochs <= 64`, and checkpoint size `<= 65536` bytes. Larger inputs fail closed with `ET_PPO_ERR_SHAPE` / `ET_PPO_ERR_FORMAT`.
- **Log-ratio overflow cap:** `logpi_new - old_logprob > 50.0f` fails closed with `ET_PPO_ERR_OVERFLOW`.
- **AOT bridge allocation:** `native/ppo_rlhf.c` performs zero heap allocations; `et_ppo_aot_create` / `et_ppo_aot_destroy` in `native/ppo_rlhf_eshkol_bridge.c` allocate/free the opaque driver context once at setup/teardown.
- **Standalone branch deliverable:** Root `Makefile` and `docs/ROADMAP.md` are intentionally untouched per strict file-scope isolation.
