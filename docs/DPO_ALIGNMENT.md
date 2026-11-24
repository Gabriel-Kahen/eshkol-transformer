# Direct Preference Optimization (DPO) Preference-Alignment Engine

This document specifies the Eshkol-native Direct Preference Optimization (DPO),
conservative DPO (cDPO), and reference-free preference-alignment loss, exact
analytic backward, reward-margin telemetry, and compiled Eshkol AOT
post-training driver.

## Artifacts and public headers

- Public C11/C++17 ABI: `include/eshkol_transformer/dpo_alignment_abi.h`
- Core implementation: `native/dpo_alignment.c` -> `build/dpo/libeshkol_transformer_dpo.a`
- Eshkol AOT bridge: `native/dpo_eshkol_bridge.h`, `native/dpo_eshkol_bridge.c` -> `build/dpo/libeshkol_transformer_dpo_aot.a`
- Native & C++17 tests: `tests/dpo/test_dpo_native.c`, `tests/dpo/header_dpo_cpp.cpp`
- Development oracle: `tests/dpo/test_dpo_oracle.py`
- Compiled Eshkol AOT post-training driver: `tests/dpo/test_dpo_aot.esk`
- Build and test scripts: `scripts/build-dpo.sh`, `scripts/test-dpo.sh`

## Mathematical contract

Given batch size `B` in `[1, 8]`, sequence length `T` in `[1, 64]`, vocabulary
size `V` in `[2, 256]`, inverse temperature `beta > 0` (finite binary64),
label-smoothing parameter `eps` in `[0.0, 0.5)` (finite binary64), and mode
flag `reference_free` in `{0, 1}`:

| Tensor | Type & Shape | Role | Mutability |
|---|---|---|---|
| `policy_chosen_logits` | `f32[B, T, V]` | Policy logits on chosen sequences | Read-only (`const float *`) |
| `policy_rejected_logits` | `f32[B, T, V]` | Policy logits on rejected sequences | Read-only (`const float *`) |
| `ref_chosen_logits` | `f32[B, T, V]` | Frozen reference logits on chosen sequences | Strictly frozen (`const float *`, optional `NULL` when `reference_free == 1`) |
| `ref_rejected_logits` | `f32[B, T, V]` | Frozen reference logits on rejected sequences | Strictly frozen (`const float *`, optional `NULL` when `reference_free == 1`) |
| `chosen_targets`, `rejected_targets` | `i64[B, T]` | Token target IDs in `[0, V - 1]` | Read-only (`const int64_t *`) |
| `chosen_mask`, `rejected_mask` | `u8[B, T]` | Response mask (`0` = prompt/pad, `1` = response; `>= 1` active per sequence) | Read-only (`const uint8_t *`) |
| `d_logits_chosen`, `d_logits_rejected` | `f32[B, T, V]` | Exact analytic logit gradients (`+0.0f` wherever `mask == 0`) | Output (`float *`) |

### Masked sequence log-probabilities (binary64 log-sum-exp)

For each batch sequence `b` in `[0, B - 1]` and active response token `t` where
`mask[b, t] == 1`, let `z[v] = (double)logits[b, t, v]` and `y = target[b, t]`:

```text
m          = max_{v=0..V-1} z[v]
s          = sum_{v=0..V-1} exp(z[v] - m)
tok_logp   = (z[y] - m) - log(s)
logp[b]    = sum_{t: mask[b, t] == 1} tok_logp[b, t]
```

Evaluating `(z[y] - m) - log(s)` cancels any constant logit shift before
subtracting `log(s)`, preserving full binary64 precision under extreme uniform
shifts (verified at `+1e6f`).

### DPO / cDPO objective and telemetry

For each pair `b` in `[0, B - 1]`:

- When `reference_free == 0` (standard DPO / cDPO):
  `delta_w[b] = logp_pi_w[b] - logp_ref_w[b]`,
  `delta_l[b] = logp_pi_l[b] - logp_ref_l[b]`.
- When `reference_free == 1` (reference-free / SimPO-style):
  `delta_w[b] = logp_pi_w[b]`, `delta_l[b] = logp_pi_l[b]`.

Implicit rewards, preference margin, and pair loss:

```text
r_w[b]    = beta * delta_w[b]
r_l[b]    = beta * delta_l[b]
margin[b] = beta * (delta_w[b] - delta_l[b])
loss[b]   = (1 - eps) * softplus(-margin[b]) + eps * softplus(margin[b])
```

Batch-averaged telemetry (`et_dpo_telemetry`, binary64):

- `mean_loss = (1 / B) * sum_b loss[b]`
- `chosen_reward_mean = (1 / B) * sum_b r_w[b]`
- `rejected_reward_mean = (1 / B) * sum_b r_l[b]`
- `reward_margin_mean = (1 / B) * sum_b margin[b]`
- `preference_accuracy = (1 / B) * sum_b (margin[b] > 0.0 ? 1.0 : 0.0)`

### Numerical stability and exact analytic backward

Overflow- and underflow-safe binary64 primitives handle arbitrary finite margins
(tested beyond `|margin| >= 200`):

```text
softplus(x) = x > 0  ? x + log1p(exp(-x)) : log1p(exp(x))
sigmoid(x)  = x >= 0 ? 1 / (1 + exp(-x))  : exp(x) / (1 + exp(x))
```

Exact derivative with respect to pair margin `m = margin[b]`:

```text
dl/dm = (m >= 0) ? (eps - sigmoid(-m)) : (sigmoid(m) - (1 - eps))
g_w[b] = (beta / B) * dl/dm
g_l[b] = -g_w[b]
```

Exact analytic logit gradients per position `(b, t)` and vocabulary index `v`:

- If `mask[b, t] == 0`: `d_logits[b, t, v] = +0.0f` (exact IEEE-754 zero bits).
- If `mask[b, t] == 1`:
  `p[v] = exp(z[v] - m) / s`,
  `d_logits[b, t, v] = (float)(g[b] * ((v == target[b, t] ? 1.0 : 0.0) - p[v]))`.

Reference logits are strictly `const` and never receive gradients or writes.

## ABI contract, safety, and failure atomicity

- **C11 / C++17 compatibility**: `et_dpo_config` (32 bytes) and
  `et_dpo_telemetry` (40 bytes) are zero-padding standard-layout structs with
  compile-time size/offset assertions (`clang-21 -std=c11` and
  `clang++-21 -std=c++17`).
- **FP environment**: Compiled with `#pragma STDC FENV_ACCESS ON`,
  `#pragma STDC FP_CONTRACT OFF`, `-ffp-contract=off -fexcess-precision=standard
  -fno-fast-math`. On x86-64, x87 (`0x0c3f == 0x003f`), MXCSR
  (`0xffc0 == 0x1f80`, FTZ/DAZ disabled, exceptions masked), and
  `fegetround() == FE_TONEAREST` are verified on every entry point
  (`ET_DPO_ERR_FP_ENV`).
- **Zero heap allocation in core**: `dpo_alignment.o` imports only
  `exp`, `log`, `log1p`, `fegetround`, `memcpy`, `memset`, `__errno_location`,
  and stack-protector support.
- **Misaligned-pointer UB safety**: `et_dpo_*_raw` entry points accept `void *`
  / `const void *` and validate alignment and non-overflowing address spans
  prior to any typed access (`ET_DPO_ERR_ALIGNMENT` / `ET_DPO_ERR_OVERFLOW`).
- **Disjointness & validation**: All non-NULL buffers (including `cfg` and
  `out_telemetry`) must be pairwise disjoint (`ET_DPO_ERR_ALIAS`). All shapes,
  hyperparameters, masks (`0` or `1`, at least one active token per sequence),
  targets (`[0, V - 1]`), and finite logits (including masked positions) are
  validated before computation.
- **Failure atomicity**: A dry-run preflight verifies all intermediates and
  binary32 gradient representability before writing any output; any error
  leaves `out_telemetry`, `d_logits_chosen`, and `d_logits_rejected`
  byte-for-byte untouched and preserves `errno`.

## Measured verification results

Verified inside the pinned Ubuntu 22.04 container (`eshkol-r0-debug:20260928`,
Clang 21.1.8, pinned Eshkol `fe9dfd52`) via `bash scripts/test-dpo.sh`,
`python3 -m unittest tests/q0/test_python_isolation.py`, and
`bash scripts/check-ci-topology.sh`:

| Check | Measured result |
|---|---|
| Core deterministic archive (`libeshkol_transformer_dpo.a`) | SHA-256 `fe7851cf56eeb2dc3c17a0b949f85f34b5b75face2268106e343abd6edf75560` |
| Bridge deterministic archive (`libeshkol_transformer_dpo_aot.a`) | SHA-256 `47fa4ac0461fc54b574604c1053bde4550f0e23680a885cb052a052e6d91caa1` |
| C++17 header & archive linkage (`clang++-21 -std=c++17`) | PASS (normal + ASan/UBSan/LSan) |
| Native central FD gradients (DPO, cDPO, ref-free, masked prefixes) | 312 checks, `max_abs_err = 1.990e-08` |
| Extreme margins & shift invariance | PASS (`pos_margin = +267.73`, `neg_margin = -267.73`, `+1e6f` uniform shift exact to `<= 1e-12`) |
| Python binary64 oracle (`tests/dpo/test_dpo_oracle.py`) | 8 cases up to `[8, 64, 256]`, 265,324 values, `max_abs_diff = 1.177e-07` (`<= 1e-6`) |
| Sanitizers (`ASan`, `UBSan`, `LSan detect_leaks=1`) | PASS (C11 + C++17 suites) |
| Compiled Eshkol AOT (`tests/dpo/test_dpo_aot.esk`) | Byte-identical ELF & stdout across 2 fresh builds; ELF SHA-256 `c95a96809ae9c86fb0da0f08799a918e60cbe9735ae3669fe9926e133b3a4328` |
| AOT frozen reference immutability | 100% bit-identical SHA-256 (`2320861590233254187 5469269098677836280 -7023172456720796241 5036665916051968037`) and all f32 weight/bias bits across all 12 steps |
| Negative & failure-atomicity suite (native + AOT) | PASS (null, alias, misaligned `_raw`, OOB shapes/targets, bad/all-zero masks, invalid `beta`/`label_smoothing`/`reference_free`, non-finite logits, gradient overflow, x87/MXCSR FP-env) |
| Python isolation & CI topology | PASS (`test_python_isolation.py`: 5/5 OK; `check-ci-topology.sh`: PASS) |

### Compiled Eshkol AOT 12-step alignment trace

Configuration: `B=4, T=6` (2-token masked prompt prefix + 4-token response),
`V=12, D=8, beta=0.5, label_smoothing=0.0, reference_free=0, lr=0.4`
(`--strict-types --optimize 0 --no-stdlib`), policy head initialized as an exact
bitwise copy of the frozen reference head, 10 central FD checks across policy
logits and head weights/biases:

| Step | Mean DPO loss | Reward margin | Chosen reward | Rejected reward | Preference accuracy |
|---:|---:|---:|---:|---:|---:|
| 0 | `0.6931471805599453` | `0` | `0` | `0` | `0` |
| 1 | `0.5350922889733117` | `0.34669911716170065` | `0.19909167565731112` | `-0.14760744150438954` | `1` |
| 2 | `0.4270654844351293` | `0.632002610405733` | `0.35682895021286165` | `-0.27517366019287137` | `1` |
| 3 | `0.3510365812220513` | `0.8698413599516106` | `0.4840081363604771` | `-0.3858332235911335` | `1` |
| 4 | `0.2957241230115982` | `1.0714370933098474` | `0.5886525614623184` | `-0.482784531847529` | `1` |
| 5 | `0.2542081015242799` | `1.2451167799934146` | `0.6764352502356865` | `-0.5686815297577281` | `1` |
| 6 | `0.22217380641211962` | `1.3969474924476672` | `0.7513491940567285` | `-0.6455982983909387` | `1` |
| 7 | `0.19685585811364045` | `1.5313741842285116` | `0.8162399107715452` | `-0.7151342734569663` | `1` |
| 8 | `0.17642838452556114` | `1.6516994743147702` | `0.8731722132259768` | `-0.7785272610887934` | `1` |
| 9 | `0.15965056892214619` | `1.7604192752949652` | `0.9236744287663028` | `-0.8367448465286624` | `1` |
| 10 | `0.14565643836737913` | `1.8594520352511605` | `0.9688994858681578` | `-0.8905525493830029` | `1` |
| 11 | `0.13382677746822547` | `1.950296656000887` | `1.0097330088941532` | `-0.9405636471067338` | `1` |
| 12 | `0.12370904749999075` | `2.0341419297862235` | `1.0468665140250653` | `-0.9872754157611585` | `1` |

Post-alignment mode checks on the aligned policy: cDPO (`label_smoothing=0.1`)
`loss = 0.32712324047861313` (`acc = 1.0`), reference-free (`reference_free=1`)
`loss = 0.6817456754426597`.

## Limitations and scope

- Bounded CPU serial binary32 logit / binary64 objective implementation for
  `B <= 8`, `T <= 64`, `V in [2, 256]`.
- GPU acceleration, mixed precision (fp16/bf16), length-normalized SimPO
  variants beyond reference-free log-prob sums, multi-layer full-backbone DPO
  graph integration, and performance benchmarking are outside this scope.
- `tests/dpo/test_dpo_oracle.py` is a development-only verification oracle; the
  delivered alignment loss, backward pass, telemetry, and AOT driver are fully
  Eshkol/C-native.
