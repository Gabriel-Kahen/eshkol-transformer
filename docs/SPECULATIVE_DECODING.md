# Exact Lossless Speculative Decoding

This subsystem implements exact lossless speculative decoding (draft-verify
rejection sampling, Leviathan et al. 2023 / Chen et al. 2023) in C11/C++17 with
an opaque compiled Eshkol AOT bridge, independent Python analytical and empirical
verification, and deterministic ELF reproducibility.

## Files and archives

- Public C11/C++17 ABI: `include/eshkol_transformer/spec_decode_abi.h`
- Core implementation: `native/spec_decode.c` -> `build/spec_decode/libeshkol_transformer_spec_decode.a`
  - Defined exports (4): `et_spec_logits_to_probs_v1`, `et_spec_sample_token_v1`,
    `et_spec_verify_greedy_v1`, `et_spec_verify_sample_v1`.
  - Allowed undefined imports: `exp`, `log`, `memcpy`, `memset`,
    `__stack_chk_fail`, `_GLOBAL_OFFSET_TABLE_` (and `fegetround` on non-x86-64).
  - Performs zero heap allocations (`malloc`/`calloc`/`realloc`/`free` trapped
    in native tests).
- Eshkol AOT bridge: `native/spec_decode_eshkol_bridge.{h,c}` ->
  `build/spec_decode/libeshkol_transformer_spec_decode_aot.a`
  - Defined exports (21): `et_spec_aot_*` opaque context lifecycle, buffer
    marshaling, greedy/stochastic verification, probability conversion,
    canonical row sampling, and telemetry accessors.
  - Zero heap allocations during verification, probability conversion, and
    sampling calls.

## Mathematical specification

### Canonical binary64 logits-to-probability conversion

Given finite binary32 logits $z \in \mathbb{R}^V$ ($2 \le V \le 4096$),
temperature $T \ge 0$ (finite; $-0.0$ rejected), $1 \le \text{top\_k} \le V$,
and finite $\text{top\_p} \in (0, 1]$:

1. **Greedy ($T = +0.0$)**: selects the lowest vocabulary index attaining
   $\max_v z_v$ (including signed-zero ties) and emits an exact one-hot
   binary64 distribution.
2. **Stochastic ($T > 0$)**:
   - Computes $m = \max_v z_v$ and shifted scaled logits
     $s_v = ((double)z_v - m) / T$, rejecting any non-finite scaled value.
   - Evaluates canonical log-sum-exp:
     $\log Z = \log \sum_{v=0}^{V-1} \exp(s_v)$ and
     $p^{(0)}_v = \exp(s_v - \log Z)$.
   - Sorts token indices $0 \dots V-1$ using a deterministic non-recursive
     bottom-up merge sort by descending $p^{(0)}_v$, breaking ties by lowest
     token index.
   - **Top-$k$**: zeroes ranks $\ge \text{top\_k}$ and renormalizes in ID order.
   - **Top-$p$**: when $\text{top\_p} < 1$, keeps the shortest prefix of the
     sorted top-$k$ tokens whose cumulative probability reaches $\ge \text{top\_p}$
     (always retaining at least rank 0), zeroes the tail, and renormalizes in
     ID order. When $\text{top\_p} = 1$, no nucleus truncation is applied.

### Deterministic counter RNG

`et_spec_rng_state_v1` carries `{seed, counter}` (`uint64_t`). Each uniform
variate in $[0, 1)$ applies SplitMix64 to
$\text{seed} + \text{counter} \cdot \text{0x9e3779b97f4a7c15}$ and scales the
top 53 bits by $2^{-53}$. Counter overflow is rejected before numerical work.

### Exact greedy verification (`et_spec_verify_greedy_v1`)

Given horizon $K \in [1, 16]$, draft tokens $x_0, \dots, x_{K-1} \in [0, V)$,
and target logit rows $[K+1, V]$:

- For $i = 0 \dots K-1$, computes $t_i = \arg\max_v z^{\text{target}}_{i,v}$
  (lowest-index tie-break).
- Emits $t_i$; if $x_i = t_i$, increments `accepted_count` and continues
  (stopping immediately if $t_i = \text{eos\_token\_id}$).
- On the first mismatch $x_i \ne t_i$, emits the correction token $t_i$ and
  stops.
- If all $K$ draft tokens match without hitting EOS, emits the bonus token
  $t_K = \arg\max_v z^{\text{target}}_{K,v}$.
- Total emitted tokens per call is strictly in $[1, K+1]$ and produces
  token-for-token identical sequences to target-only autoregressive greedy
  decoding.

### Exact stochastic rejection-sampling verification (`et_spec_verify_sample_v1`)

Given draft tokens $x_0, \dots, x_{K-1}$, draft logits $[K, V]$, target logits
$[K+1, V]$, sampling parameters, and RNG state `(seed, counter)`:

- For $i = 0 \dots K-1$:
  - Computes canonical binary64 filtered distributions $q_i$ and $p_i$.
  - Draws $u_i \in [0, 1)$ from SplitMix64 at the current counter.
  - **Acceptance rule** (no division by zero, exact on half-open $[0, 1)$):
    - If $p_i(x_i) = 0$: reject unconditionally (prevents accepting a
      zero-probability target token even when $u_i = 0$).
    - Else if $q_i(x_i) \le p_i(x_i)$: accept unconditionally (cleanly handles
      $q_i(x_i) = 0, p_i(x_i) > 0$ without division, and guarantees 100%
      acceptance when $p_i = q_i$).
    - Else ($0 < p_i(x_i) < q_i(x_i)$): accept iff $u_i \cdot q_i(x_i) < p_i(x_i)$.
  - On acceptance: emit $x_i$, increment `accepted_count`, and stop immediately
    if $x_i = \text{eos\_token\_id}$.
  - On first rejection: form the exact residual
    $r(v) = \max(0, p_i(v) - q_i(v)) / \sum_w \max(0, p_i(w) - q_i(w))$
    (at the measure-zero boundary $\sum_w \max(0, p_i(w) - q_i(w)) = 0$ where
    $p_i = q_i$ everywhere and an adversarial $x_i$ had $q_i(x_i)=p_i(x_i)=0$,
    sample directly from exact $p_i$), draw $u_{\text{res}} \in [0, 1)$, emit
    the correction token from $r$, and stop.
- If all $K$ draft tokens are accepted without EOS, sample a bonus token from
  $p_K$ using one additional SplitMix64 draw.

For any single position with $x \sim q$ and acceptance probability
$\alpha = \sum_v \min(q(v), p(v))$, the exact marginal output distribution is:

$$P_{\text{spec}}(v) = q(v) \min\!\left(1, \frac{p(v)}{q(v)}\right) + (1 - \alpha)\, r(v) = p(v)$$

## ABI contract, validation, and failure atomicity

| Status | Value | Condition |
|---|---:|---|
| `ET_SPEC_OK` | `0` | Success |
| `ET_SPEC_ERR_NULL` | `-1` | Any required pointer argument is `NULL` |
| `ET_SPEC_ERR_ALIAS` | `-2` | Any two pointer spans overlap or wrap address space |
| `ET_SPEC_ERR_BOUNDS` | `-3` | $K \notin [1,16]$, $V \notin [2,4096]$, `out_capacity < K+1`, or RNG counter overflow |
| `ET_SPEC_ERR_PARAM` | `-4` | Invalid `temperature` ($< 0$ or $-0.0$), `top_k` ($\notin [1, V]$), or `top_p` ($\notin (0, 1]$) |
| `ET_SPEC_ERR_NONFINITE` | `-5` | Any NaN/Inf logit, parameter, or scaled intermediate |
| `ET_SPEC_ERR_TOKEN_RANGE` | `-6` | Any draft token $\ge V$ or invalid `eos_token_id` ($\ne \text{UINT32\_MAX}$ and $\ge V$) |
| `ET_SPEC_ERR_FP_ENV` | `-7` | Non-default FP environment (requires round-to-nearest, masked exceptions, FTZ=0, DAZ=0) |
| `ET_SPEC_ERR_ALIGNMENT` | `-8` | Pointer does not satisfy its element/struct alignment |

All inputs (including draft tokens and all rows beyond any early rejection or
EOS position) are validated upfront before any output write. On any error
status, every output buffer remains byte-for-byte unchanged.
The core C ABI carries unsigned 64-bit RNG counters. The Eshkol AOT bridge
uses signed 64-bit arguments and accessors, so it rejects a starting counter
that could produce an unrepresentable next counter on a successful call.

## Verification results (pinned container `eshkol-r0-debug:20260928`)

Validated with `clang-21` / `clang++-21` (LLVM 21.1.8) and the pinned Eshkol
toolchain via `bash scripts/test-spec-decode.sh`:

- **Build & symbol audit**: `PASS` (`libeshkol_transformer_spec_decode.a` and
  `libeshkol_transformer_spec_decode_aot.a`, deterministic `ar rcsD`, exact
  exports/imports, C++17 `-std=c++17` header & linkage check).
- **Native C11 tests**: `PASS` in both `-O2` (with `--wrap=malloc/calloc/realloc/free`
  trapping heap calls during verification/sampling) and
  ASan/UBSan/LSan (`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`,
  `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`), covering 32-token greedy
  rollouts (identical, partial, completely wrong drafts across $K \in \{1,4,16\}$),
  seeded replay, EOS early stop, $q(x)=0$ / $p(x)=0$ / $p=q$ edge cases,
  $K=1,V=2$ and $K=16,V=4096$ bounds, FP-environment rejection, alignment,
  aliasing, and sentinel byte-for-byte preservation.
- **Python analytical & empirical oracle** (`tests/spec_decode/test_spec_decode_oracle.py`):
  - `et_spec_logits_to_probs_v1` vs independent Python reference:
    `max_abs_err = 0.000e+00`.
  - Analytical marginal parity across 9 diverse `(q, p)` scenarios (including
    top-$k$, top-$p$, temperature, ties, $V=64$, and completely disjoint
    supports): `max_abs_err = 1.110e-16` (requirement: $\le 10^{-12}$).
  - Exact SplitMix64 single-step replay against Python over 250 consecutive steps: `PASS`.
  - Empirical sampling parity (60,000 C engine samples per scenario across 4
    scenarios, 240,000 total draws; 0 emissions of any $p(v)=0$ token):

| Scenario | Samples | $L_\infty$ error | TV distance | Expected $\alpha$ | Empirical $\alpha$ |
|---|---:|---:|---:|---:|---:|
| `mild_shift_temp_0.8` | 60,000 | 0.00292 | 0.00392 | 0.77845 | 0.77843 |
| `topk_partial_overlap` | 60,000 | 0.00149 | 0.00149 | 0.30151 | 0.30208 |
| `topp_nucleus_filtered` | 60,000 | 0.00446 | 0.00446 | 0.36974 | 0.36420 |
| `completely_disjoint_support` | 60,000 | 0.00225 | 0.00225 | 0.00000 | 0.00000 |

- **Compiled Eshkol AOT verification** (`tests/spec_decode/test_spec_decode_aot.esk`,
  `--strict-types --optimize 0 --no-stdlib`, $K=4, V=16$):
  - Greedy (`identical` draft, 32 tokens): `target_calls=7` (vs 32 target-only),
    `accepted=28`, `acc_per_step=4`, `exact_match=PASS`.
  - Greedy (`partial` draft, 32 tokens): `target_calls=8` (vs 32 target-only),
    `accepted=24`, `acc_per_step=3`, `exact_match=PASS`.
  - Stochastic seeded replay (24 tokens): `target_calls=6`, `accepted=18`,
    `final_counter=53`, `replay=PASS`.
  - EOS early termination: `greedy` (`emitted=6, eos_hit=1, PASS`) and
    `stochastic` (`emitted=7, eos_hit=1, PASS`).
  - Negative/non-finite rejections with telemetry preservation: `SPEC AOT NEGATIVE PASS`.
  - Two fresh compilations produced byte-identical ELFs and byte-identical stdout:
    `SHA-256 = 45f47170cfa0d4713fa01fc95a4742585995afbbe3231ce8154ac9ab495ab58f`.

## Scope and limitations

- Bounded to CPU binary32 input logits, binary64 probability computation,
  horizon $K \in [1, 16]$, and vocabulary $V \in [2, 4096]$.
- Does not claim GPU execution, KV-cache rollback integration, or production
  throughput benchmarks; model KV-cache truncation/commit around accepted prefix
  lengths remains the responsibility of downstream generation orchestration.
- The AOT witness uses deterministic mock draft and target logits assembled
  before verification. Its `target_calls` count represents those synthetic
  batches; it does not measure live model calls or establish a cache rollback
  witness.
