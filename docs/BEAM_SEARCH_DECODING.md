# Deterministic Beam Search & Logit-Processor Engine

## 1. Overview and Scope

This module delivers a standalone, Eshkol-native deterministic beam search and composable logit-processor engine in IEEE-754 binary64 with zero heap allocations across the entire native core and Eshkol AOT bridge.

Core files:

- Public C11 / C++17 ABI: `include/eshkol_transformer/beam_search_abi.h`
- Core C11 implementation: `native/beam_search.c` -> `build/beam_search/libeshkol_transformer_beam_search.a`
- Eshkol AOT bridge: `native/beam_search_eshkol_bridge.h`, `native/beam_search_eshkol_bridge.c` -> `build/beam_search/libeshkol_transformer_beam_search_aot.a`
- Tests:
  - `tests/beam_search/header_beam_cpp.cpp` (C++17 header, `std::is_standard_layout_v`, and linkage check)
  - `tests/beam_search/test_beam_search_native.c` (C11 unit, trap-tree, cyclic N-gram, Wu ranking shift, tie-breaking, early stopping, negative fail-closed, and `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free` zero-allocation hot-path tests)
  - `tests/beam_search/test_beam_search_oracle.py` (independent Python binary64 reference oracle across 10 decoding scenarios, 4 processor stages, and 10 negative status codes)
  - `tests/beam_search/test_beam_search_aot.esk` (compiled Eshkol AOT driver)
- Build and test scripts: `scripts/build-beam-search.sh`, `scripts/test-beam-search.sh`

## 2. Mathematical Contracts and Invariants

### 2.1 Floating-Point Environment and Zero-Allocation Contract

- Every native function executes in IEEE-754 binary64 under `#pragma STDC FENV_ACCESS ON`, `#pragma STDC FP_CONTRACT OFF`, `-ffp-contract=off -fexcess-precision=standard -fno-fast-math`, and validates default `FE_TONEAREST` rounding and default x87/MXCSR control state (`ET_BEAM_ERR_FP_ENV = -5` otherwise).
- Both `beam_search.o` and `beam_search_eshkol_bridge.o` contain **zero heap allocator imports** (`nm -u` audits no `malloc`, `calloc`, `realloc`, or `free`). All runtime hot paths are additionally wrapped with `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free` in both `-O2` and ASan/UBSan/LSan builds.
- Every API entry validates pointer alignment, non-overlapping spans (exact in-place `logits_in == logits_out` is permitted for logit processors and log-softmax; partial overlaps fail closed with `ET_BEAM_ERR_ALIASING = -4`), finiteness (`NaN` and `+INFINITY` fail closed with `ET_BEAM_ERR_NONFINITE = -2`), and non-empty valid token sets (all `-INFINITY` fails closed with `ET_BEAM_ERR_NO_VALID_TOKENS = -3`). All mutating APIs are failure-atomic and leave outputs/committed state untouched on error.

### 2.2 Composable Logit Processors

Within each beam step, logit processors run in canonical deterministic order before binary64 log-softmax:

1. **Keskar/CTRL Repetition Penalty (`et_beam_apply_repetition_penalty_v1`)**:
   - Requires finite `penalty >= 1.0` (`ET_BEAM_ERR_INVALID_ARG` if `< 1.0`).
   - When `penalty == 1.0`, performs a bit-exact identity copy (preserving `-0.0` sign bits and `-INFINITY` masks).
   - Otherwise, for each unique token ID `t` present in `history_tokens[0..history_len-1]` (tracked via a stack 64-word bitset so duplicate history occurrences are penalized exactly once):
     - If finite `logit[t] > 0.0`: `logit_out[t] = logit[t] / penalty`
     - Else if finite `logit[t] <= 0.0`: `logit_out[t] = logit[t] * penalty`
   - Existing `-INFINITY` entries are preserved; finite scaling overflow fails closed with `ET_BEAM_ERR_NONFINITE`.

2. **Exact No-Repeat N-Gram Blocking (`et_beam_apply_no_repeat_ngram_v1`)**:
   - `ngram_size == 0` disables blocking (validated bit-exact copy); `ngram_size == 1` is rejected (`ET_BEAM_ERR_INVALID_ARG`); `ngram_size > 256` fails with `ET_BEAM_ERR_LIMIT`.
   - For `ngram_size = N >= 2` and `seq_len >= N`, scans all `start in [0, seq_len - N]` and masks `logits_out[seq[start + N - 1]] = -INFINITY` whenever `seq[start .. start + N - 2] == seq[seq_len - (N - 1) .. seq_len - 1]`.
   - Guarantees **zero repeated N-grams** across every surviving beam hypothesis.

3. **Min-Length EOS Suppression (`et_beam_apply_min_length_v1`)**:
   - Whenever `generated_len < min_new_tokens`, masks `logits_out[eos_token_id] = -INFINITY`.

4. **Binary64 Log-Softmax (`et_beam_log_softmax_v1`)**:
   - Computes `m = max_{v finite} logits[v]`, `log_sum_exp = log(sum_{v finite} exp(logits[v] - m))`, and `log_probs[v] = (logits[v] - m) - log_sum_exp` for finite `v` (`-INFINITY` for masked `v`).

### 2.3 Wu et al. (2016) Length Penalty

For generated token count `L = generated_len` (including `eos_token_id` when finished, `1 <= L <= max_new_tokens`) and finite `alpha = length_penalty >= 0.0`:

```
LP(L, alpha) =
    1.0                         if alpha == 0.0
    (5.0 + L) / 6.0             if alpha == 1.0
    ((5.0 + L) / 6.0) ^ alpha   otherwise

score(sum_log_prob, L, alpha) =
    sum_log_prob                if alpha == 0.0
    sum_log_prob / LP(L, alpha) otherwise
```

Because `sum_log_prob <= 0.0` and `LP(L, alpha)` is monotonically non-decreasing in `L` for `alpha >= 0.0`, larger `alpha` favors longer hypotheses while `alpha == 0.0` ranks strictly by raw cumulative log-probability.

### 2.4 Beam Expansion, Strict Total-Order Tie-Breaking, and Early Stopping

- **Capacity Bounds**: `num_beams B in [1, 8]`, `vocab_size V in [2, 4096]`, `prompt_len + max_new_tokens <= 256`, `prompt_len >= 1`, `max_new_tokens >= 1`, `min_new_tokens <= max_new_tokens`.
- **Initialization**: Step 0 starts with `active_count = 1` (the prompt with `log_prob = 0.0`) to avoid `B`-way duplicate expansion of identical prompts.
- **Per-Step Candidate Total Order**: Across all `active_count * V` valid `(parent_idx, token_id)` continuations, the top `2 * B` candidates are selected under the strict total order:
  1. Higher `cand_log_prob = parent_log_prob + step_log_prob`
  2. Lower `parent_idx`
  3. Lower `token_id`
  Because `(parent_idx, token_id)` is unique per step, ordering is 100% deterministic and independent of sort stability.
- **Finished Pool (`capacity = 2 * B`)**: Scanning the top `2 * B` candidates in rank order `r = 0 .. top_count - 1`:
  - If `token_id == eos_token_id` and `r < B`, insert the completed hypothesis with its Wu score into `finished_pool` (sorted by `(score DESC, sum_log_prob DESC, seq_len ASC, tokens LEX ASC)`, retaining the best `2 * B`).
  - If `token_id != eos_token_id` and `next_active_count < B`, add it to `next_active`.
- **Exact Admissible Early Stopping (`early_stopping == 1`)**:
  - At the end of a step, decoding stops if `steps_done >= max_new_tokens` or `active_count == 0`.
  - Otherwise, when `early_stopping == 1` and `finished_count >= B`, compute the exact upper bound on any unfinished beam's eventual Wu score:
    `best_possible_score = active_log_probs[0] / LP(max_new_tokens, alpha)`.
  - Because `active_log_probs[0] <= 0.0` and `LP(L, alpha) <= LP(max_new_tokens, alpha)`, no future completion can exceed `best_possible_score`. When `best_possible_score <= finished_pool[B - 1].score`, decoding stops early (`stopped_early = 1`) with exact optimality preserved.
- **Finalization (`et_beam_finalize_v1`)**:
  - When `finished_count >= B`, emits the top `min(max_hyps, B)` completed hypotheses from `finished_pool`.
  - When `finished_count < B`, scores remaining active beams at `steps_done`, merges them with `finished_pool` under the same strict total order, and emits the top `min(max_hyps, B, total)`.
- **`B = 1` Greedy Equivalence**: With `B = 1` and `length_penalty = 0.0`, step-by-step selection and early stopping match exact greedy decoding token-for-token and log-prob-for-log-prob.

## 3. Verification Evidence

Verified inside container `eshkol-r0-debug:20260928` (`clang-21`, `clang++-21`, pinned Eshkol toolchain):

- **Deterministic archives & zero-alloc symbol audit** (`scripts/build-beam-search.sh`): PASS
  - `libeshkol_transformer_beam_search.a` SHA-256: `f411fdbe11677114a22b6183c02693b234a1b68a008036927cf79044597bdf4d`
  - `libeshkol_transformer_beam_search_aot.a` SHA-256: `ca8571a9677934cf2f19f82ce36047311d350fbc48f80a59a84ce7c12a767c43`
- **C++17 header, standard-layout & linkage** (`tests/beam_search/header_beam_cpp.cpp`): PASS (normal `-O2` and ASan/UBSan/LSan)
- **Native C11 tests with zero-allocation trap** (`tests/beam_search/test_beam_search_native.c`, `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free`): PASS (normal `-O2` and `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`, `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`)
- **Independent Python binary64 oracle** (`tests/beam_search/test_beam_search_oracle.py`): PASS
  - Verified bit-exact parity (`max_abs_err = 0.000e+00`) across 10 decoding scenarios (trap-tree `B=1` vs `B=4`, Wu shift `alpha=0.0` vs `1.5`, and 6 multi-parameter hash-model stress cases covering `B in {1, 3, 4, 6, 8}`, `ngram in {0, 2, 3}`, `min_new_tokens`, `repetition_penalty`, `length_penalty`, and `early_stopping in {0, 1}`), all 4 logit-processor / log-softmax stages, and 10 fail-closed negative status codes with sentinel preservation.
- **Compiled Eshkol AOT driver** (`tests/beam_search/test_beam_search_aot.esk`, `--strict-types --optimize 0 --no-stdlib`): PASS
  - Dual fresh compilations produced byte-identical ELF binaries and byte-identical stdout:
  - AOT ELF SHA-256: `121a31d839ac39fd90c33fe1dcdeca100d0c90622a798113192f7750b01938ca`
  - Verified `B=1` greedy trap `[1, 2, 4, 0]` vs `B=4` global optimum `[1, 3, 5, 6, 0]` (`> +3.5` nats gain), `no_repeat_ngram_size = 2` and `3` eliminating loops on cyclic logits, `min_new_tokens = 4` EOS suppression (`gen_len = 5`), Wu ranking shift (`alpha = 0.0` short winner `gen_len = 2` vs `alpha = 1.5` long winner `gen_len = 5`), and negative fail-closed rejections with sentinel preservation.
- **Repository gates**: `tests/q0/test_python_isolation.py` (5/5 OK) and `scripts/check-ci-topology.sh` (PASS).

## 4. Measured Limitations and Unsupported Cases

- **Hard capacity bounds**: `num_beams <= 8`, `vocab_size in [2, 4096]`, `prompt_len + max_new_tokens <= 256`, `finished_pool <= 16`. Exceeding these bounds fails explicitly with `ET_BEAM_ERR_LIMIT (-7)` or `ET_BEAM_ERR_INVALID_ARG (-1)`.
- **Single-sequence batching**: One `et_beam_state_v1` tracks a single prompt with up to `B` active beams; multi-request batching is outside this engine and composes cleanly at the caller/scheduler layer.
- **CPU IEEE-754 binary64 only**: All logit processing, log-softmax, and Wu scoring execute on CPU in binary64 under default `FE_TONEAREST` MXCSR/x87 state; non-default FP environments fail closed with `ET_BEAM_ERR_FP_ENV (-5)`.
- **Diverse beam search / group penalties**: Hamming diversity penalties, constrained multi-beam Trie intersections, and stochastic beam sampling are not implemented in v1.
