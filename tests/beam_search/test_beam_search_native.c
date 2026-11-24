#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/beam_search_abi.h"
#include "beam_search_eshkol_bridge.h"

#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* ---- Allocation Trap for Hot-Path Zero-Allocation Verification ---- */

static int hot_path_active = 0;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!hot_path_active);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!hot_path_active);
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!hot_path_active);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!hot_path_active);
  __real_free(ptr);
}

static uint64_t f64_bits(double v) {
  uint64_t b = 0u;
  memcpy(&b, &v, sizeof(b));
  return b;
}

static et_beam_config_v1 make_cfg(uint32_t num_beams, uint32_t vocab_size,
                                  uint32_t eos_id, uint32_t max_new,
                                  uint32_t min_new, uint32_t ngram,
                                  uint32_t early_stop, double lp, double rp) {
  et_beam_config_v1 c;
  memset(&c, 0, sizeof(c));
  c.struct_size = sizeof(et_beam_config_v1);
  c.num_beams = num_beams;
  c.vocab_size = vocab_size;
  c.eos_token_id = eos_id;
  c.max_new_tokens = max_new;
  c.min_new_tokens = min_new;
  c.no_repeat_ngram_size = ngram;
  c.early_stopping = early_stop;
  c.reserved = 0u;
  c.length_penalty = lp;
  c.repetition_penalty = rp;
  return c;
}

/* ---- Deterministic Synthetic Logit Models (Shared with Python Oracle) ---- */

/* Trap-logits tree (V=8, EOS=0, prompt=[1]):
 *   [1]         -> tok 2 (+1.2, greedy trap), tok 3 (+0.9, global optimum), else -6.0
 *   [..., 2]    -> tok 4 (+0.2), all other tokens (0.0)
 *   [..., 4]    -> uniform 0.0 across all 8 tokens (EOS tie-broken first)
 *   [..., 3]    -> tok 5 (+8.0), else -4.0
 *   [..., 5]    -> tok 6 (+8.0), else -4.0
 *   [..., 6]    -> tok 0/EOS (+8.0), else -4.0
 *   other       -> tok 0 (+0.5), else 0.0
 */
static void trap_tree_logits(const uint32_t *seq, size_t len, double out[8]) {
  const uint32_t last = seq[len - 1u];
  if (len == 1u && last == 1u) {
    for (size_t v = 0u; v < 8u; ++v)
      out[v] = -6.0;
    out[2] = 1.2;
    out[3] = 0.9;
  } else if (last == 2u) {
    for (size_t v = 0u; v < 8u; ++v)
      out[v] = 0.0;
    out[4] = 0.2;
  } else if (last == 4u) {
    for (size_t v = 0u; v < 8u; ++v)
      out[v] = 0.0;
  } else if (last == 3u) {
    for (size_t v = 0u; v < 8u; ++v)
      out[v] = -4.0;
    out[5] = 8.0;
  } else if (last == 5u) {
    for (size_t v = 0u; v < 8u; ++v)
      out[v] = -4.0;
    out[6] = 8.0;
  } else if (last == 6u) {
    for (size_t v = 0u; v < 8u; ++v)
      out[v] = -4.0;
    out[0] = 8.0;
  } else {
    for (size_t v = 0u; v < 8u; ++v)
      out[v] = 0.0;
    out[0] = 0.5;
  }
}

/* Cyclic logits model (V=8, EOS=0, prompt=[1, 2]):
 * Strongly cycles 1 -> 2 -> 3 -> 1 -> ... with fallback tokens 4, 5, 6, 7.
 */
static void cyclic_logits(const uint32_t *seq, size_t len, double out[8]) {
  const uint32_t last = seq[len - 1u];
  out[0] = -8.0;
  out[1] = -1.0;
  out[2] = -1.0;
  out[3] = -1.0;
  out[4] = 2.0;
  out[5] = 1.5;
  out[6] = 1.0;
  out[7] = 0.5;
  if (last == 1u)
    out[2] = 6.0;
  else if (last == 2u)
    out[3] = 6.0;
  else if (last == 3u)
    out[1] = 6.0;
  else if (last == 4u)
    out[1] = 5.5;
  else if (last == 5u)
    out[2] = 5.5;
  else if (last == 6u)
    out[3] = 5.5;
  else
    out[1] = 5.0;
}

/* Wu length-penalty ranking shift graph (V=6, EOS=0, prompt=[1]):
 *   Short path: [1, 2, 0] (gen_len=2, sum_log_prob ~ -0.5981)
 *   Long path:  [1, 3, 4, 5, 2, 0] (gen_len=5, sum_log_prob ~ -0.7981)
 *   At alpha=0.0: short wins (-0.5981 > -0.7981).
 *   At alpha=1.5: long wins (-0.3709 > -0.4746).
 */
static void wu_shift_logits(const uint32_t *seq, size_t len, double out[6]) {
  for (size_t v = 0u; v < 6u; ++v)
    out[v] = -10.0;
  const uint32_t last = seq[len - 1u];
  if (len == 1u && last == 1u) {
    out[2] = 0.0;
    out[3] = -0.2;
  } else if (last == 2u) {
    out[0] = 10.0;
  } else if (last == 3u) {
    out[4] = 10.0;
  } else if (last == 4u) {
    out[5] = 10.0;
  } else if (last == 5u) {
    out[2] = 10.0;
  } else {
    out[0] = 10.0;
  }
}

/* Hash-based pseudo-random deterministic logits (V=16) for oracle stress. */
static void hash_model_logits(uint64_t seed, const uint32_t *seq, size_t len,
                              size_t vocab, double *out) {
  uint64_t h = seed ^ UINT64_C(0x9e3779b97f4a7c15);
  for (size_t i = 0u; i < len; ++i) {
    h ^= (uint64_t)seq[i] + UINT64_C(0x9e3779b97f4a7c15) + (h << 6u) +
         (h >> 2u);
  }
  for (size_t v = 0u; v < vocab; ++v) {
    uint64_t x = h ^ ((uint64_t)v * UINT64_C(0xbf58476d1ce4e5b9));
    x ^= x >> 30u;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27u;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31u;
    int64_t bucket = (int64_t)(x % 41u) - 20; /* [-20, 20] */
    out[v] = (double)bucket * 0.125;
  }
}

static int has_repeated_ngram(const uint32_t *seq, size_t len, size_t n) {
  if (n < 2u || len < n)
    return 0;
  for (size_t i = 0u; i + n <= len; ++i) {
    for (size_t j = i + 1u; j + n <= len; ++j) {
      if (memcmp(seq + i, seq + j, n * sizeof(uint32_t)) == 0)
        return 1;
    }
  }
  return 0;
}

/* ---- Unit Test 1: Logit Processors & Log-Softmax ---- */

static void test_logit_processors(void) {
  const uint32_t hist[4] = {1u, 3u, 1u, 4u}; /* token 1 appears twice -> once! */
  const double raw[6] = {2.0, 4.0, -3.0, -2.0, -0.0, -INFINITY};
  double out[6];

  /* Repetition penalty == 1.0 -> bit-exact identity (including -0.0 & -Inf). */
  hot_path_active = 1;
  CHECK(et_beam_apply_repetition_penalty_v1(hist, 4u, 6u, 1.0, raw, out) ==
        ET_BEAM_OK);
  hot_path_active = 0;
  for (size_t i = 0u; i < 6u; ++i)
    CHECK(f64_bits(out[i]) == f64_bits(raw[i]));

  /* Repetition penalty == 2.0:
   *   tok 0: not in hist -> 2.0
   *   tok 1: in hist, > 0 -> 4.0 / 2.0 = 2.0 (NOT divided twice!)
   *   tok 2: not in hist -> -3.0
   *   tok 3: in hist, <= 0 -> -2.0 * 2.0 = -4.0
   *   tok 4: in hist, -0.0 <= 0 -> -0.0 * 2.0 = -0.0
   *   tok 5: -INFINITY preserved
   */
  hot_path_active = 1;
  CHECK(et_beam_apply_repetition_penalty_v1(hist, 4u, 6u, 2.0, raw, out) ==
        ET_BEAM_OK);
  hot_path_active = 0;
  CHECK(out[0] == 2.0);
  CHECK(out[1] == 2.0);
  CHECK(out[2] == -3.0);
  CHECK(out[3] == -4.0);
  CHECK(f64_bits(out[4]) == f64_bits(-0.0));
  CHECK(isinf(out[5]) && out[5] < 0.0);

  /* Exact in-place repetition penalty. */
  double inplace[6];
  memcpy(inplace, raw, sizeof(raw));
  hot_path_active = 1;
  CHECK(et_beam_apply_repetition_penalty_v1(hist, 4u, 6u, 2.0, inplace,
                                            inplace) == ET_BEAM_OK);
  hot_path_active = 0;
  for (size_t i = 0u; i < 6u; ++i)
    CHECK(f64_bits(inplace[i]) == f64_bits(out[i]));

  /* No-repeat 2-gram: seq = [1, 2, 3, 1] -> suffix [1] -> bans 2. */
  const uint32_t seq2[4] = {1u, 2u, 3u, 1u};
  const double logits_ng[6] = {0.5, 1.0, 2.0, 3.0, -0.0, 5.0};
  hot_path_active = 1;
  CHECK(et_beam_apply_no_repeat_ngram_v1(seq2, 4u, 6u, 2u, logits_ng, out) ==
        ET_BEAM_OK);
  hot_path_active = 0;
  CHECK(isinf(out[2]) && out[2] < 0.0);
  CHECK(out[0] == 0.5 && out[1] == 1.0 && out[3] == 3.0 &&
        f64_bits(out[4]) == f64_bits(-0.0) && out[5] == 5.0);

  /* No-repeat 3-gram: seq = [1, 2, 3, 4, 1, 2] -> suffix [1, 2] -> bans 3. */
  const uint32_t seq3[6] = {1u, 2u, 3u, 4u, 1u, 2u};
  hot_path_active = 1;
  CHECK(et_beam_apply_no_repeat_ngram_v1(seq3, 6u, 6u, 3u, logits_ng, out) ==
        ET_BEAM_OK);
  hot_path_active = 0;
  CHECK(isinf(out[3]) && out[3] < 0.0);
  CHECK(out[2] == 2.0);

  /* Self-repeat 2-gram: seq = [1, 1] -> suffix [1] -> bans 1. */
  const uint32_t seq_self[2] = {1u, 1u};
  hot_path_active = 1;
  CHECK(et_beam_apply_no_repeat_ngram_v1(seq_self, 2u, 6u, 2u, logits_ng,
                                         out) == ET_BEAM_OK);
  hot_path_active = 0;
  CHECK(isinf(out[1]) && out[1] < 0.0);
  CHECK(out[0] == 0.5);

  /* Min-length EOS suppression. */
  hot_path_active = 1;
  CHECK(et_beam_apply_min_length_v1(2u, 3u, 0u, 6u, logits_ng, out) ==
        ET_BEAM_OK);
  hot_path_active = 0;
  CHECK(isinf(out[0]) && out[0] < 0.0);
  CHECK(out[1] == 1.0);

  hot_path_active = 1;
  CHECK(et_beam_apply_min_length_v1(3u, 3u, 0u, 6u, logits_ng, out) ==
        ET_BEAM_OK);
  hot_path_active = 0;
  CHECK(f64_bits(out[0]) == f64_bits(logits_ng[0]));

  /* Log-softmax with masked tokens: probabilities over unmasked sum to 1. */
  double lsm_in[4] = {1.0, -INFINITY, 3.0, 2.0};
  double lsm_out[4] = {0.0, 0.0, 0.0, 0.0};
  hot_path_active = 1;
  CHECK(et_beam_log_softmax_v1(4u, lsm_in, lsm_out) == ET_BEAM_OK);
  hot_path_active = 0;
  CHECK(isinf(lsm_out[1]) && lsm_out[1] < 0.0);
  const double prob_sum = exp(lsm_out[0]) + exp(lsm_out[2]) + exp(lsm_out[3]);
  CHECK(fabs(prob_sum - 1.0) <= 1.0e-14);
}

/* ---- Unit Test 2: B=1 Greedy Equivalence vs B=2/4 Global Optimum ---- */

static void test_trap_tree_and_greedy_equivalence(void) {
  et_beam_state_v1 *state =
      (et_beam_state_v1 *)calloc(1u, sizeof(et_beam_state_v1));
  CHECK(state != NULL);

  const uint32_t prompt[1] = {1u};
  double logits[ET_BEAM_MAX_BEAMS * 8u];
  uint32_t seq_buf[ET_BEAM_MAX_SEQ_LEN];

  /* Run B = 1, length_penalty = 0.0 (exact greedy). */
  const et_beam_config_v1 cfg1 =
      make_cfg(1u, 8u, 0u, 4u, 0u, 0u, 1u, 0.0, 1.0);
  hot_path_active = 1;
  CHECK(et_beam_init_v1(state, &cfg1, prompt, 1u) == ET_BEAM_OK);
  while (!et_beam_is_done_v1(state)) {
    const uint32_t na = et_beam_active_count_v1(state);
    for (uint32_t b = 0u; b < na; ++b) {
      size_t slen = 0u;
      double lp = 0.0;
      CHECK(et_beam_get_active_beam_v1(state, b, seq_buf, ET_BEAM_MAX_SEQ_LEN,
                                       &slen, &lp) == ET_BEAM_OK);
      trap_tree_logits(seq_buf, slen, logits + (size_t)b * 8u);
    }
    CHECK(et_beam_step_v1(state, logits, na) == ET_BEAM_OK);
  }
  et_beam_hypothesis_v1 h1[1];
  size_t c1 = 0u;
  CHECK(et_beam_finalize_v1(state, h1, 1u, &c1) == ET_BEAM_OK);
  hot_path_active = 0;

  CHECK(c1 == 1u);
  CHECK(h1[0].finished == 1u);
  CHECK(h1[0].seq_len == 4u);
  CHECK(h1[0].tokens[0] == 1u && h1[0].tokens[1] == 2u &&
        h1[0].tokens[2] == 4u && h1[0].tokens[3] == 0u);
  CHECK(f64_bits(h1[0].score) == f64_bits(h1[0].sum_log_prob));

  /* Independent greedy loop verifying exact token-for-token equivalence. */
  uint32_t g_seq[8] = {1u};
  size_t g_len = 1u;
  double g_lp = 0.0;
  for (size_t s = 0u; s < 4u; ++s) {
    double raw[8], lps[8];
    trap_tree_logits(g_seq, g_len, raw);
    hot_path_active = 1;
    CHECK(et_beam_log_softmax_v1(8u, raw, lps) == ET_BEAM_OK);
    hot_path_active = 0;
    uint32_t best_t = 0u;
    for (uint32_t v = 1u; v < 8u; ++v) {
      if (lps[v] > lps[best_t])
        best_t = v;
    }
    g_seq[g_len++] = best_t;
    g_lp += lps[best_t];
    if (best_t == 0u)
      break;
  }
  CHECK(g_len == h1[0].seq_len);
  CHECK(memcmp(g_seq, h1[0].tokens, g_len * sizeof(uint32_t)) == 0);
  CHECK(f64_bits(g_lp) == f64_bits(h1[0].sum_log_prob));

  /* Run B = 2 and B = 4: both MUST recover the global optimum [1, 3, 5, 6, 0]. */
  const uint32_t widths[2] = {2u, 4u};
  for (size_t wi = 0u; wi < 2u; ++wi) {
    const uint32_t B = widths[wi];
    const et_beam_config_v1 cfgB =
        make_cfg(B, 8u, 0u, 4u, 0u, 0u, 1u, 0.0, 1.0);
    hot_path_active = 1;
    CHECK(et_beam_init_v1(state, &cfgB, prompt, 1u) == ET_BEAM_OK);
    while (!et_beam_is_done_v1(state)) {
      const uint32_t na = et_beam_active_count_v1(state);
      for (uint32_t b = 0u; b < na; ++b) {
        size_t slen = 0u;
        double lp = 0.0;
        CHECK(et_beam_get_active_beam_v1(state, b, seq_buf, ET_BEAM_MAX_SEQ_LEN,
                                         &slen, &lp) == ET_BEAM_OK);
        trap_tree_logits(seq_buf, slen, logits + (size_t)b * 8u);
      }
      CHECK(et_beam_step_v1(state, logits, na) == ET_BEAM_OK);
    }
    et_beam_hypothesis_v1 hB[8];
    size_t cB = 0u;
    CHECK(et_beam_finalize_v1(state, hB, B, &cB) == ET_BEAM_OK);
    hot_path_active = 0;

    CHECK(cB == B);
    CHECK(hB[0].finished == 1u);
    CHECK(hB[0].seq_len == 5u);
    CHECK(hB[0].tokens[0] == 1u && hB[0].tokens[1] == 3u &&
          hB[0].tokens[2] == 5u && hB[0].tokens[3] == 6u &&
          hB[0].tokens[4] == 0u);
    /* Strictly beats greedy B=1 by > 3.5 nats! */
    CHECK(hB[0].sum_log_prob > h1[0].sum_log_prob + 3.5);
  }

  free(state);
}

/* ---- Unit Test 3: No-Repeat N-Gram (2 and 3), Min-Length & Wu Shift ---- */

static void test_ngram_minlen_wu_and_ties(void) {
  et_beam_state_v1 *state =
      (et_beam_state_v1 *)calloc(1u, sizeof(et_beam_state_v1));
  CHECK(state != NULL);

  double logits[ET_BEAM_MAX_BEAMS * 8u];
  uint32_t seq_buf[ET_BEAM_MAX_SEQ_LEN];

  /* 3a. Cyclic logits: ngram=0 loops; ngram=2 has 0 repeated bigrams;
   *     ngram=3 has 0 repeated trigrams. */
  const uint32_t c_prompt[2] = {1u, 2u};
  const uint32_t ngrams[3] = {0u, 2u, 3u};
  for (size_t ni = 0u; ni < 3u; ++ni) {
    const uint32_t ng = ngrams[ni];
    const et_beam_config_v1 cfg =
        make_cfg(4u, 8u, 0u, 8u, 8u, ng, 0u, 0.0, 1.0);
    hot_path_active = 1;
    CHECK(et_beam_init_v1(state, &cfg, c_prompt, 2u) == ET_BEAM_OK);
    while (!et_beam_is_done_v1(state)) {
      const uint32_t na = et_beam_active_count_v1(state);
      for (uint32_t b = 0u; b < na; ++b) {
        size_t slen = 0u;
        double lp = 0.0;
        CHECK(et_beam_get_active_beam_v1(state, b, seq_buf, ET_BEAM_MAX_SEQ_LEN,
                                         &slen, &lp) == ET_BEAM_OK);
        cyclic_logits(seq_buf, slen, logits + (size_t)b * 8u);
      }
      CHECK(et_beam_step_v1(state, logits, na) == ET_BEAM_OK);
    }
    et_beam_hypothesis_v1 hyps[4];
    size_t cnt = 0u;
    CHECK(et_beam_finalize_v1(state, hyps, 4u, &cnt) == ET_BEAM_OK);
    hot_path_active = 0;

    CHECK(cnt == 4u);
    if (ng == 0u) {
      CHECK(has_repeated_ngram(hyps[0].tokens, hyps[0].seq_len, 2u) == 1);
      CHECK(has_repeated_ngram(hyps[0].tokens, hyps[0].seq_len, 3u) == 1);
    } else {
      for (size_t i = 0u; i < cnt; ++i) {
        CHECK(has_repeated_ngram(hyps[i].tokens, hyps[i].seq_len, ng) == 0);
      }
    }
  }

  /* 3b. Wu length penalty ranking shift: alpha=0.0 -> short [1,2,0],
   *     alpha=1.5 -> long [1,3,4,5,2,0]. */
  const uint32_t w_prompt[1] = {1u};
  const double alphas[2] = {0.0, 1.5};
  for (size_t ai = 0u; ai < 2u; ++ai) {
    const et_beam_config_v1 cfg =
        make_cfg(4u, 6u, 0u, 5u, 0u, 0u, 0u, alphas[ai], 1.0);
    hot_path_active = 1;
    CHECK(et_beam_init_v1(state, &cfg, w_prompt, 1u) == ET_BEAM_OK);
    while (!et_beam_is_done_v1(state)) {
      const uint32_t na = et_beam_active_count_v1(state);
      for (uint32_t b = 0u; b < na; ++b) {
        size_t slen = 0u;
        double lp = 0.0;
        CHECK(et_beam_get_active_beam_v1(state, b, seq_buf, ET_BEAM_MAX_SEQ_LEN,
                                         &slen, &lp) == ET_BEAM_OK);
        wu_shift_logits(seq_buf, slen, logits + (size_t)b * 6u);
      }
      CHECK(et_beam_step_v1(state, logits, na) == ET_BEAM_OK);
    }
    et_beam_hypothesis_v1 hyps[4];
    size_t cnt = 0u;
    CHECK(et_beam_finalize_v1(state, hyps, 4u, &cnt) == ET_BEAM_OK);
    hot_path_active = 0;

    CHECK(cnt >= 2u);
    if (ai == 0u) {
      CHECK(hyps[0].seq_len == 3u && hyps[0].tokens[1] == 2u &&
            hyps[0].tokens[2] == 0u);
    } else {
      CHECK(hyps[0].seq_len == 6u && hyps[0].tokens[1] == 3u &&
            hyps[0].tokens[2] == 4u && hyps[0].tokens[3] == 5u &&
            hyps[0].tokens[4] == 2u && hyps[0].tokens[5] == 0u);
    }
  }

  /* 3c. Deterministic tie-breaking on identical logits:
   *     lower parent_idx first, then lower token_id first. */
  const et_beam_config_v1 tie_cfg =
      make_cfg(3u, 5u, 0u, 2u, 2u, 0u, 0u, 0.0, 1.0);
  const double flat[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
  const double flat3[15] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  hot_path_active = 1;
  CHECK(et_beam_init_v1(state, &tie_cfg, w_prompt, 1u) == ET_BEAM_OK);
  CHECK(et_beam_step_v1(state, flat, 1u) == ET_BEAM_OK);
  /* Step 0: EOS (0) masked by min_new=2, so beams must be [1,1], [1,2], [1,3]. */
  for (uint32_t b = 0u; b < 3u; ++b) {
    size_t slen = 0u;
    double lp = 0.0;
    CHECK(et_beam_get_active_beam_v1(state, b, seq_buf, ET_BEAM_MAX_SEQ_LEN,
                                     &slen, &lp) == ET_BEAM_OK);
    CHECK(slen == 2u && seq_buf[0] == 1u && seq_buf[1] == b + 1u);
  }
  CHECK(et_beam_step_v1(state, flat3, 3u) == ET_BEAM_OK);
  /* Step 1: all scores equal -> parent 0 wins all 3 slots: [1,1,1], [1,1,2], [1,1,3]. */
  et_beam_hypothesis_v1 tie_h[3];
  size_t tie_c = 0u;
  CHECK(et_beam_finalize_v1(state, tie_h, 3u, &tie_c) == ET_BEAM_OK);
  hot_path_active = 0;
  CHECK(tie_c == 3u);
  for (size_t i = 0u; i < 3u; ++i) {
    CHECK(tie_h[i].seq_len == 3u && tie_h[i].tokens[0] == 1u &&
          tie_h[i].tokens[1] == 1u && tie_h[i].tokens[2] == (uint32_t)(i + 1u));
  }

  /* 3d. Early stopping verification: stops strictly before max_new_tokens when
   *     B finished hypotheses exist and upper bound cannot beat B-th score. */
  const et_beam_config_v1 es_cfg =
      make_cfg(2u, 4u, 0u, 10u, 0u, 0u, 1u, 0.0, 1.0);
  const double step0_es[4] = {-5.0, 2.0, 1.5, -5.0};
  const double step1_es[8] = {
      8.0, -5.0, -5.0, -5.0,
      8.0, -5.0, -5.0, -5.0,
  };
  hot_path_active = 1;
  CHECK(et_beam_init_v1(state, &es_cfg, w_prompt, 1u) == ET_BEAM_OK);
  CHECK(et_beam_step_v1(state, step0_es, 1u) == ET_BEAM_OK);
  CHECK(et_beam_is_done_v1(state) == 0u);
  CHECK(et_beam_step_v1(state, step1_es, 2u) == ET_BEAM_OK);
  CHECK(et_beam_is_done_v1(state) == 1u);
  CHECK(et_beam_stopped_early_v1(state) == 1u);
  CHECK(et_beam_steps_done_v1(state) == 2u);
  CHECK(et_beam_finished_count_v1(state) == 2u);
  hot_path_active = 0;

  free(state);
}

/* ---- Unit Test 4: Comprehensive Negative Tests & Sentinel Preservation ---- */

static void test_negative_and_sentinel_preservation(void) {
  et_beam_state_v1 *state =
      (et_beam_state_v1 *)calloc(1u, sizeof(et_beam_state_v1));
  et_beam_state_v1 *snapshot =
      (et_beam_state_v1 *)calloc(1u, sizeof(et_beam_state_v1));
  CHECK(state != NULL && snapshot != NULL);

  const uint32_t prompt[1] = {1u};
  double out[6];
  const double sentinel = 1234.5625;
  for (size_t i = 0u; i < 6u; ++i)
    out[i] = sentinel;

  hot_path_active = 1;

  /* Invalid configs */
  et_beam_config_v1 bad = make_cfg(0u, 8u, 0u, 4u, 0u, 0u, 1u, 0.0, 1.0);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_INVALID_ARG);
  bad = make_cfg(9u, 8u, 0u, 4u, 0u, 0u, 1u, 0.0, 1.0);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_LIMIT);
  bad = make_cfg(2u, 1u, 0u, 4u, 0u, 0u, 1u, 0.0, 1.0);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_INVALID_ARG);
  bad = make_cfg(2u, 4097u, 0u, 4u, 0u, 0u, 1u, 0.0, 1.0);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_LIMIT);
  bad = make_cfg(2u, 8u, 8u, 4u, 0u, 0u, 1u, 0.0, 1.0);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_INVALID_ARG);
  bad = make_cfg(2u, 8u, 0u, 4u, 5u, 0u, 1u, 0.0, 1.0);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_INVALID_ARG);
  bad = make_cfg(2u, 8u, 0u, 4u, 0u, 1u, 1u, 0.0, 1.0);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_INVALID_ARG);
  bad = make_cfg(2u, 8u, 0u, 4u, 0u, 257u, 1u, 0.0, 1.0);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_LIMIT);
  bad = make_cfg(2u, 8u, 0u, 4u, 0u, 0u, 2u, 0.0, 1.0);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_INVALID_ARG);
  bad = make_cfg(2u, 8u, 0u, 4u, 0u, 0u, 1u, -0.1, 1.0);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_INVALID_ARG);
  bad = make_cfg(2u, 8u, 0u, 4u, 0u, 0u, 1u, 0.0, 0.99);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_INVALID_ARG);
  bad = make_cfg(2u, 8u, 0u, 4u, 0u, 0u, 1u, (double)NAN, 1.0);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_NONFINITE);
  bad = make_cfg(2u, 8u, 0u, 4u, 0u, 0u, 1u, 0.0, (double)INFINITY);
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_NONFINITE);
  bad = make_cfg(2u, 8u, 0u, 4u, 0u, 0u, 1u, 0.0, 1.0);
  bad.reserved = 1u;
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_INVALID_ARG);
  bad.reserved = 0u;
  bad.struct_size = sizeof(et_beam_config_v1) - 1u;
  CHECK(et_beam_validate_config_v1(&bad, 1u) == ET_BEAM_ERR_INVALID_ARG);

  /* All-masked logits -> ET_BEAM_ERR_NO_VALID_TOKENS + sentinel preserved. */
  const double all_masked[6] = {-INFINITY, -INFINITY, -INFINITY,
                                -INFINITY, -INFINITY, -INFINITY};
  CHECK(et_beam_log_softmax_v1(6u, all_masked, out) ==
        ET_BEAM_ERR_NO_VALID_TOKENS);
  CHECK(et_beam_apply_repetition_penalty_v1(prompt, 1u, 6u, 1.2, all_masked,
                                            out) ==
        ET_BEAM_ERR_NO_VALID_TOKENS);
  CHECK(et_beam_apply_no_repeat_ngram_v1(prompt, 1u, 6u, 2u, all_masked, out) ==
        ET_BEAM_ERR_NO_VALID_TOKENS);
  CHECK(et_beam_apply_min_length_v1(0u, 1u, 0u, 6u, all_masked, out) ==
        ET_BEAM_ERR_NO_VALID_TOKENS);

  /* Non-finite logits (NaN, +Inf) -> ET_BEAM_ERR_NONFINITE + sentinel intact. */
  const double nan_logits[6] = {1.0, (double)NAN, 2.0, 3.0, 4.0, 5.0};
  const double inf_logits[6] = {1.0, 2.0, (double)INFINITY, 3.0, 4.0, 5.0};
  CHECK(et_beam_log_softmax_v1(6u, nan_logits, out) == ET_BEAM_ERR_NONFINITE);
  CHECK(et_beam_log_softmax_v1(6u, inf_logits, out) == ET_BEAM_ERR_NONFINITE);
  CHECK(et_beam_apply_repetition_penalty_v1(prompt, 1u, 6u, 1.5, nan_logits,
                                            out) == ET_BEAM_ERR_NONFINITE);
  CHECK(et_beam_apply_no_repeat_ngram_v1(prompt, 1u, 6u, 2u, inf_logits, out) ==
        ET_BEAM_ERR_NONFINITE);
  CHECK(et_beam_apply_min_length_v1(0u, 2u, 0u, 6u, nan_logits, out) ==
        ET_BEAM_ERR_NONFINITE);

  /* Partial aliasing -> ET_BEAM_ERR_ALIASING + sentinel intact. */
  double overlap_buf[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  CHECK(et_beam_log_softmax_v1(4u, overlap_buf, overlap_buf + 1) ==
        ET_BEAM_ERR_ALIASING);
  CHECK(et_beam_apply_repetition_penalty_v1(prompt, 1u, 4u, 1.2, overlap_buf,
                                            overlap_buf + 2) ==
        ET_BEAM_ERR_ALIASING);
  CHECK(et_beam_apply_no_repeat_ngram_v1(prompt, 1u, 4u, 2u, overlap_buf,
                                         overlap_buf + 1) ==
        ET_BEAM_ERR_ALIASING);
  CHECK(et_beam_apply_min_length_v1(0u, 1u, 0u, 4u, overlap_buf,
                                    overlap_buf + 1) == ET_BEAM_ERR_ALIASING);

  for (size_t i = 0u; i < 6u; ++i)
    CHECK(f64_bits(out[i]) == f64_bits(sentinel));

  /* Processor-induced all-masked in step: V=2, eos=0, min_new=2, ngram=2,
   * prompt=[1, 1] -> tok 1 blocked by ngram=2, tok 0 blocked by min_new=2! */
  const et_beam_config_v1 trap_mask_cfg =
      make_cfg(1u, 2u, 0u, 2u, 2u, 2u, 1u, 0.0, 1.0);
  const uint32_t p11[2] = {1u, 1u};
  CHECK(et_beam_init_v1(state, &trap_mask_cfg, p11, 2u) == ET_BEAM_OK);
  memcpy(snapshot, state, sizeof(*state));

  const double finite2[2] = {1.0, 2.0};
  CHECK(et_beam_step_v1(state, finite2, 1u) == ET_BEAM_ERR_NO_VALID_TOKENS);
  /* Verify committed state fields are completely preserved after error! */
  CHECK(state->steps_done == snapshot->steps_done);
  CHECK(state->active_count == snapshot->active_count);
  CHECK(state->finished_count == snapshot->finished_count);
  CHECK(state->done == snapshot->done);
  CHECK(memcmp(state->active_tokens, snapshot->active_tokens,
               sizeof(state->active_tokens)) == 0);

  /* Finalize before done -> ET_BEAM_ERR_STATE, output untouched. */
  et_beam_hypothesis_v1 h_sent[1];
  memset(h_sent, 0xA5, sizeof(h_sent));
  size_t c_sent = 999u;
  CHECK(et_beam_finalize_v1(state, h_sent, 1u, &c_sent) == ET_BEAM_ERR_STATE);
  CHECK(c_sent == 999u);
  CHECK(h_sent[0].seq_len == 0xA5A5A5A5u);

  /* Non-default FP rounding mode -> ET_BEAM_ERR_FP_ENV. */
  const int saved_round = fegetround();
  CHECK(fesetround(FE_UPWARD) == 0);
  CHECK(et_beam_log_softmax_v1(2u, finite2, out) == ET_BEAM_ERR_FP_ENV);
  CHECK(et_beam_step_v1(state, finite2, 1u) == ET_BEAM_ERR_FP_ENV);
  CHECK(fesetround(saved_round) == 0);

  hot_path_active = 0;

  free(snapshot);
  free(state);
}

/* ---- Oracle Dump Mode (for Independent Python Binary64 Oracle) ---- */

static void dump_hyp_line(const char *tag, size_t rank,
                          const et_beam_hypothesis_v1 *h) {
  printf("%s rank=%zu finished=%u seq_len=%u gen_len=%u sum_lp=%.17g "
         "score=%.17g tokens=",
         tag, rank, h->finished, h->seq_len, h->generated_len, h->sum_log_prob,
         h->score);
  for (uint32_t i = 0u; i < h->seq_len; ++i) {
    printf("%s%u", i == 0u ? "" : ",", h->tokens[i]);
  }
  printf("\n");
}

static void run_oracle_dump(void) {
  et_beam_state_v1 *state =
      (et_beam_state_v1 *)calloc(1u, sizeof(et_beam_state_v1));
  CHECK(state != NULL);

  /* 1. Processor oracle vectors. */
  const uint32_t hist[5] = {2u, 5u, 2u, 0u, 7u};
  const double raw[8] = {1.5, -2.25, 3.75, 0.0, -0.0, -4.5, 2.0, 0.25};
  double p1[8], p2[8], p3[8], lsm[8];
  const uint32_t ng_seq[7] = {1u, 2u, 3u, 4u, 1u, 2u, 3u};

  hot_path_active = 1;
  CHECK(et_beam_apply_repetition_penalty_v1(hist, 5u, 8u, 1.35, raw, p1) ==
        ET_BEAM_OK);
  CHECK(et_beam_apply_no_repeat_ngram_v1(ng_seq, 7u, 8u, 3u, p1, p2) ==
        ET_BEAM_OK);
  CHECK(et_beam_apply_min_length_v1(1u, 3u, 0u, 8u, p2, p3) == ET_BEAM_OK);
  CHECK(et_beam_log_softmax_v1(8u, p3, lsm) == ET_BEAM_OK);
  hot_path_active = 0;

  printf("PROC_REP");
  for (size_t i = 0u; i < 8u; ++i)
    printf(" %.17g", p1[i]);
  printf("\nPROC_NGRAM");
  for (size_t i = 0u; i < 8u; ++i)
    printf(" %.17g", p2[i]);
  printf("\nPROC_MINLEN");
  for (size_t i = 0u; i < 8u; ++i)
    printf(" %.17g", p3[i]);
  printf("\nPROC_LSM");
  for (size_t i = 0u; i < 8u; ++i)
    printf(" %.17g", lsm[i]);
  printf("\n");

  /* 2. Trap tree B=1 and B=4. */
  double logits[ET_BEAM_MAX_BEAMS * 16u];
  uint32_t seq_buf[ET_BEAM_MAX_SEQ_LEN];
  const uint32_t p1_tok[1] = {1u};
  const uint32_t trap_bs[2] = {1u, 4u};
  for (size_t bi = 0u; bi < 2u; ++bi) {
    const uint32_t B = trap_bs[bi];
    const et_beam_config_v1 cfg =
        make_cfg(B, 8u, 0u, 4u, 0u, 0u, 1u, 0.0, 1.0);
    hot_path_active = 1;
    CHECK(et_beam_init_v1(state, &cfg, p1_tok, 1u) == ET_BEAM_OK);
    while (!et_beam_is_done_v1(state)) {
      const uint32_t na = et_beam_active_count_v1(state);
      for (uint32_t b = 0u; b < na; ++b) {
        size_t slen = 0u;
        double lp = 0.0;
        CHECK(et_beam_get_active_beam_v1(state, b, seq_buf, ET_BEAM_MAX_SEQ_LEN,
                                         &slen, &lp) == ET_BEAM_OK);
        trap_tree_logits(seq_buf, slen, logits + (size_t)b * 8u);
      }
      CHECK(et_beam_step_v1(state, logits, na) == ET_BEAM_OK);
    }
    et_beam_hypothesis_v1 hyps[8];
    size_t cnt = 0u;
    CHECK(et_beam_finalize_v1(state, hyps, B, &cnt) == ET_BEAM_OK);
    hot_path_active = 0;

    const char *tag = (B == 1u) ? "TRAP_B1" : "TRAP_B4";
    printf("%s_META steps=%u early=%u count=%zu\n", tag,
           et_beam_steps_done_v1(state), et_beam_stopped_early_v1(state), cnt);
    for (size_t i = 0u; i < cnt; ++i)
      dump_hyp_line(tag, i, &hyps[i]);
  }

  /* 3. Wu length penalty shift (alpha=0.0 vs alpha=1.5). */
  const double alphas[2] = {0.0, 1.5};
  for (size_t ai = 0u; ai < 2u; ++ai) {
    const et_beam_config_v1 cfg =
        make_cfg(4u, 6u, 0u, 5u, 0u, 0u, 0u, alphas[ai], 1.0);
    hot_path_active = 1;
    CHECK(et_beam_init_v1(state, &cfg, p1_tok, 1u) == ET_BEAM_OK);
    while (!et_beam_is_done_v1(state)) {
      const uint32_t na = et_beam_active_count_v1(state);
      for (uint32_t b = 0u; b < na; ++b) {
        size_t slen = 0u;
        double lp = 0.0;
        CHECK(et_beam_get_active_beam_v1(state, b, seq_buf, ET_BEAM_MAX_SEQ_LEN,
                                         &slen, &lp) == ET_BEAM_OK);
        wu_shift_logits(seq_buf, slen, logits + (size_t)b * 6u);
      }
      CHECK(et_beam_step_v1(state, logits, na) == ET_BEAM_OK);
    }
    et_beam_hypothesis_v1 hyps[4];
    size_t cnt = 0u;
    CHECK(et_beam_finalize_v1(state, hyps, 4u, &cnt) == ET_BEAM_OK);
    hot_path_active = 0;
    const char *tag = (ai == 0u) ? "WU_A00" : "WU_A15";
    printf("%s_META steps=%u early=%u count=%zu\n", tag,
           et_beam_steps_done_v1(state), et_beam_stopped_early_v1(state), cnt);
    for (size_t i = 0u; i < cnt; ++i)
      dump_hyp_line(tag, i, &hyps[i]);
  }

  /* 4. Deterministic hash-model stress across (B, ngram, min_new, lp, rp, es). */
  typedef struct stress_case {
    const char *name;
    uint64_t seed;
    uint32_t B;
    uint32_t V;
    uint32_t max_new;
    uint32_t min_new;
    uint32_t ngram;
    uint32_t early;
    double lp;
    double rp;
  } stress_case;

  const stress_case cases[6] = {
      {"STRESS_0", 101u, 1u, 16u, 8u, 0u, 0u, 1u, 0.0, 1.0},
      {"STRESS_1", 202u, 4u, 16u, 10u, 2u, 2u, 1u, 0.6, 1.25},
      {"STRESS_2", 303u, 6u, 16u, 12u, 3u, 3u, 0u, 1.0, 1.5},
      {"STRESS_3", 404u, 8u, 16u, 10u, 1u, 2u, 1u, 1.4, 1.15},
      {"STRESS_4", 505u, 4u, 16u, 9u, 0u, 0u, 1u, 0.0, 1.0},
      {"STRESS_5", 606u, 3u, 16u, 8u, 4u, 3u, 1u, 2.0, 1.8},
  };
  const uint32_t s_prompt[3] = {2u, 5u, 7u};
  for (size_t ci = 0u; ci < 6u; ++ci) {
    const stress_case *sc = &cases[ci];
    const et_beam_config_v1 cfg =
        make_cfg(sc->B, sc->V, 0u, sc->max_new, sc->min_new, sc->ngram,
                 sc->early, sc->lp, sc->rp);
    hot_path_active = 1;
    CHECK(et_beam_init_v1(state, &cfg, s_prompt, 3u) == ET_BEAM_OK);
    while (!et_beam_is_done_v1(state)) {
      const uint32_t na = et_beam_active_count_v1(state);
      for (uint32_t b = 0u; b < na; ++b) {
        size_t slen = 0u;
        double lp = 0.0;
        CHECK(et_beam_get_active_beam_v1(state, b, seq_buf, ET_BEAM_MAX_SEQ_LEN,
                                         &slen, &lp) == ET_BEAM_OK);
        hash_model_logits(sc->seed, seq_buf, slen, sc->V,
                          logits + (size_t)b * sc->V);
      }
      CHECK(et_beam_step_v1(state, logits, na) == ET_BEAM_OK);
    }
    et_beam_hypothesis_v1 hyps[8];
    size_t cnt = 0u;
    CHECK(et_beam_finalize_v1(state, hyps, sc->B, &cnt) == ET_BEAM_OK);
    hot_path_active = 0;

    printf("%s_META steps=%u early=%u count=%zu\n", sc->name,
           et_beam_steps_done_v1(state), et_beam_stopped_early_v1(state), cnt);
    for (size_t i = 0u; i < cnt; ++i)
      dump_hyp_line(sc->name, i, &hyps[i]);
  }

  /* 5. Negative fail-closed status dump for Python oracle verification. */
  const double all_mask[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
  const double nan_vec[4] = {1.0, (double)NAN, 2.0, 3.0};
  const double inf_vec[4] = {1.0, 2.0, (double)INFINITY, 3.0};
  double neg_out[4] = {42.5, 42.5, 42.5, 42.5};
  const uint32_t neg_prompt[1] = {1u};
  const et_beam_config_v1 bad_b = make_cfg(0u, 8u, 0u, 4u, 0u, 0u, 1u, 0.0, 1.0);
  const et_beam_config_v1 bad_bmax =
      make_cfg(9u, 8u, 0u, 4u, 0u, 0u, 1u, 0.0, 1.0);
  const et_beam_config_v1 bad_v = make_cfg(2u, 1u, 0u, 4u, 0u, 0u, 1u, 0.0, 1.0);
  const et_beam_config_v1 bad_ng =
      make_cfg(2u, 8u, 0u, 4u, 0u, 1u, 1u, 0.0, 1.0);
  const et_beam_config_v1 bad_rp =
      make_cfg(2u, 8u, 0u, 4u, 0u, 0u, 1u, 0.0, 0.9);
  const et_beam_config_v1 bad_mm =
      make_cfg(2u, 8u, 0u, 3u, 4u, 0u, 1u, 0.0, 1.0);
  hot_path_active = 1;
  const int32_t st_all_mask = et_beam_log_softmax_v1(4u, all_mask, neg_out);
  const int32_t st_nan = et_beam_log_softmax_v1(4u, nan_vec, neg_out);
  const int32_t st_inf = et_beam_apply_min_length_v1(0u, 1u, 0u, 4u, inf_vec,
                                                     neg_out);
  const int32_t st_alias = et_beam_log_softmax_v1(2u, neg_out, neg_out + 1);
  const int32_t st_b0 = et_beam_init_v1(state, &bad_b, neg_prompt, 1u);
  const int32_t st_b9 = et_beam_init_v1(state, &bad_bmax, neg_prompt, 1u);
  const int32_t st_v1 = et_beam_init_v1(state, &bad_v, neg_prompt, 1u);
  const int32_t st_ng1 = et_beam_init_v1(state, &bad_ng, neg_prompt, 1u);
  const int32_t st_rp = et_beam_init_v1(state, &bad_rp, neg_prompt, 1u);
  const int32_t st_mm = et_beam_init_v1(state, &bad_mm, neg_prompt, 1u);
  hot_path_active = 0;
  printf("NEG all_mask=%d nan=%d inf=%d alias=%d b0=%d b9=%d v1=%d ng1=%d "
         "rp=%d mm=%d sentinel=%.17g\n",
         st_all_mask, st_nan, st_inf, st_alias, st_b0, st_b9, st_v1, st_ng1,
         st_rp, st_mm, neg_out[0]);

  free(state);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    run_oracle_dump();
    return 0;
  }
  test_logit_processors();
  test_trap_tree_and_greedy_equivalence();
  test_ngram_minlen_wu_and_ties();
  test_negative_and_sentinel_preservation();
  printf("BEAM_SEARCH native C11 zero-alloc & negative tests: PASS\n");
  return 0;
}
