#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/spec_decode_abi.h"
#include "spec_decode_eshkol_bridge.h"

#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if defined(ET_SPEC_WRAP_ALLOC)
static int g_forbid_alloc = 0;

void *__real_malloc(size_t n);
void *__real_calloc(size_t nmemb, size_t size);
void *__real_realloc(void *ptr, size_t size);
void __real_free(void *ptr);

static void alloc_trap(const char *name) {
  fprintf(stderr, "FAIL: forbidden heap call during verify/sample: %s\n", name);
  abort();
}

void *__wrap_malloc(size_t n) {
  if (g_forbid_alloc)
    alloc_trap("malloc");
  return __real_malloc(n);
}

void *__wrap_calloc(size_t nmemb, size_t size) {
  if (g_forbid_alloc)
    alloc_trap("calloc");
  return __real_calloc(nmemb, size);
}

void *__wrap_realloc(void *ptr, size_t size) {
  if (g_forbid_alloc)
    alloc_trap("realloc");
  return __real_realloc(ptr, size);
}

void __wrap_free(void *ptr) {
  if (g_forbid_alloc)
    alloc_trap("free");
  __real_free(ptr);
}
#define FORBID_ALLOC_BEGIN() (g_forbid_alloc = 1)
#define FORBID_ALLOC_END() (g_forbid_alloc = 0)
#else
#define FORBID_ALLOC_BEGIN() ((void)0)
#define FORBID_ALLOC_END() ((void)0)
#endif

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL at %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static uint32_t argmax_ref(const float *logits, uint32_t v) {
  uint32_t best = 0u;
  double best_val = (double)logits[0];
  for (uint32_t i = 1u; i < v; ++i) {
    if ((double)logits[i] > best_val) {
      best_val = (double)logits[i];
      best = i;
    }
  }
  return best;
}

/* Deterministic synthetic autoregressive models for 32-token rollouts. */
static void target_model_logits(uint32_t pos, uint32_t prev_tok, uint32_t v,
                                float *out) {
  const uint32_t winner = (pos * 7u + prev_tok * 3u + 5u) % v;
  for (uint32_t i = 0u; i < v; ++i) {
    const int32_t h = (int32_t)(((pos * 13u + prev_tok * 17u + i * 29u) % 23u)) - 11;
    out[i] = (float)h * 0.15f;
  }
  out[winner] += 4.5f;
}

/* draft_mode: 0 = identical, 1 = partial (~65% match), 2 = completely wrong */
static void draft_model_logits(uint32_t pos, uint32_t prev_tok, uint32_t v,
                               int draft_mode, float *out) {
  target_model_logits(pos, prev_tok, v, out);
  if (draft_mode == 0)
    return;
  const uint32_t target_winner = argmax_ref(out, v);
  if (draft_mode == 1) {
    if ((pos % 3u) == 2u) {
      const uint32_t wrong = (target_winner + 1u) % v;
      out[wrong] = out[target_winner] + 2.0f;
    }
    return;
  }
  /* draft_mode == 2: always disagree with target argmax */
  const uint32_t wrong = (target_winner + 1u) % v;
  out[wrong] = out[target_winner] + 3.0f;
}

static void test_greedy_32_token_rollouts(void) {
  const uint32_t V = 32u;
  const uint32_t ROLLOUT_LEN = 32u;
  const uint32_t horizons[3] = {1u, 4u, 16u};

  /* Reference target-only greedy rollout. */
  uint32_t ref_tokens[32];
  uint32_t prev = 3u;
  float row[32];
  for (uint32_t t = 0u; t < ROLLOUT_LEN; ++t) {
    target_model_logits(t, prev, V, row);
    ref_tokens[t] = argmax_ref(row, V);
    prev = ref_tokens[t];
  }

  for (size_t hi = 0u; hi < 3u; ++hi) {
    const uint32_t K = horizons[hi];
    for (int mode = 0; mode < 3; ++mode) {
      uint32_t spec_tokens[64];
      uint32_t emitted_total = 0u;
      uint32_t target_calls = 0u;
      uint32_t accepted_total = 0u;
      uint32_t cur_prev = 3u;

      uint32_t draft_toks[ET_SPEC_MAX_HORIZON];
      float target_rows[(ET_SPEC_MAX_HORIZON + 1u) * 32u];
      float draft_row[32];
      uint32_t out_buf[ET_SPEC_MAX_HORIZON + 1u];
      et_spec_verify_result_v1 res;

      while (emitted_total < ROLLOUT_LEN) {
        /* Draft K tokens autoregressively. */
        uint32_t d_prev = cur_prev;
        for (uint32_t i = 0u; i < K; ++i) {
          draft_model_logits(emitted_total + i, d_prev, V, mode, draft_row);
          draft_toks[i] = argmax_ref(draft_row, V);
          d_prev = draft_toks[i];
        }
        /* Evaluate target model on K+1 positions in one batch call. */
        target_model_logits(emitted_total, cur_prev, V, target_rows);
        for (uint32_t i = 0u; i < K; ++i) {
          target_model_logits(emitted_total + i + 1u, draft_toks[i], V,
                              target_rows + (size_t)(i + 1u) * V);
        }
        target_calls++;

        FORBID_ALLOC_BEGIN();
        const et_spec_status st = et_spec_verify_greedy_v1(
            draft_toks, target_rows, K, V, ET_SPEC_NO_EOS, out_buf, K + 1u,
            &res);
        FORBID_ALLOC_END();
        CHECK(st == ET_SPEC_OK);
        CHECK(res.emitted_count >= 1u && res.emitted_count <= K + 1u);
        CHECK(res.eos_hit == 0u);
        CHECK(res.rng_draws == 0u);

        if (mode == 0) {
          CHECK(res.accepted_count == K);
          CHECK(res.emitted_count == K + 1u);
          CHECK(res.rejected == 0u);
        } else if (mode == 2) {
          CHECK(res.accepted_count == 0u);
          CHECK(res.emitted_count == 1u);
          CHECK(res.rejected == 1u);
        }

        accepted_total += res.accepted_count;
        for (uint32_t j = 0u; j < res.emitted_count && emitted_total < ROLLOUT_LEN;
             ++j) {
          spec_tokens[emitted_total++] = out_buf[j];
          cur_prev = out_buf[j];
        }
      }

      CHECK(memcmp(spec_tokens, ref_tokens, sizeof(ref_tokens)) == 0);
      if (mode == 0 && K > 1u) {
        CHECK(target_calls < ROLLOUT_LEN);
        CHECK(accepted_total >= ROLLOUT_LEN / 2u);
      }
      if (mode == 1 && K >= 4u) {
        CHECK(target_calls < ROLLOUT_LEN);
        CHECK(accepted_total > 0u);
      }
      if (mode == 2) {
        CHECK(target_calls == ROLLOUT_LEN);
        CHECK(accepted_total == 0u);
      }
    }
  }

  /* Lowest-index tie-breaking including -0.0f vs +0.0f. */
  const float tied_logits[2 * 4] = {
      -0.0f, +0.0f, -1.0f, +0.0f,
      2.5f,  2.5f,  2.5f,  2.5f,
  };
  const uint32_t tied_draft[1] = {0u};
  uint32_t out_tok[2] = {99u, 99u};
  et_spec_verify_result_v1 res;
  CHECK(et_spec_verify_greedy_v1(tied_draft, tied_logits, 1u, 4u,
                                 ET_SPEC_NO_EOS, out_tok, 2u, &res) ==
        ET_SPEC_OK);
  CHECK(res.accepted_count == 1u && res.emitted_count == 2u);
  CHECK(out_tok[0] == 0u && out_tok[1] == 0u);
}

static void test_eos_early_termination(void) {
  const uint32_t K = 4u;
  const uint32_t V = 8u;
  const uint32_t EOS = 7u;
  float target[(4 + 1) * 8];
  float draft[4 * 8];
  memset(target, 0, sizeof(target));
  memset(draft, 0, sizeof(draft));
  for (uint32_t r = 0u; r <= K; ++r) {
    target[r * V + ((r + 1u) % 6u)] = 5.0f;
    if (r < K)
      draft[r * V + ((r + 1u) % 6u)] = 5.0f;
  }

  /* Case 1: Greedy EOS at draft pos 0 (accepted). */
  target[0u * V + EOS] = 10.0f;
  const uint32_t d_eos0[4] = {EOS, 2u, 3u, 4u};
  uint32_t out_tok[5] = {0xAAu, 0xAAu, 0xAAu, 0xAAu, 0xAAu};
  et_spec_verify_result_v1 res;
  CHECK(et_spec_verify_greedy_v1(d_eos0, target, K, V, EOS, out_tok, 5u, &res) ==
        ET_SPEC_OK);
  CHECK(res.eos_hit == 1u && res.accepted_count == 1u &&
        res.emitted_count == 1u && res.rejected == 0u);
  CHECK(out_tok[0] == EOS);
  CHECK(out_tok[1] == 0xAAu); /* Trailing slots untouched */

  /* Case 2: Greedy EOS at correction token (pos 1 rejected -> target emits EOS). */
  target[0u * V + EOS] = 0.0f;
  target[1u * V + EOS] = 10.0f;
  const uint32_t d_corr[4] = {1u, 2u, 3u, 4u};
  CHECK(et_spec_verify_greedy_v1(d_corr, target, K, V, EOS, out_tok, 5u, &res) ==
        ET_SPEC_OK);
  CHECK(res.eos_hit == 1u && res.accepted_count == 1u &&
        res.emitted_count == 2u && res.rejected == 1u);
  CHECK(out_tok[0] == 1u && out_tok[1] == EOS);

  /* Case 3: Greedy EOS at bonus token (pos K). */
  target[1u * V + EOS] = 0.0f;
  target[4u * V + EOS] = 10.0f;
  const uint32_t d_bonus[4] = {1u, 2u, 3u, 4u};
  CHECK(et_spec_verify_greedy_v1(d_bonus, target, K, V, EOS, out_tok, 5u,
                                 &res) == ET_SPEC_OK);
  CHECK(res.eos_hit == 1u && res.accepted_count == 4u &&
        res.emitted_count == 5u && res.rejected == 0u);
  CHECK(out_tok[4] == EOS);

  /* Case 4: Stochastic EOS at accepted pos 0 (p == q with high logit on EOS). */
  memset(target, 0, sizeof(target));
  memset(draft, 0, sizeof(draft));
  target[0u * V + EOS] = 20.0f;
  draft[0u * V + EOS] = 20.0f;
  const et_spec_sampling_params_v1 params = {1.0, 1.0, V, EOS};
  const et_spec_rng_state_v1 rng = {UINT64_C(777), UINT64_C(10)};
  CHECK(et_spec_verify_sample_v1(d_eos0, draft, target, K, V, &params, &rng,
                                 out_tok, 5u, &res) == ET_SPEC_OK);
  CHECK(res.eos_hit == 1u && res.accepted_count == 1u &&
        res.emitted_count == 1u && res.rng_draws == 1u);
  CHECK(out_tok[0] == EOS);
  CHECK(res.next_rng.counter == 11u);
}

static void test_stochastic_edge_cases_and_replay(void) {
  const uint32_t V = 8u;
  const uint32_t K = 4u;

  /* 1. p == q -> 100% acceptance across 1,000 seeds/counters for K=1, 4, 16. */
  const uint32_t horizons[3] = {1u, 4u, 16u};
  for (size_t hi = 0u; hi < 3u; ++hi) {
    const uint32_t kh = horizons[hi];
    float draft_logits[16 * 8];
    float target_logits[17 * 8];
    for (uint32_t r = 0u; r <= kh; ++r) {
      for (uint32_t v = 0u; v < V; ++v) {
        const float val = 0.25f * (float)((r * 5u + v * 3u) % 9u) - 1.0f;
        target_logits[r * V + v] = val;
        if (r < kh)
          draft_logits[r * V + v] = val;
      }
    }
    const et_spec_sampling_params_v1 params = {0.85, 0.9, 6u, ET_SPEC_NO_EOS};
    et_spec_rng_state_v1 rng = {UINT64_C(0x123456789abcdef0), UINT64_C(0)};
    for (uint32_t trial = 0u; trial < 300u; ++trial) {
      uint32_t d_toks[16];
      for (uint32_t i = 0u; i < kh; ++i) {
        et_spec_rng_state_v1 next_r;
        CHECK(et_spec_sample_token_v1(draft_logits + i * V, V, &params, &rng,
                                      &d_toks[i], &next_r) == ET_SPEC_OK);
        rng = next_r;
      }
      uint32_t out_toks[17];
      et_spec_verify_result_v1 res;
      FORBID_ALLOC_BEGIN();
      const et_spec_status st = et_spec_verify_sample_v1(
          d_toks, draft_logits, target_logits, kh, V, &params, &rng, out_toks,
          kh + 1u, &res);
      FORBID_ALLOC_END();
      CHECK(st == ET_SPEC_OK);
      CHECK(res.accepted_count == kh);
      CHECK(res.emitted_count == kh + 1u);
      CHECK(res.rejected == 0u);
      CHECK(res.rng_draws == (uint64_t)(kh + 1u));
      for (uint32_t i = 0u; i < kh; ++i)
        CHECK(out_toks[i] == d_toks[i]);
      rng = res.next_rng;
    }
  }

  /* 2. q(x) == 0 edge case:
   * - Subcase 2a: q(x) == 0 and p(x) > 0 -> accepted with probability 1, no div/0.
   */
  {
    const float q_logits[8] = {10.0f, 9.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f};
    const float p_logits[2 * 8] = {
        -5.0f, -5.0f, 10.0f, 9.0f, -5.0f, -5.0f, -5.0f, -5.0f,
        0.0f,  1.0f,  2.0f,  3.0f, 0.0f,  0.0f,  0.0f,  0.0f,
    };
    /* With top_k = 2: q has support {0, 1} (so q[2] == 0.0!), p has support {2, 3}. */
    const et_spec_sampling_params_v1 params = {1.0, 1.0, 2u, ET_SPEC_NO_EOS};
    double q_probs[8], p_probs[8];
    CHECK(et_spec_logits_to_probs_v1(q_logits, V, &params, q_probs) == ET_SPEC_OK);
    CHECK(et_spec_logits_to_probs_v1(p_logits, V, &params, p_probs) == ET_SPEC_OK);
    CHECK(q_probs[2] == 0.0 && p_probs[2] > 0.0);

    const uint32_t d_tok[1] = {2u};
    for (uint64_t c = 0u; c < 64u; ++c) {
      const et_spec_rng_state_v1 rng = {UINT64_C(99), c};
      uint32_t out_t[2];
      et_spec_verify_result_v1 res;
      CHECK(et_spec_verify_sample_v1(d_tok, q_logits, p_logits, 1u, V, &params,
                                     &rng, out_t, 2u, &res) == ET_SPEC_OK);
      CHECK(res.accepted_count == 1u && res.emitted_count == 2u &&
            out_t[0] == 2u);
    }

    /* Subcase 2b: q(x) > 0 and p(x) == 0 -> ALWAYS rejected, residual in {2, 3}. */
    const uint32_t d_tok_zero_p[1] = {0u};
    CHECK(q_probs[0] > 0.0 && p_probs[0] == 0.0);
    for (uint64_t c = 0u; c < 200u; ++c) {
      const et_spec_rng_state_v1 rng = {UINT64_C(12345), c};
      uint32_t out_t[2];
      et_spec_verify_result_v1 res;
      CHECK(et_spec_verify_sample_v1(d_tok_zero_p, q_logits, p_logits, 1u, V,
                                     &params, &rng, out_t, 2u, &res) ==
            ET_SPEC_OK);
      CHECK(res.accepted_count == 0u && res.rejected == 1u &&
            res.emitted_count == 1u);
      CHECK(out_t[0] == 2u || out_t[0] == 3u);
    }

    /* Subcase 2c: p == q and adversarial x with q(x) == p(x) == 0 -> rejected,
     * residual_sum == 0 boundary cleanly samples from p (support {0, 1}). */
    const float pq_target[2 * 8] = {
        10.0f, 9.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f,
        10.0f, 9.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f,
    };
    const uint32_t d_tok_both_zero[1] = {7u};
    const et_spec_rng_state_v1 rng = {UINT64_C(555), UINT64_C(0)};
    uint32_t out_t[2];
    et_spec_verify_result_v1 res;
    CHECK(et_spec_verify_sample_v1(d_tok_both_zero, q_logits, pq_target, 1u, V,
                                   &params, &rng, out_t, 2u, &res) ==
          ET_SPEC_OK);
    CHECK(res.accepted_count == 0u && res.rejected == 1u &&
          res.emitted_count == 1u);
    CHECK(out_t[0] == 0u || out_t[0] == 1u);
  }

  /* 3. Exact seeded replay and temperature==0 greedy parity in verify_sample. */
  {
    float d_logits[4 * 8];
    float t_logits[5 * 8];
    uint32_t d_toks[4] = {1u, 3u, 5u, 7u};
    for (uint32_t r = 0u; r < 5u; ++r) {
      for (uint32_t v = 0u; v < V; ++v) {
        t_logits[r * V + v] = 0.3f * (float)((r * 7u + v * 5u) % 11u) - 1.5f;
        if (r < 4u)
          d_logits[r * V + v] = 0.2f * (float)((r * 3u + v * 9u) % 11u) - 1.0f;
      }
    }
    const et_spec_sampling_params_v1 params = {0.9, 0.85, 5u, ET_SPEC_NO_EOS};
    const et_spec_rng_state_v1 rng = {UINT64_C(0xdeadbeefcafebabe),
                                      UINT64_C(42)};
    uint32_t out1[5], out2[5];
    et_spec_verify_result_v1 res1, res2;
    CHECK(et_spec_verify_sample_v1(d_toks, d_logits, t_logits, K, V, &params,
                                   &rng, out1, 5u, &res1) == ET_SPEC_OK);
    CHECK(et_spec_verify_sample_v1(d_toks, d_logits, t_logits, K, V, &params,
                                   &rng, out2, 5u, &res2) == ET_SPEC_OK);
    CHECK(memcmp(&res1, &res2, sizeof(res1)) == 0);
    CHECK(memcmp(out1, out2, res1.emitted_count * sizeof(uint32_t)) == 0);

    /* temperature == 0 in verify_sample matches verify_greedy identically. */
    const et_spec_sampling_params_v1 gparams = {0.0, 1.0, V, ET_SPEC_NO_EOS};
    uint32_t out_g[5], out_sg[5];
    et_spec_verify_result_v1 res_g, res_sg;
    CHECK(et_spec_verify_greedy_v1(d_toks, t_logits, K, V, ET_SPEC_NO_EOS,
                                   out_g, 5u, &res_g) == ET_SPEC_OK);
    CHECK(et_spec_verify_sample_v1(d_toks, d_logits, t_logits, K, V, &gparams,
                                   &rng, out_sg, 5u, &res_sg) == ET_SPEC_OK);
    CHECK(res_g.accepted_count == res_sg.accepted_count);
    CHECK(res_g.emitted_count == res_sg.emitted_count);
    CHECK(res_g.rejected == res_sg.rejected);
    CHECK(memcmp(out_g, out_sg, res_g.emitted_count * sizeof(uint32_t)) == 0);
  }
}

static void test_max_and_min_bounds(void) {
  /* Min bounds: K=1, V=2 */
  {
    const uint32_t dt[1] = {0u};
    const float dl[2] = {0.5f, -0.5f};
    const float tl[4] = {0.5f, -0.5f, -0.2f, 0.8f};
    uint32_t out_t[2];
    et_spec_verify_result_v1 res;
    const et_spec_sampling_params_v1 params = {1.0, 1.0, 2u, ET_SPEC_NO_EOS};
    const et_spec_rng_state_v1 rng = {1u, 0u};
    CHECK(et_spec_verify_greedy_v1(dt, tl, 1u, 2u, ET_SPEC_NO_EOS, out_t, 2u,
                                   &res) == ET_SPEC_OK);
    CHECK(res.accepted_count == 1u && out_t[0] == 0u && out_t[1] == 1u);
    CHECK(et_spec_verify_sample_v1(dt, dl, tl, 1u, 2u, &params, &rng, out_t, 2u,
                                   &res) == ET_SPEC_OK);
    CHECK(res.emitted_count >= 1u && res.emitted_count <= 2u);
  }

  /* Max bounds: K=16, V=4096 */
  {
    const uint32_t K = ET_SPEC_MAX_HORIZON;
    const uint32_t V = ET_SPEC_MAX_VOCAB;
    uint32_t *dt = (uint32_t *)calloc(K, sizeof(uint32_t));
    float *dl = (float *)calloc((size_t)K * V, sizeof(float));
    float *tl = (float *)calloc((size_t)(K + 1u) * V, sizeof(float));
    CHECK(dt && dl && tl);
    for (uint32_t r = 0u; r < K; ++r) {
      dt[r] = (r * 137u) % V;
      dl[(size_t)r * V + dt[r]] = 6.0f;
      tl[(size_t)r * V + dt[r]] = 6.0f;
    }
    tl[(size_t)K * V + 4095u] = 6.0f;

    uint32_t out_t[ET_SPEC_MAX_HORIZON + 1u];
    et_spec_verify_result_v1 res;
    const et_spec_sampling_params_v1 params = {1.0, 0.95, 64u, ET_SPEC_NO_EOS};
    const et_spec_rng_state_v1 rng = {UINT64_C(2026), UINT64_C(0)};

    FORBID_ALLOC_BEGIN();
    CHECK(et_spec_verify_greedy_v1(dt, tl, K, V, ET_SPEC_NO_EOS, out_t, K + 1u,
                                   &res) == ET_SPEC_OK);
    CHECK(res.accepted_count == K && res.emitted_count == K + 1u &&
          out_t[K] == 4095u);
    CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, V, &params, &rng, out_t,
                                   K + 1u, &res) == ET_SPEC_OK);
    CHECK(res.accepted_count == K && res.emitted_count == K + 1u);
    FORBID_ALLOC_END();

    free(dt);
    free(dl);
    free(tl);
  }
}

static void test_negative_and_output_preservation(void) {
  const uint32_t K = 2u;
  const uint32_t V = 4u;
  uint32_t dt[2] = {0u, 1u};
  float dl[2 * 4] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  float tl[3 * 4] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                     0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  et_spec_sampling_params_v1 params = {1.0, 1.0, V, ET_SPEC_NO_EOS};
  et_spec_rng_state_v1 rng = {UINT64_C(1), UINT64_C(0)};

  uint32_t out_toks[3];
  uint32_t out_toks_ref[3];
  et_spec_verify_result_v1 out_res;
  et_spec_verify_result_v1 out_res_ref;
  double probs[4];
  double probs_ref[4];
  uint32_t sample_tok;
  uint32_t sample_tok_ref;
  et_spec_rng_state_v1 out_rng;
  et_spec_rng_state_v1 out_rng_ref;

  memset(out_toks, 0xA5, sizeof(out_toks));
  memset(out_toks_ref, 0xA5, sizeof(out_toks_ref));
  memset(&out_res, 0x5A, sizeof(out_res));
  memset(&out_res_ref, 0x5A, sizeof(out_res_ref));
  memset(probs, 0x3C, sizeof(probs));
  memset(probs_ref, 0x3C, sizeof(probs_ref));
  memset(&sample_tok, 0xC3, sizeof(sample_tok));
  memset(&sample_tok_ref, 0xC3, sizeof(sample_tok_ref));
  memset(&out_rng, 0x69, sizeof(out_rng));
  memset(&out_rng_ref, 0x69, sizeof(out_rng_ref));

#define ASSERT_PRESERVED()                                                     \
  do {                                                                         \
    CHECK(memcmp(out_toks, out_toks_ref, sizeof(out_toks)) == 0);              \
    CHECK(memcmp(&out_res, &out_res_ref, sizeof(out_res)) == 0);               \
    CHECK(memcmp(probs, probs_ref, sizeof(probs)) == 0);                       \
    CHECK(memcmp(&sample_tok, &sample_tok_ref, sizeof(sample_tok)) == 0);      \
    CHECK(memcmp(&out_rng, &out_rng_ref, sizeof(out_rng)) == 0);               \
  } while (0)

  /* Null pointers */
  CHECK(et_spec_logits_to_probs_v1(NULL, V, &params, probs) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_logits_to_probs_v1(dl, V, NULL, probs) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_logits_to_probs_v1(dl, V, &params, NULL) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_sample_token_v1(NULL, V, &params, &rng, &sample_tok, &out_rng) ==
        ET_SPEC_ERR_NULL);
  CHECK(et_spec_sample_token_v1(dl, V, &params, NULL, &sample_tok, &out_rng) ==
        ET_SPEC_ERR_NULL);
  CHECK(et_spec_sample_token_v1(dl, V, &params, &rng, NULL, &out_rng) ==
        ET_SPEC_ERR_NULL);
  CHECK(et_spec_sample_token_v1(dl, V, &params, &rng, &sample_tok, NULL) ==
        ET_SPEC_ERR_NULL);
  CHECK(et_spec_verify_greedy_v1(NULL, tl, K, V, ET_SPEC_NO_EOS, out_toks, 3u,
                                 &out_res) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_verify_greedy_v1(dt, NULL, K, V, ET_SPEC_NO_EOS, out_toks, 3u,
                                 &out_res) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_verify_greedy_v1(dt, tl, K, V, ET_SPEC_NO_EOS, NULL, 3u,
                                 &out_res) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_verify_greedy_v1(dt, tl, K, V, ET_SPEC_NO_EOS, out_toks, 3u,
                                 NULL) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_verify_sample_v1(NULL, dl, tl, K, V, &params, &rng, out_toks,
                                 3u, &out_res) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_verify_sample_v1(dt, NULL, tl, K, V, &params, &rng, out_toks,
                                 3u, &out_res) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_verify_sample_v1(dt, dl, NULL, K, V, &params, &rng, out_toks,
                                 3u, &out_res) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, V, NULL, &rng, out_toks, 3u,
                                 &out_res) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, V, &params, NULL, out_toks, 3u,
                                 &out_res) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, V, &params, &rng, NULL, 3u,
                                 &out_res) == ET_SPEC_ERR_NULL);
  CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, V, &params, &rng, out_toks, 3u,
                                 NULL) == ET_SPEC_ERR_NULL);
  ASSERT_PRESERVED();

  /* Bounds: K=0, K=17, V=0, V=1, V=4097, out_capacity < K+1, counter overflow */
  CHECK(et_spec_verify_greedy_v1(dt, tl, 0u, V, ET_SPEC_NO_EOS, out_toks, 3u,
                                 &out_res) == ET_SPEC_ERR_BOUNDS);
  CHECK(et_spec_verify_greedy_v1(dt, tl, 17u, V, ET_SPEC_NO_EOS, out_toks, 18u,
                                 &out_res) == ET_SPEC_ERR_BOUNDS);
  CHECK(et_spec_verify_greedy_v1(dt, tl, K, 1u, ET_SPEC_NO_EOS, out_toks, 3u,
                                 &out_res) == ET_SPEC_ERR_BOUNDS);
  CHECK(et_spec_verify_greedy_v1(dt, tl, K, 4097u, ET_SPEC_NO_EOS, out_toks, 3u,
                                 &out_res) == ET_SPEC_ERR_BOUNDS);
  CHECK(et_spec_verify_greedy_v1(dt, tl, K, V, ET_SPEC_NO_EOS, out_toks, K,
                                 &out_res) == ET_SPEC_ERR_BOUNDS);
  CHECK(et_spec_verify_sample_v1(dt, dl, tl, 0u, V, &params, &rng, out_toks, 3u,
                                 &out_res) == ET_SPEC_ERR_BOUNDS);
  CHECK(et_spec_verify_sample_v1(dt, dl, tl, 17u, V, &params, &rng, out_toks,
                                 18u, &out_res) == ET_SPEC_ERR_BOUNDS);
  CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, 1u, &params, &rng, out_toks, 3u,
                                 &out_res) == ET_SPEC_ERR_BOUNDS);
  CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, 4097u, &params, &rng, out_toks,
                                 3u, &out_res) == ET_SPEC_ERR_BOUNDS);
  CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, V, &params, &rng, out_toks, K,
                                 &out_res) == ET_SPEC_ERR_BOUNDS);
  const et_spec_rng_state_v1 rng_max = {1u, UINT64_MAX - 1u};
  CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, V, &params, &rng_max, out_toks,
                                 3u, &out_res) == ET_SPEC_ERR_BOUNDS);
  const et_spec_rng_state_v1 rng_sat = {1u, UINT64_MAX};
  CHECK(et_spec_sample_token_v1(dl, V, &params, &rng_sat, &sample_tok,
                                &out_rng) == ET_SPEC_ERR_BOUNDS);
  ASSERT_PRESERVED();

  /* Aliasing */
  {
    uint32_t alias_u32[4] = {0u, 1u, 0u, 0u};
    CHECK(et_spec_verify_greedy_v1(alias_u32, tl, K, V, ET_SPEC_NO_EOS,
                                   alias_u32, 3u, &out_res) ==
          ET_SPEC_ERR_ALIAS);
    CHECK(et_spec_verify_sample_v1(alias_u32, dl, tl, K, V, &params, &rng,
                                   alias_u32, 3u, &out_res) ==
          ET_SPEC_ERR_ALIAS);
    CHECK(et_spec_verify_sample_v1(dt, tl, tl, K, V, &params, &rng, out_toks,
                                   3u, &out_res) == ET_SPEC_ERR_ALIAS);
    CHECK(et_spec_sample_token_v1(dl, V, &params, &rng, &sample_tok, &rng) ==
          ET_SPEC_ERR_ALIAS);
    union {
      float f[8];
      double d[4];
    } overlap;
    memset(&overlap, 0, sizeof(overlap));
    CHECK(et_spec_logits_to_probs_v1(overlap.f, V, &params, overlap.d) ==
          ET_SPEC_ERR_ALIAS);
  }
  ASSERT_PRESERVED();

  /* Misaligned pointers */
  {
    _Alignas(16) unsigned char raw[64];
    memset(raw, 0, sizeof(raw));
    const float *bad_f = (const float *)(const void *)(raw + 1u);
    const uint32_t *bad_u = (const uint32_t *)(const void *)(raw + 2u);
    double *bad_d = (double *)(void *)(raw + 4u);
    CHECK(et_spec_logits_to_probs_v1(bad_f, V, &params, probs) ==
          ET_SPEC_ERR_ALIGNMENT);
    CHECK(et_spec_logits_to_probs_v1(dl, V, &params, bad_d) ==
          ET_SPEC_ERR_ALIGNMENT);
    CHECK(et_spec_verify_greedy_v1(bad_u, tl, K, V, ET_SPEC_NO_EOS, out_toks,
                                   3u, &out_res) == ET_SPEC_ERR_ALIGNMENT);
  }
  ASSERT_PRESERVED();

  /* Out-of-range draft token IDs (including at pos K-1 even when pos 0 is EOS) */
  {
    const uint32_t bad_dt[2] = {0u, V};
    CHECK(et_spec_verify_greedy_v1(bad_dt, tl, K, V, 0u, out_toks, 3u,
                                   &out_res) == ET_SPEC_ERR_TOKEN_RANGE);
    CHECK(et_spec_verify_sample_v1(bad_dt, dl, tl, K, V, &params, &rng,
                                   out_toks, 3u, &out_res) ==
          ET_SPEC_ERR_TOKEN_RANGE);
    CHECK(et_spec_verify_greedy_v1(dt, tl, K, V, V, out_toks, 3u, &out_res) ==
          ET_SPEC_ERR_TOKEN_RANGE);
  }
  ASSERT_PRESERVED();

  /* Invalid sampling params */
  {
    const et_spec_sampling_params_v1 bad_cases[] = {
        {-0.0, 1.0, V, ET_SPEC_NO_EOS},
        {-1.0, 1.0, V, ET_SPEC_NO_EOS},
        {1.0, 1.0, 0u, ET_SPEC_NO_EOS},
        {1.0, 1.0, V + 1u, ET_SPEC_NO_EOS},
        {1.0, 0.0, V, ET_SPEC_NO_EOS},
        {1.0, -0.5, V, ET_SPEC_NO_EOS},
        {1.0, 1.0001, V, ET_SPEC_NO_EOS},
    };
    for (size_t i = 0u; i < sizeof(bad_cases) / sizeof(bad_cases[0]); ++i) {
      CHECK(et_spec_logits_to_probs_v1(dl, V, &bad_cases[i], probs) ==
            ET_SPEC_ERR_PARAM);
      CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, V, &bad_cases[i], &rng,
                                     out_toks, 3u, &out_res) ==
            ET_SPEC_ERR_PARAM);
    }
    const et_spec_sampling_params_v1 nonfin_cases[] = {
        {(double)NAN, 1.0, V, ET_SPEC_NO_EOS},
        {(double)INFINITY, 1.0, V, ET_SPEC_NO_EOS},
        {-(double)INFINITY, 1.0, V, ET_SPEC_NO_EOS},
        {1.0, (double)NAN, V, ET_SPEC_NO_EOS},
        {1.0, (double)INFINITY, V, ET_SPEC_NO_EOS},
        {1.0e-320, 1.0, V, ET_SPEC_NO_EOS}, /* scaled overflow */
    };
    for (size_t i = 0u; i < sizeof(nonfin_cases) / sizeof(nonfin_cases[0]); ++i) {
      CHECK(et_spec_logits_to_probs_v1(dl, V, &nonfin_cases[i], probs) ==
            ET_SPEC_ERR_NONFINITE);
      CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, V, &nonfin_cases[i], &rng,
                                     out_toks, 3u, &out_res) ==
            ET_SPEC_ERR_NONFINITE);
    }
  }
  ASSERT_PRESERVED();

  /* Non-finite logits in draft & bonus target rows (even if pos 0 matches EOS) */
  {
    float bad_dl[2 * 4];
    float bad_tl[3 * 4];
    memcpy(bad_dl, dl, sizeof(dl));
    memcpy(bad_tl, tl, sizeof(tl));
    bad_dl[2 * 4 - 1] = (float)NAN;
    CHECK(et_spec_verify_sample_v1(dt, bad_dl, tl, K, V, &params, &rng,
                                   out_toks, 3u, &out_res) ==
          ET_SPEC_ERR_NONFINITE);
    bad_tl[3 * 4 - 1] = -(float)INFINITY;
    CHECK(et_spec_verify_greedy_v1(dt, bad_tl, K, V, 0u, out_toks, 3u,
                                   &out_res) == ET_SPEC_ERR_NONFINITE);
    CHECK(et_spec_verify_sample_v1(dt, dl, bad_tl, K, V, &params, &rng,
                                   out_toks, 3u, &out_res) ==
          ET_SPEC_ERR_NONFINITE);
  }
  ASSERT_PRESERVED();

  /* FP-environment rejection (non-default rounding mode / FTZ) */
  {
    fenv_t saved;
    CHECK(fegetenv(&saved) == 0);
    CHECK(fesetround(FE_UPWARD) == 0);
    CHECK(et_spec_logits_to_probs_v1(dl, V, &params, probs) ==
          ET_SPEC_ERR_FP_ENV);
    CHECK(et_spec_sample_token_v1(dl, V, &params, &rng, &sample_tok,
                                  &out_rng) == ET_SPEC_ERR_FP_ENV);
    CHECK(et_spec_verify_greedy_v1(dt, tl, K, V, ET_SPEC_NO_EOS, out_toks, 3u,
                                   &out_res) == ET_SPEC_ERR_FP_ENV);
    CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, V, &params, &rng, out_toks,
                                   3u, &out_res) == ET_SPEC_ERR_FP_ENV);
    CHECK(fesetenv(&saved) == 0);

#if defined(__x86_64__)
    const unsigned int saved_mxcsr = _mm_getcsr();
    _mm_setcsr(saved_mxcsr | 0x8000u); /* Enable FTZ */
    CHECK(et_spec_verify_greedy_v1(dt, tl, K, V, ET_SPEC_NO_EOS, out_toks, 3u,
                                   &out_res) == ET_SPEC_ERR_FP_ENV);
    CHECK(et_spec_verify_sample_v1(dt, dl, tl, K, V, &params, &rng, out_toks,
                                   3u, &out_res) == ET_SPEC_ERR_FP_ENV);
    _mm_setcsr(saved_mxcsr);
#endif
  }
  ASSERT_PRESERVED();
#undef ASSERT_PRESERVED
}

static void test_bridge_native(void) {
  et_spec_aot_context *ctx = NULL;
  CHECK(et_spec_aot_create(0, 4, &ctx) == ET_SPEC_ERR_BOUNDS && ctx == NULL);
  CHECK(et_spec_aot_create(17, 4, &ctx) == ET_SPEC_ERR_BOUNDS && ctx == NULL);
  CHECK(et_spec_aot_create(4, 1, &ctx) == ET_SPEC_ERR_BOUNDS && ctx == NULL);
  CHECK(et_spec_aot_create(4, 4097, &ctx) == ET_SPEC_ERR_BOUNDS && ctx == NULL);
  CHECK(et_spec_aot_create(4, 8, NULL) == ET_SPEC_ERR_NULL);

  CHECK(et_spec_aot_create(4, 8, &ctx) == ET_SPEC_OK && ctx != NULL);
  CHECK(et_spec_aot_horizon(ctx) == 4);
  CHECK(et_spec_aot_vocab_size(ctx) == 8);
  CHECK(et_spec_aot_accepted_count(ctx) == -1);
  CHECK(isnan(et_spec_aot_prob(ctx, 0)));

  for (int64_t r = 0; r < 4; ++r) {
    CHECK(et_spec_aot_set_draft_token(ctx, r, r) == ET_SPEC_OK);
    for (int64_t v = 0; v < 8; ++v) {
      CHECK(et_spec_aot_set_draft_logit(ctx, r, v, (v == r) ? 3.0 : 0.0) ==
            ET_SPEC_OK);
      CHECK(et_spec_aot_set_target_logit(ctx, r, v, (v == r) ? 3.0 : 0.0) ==
            ET_SPEC_OK);
    }
  }
  for (int64_t v = 0; v < 8; ++v)
    CHECK(et_spec_aot_set_target_logit(ctx, 4, v, (v == 5) ? 4.0 : 0.0) ==
          ET_SPEC_OK);

  FORBID_ALLOC_BEGIN();
  CHECK(et_spec_aot_verify_greedy(ctx, ET_SPEC_AOT_NO_EOS) == ET_SPEC_OK);
  CHECK(et_spec_aot_accepted_count(ctx) == 4);
  CHECK(et_spec_aot_emitted_count(ctx) == 5);
  CHECK(et_spec_aot_emitted_token(ctx, 4) == 5);
  CHECK(et_spec_aot_verify_sample(ctx, 0.8, 4, 0.9, ET_SPEC_AOT_NO_EOS, 123,
                                  0) == ET_SPEC_OK);
  CHECK(et_spec_aot_accepted_count(ctx) == 4);
  CHECK(et_spec_aot_next_counter(ctx) == 5);
  /* Failure atomicity on bridge: non-finite logit rejected, prior telemetry kept. */
  CHECK(et_spec_aot_set_target_logit(ctx, 4, 0, 1.0e300) == ET_SPEC_OK);
  CHECK(et_spec_aot_verify_sample(ctx, 0.8, 4, 0.9, ET_SPEC_AOT_NO_EOS, 123,
                                  5) == ET_SPEC_ERR_NONFINITE);
  CHECK(et_spec_aot_accepted_count(ctx) == 4);
  CHECK(et_spec_aot_next_counter(ctx) == 5);
  FORBID_ALLOC_END();

  et_spec_aot_destroy(ctx);
  et_spec_aot_destroy(NULL);
}

int main(void) {
  test_greedy_32_token_rollouts();
  test_eos_early_termination();
  test_stochastic_edge_cases_and_replay();
  test_max_and_min_bounds();
  test_negative_and_output_preservation();
  test_bridge_native();
  printf("SPEC_DECODE NATIVE C11 TESTS: PASS\n");
  return 0;
}
