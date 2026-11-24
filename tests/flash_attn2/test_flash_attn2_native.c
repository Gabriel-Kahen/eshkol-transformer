#include "eshkol_transformer/flash_attn2_abi.h"
#include "flash_attn2_eshkol_bridge.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__x86_64__)
#error "FlashAttention-2 native test requires x86-64"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,         \
              #condition);                                                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* Trap any heap allocation during FlashAttention-2 core / bridge dispatch. */
static int dispatching = 0;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!dispatching);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!dispatching);
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!dispatching);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!dispatching);
  __real_free(ptr);
}

static et_fa2_config_v1 make_cfg(size_t b, size_t hq, size_t hkv, size_t tq,
                                 size_t tk, size_t dh, size_t br, size_t bc,
                                 uint32_t causal, size_t window, double scale,
                                 double softcap) {
  et_fa2_config_v1 cfg;
  cfg.struct_size = sizeof(cfg);
  cfg.batch = b;
  cfg.num_heads = hq;
  cfg.num_kv_heads = hkv;
  cfg.seq_q = tq;
  cfg.seq_k = tk;
  cfg.head_dim = dh;
  cfg.tile_r = br;
  cfg.tile_c = bc;
  cfg.sliding_window = window;
  cfg.causal = causal;
  cfg.reserved = 0u;
  cfg.scale = scale;
  cfg.logit_softcap = softcap;
  return cfg;
}

static void fill_det(float *buf, size_t count, int salt, float scale) {
  for (size_t i = 0u; i < count; ++i) {
    const int v = (int)((37u * (unsigned)salt + 19u * (unsigned)i +
                         11u * (unsigned)(i / 3u)) %
                        33u) -
                  16;
    const float jitter = 1.0f + 0.015625f * (float)(i % 5u);
    buf[i] = (float)v * scale * jitter;
  }
}

static void fill_sentinel(unsigned char *buf, size_t bytes,
                          unsigned char seed) {
  for (size_t i = 0u; i < bytes; ++i)
    buf[i] = (unsigned char)(seed ^ (unsigned char)((i * 29u + 7u) & 0xffu));
}

static int sentinel_intact(const unsigned char *buf, size_t bytes,
                           unsigned char seed) {
  for (size_t i = 0u; i < bytes; ++i) {
    const unsigned char expected =
        (unsigned char)(seed ^ (unsigned char)((i * 29u + 7u) & 0xffu));
    if (buf[i] != expected)
      return 0;
  }
  return 1;
}

static double max_diff_f32(const float *a, const float *b, size_t count) {
  double best = 0.0;
  for (size_t i = 0u; i < count; ++i) {
    const double d = fabs((double)a[i] - (double)b[i]);
    if (d > best)
      best = d;
  }
  return best;
}

static double max_diff_f64(const double *a, const double *b, size_t count) {
  double best = 0.0;
  for (size_t i = 0u; i < count; ++i) {
    const double d = fabs(a[i] - b[i]);
    if (d > best)
      best = d;
  }
  return best;
}

static int ref_pair_allowed(size_t q_pos, size_t k_pos, uint32_t causal,
                            size_t window) {
  if (causal != 0u) {
    if (k_pos > q_pos)
      return 0;
    if (window > 0u && (q_pos - k_pos) >= window)
      return 0;
    return 1;
  }
  if (window > 0u) {
    const size_t dist = (q_pos >= k_pos) ? (q_pos - k_pos) : (k_pos - q_pos);
    if (dist >= window)
      return 0;
  }
  return 1;
}

/* Independent dense binary64 reference inside test binary (allocated via heap). */
static void dense_ref_fwd_bwd(const et_fa2_config_v1 *cfg, const float *q,
                              const float *k, const float *v, const float *do_in,
                              float *o_ref, double *lse_ref, float *dq_ref,
                              float *dk_ref, float *dv_ref) {
  const size_t B = cfg->batch;
  const size_t Hq = cfg->num_heads;
  const size_t Hkv = cfg->num_kv_heads;
  const size_t group = Hq / Hkv;
  const size_t Tq = cfg->seq_q;
  const size_t Tk = cfg->seq_k;
  const size_t Dh = cfg->head_dim;
  const double scale =
      (cfg->scale == 0.0) ? (1.0 / sqrt((double)Dh)) : cfg->scale;
  const double softcap = cfg->logit_softcap;

  const size_t q_elems = B * Hq * Tq * Dh;
  const size_t kv_elems = B * Hkv * Tk * Dh;
  double *dq_acc = (double *)__real_calloc(q_elems, sizeof(double));
  double *dk_acc = (double *)__real_calloc(kv_elems, sizeof(double));
  double *dv_acc = (double *)__real_calloc(kv_elems, sizeof(double));
  double *scores = (double *)__real_malloc(Tk * sizeof(double));
  double *s_raw_arr = (double *)__real_malloc(Tk * sizeof(double));
  double *probs = (double *)__real_malloc(Tk * sizeof(double));
  double *dp_arr = (double *)__real_malloc(Tk * sizeof(double));
  CHECK(dq_acc && dk_acc && dv_acc && scores && s_raw_arr && probs && dp_arr);

  for (size_t b = 0u; b < B; ++b) {
    for (size_t h = 0u; h < Hq; ++h) {
      const size_t hkv = h / group;
      for (size_t i = 0u; i < Tq; ++i) {
        const size_t q_off = (((b * Hq + h) * Tq + i) * Dh);
        double m = -INFINITY;
        for (size_t j = 0u; j < Tk; ++j) {
          if (!ref_pair_allowed(i, j, cfg->causal, cfg->sliding_window)) {
            scores[j] = -INFINITY;
            s_raw_arr[j] = 0.0;
            continue;
          }
          const size_t k_off = (((b * Hkv + hkv) * Tk + j) * Dh);
          double dot = 0.0;
          for (size_t d = 0u; d < Dh; ++d)
            dot += (double)q[q_off + d] * (double)k[k_off + d];
          const double sr = dot * scale;
          s_raw_arr[j] = sr;
          const double se =
              (softcap > 0.0) ? (softcap * tanh(sr / softcap)) : sr;
          scores[j] = se;
          if (se > m)
            m = se;
        }
        double sum_exp = 0.0;
        for (size_t j = 0u; j < Tk; ++j) {
          if (scores[j] == -INFINITY)
            continue;
          sum_exp += exp(scores[j] - m);
        }
        const double lse = m + log(sum_exp);
        lse_ref[(b * Hq + h) * Tq + i] = lse;

        double o_row[ET_FA2_MAX_HEAD_DIM];
        memset(o_row, 0, Dh * sizeof(double));
        for (size_t j = 0u; j < Tk; ++j) {
          if (scores[j] == -INFINITY) {
            probs[j] = 0.0;
            continue;
          }
          const double p = exp(scores[j] - lse);
          probs[j] = p;
          const size_t v_off = (((b * Hkv + hkv) * Tk + j) * Dh);
          for (size_t d = 0u; d < Dh; ++d)
            o_row[d] += p * (double)v[v_off + d];
        }
        for (size_t d = 0u; d < Dh; ++d)
          o_ref[q_off + d] = (float)o_row[d];

        if (do_in != NULL) {
          double d_i = 0.0;
          for (size_t j = 0u; j < Tk; ++j) {
            if (scores[j] == -INFINITY) {
              dp_arr[j] = 0.0;
              continue;
            }
            const size_t v_off = (((b * Hkv + hkv) * Tk + j) * Dh);
            double dp = 0.0;
            for (size_t d = 0u; d < Dh; ++d)
              dp += (double)do_in[q_off + d] * (double)v[v_off + d];
            dp_arr[j] = dp;
            d_i += probs[j] * dp;
          }
          for (size_t j = 0u; j < Tk; ++j) {
            if (scores[j] == -INFINITY)
              continue;
            const size_t kv_off = (((b * Hkv + hkv) * Tk + j) * Dh);
            const double p = probs[j];
            const double ds_cap = p * (dp_arr[j] - d_i);
            double sc_d = 1.0;
            if (softcap > 0.0) {
              const double t = tanh(s_raw_arr[j] / softcap);
              sc_d = 1.0 - t * t;
            }
            const double ddot = ds_cap * sc_d * scale;
            for (size_t d = 0u; d < Dh; ++d) {
              dq_acc[q_off + d] += ddot * (double)k[kv_off + d];
              dk_acc[kv_off + d] += ddot * (double)q[q_off + d];
              dv_acc[kv_off + d] += p * (double)do_in[q_off + d];
            }
          }
        }
      }
    }
  }

  if (do_in != NULL) {
    for (size_t i = 0u; i < q_elems; ++i)
      dq_ref[i] = (float)dq_acc[i];
    for (size_t i = 0u; i < kv_elems; ++i) {
      dk_ref[i] = (float)dk_acc[i];
      dv_ref[i] = (float)dv_acc[i];
    }
  }
  __real_free(dq_acc);
  __real_free(dk_acc);
  __real_free(dv_acc);
  __real_free(scores);
  __real_free(s_raw_arr);
  __real_free(probs);
  __real_free(dp_arr);
}

/* Brute-force expected tile skip count for verification. */
static uint64_t expected_skipped_tiles(const et_fa2_config_v1 *cfg) {
  uint64_t skip_per_head = 0u;
  for (size_t qs = 0u; qs < cfg->seq_q; qs += cfg->tile_r) {
    const size_t qe =
        (cfg->seq_q - qs < cfg->tile_r) ? cfg->seq_q : (qs + cfg->tile_r);
    for (size_t ks = 0u; ks < cfg->seq_k; ks += cfg->tile_c) {
      const size_t ke =
          (cfg->seq_k - ks < cfg->tile_c) ? cfg->seq_k : (ks + cfg->tile_c);
      int any_allowed = 0;
      for (size_t i = qs; i < qe && !any_allowed; ++i) {
        for (size_t j = ks; j < ke; ++j) {
          if (ref_pair_allowed(i, j, cfg->causal, cfg->sliding_window)) {
            any_allowed = 1;
            break;
          }
        }
      }
      if (!any_allowed)
        ++skip_per_head;
    }
  }
  return skip_per_head * (uint64_t)(cfg->batch * cfg->num_heads);
}

static void print_hex_f32(const float *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%a", (double)buf[i]);
  }
  putchar('\n');
}

static void print_hex_f64(const double *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%a", buf[i]);
  }
  putchar('\n');
}

/* ============================================================================
 * 1. O(B_r * B_c) Constant Tile Scratch Bytes Across T = 32, 64, 128, 256
 * ============================================================================ */
static void test_constant_tile_scratch(void) {
  const size_t tiles[] = {8u, 16u, 32u};
  for (size_t i = 0u; i < ARRAY_COUNT(tiles); ++i) {
    for (size_t j = 0u; j < ARRAY_COUNT(tiles); ++j) {
      dispatching = 1;
      const size_t bytes = et_fa2_tile_scratch_bytes_v1(tiles[i], tiles[j]);
      dispatching = 0;
      CHECK(bytes == 2u * tiles[i] * tiles[j] * sizeof(double));
    }
  }
  dispatching = 1;
  CHECK(et_fa2_tile_scratch_bytes_v1(0u, 16u) == 0u);
  CHECK(et_fa2_tile_scratch_bytes_v1(4u, 16u) == 0u);
  CHECK(et_fa2_tile_scratch_bytes_v1(12u, 16u) == 0u);
  CHECK(et_fa2_tile_scratch_bytes_v1(64u, 16u) == 0u);
  dispatching = 0;

  const size_t seqs[] = {32u, 64u, 128u, 256u};
  for (size_t s = 0u; s < ARRAY_COUNT(seqs); ++s) {
    const size_t T = seqs[s];
    const et_fa2_config_v1 cfg =
        make_cfg(1u, 2u, 1u, T, T, 8u, 16u, 16u, 1u, 32u, 0.0, 25.0);
    const size_t q_elems = 2u * T * 8u;
    const size_t kv_elems = 1u * T * 8u;
    const size_t lse_elems = 2u * T;
    float *q = (float *)__real_malloc(q_elems * sizeof(float));
    float *k = (float *)__real_malloc(kv_elems * sizeof(float));
    float *v = (float *)__real_malloc(kv_elems * sizeof(float));
    float *o = (float *)__real_malloc(q_elems * sizeof(float));
    double *lse = (double *)__real_malloc(lse_elems * sizeof(double));
    CHECK(q && k && v && o && lse);
    fill_det(q, q_elems, (int)(10 + s), 0.0625f);
    fill_det(k, kv_elems, (int)(20 + s), 0.0625f);
    fill_det(v, kv_elems, (int)(30 + s), 0.0625f);

    et_fa2_stats_v1 st;
    memset(&st, 0, sizeof(st));
    dispatching = 1;
    CHECK(et_fa2_forward_v1(&cfg, q, k, v, o, lse, &st) == ET_FA2_OK);
    dispatching = 0;
    CHECK(st.tile_scratch_bytes == 4096u);
    CHECK(st.evaluated_tiles + st.skipped_tiles == st.total_tiles);
    CHECK(st.skipped_tiles == expected_skipped_tiles(&cfg));
    CHECK(st.skipped_tiles > 0u);

    __real_free(q);
    __real_free(k);
    __real_free(v);
    __real_free(o);
    __real_free(lse);
  }
  printf("FLASH_ATTN2 native constant O(Br*Bc) scratch T=32,64,128,256: PASS\n");
}

/* ============================================================================
 * 2. Tile-Size Invariance ((8,8) vs (16,16) vs (32,16) vs (32,32)) + Parity
 * ============================================================================ */
static void test_tile_invariance_and_features(void) {
  typedef struct {
    size_t b, hq, hkv, tq, tk, dh;
    uint32_t causal;
    size_t window;
    double scale;
    double softcap;
    const char *label;
  } case_spec;

  const case_spec specs[] = {
      {2u, 4u, 4u, 25u, 25u, 16u, 0u, 0u, 0.0, 0.0, "MHA_full"},
      {2u, 4u, 2u, 32u, 32u, 12u, 1u, 0u, 0.0, 0.0, "GQA_causal"},
      {1u, 4u, 1u, 29u, 29u, 8u, 1u, 10u, 0.0, 0.0, "MQA_causal_window"},
      {1u, 4u, 2u, 27u, 27u, 16u, 0u, 9u, 0.0, 20.0, "GQA_noncausal_win_cap"},
      {2u, 6u, 2u, 36u, 36u, 8u, 1u, 14u, 0.3, 15.0, "GQA_all_features"},
      {1u, 4u, 2u, 45u, 73u, 8u, 1u, 20u, 0.0, 30.0, "GQA_nonsquare_tq_lt_tk"},
      {1u, 4u, 2u, 50u, 37u, 8u, 1u, 20u, 0.0, 12.0, "GQA_nonsquare_tq_gt_tk"},
      {1u, 2u, 1u, 256u, 256u, 16u, 1u, 32u, 0.0, 30.0, "MQA_long_T256"},
      {1u, 4u, 2u, 19u, 23u, 64u, 0u, 12u, 0.0, 50.0, "GQA_Dh64_cap50"},
  };

  const size_t tile_pairs[][2] = {{8u, 8u}, {16u, 16u}, {32u, 16u}};

  double worst_o = 0.0, worst_lse = 0.0;
  double worst_dq = 0.0, worst_dk = 0.0, worst_dv = 0.0;
  double worst_tile_o = 0.0, worst_tile_lse = 0.0, worst_tile_g = 0.0;

  for (size_t s = 0u; s < ARRAY_COUNT(specs); ++s) {
    const case_spec *sp = &specs[s];
    const size_t q_elems = sp->b * sp->hq * sp->tq * sp->dh;
    const size_t kv_elems = sp->b * sp->hkv * sp->tk * sp->dh;
    const size_t lse_elems = sp->b * sp->hq * sp->tq;

    float *q = (float *)__real_malloc(q_elems * sizeof(float));
    float *k = (float *)__real_malloc(kv_elems * sizeof(float));
    float *v = (float *)__real_malloc(kv_elems * sizeof(float));
    float *do_in = (float *)__real_malloc(q_elems * sizeof(float));
    float *o_ref = (float *)__real_malloc(q_elems * sizeof(float));
    double *lse_ref = (double *)__real_malloc(lse_elems * sizeof(double));
    float *dq_ref = (float *)__real_malloc(q_elems * sizeof(float));
    float *dk_ref = (float *)__real_malloc(kv_elems * sizeof(float));
    float *dv_ref = (float *)__real_malloc(kv_elems * sizeof(float));

    float *o0 = (float *)__real_malloc(q_elems * sizeof(float));
    double *lse0 = (double *)__real_malloc(lse_elems * sizeof(double));
    float *dq0 = (float *)__real_malloc(q_elems * sizeof(float));
    float *dk0 = (float *)__real_malloc(kv_elems * sizeof(float));
    float *dv0 = (float *)__real_malloc(kv_elems * sizeof(float));

    float *o_cur = (float *)__real_malloc(q_elems * sizeof(float));
    double *lse_cur = (double *)__real_malloc(lse_elems * sizeof(double));
    float *dq_cur = (float *)__real_malloc(q_elems * sizeof(float));
    float *dk_cur = (float *)__real_malloc(kv_elems * sizeof(float));
    float *dv_cur = (float *)__real_malloc(kv_elems * sizeof(float));

    CHECK(q && k && v && do_in && o_ref && lse_ref && dq_ref && dk_ref &&
          dv_ref && o0 && lse0 && dq0 && dk0 && dv0 && o_cur && lse_cur &&
          dq_cur && dk_cur && dv_cur);

    fill_det(q, q_elems, (int)(1 + 4 * s), 0.0625f);
    fill_det(k, kv_elems, (int)(2 + 4 * s), 0.0625f);
    fill_det(v, kv_elems, (int)(3 + 4 * s), 0.0625f);
    fill_det(do_in, q_elems, (int)(4 + 4 * s), 0.0625f);

    const et_fa2_config_v1 base_cfg = make_cfg(
        sp->b, sp->hq, sp->hkv, sp->tq, sp->tk, sp->dh, 8u, 8u, sp->causal,
        sp->window, sp->scale, sp->softcap);
    dense_ref_fwd_bwd(&base_cfg, q, k, v, do_in, o_ref, lse_ref, dq_ref, dk_ref,
                      dv_ref);

    for (size_t tp = 0u; tp < ARRAY_COUNT(tile_pairs); ++tp) {
      const et_fa2_config_v1 cfg = make_cfg(
          sp->b, sp->hq, sp->hkv, sp->tq, sp->tk, sp->dh, tile_pairs[tp][0],
          tile_pairs[tp][1], sp->causal, sp->window, sp->scale, sp->softcap);
      et_fa2_stats_v1 fst, bst;
      memset(&fst, 0, sizeof(fst));
      memset(&bst, 0, sizeof(bst));

      dispatching = 1;
      CHECK(et_fa2_forward_v1(&cfg, q, k, v, o_cur, lse_cur, &fst) ==
            ET_FA2_OK);
      CHECK(et_fa2_backward_v1(&cfg, q, k, v, o_cur, lse_cur, do_in, dq_cur,
                               dk_cur, dv_cur, &bst) == ET_FA2_OK);
      dispatching = 0;

      const uint64_t exp_skip = expected_skipped_tiles(&cfg);
      CHECK(fst.skipped_tiles == exp_skip);
      CHECK(bst.skipped_tiles == exp_skip);
      CHECK(fst.evaluated_tiles + fst.skipped_tiles == fst.total_tiles);
      CHECK(bst.evaluated_tiles + bst.skipped_tiles == bst.total_tiles);
      CHECK(fst.tile_scratch_bytes ==
            et_fa2_tile_scratch_bytes_v1(cfg.tile_r, cfg.tile_c));
      CHECK(bst.tile_scratch_bytes == fst.tile_scratch_bytes);

      const double eo = max_diff_f32(o_cur, o_ref, q_elems);
      const double else_v = max_diff_f64(lse_cur, lse_ref, lse_elems);
      const double edq = max_diff_f32(dq_cur, dq_ref, q_elems);
      const double edk = max_diff_f32(dk_cur, dk_ref, kv_elems);
      const double edv = max_diff_f32(dv_cur, dv_ref, kv_elems);

      CHECK(eo <= 1.0e-6);
      CHECK(else_v <= 1.0e-12);
      CHECK(edq <= 1.0e-6);
      CHECK(edk <= 1.0e-6);
      CHECK(edv <= 1.0e-6);

      if (eo > worst_o)
        worst_o = eo;
      if (else_v > worst_lse)
        worst_lse = else_v;
      if (edq > worst_dq)
        worst_dq = edq;
      if (edk > worst_dk)
        worst_dk = edk;
      if (edv > worst_dv)
        worst_dv = edv;

      if (tp == 0u) {
        memcpy(o0, o_cur, q_elems * sizeof(float));
        memcpy(lse0, lse_cur, lse_elems * sizeof(double));
        memcpy(dq0, dq_cur, q_elems * sizeof(float));
        memcpy(dk0, dk_cur, kv_elems * sizeof(float));
        memcpy(dv0, dv_cur, kv_elems * sizeof(float));
      } else {
        const double to = max_diff_f32(o0, o_cur, q_elems);
        const double tl = max_diff_f64(lse0, lse_cur, lse_elems);
        const double tdq = max_diff_f32(dq0, dq_cur, q_elems);
        const double tdk = max_diff_f32(dk0, dk_cur, kv_elems);
        const double tdv = max_diff_f32(dv0, dv_cur, kv_elems);
        CHECK(to <= 1.0e-6);
        CHECK(tl <= 1.0e-12);
        CHECK(tdq <= 1.0e-6);
        CHECK(tdk <= 1.0e-6);
        CHECK(tdv <= 1.0e-6);
        if (to > worst_tile_o)
          worst_tile_o = to;
        if (tl > worst_tile_lse)
          worst_tile_lse = tl;
        if (tdq > worst_tile_g)
          worst_tile_g = tdq;
        if (tdk > worst_tile_g)
          worst_tile_g = tdk;
        if (tdv > worst_tile_g)
          worst_tile_g = tdv;
      }
    }

    __real_free(q);
    __real_free(k);
    __real_free(v);
    __real_free(do_in);
    __real_free(o_ref);
    __real_free(lse_ref);
    __real_free(dq_ref);
    __real_free(dk_ref);
    __real_free(dv_ref);
    __real_free(o0);
    __real_free(lse0);
    __real_free(dq0);
    __real_free(dk0);
    __real_free(dv0);
    __real_free(o_cur);
    __real_free(lse_cur);
    __real_free(dq_cur);
    __real_free(dk_cur);
    __real_free(dv_cur);
  }

  printf("FLASH_ATTN2 native parity & tile invariance (9 configs x 3 tiles): "
         "PASS (ref_err O=%.3e LSE=%.3e dQ=%.3e dK=%.3e dV=%.3e; "
         "tile_inv O=%.3e LSE=%.3e grad=%.3e)\n",
         worst_o, worst_lse, worst_dq, worst_dk, worst_dv, worst_tile_o,
         worst_tile_lse, worst_tile_g);
}

/* ============================================================================
 * 3. Exhaustive Central Finite-Difference Checks for dQ, dK, dV
 * ============================================================================ */
static double check_fd_tensor(const et_fa2_config_v1 *cfg, float *q, float *k,
                              float *v, const float *do_in, float *target_buf,
                              const float *analytic_grad, size_t target_elems,
                              size_t stride, float *o_plus, float *o_minus,
                              double *lse_tmp, size_t *checked_out) {
  const size_t q_elems = cfg->batch * cfg->num_heads * cfg->seq_q * cfg->head_dim;
  const float eps = 0.001953125f; /* 2^-9, exact in binary32 */
  double worst = 0.0;

  for (size_t idx = 0u; idx < target_elems; idx += stride) {
    const float orig = target_buf[idx];
    target_buf[idx] = orig + eps;
    dispatching = 1;
    CHECK(et_fa2_forward_v1(cfg, q, k, v, o_plus, lse_tmp, NULL) == ET_FA2_OK);
    dispatching = 0;

    target_buf[idx] = orig - eps;
    dispatching = 1;
    CHECK(et_fa2_forward_v1(cfg, q, k, v, o_minus, lse_tmp, NULL) == ET_FA2_OK);
    dispatching = 0;

    target_buf[idx] = orig;

    /* Cancellation-free binary64 dot of (O_plus - O_minus) with dO. */
    double diff_loss = 0.0;
    for (size_t i = 0u; i < q_elems; ++i)
      diff_loss += ((double)o_plus[i] - (double)o_minus[i]) * (double)do_in[i];
    const double fd = diff_loss / (2.0 * (double)eps);
    const double an = (double)analytic_grad[idx];
    const double err = fabs(an - fd);
    CHECK(err < 1.0e-5);
    if (err > worst)
      worst = err;
    (*checked_out)++;
  }
  return worst;
}

static void test_central_fd_gradients(void) {
  const et_fa2_config_v1 cases[] = {
      make_cfg(1u, 4u, 2u, 16u, 16u, 4u, 8u, 8u, 1u, 6u, 0.0, 15.0),
      make_cfg(1u, 4u, 1u, 12u, 18u, 4u, 8u, 8u, 0u, 7u, 0.0, 10.0),
      make_cfg(2u, 2u, 2u, 14u, 10u, 6u, 8u, 8u, 1u, 8u, 0.35, 0.0),
      make_cfg(1u, 4u, 2u, 16u, 16u, 4u, 16u, 8u, 0u, 0u, 0.0, 25.0),
  };

  size_t total_checked = 0u;
  double worst_err = 0.0;

  for (size_t c = 0u; c < ARRAY_COUNT(cases); ++c) {
    const et_fa2_config_v1 *cfg = &cases[c];
    const size_t q_elems =
        cfg->batch * cfg->num_heads * cfg->seq_q * cfg->head_dim;
    const size_t kv_elems =
        cfg->batch * cfg->num_kv_heads * cfg->seq_k * cfg->head_dim;
    const size_t lse_elems = cfg->batch * cfg->num_heads * cfg->seq_q;

    float *q = (float *)__real_malloc(q_elems * sizeof(float));
    float *k = (float *)__real_malloc(kv_elems * sizeof(float));
    float *v = (float *)__real_malloc(kv_elems * sizeof(float));
    float *do_in = (float *)__real_malloc(q_elems * sizeof(float));
    float *o = (float *)__real_malloc(q_elems * sizeof(float));
    float *op = (float *)__real_malloc(q_elems * sizeof(float));
    float *om = (float *)__real_malloc(q_elems * sizeof(float));
    double *lse = (double *)__real_malloc(lse_elems * sizeof(double));
    double *lse_tmp = (double *)__real_malloc(lse_elems * sizeof(double));
    float *dq = (float *)__real_malloc(q_elems * sizeof(float));
    float *dk = (float *)__real_malloc(kv_elems * sizeof(float));
    float *dv = (float *)__real_malloc(kv_elems * sizeof(float));
    CHECK(q && k && v && do_in && o && op && om && lse && lse_tmp && dq && dk &&
          dv);

    fill_det(q, q_elems, (int)(51 + c * 4), 0.03125f);
    fill_det(k, kv_elems, (int)(52 + c * 4), 0.03125f);
    fill_det(v, kv_elems, (int)(53 + c * 4), 0.03125f);
    fill_det(do_in, q_elems, (int)(54 + c * 4), 0.03125f);

    dispatching = 1;
    CHECK(et_fa2_forward_v1(cfg, q, k, v, o, lse, NULL) == ET_FA2_OK);
    CHECK(et_fa2_backward_v1(cfg, q, k, v, o, lse, do_in, dq, dk, dv, NULL) ==
          ET_FA2_OK);
    dispatching = 0;

    const size_t q_stride = q_elems > 24u ? q_elems / 24u : 1u;
    const size_t kv_stride = kv_elems > 24u ? kv_elems / 24u : 1u;

    const double eq = check_fd_tensor(cfg, q, k, v, do_in, q, dq, q_elems,
                                      q_stride, op, om, lse_tmp, &total_checked);
    const double ek = check_fd_tensor(cfg, q, k, v, do_in, k, dk, kv_elems,
                                      kv_stride, op, om, lse_tmp, &total_checked);
    const double ev = check_fd_tensor(cfg, q, k, v, do_in, v, dv, kv_elems,
                                      kv_stride, op, om, lse_tmp, &total_checked);
    if (eq > worst_err)
      worst_err = eq;
    if (ek > worst_err)
      worst_err = ek;
    if (ev > worst_err)
      worst_err = ev;

    __real_free(q);
    __real_free(k);
    __real_free(v);
    __real_free(do_in);
    __real_free(o);
    __real_free(op);
    __real_free(om);
    __real_free(lse);
    __real_free(lse_tmp);
    __real_free(dq);
    __real_free(dk);
    __real_free(dv);
  }

  CHECK(total_checked >= 250u);
  CHECK(worst_err < 1.0e-5);
  printf("FLASH_ATTN2 native central FD gradients (dQ/dK/dV, GQA+Causal+Win+Cap): "
         "PASS (checked=%zu, max_abs_err=%.3e)\n",
         total_checked, worst_err);
}

/* ============================================================================
 * 4. Negative, Boundary, Aliasing, Failure-Atomicity & FP-Env Tests
 * ============================================================================ */
static void test_negative_and_fpenv(void) {
  const et_fa2_config_v1 good =
      make_cfg(1u, 4u, 2u, 16u, 16u, 8u, 8u, 8u, 1u, 8u, 0.0, 20.0);
  const size_t q_elems = 1u * 4u * 16u * 8u;
  const size_t kv_elems = 1u * 2u * 16u * 8u;
  const size_t lse_elems = 1u * 4u * 16u;

  float q[512], k[256], v[256], o[512], do_in[512];
  double lse[64];
  float dq[512], dk[256], dv[256];
  et_fa2_stats_v1 st;

  fill_det(q, q_elems, 7, 0.0625f);
  fill_det(k, kv_elems, 8, 0.0625f);
  fill_det(v, kv_elems, 9, 0.0625f);
  fill_det(do_in, q_elems, 10, 0.0625f);

  /* Populate valid O and LSE for backward tests. */
  dispatching = 1;
  CHECK(et_fa2_forward_v1(&good, q, k, v, o, lse, &st) == ET_FA2_OK);
  dispatching = 0;

  /* Plant sentinels in all outputs. */
  float o_out[512];
  double lse_out[64];
  fill_sentinel((unsigned char *)o_out, sizeof(o_out), 0xA1u);
  fill_sentinel((unsigned char *)lse_out, sizeof(lse_out), 0xB2u);
  fill_sentinel((unsigned char *)dq, sizeof(dq), 0xC3u);
  fill_sentinel((unsigned char *)dk, sizeof(dk), 0xD4u);
  fill_sentinel((unsigned char *)dv, sizeof(dv), 0xE5u);
  fill_sentinel((unsigned char *)&st, sizeof(st), 0xF6u);

  /* Invalid configs */
  et_fa2_config_v1 bad = good;
  bad.struct_size = sizeof(bad) - 1u;
  dispatching = 1;
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_INVALID_ARG);
  CHECK(et_fa2_forward_v1(&bad, q, k, v, o_out, lse_out, &st) ==
        ET_FA2_ERR_INVALID_ARG);

  bad = good;
  bad.reserved = 1u;
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_INVALID_ARG);

  bad = good;
  bad.causal = 2u;
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_INVALID_ARG);

  bad = good;
  bad.scale = -1.0;
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_INVALID_ARG);

  bad = good;
  bad.logit_softcap = -5.0;
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_INVALID_ARG);

  bad = good;
  bad.logit_softcap = INFINITY;
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_INVALID_ARG);

  bad = good;
  bad.batch = 5u;
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_UNSUPPORTED);

  bad = good;
  bad.seq_q = 257u;
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_UNSUPPORTED);

  bad = good;
  bad.head_dim = 65u;
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_UNSUPPORTED);

  bad = good;
  bad.tile_r = 12u;
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_UNSUPPORTED);

  bad = good;
  bad.tile_c = 64u;
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_UNSUPPORTED);

  bad = good;
  bad.num_kv_heads = 3u; /* H_q=4 % 3 != 0 */
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_SHAPE_MISMATCH);

  bad = good;
  bad.seq_q = 32u;
  bad.seq_k = 16u;
  bad.sliding_window = 16u; /* T_q - T_k == 16 >= W -> empty row 31 */
  CHECK(et_fa2_validate_config_v1(&bad, NULL) == ET_FA2_ERR_SHAPE_MISMATCH);

  /* Alignment & NULL checks */
  CHECK(et_fa2_forward_v1(NULL, q, k, v, o_out, lse_out, &st) ==
        ET_FA2_ERR_INVALID_ARG);
  CHECK(et_fa2_forward_v1(&good, (const char *)q + 1, k, v, o_out, lse_out,
                          &st) == ET_FA2_ERR_INVALID_ARG);
  CHECK(et_fa2_forward_v1(&good, q, k, v, (char *)o_out + 2, lse_out, &st) ==
        ET_FA2_ERR_INVALID_ARG);
  CHECK(et_fa2_forward_v1(&good, q, k, v, o_out, (char *)lse_out + 4, &st) ==
        ET_FA2_ERR_INVALID_ARG);

  /* Aliasing checks */
  CHECK(et_fa2_forward_v1(&good, q, k, v, q, lse_out, &st) ==
        ET_FA2_ERR_INVALID_ARG);
  CHECK(et_fa2_forward_v1(&good, q, k, v, o_out, (void *)o_out, &st) ==
        ET_FA2_ERR_INVALID_ARG);
  CHECK(et_fa2_backward_v1(&good, q, k, v, o, lse, do_in, dq, dk, dk, &st) ==
        ET_FA2_ERR_INVALID_ARG);
  CHECK(et_fa2_backward_v1(&good, q, k, v, o, lse, do_in, q, dk, dv, &st) ==
        ET_FA2_ERR_INVALID_ARG);
  CHECK(et_fa2_backward_v1(&good, q, k, v, o, lse, do_in, dq, dk, dv,
                           (et_fa2_stats_v1 *)(void *)dq) ==
        ET_FA2_ERR_INVALID_ARG);

  /* Non-finite inputs */
  q[5] = NAN;
  CHECK(et_fa2_forward_v1(&good, q, k, v, o_out, lse_out, &st) ==
        ET_FA2_ERR_NONFINITE);
  q[5] = 0.125f;

  k[3] = INFINITY;
  CHECK(et_fa2_forward_v1(&good, q, k, v, o_out, lse_out, &st) ==
        ET_FA2_ERR_NONFINITE);
  k[3] = 0.125f;

  do_in[7] = -INFINITY;
  CHECK(et_fa2_backward_v1(&good, q, k, v, o, lse, do_in, dq, dk, dv, &st) ==
        ET_FA2_ERR_NONFINITE);
  do_in[7] = 0.0625f;

  lse[2] = NAN;
  CHECK(et_fa2_backward_v1(&good, q, k, v, o, lse, do_in, dq, dk, dv, &st) ==
        ET_FA2_ERR_NONFINITE);

  /*
   * Failure-atomic staged overflow test: valid finite LSE everywhere except
   * adversarial last row lse[63] = -1000.0 so exp(S - LSE) overflows on the
   * final head. Must fail with ET_FA2_ERR_NONFINITE and leave dq/dk/dv/st
   * byte-for-byte untouched!
   */
  lse[2] = 0.0;
  lse[lse_elems - 1u] = -1000.0;
  CHECK(et_fa2_backward_v1(&good, q, k, v, o, lse, do_in, dq, dk, dv, &st) ==
        ET_FA2_ERR_NONFINITE);
  dispatching = 0;

  /* FP-environment rejection tests (rounding mode + MXCSR FTZ/DAZ). */
  fenv_t saved_env;
  CHECK(fegetenv(&saved_env) == 0);
  CHECK(fesetround(FE_TOWARDZERO) == 0);
  dispatching = 1;
  CHECK(et_fa2_forward_v1(&good, q, k, v, o_out, lse_out, &st) ==
        ET_FA2_ERR_FP_ENV);
  CHECK(et_fa2_backward_v1(&good, q, k, v, o, lse, do_in, dq, dk, dv, &st) ==
        ET_FA2_ERR_FP_ENV);
  dispatching = 0;
  CHECK(fesetenv(&saved_env) == 0);

  const unsigned int saved_mxcsr = _mm_getcsr();
  _mm_setcsr(saved_mxcsr | 0x8000u); /* Set FTZ bit */
  dispatching = 1;
  CHECK(et_fa2_forward_v1(&good, q, k, v, o_out, lse_out, &st) ==
        ET_FA2_ERR_FP_ENV);
  dispatching = 0;
  _mm_setcsr(saved_mxcsr);

  /* Verify EVERY sentinel remained byte-for-byte untouched! */
  CHECK(sentinel_intact((unsigned char *)o_out, sizeof(o_out), 0xA1u));
  CHECK(sentinel_intact((unsigned char *)lse_out, sizeof(lse_out), 0xB2u));
  CHECK(sentinel_intact((unsigned char *)dq, sizeof(dq), 0xC3u));
  CHECK(sentinel_intact((unsigned char *)dk, sizeof(dk), 0xD4u));
  CHECK(sentinel_intact((unsigned char *)dv, sizeof(dv), 0xE5u));
  CHECK(sentinel_intact((unsigned char *)&st, sizeof(st), 0xF6u));

  printf("FLASH_ATTN2 native negative, aliasing, failure-atomicity & FP-env: "
         "PASS\n");
}

/* ============================================================================
 * 5. --oracle Mode for Independent Python binary64 Reference Verification
 * ============================================================================ */
static void run_oracle_dump(void) {
  const et_fa2_config_v1 cases[] = {
      /* MHA full */
      make_cfg(1u, 2u, 2u, 12u, 12u, 4u, 8u, 8u, 0u, 0u, 0.0, 0.0),
      /* GQA causal */
      make_cfg(1u, 4u, 2u, 16u, 16u, 4u, 8u, 8u, 1u, 0u, 0.0, 0.0),
      /* MQA causal + sliding window */
      make_cfg(1u, 4u, 1u, 20u, 20u, 4u, 8u, 8u, 1u, 7u, 0.0, 0.0),
      /* GQA non-causal + sliding window */
      make_cfg(1u, 4u, 2u, 18u, 18u, 4u, 8u, 16u, 0u, 6u, 0.0, 0.0),
      /* GQA causal + softcap */
      make_cfg(1u, 4u, 2u, 16u, 16u, 6u, 8u, 8u, 1u, 0u, 0.0, 12.0),
      /* GQA causal + sliding window + softcap (16x16 tiles) */
      make_cfg(2u, 4u, 2u, 24u, 24u, 4u, 16u, 16u, 1u, 9u, 0.0, 18.0),
      /* Non-square T_q < T_k + causal + window + softcap + custom scale */
      make_cfg(1u, 4u, 2u, 13u, 21u, 4u, 8u, 8u, 1u, 10u, 0.375, 15.0),
      /* Non-square T_q > T_k + causal + window + softcap */
      make_cfg(1u, 4u, 2u, 19u, 14u, 4u, 8u, 8u, 1u, 10u, 0.0, 20.0),
      /* Tile (32,16) GQA + non-causal + window + softcap */
      make_cfg(1u, 4u, 1u, 33u, 25u, 8u, 32u, 16u, 0u, 11u, 0.0, 25.0),
  };

  for (size_t c = 0u; c < ARRAY_COUNT(cases); ++c) {
    const et_fa2_config_v1 *cfg = &cases[c];
    const size_t q_elems =
        cfg->batch * cfg->num_heads * cfg->seq_q * cfg->head_dim;
    const size_t kv_elems =
        cfg->batch * cfg->num_kv_heads * cfg->seq_k * cfg->head_dim;
    const size_t lse_elems = cfg->batch * cfg->num_heads * cfg->seq_q;

    float *q = (float *)__real_malloc(q_elems * sizeof(float));
    float *k = (float *)__real_malloc(kv_elems * sizeof(float));
    float *v = (float *)__real_malloc(kv_elems * sizeof(float));
    float *do_in = (float *)__real_malloc(q_elems * sizeof(float));
    float *o = (float *)__real_malloc(q_elems * sizeof(float));
    double *lse = (double *)__real_malloc(lse_elems * sizeof(double));
    float *dq = (float *)__real_malloc(q_elems * sizeof(float));
    float *dk = (float *)__real_malloc(kv_elems * sizeof(float));
    float *dv = (float *)__real_malloc(kv_elems * sizeof(float));
    CHECK(q && k && v && do_in && o && lse && dq && dk && dv);

    fill_det(q, q_elems, (int)(101 + c * 4), 0.0625f);
    fill_det(k, kv_elems, (int)(102 + c * 4), 0.0625f);
    fill_det(v, kv_elems, (int)(103 + c * 4), 0.0625f);
    fill_det(do_in, q_elems, (int)(104 + c * 4), 0.0625f);

    et_fa2_stats_v1 fst, bst;
    dispatching = 1;
    CHECK(et_fa2_forward_v1(cfg, q, k, v, o, lse, &fst) == ET_FA2_OK);
    CHECK(et_fa2_backward_v1(cfg, q, k, v, o, lse, do_in, dq, dk, dv, &bst) ==
          ET_FA2_OK);
    dispatching = 0;

    printf("CASE %zu %zu %zu %zu %zu %zu %zu %zu %u %zu %a %a %llu %llu %llu "
           "%llu\n",
           cfg->batch, cfg->num_heads, cfg->num_kv_heads, cfg->seq_q,
           cfg->seq_k, cfg->head_dim, cfg->tile_r, cfg->tile_c, cfg->causal,
           cfg->sliding_window, cfg->scale, cfg->logit_softcap,
           (unsigned long long)fst.evaluated_tiles,
           (unsigned long long)fst.skipped_tiles,
           (unsigned long long)bst.evaluated_tiles,
           (unsigned long long)bst.skipped_tiles);
    print_hex_f32(q, q_elems);
    print_hex_f32(k, kv_elems);
    print_hex_f32(v, kv_elems);
    print_hex_f32(do_in, q_elems);
    print_hex_f32(o, q_elems);
    print_hex_f64(lse, lse_elems);
    print_hex_f32(dq, q_elems);
    print_hex_f32(dk, kv_elems);
    print_hex_f32(dv, kv_elems);

    __real_free(q);
    __real_free(k);
    __real_free(v);
    __real_free(do_in);
    __real_free(o);
    __real_free(lse);
    __real_free(dq);
    __real_free(dk);
    __real_free(dv);
  }
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    run_oracle_dump();
    return 0;
  }
  test_constant_tile_scratch();
  test_tile_invariance_and_features();
  test_central_fd_gradients();
  test_negative_and_fpenv();
  printf("FLASH_ATTN2 native C11 test suite: ALL PASS\n");
  return 0;
}
