#define _GNU_SOURCE
#include "eshkol_transformer/ring_attn_abi.h"
#include "ring_attn_eshkol_bridge.h"

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void die_msg(const char *msg, int line) {
  fprintf(stderr, "FAIL (line %d): %s\n", line, msg);
  exit(1);
}

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond))                                                               \
      die_msg(#cond, __LINE__);                                                \
  } while (0)

static int count_open_fds(void) {
  DIR *d = opendir("/proc/self/fd");
  CHECK(d != NULL);
  int count = 0;
  struct dirent *ent;
  while ((ent = readdir(d)) != NULL) {
    if (ent->d_name[0] != '.')
      ++count;
  }
  closedir(d);
  return count;
}

static int64_t now_ms(void) {
  struct timespec ts;
  CHECK(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
  return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void fill_deterministic(float *buf, size_t n, uint32_t seed,
                               float scale) {
  uint64_t state = 0x9E3779B97F4A7C15ULL ^ (uint64_t)seed * 0xBF58476D1CE4E5B9ULL;
  for (size_t i = 0; i < n; ++i) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    const uint64_t x = state * 0x2545F4914F6CDD1DULL;
    const int v = (int)((x >> 48) % 33u) - 16;
    buf[i] = (float)v * scale;
  }
}

static double max_abs_diff_f32(const float *a, const float *b, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = fabs((double)a[i] - (double)b[i]);
    if (d > m)
      m = d;
  }
  return m;
}

static double max_abs_diff_f64(const double *a, const double *b, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = fabs(a[i] - b[i]);
    if (d > m)
      m = d;
  }
  return m;
}

/* ============================================================================
 * 1. Single-Process Block-Step Online Softmax & Backward Verification
 * ============================================================================ */
static void test_block_step_unit(void) {
  const et_ring_attn_config_v1 cfg = {
      sizeof(cfg), 2u, 8u, 4u, 2u, 4u, 4u, 1u, 0.0, 5000};
  const size_t N = cfg.batch, T = cfg.seq_len, H = cfg.num_heads,
               Hkv = cfg.num_kv_heads, Dh = cfg.head_dim;
  const uint32_t W = cfg.world_size;
  const size_t T_loc = T / W;
  const size_t q_elems = N * T * H * Dh;
  const size_t kv_elems = N * T * Hkv * Dh;
  const size_t lse_elems = N * T * H;

  float *q = malloc(q_elems * sizeof(float));
  float *k = malloc(kv_elems * sizeof(float));
  float *v = malloc(kv_elems * sizeof(float));
  float *do_f = malloc(q_elems * sizeof(float));
  float *o_ref = calloc(q_elems, sizeof(float));
  double *lse_ref = calloc(lse_elems, sizeof(double));
  float *dq_ref = calloc(q_elems, sizeof(float));
  float *dk_ref = calloc(kv_elems, sizeof(float));
  float *dv_ref = calloc(kv_elems, sizeof(float));
  CHECK(q && k && v && do_f && o_ref && lse_ref && dq_ref && dk_ref && dv_ref);

  fill_deterministic(q, q_elems, 101u, 0.0625f);
  fill_deterministic(k, kv_elems, 202u, 0.0625f);
  fill_deterministic(v, kv_elems, 303u, 0.0625f);
  fill_deterministic(do_f, q_elems, 404u, 0.0625f);

  CHECK(et_ring_attn_dense_ref_forward_v1(&cfg, q, k, v, o_ref, lse_ref) ==
        ET_RING_ATTN_OK);
  CHECK(et_ring_attn_dense_ref_backward_v1(&cfg, q, k, v, o_ref, lse_ref, do_f,
                                           dq_ref, dk_ref, dv_ref) ==
        ET_RING_ATTN_OK);

  const size_t q_loc = N * T_loc * H * Dh;
  const size_t kv_loc = N * T_loc * Hkv * Dh;
  const size_t lse_loc = N * T_loc * H;

  float *o_blk = calloc(q_elems, sizeof(float));
  double *lse_blk = calloc(lse_elems, sizeof(double));
  float *dq_blk = calloc(q_elems, sizeof(float));
  double *dk_global = calloc(kv_elems, sizeof(double));
  double *dv_global = calloc(kv_elems, sizeof(double));

  float *q_s = malloc(q_loc * sizeof(float));
  float *k_s = malloc(kv_loc * sizeof(float));
  float *v_s = malloc(kv_loc * sizeof(float));
  float *do_s = malloc(q_loc * sizeof(float));
  float *o_s = malloc(q_loc * sizeof(float));
  double *lse_s = malloc(lse_loc * sizeof(double));
  double *m_st = malloc(lse_loc * sizeof(double));
  double *l_st = malloc(lse_loc * sizeof(double));
  double *o_acc = malloc(q_loc * sizeof(double));
  double *d_vec = malloc(lse_loc * sizeof(double));
  double *dq_acc = malloc(q_loc * sizeof(double));
  double *dk_acc = malloc(kv_loc * sizeof(double));
  double *dv_acc = malloc(kv_loc * sizeof(double));
  CHECK(o_blk && lse_blk && dq_blk && dk_global && dv_global && q_s && k_s &&
        v_s && do_s && o_s && lse_s && m_st && l_st && o_acc && d_vec &&
        dq_acc && dk_acc && dv_acc);

  et_ring_attn_stats_v1 fwd_agg = {0}, bwd_agg = {0};

  for (uint32_t qr = 0; qr < W; ++qr) {
    /* Extract Q_qr and dO_qr */
    for (size_t n = 0; n < N; ++n) {
      memcpy(q_s + n * T_loc * H * Dh,
             q + (n * T + qr * T_loc) * H * Dh,
             T_loc * H * Dh * sizeof(float));
      memcpy(do_s + n * T_loc * H * Dh,
             do_f + (n * T + qr * T_loc) * H * Dh,
             T_loc * H * Dh * sizeof(float));
    }
    for (size_t i = 0; i < lse_loc; ++i) {
      m_st[i] = -INFINITY;
      l_st[i] = 0.0;
    }
    memset(o_acc, 0, q_loc * sizeof(double));

    /* Visit KV blocks in reversed ring order to prove online softmax invariance. */
    et_ring_attn_stats_v1 rank_fwd = {0};
    for (uint32_t step = 0; step < W; ++step) {
      const uint32_t kvr = (qr + W - step) % W;
      for (size_t n = 0; n < N; ++n) {
        memcpy(k_s + n * T_loc * Hkv * Dh,
               k + (n * T + kvr * T_loc) * Hkv * Dh,
               T_loc * Hkv * Dh * sizeof(float));
        memcpy(v_s + n * T_loc * Hkv * Dh,
               v + (n * T + kvr * T_loc) * Hkv * Dh,
               T_loc * Hkv * Dh * sizeof(float));
      }
      CHECK(et_ring_attn_fwd_block_step_v1(&cfg, qr, kvr, q_s, k_s, v_s, m_st,
                                           l_st, o_acc, &rank_fwd) ==
            ET_RING_ATTN_OK);
    }
    CHECK(rank_fwd.total_block_steps == W);
    CHECK(rank_fwd.skipped_causal_blocks == (uint64_t)(W - 1u - qr));
    CHECK(rank_fwd.executed_blocks == (uint64_t)(qr + 1u));
    fwd_agg.total_block_steps += rank_fwd.total_block_steps;
    fwd_agg.executed_blocks += rank_fwd.executed_blocks;
    fwd_agg.skipped_causal_blocks += rank_fwd.skipped_causal_blocks;

    CHECK(et_ring_attn_fwd_finalize_v1(&cfg, m_st, l_st, o_acc, o_s, lse_s) ==
          ET_RING_ATTN_OK);
    for (size_t n = 0; n < N; ++n) {
      memcpy(o_blk + (n * T + qr * T_loc) * H * Dh,
             o_s + n * T_loc * H * Dh,
             T_loc * H * Dh * sizeof(float));
      memcpy(lse_blk + (n * T + qr * T_loc) * H,
             lse_s + n * T_loc * H,
             T_loc * H * sizeof(double));
    }

    /* Block backward */
    for (size_t idx = 0; idx < lse_loc; ++idx) {
      double acc = 0.0;
      for (size_t d = 0; d < Dh; ++d)
        acc += (double)do_s[idx * Dh + d] * (double)o_s[idx * Dh + d];
      d_vec[idx] = acc;
    }
    memset(dq_acc, 0, q_loc * sizeof(double));
    et_ring_attn_stats_v1 rank_bwd = {0};
    for (uint32_t kvr = 0; kvr < W; ++kvr) {
      for (size_t n = 0; n < N; ++n) {
        memcpy(k_s + n * T_loc * Hkv * Dh,
               k + (n * T + kvr * T_loc) * Hkv * Dh,
               T_loc * Hkv * Dh * sizeof(float));
        memcpy(v_s + n * T_loc * Hkv * Dh,
               v + (n * T + kvr * T_loc) * Hkv * Dh,
               T_loc * Hkv * Dh * sizeof(float));
      }
      memset(dk_acc, 0, kv_loc * sizeof(double));
      memset(dv_acc, 0, kv_loc * sizeof(double));
      CHECK(et_ring_attn_bwd_block_step_v1(&cfg, qr, kvr, q_s, do_s, lse_s,
                                           d_vec, k_s, v_s, dq_acc, dk_acc,
                                           dv_acc, &rank_bwd) ==
            ET_RING_ATTN_OK);
      for (size_t n = 0; n < N; ++n) {
        const size_t dst_off = (n * T + kvr * T_loc) * Hkv * Dh;
        const size_t src_off = n * T_loc * Hkv * Dh;
        for (size_t j = 0; j < T_loc * Hkv * Dh; ++j) {
          dk_global[dst_off + j] += dk_acc[src_off + j];
          dv_global[dst_off + j] += dv_acc[src_off + j];
        }
      }
    }
    CHECK(rank_bwd.skipped_causal_blocks == (uint64_t)(W - 1u - qr));
    bwd_agg.total_block_steps += rank_bwd.total_block_steps;
    bwd_agg.executed_blocks += rank_bwd.executed_blocks;
    bwd_agg.skipped_causal_blocks += rank_bwd.skipped_causal_blocks;

    for (size_t n = 0; n < N; ++n) {
      const size_t dst_off = (n * T + qr * T_loc) * H * Dh;
      const size_t src_off = n * T_loc * H * Dh;
      for (size_t j = 0; j < T_loc * H * Dh; ++j)
        dq_blk[dst_off + j] = (float)dq_acc[src_off + j];
    }
  }

  CHECK(fwd_agg.total_block_steps == 16u && fwd_agg.skipped_causal_blocks == 6u &&
        fwd_agg.executed_blocks == 10u);
  CHECK(bwd_agg.total_block_steps == 16u && bwd_agg.skipped_causal_blocks == 6u &&
        bwd_agg.executed_blocks == 10u);

  float *dk_blk = malloc(kv_elems * sizeof(float));
  float *dv_blk = malloc(kv_elems * sizeof(float));
  CHECK(dk_blk && dv_blk);
  for (size_t i = 0; i < kv_elems; ++i) {
    dk_blk[i] = (float)dk_global[i];
    dv_blk[i] = (float)dv_global[i];
  }

  CHECK(max_abs_diff_f32(o_ref, o_blk, q_elems) <= 1e-6);
  CHECK(max_abs_diff_f64(lse_ref, lse_blk, lse_elems) <= 1e-12);
  CHECK(max_abs_diff_f32(dq_ref, dq_blk, q_elems) <= 1e-6);
  CHECK(max_abs_diff_f32(dk_ref, dk_blk, kv_elems) <= 1e-6);
  CHECK(max_abs_diff_f32(dv_ref, dv_blk, kv_elems) <= 1e-6);

  free(q); free(k); free(v); free(do_f);
  free(o_ref); free(lse_ref); free(dq_ref); free(dk_ref); free(dv_ref);
  free(o_blk); free(lse_blk); free(dq_blk); free(dk_global); free(dv_global);
  free(dk_blk); free(dv_blk);
  free(q_s); free(k_s); free(v_s); free(do_s); free(o_s); free(lse_s);
  free(m_st); free(l_st); free(o_acc); free(d_vec);
  free(dq_acc); free(dk_acc); free(dv_acc);
}

/* ============================================================================
 * 2. Multi-Process Parity Matrix across W = 1, 2, 4 (MHA & GQA, Causal & Non)
 * ============================================================================ */
static void test_multiprocess_parity_matrix(void) {
  typedef struct shape_case {
    size_t N, T, H, Hkv, Dh;
    uint32_t causal;
    double scale;
    uint32_t seed;
  } shape_case;

  const shape_case cases[] = {
      {1u, 8u, 4u, 4u, 4u, 1u, 0.0, 11u},   /* MHA causal */
      {1u, 8u, 4u, 4u, 4u, 0u, 0.0, 12u},   /* MHA non-causal */
      {1u, 8u, 4u, 2u, 4u, 1u, 0.0, 13u},   /* GQA 2:1 causal */
      {1u, 8u, 4u, 2u, 4u, 0u, 0.0, 14u},   /* GQA 2:1 non-causal */
      {2u, 12u, 4u, 1u, 8u, 1u, 0.0, 15u},  /* MQA (GQA 4:1) batch=2 causal */
      {2u, 16u, 8u, 2u, 8u, 1u, 0.25, 16u}, /* Custom scale GQA 4:1 causal */
      {1u, 4u, 2u, 2u, 16u, 1u, 0.0, 17u},  /* T_local=1 at W=4 */
  };
  const uint32_t worlds[] = {1u, 2u, 4u};

  double worst_o = 0.0, worst_lse = 0.0, worst_dq = 0.0, worst_dk = 0.0,
         worst_dv = 0.0;

  for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
    const shape_case c = cases[ci];
    const size_t q_elems = c.N * c.T * c.H * c.Dh;
    const size_t kv_elems = c.N * c.T * c.Hkv * c.Dh;
    const size_t lse_elems = c.N * c.T * c.H;

    float *q = malloc(q_elems * sizeof(float));
    float *k = malloc(kv_elems * sizeof(float));
    float *v = malloc(kv_elems * sizeof(float));
    float *do_f = malloc(q_elems * sizeof(float));
    float *o_ref = malloc(q_elems * sizeof(float));
    double *lse_ref = malloc(lse_elems * sizeof(double));
    float *dq_ref = malloc(q_elems * sizeof(float));
    float *dk_ref = malloc(kv_elems * sizeof(float));
    float *dv_ref = malloc(kv_elems * sizeof(float));
    float *o_mp = malloc(q_elems * sizeof(float));
    double *lse_mp = malloc(lse_elems * sizeof(double));
    float *dq_mp = malloc(q_elems * sizeof(float));
    float *dk_mp = malloc(kv_elems * sizeof(float));
    float *dv_mp = malloc(kv_elems * sizeof(float));
    CHECK(q && k && v && do_f && o_ref && lse_ref && dq_ref && dk_ref &&
          dv_ref && o_mp && lse_mp && dq_mp && dk_mp && dv_mp);

    fill_deterministic(q, q_elems, c.seed * 10u + 1u, 0.0625f);
    fill_deterministic(k, kv_elems, c.seed * 10u + 2u, 0.0625f);
    fill_deterministic(v, kv_elems, c.seed * 10u + 3u, 0.0625f);
    fill_deterministic(do_f, q_elems, c.seed * 10u + 4u, 0.0625f);

    const et_ring_attn_config_v1 ref_cfg = {
        sizeof(ref_cfg), c.N, c.T, c.H, c.Hkv, c.Dh, 1u, c.causal, c.scale,
        10000};
    CHECK(et_ring_attn_dense_ref_forward_v1(&ref_cfg, q, k, v, o_ref,
                                            lse_ref) == ET_RING_ATTN_OK);
    CHECK(et_ring_attn_dense_ref_backward_v1(&ref_cfg, q, k, v, o_ref, lse_ref,
                                             do_f, dq_ref, dk_ref, dv_ref) ==
          ET_RING_ATTN_OK);

    for (size_t wi = 0; wi < 3u; ++wi) {
      const uint32_t W = worlds[wi];
      const et_ring_attn_config_v1 cfg = {
          sizeof(cfg), c.N, c.T, c.H, c.Hkv, c.Dh, W, c.causal, c.scale, 10000};
      et_ring_attn_stats_v1 fstats = {0}, bstats = {0};

      CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o_mp, lse_mp,
                                                 &fstats) == ET_RING_ATTN_OK);
      CHECK(et_ring_attn_multiprocess_backward_v1(&cfg, q, k, v, o_mp, lse_mp,
                                                  do_f, dq_mp, dk_mp, dv_mp,
                                                  &bstats) == ET_RING_ATTN_OK);

      const uint64_t expected_total = (uint64_t)W * (uint64_t)W;
      const uint64_t expected_skips =
          c.causal ? ((uint64_t)W * (uint64_t)(W - 1u) / 2u) : 0u;
      const uint64_t expected_exec = expected_total - expected_skips;
      const uint32_t expected_fwd_rounds = (W == 1u) ? 0u : (W - 1u);
      const uint32_t expected_bwd_rounds = (W == 1u) ? 0u : W;

      CHECK(fstats.total_block_steps == expected_total);
      CHECK(fstats.skipped_causal_blocks == expected_skips);
      CHECK(fstats.executed_blocks == expected_exec);
      CHECK(fstats.ring_rounds == expected_fwd_rounds);
      CHECK(fstats.bytes_sent == fstats.bytes_received);
      if (W > 1u)
        CHECK(fstats.bytes_sent > 0u);
      else
        CHECK(fstats.bytes_sent == 0u);

      CHECK(bstats.total_block_steps == expected_total);
      CHECK(bstats.skipped_causal_blocks == expected_skips);
      CHECK(bstats.executed_blocks == expected_exec);
      CHECK(bstats.ring_rounds == expected_bwd_rounds);
      CHECK(bstats.bytes_sent == bstats.bytes_received);
      if (W > 1u)
        CHECK(bstats.bytes_sent > 0u);
      else
        CHECK(bstats.bytes_sent == 0u);

      const double eo = max_abs_diff_f32(o_ref, o_mp, q_elems);
      const double else_d = max_abs_diff_f64(lse_ref, lse_mp, lse_elems);
      const double edq = max_abs_diff_f32(dq_ref, dq_mp, q_elems);
      const double edk = max_abs_diff_f32(dk_ref, dk_mp, kv_elems);
      const double edv = max_abs_diff_f32(dv_ref, dv_mp, kv_elems);

      CHECK(eo <= 1e-6);
      CHECK(else_d <= 1e-6);
      CHECK(edq <= 1e-6);
      CHECK(edk <= 1e-6);
      CHECK(edv <= 1e-6);

      if (eo > worst_o) worst_o = eo;
      if (else_d > worst_lse) worst_lse = else_d;
      if (edq > worst_dq) worst_dq = edq;
      if (edk > worst_dk) worst_dk = edk;
      if (edv > worst_dv) worst_dv = edv;
    }

    free(q); free(k); free(v); free(do_f);
    free(o_ref); free(lse_ref); free(dq_ref); free(dk_ref); free(dv_ref);
    free(o_mp); free(lse_mp); free(dq_mp); free(dk_mp); free(dv_mp);
  }

  printf("RING_ATTN parity W=1/2/4 vs dense_ref: PASS "
         "(max_err O=%.3e LSE=%.3e dQ=%.3e dK=%.3e dV=%.3e)\n",
         worst_o, worst_lse, worst_dq, worst_dk, worst_dv);
}

/* ============================================================================
 * 3. Exhaustive Central Finite-Difference Gradient Checks (Q, K, V)
 * ============================================================================ */
static double eval_weighted_loss(const et_ring_attn_config_v1 *cfg,
                                 const float *q, const float *k, const float *v,
                                 const float *w_probe, float *o_tmp,
                                 double *lse_tmp, size_t q_elems) {
  CHECK(et_ring_attn_multiprocess_forward_v1(cfg, q, k, v, o_tmp, lse_tmp,
                                             NULL) == ET_RING_ATTN_OK);
  double loss = 0.0;
  for (size_t i = 0; i < q_elems; ++i)
    loss += (double)o_tmp[i] * (double)w_probe[i];
  return loss;
}

static void test_finite_differences(void) {
  const uint32_t causal_modes[2] = {1u, 0u};
  const uint32_t worlds[3] = {1u, 2u, 4u};
  double worst_err = 0.0;
  size_t total_checked = 0u;

  for (size_t ci = 0; ci < 2u; ++ci) {
    for (size_t wi = 0; wi < 3u; ++wi) {
      /* N=1, T=4, H=4, Hkv=2 (GQA), Dh=2 => Q=32, K=16, V=16 -> check ALL 64 elems! */
      const et_ring_attn_config_v1 cfg = {
          sizeof(cfg), 1u, 4u, 4u, 2u, 2u, worlds[wi], causal_modes[ci], 0.0,
          10000};
      const size_t q_elems = 1u * 4u * 4u * 2u;
      const size_t kv_elems = 1u * 4u * 2u * 2u;

      float q[32], k[16], v[16], w_probe[32], o[32], o_tmp[32];
      double lse[16], lse_tmp[16];
      float dq[32], dk[16], dv[16];

      fill_deterministic(q, q_elems, (uint32_t)(500u + ci * 10u + wi), 0.0625f);
      fill_deterministic(k, kv_elems, (uint32_t)(600u + ci * 10u + wi), 0.0625f);
      fill_deterministic(v, kv_elems, (uint32_t)(700u + ci * 10u + wi), 0.0625f);
      fill_deterministic(w_probe, q_elems, (uint32_t)(800u + ci * 10u + wi),
                         0.0625f);

      CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
            ET_RING_ATTN_OK);
      CHECK(et_ring_attn_multiprocess_backward_v1(
                &cfg, q, k, v, o, lse, w_probe, dq, dk, dv, NULL) ==
            ET_RING_ATTN_OK);

      const float eps = 0.00390625f; /* exact 2^-8 */
      float *tensors[3] = {q, k, v};
      const float *grads[3] = {dq, dk, dv};
      const size_t counts[3] = {q_elems, kv_elems, kv_elems};

      for (size_t t = 0; t < 3u; ++t) {
        for (size_t i = 0; i < counts[t]; ++i) {
          const float orig = tensors[t][i];
          tensors[t][i] = orig + eps;
          const double lp =
              eval_weighted_loss(&cfg, q, k, v, w_probe, o_tmp, lse_tmp, q_elems);
          tensors[t][i] = orig - eps;
          const double lm =
              eval_weighted_loss(&cfg, q, k, v, w_probe, o_tmp, lse_tmp, q_elems);
          tensors[t][i] = orig;
          const double fd = (lp - lm) / (2.0 * (double)eps);
          const double an = (double)grads[t][i];
          const double err = fabs(an - fd);
          CHECK(err <= 1e-4 + 1e-2 * fabs(fd));
          if (err > worst_err)
            worst_err = err;
          ++total_checked;
        }
      }
    }
  }

  printf("RING_ATTN numerical FD gradients (Q/K/V, W=1/2/4, causal & non-causal): "
         "PASS (checked=%zu, max_abs_err=%.3e)\n",
         total_checked, worst_err);
}

/* ============================================================================
 * 4. Deadlock-Free Large-Payload Ring Exchange (> Kernel Socket Buffer)
 * ============================================================================ */
static void test_large_payload_ring_exchange(void) {
  const uint32_t W = 4u;
  const size_t payload_bytes = 512u * 1024u; /* 512 KiB > typical AF_UNIX SNDBUF */
  int edges[4][2];
  pid_t pids[4] = {-1, -1, -1, -1};
  for (uint32_t r = 0; r < W; ++r)
    CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, edges[r]) == 0);

  for (uint32_t r = 0; r < W; ++r) {
    pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
      const uint32_t prev = (r + W - 1u) % W;
      const int sfd = edges[r][0];
      const int rfd = edges[prev][1];
      for (uint32_t i = 0; i < W; ++i) {
        if (edges[i][0] != sfd) close(edges[i][0]);
        if (edges[i][1] != rfd) close(edges[i][1]);
      }
      et_ring_attn_comm_t *comm = NULL;
      int rc = 0;
      if (et_ring_attn_comm_create_v1(ET_RING_ATTN_ABI_MAJOR,
                                      ET_RING_ATTN_ABI_MINOR, r, W, sfd, rfd,
                                      &comm) != ET_RING_ATTN_OK)
        rc = 1;
      close(sfd);
      close(rfd);
      unsigned char *tx = malloc(payload_bytes);
      unsigned char *rx = malloc(payload_bytes);
      if (!rc && tx && rx) {
        memset(tx, (int)(0xA0u + r), payload_bytes);
        memset(rx, 0, payload_bytes);
        if (et_ring_attn_ring_exchange_v1(comm, tx, rx, payload_bytes,
                                          ET_RING_ATTN_PHASE_TEST, 0u, r, prev,
                                          5000) != ET_RING_ATTN_OK)
          rc = 2;
        for (size_t i = 0; !rc && i < payload_bytes; ++i) {
          if (rx[i] != (unsigned char)(0xA0u + prev))
            rc = 3;
        }
      } else {
        rc = 4;
      }
      et_ring_attn_comm_destroy_v1(comm);
      free(tx);
      free(rx);
      _exit(rc);
    }
    pids[r] = pid;
  }

  for (uint32_t r = 0; r < W; ++r) {
    close(edges[r][0]);
    close(edges[r][1]);
  }
  for (uint32_t r = 0; r < W; ++r) {
    int wst = 0;
    CHECK(waitpid(pids[r], &wst, 0) == pids[r]);
    CHECK(WIFEXITED(wst) && WEXITSTATUS(wst) == 0);
  }
  printf("RING_ATTN deadlock-free 512KiB x 4-rank AF_UNIX ring exchange: PASS\n");
}

/* ============================================================================
 * 5. Negative Tests + Timeout + Peer Disconnect + Zero FD Leak Proof
 * ============================================================================ */
static void test_negatives_and_fd_leaks(void) {
  const int fds_start = count_open_fds();

  et_ring_attn_config_v1 cfg = {
      sizeof(cfg), 1u, 8u, 4u, 2u, 4u, 4u, 1u, 0.0, 2000};
  float q[128], k[64], v[64], do_f[128], o[128], dq[128], dk[64], dv[64];
  double lse[32];
  fill_deterministic(q, 128u, 1u, 0.125f);
  fill_deterministic(k, 64u, 2u, 0.125f);
  fill_deterministic(v, 64u, 3u, 0.125f);
  fill_deterministic(do_f, 128u, 4u, 0.125f);
  fill_deterministic(o, 128u, 5u, 0.125f);
  for (size_t i = 0; i < 32u; ++i)
    lse[i] = 1.25;

  /* Plant sentinels in output buffers. */
  o[0] = 19.5f;
  lse[0] = -42.25;
  dq[0] = 7.75f;
  dk[0] = -3.5f;
  dv[0] = 13.25f;

  /* 5a. Non-divisible T % W != 0 */
  cfg.seq_len = 6u;
  cfg.world_size = 4u;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_SHAPE_MISMATCH);
  cfg.seq_len = 5u;
  cfg.world_size = 2u;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_SHAPE_MISMATCH);

  /* 5b. Invalid world sizes */
  cfg.seq_len = 8u;
  cfg.world_size = 0u;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_UNSUPPORTED);
  cfg.world_size = 3u;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_UNSUPPORTED);
  cfg.world_size = 8u;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_UNSUPPORTED);

  /* 5c. Invalid GQA, scale, causal, timeout */
  cfg.world_size = 4u;
  cfg.num_kv_heads = 3u;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_SHAPE_MISMATCH);
  cfg.num_kv_heads = 2u;
  cfg.causal = 2u;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_INVALID_ARG);
  cfg.causal = 1u;
  cfg.scale = -0.5;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_INVALID_ARG);
  cfg.scale = NAN;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_INVALID_ARG);
  cfg.scale = 0.0;
  cfg.timeout_ms = 0;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_INVALID_ARG);
  cfg.timeout_ms = 2000;

  /* 5d. Non-finite Q, K, V, dO, LSE */
  const float q_saved = q[5];
  q[5] = NAN;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_NONFINITE);
  q[5] = INFINITY;
  CHECK(et_ring_attn_dense_ref_forward_v1(&cfg, q, k, v, o, lse) ==
        ET_RING_ATTN_ERR_NONFINITE);
  q[5] = q_saved;

  const float k_saved = k[3];
  k[3] = -INFINITY;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_NONFINITE);
  k[3] = k_saved;

  const float v_saved = v[2];
  v[2] = NAN;
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse, NULL) ==
        ET_RING_ATTN_ERR_NONFINITE);
  v[2] = v_saved;

  const float do_saved = do_f[7];
  do_f[7] = INFINITY;
  CHECK(et_ring_attn_multiprocess_backward_v1(&cfg, q, k, v, o, lse, do_f, dq,
                                              dk, dv, NULL) ==
        ET_RING_ATTN_ERR_NONFINITE);
  do_f[7] = do_saved;

  /* 5e. Misaligned buffers */
  unsigned char raw_buf[1024];
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, raw_buf + 1, k, v, o, lse,
                                             NULL) ==
        ET_RING_ATTN_ERR_INVALID_ARG);
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, raw_buf + 4,
                                             NULL) ==
        ET_RING_ATTN_ERR_INVALID_ARG);

  /* 5f. Aliased buffers */
  CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, q, lse, NULL) ==
        ET_RING_ATTN_ERR_INVALID_ARG);
  CHECK(et_ring_attn_multiprocess_backward_v1(&cfg, q, k, v, o, lse, do_f, dq,
                                              dk, dk, NULL) ==
        ET_RING_ATTN_ERR_INVALID_ARG);
  CHECK(et_ring_attn_multiprocess_backward_v1(&cfg, q, k, v, o, lse, do_f, q,
                                              dk, dv, NULL) ==
        ET_RING_ATTN_ERR_INVALID_ARG);

  /* Verify sentinels remained completely untouched! */
  CHECK(o[0] == 19.5f && lse[0] == -42.25 && dq[0] == 7.75f &&
        dk[0] == -3.5f && dv[0] == 13.25f);

  /* 5g. Bad comm creation (ABI mismatch, non-socket FD, duplicate FD for W=4) */
  et_ring_attn_comm_t *bad_comm = NULL;
  CHECK(et_ring_attn_comm_create_v1(99u, 0u, 0u, 1u, -1, -1, &bad_comm) ==
        ET_RING_ATTN_ERR_INVALID_ARG);
  CHECK(et_ring_attn_comm_create_v1(1u, 1u, 0u, 1u, -1, -1, &bad_comm) ==
        ET_RING_ATTN_ERR_INVALID_ARG);
  CHECK(et_ring_attn_comm_create_v1(1u, 0u, 1u, 1u, -1, -1, &bad_comm) ==
        ET_RING_ATTN_ERR_INVALID_ARG);
  CHECK(et_ring_attn_comm_create_v1(1u, 0u, 0u, 3u, -1, -1, &bad_comm) ==
        ET_RING_ATTN_ERR_UNSUPPORTED);
  int pipefds[2] = {-1, -1};
  CHECK(pipe(pipefds) == 0);
  CHECK(et_ring_attn_comm_create_v1(1u, 0u, 0u, 2u, pipefds[0], pipefds[1],
                                    &bad_comm) == ET_RING_ATTN_ERR_INVALID_ARG);
  close(pipefds[0]);
  close(pipefds[1]);

  /* 5h. Peer disconnect during ring forward -> ET_RING_ATTN_ERR_PEER_DISCONNECTED */
  {
    int e0[2], e1[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, e0) == 0);
    CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, e1) == 0);
    pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
      close(e0[0]);
      close(e1[1]);
      /* Immediately disconnect both ends. */
      close(e0[1]);
      close(e1[0]);
      _exit(0);
    }
    close(e0[1]);
    close(e1[0]);

    et_ring_attn_comm_t *comm = NULL;
    CHECK(et_ring_attn_comm_create_v1(1u, 0u, 0u, 2u, e0[0], e1[1], &comm) ==
          ET_RING_ATTN_OK);
    close(e0[0]);
    close(e1[1]);

    et_ring_attn_config_v1 c2 = {
        sizeof(c2), 1u, 8u, 4u, 2u, 4u, 2u, 1u, 0.0, 2000};
    float o_loc[64];
    double lse_loc[16];
    o_loc[0] = 5.5f;
    lse_loc[0] = -9.25;
    const et_ring_attn_status_t st =
        et_ring_attn_forward_v1(comm, &c2, q, k, v, o_loc, lse_loc, NULL);
    CHECK(st == ET_RING_ATTN_ERR_PEER_DISCONNECTED);
    CHECK(o_loc[0] == 5.5f && lse_loc[0] == -9.25);

    et_ring_attn_comm_destroy_v1(comm);
    int wst = 0;
    CHECK(waitpid(pid, &wst, 0) == pid);
  }

  /* 5i. Peer stall -> ET_RING_ATTN_ERR_TIMEOUT */
  {
    int e0[2], e1[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, e0) == 0);
    CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, e1) == 0);
    pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
      close(e0[0]);
      close(e1[1]);
      /* Hold sockets open silently until killed. */
      struct timespec ts = {5, 0};
      nanosleep(&ts, NULL);
      close(e0[1]);
      close(e1[0]);
      _exit(0);
    }
    close(e0[1]);
    close(e1[0]);

    et_ring_attn_comm_t *comm = NULL;
    CHECK(et_ring_attn_comm_create_v1(1u, 0u, 0u, 2u, e0[0], e1[1], &comm) ==
          ET_RING_ATTN_OK);
    close(e0[0]);
    close(e1[1]);

    et_ring_attn_config_v1 c2 = {
        sizeof(c2), 1u, 8u, 4u, 2u, 4u, 2u, 1u, 0.0, 80};
    float o_loc[64];
    double lse_loc[16];
    o_loc[0] = 3.25f;
    lse_loc[0] = -1.75;
    const int64_t t0 = now_ms();
    const et_ring_attn_status_t st =
        et_ring_attn_forward_v1(comm, &c2, q, k, v, o_loc, lse_loc, NULL);
    const int64_t dt = now_ms() - t0;
    CHECK(st == ET_RING_ATTN_ERR_TIMEOUT);
    CHECK(dt >= 60 && dt < 2000);
    CHECK(o_loc[0] == 3.25f && lse_loc[0] == -1.75);

    et_ring_attn_comm_destroy_v1(comm);
    kill(pid, SIGKILL);
    int wst = 0;
    CHECK(waitpid(pid, &wst, 0) == pid);
  }

  /* 5j. Protocol mismatch -> ET_RING_ATTN_ERR_PROTOCOL */
  {
    int e0[2], e1[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, e0) == 0);
    CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, e1) == 0);
    pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
      close(e0[0]);
      close(e1[1]);
      unsigned char junk[32];
      memset(junk, 0x55, sizeof(junk));
      (void)send(e1[0], junk, sizeof(junk), MSG_NOSIGNAL);
      /* Drain whatever rank 0 sends */
      unsigned char sink[64];
      (void)recv(e0[1], sink, sizeof(sink), 0);
      close(e0[1]);
      close(e1[0]);
      _exit(0);
    }
    close(e0[1]);
    close(e1[0]);

    et_ring_attn_comm_t *comm = NULL;
    CHECK(et_ring_attn_comm_create_v1(1u, 0u, 0u, 2u, e0[0], e1[1], &comm) ==
          ET_RING_ATTN_OK);
    close(e0[0]);
    close(e1[1]);

    unsigned char tx[16] = {0}, rx[16] = {0};
    CHECK(et_ring_attn_ring_exchange_v1(comm, tx, rx, sizeof(tx),
                                        ET_RING_ATTN_PHASE_TEST, 0u, 0u, 1u,
                                        2000) == ET_RING_ATTN_ERR_PROTOCOL);
    et_ring_attn_comm_destroy_v1(comm);
    int wst = 0;
    CHECK(waitpid(pid, &wst, 0) == pid);
  }

  const int fds_end = count_open_fds();
  CHECK(fds_start == fds_end);
  printf("RING_ATTN negative rejections, disconnect, timeout, protocol & "
         "zero-FD-leak (%d==%d): PASS\n",
         fds_start, fds_end);
}

/* ============================================================================
 * 6. Oracle Emitter (--oracle) for Independent Python Binary64 Verification
 * ============================================================================ */
static void print_f32_hex(const float *buf, size_t n) {
  for (size_t i = 0; i < n; ++i)
    printf("%s%a", i ? " " : "", (double)buf[i]);
  printf("\n");
}

static void print_f64_hex(const double *buf, size_t n) {
  for (size_t i = 0; i < n; ++i)
    printf("%s%a", i ? " " : "", buf[i]);
  printf("\n");
}

static int run_oracle_emitter(void) {
  typedef struct ocase {
    size_t N, T, H, Hkv, Dh;
    uint32_t W, causal;
    uint32_t seed;
  } ocase;

  const ocase cases[] = {
      {1u, 4u, 2u, 2u, 4u, 1u, 1u, 1001u}, /* W=1 MHA causal */
      {1u, 8u, 4u, 4u, 4u, 2u, 1u, 1002u}, /* W=2 MHA causal */
      {1u, 8u, 4u, 4u, 4u, 4u, 1u, 1003u}, /* W=4 MHA causal */
      {1u, 8u, 4u, 4u, 4u, 4u, 0u, 1004u}, /* W=4 MHA non-causal */
      {1u, 8u, 4u, 2u, 4u, 2u, 1u, 1005u}, /* W=2 GQA causal */
      {1u, 8u, 4u, 2u, 4u, 4u, 1u, 1006u}, /* W=4 GQA causal */
      {1u, 8u, 4u, 2u, 4u, 4u, 0u, 1007u}, /* W=4 GQA non-causal */
      {2u, 8u, 4u, 1u, 4u, 4u, 1u, 1008u}, /* W=4 MQA batch=2 causal */
      {2u, 8u, 4u, 2u, 4u, 2u, 0u, 1009u}, /* W=2 GQA batch=2 non-causal */
  };

  for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
    const ocase c = cases[ci];
    const size_t q_elems = c.N * c.T * c.H * c.Dh;
    const size_t kv_elems = c.N * c.T * c.Hkv * c.Dh;
    const size_t lse_elems = c.N * c.T * c.H;

    float *q = malloc(q_elems * sizeof(float));
    float *k = malloc(kv_elems * sizeof(float));
    float *v = malloc(kv_elems * sizeof(float));
    float *do_f = malloc(q_elems * sizeof(float));
    float *o = malloc(q_elems * sizeof(float));
    double *lse = malloc(lse_elems * sizeof(double));
    float *dq = malloc(q_elems * sizeof(float));
    float *dk = malloc(kv_elems * sizeof(float));
    float *dv = malloc(kv_elems * sizeof(float));
    CHECK(q && k && v && do_f && o && lse && dq && dk && dv);

    fill_deterministic(q, q_elems, c.seed + 1u, 0.0625f);
    fill_deterministic(k, kv_elems, c.seed + 2u, 0.0625f);
    fill_deterministic(v, kv_elems, c.seed + 3u, 0.0625f);
    fill_deterministic(do_f, q_elems, c.seed + 4u, 0.0625f);

    const et_ring_attn_config_v1 cfg = {
        sizeof(cfg), c.N, c.T, c.H, c.Hkv, c.Dh, c.W, c.causal, 0.0, 10000};
    et_ring_attn_stats_v1 fstats = {0}, bstats = {0};
    CHECK(et_ring_attn_multiprocess_forward_v1(&cfg, q, k, v, o, lse,
                                               &fstats) == ET_RING_ATTN_OK);
    CHECK(et_ring_attn_multiprocess_backward_v1(&cfg, q, k, v, o, lse, do_f, dq,
                                                dk, dv, &bstats) ==
          ET_RING_ATTN_OK);

    printf("CASE %zu %zu %zu %zu %zu %u %u %llu %llu\n", c.N, c.T, c.H, c.Hkv,
           c.Dh, c.W, c.causal,
           (unsigned long long)fstats.skipped_causal_blocks,
           (unsigned long long)bstats.skipped_causal_blocks);
    print_f32_hex(q, q_elems);
    print_f32_hex(k, kv_elems);
    print_f32_hex(v, kv_elems);
    print_f32_hex(do_f, q_elems);
    print_f32_hex(o, q_elems);
    print_f64_hex(lse, lse_elems);
    print_f32_hex(dq, q_elems);
    print_f32_hex(dk, kv_elems);
    print_f32_hex(dv, kv_elems);

    free(q); free(k); free(v); free(do_f);
    free(o); free(lse); free(dq); free(dk); free(dv);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0)
    return run_oracle_emitter();

  const int fds_before = count_open_fds();
  test_block_step_unit();
  test_multiprocess_parity_matrix();
  test_finite_differences();
  test_large_payload_ring_exchange();
  test_negatives_and_fd_leaks();
  const int fds_after = count_open_fds();
  CHECK(fds_before == fds_after);
  printf("RING_ATTN native C11 test suite: ALL PASS (open_fds=%d)\n", fds_after);
  return 0;
}
