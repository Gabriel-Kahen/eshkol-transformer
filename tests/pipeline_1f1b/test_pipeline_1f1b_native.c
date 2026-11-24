#define _GNU_SOURCE
#include "eshkol_transformer/pipeline_1f1b_abi.h"
#include "pipeline_1f1b_eshkol_bridge.h"

#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static void die_check(const char *msg, int line) {
  fprintf(stderr, "PIPELINE_1F1B NATIVE TEST FAIL (line %d): %s\n", line, msg);
  exit(1);
}

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond))                                                               \
      die_check(#cond, __LINE__);                                              \
  } while (0)

static int count_open_fds(void) {
  DIR *d = opendir("/proc/self/fd");
  if (d == NULL)
    die_check("cannot open /proc/self/fd", __LINE__);
  int dfd = dirfd(d);
  int count = 0;
  struct dirent *ent;
  while ((ent = readdir(d)) != NULL) {
    if (ent->d_name[0] == '.')
      continue;
    int fd = atoi(ent->d_name);
    if (fd == dfd)
      continue;
    ++count;
  }
  closedir(d);
  return count;
}

static double det_val(size_t idx, uint32_t salt, double scale) {
  uint64_t x = (uint64_t)(idx + 1u) * UINT64_C(6364136223846793005) +
               (uint64_t)salt * UINT64_C(1442695040888963407) + UINT64_C(1);
  x ^= x >> 33;
  x *= UINT64_C(0xff51afd7ed558ccd);
  x ^= x >> 33;
  int64_t bucket = (int64_t)(x % 2001u) - 1000;
  return ((double)bucket / 1000.0) * scale;
}

static void fill_params(const et_pp_config_t *cfg, double *params,
                        uint32_t salt) {
  size_t total_layers =
      (size_t)cfg->num_stages * (size_t)cfg->layers_per_stage;
  size_t per_layer = et_pp_layer_param_count_v1(cfg->d_model, cfg->d_ff);
  for (size_t l = 0; l < total_layers; ++l) {
    double *lp = params + l * per_layer;
    for (uint32_t j = 0; j < cfg->d_model; ++j)
      lp[j] = 1.0 + det_val(l * 97u + j, salt + 1u, 0.15);
    for (size_t k = cfg->d_model; k < per_layer; ++k)
      lp[k] = det_val(l * 1009u + k, salt + 2u, 0.25);
  }
  double *head = params + total_layers * per_layer;
  size_t head_w = (size_t)cfg->out_dim * (size_t)cfg->d_model;
  for (size_t k = 0; k < head_w; ++k)
    head[k] = det_val(k, salt + 3u, 0.25);
  for (uint32_t v = 0; v < cfg->out_dim; ++v)
    head[head_w + v] = det_val(v, salt + 4u, 0.10);
}

static void fill_inputs(const et_pp_config_t *cfg, double *inputs,
                        uint32_t salt) {
  size_t n = (size_t)cfg->num_microbatches * (size_t)cfg->tokens_per_mb *
             (size_t)cfg->d_model;
  for (size_t i = 0; i < n; ++i)
    inputs[i] = det_val(i, salt, 0.8);
}

static void fill_targets_ce(const et_pp_config_t *cfg, int32_t *targets,
                            uint32_t salt) {
  size_t n = (size_t)cfg->num_microbatches * (size_t)cfg->tokens_per_mb;
  for (size_t i = 0; i < n; ++i)
    targets[i] = (int32_t)((i * 7u + salt * 3u + 1u) % cfg->out_dim);
}

static void fill_targets_mse(const et_pp_config_t *cfg, double *targets,
                             uint32_t salt) {
  size_t n = (size_t)cfg->num_microbatches * (size_t)cfg->tokens_per_mb *
             (size_t)cfg->out_dim;
  for (size_t i = 0; i < n; ++i)
    targets[i] = det_val(i, salt, 0.5);
}

static double max_abs_diff(const double *a, const double *b, size_t n) {
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = fabs(a[i] - b[i]);
    if (d > m)
      m = d;
  }
  return m;
}

static et_pp_config_t make_cfg(uint32_t P, uint32_t lps, uint32_t M,
                               int32_t sched_kind, int32_t loss_type) {
  et_pp_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.abi_major = ET_PP_ABI_MAJOR;
  cfg.abi_minor = ET_PP_ABI_MINOR;
  cfg.num_stages = P;
  cfg.layers_per_stage = lps;
  cfg.num_microbatches = M;
  cfg.tokens_per_mb = 3u;
  cfg.d_model = 8u;
  cfg.d_ff = 16u;
  cfg.out_dim = 6u;
  cfg.schedule_kind = sched_kind;
  cfg.loss_type = loss_type;
  cfg.timeout_ms = 5000u;
  cfg.rms_eps = 1e-5;
  return cfg;
}

/* ------------------------------------------------------------------------- */
/* 1. Schedule invariants & 1F1B vs GPipe peak activation proof              */
/* ------------------------------------------------------------------------- */

static void test_schedules(void) {
  const uint32_t ps[2] = {2u, 4u};
  const uint32_t ms[3] = {4u, 8u, 16u};

  for (size_t pi = 0; pi < 2; ++pi) {
    uint32_t P = ps[pi];
    for (size_t mi = 0; mi < 3; ++mi) {
      uint32_t M = ms[mi];
      double expected_bubble = (double)(P - 1u) / (double)(M + P - 1u);

      for (uint32_t s = 0; s < P; ++s) {
        et_pp_schedule_t s_1f1b;
        et_pp_schedule_t s_gpipe;
        CHECK(et_pp_build_schedule_v1(P, s, M, ET_PP_SCHEDULE_1F1B, &s_1f1b) ==
              ET_PP_OK);
        CHECK(et_pp_build_schedule_v1(P, s, M, ET_PP_SCHEDULE_GPIPE,
                                      &s_gpipe) == ET_PP_OK);

        CHECK(s_1f1b.step_count == 2u * M);
        CHECK(s_1f1b.num_warmup_steps == P - 1u - s);
        CHECK(s_1f1b.num_steady_pairs == M - (P - 1u - s));
        CHECK(s_1f1b.num_cooldown_steps == P - 1u - s);
        CHECK(s_1f1b.peak_live_activations == P - s);
        CHECK(s_1f1b.peak_live_activations <= P);
        CHECK(fabs(s_1f1b.bubble_fraction - expected_bubble) <= 1e-15);

        CHECK(s_gpipe.step_count == 2u * M);
        CHECK(s_gpipe.peak_live_activations == M);
        CHECK(fabs(s_gpipe.bubble_fraction - expected_bubble) <= 1e-15);

        if (M > P)
          CHECK(s_1f1b.peak_live_activations < s_gpipe.peak_live_activations);

        /* Verify every microbatch executes F before B and live count returns to 0 */
        int seen_f[ET_PP_MAX_MICROBATCHES];
        int seen_b[ET_PP_MAX_MICROBATCHES];
        memset(seen_f, 0, sizeof(seen_f));
        memset(seen_b, 0, sizeof(seen_b));
        uint32_t live = 0;
        uint32_t max_live = 0;
        for (uint32_t k = 0; k < s_1f1b.step_count; ++k) {
          uint32_t mb = s_1f1b.steps[k].microbatch_id;
          CHECK(mb < M);
          if (s_1f1b.steps[k].op == ET_PP_OP_FORWARD) {
            CHECK(!seen_f[mb] && !seen_b[mb]);
            seen_f[mb] = 1;
            ++live;
            if (live > max_live)
              max_live = live;
          } else {
            CHECK(s_1f1b.steps[k].op == ET_PP_OP_BACKWARD);
            CHECK(seen_f[mb] && !seen_b[mb]);
            seen_b[mb] = 1;
            CHECK(live > 0u);
            --live;
          }
          CHECK(s_1f1b.steps[k].live_after == live);
        }
        CHECK(live == 0u);
        CHECK(max_live == P - s);
        for (uint32_t m = 0; m < M; ++m)
          CHECK(seen_f[m] && seen_b[m]);
      }
    }
  }
  printf("PIPELINE_1F1B NATIVE SCHEDULES P={2,4} M={4,8,16} PASS\n");
}

/* ------------------------------------------------------------------------- */
/* 2. Numerical Central Finite-Difference Gradient Checks                    */
/* ------------------------------------------------------------------------- */

static void test_numerical_gradients(void) {
  /* Part A: Exhaustive single-layer central FD check on x and all layer params */
  const uint32_t T = 3u, D = 6u, F = 10u;
  const double eps = 1e-6;
  const double rms_eps = 1e-5;
  size_t xcnt = (size_t)T * (size_t)D;
  size_t pcnt = et_pp_layer_param_count_v1(D, F);

  double *x = (double *)calloc(xcnt, sizeof(double));
  double *dy = (double *)calloc(xcnt, sizeof(double));
  double *dx = (double *)calloc(xcnt, sizeof(double));
  double *yp = (double *)calloc(xcnt, sizeof(double));
  double *ym = (double *)calloc(xcnt, sizeof(double));
  double *p = (double *)calloc(pcnt, sizeof(double));
  double *dp = (double *)calloc(pcnt, sizeof(double));
  CHECK(x && dy && dx && yp && ym && p && dp);

  for (size_t i = 0; i < xcnt; ++i) {
    x[i] = det_val(i, 10u, 0.9);
    dy[i] = det_val(i, 11u, 0.7);
  }
  for (uint32_t j = 0; j < D; ++j)
    p[j] = 1.0 + det_val(j, 12u, 0.2);
  for (size_t k = D; k < pcnt; ++k)
    p[k] = det_val(k, 13u, 0.3);

  CHECK(et_pp_layer_backward_v1(T, D, F, rms_eps, x, p, dy, dx, dp) == ET_PP_OK);

  double max_layer_x_err = 0.0;
  for (size_t i = 0; i < xcnt; ++i) {
    double orig = x[i];
    x[i] = orig + eps;
    CHECK(et_pp_layer_forward_v1(T, D, F, rms_eps, x, p, yp) == ET_PP_OK);
    x[i] = orig - eps;
    CHECK(et_pp_layer_forward_v1(T, D, F, rms_eps, x, p, ym) == ET_PP_OK);
    x[i] = orig;

    double lp = 0.0, lm = 0.0;
    for (size_t j = 0; j < xcnt; ++j) {
      lp += yp[j] * dy[j];
      lm += ym[j] * dy[j];
    }
    double num = (lp - lm) / (2.0 * eps);
    double err = fabs(num - dx[i]);
    if (err > max_layer_x_err)
      max_layer_x_err = err;
  }
  CHECK(max_layer_x_err <= 1e-8);

  double max_layer_p_err = 0.0;
  for (size_t i = 0; i < pcnt; ++i) {
    double orig = p[i];
    p[i] = orig + eps;
    CHECK(et_pp_layer_forward_v1(T, D, F, rms_eps, x, p, yp) == ET_PP_OK);
    p[i] = orig - eps;
    CHECK(et_pp_layer_forward_v1(T, D, F, rms_eps, x, p, ym) == ET_PP_OK);
    p[i] = orig;

    double lp = 0.0, lm = 0.0;
    for (size_t j = 0; j < xcnt; ++j) {
      lp += yp[j] * dy[j];
      lm += ym[j] * dy[j];
    }
    double num = (lp - lm) / (2.0 * eps);
    double err = fabs(num - dp[i]);
    if (err > max_layer_p_err)
      max_layer_p_err = err;
  }
  CHECK(max_layer_p_err <= 1e-8);

  free(x);
  free(dy);
  free(dx);
  free(yp);
  free(ym);
  free(p);
  free(dp);

  /* Part B: Full 4-layer model central FD check (both Cross-Entropy & MSE) */
  double max_full_ce_err = 0.0;
  double max_full_mse_err = 0.0;
  for (int loss_type = 0; loss_type <= 1; ++loss_type) {
    et_pp_config_t cfg = make_cfg(1u, 4u, 4u, ET_PP_SCHEDULE_1F1B, loss_type);
    cfg.tokens_per_mb = 2u;
    cfg.d_model = 4u;
    cfg.d_ff = 6u;
    cfg.out_dim = 4u;
    size_t total_p = 0;
    CHECK(et_pp_param_count_v1(&cfg, &total_p) == ET_PP_OK);
    size_t in_cnt = (size_t)cfg.num_microbatches * (size_t)cfg.tokens_per_mb *
                    (size_t)cfg.d_model;
    size_t tok_cnt = (size_t)cfg.num_microbatches * (size_t)cfg.tokens_per_mb;

    double *inputs = (double *)calloc(in_cnt, sizeof(double));
    int32_t *t_i32 = (int32_t *)calloc(tok_cnt, sizeof(int32_t));
    double *t_f64 = (double *)calloc(tok_cnt * cfg.out_dim, sizeof(double));
    double *params = (double *)calloc(total_p, sizeof(double));
    double *grads = (double *)calloc(total_p, sizeof(double));
    double *dummy_g = (double *)calloc(total_p, sizeof(double));
    CHECK(inputs && t_i32 && t_f64 && params && grads && dummy_g);

    fill_inputs(&cfg, inputs, 20u + (uint32_t)loss_type);
    fill_targets_ce(&cfg, t_i32, 21u);
    fill_targets_mse(&cfg, t_f64, 22u);
    fill_params(&cfg, params, 23u + (uint32_t)loss_type);

    double base_loss = 0.0;
    CHECK(et_pp_run_reference_step_v1(
              &cfg, inputs, loss_type == 0 ? t_i32 : NULL,
              loss_type == 1 ? t_f64 : NULL, params, grads, &base_loss) ==
          ET_PP_OK);

    double max_err = 0.0;
    for (size_t i = 0; i < total_p; ++i) {
      double orig = params[i];
      double lp = 0.0, lm = 0.0;
      params[i] = orig + eps;
      CHECK(et_pp_run_reference_step_v1(
                &cfg, inputs, loss_type == 0 ? t_i32 : NULL,
                loss_type == 1 ? t_f64 : NULL, params, dummy_g, &lp) ==
            ET_PP_OK);
      params[i] = orig - eps;
      CHECK(et_pp_run_reference_step_v1(
                &cfg, inputs, loss_type == 0 ? t_i32 : NULL,
                loss_type == 1 ? t_f64 : NULL, params, dummy_g, &lm) ==
            ET_PP_OK);
      params[i] = orig;

      double num = (lp - lm) / (2.0 * eps);
      double err = fabs(num - grads[i]);
      if (err > max_err)
        max_err = err;
    }
    CHECK(max_err <= 1e-8);
    if (loss_type == 0)
      max_full_ce_err = max_err;
    else
      max_full_mse_err = max_err;

    free(inputs);
    free(t_i32);
    free(t_f64);
    free(params);
    free(grads);
    free(dummy_g);
  }

  printf("PIPELINE_1F1B NATIVE FD_GRADIENTS layer_x=%.3e layer_p=%.3e "
         "full_ce=%.3e full_mse=%.3e PASS\n",
         max_layer_x_err, max_layer_p_err, max_full_ce_err, max_full_mse_err);
}

/* ------------------------------------------------------------------------- */
/* 3. Multi-Process P=2, P=4 Parity vs P=1 Unpipelined + 10-Step SGD         */
/* ------------------------------------------------------------------------- */

static void test_multiprocess_parity_and_sgd(void) {
  int fds_start = count_open_fds();
  const uint32_t ms[2] = {4u, 8u};
  double worst_loss_err = 0.0;
  double worst_grad_err = 0.0;

  for (int loss_type = 0; loss_type <= 1; ++loss_type) {
    for (size_t mi = 0; mi < 2; ++mi) {
      uint32_t M = ms[mi];
      et_pp_config_t cfg_ref =
          make_cfg(1u, 4u, M, ET_PP_SCHEDULE_1F1B, loss_type);
      size_t total_p = 0;
      CHECK(et_pp_param_count_v1(&cfg_ref, &total_p) == ET_PP_OK);
      size_t in_cnt = (size_t)M * (size_t)cfg_ref.tokens_per_mb *
                      (size_t)cfg_ref.d_model;
      size_t tok_cnt = (size_t)M * (size_t)cfg_ref.tokens_per_mb;

      double *inputs = (double *)calloc(in_cnt, sizeof(double));
      int32_t *t_i32 = (int32_t *)calloc(tok_cnt, sizeof(int32_t));
      double *t_f64 =
          (double *)calloc(tok_cnt * cfg_ref.out_dim, sizeof(double));
      double *params = (double *)calloc(total_p, sizeof(double));
      double *grads_ref = (double *)calloc(total_p, sizeof(double));
      double *grads_pp = (double *)calloc(total_p, sizeof(double));
      CHECK(inputs && t_i32 && t_f64 && params && grads_ref && grads_pp);

      fill_inputs(&cfg_ref, inputs, 40u + (uint32_t)loss_type * 10u + M);
      fill_targets_ce(&cfg_ref, t_i32, 41u + M);
      fill_targets_mse(&cfg_ref, t_f64, 42u + M);
      fill_params(&cfg_ref, params, 43u + (uint32_t)loss_type);

      double loss_ref = 0.0;
      CHECK(et_pp_run_reference_step_v1(
                &cfg_ref, inputs, loss_type == 0 ? t_i32 : NULL,
                loss_type == 1 ? t_f64 : NULL, params, grads_ref, &loss_ref) ==
            ET_PP_OK);

      /* Compare P=2 (2x2 layers) and P=4 (4x1 layers) for both 1F1B and GPipe */
      const uint32_t stage_cfgs[2][2] = {{2u, 2u}, {4u, 1u}};
      for (int si = 0; si < 2; ++si) {
        uint32_t P = stage_cfgs[si][0];
        uint32_t lps = stage_cfgs[si][1];
        for (int sk = 0; sk <= 1; ++sk) {
          et_pp_config_t cfg_pp = make_cfg(P, lps, M, sk, loss_type);
          et_pp_run_telemetry_t telem;
          memset(&telem, 0, sizeof(telem));
          fflush(NULL);
          CHECK(et_pp_run_multiprocess_step_v1(
                    &cfg_pp, inputs, loss_type == 0 ? t_i32 : NULL,
                    loss_type == 1 ? t_f64 : NULL, params, grads_pp, &telem) ==
                ET_PP_OK);
          CHECK(count_open_fds() == fds_start);

          CHECK(telem.distinct_pid_count == P);
          for (uint32_t s = 0; s < P; ++s) {
            uint32_t expected_peak =
                (sk == ET_PP_SCHEDULE_1F1B) ? (P - s) : M;
            CHECK(telem.stage_peak_live[s] == expected_peak);
            CHECK(telem.stage_final_live[s] == 0u);
            CHECK(telem.stage_fwd_count[s] == M);
            CHECK(telem.stage_bwd_count[s] == M);
            if (s + 1u < P)
              CHECK(telem.stage_bytes_sent[s] > 0u);
            if (s > 0u)
              CHECK(telem.stage_bytes_recv[s] > 0u);
          }

          double lerr = fabs(telem.total_loss - loss_ref);
          double gerr = max_abs_diff(grads_pp, grads_ref, total_p);
          if (lerr > worst_loss_err)
            worst_loss_err = lerr;
          if (gerr > worst_grad_err)
            worst_grad_err = gerr;
          CHECK(lerr <= 1e-6);
          CHECK(gerr <= 1e-6);
        }
      }

      free(inputs);
      free(t_i32);
      free(t_f64);
      free(params);
      free(grads_ref);
      free(grads_pp);
    }
  }

  /* 10-step SGD training parity across P=1, P=2, P=4 (M=8, Cross-Entropy) */
  et_pp_config_t cfg_p1 =
      make_cfg(1u, 4u, 8u, ET_PP_SCHEDULE_1F1B, ET_PP_LOSS_CROSS_ENTROPY);
  et_pp_config_t cfg_p2 =
      make_cfg(2u, 2u, 8u, ET_PP_SCHEDULE_1F1B, ET_PP_LOSS_CROSS_ENTROPY);
  et_pp_config_t cfg_p4 =
      make_cfg(4u, 1u, 8u, ET_PP_SCHEDULE_1F1B, ET_PP_LOSS_CROSS_ENTROPY);
  size_t total_p = 0;
  CHECK(et_pp_param_count_v1(&cfg_p1, &total_p) == ET_PP_OK);
  size_t in_cnt = 8u * (size_t)cfg_p1.tokens_per_mb * (size_t)cfg_p1.d_model;
  size_t tok_cnt = 8u * (size_t)cfg_p1.tokens_per_mb;

  double *inputs = (double *)calloc(in_cnt, sizeof(double));
  int32_t *targets = (int32_t *)calloc(tok_cnt, sizeof(int32_t));
  double *init_p = (double *)calloc(total_p, sizeof(double));
  double *w_p1 = (double *)calloc(total_p, sizeof(double));
  double *w_p2 = (double *)calloc(total_p, sizeof(double));
  double *w_p4 = (double *)calloc(total_p, sizeof(double));
  double loss_p1[10], loss_p2[10], loss_p4[10];
  et_pp_run_telemetry_t t2, t4;
  CHECK(inputs && targets && init_p && w_p1 && w_p2 && w_p4);

  fill_inputs(&cfg_p1, inputs, 77u);
  fill_targets_ce(&cfg_p1, targets, 78u);
  fill_params(&cfg_p1, init_p, 79u);

  const double lr = 0.08;
  CHECK(et_pp_train_reference_sgd_v1(&cfg_p1, 10u, lr, inputs, targets, NULL,
                                     init_p, w_p1, loss_p1) == ET_PP_OK);
  fflush(NULL);
  CHECK(et_pp_train_multiprocess_sgd_v1(&cfg_p2, 10u, lr, inputs, targets, NULL,
                                        init_p, w_p2, loss_p2, &t2) ==
        ET_PP_OK);
  CHECK(count_open_fds() == fds_start);
  fflush(NULL);
  CHECK(et_pp_train_multiprocess_sgd_v1(&cfg_p4, 10u, lr, inputs, targets, NULL,
                                        init_p, w_p4, loss_p4, &t4) ==
        ET_PP_OK);
  CHECK(count_open_fds() == fds_start);

  for (int step = 0; step < 9; ++step) {
    CHECK(loss_p1[step + 1] < loss_p1[step]);
    CHECK(loss_p2[step + 1] < loss_p2[step]);
    CHECK(loss_p4[step + 1] < loss_p4[step]);
  }
  double sgd_loss_err = 0.0;
  for (int step = 0; step < 10; ++step) {
    double e2 = fabs(loss_p2[step] - loss_p1[step]);
    double e4 = fabs(loss_p4[step] - loss_p1[step]);
    if (e2 > sgd_loss_err)
      sgd_loss_err = e2;
    if (e4 > sgd_loss_err)
      sgd_loss_err = e4;
  }
  double w2_err = max_abs_diff(w_p2, w_p1, total_p);
  double w4_err = max_abs_diff(w_p4, w_p1, total_p);
  CHECK(sgd_loss_err <= 1e-6);
  CHECK(w2_err <= 1e-6 && w4_err <= 1e-6);
  CHECK(t2.distinct_pid_count == 2u && t4.distinct_pid_count == 4u);
  CHECK(t4.stage_peak_live[0] == 4u && t4.stage_peak_live[1] == 3u &&
        t4.stage_peak_live[2] == 2u && t4.stage_peak_live[3] == 1u);

  printf("PIPELINE_1F1B NATIVE PARITY max_loss_err=%.3e max_grad_err=%.3e "
         "sgd10_loss0=%.6f sgd10_loss9=%.6f sgd_w_err=%.3e PASS\n",
         worst_loss_err, worst_grad_err, loss_p4[0], loss_p4[9],
         w2_err > w4_err ? w2_err : w4_err);

  free(inputs);
  free(targets);
  free(init_p);
  free(w_p1);
  free(w_p2);
  free(w_p4);
}

/* ------------------------------------------------------------------------- */
/* 4. Negative Tests (Config, Non-Finite, Protocol, Disconnect, Timeout)     */
/* ------------------------------------------------------------------------- */

static void test_negative_cases(void) {
  int fds_start = count_open_fds();
  et_pp_schedule_t sched;

  /* Invalid P */
  CHECK(et_pp_build_schedule_v1(0u, 0u, 4u, ET_PP_SCHEDULE_1F1B, &sched) ==
        ET_PP_ERR_INVALID_STAGES);
  CHECK(et_pp_build_schedule_v1(1u, 0u, 4u, ET_PP_SCHEDULE_1F1B, &sched) ==
        ET_PP_ERR_INVALID_STAGES);
  CHECK(et_pp_build_schedule_v1(3u, 0u, 4u, ET_PP_SCHEDULE_1F1B, &sched) ==
        ET_PP_ERR_INVALID_STAGES);
  CHECK(et_pp_build_schedule_v1(5u, 0u, 8u, ET_PP_SCHEDULE_1F1B, &sched) ==
        ET_PP_ERR_INVALID_STAGES);
  CHECK(et_pp_build_schedule_v1(4u, 4u, 8u, ET_PP_SCHEDULE_1F1B, &sched) ==
        ET_PP_ERR_INVALID_STAGES);

  /* M < P and M > max */
  CHECK(et_pp_build_schedule_v1(2u, 0u, 1u, ET_PP_SCHEDULE_1F1B, &sched) ==
        ET_PP_ERR_INVALID_MICROBATCHES);
  CHECK(et_pp_build_schedule_v1(4u, 0u, 3u, ET_PP_SCHEDULE_1F1B, &sched) ==
        ET_PP_ERR_INVALID_MICROBATCHES);
  CHECK(et_pp_build_schedule_v1(4u, 0u, 33u, ET_PP_SCHEDULE_1F1B, &sched) ==
        ET_PP_ERR_INVALID_MICROBATCHES);
  CHECK(et_pp_build_schedule_v1(4u, 0u, 8u, 99, &sched) ==
        ET_PP_ERR_INVALID_ARG);
  CHECK(et_pp_build_schedule_v1(4u, 0u, 8u, ET_PP_SCHEDULE_1F1B, NULL) ==
        ET_PP_ERR_INVALID_ARG);

  /* Multi-process config & non-finite rejections */
  et_pp_config_t cfg =
      make_cfg(2u, 1u, 4u, ET_PP_SCHEDULE_1F1B, ET_PP_LOSS_CROSS_ENTROPY);
  size_t total_p = 0;
  CHECK(et_pp_param_count_v1(&cfg, &total_p) == ET_PP_OK);
  size_t in_cnt = 4u * (size_t)cfg.tokens_per_mb * (size_t)cfg.d_model;
  size_t tok_cnt = 4u * (size_t)cfg.tokens_per_mb;
  double *inputs = (double *)calloc(in_cnt, sizeof(double));
  int32_t *targets = (int32_t *)calloc(tok_cnt, sizeof(int32_t));
  double *params = (double *)calloc(total_p, sizeof(double));
  double *grads = (double *)calloc(total_p, sizeof(double));
  et_pp_run_telemetry_t telem;
  CHECK(inputs && targets && params && grads);

  fill_inputs(&cfg, inputs, 90u);
  fill_targets_ce(&cfg, targets, 91u);
  fill_params(&cfg, params, 92u);

  /* Bad M < P in multiprocess */
  et_pp_config_t bad_m = cfg;
  bad_m.num_microbatches = 1u;
  CHECK(et_pp_run_multiprocess_step_v1(&bad_m, inputs, targets, NULL, params,
                                       grads, &telem) ==
        ET_PP_ERR_INVALID_MICROBATCHES);

  /* Bad P = 3 in multiprocess */
  et_pp_config_t bad_p = cfg;
  bad_p.num_stages = 3u;
  CHECK(et_pp_run_multiprocess_step_v1(&bad_p, inputs, targets, NULL, params,
                                       grads, &telem) ==
        ET_PP_ERR_INVALID_STAGES);

  /* Non-finite activation input */
  inputs[2] = (double)NAN;
  CHECK(et_pp_run_multiprocess_step_v1(&cfg, inputs, targets, NULL, params,
                                       grads, &telem) == ET_PP_ERR_NONFINITE);
  inputs[2] = 0.25;

  /* Non-finite weight */
  params[5] = (double)INFINITY;
  CHECK(et_pp_run_multiprocess_step_v1(&cfg, inputs, targets, NULL, params,
                                       grads, &telem) == ET_PP_ERR_NONFINITE);
  params[5] = 0.1;

  /* Target out of range */
  targets[1] = (int32_t)cfg.out_dim;
  CHECK(et_pp_run_multiprocess_step_v1(&cfg, inputs, targets, NULL, params,
                                       grads, &telem) ==
        ET_PP_ERR_TARGET_RANGE);
  targets[1] = 0;

  /* Direct AF_UNIX transport negative tests */
  int fds[2] = {-1, -1};
  double tx[2] = {0.5, -0.25};
  double rx[2] = {0.0, 0.0};

  /* 1. Mismatched microbatch_id */
  CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0);
  CHECK(et_pp_p2p_send_frame_v1(fds[0], ET_PP_MSG_FORWARD_ACT, 5u, 0u, 1u, tx,
                                sizeof(tx), 500u) == ET_PP_OK);
  CHECK(et_pp_p2p_recv_frame_v1(fds[1], ET_PP_MSG_FORWARD_ACT, 4u, 0u, 1u, rx,
                                sizeof(rx), 500u) == ET_PP_ERR_PROTOCOL);
  close(fds[0]);
  close(fds[1]);

  /* 2. Mismatched msg_type */
  CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0);
  CHECK(et_pp_p2p_send_frame_v1(fds[0], ET_PP_MSG_BACKWARD_GRAD, 0u, 1u, 0u, tx,
                                sizeof(tx), 500u) == ET_PP_OK);
  CHECK(et_pp_p2p_recv_frame_v1(fds[1], ET_PP_MSG_FORWARD_ACT, 0u, 1u, 0u, rx,
                                sizeof(rx), 500u) == ET_PP_ERR_PROTOCOL);
  close(fds[0]);
  close(fds[1]);

  /* 3. Non-finite wire activation */
  CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0);
  tx[0] = -(double)INFINITY;
  CHECK(et_pp_p2p_send_frame_v1(fds[0], ET_PP_MSG_FORWARD_ACT, 0u, 0u, 1u, tx,
                                sizeof(tx), 500u) == ET_PP_OK);
  CHECK(et_pp_p2p_recv_frame_v1(fds[1], ET_PP_MSG_FORWARD_ACT, 0u, 0u, 1u, rx,
                                sizeof(rx), 500u) == ET_PP_ERR_NONFINITE);
  close(fds[0]);
  close(fds[1]);

  /* 4. Peer disconnect */
  CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0);
  close(fds[0]);
  CHECK(et_pp_p2p_recv_frame_v1(fds[1], ET_PP_MSG_FORWARD_ACT, 0u, 0u, 1u, rx,
                                sizeof(rx), 500u) ==
        ET_PP_ERR_PEER_DISCONNECTED);
  close(fds[1]);

  /* 5. Monotonic timeout */
  CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0);
  CHECK(et_pp_p2p_recv_frame_v1(fds[1], ET_PP_MSG_FORWARD_ACT, 0u, 0u, 1u, rx,
                                sizeof(rx), 60u) == ET_PP_ERR_TIMEOUT);
  close(fds[0]);
  close(fds[1]);

  /* 6. Non-socket FD (pipe) rejected */
  int pipe_fds[2] = {-1, -1};
  CHECK(pipe(pipe_fds) == 0);
  CHECK(et_pp_p2p_send_frame_v1(pipe_fds[1], ET_PP_MSG_FORWARD_ACT, 0u, 0u, 1u,
                                tx, sizeof(tx), 100u) ==
        ET_PP_ERR_INVALID_ARG);
  close(pipe_fds[0]);
  close(pipe_fds[1]);

  /* 7. AOT bridge transport probes */
  CHECK(et_pp_aot_probe_transport_error(0, 500) == ET_PP_ERR_PROTOCOL);
  CHECK(et_pp_aot_probe_transport_error(1, 500) == ET_PP_ERR_PEER_DISCONNECTED);
  CHECK(et_pp_aot_probe_transport_error(2, 50) == ET_PP_ERR_TIMEOUT);
  CHECK(et_pp_aot_probe_transport_error(3, 500) == ET_PP_ERR_NONFINITE);

  CHECK(count_open_fds() == fds_start);
  free(inputs);
  free(targets);
  free(params);
  free(grads);
  printf("PIPELINE_1F1B NATIVE NEGATIVE checks=23 fd_leaks=0 PASS\n");
}

/* ------------------------------------------------------------------------- */
/* 5. Deterministic JSON dump for independent Python binary64 oracle         */
/* ------------------------------------------------------------------------- */

static void print_f64_array(const double *a, size_t n) {
  putchar('[');
  for (size_t i = 0; i < n; ++i) {
    if (i > 0)
      putchar(',');
    printf("%.17g", a[i]);
  }
  putchar(']');
}

static void print_i32_array(const int32_t *a, size_t n) {
  putchar('[');
  for (size_t i = 0; i < n; ++i) {
    if (i > 0)
      putchar(',');
    printf("%d", a[i]);
  }
  putchar(']');
}

static void dump_oracle_json(void) {
  const uint32_t M = 8u;
  et_pp_config_t cfg_p1 =
      make_cfg(1u, 4u, M, ET_PP_SCHEDULE_1F1B, ET_PP_LOSS_CROSS_ENTROPY);
  cfg_p1.tokens_per_mb = 2u;
  cfg_p1.d_model = 4u;
  cfg_p1.d_ff = 6u;
  cfg_p1.out_dim = 5u;

  et_pp_config_t cfg_p2 = cfg_p1;
  cfg_p2.num_stages = 2u;
  cfg_p2.layers_per_stage = 2u;

  et_pp_config_t cfg_p4 = cfg_p1;
  cfg_p4.num_stages = 4u;
  cfg_p4.layers_per_stage = 1u;

  et_pp_config_t cfg_p4_gpipe = cfg_p4;
  cfg_p4_gpipe.schedule_kind = ET_PP_SCHEDULE_GPIPE;

  et_pp_config_t cfg_mse_p4 = cfg_p4;
  cfg_mse_p4.loss_type = ET_PP_LOSS_MSE;

  size_t total_p = 0;
  CHECK(et_pp_param_count_v1(&cfg_p1, &total_p) == ET_PP_OK);
  size_t in_cnt = (size_t)M * (size_t)cfg_p1.tokens_per_mb * (size_t)cfg_p1.d_model;
  size_t tok_cnt = (size_t)M * (size_t)cfg_p1.tokens_per_mb;

  double *inputs = (double *)calloc(in_cnt, sizeof(double));
  int32_t *targets_ce = (int32_t *)calloc(tok_cnt, sizeof(int32_t));
  double *targets_mse =
      (double *)calloc(tok_cnt * cfg_p1.out_dim, sizeof(double));
  double *params = (double *)calloc(total_p, sizeof(double));
  double *g_p1 = (double *)calloc(total_p, sizeof(double));
  double *g_p2 = (double *)calloc(total_p, sizeof(double));
  double *g_p4 = (double *)calloc(total_p, sizeof(double));
  double *g_mse = (double *)calloc(total_p, sizeof(double));
  double *w_sgd_p4 = (double *)calloc(total_p, sizeof(double));
  double loss_p1 = 0.0;
  double sgd_losses_p4[10];
  et_pp_run_telemetry_t t_p2, t_p4, t_gpipe, t_mse, t_sgd;
  CHECK(inputs && targets_ce && targets_mse && params && g_p1 && g_p2 && g_p4 &&
        g_mse && w_sgd_p4);

  fill_inputs(&cfg_p1, inputs, 101u);
  fill_targets_ce(&cfg_p1, targets_ce, 102u);
  fill_targets_mse(&cfg_p1, targets_mse, 103u);
  fill_params(&cfg_p1, params, 104u);

  CHECK(et_pp_run_reference_step_v1(&cfg_p1, inputs, targets_ce, NULL, params,
                                    g_p1, &loss_p1) == ET_PP_OK);
  fflush(NULL);
  CHECK(et_pp_run_multiprocess_step_v1(&cfg_p2, inputs, targets_ce, NULL,
                                       params, g_p2, &t_p2) == ET_PP_OK);
  fflush(NULL);
  CHECK(et_pp_run_multiprocess_step_v1(&cfg_p4, inputs, targets_ce, NULL,
                                       params, g_p4, &t_p4) == ET_PP_OK);
  fflush(NULL);
  CHECK(et_pp_run_multiprocess_step_v1(&cfg_p4_gpipe, inputs, targets_ce, NULL,
                                       params, g_p4, &t_gpipe) == ET_PP_OK);
  fflush(NULL);
  CHECK(et_pp_run_multiprocess_step_v1(&cfg_mse_p4, inputs, NULL, targets_mse,
                                       params, g_mse, &t_mse) == ET_PP_OK);
  fflush(NULL);
  CHECK(et_pp_train_multiprocess_sgd_v1(&cfg_p4, 10u, 0.08, inputs, targets_ce,
                                        NULL, params, w_sgd_p4, sgd_losses_p4,
                                        &t_sgd) == ET_PP_OK);

  printf("{\n");
  printf("  \"total_layers\": 4, \"M\": %u, \"T\": %u, \"D\": %u, \"F\": %u, "
         "\"V\": %u, \"rms_eps\": %.17g, \"lr\": 0.08,\n",
         M, cfg_p1.tokens_per_mb, cfg_p1.d_model, cfg_p1.d_ff, cfg_p1.out_dim,
         cfg_p1.rms_eps);
  printf("  \"inputs\": ");
  print_f64_array(inputs, in_cnt);
  printf(",\n  \"targets_ce\": ");
  print_i32_array(targets_ce, tok_cnt);
  printf(",\n  \"targets_mse\": ");
  print_f64_array(targets_mse, tok_cnt * cfg_p1.out_dim);
  printf(",\n  \"params\": ");
  print_f64_array(params, total_p);
  printf(",\n  \"p1_ce_loss\": %.17g,\n  \"p1_ce_grads\": ", loss_p1);
  print_f64_array(g_p1, total_p);
  printf(",\n  \"p2_ce_loss\": %.17g,\n  \"p2_ce_grads\": ", t_p2.total_loss);
  print_f64_array(g_p2, total_p);
  printf(",\n  \"p2_peak_live\": [%u,%u],\n", t_p2.stage_peak_live[0],
         t_p2.stage_peak_live[1]);
  printf("  \"p4_ce_loss\": %.17g,\n  \"p4_ce_grads\": ", t_p4.total_loss);
  print_f64_array(g_p4, total_p);
  printf(",\n  \"p4_1f1b_peak_live\": [%u,%u,%u,%u],\n",
         t_p4.stage_peak_live[0], t_p4.stage_peak_live[1],
         t_p4.stage_peak_live[2], t_p4.stage_peak_live[3]);
  printf("  \"p4_gpipe_peak_live\": [%u,%u,%u,%u],\n",
         t_gpipe.stage_peak_live[0], t_gpipe.stage_peak_live[1],
         t_gpipe.stage_peak_live[2], t_gpipe.stage_peak_live[3]);
  printf("  \"p4_mse_loss\": %.17g,\n  \"p4_mse_grads\": ", t_mse.total_loss);
  print_f64_array(g_mse, total_p);
  printf(",\n  \"p4_sgd10_losses\": ");
  print_f64_array(sgd_losses_p4, 10u);
  printf(",\n  \"p4_sgd10_weights\": ");
  print_f64_array(w_sgd_p4, total_p);
  printf("\n}\n");

  free(inputs);
  free(targets_ce);
  free(targets_mse);
  free(params);
  free(g_p1);
  free(g_p2);
  free(g_p4);
  free(g_mse);
  free(w_sgd_p4);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle-dump") == 0) {
    dump_oracle_json();
    return 0;
  }
  test_schedules();
  test_numerical_gradients();
  test_multiprocess_parity_and_sgd();
  test_negative_cases();
  return 0;
}
