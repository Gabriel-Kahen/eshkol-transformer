#define _GNU_SOURCE
#include "mtp_head_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

struct et_mtp_aot_session {
  et_mtp_model_t *model;
  et_mtp_config_v1 cfg;
  et_mtp_layout_v1 layout;
  size_t batch;
  size_t seq_len;

  float *h0;
  uint32_t *tokens;
  float *d_h0;
  float *d_params;

  et_mtp_loss_stats_v1 last_loss;
  et_mtp_step_stats_v1 last_step;

  double last_fd_param_max_err;
  double last_fd_h0_max_err;
  int64_t last_fd_checked;

  int64_t spec_exact_match;
  int64_t spec_total_emitted;
  int64_t spec_verify_steps;
  double spec_tokens_per_step;

  char digest_hex[65];
};

static uint64_t bridge_splitmix64(uint64_t *state) {
  *state += UINT64_C(0x9e3779b97f4a7c15);
  uint64_t z = *state;
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

static void bytes_to_hex(const uint8_t in[32], char out[65]) {
  static const char hex[] = "0123456789abcdef";
  for (size_t i = 0u; i < 32u; ++i) {
    out[i * 2u] = hex[in[i] >> 4u];
    out[i * 2u + 1u] = hex[in[i] & 0x0fu];
  }
  out[64] = '\0';
}

static uint32_t argmax_f32_row(const float *logits, size_t V) {
  uint32_t best = 0u;
  float best_val = logits[0];
  for (size_t v = 1u; v < V; ++v) {
    if (logits[v] > best_val) {
      best_val = logits[v];
      best = (uint32_t)v;
    }
  }
  return best;
}

int32_t et_mtp_aot_open(int64_t vocab_size, int64_t hidden_dim, int64_t ffn_dim,
                        int64_t mtp_depth, int64_t tie_weights, int64_t batch,
                        int64_t seq_len, double rms_eps, double lambda_mtp,
                        double lr, double beta1, double beta2, double eps,
                        double weight_decay, double max_grad_norm,
                        et_mtp_aot_session_t **out_session) {
  if (out_session == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  *out_session = NULL;
  if (vocab_size <= 0 || hidden_dim <= 0 || ffn_dim <= 0 || mtp_depth <= 0 ||
      tie_weights < 0 || tie_weights > 1 || batch < (int64_t)ET_MTP_MIN_BATCH ||
      batch > (int64_t)ET_MTP_MAX_BATCH || seq_len <= 0 ||
      seq_len > (int64_t)ET_MTP_MAX_SEQ_LEN ||
      seq_len < mtp_depth + 2) {
    return ET_MTP_ERR_INVALID_ARG;
  }

  et_mtp_config_v1 cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.struct_size = sizeof(cfg);
  cfg.vocab_size = (uint32_t)vocab_size;
  cfg.hidden_dim = (uint32_t)hidden_dim;
  cfg.ffn_dim = (uint32_t)ffn_dim;
  cfg.mtp_depth = (uint32_t)mtp_depth;
  cfg.tie_weights = (uint32_t)tie_weights;
  cfg.rms_eps = rms_eps;
  cfg.lambda_mtp = lambda_mtp;
  cfg.lr = lr;
  cfg.beta1 = beta1;
  cfg.beta2 = beta2;
  cfg.eps = eps;
  cfg.weight_decay = weight_decay;
  cfg.max_grad_norm = max_grad_norm;

  et_mtp_model_t *model = NULL;
  et_mtp_status_t st = et_mtp_model_create_v1(&cfg, &model);
  if (st != ET_MTP_OK) {
    return st;
  }

  et_mtp_layout_v1 lay;
  st = et_mtp_model_get_layout_v1(model, &lay);
  if (st != ET_MTP_OK) {
    et_mtp_model_destroy_v1(model);
    return st;
  }

  et_mtp_aot_session_t *s =
      (et_mtp_aot_session_t *)calloc(1u, sizeof(*s));
  if (s == NULL) {
    et_mtp_model_destroy_v1(model);
    return ET_MTP_ERR_ALLOC;
  }
  const size_t B = (size_t)batch;
  const size_t T = (size_t)seq_len;
  const size_t H = (size_t)hidden_dim;
  s->h0 = (float *)calloc(B * T * H, sizeof(float));
  s->tokens = (uint32_t *)calloc(B * T, sizeof(uint32_t));
  s->d_h0 = (float *)calloc(B * T * H, sizeof(float));
  s->d_params = (float *)calloc(lay.total_params, sizeof(float));
  if (s->h0 == NULL || s->tokens == NULL || s->d_h0 == NULL ||
      s->d_params == NULL) {
    free(s->h0);
    free(s->tokens);
    free(s->d_h0);
    free(s->d_params);
    free(s);
    et_mtp_model_destroy_v1(model);
    return ET_MTP_ERR_ALLOC;
  }
  s->model = model;
  s->cfg = cfg;
  s->layout = lay;
  s->batch = B;
  s->seq_len = T;
  memset(s->digest_hex, '0', 64u);
  s->digest_hex[64] = '\0';

  *out_session = s;
  return ET_MTP_OK;
}

void et_mtp_aot_close(et_mtp_aot_session_t *session) {
  if (session == NULL) {
    return;
  }
  et_mtp_model_destroy_v1(session->model);
  free(session->h0);
  free(session->tokens);
  free(session->d_h0);
  free(session->d_params);
  free(session);
}

int32_t et_mtp_aot_init_weights(et_mtp_aot_session_t *session, int64_t seed) {
  if (session == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  return et_mtp_model_init_weights_v1(session->model, (uint64_t)seed);
}

int32_t et_mtp_aot_init_sequence(et_mtp_aot_session_t *session, int64_t seed) {
  if (session == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  uint64_t rng = (uint64_t)seed ^ UINT64_C(0x9e3779b97f4a7c15);
  const size_t B = session->batch;
  const size_t T = session->seq_len;
  const size_t H = (size_t)session->cfg.hidden_dim;
  const uint32_t V = session->cfg.vocab_size;

  for (size_t b = 0u; b < B; ++b) {
    for (size_t t = 0u; t < T; ++t) {
      session->tokens[b * T + t] =
          (uint32_t)((t * 3u + b * 5u + 1u) % (size_t)V);
      float *row = session->h0 + (b * T + t) * H;
      for (size_t j = 0u; j < H; ++j) {
        const uint64_t z = bridge_splitmix64(&rng);
        const double u = (double)(z >> 11u) * (1.0 / 9007199254740992.0);
        row[j] = (float)((u * 2.0 - 1.0) * 0.9);
      }
    }
  }
  return ET_MTP_OK;
}

int32_t et_mtp_aot_train_step(et_mtp_aot_session_t *session) {
  if (session == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  et_mtp_loss_stats_v1 loss_stats;
  et_mtp_status_t st = et_mtp_forward_backward_v1(
      session->model, session->batch, session->seq_len, session->h0,
      session->tokens, &loss_stats, session->d_h0, session->d_params,
      session->layout.total_params);
  if (st != ET_MTP_OK) {
    return st;
  }
  et_mtp_step_stats_v1 step_stats;
  st = et_mtp_adamw_step_v1(session->model, session->d_params,
                            session->layout.total_params, &step_stats);
  if (st != ET_MTP_OK) {
    return st;
  }
  session->last_loss = loss_stats;
  session->last_step = step_stats;
  return ET_MTP_OK;
}

int32_t et_mtp_aot_fd_grad_check(et_mtp_aot_session_t *session, double fd_eps) {
  if (session == NULL || !isfinite(fd_eps) || fd_eps <= 0.0) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  const size_t n = session->layout.total_params;
  const size_t h0_elems =
      session->batch * session->seq_len * (size_t)session->cfg.hidden_dim;

  et_mtp_loss_stats_v1 base_stats;
  et_mtp_status_t st = et_mtp_forward_backward_v1(
      session->model, session->batch, session->seq_len, session->h0,
      session->tokens, &base_stats, session->d_h0, session->d_params, n);
  if (st != ET_MTP_OK) {
    return st;
  }

  float *orig_params = (float *)malloc(n * sizeof(float));
  float *work_params = (float *)malloc(n * sizeof(float));
  if (orig_params == NULL || work_params == NULL) {
    free(orig_params);
    free(work_params);
    return ET_MTP_ERR_ALLOC;
  }
  st = et_mtp_model_get_params_v1(session->model, orig_params, n);
  if (st != ET_MTP_OK) {
    free(orig_params);
    free(work_params);
    return st;
  }
  memcpy(work_params, orig_params, n * sizeof(float));

  double max_param_err = 0.0;
  int64_t checked = 0;
  for (size_t i = 0u; i < n; ++i) {
    const float base_v = orig_params[i];
    const float plus_v = (float)((double)base_v + fd_eps);
    const float minus_v = (float)((double)base_v - fd_eps);
    const double actual_span = (double)plus_v - (double)minus_v;

    work_params[i] = plus_v;
    st = et_mtp_model_set_params_v1(session->model, work_params, n);
    if (st != ET_MTP_OK) {
      break;
    }
    et_mtp_loss_stats_v1 sp, sm;
    st = et_mtp_forward_v1(session->model, session->batch, session->seq_len,
                           session->h0, session->tokens, &sp);
    if (st != ET_MTP_OK) {
      break;
    }
    work_params[i] = minus_v;
    st = et_mtp_model_set_params_v1(session->model, work_params, n);
    if (st != ET_MTP_OK) {
      break;
    }
    st = et_mtp_forward_v1(session->model, session->batch, session->seq_len,
                           session->h0, session->tokens, &sm);
    if (st != ET_MTP_OK) {
      break;
    }
    work_params[i] = base_v;

    const double num = (sp.loss_total - sm.loss_total) / actual_span;
    const double err = fabs((double)session->d_params[i] - num);
    if (err > max_param_err) {
      max_param_err = err;
    }
    checked += 1;
  }

  /* Restore exact original parameters before checking d_h0. */
  const et_mtp_status_t restore_st =
      et_mtp_model_set_params_v1(session->model, orig_params, n);
  free(orig_params);
  free(work_params);
  if (st != ET_MTP_OK) {
    return st;
  }
  if (restore_st != ET_MTP_OK) {
    return restore_st;
  }

  double max_h0_err = 0.0;
  for (size_t i = 0u; i < h0_elems; ++i) {
    const float base_v = session->h0[i];
    const float plus_v = (float)((double)base_v + fd_eps);
    const float minus_v = (float)((double)base_v - fd_eps);
    const double actual_span = (double)plus_v - (double)minus_v;

    session->h0[i] = plus_v;
    et_mtp_loss_stats_v1 sp, sm;
    st = et_mtp_forward_v1(session->model, session->batch, session->seq_len,
                           session->h0, session->tokens, &sp);
    if (st != ET_MTP_OK) {
      session->h0[i] = base_v;
      return st;
    }
    session->h0[i] = minus_v;
    st = et_mtp_forward_v1(session->model, session->batch, session->seq_len,
                           session->h0, session->tokens, &sm);
    session->h0[i] = base_v;
    if (st != ET_MTP_OK) {
      return st;
    }

    const double num = (sp.loss_total - sm.loss_total) / actual_span;
    const double err = fabs((double)session->d_h0[i] - num);
    if (err > max_h0_err) {
      max_h0_err = err;
    }
    checked += 1;
  }

  session->last_fd_param_max_err = max_param_err;
  session->last_fd_h0_max_err = max_h0_err;
  session->last_fd_checked = checked;
  return ET_MTP_OK;
}

int32_t et_mtp_aot_spec_greedy_verify(et_mtp_aot_session_t *session) {
  if (session == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  const size_t T = session->seq_len;
  const size_t H = (size_t)session->cfg.hidden_dim;
  const size_t V = (size_t)session->cfg.vocab_size;
  const size_t D = (size_t)session->cfg.mtp_depth;

  /*
   * Target trajectory on batch 0:
   *   At step pos in 0 .. T - 2, main head maps h0[pos] -> logits_main[pos],
   *   with baseline greedy token g[pos] = argmax(logits_main[pos]).
   *   Self-speculative MTP starts at pos = 0 (anchor = g[0] from main head at
   *   pos = 0), and iteratively drafts D lookahead tokens for positions
   *   [cur .. cur + D - 1] from h0[cur - 1] + anchor = spec_stream[cur - 1],
   *   verifying against target_logits[cur .. cur + D] (clamped at T - 2) until
   *   all target tokens g[1 .. T - 2] are emitted!
   */
  float *all_logits = (float *)malloc(T * V * sizeof(float));
  uint32_t *baseline = (uint32_t *)malloc(T * sizeof(uint32_t));
  uint32_t *spec_stream = (uint32_t *)malloc((T + D + 2u) * sizeof(uint32_t));
  float *target_window = (float *)malloc((D + 1u) * V * sizeof(float));
  if (all_logits == NULL || baseline == NULL || spec_stream == NULL ||
      target_window == NULL) {
    free(all_logits);
    free(baseline);
    free(spec_stream);
    free(target_window);
    return ET_MTP_ERR_ALLOC;
  }

  et_mtp_status_t st = ET_MTP_OK;
  for (size_t t = 0u; t + 1u < T; ++t) {
    st = et_mtp_main_logits_v1(session->model, session->h0 + t * H,
                               all_logits + t * V);
    if (st != ET_MTP_OK) {
      break;
    }
    baseline[t] = argmax_f32_row(all_logits + t * V, V);
  }
  if (st != ET_MTP_OK) {
    free(all_logits);
    free(baseline);
    free(spec_stream);
    free(target_window);
    return st;
  }

  /* Token 0 is produced by main head at h0[0]; remaining tokens are [1..T-2]. */
  const size_t total_targets = T - 1u;
  spec_stream[0] = baseline[0];
  size_t produced = 1u;
  int64_t verify_steps = 0;
  int64_t spec_emitted_sum = 0;

  while (produced < total_targets) {
    const size_t prefix_pos = produced - 1u;
    const uint32_t anchor = spec_stream[prefix_pos];
    for (size_t j = 0u; j <= D; ++j) {
      const size_t target_idx =
          (produced + j < total_targets) ? (produced + j) : (total_targets - 1u);
      memcpy(target_window + j * V, all_logits + target_idx * V,
             V * sizeof(float));
    }
    uint32_t drafts[ET_MTP_MAX_DEPTH];
    uint32_t emitted[ET_MTP_MAX_DEPTH + 1u];
    et_mtp_verify_result_v1 vres;
    st = et_mtp_draft_verify_v1(session->model, session->h0 + prefix_pos * H,
                                anchor, target_window, 0.0, ET_MTP_NO_EOS, NULL,
                                drafts, emitted, (uint32_t)(D + 1u), &vres);
    if (st != ET_MTP_OK) {
      break;
    }
    verify_steps += 1;
    const size_t remaining = total_targets - produced;
    const size_t take =
        ((size_t)vres.emitted_count < remaining) ? (size_t)vres.emitted_count
                                                 : remaining;
    for (size_t j = 0u; j < take; ++j) {
      spec_stream[produced + j] = emitted[j];
    }
    produced += take;
    spec_emitted_sum += (int64_t)take;
  }

  int exact = (st == ET_MTP_OK && produced == total_targets) ? 1 : 0;
  for (size_t t = 0u; exact && t < total_targets; ++t) {
    if (spec_stream[t] != baseline[t]) {
      exact = 0;
    }
  }

  free(all_logits);
  free(baseline);
  free(spec_stream);
  free(target_window);
  if (st != ET_MTP_OK) {
    return st;
  }

  session->spec_exact_match = exact;
  session->spec_total_emitted = spec_emitted_sum;
  session->spec_verify_steps = verify_steps;
  session->spec_tokens_per_step =
      (verify_steps > 0) ? ((double)spec_emitted_sum / (double)verify_steps)
                         : 0.0;
  return ET_MTP_OK;
}

int32_t et_mtp_aot_save_checkpoint(const et_mtp_aot_session_t *session,
                                   const char *path) {
  if (session == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  return et_mtp_checkpoint_save_v1(session->model, path);
}

int32_t et_mtp_aot_load_checkpoint(et_mtp_aot_session_t *session,
                                   const char *path) {
  if (session == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  return et_mtp_checkpoint_load_v1(session->model, path);
}

int32_t et_mtp_aot_compute_state_digest(et_mtp_aot_session_t *session) {
  if (session == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  uint8_t digest[32];
  const et_mtp_status_t st =
      et_mtp_model_state_sha256_v1(session->model, digest);
  if (st != ET_MTP_OK) {
    return st;
  }
  bytes_to_hex(digest, session->digest_hex);
  return ET_MTP_OK;
}

int32_t et_mtp_aot_digest_byte(const et_mtp_aot_session_t *session,
                               int64_t index) {
  if (session == NULL || index < 0 || index >= 64) {
    return -1;
  }
  return (int32_t)(unsigned char)session->digest_hex[index];
}

/* --- Negative Probes --- */

int32_t et_mtp_aot_probe_invalid_depth(int64_t bad_depth) {
  et_mtp_config_v1 cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.struct_size = sizeof(cfg);
  cfg.vocab_size = 16u;
  cfg.hidden_dim = 8u;
  cfg.ffn_dim = 16u;
  cfg.mtp_depth = (uint32_t)bad_depth;
  cfg.tie_weights = 0u;
  cfg.rms_eps = 1e-6;
  cfg.lambda_mtp = 0.5;
  cfg.lr = 0.05;
  cfg.beta1 = 0.9;
  cfg.beta2 = 0.999;
  cfg.eps = 1e-8;
  cfg.weight_decay = 0.01;
  cfg.max_grad_norm = 5.0;
  return et_mtp_validate_config_v1(&cfg);
}

int32_t et_mtp_aot_probe_short_seq(const et_mtp_aot_session_t *session,
                                   int64_t short_len) {
  if (session == NULL || short_len < 0) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  et_mtp_loss_stats_v1 stats;
  return et_mtp_forward_v1(session->model, 1u, (size_t)short_len, session->h0,
                           session->tokens, &stats);
}

int32_t et_mtp_aot_probe_oob_token(const et_mtp_aot_session_t *session) {
  if (session == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  const size_t bt = session->batch * session->seq_len;
  uint32_t *bad_toks = (uint32_t *)malloc(bt * sizeof(uint32_t));
  if (bad_toks == NULL) {
    return ET_MTP_ERR_ALLOC;
  }
  memcpy(bad_toks, session->tokens, bt * sizeof(uint32_t));
  bad_toks[bt - 1u] = session->cfg.vocab_size;
  et_mtp_loss_stats_v1 stats;
  const et_mtp_status_t st =
      et_mtp_forward_v1(session->model, session->batch, session->seq_len,
                        session->h0, bad_toks, &stats);
  free(bad_toks);
  return st;
}

int32_t et_mtp_aot_probe_nonfinite_h0(const et_mtp_aot_session_t *session) {
  if (session == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  const size_t h0_elems =
      session->batch * session->seq_len * (size_t)session->cfg.hidden_dim;
  float *bad_h0 = (float *)malloc(h0_elems * sizeof(float));
  if (bad_h0 == NULL) {
    return ET_MTP_ERR_ALLOC;
  }
  memcpy(bad_h0, session->h0, h0_elems * sizeof(float));
  bad_h0[h0_elems / 2u] = NAN;
  et_mtp_loss_stats_v1 stats;
  const et_mtp_status_t st =
      et_mtp_forward_v1(session->model, session->batch, session->seq_len,
                        bad_h0, session->tokens, &stats);
  free(bad_h0);
  return st;
}

int32_t et_mtp_aot_probe_fp_env_trap(const et_mtp_aot_session_t *session) {
  if (session == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  fenv_t saved;
  if (fegetenv(&saved) != 0) {
    return ET_MTP_ERR_FP_ENV;
  }
  if (fesetround(FE_UPWARD) != 0) {
    (void)fesetenv(&saved);
    return ET_MTP_ERR_FP_ENV;
  }
  et_mtp_loss_stats_v1 stats;
  const et_mtp_status_t st =
      et_mtp_forward_v1(session->model, session->batch, session->seq_len,
                        session->h0, session->tokens, &stats);
  (void)fesetenv(&saved);
  return st;
}

int32_t et_mtp_aot_probe_ckpt_config_mismatch(const char *path,
                                              int64_t alt_depth) {
  if (path == NULL || alt_depth < (int64_t)ET_MTP_MIN_DEPTH ||
      alt_depth > (int64_t)ET_MTP_MAX_DEPTH) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  et_mtp_config_v1 cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.struct_size = sizeof(cfg);
  cfg.vocab_size = 16u;
  cfg.hidden_dim = 8u;
  cfg.ffn_dim = 16u;
  cfg.mtp_depth = (uint32_t)alt_depth;
  cfg.tie_weights = 0u;
  cfg.rms_eps = 1e-6;
  cfg.lambda_mtp = 0.5;
  cfg.lr = 0.08;
  cfg.beta1 = 0.9;
  cfg.beta2 = 0.999;
  cfg.eps = 1e-8;
  cfg.weight_decay = 0.01;
  cfg.max_grad_norm = 5.0;

  et_mtp_model_t *alt = NULL;
  et_mtp_status_t st = et_mtp_model_create_v1(&cfg, &alt);
  if (st != ET_MTP_OK) {
    return st;
  }
  st = et_mtp_checkpoint_load_v1(alt, path);
  et_mtp_model_destroy_v1(alt);
  return st;
}

int32_t et_mtp_aot_corrupt_file_byte(const char *path, int64_t byte_offset) {
  if (path == NULL || byte_offset < 0) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  int fd = open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) {
    return ET_MTP_ERR_IO;
  }
  uint8_t b = 0u;
  if (pread(fd, &b, 1u, (off_t)byte_offset) != 1) {
    close(fd);
    return ET_MTP_ERR_IO;
  }
  b ^= 0x5au;
  const ssize_t w = pwrite(fd, &b, 1u, (off_t)byte_offset);
  close(fd);
  return (w == 1) ? ET_MTP_OK : ET_MTP_ERR_IO;
}

int32_t et_mtp_aot_truncate_file(const char *path, int64_t new_size) {
  if (path == NULL || new_size < 0) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  return (truncate(path, (off_t)new_size) == 0) ? ET_MTP_OK : ET_MTP_ERR_IO;
}

int32_t et_mtp_aot_copy_file(const char *src_path, const char *dst_path) {
  if (src_path == NULL || dst_path == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  int in_fd = open(src_path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (in_fd < 0) {
    return ET_MTP_ERR_IO;
  }
  int out_fd = open(dst_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (out_fd < 0) {
    close(in_fd);
    return ET_MTP_ERR_IO;
  }
  uint8_t buf[4096];
  ssize_t r;
  int err = 0;
  while ((r = read(in_fd, buf, sizeof(buf))) > 0) {
    size_t off = 0u;
    while (off < (size_t)r) {
      const ssize_t w = write(out_fd, buf + off, (size_t)r - off);
      if (w <= 0) {
        err = 1;
        break;
      }
      off += (size_t)w;
    }
    if (err) {
      break;
    }
  }
  if (r < 0) {
    err = 1;
  }
  close(in_fd);
  close(out_fd);
  return err ? ET_MTP_ERR_IO : ET_MTP_OK;
}

int32_t et_mtp_aot_make_symlink(const char *target_path,
                                const char *link_path) {
  if (target_path == NULL || link_path == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  (void)unlink(link_path);
  return (symlink(target_path, link_path) == 0) ? ET_MTP_OK : ET_MTP_ERR_IO;
}

/* --- Scalar Accessors --- */

int64_t et_mtp_aot_total_params(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? (int64_t)session->layout.total_params : 0;
}

int64_t et_mtp_aot_step(const et_mtp_aot_session_t *session) {
  if (session == NULL) {
    return -1;
  }
  uint64_t step = 0u;
  if (et_mtp_model_get_step_v1(session->model, &step) != ET_MTP_OK) {
    return -1;
  }
  return (int64_t)step;
}

double et_mtp_aot_last_loss_total(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? session->last_loss.loss_total : NAN;
}

double et_mtp_aot_last_loss_main(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? session->last_loss.loss_main : NAN;
}

double et_mtp_aot_last_loss_mtp_avg(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? session->last_loss.loss_mtp_avg : NAN;
}

double et_mtp_aot_last_loss_depth(const et_mtp_aot_session_t *session,
                                  int64_t depth_idx) {
  if (session == NULL || depth_idx < 0 ||
      depth_idx >= (int64_t)ET_MTP_MAX_DEPTH) {
    return NAN;
  }
  return session->last_loss.loss_depth[depth_idx];
}

double et_mtp_aot_last_grad_norm(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? session->last_step.grad_norm : NAN;
}

double et_mtp_aot_last_fd_param_max_err(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? session->last_fd_param_max_err : NAN;
}

double et_mtp_aot_last_fd_h0_max_err(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? session->last_fd_h0_max_err : NAN;
}

int64_t et_mtp_aot_last_fd_checked(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? session->last_fd_checked : 0;
}

int64_t et_mtp_aot_spec_exact_match(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? session->spec_exact_match : 0;
}

int64_t et_mtp_aot_spec_total_emitted(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? session->spec_total_emitted : 0;
}

int64_t et_mtp_aot_spec_verify_steps(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? session->spec_verify_steps : 0;
}

double et_mtp_aot_spec_tokens_per_step(const et_mtp_aot_session_t *session) {
  return (session != NULL) ? session->spec_tokens_per_step : 0.0;
}
