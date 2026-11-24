#define _GNU_SOURCE
#include "pipeline_1f1b_eshkol_bridge.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int aligned_for(const void *p, size_t align) {
  return p != NULL && (((uintptr_t)p & (align - 1u)) == 0u);
}

static int u32_from_i64(int64_t v, uint32_t *out) {
  if (v < 0 || v > (int64_t)UINT32_MAX)
    return 0;
  *out = (uint32_t)v;
  return 1;
}

static int i32_from_i64(int64_t v, int32_t *out) {
  if (v < (int64_t)INT32_MIN || v > (int64_t)INT32_MAX)
    return 0;
  *out = (int32_t)v;
  return 1;
}

int64_t et_pp_aot_param_count(int64_t total_layers, int64_t d_model,
                              int64_t d_ff, int64_t out_dim) {
  if (total_layers < 1 || total_layers > (int64_t)ET_PP_MAX_TOTAL_LAYERS ||
      d_model < 2 || d_model > (int64_t)ET_PP_MAX_D_MODEL || d_ff < 2 ||
      d_ff > (int64_t)ET_PP_MAX_D_FF || out_dim < 2 ||
      out_dim > (int64_t)ET_PP_MAX_OUT_DIM)
    return (int64_t)ET_PP_ERR_SHAPE;

  size_t per_layer =
      et_pp_layer_param_count_v1((uint32_t)d_model, (uint32_t)d_ff);
  size_t head = (size_t)out_dim * (size_t)d_model + (size_t)out_dim;
  return (int64_t)((size_t)total_layers * per_layer + head);
}

int32_t et_pp_aot_build_schedule(int64_t num_stages, int64_t stage_id,
                                 int64_t num_microbatches,
                                 int64_t schedule_kind, void *out_summary_i64,
                                 void *out_bubble_f64) {
  uint32_t p = 0;
  uint32_t s = 0;
  uint32_t m = 0;
  int32_t kind = 0;
  if (!aligned_for(out_summary_i64, sizeof(int64_t)) ||
      !aligned_for(out_bubble_f64, sizeof(double)))
    return ET_PP_ERR_INVALID_ARG;
  if (!u32_from_i64(num_stages, &p) || !u32_from_i64(stage_id, &s))
    return ET_PP_ERR_INVALID_STAGES;
  if (!u32_from_i64(num_microbatches, &m))
    return ET_PP_ERR_INVALID_MICROBATCHES;
  if (!i32_from_i64(schedule_kind, &kind))
    return ET_PP_ERR_INVALID_ARG;

  et_pp_schedule_t sched;
  et_pp_status_t st = et_pp_build_schedule_v1(p, s, m, kind, &sched);
  if (st != ET_PP_OK)
    return st;

  int64_t summary[5];
  summary[0] = (int64_t)sched.num_warmup_steps;
  summary[1] = (int64_t)sched.num_steady_pairs;
  summary[2] = (int64_t)sched.num_cooldown_steps;
  summary[3] = (int64_t)sched.step_count;
  summary[4] = (int64_t)sched.peak_live_activations;
  memcpy(out_summary_i64, summary, sizeof(summary));
  memcpy(out_bubble_f64, &sched.bubble_fraction, sizeof(double));
  return ET_PP_OK;
}

int32_t et_pp_aot_layer_forward(int64_t tokens, int64_t d_model, int64_t d_ff,
                                double rms_eps, const void *x,
                                const void *layer_params, void *y) {
  uint32_t t = 0;
  uint32_t d = 0;
  uint32_t f = 0;
  if (!u32_from_i64(tokens, &t) || !u32_from_i64(d_model, &d) ||
      !u32_from_i64(d_ff, &f))
    return ET_PP_ERR_SHAPE;
  if (!aligned_for(x, sizeof(double)) ||
      !aligned_for(layer_params, sizeof(double)) ||
      !aligned_for(y, sizeof(double)))
    return ET_PP_ERR_INVALID_ARG;
  return et_pp_layer_forward_v1(t, d, f, rms_eps, (const double *)x,
                                (const double *)layer_params, (double *)y);
}

int32_t et_pp_aot_layer_backward(int64_t tokens, int64_t d_model, int64_t d_ff,
                                 double rms_eps, const void *x,
                                 const void *layer_params, const void *dy,
                                 void *dx, void *dlayer_params) {
  uint32_t t = 0;
  uint32_t d = 0;
  uint32_t f = 0;
  if (!u32_from_i64(tokens, &t) || !u32_from_i64(d_model, &d) ||
      !u32_from_i64(d_ff, &f))
    return ET_PP_ERR_SHAPE;
  if (!aligned_for(x, sizeof(double)) ||
      !aligned_for(layer_params, sizeof(double)) ||
      !aligned_for(dy, sizeof(double)) || !aligned_for(dx, sizeof(double)) ||
      !aligned_for(dlayer_params, sizeof(double)))
    return ET_PP_ERR_INVALID_ARG;
  return et_pp_layer_backward_v1(t, d, f, rms_eps, (const double *)x,
                                 (const double *)layer_params,
                                 (const double *)dy, (double *)dx,
                                 (double *)dlayer_params);
}

static et_pp_status_t build_cfg(int64_t num_stages, int64_t layers_per_stage,
                                int64_t num_microbatches, int64_t tokens_per_mb,
                                int64_t d_model, int64_t d_ff, int64_t out_dim,
                                int64_t schedule_kind, int64_t loss_type,
                                int64_t timeout_ms, double rms_eps,
                                et_pp_config_t *cfg) {
  uint32_t p = 0;
  uint32_t lps = 0;
  uint32_t m = 0;
  uint32_t t = 0;
  uint32_t d = 0;
  uint32_t f = 0;
  uint32_t v = 0;
  uint32_t to = 0;
  int32_t sk = 0;
  int32_t lt = 0;

  if (!u32_from_i64(num_stages, &p))
    return ET_PP_ERR_INVALID_STAGES;
  if (!u32_from_i64(num_microbatches, &m))
    return ET_PP_ERR_INVALID_MICROBATCHES;
  if (!u32_from_i64(layers_per_stage, &lps) ||
      !u32_from_i64(tokens_per_mb, &t) || !u32_from_i64(d_model, &d) ||
      !u32_from_i64(d_ff, &f) || !u32_from_i64(out_dim, &v))
    return ET_PP_ERR_SHAPE;
  if (!i32_from_i64(schedule_kind, &sk) || !i32_from_i64(loss_type, &lt) ||
      !u32_from_i64(timeout_ms, &to))
    return ET_PP_ERR_INVALID_ARG;

  cfg->abi_major = ET_PP_ABI_MAJOR;
  cfg->abi_minor = ET_PP_ABI_MINOR;
  cfg->num_stages = p;
  cfg->layers_per_stage = lps;
  cfg->num_microbatches = m;
  cfg->tokens_per_mb = t;
  cfg->d_model = d;
  cfg->d_ff = f;
  cfg->out_dim = v;
  cfg->schedule_kind = sk;
  cfg->loss_type = lt;
  cfg->timeout_ms = to;
  cfg->rms_eps = rms_eps;
  return ET_PP_OK;
}

int32_t et_pp_aot_run_reference_step(
    int64_t total_layers, int64_t num_microbatches, int64_t tokens_per_mb,
    int64_t d_model, int64_t d_ff, int64_t out_dim, int64_t loss_type,
    double rms_eps, const void *inputs, const void *targets, const void *params,
    void *out_grads, void *out_loss_f64) {
  et_pp_config_t cfg;
  et_pp_status_t st = build_cfg(1, total_layers, num_microbatches,
                                tokens_per_mb, d_model, d_ff, out_dim,
                                ET_PP_SCHEDULE_1F1B, loss_type, 5000, rms_eps,
                                &cfg);
  if (st != ET_PP_OK)
    return st;
  if (!aligned_for(inputs, sizeof(double)) ||
      !aligned_for(params, sizeof(double)) ||
      !aligned_for(out_grads, sizeof(double)) ||
      !aligned_for(out_loss_f64, sizeof(double)))
    return ET_PP_ERR_INVALID_ARG;

  const int32_t *t_i32 = NULL;
  const double *t_f64 = NULL;
  if (cfg.loss_type == ET_PP_LOSS_CROSS_ENTROPY) {
    if (!aligned_for(targets, sizeof(int32_t)))
      return ET_PP_ERR_INVALID_ARG;
    t_i32 = (const int32_t *)targets;
  } else {
    if (!aligned_for(targets, sizeof(double)))
      return ET_PP_ERR_INVALID_ARG;
    t_f64 = (const double *)targets;
  }

  double loss = 0.0;
  st = et_pp_run_reference_step_v1(&cfg, (const double *)inputs, t_i32, t_f64,
                                   (const double *)params, (double *)out_grads,
                                   &loss);
  if (st == ET_PP_OK)
    memcpy(out_loss_f64, &loss, sizeof(double));
  return st;
}

int32_t et_pp_aot_run_multiprocess_step(
    int64_t num_stages, int64_t layers_per_stage, int64_t num_microbatches,
    int64_t tokens_per_mb, int64_t d_model, int64_t d_ff, int64_t out_dim,
    int64_t schedule_kind, int64_t loss_type, int64_t timeout_ms,
    double rms_eps, const void *inputs, const void *targets, const void *params,
    void *out_grads, void *out_metrics_f64, void *out_stage_metrics_i64) {
  et_pp_config_t cfg;
  et_pp_status_t st =
      build_cfg(num_stages, layers_per_stage, num_microbatches, tokens_per_mb,
                d_model, d_ff, out_dim, schedule_kind, loss_type, timeout_ms,
                rms_eps, &cfg);
  if (st != ET_PP_OK)
    return st;
  if (!aligned_for(inputs, sizeof(double)) ||
      !aligned_for(params, sizeof(double)) ||
      !aligned_for(out_grads, sizeof(double)) ||
      !aligned_for(out_metrics_f64, sizeof(double)) ||
      !aligned_for(out_stage_metrics_i64, sizeof(int64_t)))
    return ET_PP_ERR_INVALID_ARG;

  const int32_t *t_i32 = NULL;
  const double *t_f64 = NULL;
  if (cfg.loss_type == ET_PP_LOSS_CROSS_ENTROPY) {
    if (!aligned_for(targets, sizeof(int32_t)))
      return ET_PP_ERR_INVALID_ARG;
    t_i32 = (const int32_t *)targets;
  } else {
    if (!aligned_for(targets, sizeof(double)))
      return ET_PP_ERR_INVALID_ARG;
    t_f64 = (const double *)targets;
  }

  et_pp_run_telemetry_t telem;
  st = et_pp_run_multiprocess_step_v1(&cfg, (const double *)inputs, t_i32,
                                      t_f64, (const double *)params,
                                      (double *)out_grads, &telem);
  if (st != ET_PP_OK)
    return st;

  double metrics_f64[2];
  metrics_f64[0] = telem.total_loss;
  metrics_f64[1] = telem.bubble_fraction;
  memcpy(out_metrics_f64, metrics_f64, sizeof(metrics_f64));

  int64_t stage_i64[17];
  for (uint32_t s = 0; s < ET_PP_MAX_STAGES; ++s) {
    stage_i64[s] = (int64_t)telem.stage_peak_live[s];
    stage_i64[4u + s] = (int64_t)telem.stage_final_live[s];
    stage_i64[8u + s] = (int64_t)telem.stage_bytes_sent[s];
    stage_i64[12u + s] = (int64_t)telem.stage_bytes_recv[s];
  }
  stage_i64[16] = (int64_t)telem.distinct_pid_count;
  memcpy(out_stage_metrics_i64, stage_i64, sizeof(stage_i64));
  return ET_PP_OK;
}

int32_t et_pp_aot_train_reference_sgd(
    int64_t total_layers, int64_t num_microbatches, int64_t tokens_per_mb,
    int64_t d_model, int64_t d_ff, int64_t out_dim, int64_t loss_type,
    int64_t num_steps, double lr, double rms_eps, const void *inputs,
    const void *targets, const void *init_params, void *out_final_params,
    void *out_step_losses) {
  uint32_t steps = 0;
  if (!u32_from_i64(num_steps, &steps))
    return ET_PP_ERR_INVALID_ARG;
  et_pp_config_t cfg;
  et_pp_status_t st = build_cfg(1, total_layers, num_microbatches,
                                tokens_per_mb, d_model, d_ff, out_dim,
                                ET_PP_SCHEDULE_1F1B, loss_type, 5000, rms_eps,
                                &cfg);
  if (st != ET_PP_OK)
    return st;
  if (!aligned_for(inputs, sizeof(double)) ||
      !aligned_for(init_params, sizeof(double)) ||
      !aligned_for(out_final_params, sizeof(double)) ||
      !aligned_for(out_step_losses, sizeof(double)))
    return ET_PP_ERR_INVALID_ARG;

  const int32_t *t_i32 = NULL;
  const double *t_f64 = NULL;
  if (cfg.loss_type == ET_PP_LOSS_CROSS_ENTROPY) {
    if (!aligned_for(targets, sizeof(int32_t)))
      return ET_PP_ERR_INVALID_ARG;
    t_i32 = (const int32_t *)targets;
  } else {
    if (!aligned_for(targets, sizeof(double)))
      return ET_PP_ERR_INVALID_ARG;
    t_f64 = (const double *)targets;
  }

  return et_pp_train_reference_sgd_v1(
      &cfg, steps, lr, (const double *)inputs, t_i32, t_f64,
      (const double *)init_params, (double *)out_final_params,
      (double *)out_step_losses);
}

int32_t et_pp_aot_train_multiprocess_sgd(
    int64_t num_stages, int64_t layers_per_stage, int64_t num_microbatches,
    int64_t tokens_per_mb, int64_t d_model, int64_t d_ff, int64_t out_dim,
    int64_t schedule_kind, int64_t loss_type, int64_t num_steps,
    int64_t timeout_ms, double lr, double rms_eps, const void *inputs,
    const void *targets, const void *init_params, void *out_final_params,
    void *out_step_losses, void *out_peak_live_i64) {
  uint32_t steps = 0;
  if (!u32_from_i64(num_steps, &steps))
    return ET_PP_ERR_INVALID_ARG;
  et_pp_config_t cfg;
  et_pp_status_t st =
      build_cfg(num_stages, layers_per_stage, num_microbatches, tokens_per_mb,
                d_model, d_ff, out_dim, schedule_kind, loss_type, timeout_ms,
                rms_eps, &cfg);
  if (st != ET_PP_OK)
    return st;
  if (!aligned_for(inputs, sizeof(double)) ||
      !aligned_for(init_params, sizeof(double)) ||
      !aligned_for(out_final_params, sizeof(double)) ||
      !aligned_for(out_step_losses, sizeof(double)) ||
      !aligned_for(out_peak_live_i64, sizeof(int64_t)))
    return ET_PP_ERR_INVALID_ARG;

  const int32_t *t_i32 = NULL;
  const double *t_f64 = NULL;
  if (cfg.loss_type == ET_PP_LOSS_CROSS_ENTROPY) {
    if (!aligned_for(targets, sizeof(int32_t)))
      return ET_PP_ERR_INVALID_ARG;
    t_i32 = (const int32_t *)targets;
  } else {
    if (!aligned_for(targets, sizeof(double)))
      return ET_PP_ERR_INVALID_ARG;
    t_f64 = (const double *)targets;
  }

  et_pp_run_telemetry_t telem;
  st = et_pp_train_multiprocess_sgd_v1(
      &cfg, steps, lr, (const double *)inputs, t_i32, t_f64,
      (const double *)init_params, (double *)out_final_params,
      (double *)out_step_losses, &telem);
  if (st != ET_PP_OK)
    return st;

  int64_t peaks[5];
  for (uint32_t s = 0; s < ET_PP_MAX_STAGES; ++s)
    peaks[s] = (int64_t)telem.stage_peak_live[s];
  peaks[4] = (int64_t)telem.distinct_pid_count;
  memcpy(out_peak_live_i64, peaks, sizeof(peaks));
  return ET_PP_OK;
}

int32_t et_pp_aot_probe_transport_error(int64_t scenario, int64_t timeout_ms) {
  uint32_t to = 0;
  if (!u32_from_i64(timeout_ms, &to) || to == 0u || to > ET_PP_MAX_TIMEOUT_MS)
    return ET_PP_ERR_INVALID_ARG;

  int fds[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) != 0)
    return ET_PP_ERR_IO;

  et_pp_status_t st = ET_PP_ERR_INVALID_ARG;
  double payload[2] = {1.25, -0.5};
  double rx[2] = {0.0, 0.0};

  if (scenario == 0) {
    /* Mismatched microbatch_id: send mb=3, expect mb=2 */
    st = et_pp_p2p_send_frame_v1(fds[0], ET_PP_MSG_FORWARD_ACT, 3u, 0u, 1u,
                                 payload, sizeof(payload), to);
    if (st == ET_PP_OK) {
      st = et_pp_p2p_recv_frame_v1(fds[1], ET_PP_MSG_FORWARD_ACT, 2u, 0u, 1u,
                                   rx, sizeof(rx), to);
    }
  } else if (scenario == 1) {
    /* Peer disconnect */
    close(fds[0]);
    fds[0] = -1;
    st = et_pp_p2p_recv_frame_v1(fds[1], ET_PP_MSG_FORWARD_ACT, 0u, 0u, 1u, rx,
                                 sizeof(rx), to);
  } else if (scenario == 2) {
    /* Monotonic timeout: peer sends nothing */
    st = et_pp_p2p_recv_frame_v1(fds[1], ET_PP_MSG_FORWARD_ACT, 0u, 0u, 1u, rx,
                                 sizeof(rx), to);
  } else if (scenario == 3) {
    /* Non-finite activation over wire */
    payload[1] = (double)NAN;
    st = et_pp_p2p_send_frame_v1(fds[0], ET_PP_MSG_FORWARD_ACT, 0u, 0u, 1u,
                                 payload, sizeof(payload), to);
    if (st == ET_PP_OK) {
      st = et_pp_p2p_recv_frame_v1(fds[1], ET_PP_MSG_FORWARD_ACT, 0u, 0u, 1u,
                                   rx, sizeof(rx), to);
    }
  }

  if (fds[0] >= 0)
    close(fds[0]);
  if (fds[1] >= 0)
    close(fds[1]);
  return st;
}

void et_pp_aot_store_f64(void *buf, int64_t index, double value) {
  if (buf == NULL || index < 0)
    return;
  unsigned char *dst =
      (unsigned char *)buf + (size_t)index * sizeof(double);
  memcpy(dst, &value, sizeof(double));
}

double et_pp_aot_load_f64(const void *buf, int64_t index) {
  double value = 0.0;
  if (buf == NULL || index < 0)
    return 0.0;
  const unsigned char *src =
      (const unsigned char *)buf + (size_t)index * sizeof(double);
  memcpy(&value, src, sizeof(double));
  return value;
}

void et_pp_aot_store_i32(void *buf, int64_t index, int64_t value) {
  if (buf == NULL || index < 0)
    return;
  int32_t v = (int32_t)value;
  unsigned char *dst =
      (unsigned char *)buf + (size_t)index * sizeof(int32_t);
  memcpy(dst, &v, sizeof(int32_t));
}

int64_t et_pp_aot_load_i64(const void *buf, int64_t index) {
  int64_t value = 0;
  if (buf == NULL || index < 0)
    return 0;
  const unsigned char *src =
      (const unsigned char *)buf + (size_t)index * sizeof(int64_t);
  memcpy(&value, src, sizeof(int64_t));
  return value;
}

double et_pp_aot_make_nonfinite(int64_t kind) {
  if (kind == 1)
    return (double)INFINITY;
  if (kind == 2)
    return -(double)INFINITY;
  return (double)NAN;
}
