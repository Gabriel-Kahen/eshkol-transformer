#define _GNU_SOURCE
#include "eshkol_transformer/pipeline_1f1b_abi.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

_Static_assert(sizeof(double) == 8 && FLT_RADIX == 2 && DBL_MANT_DIG == 53 &&
                   DBL_MAX_EXP == 1024 && FLT_EVAL_METHOD == 0,
               "pipeline_1f1b requires IEEE-754 binary64 doubles");

_Static_assert(sizeof(et_pp_frame_header_t) == 32,
               "wire frame header must be 32 bytes without padding");

/* ------------------------------------------------------------------------- */
/* Helpers: monotonic clock, finite checks, socket validation                */
/* ------------------------------------------------------------------------- */

static int retryable_errno(int err) {
  return err == EINTR || err == EAGAIN || err == EWOULDBLOCK;
}

static int64_t monotonic_now_ms(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    return -1;
  return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static int all_finite_f64(const double *buf, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(buf[i]))
      return 0;
  }
  return 1;
}

static et_pp_status_t validate_unix_stream_fd(int fd) {
  struct sockaddr_storage local;
  struct sockaddr_storage peer;
  socklen_t len = sizeof(local);
  int sock_type = 0;
  if (fd < 0)
    return ET_PP_ERR_INVALID_ARG;
  if (getsockname(fd, (struct sockaddr *)&local, &len) != 0 ||
      local.ss_family != AF_UNIX)
    return ET_PP_ERR_INVALID_ARG;
  len = sizeof(peer);
  if (getpeername(fd, (struct sockaddr *)&peer, &len) != 0 ||
      peer.ss_family != AF_UNIX)
    return ET_PP_ERR_INVALID_ARG;
  len = sizeof(sock_type);
  if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &sock_type, &len) != 0 ||
      sock_type != SOCK_STREAM)
    return ET_PP_ERR_INVALID_ARG;
  return ET_PP_OK;
}

static et_pp_status_t wait_socket_ready(int fd, short want_event,
                                        int64_t deadline_ms) {
  for (;;) {
    int64_t now = monotonic_now_ms();
    if (now < 0)
      return ET_PP_ERR_IO;
    if (now >= deadline_ms)
      return ET_PP_ERR_TIMEOUT;
    int64_t rem = deadline_ms - now;
    int poll_ms = rem > 60000 ? 60000 : (int)rem;
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = (short)(want_event | POLLRDHUP);
    pfd.revents = 0;
    int rc = poll(&pfd, 1, poll_ms);
    if (rc < 0) {
      if (errno == EINTR)
        continue;
      return ET_PP_ERR_IO;
    }
    if (rc == 0)
      continue;
    if (pfd.revents & POLLNVAL)
      return ET_PP_ERR_IO;
    /* When reading, always allow draining buffered bytes before reporting HUP. */
    if (want_event == POLLIN) {
      if (pfd.revents & POLLIN)
        return ET_PP_OK;
      if (pfd.revents & (POLLHUP | POLLRDHUP | POLLERR))
        return ET_PP_ERR_PEER_DISCONNECTED;
    } else {
      if (pfd.revents & (POLLERR | POLLHUP))
        return ET_PP_ERR_PEER_DISCONNECTED;
      if (pfd.revents & POLLOUT)
        return ET_PP_OK;
    }
  }
}

static et_pp_status_t transfer_exact(int fd, void *buf, size_t size,
                                     int writing, int64_t deadline_ms) {
  unsigned char *cursor = (unsigned char *)buf;
  size_t remaining = size;
  while (remaining > 0) {
    et_pp_status_t st =
        wait_socket_ready(fd, writing ? POLLOUT : POLLIN, deadline_ms);
    if (st != ET_PP_OK)
      return st;
    ssize_t n = writing
                    ? send(fd, cursor, remaining, MSG_DONTWAIT | MSG_NOSIGNAL)
                    : recv(fd, cursor, remaining, MSG_DONTWAIT);
    if (n < 0) {
      if (retryable_errno(errno))
        continue;
      if (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN)
        return ET_PP_ERR_PEER_DISCONNECTED;
      return ET_PP_ERR_IO;
    }
    if (n == 0)
      return ET_PP_ERR_PEER_DISCONNECTED;
    cursor += (size_t)n;
    remaining -= (size_t)n;
  }
  return ET_PP_OK;
}

/* ------------------------------------------------------------------------- */
/* Public Framed AF_UNIX Point-to-Point Transport                            */
/* ------------------------------------------------------------------------- */

et_pp_status_t et_pp_p2p_send_frame_v1(int fd, uint32_t msg_type,
                                       uint32_t microbatch_id,
                                       uint32_t src_stage, uint32_t dst_stage,
                                       const void *payload,
                                       uint64_t payload_bytes,
                                       uint32_t timeout_ms) {
  if (timeout_ms == 0 || timeout_ms > ET_PP_MAX_TIMEOUT_MS)
    return ET_PP_ERR_INVALID_ARG;
  if (payload_bytes > 0 && payload == NULL)
    return ET_PP_ERR_INVALID_ARG;
  et_pp_status_t st = validate_unix_stream_fd(fd);
  if (st != ET_PP_OK)
    return st;

  int64_t start = monotonic_now_ms();
  if (start < 0)
    return ET_PP_ERR_IO;
  int64_t deadline = start + (int64_t)timeout_ms;

  et_pp_frame_header_t hdr;
  hdr.magic = ET_PP_WIRE_MAGIC;
  hdr.msg_type = msg_type;
  hdr.microbatch_id = microbatch_id;
  hdr.src_stage = src_stage;
  hdr.dst_stage = dst_stage;
  hdr.payload_bytes = payload_bytes;

  st = transfer_exact(fd, &hdr, sizeof(hdr), 1, deadline);
  if (st != ET_PP_OK)
    return st;
  if (payload_bytes > 0) {
    /* transfer_exact does not mutate buffer when writing == 1 */
    st = transfer_exact(fd, (void *)(uintptr_t)payload, (size_t)payload_bytes,
                        1, deadline);
    if (st != ET_PP_OK)
      return st;
  }
  return ET_PP_OK;
}

et_pp_status_t et_pp_p2p_recv_frame_v1(int fd, uint32_t expected_msg_type,
                                       uint32_t expected_microbatch_id,
                                       uint32_t expected_src_stage,
                                       uint32_t expected_dst_stage,
                                       void *payload_buf,
                                       uint64_t expected_payload_bytes,
                                       uint32_t timeout_ms) {
  if (timeout_ms == 0 || timeout_ms > ET_PP_MAX_TIMEOUT_MS)
    return ET_PP_ERR_INVALID_ARG;
  if (expected_payload_bytes > 0 && payload_buf == NULL)
    return ET_PP_ERR_INVALID_ARG;
  et_pp_status_t st = validate_unix_stream_fd(fd);
  if (st != ET_PP_OK)
    return st;

  int64_t start = monotonic_now_ms();
  if (start < 0)
    return ET_PP_ERR_IO;
  int64_t deadline = start + (int64_t)timeout_ms;

  et_pp_frame_header_t hdr;
  st = transfer_exact(fd, &hdr, sizeof(hdr), 0, deadline);
  if (st != ET_PP_OK)
    return st;

  if (hdr.magic != ET_PP_WIRE_MAGIC || hdr.msg_type != expected_msg_type ||
      hdr.microbatch_id != expected_microbatch_id ||
      hdr.src_stage != expected_src_stage ||
      hdr.dst_stage != expected_dst_stage ||
      hdr.payload_bytes != expected_payload_bytes)
    return ET_PP_ERR_PROTOCOL;

  if (expected_payload_bytes > 0) {
    st = transfer_exact(fd, payload_buf, (size_t)expected_payload_bytes, 0,
                        deadline);
    if (st != ET_PP_OK)
      return st;
    if ((expected_msg_type == ET_PP_MSG_FORWARD_ACT ||
         expected_msg_type == ET_PP_MSG_BACKWARD_GRAD) &&
        (expected_payload_bytes % sizeof(double) == 0)) {
      if (!all_finite_f64((const double *)payload_buf,
                          (size_t)(expected_payload_bytes / sizeof(double))))
        return ET_PP_ERR_NONFINITE;
    }
  }
  return ET_PP_OK;
}

/* ------------------------------------------------------------------------- */
/* Deterministic 1F1B & GPipe Schedule Generator                             */
/* ------------------------------------------------------------------------- */

et_pp_status_t et_pp_build_schedule_v1(uint32_t num_stages, uint32_t stage_id,
                                       uint32_t num_microbatches,
                                       int32_t schedule_kind,
                                       et_pp_schedule_t *out_schedule) {
  if (out_schedule == NULL)
    return ET_PP_ERR_INVALID_ARG;
  if (num_stages != 2u && num_stages != 4u)
    return ET_PP_ERR_INVALID_STAGES;
  if (stage_id >= num_stages)
    return ET_PP_ERR_INVALID_STAGES;
  if (num_microbatches < num_stages ||
      num_microbatches > ET_PP_MAX_MICROBATCHES)
    return ET_PP_ERR_INVALID_MICROBATCHES;
  if (schedule_kind != ET_PP_SCHEDULE_1F1B &&
      schedule_kind != ET_PP_SCHEDULE_GPIPE)
    return ET_PP_ERR_INVALID_ARG;

  memset(out_schedule, 0, sizeof(*out_schedule));
  out_schedule->num_stages = num_stages;
  out_schedule->stage_id = stage_id;
  out_schedule->num_microbatches = num_microbatches;
  out_schedule->schedule_kind = schedule_kind;
  out_schedule->bubble_fraction =
      (double)(num_stages - 1u) / (double)(num_microbatches + num_stages - 1u);

  uint32_t idx = 0;
  uint32_t live = 0;
  uint32_t peak = 0;

  if (schedule_kind == ET_PP_SCHEDULE_1F1B) {
    uint32_t num_warmup = num_stages - 1u - stage_id;
    uint32_t num_steady = num_microbatches - num_warmup;
    uint32_t num_cooldown = num_warmup;

    out_schedule->num_warmup_steps = num_warmup;
    out_schedule->num_steady_pairs = num_steady;
    out_schedule->num_cooldown_steps = num_cooldown;

    /* Warmup phase: F(0) .. F(num_warmup - 1) */
    for (uint32_t k = 0; k < num_warmup; ++k) {
      ++live;
      if (live > peak)
        peak = live;
      out_schedule->steps[idx].op = ET_PP_OP_FORWARD;
      out_schedule->steps[idx].phase = ET_PP_PHASE_WARMUP;
      out_schedule->steps[idx].microbatch_id = k;
      out_schedule->steps[idx].live_after = live;
      ++idx;
    }

    /* Steady-state phase: interleaved (F(num_warmup + i), B(i)) pairs */
    for (uint32_t i = 0; i < num_steady; ++i) {
      ++live;
      if (live > peak)
        peak = live;
      out_schedule->steps[idx].op = ET_PP_OP_FORWARD;
      out_schedule->steps[idx].phase = ET_PP_PHASE_STEADY;
      out_schedule->steps[idx].microbatch_id = num_warmup + i;
      out_schedule->steps[idx].live_after = live;
      ++idx;

      --live;
      out_schedule->steps[idx].op = ET_PP_OP_BACKWARD;
      out_schedule->steps[idx].phase = ET_PP_PHASE_STEADY;
      out_schedule->steps[idx].microbatch_id = i;
      out_schedule->steps[idx].live_after = live;
      ++idx;
    }

    /* Cooldown phase: B(num_steady) .. B(M - 1) */
    for (uint32_t k = 0; k < num_cooldown; ++k) {
      --live;
      out_schedule->steps[idx].op = ET_PP_OP_BACKWARD;
      out_schedule->steps[idx].phase = ET_PP_PHASE_COOLDOWN;
      out_schedule->steps[idx].microbatch_id = num_steady + k;
      out_schedule->steps[idx].live_after = live;
      ++idx;
    }
  } else {
    /* GPipe: all M forwards, then all M backwards */
    out_schedule->num_warmup_steps = num_microbatches;
    out_schedule->num_steady_pairs = 0u;
    out_schedule->num_cooldown_steps = num_microbatches;

    for (uint32_t m = 0; m < num_microbatches; ++m) {
      ++live;
      if (live > peak)
        peak = live;
      out_schedule->steps[idx].op = ET_PP_OP_FORWARD;
      out_schedule->steps[idx].phase = ET_PP_PHASE_WARMUP;
      out_schedule->steps[idx].microbatch_id = m;
      out_schedule->steps[idx].live_after = live;
      ++idx;
    }
    for (uint32_t m = 0; m < num_microbatches; ++m) {
      --live;
      out_schedule->steps[idx].op = ET_PP_OP_BACKWARD;
      out_schedule->steps[idx].phase = ET_PP_PHASE_COOLDOWN;
      out_schedule->steps[idx].microbatch_id = m;
      out_schedule->steps[idx].live_after = live;
      ++idx;
    }
  }

  out_schedule->step_count = idx;
  out_schedule->peak_live_activations = peak;
  return ET_PP_OK;
}

/* ------------------------------------------------------------------------- */
/* Config & Parameter Layout Helpers                                         */
/* ------------------------------------------------------------------------- */

static et_pp_status_t validate_config_internal(const et_pp_config_t *cfg,
                                               int allow_single_stage) {
  if (cfg == NULL)
    return ET_PP_ERR_INVALID_ARG;
  if (cfg->abi_major != ET_PP_ABI_MAJOR || cfg->abi_minor > ET_PP_ABI_MINOR)
    return ET_PP_ERR_INVALID_ARG;
  if (allow_single_stage) {
    if (cfg->num_stages != 1u && cfg->num_stages != 2u && cfg->num_stages != 4u)
      return ET_PP_ERR_INVALID_STAGES;
  } else {
    if (cfg->num_stages != 2u && cfg->num_stages != 4u)
      return ET_PP_ERR_INVALID_STAGES;
  }
  if (cfg->layers_per_stage == 0u ||
      cfg->layers_per_stage > ET_PP_MAX_LAYERS_PER_STAGE)
    return ET_PP_ERR_SHAPE;
  if (cfg->num_stages * cfg->layers_per_stage > ET_PP_MAX_TOTAL_LAYERS)
    return ET_PP_ERR_SHAPE;
  if (cfg->num_microbatches < cfg->num_stages ||
      cfg->num_microbatches > ET_PP_MAX_MICROBATCHES)
    return ET_PP_ERR_INVALID_MICROBATCHES;
  if (cfg->tokens_per_mb == 0u || cfg->tokens_per_mb > ET_PP_MAX_TOKENS_PER_MB)
    return ET_PP_ERR_SHAPE;
  if (cfg->d_model < 2u || cfg->d_model > ET_PP_MAX_D_MODEL)
    return ET_PP_ERR_SHAPE;
  if (cfg->d_ff < 2u || cfg->d_ff > ET_PP_MAX_D_FF)
    return ET_PP_ERR_SHAPE;
  if (cfg->out_dim < 2u || cfg->out_dim > ET_PP_MAX_OUT_DIM)
    return ET_PP_ERR_SHAPE;
  if (cfg->schedule_kind != ET_PP_SCHEDULE_1F1B &&
      cfg->schedule_kind != ET_PP_SCHEDULE_GPIPE)
    return ET_PP_ERR_INVALID_ARG;
  if (cfg->loss_type != ET_PP_LOSS_CROSS_ENTROPY &&
      cfg->loss_type != ET_PP_LOSS_MSE)
    return ET_PP_ERR_SHAPE;
  if (cfg->timeout_ms == 0u || cfg->timeout_ms > ET_PP_MAX_TIMEOUT_MS)
    return ET_PP_ERR_INVALID_ARG;
  if (!isfinite(cfg->rms_eps) || cfg->rms_eps <= 0.0)
    return ET_PP_ERR_INVALID_ARG;
  return ET_PP_OK;
}

et_pp_status_t et_pp_validate_config_v1(const et_pp_config_t *cfg) {
  return validate_config_internal(cfg, 0);
}

size_t et_pp_layer_param_count_v1(uint32_t d_model, uint32_t d_ff) {
  return (size_t)d_model + 3u * (size_t)d_ff * (size_t)d_model;
}

static size_t head_param_count(uint32_t d_model, uint32_t out_dim) {
  return (size_t)out_dim * (size_t)d_model + (size_t)out_dim;
}

static size_t stage_param_offset(const et_pp_config_t *cfg, uint32_t stage_id) {
  size_t per_layer = et_pp_layer_param_count_v1(cfg->d_model, cfg->d_ff);
  return (size_t)stage_id * (size_t)cfg->layers_per_stage * per_layer;
}

static size_t stage_param_count(const et_pp_config_t *cfg, uint32_t stage_id) {
  size_t per_layer = et_pp_layer_param_count_v1(cfg->d_model, cfg->d_ff);
  size_t count = (size_t)cfg->layers_per_stage * per_layer;
  if (stage_id + 1u == cfg->num_stages)
    count += head_param_count(cfg->d_model, cfg->out_dim);
  return count;
}

et_pp_status_t et_pp_param_count_v1(const et_pp_config_t *cfg,
                                    size_t *out_count) {
  if (out_count == NULL)
    return ET_PP_ERR_INVALID_ARG;
  et_pp_status_t st = validate_config_internal(cfg, 1);
  if (st != ET_PP_OK)
    return st;
  size_t total_layers = (size_t)cfg->num_stages * (size_t)cfg->layers_per_stage;
  *out_count = total_layers * et_pp_layer_param_count_v1(cfg->d_model, cfg->d_ff) +
               head_param_count(cfg->d_model, cfg->out_dim);
  return ET_PP_OK;
}

static et_pp_status_t validate_inputs_and_targets(const et_pp_config_t *cfg,
                                                  const double *inputs,
                                                  const int32_t *targets_i32,
                                                  const double *targets_f64,
                                                  const double *params) {
  if (inputs == NULL || params == NULL)
    return ET_PP_ERR_INVALID_ARG;
  size_t total_params = 0;
  et_pp_status_t st = et_pp_param_count_v1(cfg, &total_params);
  if (st != ET_PP_OK)
    return st;
  size_t input_elems = (size_t)cfg->num_microbatches *
                       (size_t)cfg->tokens_per_mb * (size_t)cfg->d_model;
  if (!all_finite_f64(inputs, input_elems) ||
      !all_finite_f64(params, total_params))
    return ET_PP_ERR_NONFINITE;

  size_t total_tokens =
      (size_t)cfg->num_microbatches * (size_t)cfg->tokens_per_mb;
  if (cfg->loss_type == ET_PP_LOSS_CROSS_ENTROPY) {
    if (targets_i32 == NULL)
      return ET_PP_ERR_INVALID_ARG;
    for (size_t i = 0; i < total_tokens; ++i) {
      if (targets_i32[i] < 0 || (uint32_t)targets_i32[i] >= cfg->out_dim)
        return ET_PP_ERR_TARGET_RANGE;
    }
  } else {
    if (targets_f64 == NULL)
      return ET_PP_ERR_INVALID_ARG;
    if (!all_finite_f64(targets_f64, total_tokens * (size_t)cfg->out_dim))
      return ET_PP_ERR_NONFINITE;
  }
  return ET_PP_OK;
}

/* ------------------------------------------------------------------------- */
/* Differentiable Pre-RMSNorm + SwiGLU Residual Layer Kernels                */
/* ------------------------------------------------------------------------- */

static double sigmoid_f64(double x) {
  if (x >= 0.0) {
    double e = exp(-x);
    return 1.0 / (1.0 + e);
  }
  double e = exp(x);
  return e / (1.0 + e);
}

typedef struct layer_cache {
  double *x_in;      /* [T * D] */
  double *r_inv;     /* [T] */
  double *norm;      /* [T * D] */
  double *gate;      /* [T * F] */
  double *up;        /* [T * F] */
  double *silu_gate; /* [T * F] */
} layer_cache_t;

static void layer_forward_cached(uint32_t T, uint32_t D, uint32_t F,
                                 double rms_eps, const double *x,
                                 const double *params, double *y,
                                 layer_cache_t *cache) {
  const double *gamma = params;
  const double *w_gate = gamma + D;
  const double *w_up = w_gate + (size_t)F * (size_t)D;
  const double *w_down = w_up + (size_t)F * (size_t)D;

  if (cache != NULL)
    memcpy(cache->x_in, x, (size_t)T * (size_t)D * sizeof(double));

  for (uint32_t t = 0; t < T; ++t) {
    const double *xt = x + (size_t)t * (size_t)D;
    double *yt = y + (size_t)t * (size_t)D;

    double norm_buf[ET_PP_MAX_D_MODEL];
    double act_buf[ET_PP_MAX_D_FF];
    double *nt = (cache != NULL) ? (cache->norm + (size_t)t * (size_t)D) : norm_buf;

    double sum_sq = 0.0;
    for (uint32_t j = 0; j < D; ++j)
      sum_sq += xt[j] * xt[j];
    double r = 1.0 / sqrt(sum_sq / (double)D + rms_eps);
    if (cache != NULL)
      cache->r_inv[t] = r;

    for (uint32_t j = 0; j < D; ++j)
      nt[j] = xt[j] * r * gamma[j];

    for (uint32_t k = 0; k < F; ++k) {
      const double *wg_k = w_gate + (size_t)k * (size_t)D;
      const double *wu_k = w_up + (size_t)k * (size_t)D;
      double g = 0.0;
      double u = 0.0;
      for (uint32_t j = 0; j < D; ++j) {
        g += nt[j] * wg_k[j];
        u += nt[j] * wu_k[j];
      }
      double s = g * sigmoid_f64(g);
      if (cache != NULL) {
        cache->gate[(size_t)t * (size_t)F + k] = g;
        cache->up[(size_t)t * (size_t)F + k] = u;
        cache->silu_gate[(size_t)t * (size_t)F + k] = s;
      }
      act_buf[k] = s * u;
    }

    for (uint32_t j = 0; j < D; ++j) {
      const double *wd_j = w_down + (size_t)j * (size_t)F;
      double acc = 0.0;
      for (uint32_t k = 0; k < F; ++k)
        acc += act_buf[k] * wd_j[k];
      yt[j] = xt[j] + acc;
    }
  }
}

static void layer_backward_accumulate(uint32_t T, uint32_t D, uint32_t F,
                                      const double *params,
                                      const layer_cache_t *cache,
                                      const double *dy, double *dx,
                                      double *dparams_accum) {
  const double *gamma = params;
  const double *w_gate = gamma + D;
  const double *w_up = w_gate + (size_t)F * (size_t)D;
  const double *w_down = w_up + (size_t)F * (size_t)D;

  double *dgamma = dparams_accum;
  double *dw_gate = dgamma + D;
  double *dw_up = dw_gate + (size_t)F * (size_t)D;
  double *dw_down = dw_up + (size_t)F * (size_t)D;

  for (uint32_t t = 0; t < T; ++t) {
    const double *xt = cache->x_in + (size_t)t * (size_t)D;
    const double *nt = cache->norm + (size_t)t * (size_t)D;
    const double *gt = cache->gate + (size_t)t * (size_t)F;
    const double *ut = cache->up + (size_t)t * (size_t)F;
    const double *st = cache->silu_gate + (size_t)t * (size_t)F;
    const double *dyt = dy + (size_t)t * (size_t)D;
    double *dxt = dx + (size_t)t * (size_t)D;
    double r = cache->r_inv[t];

    double da[ET_PP_MAX_D_FF];
    for (uint32_t k = 0; k < F; ++k)
      da[k] = 0.0;

    for (uint32_t j = 0; j < D; ++j) {
      const double *wd_j = w_down + (size_t)j * (size_t)F;
      double *dwd_j = dw_down + (size_t)j * (size_t)F;
      double dyj = dyt[j];
      for (uint32_t k = 0; k < F; ++k) {
        double a_tk = st[k] * ut[k];
        dwd_j[k] += dyj * a_tk;
        da[k] += dyj * wd_j[k];
      }
    }

    double dn[ET_PP_MAX_D_MODEL];
    for (uint32_t j = 0; j < D; ++j)
      dn[j] = 0.0;

    for (uint32_t k = 0; k < F; ++k) {
      double g = gt[k];
      double sig = sigmoid_f64(g);
      double silu_prime = sig * (1.0 + g * (1.0 - sig));
      double du = da[k] * st[k];
      double dg = da[k] * ut[k] * silu_prime;

      const double *wg_k = w_gate + (size_t)k * (size_t)D;
      const double *wu_k = w_up + (size_t)k * (size_t)D;
      double *dwg_k = dw_gate + (size_t)k * (size_t)D;
      double *dwu_k = dw_up + (size_t)k * (size_t)D;
      for (uint32_t j = 0; j < D; ++j) {
        dwg_k[j] += dg * nt[j];
        dwu_k[j] += du * nt[j];
        dn[j] += dg * wg_k[j] + du * wu_k[j];
      }
    }

    double c_sum = 0.0;
    for (uint32_t j = 0; j < D; ++j) {
      dgamma[j] += dn[j] * xt[j] * r;
      c_sum += dn[j] * gamma[j] * xt[j];
    }
    double c = c_sum / (double)D;
    double r2 = r * r;
    for (uint32_t j = 0; j < D; ++j) {
      double dx_norm = r * (dn[j] * gamma[j] - xt[j] * r2 * c);
      dxt[j] = dyt[j] + dx_norm;
    }
  }
}

et_pp_status_t et_pp_layer_forward_v1(uint32_t tokens, uint32_t d_model,
                                      uint32_t d_ff, double rms_eps,
                                      const double *x,
                                      const double *layer_params, double *y) {
  if (x == NULL || layer_params == NULL || y == NULL)
    return ET_PP_ERR_INVALID_ARG;
  if (tokens == 0u || tokens > ET_PP_MAX_TOKENS_PER_MB || d_model < 2u ||
      d_model > ET_PP_MAX_D_MODEL || d_ff < 2u || d_ff > ET_PP_MAX_D_FF)
    return ET_PP_ERR_SHAPE;
  if (!isfinite(rms_eps) || rms_eps <= 0.0)
    return ET_PP_ERR_INVALID_ARG;
  if (!all_finite_f64(x, (size_t)tokens * (size_t)d_model) ||
      !all_finite_f64(layer_params, et_pp_layer_param_count_v1(d_model, d_ff)))
    return ET_PP_ERR_NONFINITE;

  layer_forward_cached(tokens, d_model, d_ff, rms_eps, x, layer_params, y, NULL);
  if (!all_finite_f64(y, (size_t)tokens * (size_t)d_model))
    return ET_PP_ERR_NONFINITE;
  return ET_PP_OK;
}

et_pp_status_t et_pp_layer_backward_v1(uint32_t tokens, uint32_t d_model,
                                       uint32_t d_ff, double rms_eps,
                                       const double *x,
                                       const double *layer_params,
                                       const double *dy, double *dx,
                                       double *dlayer_params) {
  if (x == NULL || layer_params == NULL || dy == NULL || dx == NULL ||
      dlayer_params == NULL)
    return ET_PP_ERR_INVALID_ARG;
  if (tokens == 0u || tokens > ET_PP_MAX_TOKENS_PER_MB || d_model < 2u ||
      d_model > ET_PP_MAX_D_MODEL || d_ff < 2u || d_ff > ET_PP_MAX_D_FF)
    return ET_PP_ERR_SHAPE;
  if (!isfinite(rms_eps) || rms_eps <= 0.0)
    return ET_PP_ERR_INVALID_ARG;
  size_t pcount = et_pp_layer_param_count_v1(d_model, d_ff);
  size_t xcount = (size_t)tokens * (size_t)d_model;
  if (!all_finite_f64(x, xcount) || !all_finite_f64(layer_params, pcount) ||
      !all_finite_f64(dy, xcount))
    return ET_PP_ERR_NONFINITE;

  double x_in_buf[ET_PP_MAX_TOKENS_PER_MB * ET_PP_MAX_D_MODEL];
  double r_inv_buf[ET_PP_MAX_TOKENS_PER_MB];
  double norm_buf[ET_PP_MAX_TOKENS_PER_MB * ET_PP_MAX_D_MODEL];
  double gate_buf[ET_PP_MAX_TOKENS_PER_MB * ET_PP_MAX_D_FF];
  double up_buf[ET_PP_MAX_TOKENS_PER_MB * ET_PP_MAX_D_FF];
  double silu_buf[ET_PP_MAX_TOKENS_PER_MB * ET_PP_MAX_D_FF];
  double y_tmp[ET_PP_MAX_TOKENS_PER_MB * ET_PP_MAX_D_MODEL];

  layer_cache_t cache;
  cache.x_in = x_in_buf;
  cache.r_inv = r_inv_buf;
  cache.norm = norm_buf;
  cache.gate = gate_buf;
  cache.up = up_buf;
  cache.silu_gate = silu_buf;

  layer_forward_cached(tokens, d_model, d_ff, rms_eps, x, layer_params, y_tmp,
                       &cache);
  memset(dlayer_params, 0, pcount * sizeof(double));
  layer_backward_accumulate(tokens, d_model, d_ff, layer_params, &cache, dy, dx,
                            dlayer_params);
  if (!all_finite_f64(dx, xcount) || !all_finite_f64(dlayer_params, pcount))
    return ET_PP_ERR_NONFINITE;
  return ET_PP_OK;
}

/* ------------------------------------------------------------------------- */
/* Output Head Forward + Loss + Backward                                     */
/* ------------------------------------------------------------------------- */

static double head_forward_and_loss(const et_pp_config_t *cfg,
                                    const double *h_in,
                                    const double *head_params,
                                    const int32_t *mb_targets_i32,
                                    const double *mb_targets_f64,
                                    double *logits_out) {
  uint32_t T = cfg->tokens_per_mb;
  uint32_t D = cfg->d_model;
  uint32_t V = cfg->out_dim;
  const double *w_head = head_params;
  const double *b_head = head_params + (size_t)V * (size_t)D;

  for (uint32_t t = 0; t < T; ++t) {
    const double *ht = h_in + (size_t)t * (size_t)D;
    double *zt = logits_out + (size_t)t * (size_t)V;
    for (uint32_t v = 0; v < V; ++v) {
      const double *wv = w_head + (size_t)v * (size_t)D;
      double acc = b_head[v];
      for (uint32_t j = 0; j < D; ++j)
        acc += ht[j] * wv[j];
      zt[v] = acc;
    }
  }

  double mb_loss = 0.0;
  if (cfg->loss_type == ET_PP_LOSS_CROSS_ENTROPY) {
    for (uint32_t t = 0; t < T; ++t) {
      const double *zt = logits_out + (size_t)t * (size_t)V;
      double zmax = zt[0];
      for (uint32_t v = 1; v < V; ++v) {
        if (zt[v] > zmax)
          zmax = zt[v];
      }
      double sum_exp = 0.0;
      for (uint32_t v = 0; v < V; ++v)
        sum_exp += exp(zt[v] - zmax);
      double lse = zmax + log(sum_exp);
      int32_t y_idx = mb_targets_i32[t];
      mb_loss += (lse - zt[y_idx]);
    }
    mb_loss /= (double)T;
  } else {
    for (uint32_t t = 0; t < T; ++t) {
      const double *zt = logits_out + (size_t)t * (size_t)V;
      const double *tgt = mb_targets_f64 + (size_t)t * (size_t)V;
      for (uint32_t v = 0; v < V; ++v) {
        double diff = zt[v] - tgt[v];
        mb_loss += 0.5 * diff * diff;
      }
    }
    mb_loss /= (double)((size_t)T * (size_t)V);
  }
  return mb_loss / (double)cfg->num_microbatches;
}

static void head_backward_accumulate(const et_pp_config_t *cfg,
                                     const double *h_in, const double *logits,
                                     const double *head_params,
                                     const int32_t *mb_targets_i32,
                                     const double *mb_targets_f64,
                                     double *dh_out,
                                     double *dhead_params_accum) {
  uint32_t T = cfg->tokens_per_mb;
  uint32_t D = cfg->d_model;
  uint32_t V = cfg->out_dim;
  const double *w_head = head_params;
  double *dw_head = dhead_params_accum;
  double *db_head = dhead_params_accum + (size_t)V * (size_t)D;

  for (uint32_t t = 0; t < T; ++t) {
    const double *ht = h_in + (size_t)t * (size_t)D;
    const double *zt = logits + (size_t)t * (size_t)V;
    double *dht = dh_out + (size_t)t * (size_t)D;
    double dz[ET_PP_MAX_OUT_DIM];

    if (cfg->loss_type == ET_PP_LOSS_CROSS_ENTROPY) {
      double zmax = zt[0];
      for (uint32_t v = 1; v < V; ++v) {
        if (zt[v] > zmax)
          zmax = zt[v];
      }
      double sum_exp = 0.0;
      for (uint32_t v = 0; v < V; ++v) {
        dz[v] = exp(zt[v] - zmax);
        sum_exp += dz[v];
      }
      double scale = 1.0 / ((double)cfg->num_microbatches * (double)T);
      int32_t y_idx = mb_targets_i32[t];
      for (uint32_t v = 0; v < V; ++v) {
        double p = dz[v] / sum_exp;
        dz[v] = (p - (v == (uint32_t)y_idx ? 1.0 : 0.0)) * scale;
      }
    } else {
      const double *tgt = mb_targets_f64 + (size_t)t * (size_t)V;
      double scale =
          1.0 / ((double)cfg->num_microbatches * (double)T * (double)V);
      for (uint32_t v = 0; v < V; ++v)
        dz[v] = (zt[v] - tgt[v]) * scale;
    }

    for (uint32_t j = 0; j < D; ++j)
      dht[j] = 0.0;

    for (uint32_t v = 0; v < V; ++v) {
      const double *wv = w_head + (size_t)v * (size_t)D;
      double *dwv = dw_head + (size_t)v * (size_t)D;
      double dzv = dz[v];
      db_head[v] += dzv;
      for (uint32_t j = 0; j < D; ++j) {
        dwv[j] += dzv * ht[j];
        dht[j] += dzv * wv[j];
      }
    }
  }
}

/* ------------------------------------------------------------------------- */
/* Activation Slot Pool (Enforces Peak Live Activation Memory Bound!)        */
/* ------------------------------------------------------------------------- */

typedef struct stage_slot {
  uint32_t mb_id;
  layer_cache_t layers[ET_PP_MAX_LAYERS_PER_STAGE];
  double *final_h; /* [T * D], only allocated on last stage */
  double *logits;  /* [T * V], only allocated on last stage */
} stage_slot_t;

typedef struct stage_pool {
  uint32_t slot_cap;
  uint32_t layers_per_stage;
  int is_last_stage;
  stage_slot_t *slots;
  double *arena;
} stage_pool_t;

static et_pp_status_t stage_pool_init(stage_pool_t *pool, uint32_t slot_cap,
                                      uint32_t layers_per_stage,
                                      int is_last_stage, uint32_t T, uint32_t D,
                                      uint32_t F, uint32_t V) {
  memset(pool, 0, sizeof(*pool));
  pool->slot_cap = slot_cap;
  pool->layers_per_stage = layers_per_stage;
  pool->is_last_stage = is_last_stage;

  pool->slots = (stage_slot_t *)calloc(slot_cap, sizeof(stage_slot_t));
  if (pool->slots == NULL)
    return ET_PP_ERR_ALLOC;

  size_t per_layer_doubles = 2u * (size_t)T * (size_t)D + (size_t)T +
                             3u * (size_t)T * (size_t)F;
  size_t per_slot_doubles = (size_t)layers_per_stage * per_layer_doubles;
  if (is_last_stage)
    per_slot_doubles += (size_t)T * (size_t)D + (size_t)T * (size_t)V;

  pool->arena = (double *)calloc((size_t)slot_cap * per_slot_doubles,
                                 sizeof(double));
  if (pool->arena == NULL) {
    free(pool->slots);
    pool->slots = NULL;
    return ET_PP_ERR_ALLOC;
  }

  double *ptr = pool->arena;
  for (uint32_t s = 0; s < slot_cap; ++s) {
    pool->slots[s].mb_id = UINT32_MAX;
    for (uint32_t l = 0; l < layers_per_stage; ++l) {
      layer_cache_t *lc = &pool->slots[s].layers[l];
      lc->x_in = ptr;
      ptr += (size_t)T * (size_t)D;
      lc->r_inv = ptr;
      ptr += (size_t)T;
      lc->norm = ptr;
      ptr += (size_t)T * (size_t)D;
      lc->gate = ptr;
      ptr += (size_t)T * (size_t)F;
      lc->up = ptr;
      ptr += (size_t)T * (size_t)F;
      lc->silu_gate = ptr;
      ptr += (size_t)T * (size_t)F;
    }
    if (is_last_stage) {
      pool->slots[s].final_h = ptr;
      ptr += (size_t)T * (size_t)D;
      pool->slots[s].logits = ptr;
      ptr += (size_t)T * (size_t)V;
    }
  }
  return ET_PP_OK;
}

static void stage_pool_destroy(stage_pool_t *pool) {
  free(pool->arena);
  free(pool->slots);
  memset(pool, 0, sizeof(*pool));
}

/* ------------------------------------------------------------------------- */
/* Single-Process Unpipelined Full-Model Reference                           */
/* ------------------------------------------------------------------------- */

static et_pp_status_t run_reference_step_internal(
    const et_pp_config_t *cfg, const double *inputs, const int32_t *targets_i32,
    const double *targets_f64, const double *params, double *out_grads,
    double *out_loss) {
  uint32_t L = cfg->num_stages * cfg->layers_per_stage;
  uint32_t M = cfg->num_microbatches;
  uint32_t T = cfg->tokens_per_mb;
  uint32_t D = cfg->d_model;
  uint32_t F = cfg->d_ff;
  uint32_t V = cfg->out_dim;
  size_t per_layer = et_pp_layer_param_count_v1(D, F);
  size_t total_params = (size_t)L * per_layer + head_param_count(D, V);

  /* Reference runs one microbatch completely through all L layers at a time */
  layer_cache_t *caches = (layer_cache_t *)calloc(L, sizeof(layer_cache_t));
  size_t per_layer_doubles = 2u * (size_t)T * (size_t)D + (size_t)T +
                             3u * (size_t)T * (size_t)F;
  double *arena =
      (double *)calloc((size_t)L * per_layer_doubles, sizeof(double));
  if (caches == NULL || arena == NULL) {
    free(caches);
    free(arena);
    return ET_PP_ERR_ALLOC;
  }
  double *ptr = arena;
  for (uint32_t l = 0; l < L; ++l) {
    caches[l].x_in = ptr;
    ptr += (size_t)T * (size_t)D;
    caches[l].r_inv = ptr;
    ptr += (size_t)T;
    caches[l].norm = ptr;
    ptr += (size_t)T * (size_t)D;
    caches[l].gate = ptr;
    ptr += (size_t)T * (size_t)F;
    caches[l].up = ptr;
    ptr += (size_t)T * (size_t)F;
    caches[l].silu_gate = ptr;
    ptr += (size_t)T * (size_t)F;
  }

  double buf_a[ET_PP_MAX_TOKENS_PER_MB * ET_PP_MAX_D_MODEL];
  double buf_b[ET_PP_MAX_TOKENS_PER_MB * ET_PP_MAX_D_MODEL];
  double logits[ET_PP_MAX_TOKENS_PER_MB * ET_PP_MAX_OUT_DIM];
  double total_loss = 0.0;

  memset(out_grads, 0, total_params * sizeof(double));

  for (uint32_t m = 0; m < M; ++m) {
    const double *cur_in = inputs + (size_t)m * (size_t)T * (size_t)D;
    memcpy(buf_a, cur_in, (size_t)T * (size_t)D * sizeof(double));

    for (uint32_t l = 0; l < L; ++l) {
      const double *lp = params + (size_t)l * per_layer;
      layer_forward_cached(T, D, F, cfg->rms_eps, buf_a, lp, buf_b, &caches[l]);
      memcpy(buf_a, buf_b, (size_t)T * (size_t)D * sizeof(double));
    }

    const double *head_p = params + (size_t)L * per_layer;
    double *dhead_p = out_grads + (size_t)L * per_layer;
    const int32_t *mb_ti =
        (targets_i32 != NULL) ? (targets_i32 + (size_t)m * (size_t)T) : NULL;
    const double *mb_tf = (targets_f64 != NULL)
                              ? (targets_f64 + (size_t)m * (size_t)T * (size_t)V)
                              : NULL;

    total_loss += head_forward_and_loss(cfg, buf_a, head_p, mb_ti, mb_tf, logits);
    head_backward_accumulate(cfg, buf_a, logits, head_p, mb_ti, mb_tf, buf_b,
                             dhead_p);
    memcpy(buf_a, buf_b, (size_t)T * (size_t)D * sizeof(double));

    for (uint32_t rev = 0; rev < L; ++rev) {
      uint32_t l = L - 1u - rev;
      const double *lp = params + (size_t)l * per_layer;
      double *dlp = out_grads + (size_t)l * per_layer;
      layer_backward_accumulate(T, D, F, lp, &caches[l], buf_a, buf_b, dlp);
      memcpy(buf_a, buf_b, (size_t)T * (size_t)D * sizeof(double));
    }
  }

  free(caches);
  free(arena);

  if (!isfinite(total_loss) || !all_finite_f64(out_grads, total_params))
    return ET_PP_ERR_NONFINITE;
  *out_loss = total_loss;
  return ET_PP_OK;
}

et_pp_status_t et_pp_run_reference_step_v1(const et_pp_config_t *cfg,
                                           const double *inputs,
                                           const int32_t *targets_i32,
                                           const double *targets_f64,
                                           const double *params,
                                           double *out_grads,
                                           double *out_loss) {
  if (out_grads == NULL || out_loss == NULL)
    return ET_PP_ERR_INVALID_ARG;
  et_pp_status_t st = validate_config_internal(cfg, 1);
  if (st != ET_PP_OK)
    return st;
  st = validate_inputs_and_targets(cfg, inputs, targets_i32, targets_f64,
                                   params);
  if (st != ET_PP_OK)
    return st;
  return run_reference_step_internal(cfg, inputs, targets_i32, targets_f64,
                                     params, out_grads, out_loss);
}

et_pp_status_t et_pp_train_reference_sgd_v1(
    const et_pp_config_t *cfg, uint32_t num_steps, double lr,
    const double *inputs, const int32_t *targets_i32,
    const double *targets_f64, const double *init_params,
    double *out_final_params, double *out_step_losses) {
  if (num_steps == 0u || num_steps > ET_PP_MAX_TRAIN_STEPS || !isfinite(lr) ||
      lr <= 0.0 || out_final_params == NULL || out_step_losses == NULL)
    return ET_PP_ERR_INVALID_ARG;
  et_pp_status_t st = validate_config_internal(cfg, 1);
  if (st != ET_PP_OK)
    return st;
  st = validate_inputs_and_targets(cfg, inputs, targets_i32, targets_f64,
                                   init_params);
  if (st != ET_PP_OK)
    return st;

  size_t total_params = 0;
  st = et_pp_param_count_v1(cfg, &total_params);
  if (st != ET_PP_OK)
    return st;

  double *grads = (double *)calloc(total_params, sizeof(double));
  if (grads == NULL)
    return ET_PP_ERR_ALLOC;

  memcpy(out_final_params, init_params, total_params * sizeof(double));
  for (uint32_t step = 0; step < num_steps; ++step) {
    double step_loss = 0.0;
    st = run_reference_step_internal(cfg, inputs, targets_i32, targets_f64,
                                     out_final_params, grads, &step_loss);
    if (st != ET_PP_OK) {
      free(grads);
      return st;
    }
    out_step_losses[step] = step_loss;
    for (size_t i = 0; i < total_params; ++i)
      out_final_params[i] -= lr * grads[i];
  }
  free(grads);
  return ET_PP_OK;
}

/* ------------------------------------------------------------------------- */
/* Real Multi-Process Stage Worker & Coordinator                             */
/* ------------------------------------------------------------------------- */

typedef struct stage_reply_header {
  int32_t status;
  int32_t pid;
  uint32_t peak_live;
  uint32_t final_live;
  uint32_t fwd_count;
  uint32_t bwd_count;
  uint64_t bytes_sent;
  uint64_t bytes_recv;
  double stage_loss;
} stage_reply_header_t;

static et_pp_status_t run_stage_one_step(
    const et_pp_config_t *cfg, uint32_t stage_id, const et_pp_schedule_t *sched,
    int prev_fd, int next_fd, const double *inputs, const int32_t *targets_i32,
    const double *targets_f64, const double *stage_params,
    double *stage_grads_out, stage_pool_t *pool, stage_reply_header_t *reply) {
  uint32_t P = cfg->num_stages;
  uint32_t L_s = cfg->layers_per_stage;
  uint32_t T = cfg->tokens_per_mb;
  uint32_t D = cfg->d_model;
  uint32_t F = cfg->d_ff;
  uint32_t V = cfg->out_dim;
  size_t per_layer = et_pp_layer_param_count_v1(D, F);
  size_t local_pcount = stage_param_count(cfg, stage_id);
  size_t act_elems = (size_t)T * (size_t)D;
  uint64_t act_bytes = (uint64_t)(act_elems * sizeof(double));
  int is_first = (stage_id == 0u);
  int is_last = (stage_id + 1u == P);

  memset(stage_grads_out, 0, local_pcount * sizeof(double));
  for (uint32_t i = 0; i < pool->slot_cap; ++i)
    pool->slots[i].mb_id = UINT32_MAX;

  uint32_t live = 0;
  uint32_t peak_live = 0;
  uint32_t fwd_count = 0;
  uint32_t bwd_count = 0;
  uint64_t bytes_sent = 0;
  uint64_t bytes_recv = 0;
  double total_loss = 0.0;

  double buf_a[ET_PP_MAX_TOKENS_PER_MB * ET_PP_MAX_D_MODEL];
  double buf_b[ET_PP_MAX_TOKENS_PER_MB * ET_PP_MAX_D_MODEL];

  for (uint32_t step_i = 0; step_i < sched->step_count; ++step_i) {
    const et_pp_schedule_step_t *step = &sched->steps[step_i];
    uint32_t mb = step->microbatch_id;
    uint32_t slot_idx = mb % pool->slot_cap;
    stage_slot_t *slot = &pool->slots[slot_idx];

    if (step->op == ET_PP_OP_FORWARD) {
      if (live >= pool->slot_cap || slot->mb_id != UINT32_MAX)
        return ET_PP_ERR_SCHEDULE;

      if (is_first) {
        memcpy(buf_a, inputs + (size_t)mb * act_elems, (size_t)act_bytes);
      } else {
        et_pp_status_t st = et_pp_p2p_recv_frame_v1(
            prev_fd, ET_PP_MSG_FORWARD_ACT, mb, stage_id - 1u, stage_id, buf_a,
            act_bytes, cfg->timeout_ms);
        if (st != ET_PP_OK)
          return st;
        bytes_recv += sizeof(et_pp_frame_header_t) + act_bytes;
      }

      for (uint32_t l = 0; l < L_s; ++l) {
        const double *lp = stage_params + (size_t)l * per_layer;
        layer_forward_cached(T, D, F, cfg->rms_eps, buf_a, lp, buf_b,
                             &slot->layers[l]);
        memcpy(buf_a, buf_b, (size_t)act_bytes);
      }

      if (is_last) {
        memcpy(slot->final_h, buf_a, (size_t)act_bytes);
        const double *head_p = stage_params + (size_t)L_s * per_layer;
        const int32_t *mb_ti =
            (targets_i32 != NULL) ? (targets_i32 + (size_t)mb * (size_t)T) : NULL;
        const double *mb_tf =
            (targets_f64 != NULL)
                ? (targets_f64 + (size_t)mb * (size_t)T * (size_t)V)
                : NULL;
        total_loss += head_forward_and_loss(cfg, slot->final_h, head_p, mb_ti,
                                            mb_tf, slot->logits);
      } else {
        et_pp_status_t st = et_pp_p2p_send_frame_v1(
            next_fd, ET_PP_MSG_FORWARD_ACT, mb, stage_id, stage_id + 1u, buf_a,
            act_bytes, cfg->timeout_ms);
        if (st != ET_PP_OK)
          return st;
        bytes_sent += sizeof(et_pp_frame_header_t) + act_bytes;
      }

      slot->mb_id = mb;
      ++live;
      ++fwd_count;
      if (live > peak_live)
        peak_live = live;
    } else if (step->op == ET_PP_OP_BACKWARD) {
      if (live == 0u || slot->mb_id != mb)
        return ET_PP_ERR_SCHEDULE;

      if (is_last) {
        const double *head_p = stage_params + (size_t)L_s * per_layer;
        double *dhead_p = stage_grads_out + (size_t)L_s * per_layer;
        const int32_t *mb_ti =
            (targets_i32 != NULL) ? (targets_i32 + (size_t)mb * (size_t)T) : NULL;
        const double *mb_tf =
            (targets_f64 != NULL)
                ? (targets_f64 + (size_t)mb * (size_t)T * (size_t)V)
                : NULL;
        head_backward_accumulate(cfg, slot->final_h, slot->logits, head_p,
                                 mb_ti, mb_tf, buf_a, dhead_p);
      } else {
        et_pp_status_t st = et_pp_p2p_recv_frame_v1(
            next_fd, ET_PP_MSG_BACKWARD_GRAD, mb, stage_id + 1u, stage_id,
            buf_a, act_bytes, cfg->timeout_ms);
        if (st != ET_PP_OK)
          return st;
        bytes_recv += sizeof(et_pp_frame_header_t) + act_bytes;
      }

      for (uint32_t rev = 0; rev < L_s; ++rev) {
        uint32_t l = L_s - 1u - rev;
        const double *lp = stage_params + (size_t)l * per_layer;
        double *dlp = stage_grads_out + (size_t)l * per_layer;
        layer_backward_accumulate(T, D, F, lp, &slot->layers[l], buf_a, buf_b,
                                  dlp);
        memcpy(buf_a, buf_b, (size_t)act_bytes);
      }

      if (!is_first) {
        et_pp_status_t st = et_pp_p2p_send_frame_v1(
            prev_fd, ET_PP_MSG_BACKWARD_GRAD, mb, stage_id, stage_id - 1u,
            buf_a, act_bytes, cfg->timeout_ms);
        if (st != ET_PP_OK)
          return st;
        bytes_sent += sizeof(et_pp_frame_header_t) + act_bytes;
      }

      slot->mb_id = UINT32_MAX;
      --live;
      ++bwd_count;
    } else {
      return ET_PP_ERR_SCHEDULE;
    }
  }

  if (live != 0u)
    return ET_PP_ERR_SCHEDULE;
  if (!isfinite(total_loss) || !all_finite_f64(stage_grads_out, local_pcount))
    return ET_PP_ERR_NONFINITE;

  reply->status = ET_PP_OK;
  reply->pid = (int32_t)getpid();
  reply->peak_live = peak_live;
  reply->final_live = live;
  reply->fwd_count = fwd_count;
  reply->bwd_count = bwd_count;
  reply->bytes_sent = bytes_sent;
  reply->bytes_recv = bytes_recv;
  reply->stage_loss = total_loss;
  return ET_PP_OK;
}

static void close_fd_if_open(int *fd) {
  if (*fd >= 0) {
    close(*fd);
    *fd = -1;
  }
}

static void reap_all_children(pid_t *pids, uint32_t count, int terminate_first) {
  if (terminate_first) {
    for (uint32_t s = 0; s < count; ++s) {
      if (pids[s] > 0)
        kill(pids[s], SIGTERM);
    }
  }
  for (uint32_t s = 0; s < count; ++s) {
    if (pids[s] > 0) {
      int wstatus = 0;
      while (waitpid(pids[s], &wstatus, 0) < 0) {
        if (errno != EINTR)
          break;
      }
      pids[s] = -1;
    }
  }
}

static uint32_t count_distinct_positive_pids(const int32_t *pids, uint32_t n) {
  uint32_t distinct = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (pids[i] <= 0)
      continue;
    int seen = 0;
    for (uint32_t j = 0; j < i; ++j) {
      if (pids[j] == pids[i]) {
        seen = 1;
        break;
      }
    }
    if (!seen)
      ++distinct;
  }
  return distinct;
}

/*
 * Common multi-process pipeline driver supporting both single-step gradient
 * computation (num_steps == 0) and multi-step SGD training (num_steps >= 1).
 */
static et_pp_status_t run_multiprocess_driver(
    const et_pp_config_t *cfg, uint32_t num_steps, double lr,
    const double *inputs, const int32_t *targets_i32,
    const double *targets_f64, const double *init_params, double *out_grads,
    double *out_final_params, double *out_step_losses,
    et_pp_run_telemetry_t *out_telemetry) {
  uint32_t P = cfg->num_stages;
  int link_fds[ET_PP_MAX_STAGES][2];
  int ctrl_fds[ET_PP_MAX_STAGES][2];
  pid_t pids[ET_PP_MAX_STAGES];

  for (uint32_t s = 0; s < ET_PP_MAX_STAGES; ++s) {
    link_fds[s][0] = -1;
    link_fds[s][1] = -1;
    ctrl_fds[s][0] = -1;
    ctrl_fds[s][1] = -1;
    pids[s] = -1;
  }

  /* Create P-1 adjacent stage AF_UNIX socketpairs + P coordinator sockets */
  for (uint32_t s = 0; s + 1u < P; ++s) {
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, link_fds[s]) != 0)
      goto fail_io;
  }
  for (uint32_t s = 0; s < P; ++s) {
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, ctrl_fds[s]) != 0)
      goto fail_io;
  }

  for (uint32_t s = 0; s < P; ++s) {
    pid_t pid = fork();
    if (pid < 0)
      goto fail_io;
    if (pid == 0) {
      /* Child process for stage s */
      int prev_fd = (s > 0u) ? link_fds[s - 1u][1] : -1;
      int next_fd = (s + 1u < P) ? link_fds[s][0] : -1;
      int ctrl_fd = ctrl_fds[s][1];

      for (uint32_t i = 0; i + 1u < P; ++i) {
        if (link_fds[i][0] != prev_fd && link_fds[i][0] != next_fd)
          close_fd_if_open(&link_fds[i][0]);
        if (link_fds[i][1] != prev_fd && link_fds[i][1] != next_fd)
          close_fd_if_open(&link_fds[i][1]);
      }
      for (uint32_t i = 0; i < P; ++i) {
        close_fd_if_open(&ctrl_fds[i][0]);
        if (i != s)
          close_fd_if_open(&ctrl_fds[i][1]);
      }

      et_pp_schedule_t sched;
      et_pp_status_t st = et_pp_build_schedule_v1(
          P, s, cfg->num_microbatches, cfg->schedule_kind, &sched);

      size_t p_off = stage_param_offset(cfg, s);
      size_t p_cnt = stage_param_count(cfg, s);
      double *local_params = (double *)malloc(p_cnt * sizeof(double));
      double *local_grads = (double *)calloc(p_cnt, sizeof(double));
      double *local_losses =
          (num_steps > 0u && s + 1u == P)
              ? (double *)calloc(num_steps, sizeof(double))
              : NULL;
      stage_pool_t pool;
      memset(&pool, 0, sizeof(pool));

      stage_reply_header_t reply;
      memset(&reply, 0, sizeof(reply));
      reply.pid = (int32_t)getpid();
      reply.status = ET_PP_ERR_ALLOC;

      if (st == ET_PP_OK && local_params != NULL && local_grads != NULL &&
          (num_steps == 0u || s + 1u != P || local_losses != NULL)) {
        memcpy(local_params, init_params + p_off, p_cnt * sizeof(double));
        st = stage_pool_init(&pool, sched.peak_live_activations,
                             cfg->layers_per_stage, (s + 1u == P),
                             cfg->tokens_per_mb, cfg->d_model, cfg->d_ff,
                             cfg->out_dim);
        if (st == ET_PP_OK) {
          uint32_t iters = (num_steps == 0u) ? 1u : num_steps;
          for (uint32_t iter = 0; iter < iters; ++iter) {
            st = run_stage_one_step(
                cfg, s, &sched, prev_fd, next_fd,
                (s == 0u) ? inputs : NULL,
                (s + 1u == P) ? targets_i32 : NULL,
                (s + 1u == P) ? targets_f64 : NULL,
                local_params, local_grads, &pool, &reply);
            if (st != ET_PP_OK) {
              reply.status = st;
              break;
            }
            if (num_steps > 0u) {
              if (s + 1u == P)
                local_losses[iter] = reply.stage_loss;
              for (size_t k = 0; k < p_cnt; ++k)
                local_params[k] -= lr * local_grads[k];
            }
          }
        } else {
          reply.status = st;
        }
      } else if (st != ET_PP_OK) {
        reply.status = st;
      }

      /* Close stage-to-stage links promptly so peers observe EOF on failure */
      close_fd_if_open(&prev_fd);
      close_fd_if_open(&next_fd);

      /* Assemble single framed reply to coordinator */
      size_t extra_doubles = p_cnt + ((num_steps > 0u && s + 1u == P) ? num_steps : 0u);
      size_t payload_bytes = sizeof(reply) + extra_doubles * sizeof(double);
      unsigned char *msg = (unsigned char *)calloc(1, payload_bytes);
      if (msg != NULL) {
        memcpy(msg, &reply, sizeof(reply));
        if (reply.status == ET_PP_OK) {
          const double *vec = (num_steps == 0u) ? local_grads : local_params;
          memcpy(msg + sizeof(reply), vec, p_cnt * sizeof(double));
          if (num_steps > 0u && s + 1u == P) {
            memcpy(msg + sizeof(reply) + p_cnt * sizeof(double), local_losses,
                   num_steps * sizeof(double));
          }
        }
        (void)et_pp_p2p_send_frame_v1(ctrl_fd, ET_PP_MSG_STAGE_RESULT, 0u, s, s,
                                      msg, (uint64_t)payload_bytes,
                                      cfg->timeout_ms);
        free(msg);
      }

      close_fd_if_open(&ctrl_fd);
      stage_pool_destroy(&pool);
      free(local_losses);
      free(local_grads);
      free(local_params);
      _exit(reply.status == ET_PP_OK ? 0 : 1);
    }
    pids[s] = pid;
  }

  /* Parent closes all inter-stage FDs and child ends of control sockets */
  for (uint32_t s = 0; s + 1u < P; ++s) {
    close_fd_if_open(&link_fds[s][0]);
    close_fd_if_open(&link_fds[s][1]);
  }
  for (uint32_t s = 0; s < P; ++s)
    close_fd_if_open(&ctrl_fds[s][1]);

  et_pp_run_telemetry_t telem;
  memset(&telem, 0, sizeof(telem));
  telem.num_stages = P;
  telem.num_microbatches = cfg->num_microbatches;
  telem.schedule_kind = cfg->schedule_kind;
  telem.bubble_fraction =
      (double)(P - 1u) / (double)(cfg->num_microbatches + P - 1u);

  et_pp_status_t overall = ET_PP_OK;
  for (uint32_t s = 0; s < P; ++s) {
    size_t p_off = stage_param_offset(cfg, s);
    size_t p_cnt = stage_param_count(cfg, s);
    size_t extra_doubles = p_cnt + ((num_steps > 0u && s + 1u == P) ? num_steps : 0u);
    size_t payload_bytes = sizeof(stage_reply_header_t) + extra_doubles * sizeof(double);
    unsigned char *msg = (unsigned char *)malloc(payload_bytes);
    if (msg == NULL) {
      overall = ET_PP_ERR_ALLOC;
      break;
    }
    et_pp_status_t rst = et_pp_p2p_recv_frame_v1(
        ctrl_fds[s][0], ET_PP_MSG_STAGE_RESULT, 0u, s, s, msg,
        (uint64_t)payload_bytes, cfg->timeout_ms);
    if (rst != ET_PP_OK) {
      free(msg);
      if (overall == ET_PP_OK)
        overall = rst;
      break;
    }
    stage_reply_header_t reply;
    memcpy(&reply, msg, sizeof(reply));
    if (reply.status != ET_PP_OK) {
      free(msg);
      if (overall == ET_PP_OK)
        overall = reply.status;
      break;
    }

    telem.stage_pids[s] = reply.pid;
    telem.stage_peak_live[s] = reply.peak_live;
    telem.stage_final_live[s] = reply.final_live;
    telem.stage_fwd_count[s] = reply.fwd_count;
    telem.stage_bwd_count[s] = reply.bwd_count;
    telem.stage_bytes_sent[s] = reply.bytes_sent;
    telem.stage_bytes_recv[s] = reply.bytes_recv;
    if (s + 1u == P)
      telem.total_loss = reply.stage_loss;

    if (num_steps == 0u) {
      memcpy(out_grads + p_off, msg + sizeof(reply), p_cnt * sizeof(double));
    } else {
      memcpy(out_final_params + p_off, msg + sizeof(reply),
             p_cnt * sizeof(double));
      if (s + 1u == P) {
        memcpy(out_step_losses,
               msg + sizeof(reply) + p_cnt * sizeof(double),
               num_steps * sizeof(double));
      }
    }
    free(msg);
  }

  for (uint32_t s = 0; s < P; ++s)
    close_fd_if_open(&ctrl_fds[s][0]);

  if (overall != ET_PP_OK) {
    reap_all_children(pids, P, 1);
    return overall;
  }

  for (uint32_t s = 0; s < P; ++s) {
    int wstatus = 0;
    while (waitpid(pids[s], &wstatus, 0) < 0) {
      if (errno != EINTR) {
        overall = ET_PP_ERR_IO;
        break;
      }
    }
    pids[s] = -1;
    if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) {
      if (overall == ET_PP_OK)
        overall = ET_PP_ERR_IO;
    }
  }
  if (overall != ET_PP_OK)
    return overall;

  telem.distinct_pid_count = count_distinct_positive_pids(telem.stage_pids, P);
  if (out_telemetry != NULL)
    *out_telemetry = telem;
  return ET_PP_OK;

fail_io:
  for (uint32_t s = 0; s < ET_PP_MAX_STAGES; ++s) {
    close_fd_if_open(&link_fds[s][0]);
    close_fd_if_open(&link_fds[s][1]);
    close_fd_if_open(&ctrl_fds[s][0]);
    close_fd_if_open(&ctrl_fds[s][1]);
  }
  reap_all_children(pids, P, 1);
  return ET_PP_ERR_IO;
}

et_pp_status_t et_pp_run_multiprocess_step_v1(
    const et_pp_config_t *cfg, const double *inputs, const int32_t *targets_i32,
    const double *targets_f64, const double *params, double *out_grads,
    et_pp_run_telemetry_t *out_telemetry) {
  if (out_grads == NULL || out_telemetry == NULL)
    return ET_PP_ERR_INVALID_ARG;
  et_pp_status_t st = validate_config_internal(cfg, 0);
  if (st != ET_PP_OK)
    return st;
  st = validate_inputs_and_targets(cfg, inputs, targets_i32, targets_f64,
                                   params);
  if (st != ET_PP_OK)
    return st;
  return run_multiprocess_driver(cfg, 0u, 0.0, inputs, targets_i32, targets_f64,
                                 params, out_grads, NULL, NULL, out_telemetry);
}

et_pp_status_t et_pp_train_multiprocess_sgd_v1(
    const et_pp_config_t *cfg, uint32_t num_steps, double lr,
    const double *inputs, const int32_t *targets_i32,
    const double *targets_f64, const double *init_params,
    double *out_final_params, double *out_step_losses,
    et_pp_run_telemetry_t *out_last_telemetry) {
  if (num_steps == 0u || num_steps > ET_PP_MAX_TRAIN_STEPS || !isfinite(lr) ||
      lr <= 0.0 || out_final_params == NULL || out_step_losses == NULL)
    return ET_PP_ERR_INVALID_ARG;
  et_pp_status_t st = validate_config_internal(cfg, 0);
  if (st != ET_PP_OK)
    return st;
  st = validate_inputs_and_targets(cfg, inputs, targets_i32, targets_f64,
                                   init_params);
  if (st != ET_PP_OK)
    return st;
  return run_multiprocess_driver(cfg, num_steps, lr, inputs, targets_i32,
                                 targets_f64, init_params, NULL,
                                 out_final_params, out_step_losses,
                                 out_last_telemetry);
}
