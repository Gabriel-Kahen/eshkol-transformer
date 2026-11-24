#define _GNU_SOURCE
#include "tensor_parallel_eshkol_bridge.h"

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

enum {
  TP_CMD_FORWARD = 1,
  TP_CMD_BACKWARD = 2,
  TP_CMD_SGD_STEP = 3,
  TP_CMD_GATHER_WEIGHTS = 4,
  TP_CMD_GATHER_GRADS = 5,
  TP_CMD_RESET_STATS = 6,
  TP_CMD_QUIT = 7
};

typedef struct tp_ctrl_msg {
  int32_t cmd;
  int32_t pad;
  uint64_t tag;
  double scalar;
} tp_ctrl_msg_t;

typedef struct tp_block_storage {
  float *gamma1;
  float *w_q;
  float *w_k;
  float *w_v;
  float *w_o;
  float *gamma2;
  float *w_gate;
  float *w_up;
  float *w_down;
} tp_block_storage_t;

typedef struct tp_cache_storage {
  float *x;
  float *norm1;
  float *q_r;
  float *k_r;
  float *v_r;
  float *attn_r;
  float *h1;
  float *norm2;
  float *gate_r;
  float *up_r;
} tp_cache_storage_t;

struct et_tp_aot_session {
  et_tp_block_config_t cfg;
  uint32_t world_size;
  int32_t timeout_ms;
  uint64_t tag_seq;
  et_tp_group_t *group;
  et_tp_block_memory_stats_t mem_stats;

  pid_t worker_pids[ET_TP_MAX_WORLD_SIZE];
  int ctrl_fds[ET_TP_MAX_WORLD_SIZE];

  float *x;
  float *target;
  float *y;
  float *dy;
  float *dx;

  tp_block_storage_t shard_w;
  tp_block_storage_t shard_g;
  tp_cache_storage_t cache;

  tp_block_storage_t gathered_w;
  tp_block_storage_t gathered_g;
};

static int ctrl_write_all(int fd, const void *buf, size_t n) {
  const unsigned char *p = (const unsigned char *)buf;
  size_t done = 0u;
  while (done < n) {
    const ssize_t rc = send(fd, p + done, n - done, MSG_NOSIGNAL);
    if (rc > 0) {
      done += (size_t)rc;
      continue;
    }
    if (rc < 0 && errno == EINTR)
      continue;
    return -1;
  }
  return 0;
}

static int ctrl_read_all(int fd, void *buf, size_t n) {
  unsigned char *p = (unsigned char *)buf;
  size_t done = 0u;
  while (done < n) {
    const ssize_t rc = recv(fd, p + done, n - done, 0);
    if (rc > 0) {
      done += (size_t)rc;
      continue;
    }
    if (rc < 0 && errno == EINTR)
      continue;
    return -1;
  }
  return 0;
}

static size_t param_elems_for_world(const et_tp_block_config_t *cfg,
                                    uint32_t world_size, int64_t param_id) {
  const size_t D = cfg->d_model;
  const size_t F_loc = (size_t)(cfg->d_ff / world_size);
  const size_t q_loc =
      ((size_t)cfg->num_q_heads * (size_t)cfg->head_dim) / world_size;
  const size_t kv_loc =
      ((size_t)cfg->num_kv_heads * (size_t)cfg->head_dim) / world_size;
  switch (param_id) {
  case ET_TP_AOT_PARAM_GAMMA1:
    return D;
  case ET_TP_AOT_PARAM_WQ:
    return q_loc * D;
  case ET_TP_AOT_PARAM_WK:
    return kv_loc * D;
  case ET_TP_AOT_PARAM_WV:
    return kv_loc * D;
  case ET_TP_AOT_PARAM_WO:
    return D * q_loc;
  case ET_TP_AOT_PARAM_GAMMA2:
    return D;
  case ET_TP_AOT_PARAM_WGATE:
    return F_loc * D;
  case ET_TP_AOT_PARAM_WUP:
    return F_loc * D;
  case ET_TP_AOT_PARAM_WDOWN:
    return D * F_loc;
  default:
    return 0u;
  }
}

static float *storage_param_ptr(const tp_block_storage_t *st,
                                int64_t param_id) {
  switch (param_id) {
  case ET_TP_AOT_PARAM_GAMMA1:
    return st->gamma1;
  case ET_TP_AOT_PARAM_WQ:
    return st->w_q;
  case ET_TP_AOT_PARAM_WK:
    return st->w_k;
  case ET_TP_AOT_PARAM_WV:
    return st->w_v;
  case ET_TP_AOT_PARAM_WO:
    return st->w_o;
  case ET_TP_AOT_PARAM_GAMMA2:
    return st->gamma2;
  case ET_TP_AOT_PARAM_WGATE:
    return st->w_gate;
  case ET_TP_AOT_PARAM_WUP:
    return st->w_up;
  case ET_TP_AOT_PARAM_WDOWN:
    return st->w_down;
  default:
    return NULL;
  }
}

static et_tp_block_weights_t storage_as_weights(const tp_block_storage_t *st) {
  et_tp_block_weights_t w = {st->gamma1, st->w_q,    st->w_k,
                             st->w_v,    st->w_o,    st->gamma2,
                             st->w_gate, st->w_up,   st->w_down};
  return w;
}

static et_tp_block_grads_t storage_as_grads(const tp_block_storage_t *st) {
  et_tp_block_grads_t g = {st->gamma1, st->w_q,    st->w_k,
                           st->w_v,    st->w_o,    st->gamma2,
                           st->w_gate, st->w_up,   st->w_down};
  return g;
}

static et_tp_block_cache_t storage_as_cache(const tp_cache_storage_t *c) {
  et_tp_block_cache_t out = {c->x,     c->norm1, c->q_r,    c->k_r,    c->v_r,
                             c->attn_r, c->h1,   c->norm2,  c->gate_r, c->up_r};
  return out;
}

static void free_block_storage(tp_block_storage_t *st) {
  free(st->gamma1);
  free(st->w_q);
  free(st->w_k);
  free(st->w_v);
  free(st->w_o);
  free(st->gamma2);
  free(st->w_gate);
  free(st->w_up);
  free(st->w_down);
  memset(st, 0, sizeof(*st));
}

static int alloc_block_storage(const et_tp_block_config_t *cfg,
                               uint32_t world_size, tp_block_storage_t *st) {
  memset(st, 0, sizeof(*st));
  st->gamma1 =
      (float *)calloc(param_elems_for_world(cfg, world_size, 0), sizeof(float));
  st->w_q =
      (float *)calloc(param_elems_for_world(cfg, world_size, 1), sizeof(float));
  st->w_k =
      (float *)calloc(param_elems_for_world(cfg, world_size, 2), sizeof(float));
  st->w_v =
      (float *)calloc(param_elems_for_world(cfg, world_size, 3), sizeof(float));
  st->w_o =
      (float *)calloc(param_elems_for_world(cfg, world_size, 4), sizeof(float));
  st->gamma2 =
      (float *)calloc(param_elems_for_world(cfg, world_size, 5), sizeof(float));
  st->w_gate =
      (float *)calloc(param_elems_for_world(cfg, world_size, 6), sizeof(float));
  st->w_up =
      (float *)calloc(param_elems_for_world(cfg, world_size, 7), sizeof(float));
  st->w_down =
      (float *)calloc(param_elems_for_world(cfg, world_size, 8), sizeof(float));
  if (st->gamma1 == NULL || st->w_q == NULL || st->w_k == NULL ||
      st->w_v == NULL || st->w_o == NULL || st->gamma2 == NULL ||
      st->w_gate == NULL || st->w_up == NULL || st->w_down == NULL) {
    free_block_storage(st);
    return -1;
  }
  return 0;
}

static void free_cache_storage(tp_cache_storage_t *c) {
  free(c->x);
  free(c->norm1);
  free(c->q_r);
  free(c->k_r);
  free(c->v_r);
  free(c->attn_r);
  free(c->h1);
  free(c->norm2);
  free(c->gate_r);
  free(c->up_r);
  memset(c, 0, sizeof(*c));
}

static int alloc_cache_storage(const et_tp_block_config_t *cfg,
                               uint32_t world_size, tp_cache_storage_t *c) {
  memset(c, 0, sizeof(*c));
  const size_t M = (size_t)cfg->batch * (size_t)cfg->seq_len;
  const size_t D = cfg->d_model;
  const size_t F_loc = (size_t)(cfg->d_ff / world_size);
  const size_t q_loc =
      ((size_t)cfg->num_q_heads * (size_t)cfg->head_dim) / world_size;
  const size_t kv_loc =
      ((size_t)cfg->num_kv_heads * (size_t)cfg->head_dim) / world_size;

  c->x = (float *)calloc(M * D, sizeof(float));
  c->norm1 = (float *)calloc(M * D, sizeof(float));
  c->q_r = (float *)calloc(M * q_loc, sizeof(float));
  c->k_r = (float *)calloc(M * kv_loc, sizeof(float));
  c->v_r = (float *)calloc(M * kv_loc, sizeof(float));
  c->attn_r = (float *)calloc(M * q_loc, sizeof(float));
  c->h1 = (float *)calloc(M * D, sizeof(float));
  c->norm2 = (float *)calloc(M * D, sizeof(float));
  c->gate_r = (float *)calloc(M * F_loc, sizeof(float));
  c->up_r = (float *)calloc(M * F_loc, sizeof(float));
  if (c->x == NULL || c->norm1 == NULL || c->q_r == NULL || c->k_r == NULL ||
      c->v_r == NULL || c->attn_r == NULL || c->h1 == NULL ||
      c->norm2 == NULL || c->gate_r == NULL || c->up_r == NULL) {
    free_cache_storage(c);
    return -1;
  }
  return 0;
}

static float det_val(uint64_t seed, size_t idx, double scale) {
  uint64_t x = seed * UINT64_C(6364136223846793005) +
               (uint64_t)(idx + 1u) * UINT64_C(1442695040888963407);
  x ^= x >> 33;
  x *= UINT64_C(0xff51afd7ed558ccd);
  x ^= x >> 33;
  const int32_t bucket = (int32_t)(x % 2001u) - 1000;
  return (float)(((double)bucket / 1000.0) * scale);
}

static void init_deterministic_problem(const et_tp_block_config_t *cfg,
                                       tp_block_storage_t *full_w, float *x,
                                       float *target) {
  const size_t xd_cnt =
      (size_t)cfg->batch * (size_t)cfg->seq_len * (size_t)cfg->d_model;
  for (size_t i = 0u; i < xd_cnt; ++i) {
    x[i] = det_val(101u, i, 0.45);
    target[i] = det_val(202u, i, 0.55);
  }
  for (int64_t p = 0; p < ET_TP_AOT_PARAM_COUNT; ++p) {
    float *buf = storage_param_ptr(full_w, p);
    const size_t n = param_elems_for_world(cfg, 1u, p);
    for (size_t i = 0u; i < n; ++i) {
      if (p == ET_TP_AOT_PARAM_GAMMA1 || p == ET_TP_AOT_PARAM_GAMMA2) {
        buf[i] = 1.0f + det_val((uint64_t)(300 + p), i, 0.12);
      } else {
        buf[i] = det_val((uint64_t)(300 + p), i, 0.18);
      }
    }
  }
}

static void apply_local_sgd(const et_tp_block_config_t *cfg,
                            uint32_t world_size, tp_block_storage_t *w,
                            const tp_block_storage_t *g, double lr) {
  for (int64_t p = 0; p < ET_TP_AOT_PARAM_COUNT; ++p) {
    float *wb = storage_param_ptr(w, p);
    const float *gb = storage_param_ptr(g, p);
    const size_t n = param_elems_for_world(cfg, world_size, p);
    for (size_t i = 0u; i < n; ++i)
      wb[i] = (float)((double)wb[i] - lr * (double)gb[i]);
  }
}

static double compute_mse_and_dy(size_t count, const float *y,
                                 const float *target, float *dy) {
  const double inv_n = 1.0 / (double)count;
  double sum_sq = 0.0;
  for (size_t i = 0u; i < count; ++i) {
    const double diff = (double)y[i] - (double)target[i];
    sum_sq += diff * diff;
    dy[i] = (float)(diff * inv_n);
  }
  return 0.5 * sum_sq * inv_n;
}

static void run_worker_process(const et_tp_block_config_t *cfg, uint32_t rank,
                               uint32_t world_size, int data_fd, int ctrl_fd,
                               int32_t timeout_ms) {
  et_tp_group_t *group = NULL;
  tp_block_storage_t full_init;
  tp_block_storage_t shard_w;
  tp_block_storage_t shard_g;
  tp_cache_storage_t cache;
  memset(&full_init, 0, sizeof(full_init));
  memset(&shard_w, 0, sizeof(shard_w));
  memset(&shard_g, 0, sizeof(shard_g));
  memset(&cache, 0, sizeof(cache));

  const size_t xd_cnt =
      (size_t)cfg->batch * (size_t)cfg->seq_len * (size_t)cfg->d_model;
  float *x = (float *)calloc(xd_cnt, sizeof(float));
  float *target = (float *)calloc(xd_cnt, sizeof(float));
  float *y = (float *)calloc(xd_cnt, sizeof(float));
  float *dy = (float *)calloc(xd_cnt, sizeof(float));
  float *dx = (float *)calloc(xd_cnt, sizeof(float));

  int rc = 1;
  if (x == NULL || target == NULL || y == NULL || dy == NULL || dx == NULL)
    goto done;
  if (et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, rank, world_size,
                         &data_fd, 1u, timeout_ms, &group) != ET_TP_OK)
    goto done;
  close(data_fd);
  data_fd = -1;

  if (alloc_block_storage(cfg, 1u, &full_init) != 0 ||
      alloc_block_storage(cfg, world_size, &shard_w) != 0 ||
      alloc_block_storage(cfg, world_size, &shard_g) != 0 ||
      alloc_cache_storage(cfg, world_size, &cache) != 0)
    goto done;

  init_deterministic_problem(cfg, &full_init, x, target);
  {
    const et_tp_block_weights_t fw = storage_as_weights(&full_init);
    et_tp_block_weights_t rw = storage_as_weights(&shard_w);
    if (et_tp_block_shard_weights_v1(cfg, rank, world_size, &fw, &rw) !=
        ET_TP_OK)
      goto done;
  }
  /* Release full weights immediately so worker holds only its 1/W shard. */
  free_block_storage(&full_init);

  rc = 0;
  for (;;) {
    tp_ctrl_msg_t msg;
    if (ctrl_read_all(ctrl_fd, &msg, sizeof(msg)) != 0) {
      rc = 2;
      break;
    }
    if (msg.cmd == TP_CMD_QUIT) {
      rc = 0;
      break;
    }
    int32_t st = ET_TP_OK;
    if (msg.cmd == TP_CMD_FORWARD) {
      const et_tp_block_weights_t rw = storage_as_weights(&shard_w);
      et_tp_block_cache_t c = storage_as_cache(&cache);
      st = et_tp_block_forward_v1(group, cfg, &rw, x, y, &c, msg.tag,
                                  timeout_ms);
    } else if (msg.cmd == TP_CMD_BACKWARD) {
      (void)compute_mse_and_dy(xd_cnt, y, target, dy);
      const et_tp_block_weights_t rw = storage_as_weights(&shard_w);
      const et_tp_block_cache_t c = storage_as_cache(&cache);
      et_tp_block_grads_t rg = storage_as_grads(&shard_g);
      st = et_tp_block_backward_v1(group, cfg, &rw, &c, dy, dx, &rg, msg.tag,
                                   timeout_ms);
    } else if (msg.cmd == TP_CMD_SGD_STEP) {
      apply_local_sgd(cfg, world_size, &shard_w, &shard_g, msg.scalar);
      st = ET_TP_OK;
    } else if (msg.cmd == TP_CMD_GATHER_WEIGHTS ||
               msg.cmd == TP_CMD_GATHER_GRADS) {
      tp_block_storage_t tmp_full;
      if (alloc_block_storage(cfg, 1u, &tmp_full) != 0) {
        st = ET_TP_ERR_ALLOC;
      } else {
        const et_tp_block_weights_t rw = storage_as_weights(
            msg.cmd == TP_CMD_GATHER_WEIGHTS ? &shard_w : &shard_g);
        et_tp_block_weights_t fw = storage_as_weights(&tmp_full);
        st = et_tp_block_gather_weights_v1(group, cfg, &rw, &fw, msg.tag,
                                           timeout_ms);
        free_block_storage(&tmp_full);
      }
    } else if (msg.cmd == TP_CMD_RESET_STATS) {
      st = et_tp_group_reset_stats(group);
    } else {
      st = ET_TP_ERR_INVALID_ARG;
    }
    if (ctrl_write_all(ctrl_fd, &st, sizeof(st)) != 0) {
      rc = 3;
      break;
    }
  }

done:
  if (data_fd >= 0)
    close(data_fd);
  if (ctrl_fd >= 0)
    close(ctrl_fd);
  et_tp_group_destroy(group);
  free_block_storage(&full_init);
  free_block_storage(&shard_w);
  free_block_storage(&shard_g);
  free_cache_storage(&cache);
  free(x);
  free(target);
  free(y);
  free(dy);
  free(dx);
  _exit(rc);
}

static int32_t dispatch_workers(et_tp_aot_session_t *s, int32_t cmd,
                                uint64_t tag, double scalar) {
  if (s->world_size == 1u)
    return ET_TP_OK;
  tp_ctrl_msg_t msg = {cmd, 0, tag, scalar};
  for (uint32_t r = 1u; r < s->world_size; ++r) {
    if (ctrl_write_all(s->ctrl_fds[r], &msg, sizeof(msg)) != 0)
      return ET_TP_ERR_PEER_DISCONNECTED;
  }
  return ET_TP_OK;
}

static int32_t collect_worker_acks(et_tp_aot_session_t *s) {
  if (s->world_size == 1u)
    return ET_TP_OK;
  int32_t first_err = ET_TP_OK;
  for (uint32_t r = 1u; r < s->world_size; ++r) {
    int32_t wst = ET_TP_OK;
    if (ctrl_read_all(s->ctrl_fds[r], &wst, sizeof(wst)) != 0) {
      if (first_err == ET_TP_OK)
        first_err = ET_TP_ERR_PEER_DISCONNECTED;
    } else if (wst != ET_TP_OK && first_err == ET_TP_OK) {
      first_err = wst;
    }
  }
  return first_err;
}

int32_t et_tp_aot_session_create(int64_t world_size, int64_t batch,
                                 int64_t seq_len, int64_t d_model, int64_t d_ff,
                                 int64_t num_q_heads, int64_t num_kv_heads,
                                 int64_t head_dim, double rms_eps,
                                 et_tp_aot_session_t **out_session) {
  if (out_session == NULL)
    return ET_TP_ERR_INVALID_ARG;
  *out_session = NULL;

  if (world_size <= 0 || world_size > (int64_t)ET_TP_MAX_WORLD_SIZE ||
      batch <= 0 || seq_len <= 0 || d_model <= 0 || d_ff <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      !isfinite(rms_eps) || !(rms_eps > 0.0))
    return ET_TP_ERR_INVALID_ARG;

  et_tp_block_config_t cfg = {(uint32_t)batch,        (uint32_t)seq_len,
                              (uint32_t)d_model,      (uint32_t)d_ff,
                              (uint32_t)num_q_heads,  (uint32_t)num_kv_heads,
                              (uint32_t)head_dim,     (float)rms_eps};
  et_tp_status_t st = et_tp_block_validate_config(&cfg, (uint32_t)world_size);
  if (st != ET_TP_OK)
    return st;

  et_tp_aot_session_t *s =
      (et_tp_aot_session_t *)calloc(1u, sizeof(*s));
  if (s == NULL)
    return ET_TP_ERR_ALLOC;
  s->cfg = cfg;
  s->world_size = (uint32_t)world_size;
  s->timeout_ms = 10000;
  s->tag_seq = 100u;
  for (uint32_t i = 0u; i < ET_TP_MAX_WORLD_SIZE; ++i) {
    s->worker_pids[i] = -1;
    s->ctrl_fds[i] = -1;
  }

  st = et_tp_block_memory_stats_v1(&cfg, s->world_size, &s->mem_stats);
  if (st != ET_TP_OK) {
    et_tp_aot_session_destroy(s);
    return st;
  }

  const size_t xd_cnt =
      (size_t)cfg.batch * (size_t)cfg.seq_len * (size_t)cfg.d_model;
  s->x = (float *)calloc(xd_cnt, sizeof(float));
  s->target = (float *)calloc(xd_cnt, sizeof(float));
  s->y = (float *)calloc(xd_cnt, sizeof(float));
  s->dy = (float *)calloc(xd_cnt, sizeof(float));
  s->dx = (float *)calloc(xd_cnt, sizeof(float));

  tp_block_storage_t full_init;
  memset(&full_init, 0, sizeof(full_init));

  if (s->x == NULL || s->target == NULL || s->y == NULL || s->dy == NULL ||
      s->dx == NULL || alloc_block_storage(&cfg, 1u, &full_init) != 0 ||
      alloc_block_storage(&cfg, s->world_size, &s->shard_w) != 0 ||
      alloc_block_storage(&cfg, s->world_size, &s->shard_g) != 0 ||
      alloc_block_storage(&cfg, 1u, &s->gathered_w) != 0 ||
      alloc_block_storage(&cfg, 1u, &s->gathered_g) != 0 ||
      alloc_cache_storage(&cfg, s->world_size, &s->cache) != 0) {
    free_block_storage(&full_init);
    et_tp_aot_session_destroy(s);
    return ET_TP_ERR_ALLOC;
  }

  init_deterministic_problem(&cfg, &full_init, s->x, s->target);
  {
    const et_tp_block_weights_t fw = storage_as_weights(&full_init);
    et_tp_block_weights_t rw = storage_as_weights(&s->shard_w);
    st = et_tp_block_shard_weights_v1(&cfg, 0u, s->world_size, &fw, &rw);
  }
  free_block_storage(&full_init);
  if (st != ET_TP_OK) {
    et_tp_aot_session_destroy(s);
    return st;
  }

  if (s->world_size == 1u) {
    st = et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 1u, NULL, 0u,
                            s->timeout_ms, &s->group);
    if (st != ET_TP_OK) {
      et_tp_aot_session_destroy(s);
      return st;
    }
    *out_session = s;
    return ET_TP_OK;
  }

  /* Create AF_UNIX SOCK_STREAM pairs and fork (W - 1) real OS processes. */
  int data_pairs[ET_TP_MAX_WORLD_SIZE][2];
  int ctrl_pairs[ET_TP_MAX_WORLD_SIZE][2];
  for (uint32_t r = 0u; r < ET_TP_MAX_WORLD_SIZE; ++r) {
    data_pairs[r][0] = data_pairs[r][1] = -1;
    ctrl_pairs[r][0] = ctrl_pairs[r][1] = -1;
  }
  for (uint32_t r = 1u; r < s->world_size; ++r) {
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, data_pairs[r]) !=
            0 ||
        socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, ctrl_pairs[r]) !=
            0) {
      for (uint32_t k = 1u; k <= r; ++k) {
        if (data_pairs[k][0] >= 0)
          close(data_pairs[k][0]);
        if (data_pairs[k][1] >= 0)
          close(data_pairs[k][1]);
        if (ctrl_pairs[k][0] >= 0)
          close(ctrl_pairs[k][0]);
        if (ctrl_pairs[k][1] >= 0)
          close(ctrl_pairs[k][1]);
      }
      et_tp_aot_session_destroy(s);
      return ET_TP_ERR_IO;
    }
  }

  (void)fflush(NULL);
  for (uint32_t r = 1u; r < s->world_size; ++r) {
    const pid_t pid = fork();
    if (pid < 0) {
      for (uint32_t k = 1u; k < s->world_size; ++k) {
        if (data_pairs[k][0] >= 0)
          close(data_pairs[k][0]);
        if (data_pairs[k][1] >= 0)
          close(data_pairs[k][1]);
        if (ctrl_pairs[k][0] >= 0)
          close(ctrl_pairs[k][0]);
        if (ctrl_pairs[k][1] >= 0)
          close(ctrl_pairs[k][1]);
      }
      et_tp_aot_session_destroy(s);
      return ET_TP_ERR_IO;
    }
    if (pid == 0) {
      /* Child process: keep only rank r's endpoints, close all others. */
      const int my_data = data_pairs[r][1];
      const int my_ctrl = ctrl_pairs[r][1];
      for (uint32_t k = 1u; k < s->world_size; ++k) {
        if (data_pairs[k][0] >= 0)
          close(data_pairs[k][0]);
        if (k != r && data_pairs[k][1] >= 0)
          close(data_pairs[k][1]);
        if (ctrl_pairs[k][0] >= 0)
          close(ctrl_pairs[k][0]);
        if (k != r && ctrl_pairs[k][1] >= 0)
          close(ctrl_pairs[k][1]);
      }
      /* Free inherited parent heap copies before entering worker loop. */
      free(s->x);
      free(s->target);
      free(s->y);
      free(s->dy);
      free(s->dx);
      free_block_storage(&s->shard_w);
      free_block_storage(&s->shard_g);
      free_block_storage(&s->gathered_w);
      free_block_storage(&s->gathered_g);
      free_cache_storage(&s->cache);
      free(s);
      run_worker_process(&cfg, r, (uint32_t)world_size, my_data, my_ctrl,
                         10000);
    }
    s->worker_pids[r] = pid;
    s->ctrl_fds[r] = ctrl_pairs[r][0];
    close(ctrl_pairs[r][1]);
    ctrl_pairs[r][1] = -1;
    close(data_pairs[r][1]);
    data_pairs[r][1] = -1;
  }

  int rank0_fds[ET_TP_MAX_WORLD_SIZE];
  for (uint32_t r = 1u; r < s->world_size; ++r)
    rank0_fds[r - 1u] = data_pairs[r][0];

  st = et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, s->world_size,
                          rank0_fds, (size_t)(s->world_size - 1u),
                          s->timeout_ms, &s->group);
  for (uint32_t r = 1u; r < s->world_size; ++r)
    close(data_pairs[r][0]);

  if (st != ET_TP_OK) {
    et_tp_aot_session_destroy(s);
    return st;
  }

  *out_session = s;
  return ET_TP_OK;
}

void et_tp_aot_session_destroy(et_tp_aot_session_t *s) {
  if (s == NULL)
    return;
  if (s->world_size > 1u) {
    tp_ctrl_msg_t quit_msg = {TP_CMD_QUIT, 0, 0u, 0.0};
    for (uint32_t r = 1u; r < s->world_size; ++r) {
      if (s->ctrl_fds[r] >= 0) {
        (void)ctrl_write_all(s->ctrl_fds[r], &quit_msg, sizeof(quit_msg));
        close(s->ctrl_fds[r]);
        s->ctrl_fds[r] = -1;
      }
    }
    for (uint32_t r = 1u; r < s->world_size; ++r) {
      if (s->worker_pids[r] > 0) {
        int status = 0;
        while (waitpid(s->worker_pids[r], &status, 0) < 0 && errno == EINTR) {
        }
        s->worker_pids[r] = -1;
      }
    }
  }
  et_tp_group_destroy(s->group);
  free(s->x);
  free(s->target);
  free(s->y);
  free(s->dy);
  free(s->dx);
  free_block_storage(&s->shard_w);
  free_block_storage(&s->shard_g);
  free_block_storage(&s->gathered_w);
  free_block_storage(&s->gathered_g);
  free_cache_storage(&s->cache);
  free(s);
}

int32_t et_tp_aot_forward(et_tp_aot_session_t *s) {
  if (s == NULL)
    return ET_TP_ERR_INVALID_ARG;
  const uint64_t tag = s->tag_seq;
  s->tag_seq += 8u;
  int32_t st = dispatch_workers(s, TP_CMD_FORWARD, tag, 0.0);
  if (st != ET_TP_OK)
    return st;
  const et_tp_block_weights_t rw = storage_as_weights(&s->shard_w);
  et_tp_block_cache_t c = storage_as_cache(&s->cache);
  st = et_tp_block_forward_v1(s->group, &s->cfg, &rw, s->x, s->y, &c, tag,
                              s->timeout_ms);
  const int32_t wst = collect_worker_acks(s);
  return (st != ET_TP_OK) ? st : wst;
}

double et_tp_aot_mse_loss(et_tp_aot_session_t *s) {
  if (s == NULL)
    return NAN;
  const size_t xd_cnt =
      (size_t)s->cfg.batch * (size_t)s->cfg.seq_len * (size_t)s->cfg.d_model;
  return compute_mse_and_dy(xd_cnt, s->y, s->target, s->dy);
}

int32_t et_tp_aot_backward(et_tp_aot_session_t *s) {
  if (s == NULL)
    return ET_TP_ERR_INVALID_ARG;
  const uint64_t tag = s->tag_seq;
  s->tag_seq += 8u;
  int32_t st = dispatch_workers(s, TP_CMD_BACKWARD, tag, 0.0);
  if (st != ET_TP_OK)
    return st;
  const et_tp_block_weights_t rw = storage_as_weights(&s->shard_w);
  const et_tp_block_cache_t c = storage_as_cache(&s->cache);
  et_tp_block_grads_t rg = storage_as_grads(&s->shard_g);
  st = et_tp_block_backward_v1(s->group, &s->cfg, &rw, &c, s->dy, s->dx, &rg,
                               tag, s->timeout_ms);
  const int32_t wst = collect_worker_acks(s);
  return (st != ET_TP_OK) ? st : wst;
}

int32_t et_tp_aot_sgd_step(et_tp_aot_session_t *s, double lr) {
  if (s == NULL || !isfinite(lr) || !(lr > 0.0))
    return ET_TP_ERR_INVALID_ARG;
  int32_t st = dispatch_workers(s, TP_CMD_SGD_STEP, 0u, lr);
  if (st != ET_TP_OK)
    return st;
  apply_local_sgd(&s->cfg, s->world_size, &s->shard_w, &s->shard_g, lr);
  return collect_worker_acks(s);
}

int32_t et_tp_aot_gather_weights(et_tp_aot_session_t *s) {
  if (s == NULL)
    return ET_TP_ERR_INVALID_ARG;
  const uint64_t tag = s->tag_seq;
  s->tag_seq += 16u;
  int32_t st = dispatch_workers(s, TP_CMD_GATHER_WEIGHTS, tag, 0.0);
  if (st != ET_TP_OK)
    return st;
  const et_tp_block_weights_t rw = storage_as_weights(&s->shard_w);
  et_tp_block_weights_t fw = storage_as_weights(&s->gathered_w);
  st = et_tp_block_gather_weights_v1(s->group, &s->cfg, &rw, &fw, tag,
                                     s->timeout_ms);
  const int32_t wst = collect_worker_acks(s);
  return (st != ET_TP_OK) ? st : wst;
}

int32_t et_tp_aot_gather_grads(et_tp_aot_session_t *s) {
  if (s == NULL)
    return ET_TP_ERR_INVALID_ARG;
  const uint64_t tag = s->tag_seq;
  s->tag_seq += 16u;
  int32_t st = dispatch_workers(s, TP_CMD_GATHER_GRADS, tag, 0.0);
  if (st != ET_TP_OK)
    return st;
  const et_tp_block_weights_t rg = storage_as_weights(&s->shard_g);
  et_tp_block_weights_t fg = storage_as_weights(&s->gathered_g);
  st = et_tp_block_gather_weights_v1(s->group, &s->cfg, &rg, &fg, tag,
                                     s->timeout_ms);
  const int32_t wst = collect_worker_acks(s);
  return (st != ET_TP_OK) ? st : wst;
}

int32_t et_tp_aot_reset_stats(et_tp_aot_session_t *s) {
  if (s == NULL)
    return ET_TP_ERR_INVALID_ARG;
  int32_t st = dispatch_workers(s, TP_CMD_RESET_STATS, 0u, 0.0);
  if (st != ET_TP_OK)
    return st;
  st = et_tp_group_reset_stats(s->group);
  const int32_t wst = collect_worker_acks(s);
  return (st != ET_TP_OK) ? st : wst;
}

int64_t et_tp_aot_allreduce_calls(const et_tp_aot_session_t *s) {
  et_tp_collective_stats_t st;
  if (s == NULL || et_tp_group_get_stats(s->group, &st) != ET_TP_OK)
    return -1;
  return (int64_t)st.allreduce_calls;
}

int64_t et_tp_aot_ipc_allreduce_calls(const et_tp_aot_session_t *s) {
  et_tp_collective_stats_t st;
  if (s == NULL || et_tp_group_get_stats(s->group, &st) != ET_TP_OK)
    return -1;
  return (int64_t)st.ipc_allreduce_calls;
}

int64_t et_tp_aot_allgather_calls(const et_tp_aot_session_t *s) {
  et_tp_collective_stats_t st;
  if (s == NULL || et_tp_group_get_stats(s->group, &st) != ET_TP_OK)
    return -1;
  return (int64_t)st.allgather_calls;
}

int64_t et_tp_aot_ipc_allgather_calls(const et_tp_aot_session_t *s) {
  et_tp_collective_stats_t st;
  if (s == NULL || et_tp_group_get_stats(s->group, &st) != ET_TP_OK)
    return -1;
  return (int64_t)st.ipc_allgather_calls;
}

int64_t et_tp_aot_full_sharded_weight_bytes(const et_tp_aot_session_t *s) {
  return (s == NULL) ? -1 : (int64_t)s->mem_stats.full_sharded_weight_bytes;
}

int64_t et_tp_aot_rank_sharded_weight_bytes(const et_tp_aot_session_t *s) {
  return (s == NULL) ? -1 : (int64_t)s->mem_stats.rank_sharded_weight_bytes;
}

int64_t et_tp_aot_replicated_weight_bytes(const et_tp_aot_session_t *s) {
  return (s == NULL) ? -1 : (int64_t)s->mem_stats.replicated_weight_bytes;
}

int64_t et_tp_aot_output_count(const et_tp_aot_session_t *s) {
  if (s == NULL)
    return -1;
  return (int64_t)((size_t)s->cfg.batch * (size_t)s->cfg.seq_len *
                   (size_t)s->cfg.d_model);
}

int64_t et_tp_aot_param_count(const et_tp_aot_session_t *s, int64_t param_id) {
  if (s == NULL || param_id < 0 || param_id >= ET_TP_AOT_PARAM_COUNT)
    return -1;
  return (int64_t)param_elems_for_world(&s->cfg, 1u, param_id);
}

double et_tp_aot_get_input(const et_tp_aot_session_t *s, int64_t idx) {
  if (s == NULL || idx < 0 || idx >= et_tp_aot_output_count(s))
    return NAN;
  return (double)s->x[idx];
}

int32_t et_tp_aot_set_input(et_tp_aot_session_t *s, int64_t idx, double value) {
  if (s == NULL || s->world_size != 1u || idx < 0 ||
      idx >= et_tp_aot_output_count(s))
    return ET_TP_ERR_INVALID_ARG;
  s->x[idx] = (float)value;
  return ET_TP_OK;
}

double et_tp_aot_get_output(const et_tp_aot_session_t *s, int64_t idx) {
  if (s == NULL || idx < 0 || idx >= et_tp_aot_output_count(s))
    return NAN;
  return (double)s->y[idx];
}

double et_tp_aot_get_dx(const et_tp_aot_session_t *s, int64_t idx) {
  if (s == NULL || idx < 0 || idx >= et_tp_aot_output_count(s))
    return NAN;
  return (double)s->dx[idx];
}

double et_tp_aot_get_weight(const et_tp_aot_session_t *s, int64_t param_id,
                            int64_t idx) {
  if (s == NULL || s->world_size != 1u || param_id < 0 ||
      param_id >= ET_TP_AOT_PARAM_COUNT)
    return NAN;
  const size_t n = param_elems_for_world(&s->cfg, 1u, param_id);
  if (idx < 0 || (size_t)idx >= n)
    return NAN;
  return (double)storage_param_ptr(&s->shard_w, param_id)[idx];
}

int32_t et_tp_aot_set_weight(et_tp_aot_session_t *s, int64_t param_id,
                             int64_t idx, double value) {
  if (s == NULL || s->world_size != 1u || param_id < 0 ||
      param_id >= ET_TP_AOT_PARAM_COUNT)
    return ET_TP_ERR_INVALID_ARG;
  const size_t n = param_elems_for_world(&s->cfg, 1u, param_id);
  if (idx < 0 || (size_t)idx >= n)
    return ET_TP_ERR_INVALID_ARG;
  storage_param_ptr(&s->shard_w, param_id)[idx] = (float)value;
  return ET_TP_OK;
}

double et_tp_aot_get_gathered_weight(const et_tp_aot_session_t *s,
                                     int64_t param_id, int64_t idx) {
  if (s == NULL || param_id < 0 || param_id >= ET_TP_AOT_PARAM_COUNT)
    return NAN;
  const size_t n = param_elems_for_world(&s->cfg, 1u, param_id);
  if (idx < 0 || (size_t)idx >= n)
    return NAN;
  return (double)storage_param_ptr(&s->gathered_w, param_id)[idx];
}

double et_tp_aot_get_gathered_grad(const et_tp_aot_session_t *s,
                                   int64_t param_id, int64_t idx) {
  if (s == NULL || param_id < 0 || param_id >= ET_TP_AOT_PARAM_COUNT)
    return NAN;
  const size_t n = param_elems_for_world(&s->cfg, 1u, param_id);
  if (idx < 0 || (size_t)idx >= n)
    return NAN;
  return (double)storage_param_ptr(&s->gathered_g, param_id)[idx];
}

static double max_abs_diff_vec(const float *a, const float *b, size_t n) {
  double m = 0.0;
  for (size_t i = 0u; i < n; ++i) {
    const double d = fabs((double)a[i] - (double)b[i]);
    if (d > m)
      m = d;
  }
  return m;
}

double et_tp_aot_max_output_diff(const et_tp_aot_session_t *a,
                                 const et_tp_aot_session_t *b) {
  if (a == NULL || b == NULL)
    return NAN;
  const size_t n = (size_t)et_tp_aot_output_count(a);
  if (n != (size_t)et_tp_aot_output_count(b))
    return NAN;
  return max_abs_diff_vec(a->y, b->y, n);
}

double et_tp_aot_max_dx_diff(const et_tp_aot_session_t *a,
                             const et_tp_aot_session_t *b) {
  if (a == NULL || b == NULL)
    return NAN;
  const size_t n = (size_t)et_tp_aot_output_count(a);
  if (n != (size_t)et_tp_aot_output_count(b))
    return NAN;
  return max_abs_diff_vec(a->dx, b->dx, n);
}

static double max_storage_diff(const et_tp_block_config_t *cfg,
                               const tp_block_storage_t *a,
                               const tp_block_storage_t *b) {
  double m = 0.0;
  for (int64_t p = 0; p < ET_TP_AOT_PARAM_COUNT; ++p) {
    const size_t n = param_elems_for_world(cfg, 1u, p);
    const double d =
        max_abs_diff_vec(storage_param_ptr(a, p), storage_param_ptr(b, p), n);
    if (d > m)
      m = d;
  }
  return m;
}

double et_tp_aot_max_grad_diff(const et_tp_aot_session_t *a,
                               const et_tp_aot_session_t *b) {
  if (a == NULL || b == NULL)
    return NAN;
  return max_storage_diff(&a->cfg, &a->gathered_g, &b->gathered_g);
}

double et_tp_aot_max_weight_diff(const et_tp_aot_session_t *a,
                                 const et_tp_aot_session_t *b) {
  if (a == NULL || b == NULL)
    return NAN;
  return max_storage_diff(&a->cfg, &a->gathered_w, &b->gathered_w);
}

static int count_open_fds(void) {
  DIR *dir = opendir("/proc/self/fd");
  if (dir == NULL)
    return -1;
  int count = 0;
  while (readdir(dir) != NULL)
    ++count;
  closedir(dir);
  return count;
}

int32_t et_tp_aot_check_negative_case(int64_t case_id) {
  /* Case 0: Invalid world_size = 3 -> ET_TP_ERR_INVALID_ARG */
  if (case_id == 0) {
    et_tp_group_t *g = NULL;
    return et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 3u, NULL,
                              0u, 1000, &g);
  }
  /* Case 1: ColumnParallel Out % W != 0 (Out=5, W=2) -> ET_TP_ERR_DIVISIBILITY */
  if (case_id == 1) {
    int sp[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp) != 0)
      return ET_TP_ERR_IO;
    et_tp_group_t *g = NULL;
    int32_t st = et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 2u,
                                    &sp[0], 1u, 1000, &g);
    close(sp[0]);
    close(sp[1]);
    if (st != ET_TP_OK)
      return st;
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[10] = {0.0f};
    float y[10] = {0.0f};
    st = et_tp_column_linear_forward_v1(g, 1u, 4u, 5u, x, w, NULL, y);
    et_tp_group_destroy(g);
    return st;
  }
  /* Case 2: RowParallel In % W != 0 (In=5, W=2) -> ET_TP_ERR_DIVISIBILITY */
  if (case_id == 2) {
    int sp[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp) != 0)
      return ET_TP_ERR_IO;
    et_tp_group_t *g = NULL;
    int32_t st = et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 2u,
                                    &sp[0], 1u, 1000, &g);
    close(sp[0]);
    close(sp[1]);
    if (st != ET_TP_OK)
      return st;
    float x[5] = {0.0f};
    float w[10] = {0.0f};
    float y[4] = {0.0f};
    st = et_tp_row_linear_forward_v1(g, 1u, 5u, 4u, x, w, NULL, y, 1u, 1000);
    et_tp_group_destroy(g);
    return st;
  }
  /* Case 3: SwiGLU d_ff % W != 0 (d_ff=6, W=4) -> ET_TP_ERR_DIVISIBILITY */
  if (case_id == 3) {
    const et_tp_swiglu_config_t cfg = {2u, 4u, 6u};
    return et_tp_swiglu_validate_config(&cfg, 4u);
  }
  /* Case 4: GQA H_q % W != 0 (H_q=6, H_kv=4, W=4) -> ET_TP_ERR_DIVISIBILITY */
  if (case_id == 4) {
    const et_tp_gqa_config_t cfg = {1u, 4u, 12u, 6u, 4u, 2u};
    return et_tp_gqa_validate_config(&cfg, 4u);
  }
  /* Case 5: GQA H_kv % W != 0 (H_q=8, H_kv=2, W=4) -> ET_TP_ERR_DIVISIBILITY */
  if (case_id == 5) {
    const et_tp_gqa_config_t cfg = {1u, 4u, 16u, 8u, 2u, 2u};
    return et_tp_gqa_validate_config(&cfg, 4u);
  }
  /* Case 6: GQA H_q % H_kv != 0 (H_q=6, H_kv=4, W=2) -> ET_TP_ERR_DIVISIBILITY */
  if (case_id == 6) {
    const et_tp_gqa_config_t cfg = {1u, 4u, 12u, 6u, 4u, 2u};
    return et_tp_gqa_validate_config(&cfg, 2u);
  }
  /* Case 7: Non-finite input (NaN) -> ET_TP_ERR_NONFINITE */
  if (case_id == 7) {
    et_tp_group_t *g = NULL;
    int32_t st = et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 1u,
                                    NULL, 0u, 1000, &g);
    if (st != ET_TP_OK)
      return st;
    float x[4] = {1.0f, NAN, 0.5f, -0.5f};
    float w[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float y[2] = {0.0f};
    st = et_tp_column_linear_forward_v1(g, 1u, 4u, 2u, x, w, NULL, y);
    et_tp_group_destroy(g);
    return st;
  }
  /* Case 8: Misaligned float pointer -> ET_TP_ERR_ALIGNMENT */
  if (case_id == 8) {
    et_tp_group_t *g = NULL;
    int32_t st = et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 1u,
                                    NULL, 0u, 1000, &g);
    if (st != ET_TP_OK)
      return st;
    unsigned char raw[64];
    memset(raw, 0, sizeof(raw));
    const float *misaligned = (const float *)(const void *)(raw + 1u);
    float out[4] = {0.0f};
    st = et_tp_allreduce_sum_f32(g, misaligned, out, 4u, 1u, 1000);
    et_tp_group_destroy(g);
    return st;
  }
  /* Case 9: Overlapping alias in AllGather -> ET_TP_ERR_ALIAS */
  if (case_id == 9) {
    et_tp_group_t *g = NULL;
    int32_t st = et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 1u,
                                    NULL, 0u, 1000, &g);
    if (st != ET_TP_OK)
      return st;
    float buf[8] = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    st = et_tp_allgather_f32(g, buf, 4u, buf + 2u, 1u, 1000);
    et_tp_group_destroy(g);
    return st;
  }
  /* Case 10: Monotonic timeout + zero FD leak -> ET_TP_ERR_TIMEOUT */
  if (case_id == 10) {
    const int fds_before = count_open_fds();
    int sp[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp) != 0)
      return ET_TP_ERR_IO;
    et_tp_group_t *g = NULL;
    int32_t st = et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 2u,
                                    &sp[0], 1u, 40, &g);
    close(sp[0]);
    if (st != ET_TP_OK) {
      close(sp[1]);
      return st;
    }
    float send_buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float recv_buf[4] = {9.0f, 9.0f, 9.0f, 9.0f};
    /* Peer end sp[1] stays open and silent -> triggers ET_TP_ERR_TIMEOUT */
    st = et_tp_allreduce_sum_f32(g, send_buf, recv_buf, 4u, 77u, 40);
    et_tp_group_destroy(g);
    close(sp[1]);
    const int fds_after = count_open_fds();
    if (fds_before >= 0 && fds_after != fds_before)
      return ET_TP_ERR_IO;
    if (recv_buf[0] != 9.0f)
      return ET_TP_ERR_IO;
    return st;
  }
  /* Case 11: Peer disconnect + zero FD leak -> ET_TP_ERR_PEER_DISCONNECTED */
  if (case_id == 11) {
    const int fds_before = count_open_fds();
    int sp[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp) != 0)
      return ET_TP_ERR_IO;
    et_tp_group_t *g = NULL;
    int32_t st = et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 2u,
                                    &sp[0], 1u, 1000, &g);
    close(sp[0]);
    close(sp[1]); /* peer disconnects before collective */
    if (st != ET_TP_OK)
      return st;
    float send_buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float recv_buf[4] = {8.0f, 8.0f, 8.0f, 8.0f};
    st = et_tp_allreduce_sum_f32(g, send_buf, recv_buf, 4u, 88u, 1000);
    et_tp_group_destroy(g);
    const int fds_after = count_open_fds();
    if (fds_before >= 0 && fds_after != fds_before)
      return ET_TP_ERR_IO;
    if (recv_buf[0] != 8.0f)
      return ET_TP_ERR_IO;
    return st;
  }
  return ET_TP_ERR_INVALID_ARG;
}
