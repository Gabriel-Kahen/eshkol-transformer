#define _GNU_SOURCE
#include "fsdp_zero_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct et_fsdp_aot_session {
  et_fsdp_comm_t *comm;
  et_fsdp_engine_t *engine;

  size_t in_dim;
  size_t hidden_dim;
  size_t num_classes;
  size_t global_batch;
  size_t local_batch;
  size_t num_params;

  float *x_buf;
  int32_t *y_buf;
  float *grads_buf;

  double last_loss;
  double last_grad_norm;
  double last_clip_scale;
  int64_t last_clipped;
  double last_fd_max_err;
  int64_t ephemeral_poison_verified;

  char full_digest_hex[65];
  char shard_digest_hex[65];
  char shard_mark_hex[65];
};

static void bytes_to_hex(const uint8_t in[32], char out[65]) {
  static const char hex[] = "0123456789abcdef";
  for (size_t i = 0u; i < 32u; ++i) {
    out[i * 2u] = hex[in[i] >> 4u];
    out[i * 2u + 1u] = hex[in[i] & 0x0fu];
  }
  out[64] = '\0';
}

static uint32_t f32_bits(float v) {
  uint32_t b = 0u;
  memcpy(&b, &v, sizeof(b));
  return b;
}

static float bits_f32(uint32_t b) {
  float v = 0.0f;
  memcpy(&v, &b, sizeof(v));
  return v;
}

static int verify_all_poisoned(const float *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (f32_bits(buf[i]) != ET_FSDP_POISON_F32_BITS) {
      return 0;
    }
  }
  return 1;
}

ET_FSDP_API int32_t et_fsdp_aot_open(
    const char *sock_dir, int64_t rank, int64_t world_size, int64_t stage,
    int64_t in_dim, int64_t hidden_dim, int64_t num_classes,
    int64_t global_batch, double lr, double beta1, double beta2, double eps,
    double weight_decay, double max_grad_norm, int64_t timeout_ms,
    et_fsdp_aot_session_t **out_session) {
  if (out_session == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  *out_session = NULL;

  if (rank < 0 || world_size <= 0 || world_size > (int64_t)ET_FSDP_MAX_WORLD_SIZE ||
      rank >= world_size || stage < 1 || stage > 3 || in_dim <= 0 ||
      hidden_dim <= 0 || num_classes < 2 || global_batch <= 0 ||
      (global_batch % world_size) != 0 || timeout_ms <= 0 ||
      timeout_ms > 600000) {
    return ET_FSDP_ERR_INVALID_ARG;
  }

  uint64_t num_params_u64 = 0u;
  et_fsdp_status_t st = et_fsdp_mlp_num_params(
      (size_t)in_dim, (size_t)hidden_dim, (size_t)num_classes, &num_params_u64);
  if (st != ET_FSDP_OK) {
    return st;
  }

  et_fsdp_aot_session_t *s =
      (et_fsdp_aot_session_t *)calloc(1u, sizeof(*s));
  if (s == NULL) {
    return ET_FSDP_ERR_ALLOC;
  }

  s->in_dim = (size_t)in_dim;
  s->hidden_dim = (size_t)hidden_dim;
  s->num_classes = (size_t)num_classes;
  s->global_batch = (size_t)global_batch;
  s->local_batch = (size_t)(global_batch / world_size);
  s->num_params = (size_t)num_params_u64;

  st = et_fsdp_comm_open_unix(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, sock_dir,
                              (uint32_t)rank, (uint32_t)world_size,
                              (int)timeout_ms, &s->comm);
  if (st != ET_FSDP_OK) {
    free(s);
    return st;
  }

  et_fsdp_config_v1 cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.stage = (uint32_t)stage;
  cfg.reserved = 0u;
  cfg.num_params = num_params_u64;
  cfg.lr = lr;
  cfg.beta1 = beta1;
  cfg.beta2 = beta2;
  cfg.eps = eps;
  cfg.weight_decay = weight_decay;
  cfg.max_grad_norm = max_grad_norm;

  st = et_fsdp_engine_create(s->comm, &cfg, &s->engine);
  if (st != ET_FSDP_OK) {
    et_fsdp_comm_destroy(s->comm);
    free(s);
    return st;
  }

  s->x_buf = (float *)calloc(s->local_batch * s->in_dim, sizeof(float));
  s->y_buf = (int32_t *)calloc(s->local_batch, sizeof(int32_t));
  s->grads_buf = (float *)calloc(s->num_params, sizeof(float));
  if (s->x_buf == NULL || s->y_buf == NULL || s->grads_buf == NULL) {
    et_fsdp_aot_close(s);
    return ET_FSDP_ERR_ALLOC;
  }

  *out_session = s;
  return ET_FSDP_OK;
}

ET_FSDP_API void et_fsdp_aot_close(et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return;
  }
  free(session->x_buf);
  free(session->y_buf);
  free(session->grads_buf);
  et_fsdp_engine_destroy(session->engine);
  et_fsdp_comm_destroy(session->comm);
  free(session);
}

ET_FSDP_API int32_t et_fsdp_aot_init_params(et_fsdp_aot_session_t *session,
                                            int64_t seed) {
  if (session == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  float *init_buf = (float *)malloc(session->num_params * sizeof(float));
  if (init_buf == NULL) {
    return ET_FSDP_ERR_ALLOC;
  }
  et_fsdp_status_t st = et_fsdp_mlp_init_params(
      session->in_dim, session->hidden_dim, session->num_classes,
      (uint64_t)seed, init_buf, session->num_params);
  if (st == ET_FSDP_OK) {
    st = et_fsdp_engine_load_initial_params(session->engine, init_buf,
                                            session->num_params);
  }
  (void)et_fsdp_engine_poison_ephemeral(init_buf, session->num_params);
  free(init_buf);
  return st;
}

ET_FSDP_API int32_t et_fsdp_aot_train_step(et_fsdp_aot_session_t *session,
                                           int64_t step_idx, int64_t tag,
                                           int64_t timeout_ms) {
  if (session == NULL || step_idx < 0 || timeout_ms <= 0 || timeout_ms > 600000) {
    return ET_FSDP_ERR_INVALID_ARG;
  }

  /* 1. Ephemeral full parameter materialization (AllGather for Stage 3 FSDP). */
  float *ephemeral_params = (float *)malloc(session->num_params * sizeof(float));
  if (ephemeral_params == NULL) {
    return ET_FSDP_ERR_ALLOC;
  }
  et_fsdp_status_t st = et_fsdp_engine_gather_params(
      session->engine, ephemeral_params, session->num_params, (uint64_t)tag,
      (int)timeout_ms);
  if (st != ET_FSDP_OK) {
    free(ephemeral_params);
    return st;
  }

  /* 2. Rank-local microbatch forward + backward. */
  st = et_fsdp_mlp_make_batch(
      session->in_dim, session->num_classes, session->global_batch,
      et_fsdp_comm_rank(session->comm), et_fsdp_comm_world_size(session->comm),
      (uint64_t)step_idx, session->x_buf, session->y_buf);
  double local_loss = 0.0;
  if (st == ET_FSDP_OK) {
    st = et_fsdp_mlp_forward_backward(
        session->in_dim, session->hidden_dim, session->num_classes,
        session->local_batch, ephemeral_params, session->x_buf, session->y_buf,
        &local_loss, session->grads_buf);
  }

  /* 3. Poison and release ephemeral full parameter buffer immediately. */
  (void)et_fsdp_engine_poison_ephemeral(ephemeral_params, session->num_params);
  session->ephemeral_poison_verified =
      verify_all_poisoned(ephemeral_params, session->num_params) ? 1 : 0;
  free(ephemeral_params);
  if (st != ET_FSDP_OK) {
    return st;
  }

  /* 4. Exact global mean loss via AllReduceSum_f64. */
  double sum_loss = 0.0;
  st = et_fsdp_allreduce_sum_f64(session->comm, &local_loss, &sum_loss, 1u,
                                 (uint64_t)tag + UINT64_C(0x01000000),
                                 (int)timeout_ms);
  if (st != ET_FSDP_OK) {
    return st;
  }
  session->last_loss =
      sum_loss / (double)et_fsdp_comm_world_size(session->comm);

  /* 5. Synchronized ZeRO step (ReduceScatter/AllReduce + global clip + AdamW). */
  et_fsdp_step_info_v1 info;
  memset(&info, 0, sizeof(info));
  st = et_fsdp_engine_step(session->engine, session->grads_buf,
                           session->num_params,
                           (uint64_t)tag + UINT64_C(0x02000000),
                           (int)timeout_ms, &info);
  if (st != ET_FSDP_OK) {
    return st;
  }
  session->last_grad_norm = info.global_grad_norm;
  session->last_clip_scale = info.clip_scale;
  session->last_clipped = (int64_t)info.clipped;
  return ET_FSDP_OK;
}

ET_FSDP_API int32_t et_fsdp_aot_step_inject_nonfinite(
    et_fsdp_aot_session_t *session, int64_t step_idx, int64_t target_rank,
    int64_t grad_index, int64_t bad_bits_u32, int64_t tag, int64_t timeout_ms) {
  if (session == NULL || step_idx < 0 || grad_index < 0 ||
      (size_t)grad_index >= session->num_params || timeout_ms <= 0 ||
      timeout_ms > 600000) {
    return ET_FSDP_ERR_INVALID_ARG;
  }

  float *ephemeral_params = (float *)malloc(session->num_params * sizeof(float));
  if (ephemeral_params == NULL) {
    return ET_FSDP_ERR_ALLOC;
  }
  et_fsdp_status_t st = et_fsdp_engine_gather_params(
      session->engine, ephemeral_params, session->num_params, (uint64_t)tag,
      (int)timeout_ms);
  if (st != ET_FSDP_OK) {
    free(ephemeral_params);
    return st;
  }

  double local_loss = 0.0;
  st = et_fsdp_mlp_make_batch(
      session->in_dim, session->num_classes, session->global_batch,
      et_fsdp_comm_rank(session->comm), et_fsdp_comm_world_size(session->comm),
      (uint64_t)step_idx, session->x_buf, session->y_buf);
  if (st == ET_FSDP_OK) {
    st = et_fsdp_mlp_forward_backward(
        session->in_dim, session->hidden_dim, session->num_classes,
        session->local_batch, ephemeral_params, session->x_buf, session->y_buf,
        &local_loss, session->grads_buf);
  }
  (void)et_fsdp_engine_poison_ephemeral(ephemeral_params, session->num_params);
  free(ephemeral_params);
  if (st != ET_FSDP_OK) {
    return st;
  }

  if ((int64_t)et_fsdp_comm_rank(session->comm) == target_rank) {
    session->grads_buf[(size_t)grad_index] =
        bits_f32((uint32_t)((uint64_t)bad_bits_u32 & UINT64_C(0xffffffff)));
  }

  et_fsdp_step_info_v1 info;
  return et_fsdp_engine_step(session->engine, session->grads_buf,
                             session->num_params,
                             (uint64_t)tag + UINT64_C(0x02000000),
                             (int)timeout_ms, &info);
}

ET_FSDP_API int32_t et_fsdp_aot_probe_tag_mismatch(
    et_fsdp_aot_session_t *session, int64_t tag, int64_t alt_delta,
    int64_t timeout_ms) {
  if (session == NULL || timeout_ms <= 0 || timeout_ms > 600000) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  double send_val = 1.0;
  double recv_val = 0.0;
  const uint64_t effective_tag =
      (et_fsdp_comm_rank(session->comm) == 0u)
          ? (uint64_t)tag
          : (uint64_t)(tag + alt_delta);
  return et_fsdp_allreduce_sum_f64(session->comm, &send_val, &recv_val, 1u,
                                   effective_tag, (int)timeout_ms);
}

ET_FSDP_API int32_t et_fsdp_aot_fd_grad_check(et_fsdp_aot_session_t *session,
                                              int64_t step_idx, double fd_eps) {
  if (session == NULL || step_idx < 0 || !isfinite(fd_eps) || fd_eps <= 0.0) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  if (et_fsdp_comm_world_size(session->comm) != 1u) {
    return ET_FSDP_ERR_INVALID_ARG;
  }

  float *params = (float *)malloc(session->num_params * sizeof(float));
  float *grads = (float *)malloc(session->num_params * sizeof(float));
  if (params == NULL || grads == NULL) {
    free(params);
    free(grads);
    return ET_FSDP_ERR_ALLOC;
  }

  et_fsdp_status_t st = et_fsdp_engine_gather_params(
      session->engine, params, session->num_params, UINT64_C(0x700001), 5000);
  if (st == ET_FSDP_OK) {
    st = et_fsdp_mlp_make_batch(session->in_dim, session->num_classes,
                                session->global_batch, 0u, 1u,
                                (uint64_t)step_idx, session->x_buf,
                                session->y_buf);
  }
  double base_loss = 0.0;
  if (st == ET_FSDP_OK) {
    st = et_fsdp_mlp_forward_backward(
        session->in_dim, session->hidden_dim, session->num_classes,
        session->local_batch, params, session->x_buf, session->y_buf,
        &base_loss, grads);
  }
  if (st != ET_FSDP_OK) {
    free(params);
    free(grads);
    return st;
  }

  /* Check every single parameter with central finite differences! */
  double max_err = 0.0;
  for (size_t i = 0u; i < session->num_params; ++i) {
    const float orig = params[i];
    const float p_plus = (float)((double)orig + fd_eps);
    const float p_minus = (float)((double)orig - fd_eps);
    const double actual_span = (double)p_plus - (double)p_minus;
    if (actual_span <= 0.0) {
      st = ET_FSDP_ERR_INVALID_ARG;
      break;
    }
    double l_plus = 0.0;
    double l_minus = 0.0;
    params[i] = p_plus;
    st = et_fsdp_mlp_forward_backward(
        session->in_dim, session->hidden_dim, session->num_classes,
        session->local_batch, params, session->x_buf, session->y_buf, &l_plus,
        NULL);
    if (st != ET_FSDP_OK) {
      break;
    }
    params[i] = p_minus;
    st = et_fsdp_mlp_forward_backward(
        session->in_dim, session->hidden_dim, session->num_classes,
        session->local_batch, params, session->x_buf, session->y_buf, &l_minus,
        NULL);
    params[i] = orig;
    if (st != ET_FSDP_OK) {
      break;
    }
    const double fd = (l_plus - l_minus) / actual_span;
    const double err = fabs((double)grads[i] - fd);
    if (err > max_err) {
      max_err = err;
    }
  }

  free(params);
  free(grads);
  if (st == ET_FSDP_OK) {
    session->last_fd_max_err = max_err;
  }
  return st;
}

ET_FSDP_API int32_t et_fsdp_aot_save_checkpoint(
    const et_fsdp_aot_session_t *session, const char *path) {
  if (session == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  return et_fsdp_engine_save_checkpoint(session->engine, path);
}

ET_FSDP_API int32_t et_fsdp_aot_load_checkpoint(et_fsdp_aot_session_t *session,
                                                const char *path) {
  if (session == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  return et_fsdp_engine_load_checkpoint(session->engine, path);
}

ET_FSDP_API int32_t et_fsdp_aot_gather_digest(et_fsdp_aot_session_t *session,
                                              int64_t tag, int64_t timeout_ms,
                                              const char *dump_path) {
  if (session == NULL || timeout_ms <= 0 || timeout_ms > 600000) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  float *full = (float *)malloc(session->num_params * sizeof(float));
  if (full == NULL) {
    return ET_FSDP_ERR_ALLOC;
  }
  et_fsdp_status_t st = et_fsdp_engine_gather_params(
      session->engine, full, session->num_params, (uint64_t)tag,
      (int)timeout_ms);
  if (st != ET_FSDP_OK) {
    free(full);
    return st;
  }

  uint8_t *raw = (uint8_t *)malloc(session->num_params * 4u);
  if (raw == NULL) {
    (void)et_fsdp_engine_poison_ephemeral(full, session->num_params);
    free(full);
    return ET_FSDP_ERR_ALLOC;
  }
  for (size_t i = 0u; i < session->num_params; ++i) {
    const float v = (full[i] == 0.0f) ? 0.0f : full[i];
    const uint32_t b = f32_bits(v);
    raw[i * 4u + 0u] = (uint8_t)(b & 0xffu);
    raw[i * 4u + 1u] = (uint8_t)((b >> 8u) & 0xffu);
    raw[i * 4u + 2u] = (uint8_t)((b >> 16u) & 0xffu);
    raw[i * 4u + 3u] = (uint8_t)((b >> 24u) & 0xffu);
  }
  uint8_t digest[32];
  st = et_fsdp_sha256_bytes(raw, session->num_params * 4u, digest);
  if (st == ET_FSDP_OK) {
    bytes_to_hex(digest, session->full_digest_hex);
    if (dump_path != NULL && dump_path[0] != '\0') {
      const int fd = open(dump_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                          (mode_t)0600);
      if (fd < 0) {
        st = ET_FSDP_ERR_IO;
      } else {
        size_t off = 0u;
        const size_t total = session->num_params * 4u;
        while (off < total) {
          const ssize_t n = write(fd, raw + off, total - off);
          if (n > 0) {
            off += (size_t)n;
          } else if (n < 0 && errno == EINTR) {
            continue;
          } else {
            st = ET_FSDP_ERR_IO;
            break;
          }
        }
        if (close(fd) != 0 && st == ET_FSDP_OK) {
          st = ET_FSDP_ERR_IO;
        }
      }
    }
  }
  (void)et_fsdp_engine_poison_ephemeral(full, session->num_params);
  free(raw);
  free(full);
  return st;
}

ET_FSDP_API int32_t et_fsdp_aot_hash_local_shard(
    et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  et_fsdp_memory_report_v1 rep;
  et_fsdp_status_t st = et_fsdp_engine_get_memory_report(session->engine, &rep);
  if (st != ET_FSDP_OK) {
    return st;
  }
  const size_t s = (size_t)rep.shard_len;
  float *p = (float *)malloc(s * sizeof(float));
  float *m = (float *)malloc(s * sizeof(float));
  float *v = (float *)malloc(s * sizeof(float));
  uint8_t *raw = (uint8_t *)malloc(s * 12u + 8u);
  if (p == NULL || m == NULL || v == NULL || raw == NULL) {
    free(p);
    free(m);
    free(v);
    free(raw);
    return ET_FSDP_ERR_ALLOC;
  }
  st = et_fsdp_engine_get_param_shard(session->engine, p, s);
  if (st == ET_FSDP_OK) {
    st = et_fsdp_engine_get_moments_shard(session->engine, m, v, s);
  }
  if (st == ET_FSDP_OK) {
    const uint64_t step = et_fsdp_engine_get_step(session->engine);
    for (size_t i = 0u; i < 8u; ++i) {
      raw[i] = (uint8_t)((step >> (8u * i)) & 0xffu);
    }
    memcpy(raw + 8u, p, s * 4u);
    memcpy(raw + 8u + s * 4u, m, s * 4u);
    memcpy(raw + 8u + s * 8u, v, s * 4u);
    uint8_t digest[32];
    st = et_fsdp_sha256_bytes(raw, s * 12u + 8u, digest);
    if (st == ET_FSDP_OK) {
      bytes_to_hex(digest, session->shard_digest_hex);
    }
  }
  free(p);
  free(m);
  free(v);
  free(raw);
  return st;
}

ET_FSDP_API int32_t et_fsdp_aot_save_shard_digest_mark(
    et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  const et_fsdp_status_t st = et_fsdp_aot_hash_local_shard(session);
  if (st != ET_FSDP_OK) {
    return st;
  }
  memcpy(session->shard_mark_hex, session->shard_digest_hex, 65u);
  return ET_FSDP_OK;
}

ET_FSDP_API int32_t et_fsdp_aot_shard_digest_matches_mark(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL || session->shard_mark_hex[0] == '\0' ||
      session->shard_digest_hex[0] == '\0') {
    return 0;
  }
  return memcmp(session->shard_mark_hex, session->shard_digest_hex, 64u) == 0
             ? 1
             : 0;
}

ET_FSDP_API int32_t et_fsdp_aot_full_digest_byte(
    const et_fsdp_aot_session_t *session, int64_t index) {
  if (session == NULL || index < 0 || index >= 64) {
    return -1;
  }
  return (int32_t)(unsigned char)session->full_digest_hex[(size_t)index];
}

ET_FSDP_API int32_t et_fsdp_aot_corrupt_file_byte(const char *path,
                                                  int64_t byte_offset) {
  if (path == NULL || byte_offset < 0) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  const int fd = open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) {
    return ET_FSDP_ERR_IO;
  }
  uint8_t b = 0u;
  if (pread(fd, &b, 1u, (off_t)byte_offset) != 1) {
    close(fd);
    return ET_FSDP_ERR_IO;
  }
  b ^= 0x5au;
  const ssize_t w = pwrite(fd, &b, 1u, (off_t)byte_offset);
  close(fd);
  return (w == 1) ? ET_FSDP_OK : ET_FSDP_ERR_IO;
}

ET_FSDP_API int32_t et_fsdp_aot_truncate_file(const char *path,
                                              int64_t new_size) {
  if (path == NULL || new_size < 0) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  return truncate(path, (off_t)new_size) == 0 ? ET_FSDP_OK : ET_FSDP_ERR_IO;
}

ET_FSDP_API int32_t et_fsdp_aot_make_symlink(const char *target_path,
                                             const char *link_path) {
  if (target_path == NULL || link_path == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  (void)unlink(link_path);
  return symlink(target_path, link_path) == 0 ? ET_FSDP_OK : ET_FSDP_ERR_IO;
}

/* --- Scalar Accessors --- */

ET_FSDP_API int64_t et_fsdp_aot_rank(const et_fsdp_aot_session_t *session) {
  return session == NULL ? -1 : (int64_t)et_fsdp_comm_rank(session->comm);
}

ET_FSDP_API int64_t et_fsdp_aot_world_size(
    const et_fsdp_aot_session_t *session) {
  return session == NULL ? -1 : (int64_t)et_fsdp_comm_world_size(session->comm);
}

ET_FSDP_API int64_t et_fsdp_aot_stage(const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1;
  }
  et_fsdp_memory_report_v1 rep;
  if (et_fsdp_engine_get_memory_report(session->engine, &rep) != ET_FSDP_OK) {
    return -1;
  }
  return (int64_t)rep.stage;
}

ET_FSDP_API int64_t et_fsdp_aot_num_params(
    const et_fsdp_aot_session_t *session) {
  return session == NULL ? -1 : (int64_t)session->num_params;
}

ET_FSDP_API int64_t et_fsdp_aot_padded_params(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1;
  }
  et_fsdp_memory_report_v1 rep;
  if (et_fsdp_engine_get_memory_report(session->engine, &rep) != ET_FSDP_OK) {
    return -1;
  }
  return (int64_t)rep.padded_params;
}

ET_FSDP_API int64_t et_fsdp_aot_shard_len(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1;
  }
  et_fsdp_memory_report_v1 rep;
  if (et_fsdp_engine_get_memory_report(session->engine, &rep) != ET_FSDP_OK) {
    return -1;
  }
  return (int64_t)rep.shard_len;
}

ET_FSDP_API int64_t et_fsdp_aot_step(const et_fsdp_aot_session_t *session) {
  return session == NULL ? -1 : (int64_t)et_fsdp_engine_get_step(session->engine);
}

ET_FSDP_API int64_t et_fsdp_aot_persistent_param_bytes(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1;
  }
  et_fsdp_memory_report_v1 rep;
  if (et_fsdp_engine_get_memory_report(session->engine, &rep) != ET_FSDP_OK) {
    return -1;
  }
  return (int64_t)rep.persistent_param_bytes;
}

ET_FSDP_API int64_t et_fsdp_aot_persistent_grad_bytes(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1;
  }
  et_fsdp_memory_report_v1 rep;
  if (et_fsdp_engine_get_memory_report(session->engine, &rep) != ET_FSDP_OK) {
    return -1;
  }
  return (int64_t)rep.persistent_grad_bytes;
}

ET_FSDP_API int64_t et_fsdp_aot_persistent_opt_bytes(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1;
  }
  et_fsdp_memory_report_v1 rep;
  if (et_fsdp_engine_get_memory_report(session->engine, &rep) != ET_FSDP_OK) {
    return -1;
  }
  return (int64_t)rep.persistent_opt_bytes;
}

ET_FSDP_API int64_t et_fsdp_aot_persistent_total_bytes(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1;
  }
  et_fsdp_memory_report_v1 rep;
  if (et_fsdp_engine_get_memory_report(session->engine, &rep) != ET_FSDP_OK) {
    return -1;
  }
  return (int64_t)rep.persistent_total_bytes;
}

ET_FSDP_API int64_t et_fsdp_aot_unsharded_baseline_bytes(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1;
  }
  et_fsdp_memory_report_v1 rep;
  if (et_fsdp_engine_get_memory_report(session->engine, &rep) != ET_FSDP_OK) {
    return -1;
  }
  return (int64_t)rep.unsharded_baseline_bytes;
}

ET_FSDP_API double et_fsdp_aot_opt_reduction_ratio(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1.0;
  }
  et_fsdp_memory_report_v1 rep;
  if (et_fsdp_engine_get_memory_report(session->engine, &rep) != ET_FSDP_OK) {
    return -1.0;
  }
  return rep.optimizer_reduction_ratio;
}

ET_FSDP_API double et_fsdp_aot_total_reduction_ratio(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1.0;
  }
  et_fsdp_memory_report_v1 rep;
  if (et_fsdp_engine_get_memory_report(session->engine, &rep) != ET_FSDP_OK) {
    return -1.0;
  }
  return rep.total_persistent_reduction_ratio;
}

ET_FSDP_API double et_fsdp_aot_last_loss(const et_fsdp_aot_session_t *session) {
  return session == NULL ? NAN : session->last_loss;
}

ET_FSDP_API double et_fsdp_aot_last_grad_norm(
    const et_fsdp_aot_session_t *session) {
  return session == NULL ? NAN : session->last_grad_norm;
}

ET_FSDP_API double et_fsdp_aot_last_clip_scale(
    const et_fsdp_aot_session_t *session) {
  return session == NULL ? NAN : session->last_clip_scale;
}

ET_FSDP_API int64_t et_fsdp_aot_last_clipped(
    const et_fsdp_aot_session_t *session) {
  return session == NULL ? -1 : session->last_clipped;
}

ET_FSDP_API double et_fsdp_aot_last_fd_max_err(
    const et_fsdp_aot_session_t *session) {
  return session == NULL ? NAN : session->last_fd_max_err;
}

ET_FSDP_API int64_t et_fsdp_aot_ephemeral_poison_verified(
    const et_fsdp_aot_session_t *session) {
  return session == NULL ? -1 : session->ephemeral_poison_verified;
}

ET_FSDP_API int64_t et_fsdp_aot_comm_bytes_sent(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1;
  }
  et_fsdp_comm_stats_v1 st;
  if (et_fsdp_comm_get_stats(session->comm, &st) != ET_FSDP_OK) {
    return -1;
  }
  return (int64_t)st.payload_bytes_sent;
}

ET_FSDP_API int64_t et_fsdp_aot_comm_bytes_recv(
    const et_fsdp_aot_session_t *session) {
  if (session == NULL) {
    return -1;
  }
  et_fsdp_comm_stats_v1 st;
  if (et_fsdp_comm_get_stats(session->comm, &st) != ET_FSDP_OK) {
    return -1;
  }
  return (int64_t)st.payload_bytes_recv;
}
