#define _GNU_SOURCE
#include "hybrid_mesh_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct et_mesh_aot_session {
  et_mesh_t *mesh;
  et_mesh_engine_t *engine;
  et_mesh_coords_v1 coords;
  et_mesh_model_config_v1 mcfg;
  et_mesh_optim_config_v1 ocfg;
  et_mesh_memory_report_v1 mem;
  et_mesh_step_info_v1 last_step;
  double last_fd_max_err;
  float *x_local;
  int32_t *targets_local;
  uint8_t full_digest[32];
  char full_digest_hex[65];
  uint8_t local_digest[32];
  char local_digest_hex[65];
  uint8_t marked_digest[32];
  int has_full_digest;
  int has_local_digest;
  int has_mark;
};

static void bytes_to_hex(const uint8_t digest[32], char hex[65]) {
  static const char lut[] = "0123456789abcdef";
  for (size_t i = 0u; i < 32u; ++i) {
    hex[i * 2u] = lut[(digest[i] >> 4u) & 0x0fu];
    hex[i * 2u + 1u] = lut[digest[i] & 0x0fu];
  }
  hex[64] = '\0';
}

static int write_all_raw(int fd, const uint8_t *buf, size_t len) {
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = write(fd, buf + off, len - off);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    if (n == 0) {
      return -1;
    }
    off += (size_t)n;
  }
  return 0;
}

ET_MESH_AOT_API int32_t et_mesh_aot_open(
    const char *sock_dir, int64_t global_rank, int64_t dp_size, int64_t tp_size,
    int64_t d_model, int64_t d_ff, int64_t num_classes, int64_t global_batch,
    double rms_eps, double lr, double beta1, double beta2, double eps,
    double weight_decay, double max_grad_norm, int64_t timeout_ms,
    et_mesh_aot_session_t **out_sess) {
  if (out_sess == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  *out_sess = NULL;
  if (global_rank < 0 || dp_size <= 0 || tp_size <= 0 || d_model <= 0 ||
      d_ff <= 0 || num_classes <= 0 || global_batch <= 0 || timeout_ms <= 0 ||
      timeout_ms > (int64_t)INT_MAX || !isfinite(rms_eps) || rms_eps <= 0.0) {
    return ET_MESH_ERR_INVALID_ARG;
  }

  et_mesh_model_config_v1 mcfg;
  memset(&mcfg, 0, sizeof(mcfg));
  mcfg.d_model = (uint32_t)d_model;
  mcfg.d_ff = (uint32_t)d_ff;
  mcfg.num_classes = (uint32_t)num_classes;
  mcfg.global_batch = (uint32_t)global_batch;
  mcfg.rms_eps = (float)rms_eps;
  mcfg.reserved = 0u;

  et_mesh_optim_config_v1 ocfg;
  ocfg.lr = lr;
  ocfg.beta1 = beta1;
  ocfg.beta2 = beta2;
  ocfg.eps = eps;
  ocfg.weight_decay = weight_decay;
  ocfg.max_grad_norm = max_grad_norm;

  /* Validate model/optim configs BEFORE touching sockets so bad configs fail fast. */
  et_mesh_status_t st = et_mesh_validate_configs(
      &mcfg, &ocfg, (uint32_t)dp_size, (uint32_t)tp_size);
  if (st != ET_MESH_OK) {
    return st;
  }

  et_mesh_t *mesh = NULL;
  st = et_mesh_open_unix(ET_MESH_ABI_MAJOR, ET_MESH_ABI_MINOR, sock_dir,
                         (uint32_t)global_rank, (uint32_t)dp_size,
                         (uint32_t)tp_size, (int)timeout_ms, &mesh);
  if (st != ET_MESH_OK) {
    return st;
  }

  et_mesh_engine_t *engine = NULL;
  st = et_mesh_engine_create(mesh, &mcfg, &ocfg, &engine);
  if (st != ET_MESH_OK) {
    et_mesh_destroy(mesh);
    return st;
  }

  et_mesh_aot_session_t *s = (et_mesh_aot_session_t *)calloc(1u, sizeof(*s));
  if (s == NULL) {
    et_mesh_engine_destroy(engine);
    et_mesh_destroy(mesh);
    return ET_MESH_ERR_ALLOC;
  }
  s->mesh = mesh;
  s->engine = engine;
  s->mcfg = mcfg;
  s->ocfg = ocfg;
  (void)et_mesh_get_coords(mesh, &s->coords);
  (void)et_mesh_engine_get_memory_report(engine, &s->mem);

  const size_t local_batch = (size_t)(mcfg.global_batch / s->coords.dp_size);
  s->x_local = (float *)calloc(local_batch * (size_t)mcfg.d_model, sizeof(float));
  s->targets_local = (int32_t *)calloc(local_batch, sizeof(int32_t));
  if (s->x_local == NULL || s->targets_local == NULL) {
    et_mesh_aot_close(s);
    return ET_MESH_ERR_ALLOC;
  }

  *out_sess = s;
  return ET_MESH_OK;
}

ET_MESH_AOT_API void et_mesh_aot_close(et_mesh_aot_session_t *sess) {
  if (sess == NULL) {
    return;
  }
  free(sess->x_local);
  free(sess->targets_local);
  et_mesh_engine_destroy(sess->engine);
  et_mesh_destroy(sess->mesh);
  free(sess);
}

ET_MESH_AOT_API int32_t et_mesh_aot_init_params(et_mesh_aot_session_t *sess,
                                                int64_t seed) {
  if (sess == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  const size_t fp = (size_t)sess->mem.full_total_params;
  float *full = (float *)malloc(fp * sizeof(float));
  if (full == NULL) {
    return ET_MESH_ERR_ALLOC;
  }
  et_mesh_status_t st =
      et_mesh_init_full_params(&sess->mcfg, (uint64_t)seed, full, fp);
  if (st == ET_MESH_OK) {
    st = et_mesh_engine_load_full_params(sess->engine, full, fp);
  }
  free(full);
  return st;
}

ET_MESH_AOT_API int32_t et_mesh_aot_train_step(et_mesh_aot_session_t *sess,
                                               int64_t step_idx,
                                               int64_t tag_base,
                                               int64_t timeout_ms) {
  if (sess == NULL || step_idx < 0 || timeout_ms <= 0 ||
      timeout_ms > (int64_t)INT_MAX) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  et_mesh_status_t st = et_mesh_make_microbatch(
      &sess->mcfg, sess->coords.dp_rank, sess->coords.dp_size,
      (uint64_t)step_idx, sess->x_local, sess->targets_local);
  if (st != ET_MESH_OK) {
    return st;
  }
  return et_mesh_engine_train_step(sess->engine, sess->x_local,
                                   sess->targets_local, (uint64_t)tag_base,
                                   (int)timeout_ms, &sess->last_step);
}

ET_MESH_AOT_API int32_t et_mesh_aot_step_inject_nonfinite(
    et_mesh_aot_session_t *sess, int64_t step_idx, int64_t target_global_rank,
    int64_t grad_index, int64_t bad_f32_bits, int64_t tag_base,
    int64_t timeout_ms) {
  if (sess == NULL || step_idx < 0 || target_global_rank < 0 ||
      grad_index < 0 || timeout_ms <= 0 || timeout_ms > (int64_t)INT_MAX) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  et_mesh_status_t st = et_mesh_make_microbatch(
      &sess->mcfg, sess->coords.dp_rank, sess->coords.dp_size,
      (uint64_t)step_idx, sess->x_local, sess->targets_local);
  if (st != ET_MESH_OK) {
    return st;
  }
  return et_mesh_engine_train_step_inject_nonfinite(
      sess->engine, sess->x_local, sess->targets_local,
      (uint32_t)target_global_rank, (size_t)grad_index, (uint32_t)bad_f32_bits,
      (uint64_t)tag_base, (int)timeout_ms, &sess->last_step);
}

/* Central finite-difference check on full dense block + loss (for 1x1 session). */
ET_MESH_AOT_API int32_t et_mesh_aot_fd_grad_check(et_mesh_aot_session_t *sess,
                                                  int64_t step_idx,
                                                  double fd_eps,
                                                  int64_t timeout_ms) {
  if (sess == NULL || step_idx < 0 || !isfinite(fd_eps) || fd_eps <= 0.0 ||
      timeout_ms <= 0 || timeout_ms > (int64_t)INT_MAX ||
      sess->coords.world_size != 1u) {
    return ET_MESH_ERR_INVALID_ARG;
  }

  et_mesh_status_t st = et_mesh_make_microbatch(
      &sess->mcfg, 0u, 1u, (uint64_t)step_idx, sess->x_local,
      sess->targets_local);
  if (st != ET_MESH_OK) {
    return st;
  }

  const size_t fp = (size_t)sess->mem.full_total_params;
  float *orig_params = (float *)malloc(fp * sizeof(float));
  float *work_params = (float *)malloc(fp * sizeof(float));
  float *analytic_grads = (float *)malloc(fp * sizeof(float));
  if (orig_params == NULL || work_params == NULL || analytic_grads == NULL) {
    free(orig_params);
    free(work_params);
    free(analytic_grads);
    return ET_MESH_ERR_ALLOC;
  }

  st = et_mesh_engine_get_local_params(sess->engine, orig_params, fp);
  if (st != ET_MESH_OK) {
    goto cleanup;
  }

  double base_loss = 0.0;
  st = et_mesh_engine_forward(sess->engine, sess->x_local, sess->targets_local,
                              UINT64_C(900001), (int)timeout_ms, &base_loss);
  if (st != ET_MESH_OK) {
    goto cleanup;
  }
  st = et_mesh_engine_backward(sess->engine, UINT64_C(900002), (int)timeout_ms,
                               NULL);
  if (st != ET_MESH_OK) {
    goto cleanup;
  }
  st = et_mesh_engine_get_local_grads(sess->engine, analytic_grads, fp);
  if (st != ET_MESH_OK) {
    goto cleanup;
  }

  memcpy(work_params, orig_params, fp * sizeof(float));
  double max_err = 0.0;
  for (size_t i = 0u; i < fp; ++i) {
    const double base_w = (double)orig_params[i];
    const float w_plus = (float)(base_w + fd_eps);
    const float w_minus = (float)(base_w - fd_eps);
    const double actual_span = (double)w_plus - (double)w_minus;
    if (actual_span <= 0.0) {
      st = ET_MESH_ERR_INVALID_ARG;
      goto cleanup;
    }

    work_params[i] = w_plus;
    st = et_mesh_engine_load_full_params(sess->engine, work_params, fp);
    if (st != ET_MESH_OK) {
      goto cleanup;
    }
    double l_plus = 0.0;
    st = et_mesh_engine_forward(sess->engine, sess->x_local,
                                sess->targets_local, UINT64_C(910000) + i * 2u,
                                (int)timeout_ms, &l_plus);
    if (st != ET_MESH_OK) {
      goto cleanup;
    }

    work_params[i] = w_minus;
    st = et_mesh_engine_load_full_params(sess->engine, work_params, fp);
    if (st != ET_MESH_OK) {
      goto cleanup;
    }
    double l_minus = 0.0;
    st = et_mesh_engine_forward(
        sess->engine, sess->x_local, sess->targets_local,
        UINT64_C(910001) + i * 2u, (int)timeout_ms, &l_minus);
    if (st != ET_MESH_OK) {
      goto cleanup;
    }

    work_params[i] = orig_params[i];
    const double fd_grad = (l_plus - l_minus) / actual_span;
    const double err = fabs(fd_grad - (double)analytic_grads[i]);
    if (err > max_err) {
      max_err = err;
    }
  }

  /* Restore original parameters and reset stats from FD probes. */
  st = et_mesh_engine_load_full_params(sess->engine, orig_params, fp);
  if (st == ET_MESH_OK) {
    sess->last_fd_max_err = max_err;
    (void)et_mesh_reset_stats(sess->mesh);
  }

cleanup:
  free(orig_params);
  free(work_params);
  free(analytic_grads);
  return st;
}

ET_MESH_AOT_API int32_t et_mesh_aot_probe_tag_mismatch(
    et_mesh_aot_session_t *sess, int64_t base_tag, int64_t diverge_on_rank,
    int64_t timeout_ms) {
  if (sess == NULL || timeout_ms <= 0 || timeout_ms > (int64_t)INT_MAX) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  uint64_t tag = (uint64_t)base_tag;
  if (diverge_on_rank >= 0 &&
      sess->coords.global_rank == (uint32_t)diverge_on_rank) {
    tag += 1u;
  }
  float send_v = 1.0f;
  float recv_v = 0.0f;
  if (sess->coords.tp_size > 1u) {
    return et_mesh_tp_allreduce_sum_f32(sess->mesh, &send_v, &recv_v, 1u, tag,
                                        (int)timeout_ms);
  }
  return et_mesh_dp_allreduce_avg_f32(sess->mesh, &send_v, &recv_v, 1u, tag,
                                      (int)timeout_ms);
}

ET_MESH_AOT_API int32_t et_mesh_aot_save_checkpoint(
    et_mesh_aot_session_t *sess, const char *path) {
  if (sess == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  return et_mesh_engine_save_checkpoint(sess->engine, path);
}

ET_MESH_AOT_API int32_t et_mesh_aot_load_checkpoint(
    et_mesh_aot_session_t *sess, const char *path) {
  if (sess == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  return et_mesh_engine_load_checkpoint(sess->engine, path);
}

ET_MESH_AOT_API int32_t et_mesh_aot_gather_digest(
    et_mesh_aot_session_t *sess, int64_t tag, int64_t timeout_ms,
    const char *dump_weights_path) {
  if (sess == NULL || timeout_ms <= 0 || timeout_ms > (int64_t)INT_MAX) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  const size_t fp = (size_t)sess->mem.full_total_params;
  float *full = (float *)malloc(fp * sizeof(float));
  if (full == NULL) {
    return ET_MESH_ERR_ALLOC;
  }
  et_mesh_status_t st = et_mesh_engine_gather_full_params(
      sess->engine, full, fp, (uint64_t)tag, (int)timeout_ms);
  if (st != ET_MESH_OK) {
    free(full);
    return st;
  }
  st = et_mesh_sha256_bytes((const uint8_t *)full, fp * sizeof(float),
                            sess->full_digest);
  if (st == ET_MESH_OK) {
    bytes_to_hex(sess->full_digest, sess->full_digest_hex);
    sess->has_full_digest = 1;
    if (dump_weights_path != NULL && dump_weights_path[0] != '\0') {
      const int fd = open(dump_weights_path,
                          O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC,
                          0600);
      if (fd < 0) {
        st = ET_MESH_ERR_IO;
      } else {
        if (write_all_raw(fd, (const uint8_t *)full, fp * sizeof(float)) != 0) {
          st = ET_MESH_ERR_IO;
        }
        (void)close(fd);
      }
    }
  }
  free(full);
  return st;
}

ET_MESH_AOT_API int32_t et_mesh_aot_hash_local_shard(
    et_mesh_aot_session_t *sess) {
  if (sess == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  et_mesh_status_t st =
      et_mesh_engine_local_state_sha256(sess->engine, sess->local_digest);
  if (st == ET_MESH_OK) {
    bytes_to_hex(sess->local_digest, sess->local_digest_hex);
    sess->has_local_digest = 1;
  }
  return st;
}

ET_MESH_AOT_API int32_t et_mesh_aot_save_shard_mark(
    et_mesh_aot_session_t *sess) {
  et_mesh_status_t st = et_mesh_aot_hash_local_shard(sess);
  if (st == ET_MESH_OK) {
    memcpy(sess->marked_digest, sess->local_digest, 32u);
    sess->has_mark = 1;
  }
  return st;
}

ET_MESH_AOT_API int32_t et_mesh_aot_shard_matches_mark(
    const et_mesh_aot_session_t *sess) {
  if (sess == NULL || !sess->has_mark || !sess->has_local_digest) {
    return 0;
  }
  return memcmp(sess->local_digest, sess->marked_digest, 32u) == 0 ? 1 : 0;
}

ET_MESH_AOT_API int32_t et_mesh_aot_full_digest_byte(
    const et_mesh_aot_session_t *sess, int64_t index) {
  if (sess == NULL || !sess->has_full_digest || index < 0 || index >= 64) {
    return -1;
  }
  return (int32_t)(unsigned char)sess->full_digest_hex[index];
}

ET_MESH_AOT_API int32_t et_mesh_aot_local_digest_byte(
    const et_mesh_aot_session_t *sess, int64_t index) {
  if (sess == NULL || !sess->has_local_digest || index < 0 || index >= 64) {
    return -1;
  }
  return (int32_t)(unsigned char)sess->local_digest_hex[index];
}

ET_MESH_AOT_API int32_t et_mesh_aot_corrupt_file_byte(const char *path,
                                                      int64_t offset) {
  if (path == NULL || offset < 0) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  const int fd = open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) {
    return ET_MESH_ERR_IO;
  }
  uint8_t b = 0u;
  if (pread(fd, &b, 1u, (off_t)offset) != 1) {
    (void)close(fd);
    return ET_MESH_ERR_IO;
  }
  b ^= 0xffu;
  if (pwrite(fd, &b, 1u, (off_t)offset) != 1) {
    (void)close(fd);
    return ET_MESH_ERR_IO;
  }
  (void)close(fd);
  return ET_MESH_OK;
}

ET_MESH_AOT_API int32_t et_mesh_aot_truncate_file(const char *path,
                                                  int64_t new_size) {
  if (path == NULL || new_size < 0) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  return truncate(path, (off_t)new_size) == 0 ? ET_MESH_OK : ET_MESH_ERR_IO;
}

ET_MESH_AOT_API int32_t et_mesh_aot_make_symlink(const char *target,
                                                 const char *link_path) {
  if (target == NULL || link_path == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  (void)unlink(link_path);
  return symlink(target, link_path) == 0 ? ET_MESH_OK : ET_MESH_ERR_IO;
}

/* --- Getters --- */

ET_MESH_AOT_API int64_t et_mesh_aot_global_rank(const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)s->coords.global_rank;
}
ET_MESH_AOT_API int64_t et_mesh_aot_dp_size(const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)s->coords.dp_size;
}
ET_MESH_AOT_API int64_t et_mesh_aot_tp_size(const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)s->coords.tp_size;
}
ET_MESH_AOT_API int64_t et_mesh_aot_dp_rank(const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)s->coords.dp_rank;
}
ET_MESH_AOT_API int64_t et_mesh_aot_tp_rank(const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)s->coords.tp_rank;
}
ET_MESH_AOT_API int64_t et_mesh_aot_step(const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)et_mesh_engine_get_step(s->engine);
}
ET_MESH_AOT_API int64_t et_mesh_aot_full_params(const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)s->mem.full_total_params;
}
ET_MESH_AOT_API int64_t et_mesh_aot_rank_params(const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)s->mem.rank_total_params;
}
ET_MESH_AOT_API int64_t et_mesh_aot_full_sharded_bytes(
    const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)s->mem.full_sharded_weight_bytes;
}
ET_MESH_AOT_API int64_t et_mesh_aot_rank_sharded_bytes(
    const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)s->mem.rank_sharded_weight_bytes;
}
ET_MESH_AOT_API int64_t et_mesh_aot_replicated_bytes(
    const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)s->mem.replicated_weight_bytes;
}
ET_MESH_AOT_API double et_mesh_aot_tp_sharding_ratio(
    const et_mesh_aot_session_t *s) {
  return s == NULL ? 0.0 : s->mem.tp_sharding_ratio;
}
ET_MESH_AOT_API double et_mesh_aot_last_local_loss(
    const et_mesh_aot_session_t *s) {
  return s == NULL ? NAN : s->last_step.local_loss;
}
ET_MESH_AOT_API double et_mesh_aot_last_global_loss(
    const et_mesh_aot_session_t *s) {
  return s == NULL ? NAN : s->last_step.global_loss;
}
ET_MESH_AOT_API double et_mesh_aot_last_grad_norm(
    const et_mesh_aot_session_t *s) {
  return s == NULL ? NAN : s->last_step.global_grad_norm;
}
ET_MESH_AOT_API int64_t et_mesh_aot_last_clipped(
    const et_mesh_aot_session_t *s) {
  return s == NULL ? -1 : (int64_t)s->last_step.clipped;
}
ET_MESH_AOT_API double et_mesh_aot_last_fd_max_err(
    const et_mesh_aot_session_t *s) {
  return s == NULL ? NAN : s->last_fd_max_err;
}
ET_MESH_AOT_API int64_t et_mesh_aot_tp_allreduce_calls(
    const et_mesh_aot_session_t *s) {
  if (s == NULL) {
    return -1;
  }
  et_mesh_comm_stats_v1 st;
  (void)et_mesh_get_stats(s->mesh, &st);
  return (int64_t)st.tp_allreduce_calls;
}
ET_MESH_AOT_API int64_t et_mesh_aot_dp_allreduce_calls(
    const et_mesh_aot_session_t *s) {
  if (s == NULL) {
    return -1;
  }
  et_mesh_comm_stats_v1 st;
  (void)et_mesh_get_stats(s->mesh, &st);
  return (int64_t)st.dp_allreduce_calls;
}
ET_MESH_AOT_API int64_t et_mesh_aot_tp_allgather_calls(
    const et_mesh_aot_session_t *s) {
  if (s == NULL) {
    return -1;
  }
  et_mesh_comm_stats_v1 st;
  (void)et_mesh_get_stats(s->mesh, &st);
  return (int64_t)st.tp_allgather_calls;
}
ET_MESH_AOT_API int64_t et_mesh_aot_tp_bytes_sent(
    const et_mesh_aot_session_t *s) {
  if (s == NULL) {
    return -1;
  }
  et_mesh_comm_stats_v1 st;
  (void)et_mesh_get_stats(s->mesh, &st);
  return (int64_t)st.tp_bytes_sent;
}
ET_MESH_AOT_API int64_t et_mesh_aot_dp_bytes_sent(
    const et_mesh_aot_session_t *s) {
  if (s == NULL) {
    return -1;
  }
  et_mesh_comm_stats_v1 st;
  (void)et_mesh_get_stats(s->mesh, &st);
  return (int64_t)st.dp_bytes_sent;
}
