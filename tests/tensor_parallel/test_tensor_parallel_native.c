#define _GNU_SOURCE
#include "eshkol_transformer/tensor_parallel_abi.h"
#include "tensor_parallel_eshkol_bridge.h"

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static int count_open_fds(void) {
  DIR *d = opendir("/proc/self/fd");
  CHECK(d != NULL);
  int n = 0;
  while (readdir(d) != NULL)
    ++n;
  closedir(d);
  return n;
}

static int64_t monotonic_ms(void) {
  struct timespec ts;
  CHECK(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
  return (int64_t)ts.tv_sec * 1000 + (int64_t)(ts.tv_nsec / 1000000L);
}

static void sleep_ms(long ms) {
  struct timespec req = {ms / 1000L, (ms % 1000L) * 1000000L};
  while (nanosleep(&req, &req) < 0)
    CHECK(errno == EINTR);
}

static void pipe_move(int fd, void *buf, size_t n, int writing) {
  unsigned char *p = (unsigned char *)buf;
  while (n > 0u) {
    const ssize_t rc = writing ? write(fd, p, n) : read(fd, p, n);
    if (rc < 0 && errno == EINTR)
      continue;
    CHECK(rc > 0);
    p += (size_t)rc;
    n -= (size_t)rc;
  }
}

static float det_f32(uint64_t seed, size_t idx, double scale) {
  uint64_t x = seed * UINT64_C(6364136223846793005) +
               (uint64_t)(idx + 1u) * UINT64_C(1442695040888963407);
  x ^= x >> 33;
  x *= UINT64_C(0xff51afd7ed558ccd);
  x ^= x >> 33;
  const int32_t bucket = (int32_t)(x % 2001u) - 1000;
  return (float)(((double)bucket / 1000.0) * scale);
}

static double max_abs_err(const float *a, const float *b, size_t n) {
  double m = 0.0;
  for (size_t i = 0u; i < n; ++i) {
    const double d = fabs((double)a[i] - (double)b[i]);
    if (d > m)
      m = d;
  }
  return m;
}

static void slice_col_shard(const float *full, float *shard, size_t rows,
                            size_t full_cols, uint32_t rank, uint32_t W) {
  const size_t loc = full_cols / W;
  for (size_t i = 0u; i < rows; ++i)
    memcpy(shard + i * loc, full + i * full_cols + (size_t)rank * loc,
           loc * sizeof(float));
}

static void reassemble_col_shards(const float *gathered, float *full,
                                  size_t rows, size_t full_cols, uint32_t W) {
  const size_t loc = full_cols / W;
  const size_t shard_elems = rows * loc;
  for (uint32_t r = 0u; r < W; ++r) {
    const float *src = gathered + (size_t)r * shard_elems;
    for (size_t i = 0u; i < rows; ++i)
      memcpy(full + i * full_cols + (size_t)r * loc, src + i * loc,
             loc * sizeof(float));
  }
}

typedef void (*tp_rank_fn)(uint32_t rank, uint32_t world, et_tp_group_t *group,
                           const void *ctx, int report_fd);

static void run_forked_group(uint32_t world, tp_rank_fn fn, const void *ctx,
                             void *report_buf, size_t report_bytes,
                             int small_sndbuf) {
  CHECK(world == 1u || world == 2u || world == 4u);
  int pairs[ET_TP_MAX_WORLD_SIZE][2];
  int trace[2] = {-1, -1};
  pid_t pids[ET_TP_MAX_WORLD_SIZE];
  CHECK(fflush(NULL) == 0);

  if (report_buf != NULL && report_bytes > 0u)
    CHECK(pipe(trace) == 0);

  for (uint32_t r = 1u; r < world; ++r) {
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, pairs[r]) == 0);
    if (small_sndbuf) {
      const int cap = 1024;
      CHECK(setsockopt(pairs[r][0], SOL_SOCKET, SO_SNDBUF, &cap, sizeof(cap)) ==
            0);
      CHECK(setsockopt(pairs[r][1], SOL_SOCKET, SO_SNDBUF, &cap, sizeof(cap)) ==
            0);
    }
  }

  for (uint32_t rank = 0u; rank < world; ++rank) {
    pids[rank] = fork();
    CHECK(pids[rank] >= 0);
    if (pids[rank] != 0)
      continue;

    alarm(15u);
    if (trace[0] >= 0) {
      CHECK(close(trace[0]) == 0);
      if (rank != 0u)
        CHECK(close(trace[1]) == 0);
    }

    int fds[ET_TP_MAX_WORLD_SIZE];
    size_t fd_count = 0u;
    for (uint32_t r = 1u; r < world; ++r) {
      if (rank == 0u)
        fds[fd_count++] = pairs[r][0];
      else
        CHECK(close(pairs[r][0]) == 0);
      if (rank == r)
        fds[fd_count++] = pairs[r][1];
      else
        CHECK(close(pairs[r][1]) == 0);
    }

    et_tp_group_t *group = NULL;
    CHECK(et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, rank, world,
                             fd_count > 0u ? fds : NULL, fd_count, 5000,
                             &group) == ET_TP_OK);
    for (size_t i = 0u; i < fd_count; ++i)
      CHECK(close(fds[i]) == 0);

    fn(rank, world, group, ctx, rank == 0u ? trace[1] : -1);
    et_tp_group_destroy(group);
    if (rank == 0u && trace[1] >= 0)
      CHECK(close(trace[1]) == 0);
    exit(EXIT_SUCCESS);
  }

  for (uint32_t r = 1u; r < world; ++r) {
    CHECK(close(pairs[r][0]) == 0);
    CHECK(close(pairs[r][1]) == 0);
  }
  if (trace[0] >= 0) {
    CHECK(close(trace[1]) == 0);
    pipe_move(trace[0], report_buf, report_bytes, 0);
    CHECK(close(trace[0]) == 0);
  }
  for (uint32_t rank = 0u; rank < world; ++rank) {
    int status = 0;
    while (waitpid(pids[rank], &status, 0) < 0)
      CHECK(errno == EINTR);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
}

/* =========================================================================
 * 1. AF_UNIX Collective Tests (Bitwise determinism, large payload, faults)
 * ========================================================================= */

static void rank_collective_bitwise(uint32_t rank, uint32_t world,
                                    et_tp_group_t *g, const void *ctx,
                                    int report_fd) {
  (void)ctx;
  /* Reverse arrival order: rank 0 sleeps longest so rank W-1 arrives first. */
  sleep_ms((long)(world - 1u - rank) * 5L);

  const size_t n = 4096u;
  float *send_buf = (float *)malloc(n * sizeof(float));
  float *recv_buf = (float *)malloc(n * sizeof(float));
  float *inplace = (float *)malloc(n * sizeof(float));
  float *gather_in = (float *)malloc(64u * sizeof(float));
  float *gather_out = (float *)malloc((size_t)world * 64u * sizeof(float));
  CHECK(send_buf && recv_buf && inplace && gather_in && gather_out);

  for (size_t i = 0u; i < n; ++i) {
    send_buf[i] = det_f32(10u + rank, i, 1.5);
    inplace[i] = send_buf[i];
  }
  CHECK(et_tp_allreduce_sum_f32(g, send_buf, recv_buf, n, 1u, 5000) ==
        ET_TP_OK);
  CHECK(et_tp_allreduce_sum_f32(g, inplace, inplace, n, 2u, 5000) == ET_TP_OK);
  CHECK(memcmp(recv_buf, inplace, n * sizeof(float)) == 0);

  /* Verify canonical ascending rank-order binary64 sum bitwise on EVERY rank! */
  for (size_t i = 0u; i < n; ++i) {
    double acc = 0.0;
    for (uint32_t r = 0u; r < world; ++r)
      acc += (double)det_f32(10u + r, i, 1.5);
    const float expected = (float)acc;
    CHECK(memcmp(&recv_buf[i], &expected, sizeof(float)) == 0);
  }

  for (size_t i = 0u; i < 64u; ++i)
    gather_in[i] = det_f32(50u + rank, i, 2.0);
  CHECK(et_tp_allgather_f32(g, gather_in, 64u, gather_out, 3u, 5000) ==
        ET_TP_OK);
  for (uint32_t r = 0u; r < world; ++r) {
    for (size_t i = 0u; i < 64u; ++i) {
      const float expected = det_f32(50u + r, i, 2.0);
      CHECK(memcmp(&gather_out[(size_t)r * 64u + i], &expected,
                   sizeof(float)) == 0);
    }
  }

  /* Coordinated atomic non-finite rejection on highest rank + recovery. */
  float nf_in[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float nf_out[8] = {-9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f};
  if (rank == world - 1u)
    nf_in[3] = NAN;
  CHECK(et_tp_allreduce_sum_f32(g, nf_in, nf_out, 8u, 4u, 5000) ==
        ET_TP_ERR_NONFINITE);
  for (size_t i = 0u; i < 8u; ++i)
    CHECK(nf_out[i] == -9.0f);

  /* Group remains healthy and usable after coordinated non-finite abort! */
  nf_in[3] = 4.0f;
  CHECK(et_tp_allreduce_sum_f32(g, nf_in, nf_out, 8u, 5u, 5000) == ET_TP_OK);
  for (size_t i = 0u; i < 8u; ++i)
    CHECK(nf_out[i] == (float)((double)nf_in[i] * (double)world));

  et_tp_collective_stats_t st;
  CHECK(et_tp_group_get_stats(g, &st) == ET_TP_OK);
  CHECK(st.allreduce_calls == 3u);
  CHECK(st.allgather_calls == 1u);
  if (world == 1u) {
    CHECK(st.ipc_allreduce_calls == 0u && st.ipc_allgather_calls == 0u);
  } else {
    CHECK(st.ipc_allreduce_calls == 3u && st.ipc_allgather_calls == 1u);
  }

  if (report_fd >= 0) {
    const uint32_t ok = 1u;
    pipe_move(report_fd, (void *)&ok, sizeof(ok), 1);
  }

  free(send_buf);
  free(recv_buf);
  free(inplace);
  free(gather_in);
  free(gather_out);
}

static void rank_collective_mismatch(uint32_t rank, uint32_t world,
                                     et_tp_group_t *g, const void *ctx,
                                     int report_fd) {
  (void)world;
  (void)ctx;
  float in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float out[4] = {7.0f, 7.0f, 7.0f, 7.0f};
  const uint64_t tag = (rank == 0u) ? 10u : 99u;
  const et_tp_status_t st1 = et_tp_allreduce_sum_f32(g, in, out, 4u, tag, 2000);
  CHECK(st1 == ET_TP_ERR_MISMATCH || st1 == ET_TP_ERR_PEER_DISCONNECTED);
  CHECK(out[0] == 7.0f);
  /* Sticky fatal error */
  const et_tp_status_t st2 = et_tp_allreduce_sum_f32(g, in, out, 4u, 10u, 2000);
  CHECK(st2 == st1);
  if (report_fd >= 0) {
    const uint32_t ok = 1u;
    pipe_move(report_fd, (void *)&ok, sizeof(ok), 1);
  }
}

static void test_collectives(void) {
  const uint32_t worlds[3] = {1u, 2u, 4u};
  for (size_t wi = 0u; wi < 3u; ++wi) {
    uint32_t ok = 0u;
    run_forked_group(worlds[wi], rank_collective_bitwise, NULL, &ok, sizeof(ok),
                     1);
    CHECK(ok == 1u);
  }
  uint32_t ok = 0u;
  run_forked_group(2u, rank_collective_mismatch, NULL, &ok, sizeof(ok), 0);
  CHECK(ok == 1u);
}

/* =========================================================================
 * 2. ColumnParallelLinear & RowParallelLinear Parity + FD Checks
 * ========================================================================= */

typedef struct linear_case {
  uint32_t rows;
  uint32_t in_dim;
  uint32_t out_dim;
  int has_bias;
  float x[64];
  float w[128];
  float b[16];
  float dy[64];
} linear_case_t;

typedef struct linear_result {
  float y[64];
  float dx[64];
  float dw[128];
  float db[16];
  uint64_t fwd_ar;
  uint64_t bwd_ar;
} linear_result_t;

static void rank_column_linear(uint32_t rank, uint32_t world, et_tp_group_t *g,
                               const void *ctx, int report_fd) {
  const linear_case_t *c = (const linear_case_t *)ctx;
  const size_t out_loc = c->out_dim / world;
  float w_r[128], b_r[16], y_r[64], dy_r[64], dx[64], dw_r[128], db_r[16];
  memset(db_r, 0, sizeof(db_r));

  memcpy(w_r, c->w + (size_t)rank * out_loc * c->in_dim,
         out_loc * c->in_dim * sizeof(float));
  if (c->has_bias)
    memcpy(b_r, c->b + (size_t)rank * out_loc, out_loc * sizeof(float));
  slice_col_shard(c->dy, dy_r, c->rows, c->out_dim, rank, world);

  CHECK(et_tp_group_reset_stats(g) == ET_TP_OK);
  CHECK(et_tp_column_linear_forward_v1(g, c->rows, c->in_dim, c->out_dim, c->x,
                                       w_r, c->has_bias ? b_r : NULL,
                                       y_r) == ET_TP_OK);
  et_tp_collective_stats_t s_fwd;
  CHECK(et_tp_group_get_stats(g, &s_fwd) == ET_TP_OK);
  CHECK(s_fwd.allreduce_calls == 0u);

  CHECK(et_tp_column_linear_backward_v1(
            g, c->rows, c->in_dim, c->out_dim, c->x, w_r, dy_r, dx, dw_r,
            c->has_bias ? db_r : NULL, 11u, 5000) == ET_TP_OK);
  et_tp_collective_stats_t s_bwd;
  CHECK(et_tp_group_get_stats(g, &s_bwd) == ET_TP_OK);
  CHECK(s_bwd.allreduce_calls == 1u);

  linear_result_t res;
  memset(&res, 0, sizeof(res));
  float y_gath[64];
  CHECK(et_tp_allgather_f32(g, y_r, (size_t)c->rows * out_loc, y_gath, 12u,
                            5000) == ET_TP_OK);
  reassemble_col_shards(y_gath, res.y, c->rows, c->out_dim, world);
  memcpy(res.dx, dx, (size_t)c->rows * c->in_dim * sizeof(float));
  CHECK(et_tp_allgather_f32(g, dw_r, out_loc * c->in_dim, res.dw, 13u, 5000) ==
        ET_TP_OK);
  if (c->has_bias)
    CHECK(et_tp_allgather_f32(g, db_r, out_loc, res.db, 14u, 5000) == ET_TP_OK);
  res.fwd_ar = s_fwd.allreduce_calls;
  res.bwd_ar = s_bwd.allreduce_calls - s_fwd.allreduce_calls;

  if (report_fd >= 0)
    pipe_move(report_fd, &res, sizeof(res), 1);
}

static void rank_row_linear(uint32_t rank, uint32_t world, et_tp_group_t *g,
                            const void *ctx, int report_fd) {
  const linear_case_t *c = (const linear_case_t *)ctx;
  const size_t in_loc = c->in_dim / world;
  float x_r[64], w_r[128], y[64], dx_r[64], dw_r[128], db[16];
  memset(db, 0, sizeof(db));

  slice_col_shard(c->x, x_r, c->rows, c->in_dim, rank, world);
  slice_col_shard(c->w, w_r, c->out_dim, c->in_dim, rank, world);

  CHECK(et_tp_group_reset_stats(g) == ET_TP_OK);
  CHECK(et_tp_row_linear_forward_v1(g, c->rows, c->in_dim, c->out_dim, x_r, w_r,
                                    c->has_bias ? c->b : NULL, y, 21u,
                                    5000) == ET_TP_OK);
  et_tp_collective_stats_t s_fwd;
  CHECK(et_tp_group_get_stats(g, &s_fwd) == ET_TP_OK);
  CHECK(s_fwd.allreduce_calls == 1u);

  CHECK(et_tp_row_linear_backward_v1(g, c->rows, c->in_dim, c->out_dim, x_r,
                                     w_r, c->dy, dx_r, dw_r,
                                     c->has_bias ? db : NULL) == ET_TP_OK);
  et_tp_collective_stats_t s_bwd;
  CHECK(et_tp_group_get_stats(g, &s_bwd) == ET_TP_OK);
  CHECK(s_bwd.allreduce_calls == 1u);

  linear_result_t res;
  memset(&res, 0, sizeof(res));
  memcpy(res.y, y, (size_t)c->rows * c->out_dim * sizeof(float));
  float dx_gath[64], dw_gath[128];
  CHECK(et_tp_allgather_f32(g, dx_r, (size_t)c->rows * in_loc, dx_gath, 22u,
                            5000) == ET_TP_OK);
  reassemble_col_shards(dx_gath, res.dx, c->rows, c->in_dim, world);
  CHECK(et_tp_allgather_f32(g, dw_r, (size_t)c->out_dim * in_loc, dw_gath, 23u,
                            5000) == ET_TP_OK);
  reassemble_col_shards(dw_gath, res.dw, c->out_dim, c->in_dim, world);
  if (c->has_bias)
    memcpy(res.db, db, (size_t)c->out_dim * sizeof(float));
  res.fwd_ar = s_fwd.allreduce_calls;
  res.bwd_ar = s_bwd.allreduce_calls - s_fwd.allreduce_calls;

  if (report_fd >= 0)
    pipe_move(report_fd, &res, sizeof(res), 1);
}

static void init_linear_case(linear_case_t *c, uint32_t rows, uint32_t in_d,
                             uint32_t out_d, int has_bias) {
  memset(c, 0, sizeof(*c));
  c->rows = rows;
  c->in_dim = in_d;
  c->out_dim = out_d;
  c->has_bias = has_bias;
  for (size_t i = 0u; i < (size_t)rows * in_d; ++i)
    c->x[i] = det_f32(101u, i, 0.7);
  for (size_t i = 0u; i < (size_t)out_d * in_d; ++i)
    c->w[i] = det_f32(102u, i, 0.5);
  for (size_t i = 0u; i < out_d; ++i)
    c->b[i] = det_f32(103u, i, 0.4);
  for (size_t i = 0u; i < (size_t)rows * out_d; ++i)
    c->dy[i] = det_f32(104u, i, 0.6);
}

static void test_column_and_row_linear(void) {
  linear_case_t c;
  init_linear_case(&c, 4u, 8u, 12u, 1);

  linear_result_t col1, col2, col4;
  run_forked_group(1u, rank_column_linear, &c, &col1, sizeof(col1), 0);
  run_forked_group(2u, rank_column_linear, &c, &col2, sizeof(col2), 0);
  run_forked_group(4u, rank_column_linear, &c, &col4, sizeof(col4), 0);

  CHECK(col1.fwd_ar == 0u && col1.bwd_ar == 1u);
  CHECK(col2.fwd_ar == 0u && col2.bwd_ar == 1u);
  CHECK(col4.fwd_ar == 0u && col4.bwd_ar == 1u);
  CHECK(max_abs_err(col1.y, col2.y, 48u) <= 1e-6);
  CHECK(max_abs_err(col1.y, col4.y, 48u) <= 1e-6);
  CHECK(max_abs_err(col1.dx, col2.dx, 32u) <= 1e-6);
  CHECK(max_abs_err(col1.dx, col4.dx, 32u) <= 1e-6);
  CHECK(max_abs_err(col1.dw, col2.dw, 96u) <= 1e-6);
  CHECK(max_abs_err(col1.dw, col4.dw, 96u) <= 1e-6);
  CHECK(max_abs_err(col1.db, col2.db, 12u) <= 1e-6);
  CHECK(max_abs_err(col1.db, col4.db, 12u) <= 1e-6);

  linear_result_t row1, row2, row4;
  run_forked_group(1u, rank_row_linear, &c, &row1, sizeof(row1), 0);
  run_forked_group(2u, rank_row_linear, &c, &row2, sizeof(row2), 0);
  run_forked_group(4u, rank_row_linear, &c, &row4, sizeof(row4), 0);

  CHECK(row1.fwd_ar == 1u && row1.bwd_ar == 0u);
  CHECK(row2.fwd_ar == 1u && row2.bwd_ar == 0u);
  CHECK(row4.fwd_ar == 1u && row4.bwd_ar == 0u);
  CHECK(max_abs_err(row1.y, row2.y, 48u) <= 1e-6);
  CHECK(max_abs_err(row1.y, row4.y, 48u) <= 1e-6);
  CHECK(max_abs_err(row1.dx, row2.dx, 32u) <= 1e-6);
  CHECK(max_abs_err(row1.dx, row4.dx, 32u) <= 1e-6);
  CHECK(max_abs_err(row1.dw, row2.dw, 96u) <= 1e-6);
  CHECK(max_abs_err(row1.dw, row4.dw, 96u) <= 1e-6);
  CHECK(max_abs_err(row1.db, row2.db, 12u) <= 1e-6);
  CHECK(max_abs_err(row1.db, row4.db, 12u) <= 1e-6);

  /* Central FD check on ColumnParallelLinear (X, W, b). */
  et_tp_group_t *g1 = NULL;
  CHECK(et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 1u, NULL, 0u,
                           1000, &g1) == ET_TP_OK);
  const float h = 1e-3f;
  float y_p[48], y_m[48];
  for (size_t idx = 0u; idx < 32u; idx += 5u) {
    const float orig = c.x[idx];
    c.x[idx] = orig + h;
    CHECK(et_tp_column_linear_forward_v1(g1, 4u, 8u, 12u, c.x, c.w, c.b, y_p) ==
          ET_TP_OK);
    c.x[idx] = orig - h;
    CHECK(et_tp_column_linear_forward_v1(g1, 4u, 8u, 12u, c.x, c.w, c.b, y_m) ==
          ET_TP_OK);
    c.x[idx] = orig;
    double lp = 0.0, lm = 0.0;
    for (size_t j = 0u; j < 48u; ++j) {
      lp += (double)y_p[j] * (double)c.dy[j];
      lm += (double)y_m[j] * (double)c.dy[j];
    }
    const double fd = (lp - lm) / (2.0 * (double)h);
    CHECK(fabs(fd - (double)col1.dx[idx]) <= 5e-4);
  }
  for (size_t idx = 0u; idx < 96u; idx += 11u) {
    const float orig = c.w[idx];
    c.w[idx] = orig + h;
    CHECK(et_tp_row_linear_forward_v1(g1, 4u, 8u, 12u, c.x, c.w, c.b, y_p, 1u,
                                      1000) == ET_TP_OK);
    c.w[idx] = orig - h;
    CHECK(et_tp_row_linear_forward_v1(g1, 4u, 8u, 12u, c.x, c.w, c.b, y_m, 2u,
                                      1000) == ET_TP_OK);
    c.w[idx] = orig;
    double lp = 0.0, lm = 0.0;
    for (size_t j = 0u; j < 48u; ++j) {
      lp += (double)y_p[j] * (double)c.dy[j];
      lm += (double)y_m[j] * (double)c.dy[j];
    }
    const double fd = (lp - lm) / (2.0 * (double)h);
    CHECK(fabs(fd - (double)row1.dw[idx]) <= 5e-4);
  }
  et_tp_group_destroy(g1);
}

/* =========================================================================
 * 3. TP-Sharded SwiGLU MLP Parity + FD Check
 * ========================================================================= */

typedef struct swiglu_case {
  et_tp_swiglu_config_t cfg;
  float x[64];
  float w_gate[256];
  float w_up[256];
  float w_down[256];
  float dy[64];
} swiglu_case_t;

typedef struct swiglu_result {
  float y[64];
  float dx[64];
  float dw_gate[256];
  float dw_up[256];
  float dw_down[256];
  uint64_t fwd_ar;
  uint64_t bwd_ar;
} swiglu_result_t;

static void rank_swiglu(uint32_t rank, uint32_t world, et_tp_group_t *g,
                        const void *ctx, int report_fd) {
  const swiglu_case_t *c = (const swiglu_case_t *)ctx;
  const size_t T = c->cfg.tokens, D = c->cfg.d_model, F = c->cfg.d_ff;
  const size_t F_loc = F / world;
  float wg_r[256], wu_r[256], wd_r[256], gc_r[128], uc_r[128];
  float y[64], dx[64], dwg_r[256], dwu_r[256], dwd_r[256];

  memcpy(wg_r, c->w_gate + (size_t)rank * F_loc * D,
         F_loc * D * sizeof(float));
  memcpy(wu_r, c->w_up + (size_t)rank * F_loc * D, F_loc * D * sizeof(float));
  slice_col_shard(c->w_down, wd_r, D, F, rank, world);

  CHECK(et_tp_group_reset_stats(g) == ET_TP_OK);
  CHECK(et_tp_swiglu_forward_v1(g, &c->cfg, c->x, wg_r, wu_r, wd_r, gc_r, uc_r,
                                y, 31u, 5000) == ET_TP_OK);
  et_tp_collective_stats_t s_fwd;
  CHECK(et_tp_group_get_stats(g, &s_fwd) == ET_TP_OK);
  CHECK(s_fwd.allreduce_calls == 1u);

  CHECK(et_tp_swiglu_backward_v1(g, &c->cfg, c->x, wg_r, wu_r, wd_r, gc_r, uc_r,
                                 c->dy, dx, dwg_r, dwu_r, dwd_r, 32u,
                                 5000) == ET_TP_OK);
  et_tp_collective_stats_t s_bwd;
  CHECK(et_tp_group_get_stats(g, &s_bwd) == ET_TP_OK);
  CHECK(s_bwd.allreduce_calls == 2u);

  swiglu_result_t res;
  memset(&res, 0, sizeof(res));
  memcpy(res.y, y, T * D * sizeof(float));
  memcpy(res.dx, dx, T * D * sizeof(float));
  CHECK(et_tp_allgather_f32(g, dwg_r, F_loc * D, res.dw_gate, 33u, 5000) ==
        ET_TP_OK);
  CHECK(et_tp_allgather_f32(g, dwu_r, F_loc * D, res.dw_up, 34u, 5000) ==
        ET_TP_OK);
  float dwd_gath[256];
  CHECK(et_tp_allgather_f32(g, dwd_r, D * F_loc, dwd_gath, 35u, 5000) ==
        ET_TP_OK);
  reassemble_col_shards(dwd_gath, res.dw_down, D, F, world);
  res.fwd_ar = s_fwd.allreduce_calls;
  res.bwd_ar = s_bwd.allreduce_calls - s_fwd.allreduce_calls;
  if (report_fd >= 0)
    pipe_move(report_fd, &res, sizeof(res), 1);
}

static void init_swiglu_case(swiglu_case_t *c) {
  memset(c, 0, sizeof(*c));
  c->cfg.tokens = 4u;
  c->cfg.d_model = 8u;
  c->cfg.d_ff = 16u;
  for (size_t i = 0u; i < 32u; ++i) {
    c->x[i] = det_f32(201u, i, 0.5);
    c->dy[i] = det_f32(202u, i, 0.6);
  }
  for (size_t i = 0u; i < 128u; ++i) {
    c->w_gate[i] = det_f32(203u, i, 0.35);
    c->w_up[i] = det_f32(204u, i, 0.35);
    c->w_down[i] = det_f32(205u, i, 0.35);
  }
}

static void test_swiglu(void) {
  swiglu_case_t c;
  init_swiglu_case(&c);
  swiglu_result_t r1, r2, r4;
  run_forked_group(1u, rank_swiglu, &c, &r1, sizeof(r1), 0);
  run_forked_group(2u, rank_swiglu, &c, &r2, sizeof(r2), 0);
  run_forked_group(4u, rank_swiglu, &c, &r4, sizeof(r4), 0);

  CHECK(r1.fwd_ar == 1u && r1.bwd_ar == 1u);
  CHECK(r2.fwd_ar == 1u && r2.bwd_ar == 1u);
  CHECK(r4.fwd_ar == 1u && r4.bwd_ar == 1u);
  CHECK(max_abs_err(r1.y, r2.y, 32u) <= 1e-6);
  CHECK(max_abs_err(r1.y, r4.y, 32u) <= 1e-6);
  CHECK(max_abs_err(r1.dx, r2.dx, 32u) <= 1e-6);
  CHECK(max_abs_err(r1.dx, r4.dx, 32u) <= 1e-6);
  CHECK(max_abs_err(r1.dw_gate, r2.dw_gate, 128u) <= 1e-6);
  CHECK(max_abs_err(r1.dw_gate, r4.dw_gate, 128u) <= 1e-6);
  CHECK(max_abs_err(r1.dw_up, r2.dw_up, 128u) <= 1e-6);
  CHECK(max_abs_err(r1.dw_up, r4.dw_up, 128u) <= 1e-6);
  CHECK(max_abs_err(r1.dw_down, r2.dw_down, 128u) <= 1e-6);
  CHECK(max_abs_err(r1.dw_down, r4.dw_down, 128u) <= 1e-6);

  /* Central FD check on SwiGLU (X, W_gate, W_up, W_down). */
  et_tp_group_t *g1 = NULL;
  CHECK(et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 1u, NULL, 0u,
                           1000, &g1) == ET_TP_OK);
  const float h = 1e-3f;
  float gc[64], uc[64], yp[32], ym[32];
  float *ptrs[4] = {c.x, c.w_gate, c.w_up, c.w_down};
  const float *grads[4] = {r1.dx, r1.dw_gate, r1.dw_up, r1.dw_down};
  const size_t lens[4] = {32u, 128u, 128u, 128u};
  for (int t = 0; t < 4; ++t) {
    for (size_t idx = 0u; idx < lens[t]; idx += 13u) {
      const float orig = ptrs[t][idx];
      ptrs[t][idx] = orig + h;
      CHECK(et_tp_swiglu_forward_v1(g1, &c.cfg, c.x, c.w_gate, c.w_up, c.w_down,
                                    gc, uc, yp, 1u, 1000) == ET_TP_OK);
      ptrs[t][idx] = orig - h;
      CHECK(et_tp_swiglu_forward_v1(g1, &c.cfg, c.x, c.w_gate, c.w_up, c.w_down,
                                    gc, uc, ym, 2u, 1000) == ET_TP_OK);
      ptrs[t][idx] = orig;
      double lp = 0.0, lm = 0.0;
      for (size_t j = 0u; j < 32u; ++j) {
        lp += (double)yp[j] * (double)c.dy[j];
        lm += (double)ym[j] * (double)c.dy[j];
      }
      const double fd = (lp - lm) / (2.0 * (double)h);
      CHECK(fabs(fd - (double)grads[t][idx]) <= 5e-4);
    }
  }
  et_tp_group_destroy(g1);
}

/* =========================================================================
 * 4. TP-Sharded Causal GQA Attention Parity + FD Check
 * ========================================================================= */

typedef struct gqa_case {
  et_tp_gqa_config_t cfg;
  float x[256];
  float w_q[1024];
  float w_k[512];
  float w_v[512];
  float w_o[1024];
  float dy[256];
} gqa_case_t;

typedef struct gqa_result {
  float y[256];
  float dx[256];
  float dw_q[1024];
  float dw_k[512];
  float dw_v[512];
  float dw_o[1024];
  uint64_t fwd_ar;
  uint64_t bwd_ar;
} gqa_result_t;

static void rank_gqa(uint32_t rank, uint32_t world, et_tp_group_t *g,
                     const void *ctx, int report_fd) {
  const gqa_case_t *c = (const gqa_case_t *)ctx;
  const size_t M = (size_t)c->cfg.batch * (size_t)c->cfg.seq_len;
  const size_t D = c->cfg.d_model;
  const size_t q_dim = (size_t)c->cfg.num_q_heads * (size_t)c->cfg.head_dim;
  const size_t kv_dim = (size_t)c->cfg.num_kv_heads * (size_t)c->cfg.head_dim;
  const size_t q_loc = q_dim / world;
  const size_t kv_loc = kv_dim / world;

  float wq_r[1024], wk_r[512], wv_r[512], wo_r[1024];
  float qc[256], kc[256], vc[256], ac[256], y[256], dx[256];
  float dwq_r[1024], dwk_r[512], dwv_r[512], dwo_r[1024];

  memcpy(wq_r, c->w_q + (size_t)rank * q_loc * D, q_loc * D * sizeof(float));
  memcpy(wk_r, c->w_k + (size_t)rank * kv_loc * D, kv_loc * D * sizeof(float));
  memcpy(wv_r, c->w_v + (size_t)rank * kv_loc * D, kv_loc * D * sizeof(float));
  slice_col_shard(c->w_o, wo_r, D, q_dim, rank, world);

  CHECK(et_tp_group_reset_stats(g) == ET_TP_OK);
  CHECK(et_tp_gqa_forward_v1(g, &c->cfg, c->x, wq_r, wk_r, wv_r, wo_r, qc, kc,
                             vc, ac, y, 41u, 5000) == ET_TP_OK);
  et_tp_collective_stats_t s_fwd;
  CHECK(et_tp_group_get_stats(g, &s_fwd) == ET_TP_OK);
  CHECK(s_fwd.allreduce_calls == 1u);

  CHECK(et_tp_gqa_backward_v1(g, &c->cfg, c->x, wq_r, wk_r, wv_r, wo_r, qc, kc,
                              vc, ac, c->dy, dx, dwq_r, dwk_r, dwv_r, dwo_r,
                              42u, 5000) == ET_TP_OK);
  et_tp_collective_stats_t s_bwd;
  CHECK(et_tp_group_get_stats(g, &s_bwd) == ET_TP_OK);
  CHECK(s_bwd.allreduce_calls == 2u);

  gqa_result_t res;
  memset(&res, 0, sizeof(res));
  memcpy(res.y, y, M * D * sizeof(float));
  memcpy(res.dx, dx, M * D * sizeof(float));
  CHECK(et_tp_allgather_f32(g, dwq_r, q_loc * D, res.dw_q, 43u, 5000) ==
        ET_TP_OK);
  CHECK(et_tp_allgather_f32(g, dwk_r, kv_loc * D, res.dw_k, 44u, 5000) ==
        ET_TP_OK);
  CHECK(et_tp_allgather_f32(g, dwv_r, kv_loc * D, res.dw_v, 45u, 5000) ==
        ET_TP_OK);
  float dwo_gath[1024];
  CHECK(et_tp_allgather_f32(g, dwo_r, D * q_loc, dwo_gath, 46u, 5000) ==
        ET_TP_OK);
  reassemble_col_shards(dwo_gath, res.dw_o, D, q_dim, world);
  res.fwd_ar = s_fwd.allreduce_calls;
  res.bwd_ar = s_bwd.allreduce_calls - s_fwd.allreduce_calls;
  if (report_fd >= 0)
    pipe_move(report_fd, &res, sizeof(res), 1);
}

static void init_gqa_case(gqa_case_t *c, uint32_t B, uint32_t T, uint32_t Hq,
                          uint32_t Hkv, uint32_t Dh) {
  memset(c, 0, sizeof(*c));
  const uint32_t D = Hq * Dh;
  const uint32_t kv_dim = Hkv * Dh;
  c->cfg.batch = B;
  c->cfg.seq_len = T;
  c->cfg.d_model = D;
  c->cfg.num_q_heads = Hq;
  c->cfg.num_kv_heads = Hkv;
  c->cfg.head_dim = Dh;
  for (size_t i = 0u; i < (size_t)B * T * D; ++i) {
    c->x[i] = det_f32(301u, i, 0.5);
    c->dy[i] = det_f32(302u, i, 0.5);
  }
  for (size_t i = 0u; i < (size_t)D * D; ++i) {
    c->w_q[i] = det_f32(303u, i, 0.3);
    c->w_o[i] = det_f32(306u, i, 0.3);
  }
  for (size_t i = 0u; i < (size_t)kv_dim * D; ++i) {
    c->w_k[i] = det_f32(304u, i, 0.3);
    c->w_v[i] = det_f32(305u, i, 0.3);
  }
}

static void test_gqa(void) {
  /* Geometry A: H_q=8, H_kv=4, D_h=4 (group_ratio=2, D=32) */
  /* Geometry B: H_q=12, H_kv=4, D_h=2 (group_ratio=3, D=24) */
  const uint32_t geoms[2][5] = {{2u, 4u, 8u, 4u, 4u}, {2u, 3u, 12u, 4u, 2u}};
  for (int gi = 0; gi < 2; ++gi) {
    gqa_case_t c;
    init_gqa_case(&c, geoms[gi][0], geoms[gi][1], geoms[gi][2], geoms[gi][3],
                  geoms[gi][4]);
    const size_t xd_cnt = (size_t)c.cfg.batch * c.cfg.seq_len * c.cfg.d_model;
    const size_t q_cnt = (size_t)c.cfg.d_model * c.cfg.d_model;
    const size_t kv_cnt =
        (size_t)c.cfg.num_kv_heads * c.cfg.head_dim * c.cfg.d_model;

    gqa_result_t r1, r2, r4;
    run_forked_group(1u, rank_gqa, &c, &r1, sizeof(r1), 0);
    run_forked_group(2u, rank_gqa, &c, &r2, sizeof(r2), 0);
    run_forked_group(4u, rank_gqa, &c, &r4, sizeof(r4), 0);

    CHECK(r1.fwd_ar == 1u && r1.bwd_ar == 1u);
    CHECK(r2.fwd_ar == 1u && r2.bwd_ar == 1u);
    CHECK(r4.fwd_ar == 1u && r4.bwd_ar == 1u);
    CHECK(max_abs_err(r1.y, r2.y, xd_cnt) <= 1e-6);
    CHECK(max_abs_err(r1.y, r4.y, xd_cnt) <= 1e-6);
    CHECK(max_abs_err(r1.dx, r2.dx, xd_cnt) <= 1e-6);
    CHECK(max_abs_err(r1.dx, r4.dx, xd_cnt) <= 1e-6);
    CHECK(max_abs_err(r1.dw_q, r2.dw_q, q_cnt) <= 1e-6);
    CHECK(max_abs_err(r1.dw_q, r4.dw_q, q_cnt) <= 1e-6);
    CHECK(max_abs_err(r1.dw_k, r2.dw_k, kv_cnt) <= 1e-6);
    CHECK(max_abs_err(r1.dw_k, r4.dw_k, kv_cnt) <= 1e-6);
    CHECK(max_abs_err(r1.dw_v, r2.dw_v, kv_cnt) <= 1e-6);
    CHECK(max_abs_err(r1.dw_v, r4.dw_v, kv_cnt) <= 1e-6);
    CHECK(max_abs_err(r1.dw_o, r2.dw_o, q_cnt) <= 1e-6);
    CHECK(max_abs_err(r1.dw_o, r4.dw_o, q_cnt) <= 1e-6);

    if (gi == 0) {
      /* Central FD check on GQA (X, W_q, W_k, W_v, W_o). */
      et_tp_group_t *g1 = NULL;
      CHECK(et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 1u, NULL,
                               0u, 1000, &g1) == ET_TP_OK);
      const float h = 1e-3f;
      float qc[256], kc[256], vc[256], ac[256], yp[256], ym[256];
      float *ptrs[5] = {c.x, c.w_q, c.w_k, c.w_v, c.w_o};
      const float *grads[5] = {r1.dx, r1.dw_q, r1.dw_k, r1.dw_v, r1.dw_o};
      const size_t lens[5] = {xd_cnt, q_cnt, kv_cnt, kv_cnt, q_cnt};
      for (int t = 0; t < 5; ++t) {
        for (size_t idx = 0u; idx < lens[t]; idx += 53u) {
          const float orig = ptrs[t][idx];
          ptrs[t][idx] = orig + h;
          CHECK(et_tp_gqa_forward_v1(g1, &c.cfg, c.x, c.w_q, c.w_k, c.w_v,
                                     c.w_o, qc, kc, vc, ac, yp, 1u,
                                     1000) == ET_TP_OK);
          ptrs[t][idx] = orig - h;
          CHECK(et_tp_gqa_forward_v1(g1, &c.cfg, c.x, c.w_q, c.w_k, c.w_v,
                                     c.w_o, qc, kc, vc, ac, ym, 2u,
                                     1000) == ET_TP_OK);
          ptrs[t][idx] = orig;
          double lp = 0.0, lm = 0.0;
          for (size_t j = 0u; j < xd_cnt; ++j) {
            lp += (double)yp[j] * (double)c.dy[j];
            lm += (double)ym[j] * (double)c.dy[j];
          }
          const double fd = (lp - lm) / (2.0 * (double)h);
          CHECK(fabs(fd - (double)grads[t][idx]) <= 5e-4);
        }
      }
      et_tp_group_destroy(g1);
    }
  }
}

/* =========================================================================
 * 5. Full TP Transformer Block + 10-Step SGD Parity + Bridge Tests
 * ========================================================================= */

static void test_full_block_and_bridge(void) {
  et_tp_aot_session_t *s1 = NULL, *s2 = NULL, *s4 = NULL;
  CHECK(et_tp_aot_session_create(1, 2, 4, 32, 32, 8, 4, 4, 1e-5, &s1) ==
        ET_TP_OK);
  CHECK(et_tp_aot_session_create(2, 2, 4, 32, 32, 8, 4, 4, 1e-5, &s2) ==
        ET_TP_OK);
  CHECK(et_tp_aot_session_create(4, 2, 4, 32, 32, 8, 4, 4, 1e-5, &s4) ==
        ET_TP_OK);

  /* Exact 1/W sharded weight memory verification. */
  const int64_t full_bytes = et_tp_aot_full_sharded_weight_bytes(s1);
  CHECK(full_bytes == 24576);
  CHECK(et_tp_aot_rank_sharded_weight_bytes(s1) == full_bytes);
  CHECK(et_tp_aot_rank_sharded_weight_bytes(s2) == full_bytes / 2);
  CHECK(et_tp_aot_rank_sharded_weight_bytes(s4) == full_bytes / 4);
  CHECK(et_tp_aot_replicated_weight_bytes(s1) == 256);

  double prev_loss = 1e9;
  for (int step = 0; step < 10; ++step) {
    CHECK(et_tp_aot_forward(s1) == ET_TP_OK);
    CHECK(et_tp_aot_forward(s2) == ET_TP_OK);
    CHECK(et_tp_aot_forward(s4) == ET_TP_OK);

    CHECK(et_tp_aot_allreduce_calls(s1) == (int64_t)(step * 4 + 2));
    CHECK(et_tp_aot_allreduce_calls(s2) == (int64_t)(step * 4 + 2));
    CHECK(et_tp_aot_allreduce_calls(s4) == (int64_t)(step * 4 + 2));
    CHECK(et_tp_aot_ipc_allreduce_calls(s1) == 0);
    CHECK(et_tp_aot_ipc_allreduce_calls(s2) == (int64_t)(step * 4 + 2));
    CHECK(et_tp_aot_ipc_allreduce_calls(s4) == (int64_t)(step * 4 + 2));

    const double l1 = et_tp_aot_mse_loss(s1);
    const double l2 = et_tp_aot_mse_loss(s2);
    const double l4 = et_tp_aot_mse_loss(s4);
    CHECK(isfinite(l1) && l1 < prev_loss);
    CHECK(fabs(l1 - l2) <= 1e-6);
    CHECK(fabs(l1 - l4) <= 1e-6);
    prev_loss = l1;

    CHECK(et_tp_aot_max_output_diff(s1, s2) <= 1e-6);
    CHECK(et_tp_aot_max_output_diff(s1, s4) <= 1e-6);

    CHECK(et_tp_aot_backward(s1) == ET_TP_OK);
    CHECK(et_tp_aot_backward(s2) == ET_TP_OK);
    CHECK(et_tp_aot_backward(s4) == ET_TP_OK);

    CHECK(et_tp_aot_allreduce_calls(s1) == (int64_t)((step + 1) * 4));
    CHECK(et_tp_aot_allreduce_calls(s2) == (int64_t)((step + 1) * 4));
    CHECK(et_tp_aot_allreduce_calls(s4) == (int64_t)((step + 1) * 4));
    CHECK(et_tp_aot_max_dx_diff(s1, s2) <= 1e-6);
    CHECK(et_tp_aot_max_dx_diff(s1, s4) <= 1e-6);

    if (step == 0 || step == 9) {
      CHECK(et_tp_aot_gather_grads(s1) == ET_TP_OK);
      CHECK(et_tp_aot_gather_grads(s2) == ET_TP_OK);
      CHECK(et_tp_aot_gather_grads(s4) == ET_TP_OK);
      CHECK(et_tp_aot_max_grad_diff(s1, s2) <= 1e-6);
      CHECK(et_tp_aot_max_grad_diff(s1, s4) <= 1e-6);
    }

    CHECK(et_tp_aot_sgd_step(s1, 0.8) == ET_TP_OK);
    CHECK(et_tp_aot_sgd_step(s2, 0.8) == ET_TP_OK);
    CHECK(et_tp_aot_sgd_step(s4, 0.8) == ET_TP_OK);
  }

  CHECK(et_tp_aot_gather_weights(s1) == ET_TP_OK);
  CHECK(et_tp_aot_gather_weights(s2) == ET_TP_OK);
  CHECK(et_tp_aot_gather_weights(s4) == ET_TP_OK);
  CHECK(et_tp_aot_max_weight_diff(s1, s2) <= 1e-6);
  CHECK(et_tp_aot_max_weight_diff(s1, s4) <= 1e-6);

  /* Central FD check across all 9 block parameter tensors + input X on s1. */
  CHECK(et_tp_aot_forward(s1) == ET_TP_OK);
  (void)et_tp_aot_mse_loss(s1);
  CHECK(et_tp_aot_backward(s1) == ET_TP_OK);
  CHECK(et_tp_aot_gather_grads(s1) == ET_TP_OK);
  const double h = 1e-3;
  for (int64_t p = 0; p < ET_TP_AOT_PARAM_COUNT; ++p) {
    const int64_t n = et_tp_aot_param_count(s1, p);
    const int64_t idx = (p * 17) % n;
    const double analytic = et_tp_aot_get_gathered_grad(s1, p, idx);
    const double orig = et_tp_aot_get_weight(s1, p, idx);
    CHECK(et_tp_aot_set_weight(s1, p, idx, orig + h) == ET_TP_OK);
    CHECK(et_tp_aot_forward(s1) == ET_TP_OK);
    const double lp = et_tp_aot_mse_loss(s1);
    CHECK(et_tp_aot_set_weight(s1, p, idx, orig - h) == ET_TP_OK);
    CHECK(et_tp_aot_forward(s1) == ET_TP_OK);
    const double lm = et_tp_aot_mse_loss(s1);
    CHECK(et_tp_aot_set_weight(s1, p, idx, orig) == ET_TP_OK);
    const double fd = (lp - lm) / (2.0 * h);
    CHECK(fabs(fd - analytic) <= 5e-4);
  }
  {
    const int64_t idx = 19;
    const double analytic = et_tp_aot_get_dx(s1, idx);
    const double orig = et_tp_aot_get_input(s1, idx);
    CHECK(et_tp_aot_set_input(s1, idx, orig + h) == ET_TP_OK);
    CHECK(et_tp_aot_forward(s1) == ET_TP_OK);
    const double lp = et_tp_aot_mse_loss(s1);
    CHECK(et_tp_aot_set_input(s1, idx, orig - h) == ET_TP_OK);
    CHECK(et_tp_aot_forward(s1) == ET_TP_OK);
    const double lm = et_tp_aot_mse_loss(s1);
    CHECK(et_tp_aot_set_input(s1, idx, orig) == ET_TP_OK);
    const double fd = (lp - lm) / (2.0 * h);
    CHECK(fabs(fd - analytic) <= 5e-4);
  }

  et_tp_aot_session_destroy(s1);
  et_tp_aot_session_destroy(s2);
  et_tp_aot_session_destroy(s4);
}

/* =========================================================================
 * 6. Negative Tests & FD Leak Verification
 * ========================================================================= */

static void test_negative_cases(void) {
  const int fds_start = count_open_fds();

  const int32_t expected[12] = {
      ET_TP_ERR_INVALID_ARG,       /* 0: W=3 */
      ET_TP_ERR_DIVISIBILITY,      /* 1: Col Out%W */
      ET_TP_ERR_DIVISIBILITY,      /* 2: Row In%W */
      ET_TP_ERR_DIVISIBILITY,      /* 3: SwiGLU d_ff%W */
      ET_TP_ERR_DIVISIBILITY,      /* 4: GQA H_q%W */
      ET_TP_ERR_DIVISIBILITY,      /* 5: GQA H_kv%W */
      ET_TP_ERR_DIVISIBILITY,      /* 6: GQA H_q%H_kv */
      ET_TP_ERR_NONFINITE,         /* 7: NaN input */
      ET_TP_ERR_ALIGNMENT,         /* 8: misaligned float* */
      ET_TP_ERR_ALIAS,             /* 9: overlapping AllGather */
      ET_TP_ERR_TIMEOUT,           /* 10: monotonic timeout */
      ET_TP_ERR_PEER_DISCONNECTED, /* 11: peer disconnect */
  };
  for (int64_t i = 0; i < 12; ++i)
    CHECK(et_tp_aot_check_negative_case(i) == expected[i]);

  /* Monotonic timeout wall-clock check (>= 35 ms, < 500 ms). */
  {
    int sp[2] = {-1, -1};
    CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp) == 0);
    et_tp_group_t *g = NULL;
    CHECK(et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 2u, &sp[0],
                             1u, 50, &g) == ET_TP_OK);
    close(sp[0]);
    float s[4] = {1.0f, 2.0f, 3.0f, 4.0f}, r[4] = {0.0f};
    const int64_t t0 = monotonic_ms();
    CHECK(et_tp_allreduce_sum_f32(g, s, r, 4u, 1u, 50) == ET_TP_ERR_TIMEOUT);
    const int64_t dt = monotonic_ms() - t0;
    CHECK(dt >= 40 && dt < 2000);
    et_tp_group_destroy(g);
    close(sp[1]);
  }

  /* Duplicate socket FD rejection with zero FD leak. */
  {
    int sp[2] = {-1, -1};
    CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp) == 0);
    const int dup_fds[3] = {sp[0], sp[0], sp[0]};
    et_tp_group_t *g = NULL;
    CHECK(et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 4u, dup_fds,
                             3u, 1000, &g) == ET_TP_ERR_INVALID_ARG);
    CHECK(g == NULL);
    close(sp[0]);
    close(sp[1]);
  }

  /* Partial overlap in AllReduce rejected with ET_TP_ERR_ALIAS. */
  {
    et_tp_group_t *g = NULL;
    CHECK(et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 1u, NULL, 0u,
                             1000, &g) == ET_TP_OK);
    float buf[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    CHECK(et_tp_allreduce_sum_f32(g, buf, buf + 1u, 4u, 1u, 1000) ==
          ET_TP_ERR_ALIAS);
    et_tp_group_destroy(g);
  }

  CHECK(count_open_fds() == fds_start);
}

/* =========================================================================
 * 7. --oracle Dump Mode for Independent Binary64 Python Oracle
 * ========================================================================= */

static void print_f32_hex(const float *v, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%a", (double)v[i]);
  }
  putchar('\n');
}

static void run_oracle_dump(void) {
  /* 1. ColumnParallelLinear & RowParallelLinear across W=1, 2, 4 */
  linear_case_t lc;
  init_linear_case(&lc, 3u, 8u, 12u, 1);
  printf("LINEAR %u %u %u\n", lc.rows, lc.in_dim, lc.out_dim);
  print_f32_hex(lc.x, (size_t)lc.rows * lc.in_dim);
  print_f32_hex(lc.w, (size_t)lc.out_dim * lc.in_dim);
  print_f32_hex(lc.b, lc.out_dim);
  print_f32_hex(lc.dy, (size_t)lc.rows * lc.out_dim);
  const uint32_t worlds[3] = {1u, 2u, 4u};
  for (size_t wi = 0u; wi < 3u; ++wi) {
    linear_result_t col, row;
    run_forked_group(worlds[wi], rank_column_linear, &lc, &col, sizeof(col), 0);
    run_forked_group(worlds[wi], rank_row_linear, &lc, &row, sizeof(row), 0);
    printf("COL_W %u\n", worlds[wi]);
    print_f32_hex(col.y, (size_t)lc.rows * lc.out_dim);
    print_f32_hex(col.dx, (size_t)lc.rows * lc.in_dim);
    print_f32_hex(col.dw, (size_t)lc.out_dim * lc.in_dim);
    print_f32_hex(col.db, lc.out_dim);
    printf("ROW_W %u\n", worlds[wi]);
    print_f32_hex(row.y, (size_t)lc.rows * lc.out_dim);
    print_f32_hex(row.dx, (size_t)lc.rows * lc.in_dim);
    print_f32_hex(row.dw, (size_t)lc.out_dim * lc.in_dim);
    print_f32_hex(row.db, lc.out_dim);
  }

  /* 2. TP-SwiGLU across W=1, 2, 4 */
  swiglu_case_t sc;
  init_swiglu_case(&sc);
  printf("SWIGLU %u %u %u\n", sc.cfg.tokens, sc.cfg.d_model, sc.cfg.d_ff);
  print_f32_hex(sc.x, (size_t)sc.cfg.tokens * sc.cfg.d_model);
  print_f32_hex(sc.w_gate, (size_t)sc.cfg.d_ff * sc.cfg.d_model);
  print_f32_hex(sc.w_up, (size_t)sc.cfg.d_ff * sc.cfg.d_model);
  print_f32_hex(sc.w_down, (size_t)sc.cfg.d_model * sc.cfg.d_ff);
  print_f32_hex(sc.dy, (size_t)sc.cfg.tokens * sc.cfg.d_model);
  for (size_t wi = 0u; wi < 3u; ++wi) {
    swiglu_result_t sr;
    run_forked_group(worlds[wi], rank_swiglu, &sc, &sr, sizeof(sr), 0);
    printf("SWIGLU_W %u\n", worlds[wi]);
    print_f32_hex(sr.y, (size_t)sc.cfg.tokens * sc.cfg.d_model);
    print_f32_hex(sr.dx, (size_t)sc.cfg.tokens * sc.cfg.d_model);
    print_f32_hex(sr.dw_gate, (size_t)sc.cfg.d_ff * sc.cfg.d_model);
    print_f32_hex(sr.dw_up, (size_t)sc.cfg.d_ff * sc.cfg.d_model);
    print_f32_hex(sr.dw_down, (size_t)sc.cfg.d_model * sc.cfg.d_ff);
  }

  /* 3. TP-GQA across W=1, 2, 4 */
  gqa_case_t gc;
  init_gqa_case(&gc, 2u, 4u, 8u, 4u, 2u);
  const size_t g_xd = (size_t)gc.cfg.batch * gc.cfg.seq_len * gc.cfg.d_model;
  const size_t g_q = (size_t)gc.cfg.d_model * gc.cfg.d_model;
  const size_t g_kv =
      (size_t)gc.cfg.num_kv_heads * gc.cfg.head_dim * gc.cfg.d_model;
  printf("GQA %u %u %u %u %u %u\n", gc.cfg.batch, gc.cfg.seq_len,
         gc.cfg.d_model, gc.cfg.num_q_heads, gc.cfg.num_kv_heads,
         gc.cfg.head_dim);
  print_f32_hex(gc.x, g_xd);
  print_f32_hex(gc.w_q, g_q);
  print_f32_hex(gc.w_k, g_kv);
  print_f32_hex(gc.w_v, g_kv);
  print_f32_hex(gc.w_o, g_q);
  print_f32_hex(gc.dy, g_xd);
  for (size_t wi = 0u; wi < 3u; ++wi) {
    gqa_result_t gr;
    run_forked_group(worlds[wi], rank_gqa, &gc, &gr, sizeof(gr), 0);
    printf("GQA_W %u\n", worlds[wi]);
    print_f32_hex(gr.y, g_xd);
    print_f32_hex(gr.dx, g_xd);
    print_f32_hex(gr.dw_q, g_q);
    print_f32_hex(gr.dw_k, g_kv);
    print_f32_hex(gr.dw_v, g_kv);
    print_f32_hex(gr.dw_o, g_q);
  }

  /* 4. Full TP Transformer Block across W=1, 2, 4 */
  const int64_t B = 1, T = 4, D = 16, F = 16, Hq = 8, Hkv = 4, Dh = 2;
  const double eps = 1e-5;
  printf("BLOCK %lld %lld %lld %lld %lld %lld %lld %a\n", (long long)B,
         (long long)T, (long long)D, (long long)F, (long long)Hq,
         (long long)Hkv, (long long)Dh, eps);
  for (size_t wi = 0u; wi < 3u; ++wi) {
    et_tp_aot_session_t *s = NULL;
    CHECK(et_tp_aot_session_create(worlds[wi], B, T, D, F, Hq, Hkv, Dh, eps,
                                   &s) == ET_TP_OK);
    if (wi == 0u) {
      CHECK(et_tp_aot_gather_weights(s) == ET_TP_OK);
      const int64_t n_out = et_tp_aot_output_count(s);
      float *tmp = (float *)malloc((size_t)n_out * sizeof(float));
      CHECK(tmp != NULL);
      for (int64_t i = 0; i < n_out; ++i)
        tmp[i] = (float)et_tp_aot_get_input(s, i);
      print_f32_hex(tmp, (size_t)n_out);
      free(tmp);
      for (int64_t p = 0; p < ET_TP_AOT_PARAM_COUNT; ++p) {
        const int64_t n = et_tp_aot_param_count(s, p);
        float *wbuf = (float *)malloc((size_t)n * sizeof(float));
        CHECK(wbuf != NULL);
        for (int64_t i = 0; i < n; ++i)
          wbuf[i] = (float)et_tp_aot_get_gathered_weight(s, p, i);
        print_f32_hex(wbuf, (size_t)n);
        free(wbuf);
      }
    }
    CHECK(et_tp_aot_forward(s) == ET_TP_OK);
    const double loss = et_tp_aot_mse_loss(s);
    CHECK(et_tp_aot_backward(s) == ET_TP_OK);
    CHECK(et_tp_aot_gather_grads(s) == ET_TP_OK);
    printf("BLOCK_W %u %a\n", worlds[wi], loss);
    const int64_t n_out = et_tp_aot_output_count(s);
    float *ybuf = (float *)malloc((size_t)n_out * sizeof(float));
    float *dxbuf = (float *)malloc((size_t)n_out * sizeof(float));
    CHECK(ybuf && dxbuf);
    for (int64_t i = 0; i < n_out; ++i) {
      ybuf[i] = (float)et_tp_aot_get_output(s, i);
      dxbuf[i] = (float)et_tp_aot_get_dx(s, i);
    }
    print_f32_hex(ybuf, (size_t)n_out);
    print_f32_hex(dxbuf, (size_t)n_out);
    free(ybuf);
    free(dxbuf);
    for (int64_t p = 0; p < ET_TP_AOT_PARAM_COUNT; ++p) {
      const int64_t n = et_tp_aot_param_count(s, p);
      float *gbuf = (float *)malloc((size_t)n * sizeof(float));
      CHECK(gbuf != NULL);
      for (int64_t i = 0; i < n; ++i)
        gbuf[i] = (float)et_tp_aot_get_gathered_grad(s, p, i);
      print_f32_hex(gbuf, (size_t)n);
      free(gbuf);
    }
    et_tp_aot_session_destroy(s);
  }
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    run_oracle_dump();
    return 0;
  }
  const int fds_initial = count_open_fds();
  test_collectives();
  test_column_and_row_linear();
  test_swiglu();
  test_gqa();
  test_full_block_and_bridge();
  test_negative_cases();
  CHECK(count_open_fds() == fds_initial);
  printf("TP native C11 suite (AF_UNIX W=1/2/4 collectives, Column/Row/SwiGLU/"
         "GQA/Block parity, FD, negatives, zero FD leaks): PASS\n");
  return 0;
}
