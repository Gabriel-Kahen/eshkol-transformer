#define _GNU_SOURCE
#include "dist5_eshkol_bridge.h"

#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,       \
              #cond);                                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

enum { TEST_TIMEOUT_MS = 10000 };

static int count_open_fds(void) {
  DIR *dir = opendir("/proc/self/fd");
  CHECK(dir != NULL);
  int count = 0;
  struct dirent *ent;
  while ((ent = readdir(dir)) != NULL) {
    if (ent->d_name[0] != '.')
      ++count;
  }
  closedir(dir);
  return count;
}

static void make_temp_sock_dir(char *buf, size_t cap) {
  CHECK(cap >= 32u);
  snprintf(buf, cap, "/tmp/dist5_br_XXXXXX");
  CHECK(mkdtemp(buf) != NULL);
}

static void remove_sock_dir(const char *dir) {
  char path[256];
  snprintf(path, sizeof(path), "%s/dist5_r0.sock", dir);
  (void)unlink(path);
  (void)rmdir(dir);
}

static void test_open_negative_and_fd_leaks(void) {
  const int fds_before = count_open_fds();
  char sock_dir[64];
  make_temp_sock_dir(sock_dir, sizeof(sock_dir));

  et_dist5_aot_handle_t *h = (et_dist5_aot_handle_t *)0x1;
  CHECK(et_dist5_aot_open(sock_dir, 0, 1, 1000, NULL) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_open(sock_dir, 0, 0, 1000, &h) ==
            ET_DIST5_ERR_INVALID_ARG &&
        h == NULL);
  CHECK(et_dist5_aot_open(sock_dir, 0, ET_DIST5_MAX_WORLD_SIZE + 1, 1000, &h) ==
            ET_DIST5_ERR_INVALID_ARG &&
        h == NULL);
  CHECK(et_dist5_aot_open(sock_dir, -1, 2, 1000, &h) ==
            ET_DIST5_ERR_INVALID_ARG &&
        h == NULL);
  CHECK(et_dist5_aot_open(sock_dir, 2, 2, 1000, &h) ==
            ET_DIST5_ERR_INVALID_ARG &&
        h == NULL);
  CHECK(et_dist5_aot_open(sock_dir, 0, 1, 0, &h) == ET_DIST5_ERR_INVALID_ARG &&
        h == NULL);
  CHECK(et_dist5_aot_open(sock_dir, 0, 1, -5, &h) == ET_DIST5_ERR_INVALID_ARG &&
        h == NULL);
  CHECK(et_dist5_aot_open(NULL, 0, 2, 1000, &h) == ET_DIST5_ERR_INVALID_ARG &&
        h == NULL);
  CHECK(et_dist5_aot_open("", 0, 2, 1000, &h) == ET_DIST5_ERR_INVALID_ARG &&
        h == NULL);
  CHECK(et_dist5_aot_open("/tmp/definitely_missing_dist5_dir_999999", 0, 2,
                          1000, &h) == ET_DIST5_ERR_INVALID_ARG &&
        h == NULL);

  /* Overlong socket path must be rejected explicitly (never truncated). */
  char long_dir[160];
  memset(long_dir, 'a', sizeof(long_dir) - 1u);
  long_dir[0] = '/';
  long_dir[sizeof(long_dir) - 1u] = '\0';
  CHECK(et_dist5_aot_open(long_dir, 0, 2, 1000, &h) ==
            ET_DIST5_ERR_INVALID_ARG &&
        h == NULL);

  /* Timeout waiting for peer (both rank 0 and rank 1 paths). */
  CHECK(et_dist5_aot_open(sock_dir, 0, 2, 50, &h) == ET_DIST5_ERR_TIMEOUT &&
        h == NULL);
  CHECK(et_dist5_aot_open(sock_dir, 1, 2, 50, &h) == ET_DIST5_ERR_TIMEOUT &&
        h == NULL);

  /* Metadata queries on NULL. */
  CHECK(et_dist5_aot_rank(NULL) == -1);
  CHECK(et_dist5_aot_world_size(NULL) == -1);
  et_dist5_aot_close(NULL);

  remove_sock_dir(sock_dir);
  CHECK(count_open_fds() == fds_before);
}

static void test_buffer_helpers_and_alignment(void) {
  CHECK(et_dist5_aot_alloc_f32(0) == NULL);
  CHECK(et_dist5_aot_alloc_f32(-4) == NULL);
  CHECK(et_dist5_aot_alloc_f32((int64_t)(ET_DIST5_MAX_BUFFER_BYTES / 4u) + 1) ==
        NULL);
  et_dist5_aot_free_f32(NULL);

  void *a = et_dist5_aot_alloc_f32(4);
  void *b = et_dist5_aot_alloc_f32(4);
  CHECK(a != NULL && b != NULL);

  /* Known SHA-256 vector for 4 zero bytes (one f32 +0.0f). */
  char hex[65];
  memset(hex, 'Z', sizeof(hex));
  CHECK(et_dist5_aot_sha256_hex_f32(a, 1, hex, sizeof(hex)) == ET_DIST5_OK);
  CHECK(strcmp(hex,
               "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119") ==
        0);
  CHECK(et_dist5_aot_read_byte(hex, 65, 0) == 'd');
  CHECK(et_dist5_aot_read_byte(hex, 65, 63) == '9');
  CHECK(et_dist5_aot_read_byte(hex, 65, -1) == -1);
  CHECK(et_dist5_aot_read_byte(hex, 65, 65) == -1);
  CHECK(et_dist5_aot_read_byte(NULL, 65, 0) == -1);

  /* Store/load round-trips including -0.0, NaN/Inf clamping, and bit patterns. */
  CHECK(et_dist5_aot_store_f32(a, 4, 0, -0.0) == ET_DIST5_OK);
  CHECK(et_dist5_aot_load_f32_bits(a, 4, 0) == INT64_C(0x80000000));
  CHECK(et_dist5_aot_store_f32(a, 4, 1, 1.5) == ET_DIST5_OK);
  CHECK(et_dist5_aot_load_f32(a, 4, 1) == 1.5);
  CHECK(et_dist5_aot_store_f32(a, 4, 2, DBL_MAX) == ET_DIST5_OK);
  CHECK(et_dist5_aot_load_f32_bits(a, 4, 2) == INT64_C(0x7f800000));
  CHECK(et_dist5_aot_store_f32_bits(a, 4, 3, INT64_C(0x7fc00000)) ==
        ET_DIST5_OK);
  CHECK(et_dist5_aot_load_f32_bits(a, 4, 3) == INT64_C(0x7fc00000));
  CHECK(et_dist5_aot_equal_bits_f32(a, b, 4) == 0);

  /* Deterministic f32 accumulation and scalar division helpers. */
  for (int64_t i = 0; i < 4; ++i) {
    CHECK(et_dist5_aot_store_f32(a, 4, i, (double)(i + 1)) == ET_DIST5_OK);
    CHECK(et_dist5_aot_store_f32(b, 4, i, (double)(2 * i + 1)) == ET_DIST5_OK);
  }
  CHECK(et_dist5_aot_f32_add_inplace(a, b, 4) == ET_DIST5_OK);
  CHECK(et_dist5_aot_f32_div_scalar_inplace(a, 4, 2.0) == ET_DIST5_OK);
  CHECK(et_dist5_aot_load_f32(a, 4, 0) == 1.0);
  CHECK(et_dist5_aot_load_f32(a, 4, 1) == 2.5);
  CHECK(et_dist5_aot_load_f32(a, 4, 2) == 4.0);
  CHECK(et_dist5_aot_load_f32(a, 4, 3) == 5.5);

  /* Non-finite and invalid-arg guards leave dst untouched. */
  CHECK(et_dist5_aot_f32_div_scalar_inplace(a, 4, 0.0) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_f32_div_scalar_inplace(a, 4, NAN) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_store_f32_bits(b, 4, 2, INT64_C(0x7fc00000)) ==
        ET_DIST5_OK);
  CHECK(et_dist5_aot_f32_add_inplace(a, b, 4) == ET_DIST5_ERR_NONFINITE);
  CHECK(et_dist5_aot_load_f32(a, 4, 0) == 1.0);
  CHECK(et_dist5_aot_load_f32(a, 4, 3) == 5.5);

  /* Misaligned pointer negative checks (via void * — zero UB). */
  unsigned char raw[32];
  memset(raw, 0xa5, sizeof(raw));
  unsigned char before[32];
  memcpy(before, raw, sizeof(raw));
  void *mis1 = raw + 1;
  void *mis2 = raw + 2;

  CHECK(et_dist5_aot_store_f32(NULL, 4, 0, 1.0) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_store_f32(mis1, 4, 0, 1.0) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_store_f32(mis2, 4, 0, 1.0) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_store_f32(a, 4, -1, 1.0) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_store_f32(a, 4, 4, 1.0) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(isnan(et_dist5_aot_load_f32(NULL, 4, 0)));
  CHECK(isnan(et_dist5_aot_load_f32(mis1, 4, 0)));
  CHECK(isnan(et_dist5_aot_load_f32(a, 4, 4)));
  CHECK(et_dist5_aot_store_f32_bits(mis1, 4, 0, 0) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_store_f32_bits(a, 4, 0, -1) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_store_f32_bits(a, 4, 0, INT64_C(0x100000000)) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_load_f32_bits(mis1, 4, 0) == -1);
  CHECK(et_dist5_aot_equal_bits_f32(mis1, a, 4) == -1);
  CHECK(et_dist5_aot_equal_bits_f32(a, NULL, 4) == -1);
  CHECK(et_dist5_aot_equal_bits_f32(a, b, 0) == -1);
  CHECK(et_dist5_aot_sha256_hex_f32(mis1, 4, hex, sizeof(hex)) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_sha256_hex_f32(a, 4, hex, 64) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_sha256_hex_f32(a, 4, NULL, 65) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_f32_add_inplace(mis1, b, 4) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_f32_div_scalar_inplace(mis2, 4, 2.0) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(memcmp(raw, before, sizeof(raw)) == 0);

  et_dist5_aot_free_f32(a);
  et_dist5_aot_free_f32(b);
}

static void test_w1_negatives_and_mod5(void) {
  et_dist5_aot_handle_t *h = NULL;
  CHECK(et_dist5_aot_open(NULL, 0, 1, TEST_TIMEOUT_MS, &h) == ET_DIST5_OK &&
        h != NULL);

  unsigned char raw[32];
  memset(raw, 0x5a, sizeof(raw));
  float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float dst[4] = {9.0f, 9.0f, 9.0f, 9.0f};
  float dst_before[4];
  memcpy(dst_before, dst, sizeof(dst));

  /* Null handle, negative tag, nonpositive timeout. */
  CHECK(et_dist5_aot_barrier(NULL, 0, 1000) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_barrier(h, -1, 1000) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_barrier(h, 0, 0) == ET_DIST5_ERR_INVALID_ARG);

  /* Null/misaligned/invalid collective arguments on W=1 leave dst intact. */
  CHECK(et_dist5_aot_broadcast_f32(h, raw + 1, 4, 0, 1, 1000) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_broadcast_f32(h, dst, 4, 1, 2, 1000) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_allreduce_f32(h, src, raw + 1, 4, ET_DIST5_SUM, 3, 1000) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_allreduce_f32(h, src, dst, 4, 99, 4, 1000) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_allgather_f32(h, src, 0, dst, 5, 1000) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_reduce_scatter_f32(h, src, dst, -1, ET_DIST5_SUM, 6,
                                        1000) == ET_DIST5_ERR_INVALID_ARG);
  CHECK(et_dist5_aot_sync_gradients_f32(h, raw + 1, dst, 4, 7, 1000) ==
        ET_DIST5_ERR_INVALID_ARG);

  /* Non-finite sync_gradients_f32 on W=1 leaves dst intact and stays usable. */
  src[2] = NAN;
  CHECK(et_dist5_aot_sync_gradients_f32(h, src, dst, 4, 8, 1000) ==
        ET_DIST5_ERR_NONFINITE);
  CHECK(memcmp(dst, dst_before, sizeof(dst)) == 0);
  src[2] = 3.0f;
  CHECK(et_dist5_aot_sync_gradients_f32(h, src, dst, 4, 9, 1000) ==
        ET_DIST5_OK);
  CHECK(et_dist5_aot_equal_bits_f32(src, dst, 4) == 1);

  /* MOD5 K1 negative checks (null handle, misaligned pointer, bad kv_heads). */
  float x[128] = {0}, gamma[16] = {0}, y[128] = {0}, eps = 0.125f;
  float gate[256] = {0}, up[256] = {0}, glu_out[256] = {0};
  float q[128] = {0}, k[32] = {0}, v[32] = {0}, slopes[4] = {0};
  int64_t qpos[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int64_t kpos[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t mask[64];
  memset(mask, 1, sizeof(mask));
  CHECK(et_dist5_aot_mod5_rms_forward(NULL, x, gamma, &eps, y) != 0);
  CHECK(et_dist5_aot_mod5_rms_forward(h, raw + 1, gamma, &eps, y) != 0);
  CHECK(et_dist5_aot_mod5_swiglu_forward(h, gate, up, glu_out) == 0);
  CHECK(et_dist5_aot_mod5_swiglu_forward(h, raw + 1, up, glu_out) != 0);
  CHECK(et_dist5_aot_mod5_attention_forward(h, 3, q, k, v, qpos, kpos, mask,
                                            slopes, 4, y) != 0);
  CHECK(et_dist5_aot_mod5_attention_forward(h, 1, q, k, v, qpos, kpos, mask,
                                            slopes, 4, y) == 0);

  et_dist5_aot_close(h);
}

typedef void (*rank_fn)(const char *sock_dir, int64_t rank, int64_t world);

static void run_multiprocess(int64_t world, rank_fn fn, int reverse_start) {
  char sock_dir[64];
  make_temp_sock_dir(sock_dir, sizeof(sock_dir));

  pid_t pids[ET_DIST5_MAX_WORLD_SIZE];
  for (int64_t i = 0; i < world; ++i) {
    const int64_t r = reverse_start ? (world - 1 - i) : i;
    const pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) {
      if (reverse_start && r == 0) {
        /* Delay coordinator slightly so workers exercise retry-connect. */
        struct timespec ts = {0, 20000000};
        nanosleep(&ts, NULL);
      }
      fn(sock_dir, r, world);
      _exit(0);
    }
    pids[r] = pid;
  }

  for (int64_t r = 0; r < world; ++r) {
    int wstatus = 0;
    while (waitpid(pids[r], &wstatus, 0) < 0) {
      CHECK(errno == EINTR);
    }
    CHECK(WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 0);
  }
  remove_sock_dir(sock_dir);
}

static void worker_all_collectives_and_nonfinite(const char *sock_dir,
                                                 int64_t rank, int64_t world) {
  const int fds_before = count_open_fds();
  et_dist5_aot_handle_t *h = NULL;
  CHECK(et_dist5_aot_open(sock_dir, rank, world, TEST_TIMEOUT_MS, &h) ==
            ET_DIST5_OK &&
        h != NULL);
  CHECK(et_dist5_aot_rank(h) == rank);
  CHECK(et_dist5_aot_world_size(h) == world);

  /* Barrier. */
  CHECK(et_dist5_aot_barrier(h, 1, TEST_TIMEOUT_MS) == ET_DIST5_OK);

  /* Broadcast from rank 0. */
  float bcast[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (rank == 0) {
    bcast[0] = 1.25f;
    bcast[1] = -2.5f;
    bcast[2] = 3.75f;
    bcast[3] = -0.0f;
  }
  CHECK(et_dist5_aot_broadcast_f32(h, bcast, 4, 0, 2, TEST_TIMEOUT_MS) ==
        ET_DIST5_OK);
  CHECK(et_dist5_aot_load_f32(bcast, 4, 0) == 1.25);
  CHECK(et_dist5_aot_load_f32(bcast, 4, 1) == -2.5);
  CHECK(et_dist5_aot_load_f32(bcast, 4, 2) == 3.75);
  CHECK(et_dist5_aot_load_f32_bits(bcast, 4, 3) == INT64_C(0x80000000));

  /* Canonical ascending rank-order allreduce:
   * With W=4: rank0=16777216.0f, rank1=1.0f, rank2=-16777216.0f, rank3=2.0f.
   * Ascending order: (((16777216.0f + 1.0f) + -16777216.0f) + 2.0f) == 2.0f!
   * Reversed order would give 3.0f. */
  if (world == 4) {
    const float vals[4] = {16777216.0f, 1.0f, -16777216.0f, 2.0f};
    float in_v = vals[rank], out_v = -99.0f;
    CHECK(et_dist5_aot_allreduce_f32(h, &in_v, &out_v, 1, ET_DIST5_SUM, 3,
                                     TEST_TIMEOUT_MS) == ET_DIST5_OK);
    CHECK(out_v == 2.0f);
  }

  /* Allgather. */
  float send_ag[2] = {(float)(rank * 10 + 1), (float)(rank * 10 + 2)};
  float recv_ag[8] = {0};
  CHECK(et_dist5_aot_allgather_f32(h, send_ag, 2, recv_ag, 4,
                                   TEST_TIMEOUT_MS) == ET_DIST5_OK);
  for (int64_t r = 0; r < world; ++r) {
    CHECK(recv_ag[2 * r + 0] == (float)(r * 10 + 1));
    CHECK(recv_ag[2 * r + 1] == (float)(r * 10 + 2));
  }

  /* Reduce-scatter. */
  float send_rs[8];
  for (int64_t i = 0; i < world * 2; ++i)
    send_rs[i] = (float)(rank + 1) * (float)(i + 1);
  float recv_rs[2] = {0};
  CHECK(et_dist5_aot_reduce_scatter_f32(h, send_rs, recv_rs, 2, ET_DIST5_SUM, 5,
                                        TEST_TIMEOUT_MS) == ET_DIST5_OK);
  const float rank_sum = (float)(world * (world + 1) / 2);
  CHECK(recv_rs[0] == rank_sum * (float)(rank * 2 + 1));
  CHECK(recv_rs[1] == rank_sum * (float)(rank * 2 + 2));

  /* Coordinated misaligned-pointer rejection on rank 1 without deadlock. */
  unsigned char raw[32];
  memset(raw, 0x33, sizeof(raw));
  float dst[4] = {7.0f, 7.0f, 7.0f, 7.0f};
  float dst_snap[4];
  memcpy(dst_snap, dst, sizeof(dst));
  CHECK(et_dist5_aot_sync_gradients_f32(h, rank == 1 ? (void *)(raw + 1)
                                                     : (void *)bcast,
                                        dst, 4, 6, TEST_TIMEOUT_MS) ==
        ET_DIST5_ERR_INVALID_ARG);
  CHECK(memcmp(dst, dst_snap, sizeof(dst)) == 0);

  /* Atomic non-finite abort on rank world-1 (NaN and +Inf). */
  float grad[4] = {(float)(rank + 1), 2.0f, 3.0f, 4.0f};
  if (rank == world - 1)
    grad[1] = NAN;
  CHECK(et_dist5_aot_sync_gradients_f32(h, grad, dst, 4, 7, TEST_TIMEOUT_MS) ==
        ET_DIST5_ERR_NONFINITE);
  CHECK(memcmp(dst, dst_snap, sizeof(dst)) == 0);

  if (rank == world - 1)
    grad[1] = INFINITY;
  CHECK(et_dist5_aot_sync_gradients_f32(h, grad, dst, 4, 8, TEST_TIMEOUT_MS) ==
        ET_DIST5_ERR_NONFINITE);
  CHECK(memcmp(dst, dst_snap, sizeof(dst)) == 0);

  /* Communicator remains reusable after coordinated non-finite abort. */
  grad[1] = 2.0f;
  CHECK(et_dist5_aot_sync_gradients_f32(h, grad, dst, 4, 9, TEST_TIMEOUT_MS) ==
        ET_DIST5_OK);
  CHECK(dst[0] == (float)(world + 1) * 0.5f);
  CHECK(dst[1] == 2.0f && dst[2] == 3.0f && dst[3] == 4.0f);

  et_dist5_aot_close(h);
  CHECK(count_open_fds() == fds_before);
}

static void worker_kind_and_tag_mismatch(const char *sock_dir, int64_t rank,
                                         int64_t world) {
  et_dist5_aot_handle_t *h = NULL;
  CHECK(et_dist5_aot_open(sock_dir, rank, world, TEST_TIMEOUT_MS, &h) ==
            ET_DIST5_OK &&
        h != NULL);
  float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  int32_t st;
  if (rank == 0) {
    st = et_dist5_aot_barrier(h, 10, TEST_TIMEOUT_MS);
  } else {
    st = et_dist5_aot_broadcast_f32(h, buf, 4, 0, 10, TEST_TIMEOUT_MS);
  }
  CHECK(st == ET_DIST5_ERR_MISMATCH);
  /* Sticky failure after mismatch. */
  CHECK(et_dist5_aot_barrier(h, 11, TEST_TIMEOUT_MS) == ET_DIST5_ERR_MISMATCH);
  et_dist5_aot_close(h);
}

static void worker_tag_mismatch(const char *sock_dir, int64_t rank,
                                int64_t world) {
  et_dist5_aot_handle_t *h = NULL;
  CHECK(et_dist5_aot_open(sock_dir, rank, world, TEST_TIMEOUT_MS, &h) ==
            ET_DIST5_OK &&
        h != NULL);
  CHECK(et_dist5_aot_barrier(h, rank == 0 ? 42 : 99, TEST_TIMEOUT_MS) ==
        ET_DIST5_ERR_MISMATCH);
  et_dist5_aot_close(h);
}

static void test_duplicate_rank_rejection(void) {
  const int fds_before = count_open_fds();
  char sock_dir[64];
  make_temp_sock_dir(sock_dir, sizeof(sock_dir));

  pid_t pids[3];
  for (int i = 0; i < 3; ++i) {
    pids[i] = fork();
    CHECK(pids[i] >= 0);
    if (pids[i] == 0) {
      et_dist5_aot_handle_t *h = NULL;
      /* Both workers claim rank 1 for world=3! */
      const int64_t claimed_rank = i == 0 ? 0 : 1;
      const int32_t st =
          et_dist5_aot_open(sock_dir, claimed_rank, 3, 2000, &h);
      CHECK(st != ET_DIST5_OK && h == NULL);
      if (i == 0)
        CHECK(st == ET_DIST5_ERR_MISMATCH);
      _exit(0);
    }
  }
  for (int i = 0; i < 3; ++i) {
    int ws = 0;
    while (waitpid(pids[i], &ws, 0) < 0)
      CHECK(errno == EINTR);
    CHECK(WIFEXITED(ws) && WEXITSTATUS(ws) == 0);
  }
  remove_sock_dir(sock_dir);
  CHECK(count_open_fds() == fds_before);
}

int main(void) {
  alarm(45);
  const int fds_start = count_open_fds();

  test_open_negative_and_fd_leaks();
  test_buffer_helpers_and_alignment();
  test_w1_negatives_and_mod5();
  run_multiprocess(2, worker_all_collectives_and_nonfinite, 0);
  run_multiprocess(2, worker_all_collectives_and_nonfinite, 1);
  run_multiprocess(4, worker_all_collectives_and_nonfinite, 1);
  run_multiprocess(2, worker_kind_and_tag_mismatch, 0);
  run_multiprocess(2, worker_tag_mismatch, 1);
  test_duplicate_rank_rejection();

  CHECK(count_open_fds() == fds_start);
  puts("DIST5 AOT bridge tests: PASS (open/FD/alignment/collectives/nonfinite/mismatch)");
  return 0;
}
