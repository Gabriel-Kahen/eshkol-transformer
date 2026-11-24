#define _GNU_SOURCE
#include "eshkol_transformer/hybrid_mesh_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

_Static_assert(sizeof(float) == 4u && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   FLT_MAX_EXP == 128 && FLT_EVAL_METHOD == 0,
               "Hybrid 2D mesh engine requires IEEE-754 binary32");
_Static_assert(sizeof(double) == 8u && DBL_MANT_DIG == 53,
               "Hybrid 2D mesh engine requires IEEE-754 binary64");

#define MESH_HELLO_MAGIC UINT64_C(0x4554324448454c4f) /* "ET2DHELO" */
#define MESH_WIRE_MAGIC UINT64_C(0x4554324457495201)  /* "ET2DWIR\x01" */

enum {
  HELLO_WORDS = 8,
  ACK_WORDS = 2,
  REQ_WORDS = 7,
  REP_WORDS = 3
};

enum mesh_subgroup_kind {
  SUBGROUP_TP = 1,
  SUBGROUP_DP = 2
};

enum mesh_op_kind {
  OP_ALLREDUCE_SUM_F32 = 1,
  OP_ALLREDUCE_SUM_F64 = 2,
  OP_ALLGATHER_F32 = 3,
  OP_ALLREDUCE_AVG_F32 = 4,
  OP_ALLREDUCE_AVG_F64 = 5,
  OP_CONSENSUS_OR_U64 = 6
};

enum mesh_phase {
  PHASE_GRANT = 1,
  PHASE_RESULT = 2,
  PHASE_ABORT_NONFINITE = 3,
  PHASE_FATAL = 4
};

typedef struct et_subcomm {
  uint32_t sub_rank;
  uint32_t sub_size;
  uint64_t sequence;
  et_mesh_status_t fatal;
  int fd[ET_MESH_MAX_WORLD_SIZE];
} et_subcomm_t;

struct et_mesh {
  et_mesh_coords_v1 coords;
  et_subcomm_t tp_group;
  et_subcomm_t dp_group;
  et_mesh_comm_stats_v1 stats;
};

struct et_mesh_engine {
  et_mesh_t *mesh;
  et_mesh_model_config_v1 mcfg;
  et_mesh_optim_config_v1 ocfg;
  uint32_t d_ff_local;
  uint32_t local_batch;
  uint64_t replicated_params;
  uint64_t full_sharded_params;
  uint64_t rank_sharded_params;
  uint64_t full_total_params;
  uint64_t rank_total_params;
  uint64_t step;
  int has_forward_cache;

  /* Persistent per-rank state: layout
   * [gamma(D), b_down(D), W_head(C*D), b_head(C),
   *  W_gate_r(H_loc*D), W_up_r(H_loc*D), W_down_r(D*H_loc)] */
  float *params;
  float *grads;
  float *m;
  float *v;

  /* Forward activation cache (binary64 internal precision for numerical accuracy). */
  float *cache_x;        /* [B_loc, D] */
  int32_t *cache_target; /* [B_loc] */
  double *cache_rrms;    /* [B_loc] */
  double *cache_u;       /* [B_loc, D] */
  double *cache_gate;    /* [B_loc, H_loc] */
  double *cache_up;      /* [B_loc, H_loc] */
  double *cache_act;     /* [B_loc, H_loc] */
  double *cache_h;       /* [B_loc, D] */
  double *cache_probs;   /* [B_loc, C] */
  double cache_local_loss;

  /* Reusable scratch for TP f32 collectives ([B_loc, D]). */
  float *scratch_bd_send;
  float *scratch_bd_recv;
};

/* --- Bit / Endian / FP / Overflow Helpers --- */

static uint32_t f32_to_bits(float v) {
  uint32_t b = 0u;
  memcpy(&b, &v, sizeof(b));
  return b;
}

static float bits_to_f32(uint32_t b) {
  float v = 0.0f;
  memcpy(&v, &b, sizeof(v));
  return v;
}

static uint64_t f64_to_bits(double v) {
  uint64_t b = 0u;
  memcpy(&b, &v, sizeof(b));
  return b;
}

static double bits_to_f64(uint64_t b) {
  double v = 0.0;
  memcpy(&v, &b, sizeof(v));
  return v;
}

static float canonical_f32(float v) { return v == 0.0f ? 0.0f : v; }

static double canonical_f64(double v) { return v == 0.0 ? 0.0 : v; }

static void put_u32_le(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
  dst[2] = (uint8_t)((v >> 16u) & 0xffu);
  dst[3] = (uint8_t)((v >> 24u) & 0xffu);
}

static void put_u64_le(uint8_t *dst, uint64_t v) {
  for (size_t i = 0u; i < 8u; ++i) {
    dst[i] = (uint8_t)((v >> (8u * i)) & 0xffu);
  }
}

static uint32_t get_u32_le(const uint8_t *src) {
  return ((uint32_t)src[0]) | ((uint32_t)src[1] << 8u) |
         ((uint32_t)src[2] << 16u) | ((uint32_t)src[3] << 24u);
}

static uint64_t get_u64_le(const uint8_t *src) {
  uint64_t v = 0u;
  for (size_t i = 0u; i < 8u; ++i) {
    v |= ((uint64_t)src[i]) << (8u * i);
  }
  return v;
}

static int mul_u64_overflow(uint64_t a, uint64_t b, uint64_t *out) {
  if (a != 0u && b > UINT64_MAX / a) {
    return 1;
  }
  *out = a * b;
  return 0;
}

static int add_u64_overflow(uint64_t a, uint64_t b, uint64_t *out) {
  if (a > UINT64_MAX - b) {
    return 1;
  }
  *out = a + b;
  return 0;
}

static int ranges_overlap(const void *a, size_t a_bytes, const void *b,
                          size_t b_bytes) {
  if (a == NULL || b == NULL || a_bytes == 0u || b_bytes == 0u) {
    return 0;
  }
  const uintptr_t la = (uintptr_t)a;
  const uintptr_t lb = (uintptr_t)b;
  if (la > UINTPTR_MAX - a_bytes || lb > UINTPTR_MAX - b_bytes) {
    return 1;
  }
  return la < lb + b_bytes && lb < la + a_bytes;
}

static int has_nonfinite_f32(const float *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!isfinite(buf[i])) {
      return 1;
    }
  }
  return 0;
}

static int has_nonfinite_f64(const double *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!isfinite(buf[i])) {
      return 1;
    }
  }
  return 0;
}

/* --- Standalone FIPS 180-4 SHA-256 --- */

typedef struct mesh_sha256_ctx {
  uint32_t state[8];
  uint64_t bitlen;
  uint8_t buf[64];
  size_t buflen;
} mesh_sha256_ctx;

static uint32_t rotr32(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32u - n));
}

static void sha256_transform(mesh_sha256_ctx *ctx, const uint8_t block[64]) {
  static const uint32_t K[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

  uint32_t w[64];
  for (size_t i = 0u; i < 16u; ++i) {
    w[i] = ((uint32_t)block[i * 4u] << 24u) |
           ((uint32_t)block[i * 4u + 1u] << 16u) |
           ((uint32_t)block[i * 4u + 2u] << 8u) |
           ((uint32_t)block[i * 4u + 3u]);
  }
  for (size_t i = 16u; i < 64u; ++i) {
    const uint32_t s0 =
        rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
    const uint32_t s1 =
        rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }

  uint32_t a = ctx->state[0];
  uint32_t b = ctx->state[1];
  uint32_t c = ctx->state[2];
  uint32_t d = ctx->state[3];
  uint32_t e = ctx->state[4];
  uint32_t f = ctx->state[5];
  uint32_t g = ctx->state[6];
  uint32_t h = ctx->state[7];

  for (size_t i = 0u; i < 64u; ++i) {
    const uint32_t S1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t temp1 = h + S1 + ch + K[i] + w[i];
    const uint32_t S0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = S0 + maj;

    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

static void sha256_init(mesh_sha256_ctx *ctx) {
  ctx->state[0] = 0x6a09e667u;
  ctx->state[1] = 0xbb67ae85u;
  ctx->state[2] = 0x3c6ef372u;
  ctx->state[3] = 0xa54ff53au;
  ctx->state[4] = 0x510e527fu;
  ctx->state[5] = 0x9b05688cu;
  ctx->state[6] = 0x1f83d9abu;
  ctx->state[7] = 0x5be0cd19u;
  ctx->bitlen = 0u;
  ctx->buflen = 0u;
}

static void sha256_update(mesh_sha256_ctx *ctx, const uint8_t *data,
                          size_t len) {
  for (size_t i = 0u; i < len; ++i) {
    ctx->buf[ctx->buflen++] = data[i];
    if (ctx->buflen == 64u) {
      sha256_transform(ctx, ctx->buf);
      ctx->bitlen += 512u;
      ctx->buflen = 0u;
    }
  }
}

static void sha256_final(mesh_sha256_ctx *ctx, uint8_t digest[32]) {
  const uint64_t total_bits = ctx->bitlen + (uint64_t)ctx->buflen * 8u;
  ctx->buf[ctx->buflen++] = 0x80u;
  if (ctx->buflen > 56u) {
    while (ctx->buflen < 64u) {
      ctx->buf[ctx->buflen++] = 0x00u;
    }
    sha256_transform(ctx, ctx->buf);
    ctx->buflen = 0u;
  }
  while (ctx->buflen < 56u) {
    ctx->buf[ctx->buflen++] = 0x00u;
  }
  for (size_t i = 0u; i < 8u; ++i) {
    ctx->buf[56u + i] = (uint8_t)((total_bits >> (56u - 8u * i)) & 0xffu);
  }
  sha256_transform(ctx, ctx->buf);
  for (size_t i = 0u; i < 8u; ++i) {
    digest[i * 4u + 0u] = (uint8_t)((ctx->state[i] >> 24u) & 0xffu);
    digest[i * 4u + 1u] = (uint8_t)((ctx->state[i] >> 16u) & 0xffu);
    digest[i * 4u + 2u] = (uint8_t)((ctx->state[i] >> 8u) & 0xffu);
    digest[i * 4u + 3u] = (uint8_t)(ctx->state[i] & 0xffu);
  }
}

ET_MESH_API et_mesh_status_t et_mesh_sha256_bytes(const uint8_t *data,
                                                  size_t len,
                                                  uint8_t out_digest[32]) {
  if (out_digest == NULL || (len > 0u && data == NULL)) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  mesh_sha256_ctx ctx;
  sha256_init(&ctx);
  if (len > 0u) {
    sha256_update(&ctx, data, len);
  }
  sha256_final(&ctx, out_digest);
  return ET_MESH_OK;
}

/* --- Monotonic Clock & Socket I/O Helpers --- */

static int monotonic_now_ns(uint64_t *out_ns) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0 || ts.tv_sec < 0 ||
      ts.tv_nsec < 0) {
    return -1;
  }
  *out_ns = (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
  return 0;
}

static int compute_deadline_ns(int timeout_ms, uint64_t *out_deadline) {
  if (timeout_ms <= 0) {
    return -1;
  }
  uint64_t now = 0u;
  if (monotonic_now_ns(&now) != 0) {
    return -1;
  }
  const uint64_t delta = (uint64_t)timeout_ms * UINT64_C(1000000);
  *out_deadline = (now > UINT64_MAX - delta) ? UINT64_MAX : (now + delta);
  return 0;
}

static int remaining_ms_from_deadline(uint64_t deadline_ns, int *out_ms) {
  uint64_t now = 0u;
  if (monotonic_now_ns(&now) != 0) {
    return -1;
  }
  if (now >= deadline_ns) {
    *out_ms = 0;
    return 0;
  }
  const uint64_t diff_ns = deadline_ns - now;
  const uint64_t ms = (diff_ns + UINT64_C(999999)) / UINT64_C(1000000);
  *out_ms = (ms > (uint64_t)INT_MAX) ? INT_MAX : (int)ms;
  return 0;
}

static et_mesh_status_t wait_fd_event(int fd, short events,
                                      uint64_t deadline_ns) {
  for (;;) {
    int rem_ms = 0;
    if (remaining_ms_from_deadline(deadline_ns, &rem_ms) != 0) {
      return ET_MESH_ERR_IO;
    }
    if (rem_ms <= 0) {
      return ET_MESH_ERR_TIMEOUT;
    }
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = events;
    pfd.revents = 0;
    const int rc = poll(&pfd, 1u, rem_ms);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      return ET_MESH_ERR_IO;
    }
    if (rc == 0) {
      return ET_MESH_ERR_TIMEOUT;
    }
    if ((pfd.revents & POLLNVAL) != 0) {
      return ET_MESH_ERR_IO;
    }
    if ((pfd.revents & (POLLERR | POLLHUP)) != 0 &&
        (pfd.revents & events) == 0) {
      return ET_MESH_ERR_PEER_DISCONNECTED;
    }
    return ET_MESH_OK;
  }
}

static et_mesh_status_t send_all_deadline(int fd, const void *buf, size_t len,
                                          uint64_t deadline_ns) {
  const uint8_t *p = (const uint8_t *)buf;
  size_t sent = 0u;
  while (sent < len) {
    et_mesh_status_t wst = wait_fd_event(fd, POLLOUT, deadline_ns);
    if (wst != ET_MESH_OK) {
      return wst;
    }
    const ssize_t n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      if (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN) {
        return ET_MESH_ERR_PEER_DISCONNECTED;
      }
      return ET_MESH_ERR_IO;
    }
    if (n == 0) {
      return ET_MESH_ERR_PEER_DISCONNECTED;
    }
    sent += (size_t)n;
  }
  return ET_MESH_OK;
}

static et_mesh_status_t recv_all_deadline(int fd, void *buf, size_t len,
                                          uint64_t deadline_ns) {
  uint8_t *p = (uint8_t *)buf;
  size_t got = 0u;
  while (got < len) {
    et_mesh_status_t wst = wait_fd_event(fd, POLLIN, deadline_ns);
    if (wst != ET_MESH_OK) {
      return wst;
    }
    const ssize_t n = recv(fd, p + got, len - got, 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      if (errno == ECONNRESET || errno == ENOTCONN) {
        return ET_MESH_ERR_PEER_DISCONNECTED;
      }
      return ET_MESH_ERR_IO;
    }
    if (n == 0) {
      return ET_MESH_ERR_PEER_DISCONNECTED;
    }
    got += (size_t)n;
  }
  return ET_MESH_OK;
}

static void encode_words_le(uint8_t *dst, const uint64_t *words, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    put_u64_le(dst + i * 8u, words[i]);
  }
}

static void decode_words_le(const uint8_t *src, uint64_t *words, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    words[i] = get_u64_le(src + i * 8u);
  }
}

static et_mesh_status_t send_words(int fd, const uint64_t *words, size_t n,
                                   uint64_t deadline_ns) {
  uint8_t raw[HELLO_WORDS * 8u];
  if (n > HELLO_WORDS) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  encode_words_le(raw, words, n);
  return send_all_deadline(fd, raw, n * 8u, deadline_ns);
}

static et_mesh_status_t recv_words(int fd, uint64_t *words, size_t n,
                                   uint64_t deadline_ns) {
  uint8_t raw[HELLO_WORDS * 8u];
  if (n > HELLO_WORDS) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  et_mesh_status_t st = recv_all_deadline(fd, raw, n * 8u, deadline_ns);
  if (st != ET_MESH_OK) {
    return st;
  }
  decode_words_le(raw, words, n);
  return ET_MESH_OK;
}

static int validate_unix_stream_fd(int fd) {
  if (fd < 0) {
    return 0;
  }
  int sock_type = 0;
  socklen_t opt_len = sizeof(sock_type);
  if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &sock_type, &opt_len) != 0 ||
      sock_type != SOCK_STREAM) {
    return 0;
  }
  struct sockaddr_un local_addr;
  socklen_t local_len = sizeof(local_addr);
  memset(&local_addr, 0, sizeof(local_addr));
  if (getsockname(fd, (struct sockaddr *)&local_addr, &local_len) != 0 ||
      local_addr.sun_family != AF_UNIX) {
    return 0;
  }
  struct sockaddr_un peer_addr;
  socklen_t peer_len = sizeof(peer_addr);
  memset(&peer_addr, 0, sizeof(peer_addr));
  if (getpeername(fd, (struct sockaddr *)&peer_addr, &peer_len) != 0 ||
      peer_addr.sun_family != AF_UNIX) {
    return 0;
  }
  return 1;
}

static void subcomm_init_empty(et_subcomm_t *sub, uint32_t sub_rank,
                               uint32_t sub_size) {
  sub->sub_rank = sub_rank;
  sub->sub_size = sub_size;
  sub->sequence = 1u;
  sub->fatal = ET_MESH_OK;
  for (size_t i = 0u; i < ET_MESH_MAX_WORLD_SIZE; ++i) {
    sub->fd[i] = -1;
  }
}

static void subcomm_close(et_subcomm_t *sub) {
  if (sub == NULL) {
    return;
  }
  for (size_t i = 0u; i < ET_MESH_MAX_WORLD_SIZE; ++i) {
    if (sub->fd[i] >= 0) {
      (void)close(sub->fd[i]);
      sub->fd[i] = -1;
    }
  }
}

static et_mesh_status_t subcomm_set_fds(et_subcomm_t *sub, const int *fds,
                                        size_t fd_count) {
  if (sub->sub_size == 1u) {
    if (fd_count != 0u) {
      return ET_MESH_ERR_INVALID_ARG;
    }
    return ET_MESH_OK;
  }
  const size_t expected =
      (sub->sub_rank == 0u) ? (size_t)(sub->sub_size - 1u) : 1u;
  if (fd_count != expected || fds == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  for (size_t i = 0u; i < fd_count; ++i) {
    if (!validate_unix_stream_fd(fds[i])) {
      subcomm_close(sub);
      return ET_MESH_ERR_INVALID_ARG;
    }
    const int dup_fd = fcntl(fds[i], F_DUPFD_CLOEXEC, 0);
    if (dup_fd < 0) {
      subcomm_close(sub);
      return ET_MESH_ERR_IO;
    }
    if (sub->sub_rank == 0u) {
      sub->fd[i + 1u] = dup_fd;
    } else {
      sub->fd[0] = dup_fd;
    }
  }
  return ET_MESH_OK;
}

static et_mesh_status_t subcomm_open_unix(et_subcomm_t *sub,
                                          const char *sock_dir,
                                          uint32_t dp_size, uint32_t tp_size,
                                          uint32_t subgroup_kind,
                                          uint32_t subgroup_id,
                                          uint64_t deadline_ns) {
  if (sub->sub_size == 1u) {
    return ET_MESH_OK;
  }
  if (sock_dir == NULL || sock_dir[0] == '\0') {
    return ET_MESH_ERR_INVALID_ARG;
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  const char *tag_str = (subgroup_kind == SUBGROUP_TP) ? "tp_dp" : "dp_tp";
  const int n = snprintf(addr.sun_path, sizeof(addr.sun_path),
                         "%s/mesh_%s%u.sock", sock_dir, tag_str, subgroup_id);
  if (n <= 0 || (size_t)n >= sizeof(addr.sun_path)) {
    return ET_MESH_ERR_INVALID_ARG;
  }

  if (sub->sub_rank == 0u) {
    (void)unlink(addr.sun_path);
    const int lfd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (lfd < 0) {
      return ET_MESH_ERR_IO;
    }
    if (bind(lfd, (const struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(lfd, (int)sub->sub_size) != 0) {
      (void)close(lfd);
      (void)unlink(addr.sun_path);
      return ET_MESH_ERR_IO;
    }

    uint32_t accepted = 0u;
    et_mesh_status_t err = ET_MESH_OK;
    while (accepted < sub->sub_size - 1u) {
      err = wait_fd_event(lfd, POLLIN, deadline_ns);
      if (err != ET_MESH_OK) {
        break;
      }
      const int cfd = accept4(lfd, NULL, NULL, SOCK_CLOEXEC);
      if (cfd < 0) {
        if (errno == EINTR) {
          continue;
        }
        err = ET_MESH_ERR_IO;
        break;
      }
      uint64_t hello[HELLO_WORDS];
      err = recv_words(cfd, hello, HELLO_WORDS, deadline_ns);
      if (err != ET_MESH_OK) {
        (void)close(cfd);
        break;
      }
      if (hello[0] != MESH_HELLO_MAGIC || hello[1] != ET_MESH_ABI_MAJOR ||
          hello[2] != ET_MESH_ABI_MINOR || hello[3] != (uint64_t)dp_size ||
          hello[4] != (uint64_t)tp_size ||
          hello[5] != (uint64_t)subgroup_kind ||
          hello[6] != (uint64_t)subgroup_id || hello[7] == 0u ||
          hello[7] >= (uint64_t)sub->sub_size) {
        (void)close(cfd);
        err = ET_MESH_ERR_MISMATCH;
        break;
      }
      const uint32_t peer_r = (uint32_t)hello[7];
      if (sub->fd[peer_r] >= 0) {
        (void)close(cfd);
        err = ET_MESH_ERR_MISMATCH;
        break;
      }
      sub->fd[peer_r] = cfd;
      ++accepted;
    }

    (void)close(lfd);
    (void)unlink(addr.sun_path);

    if (err != ET_MESH_OK) {
      subcomm_close(sub);
      return err;
    }

    const uint64_t ack[ACK_WORDS] = {MESH_HELLO_MAGIC, (uint64_t)ET_MESH_OK};
    for (uint32_t r = 1u; r < sub->sub_size; ++r) {
      err = send_words(sub->fd[r], ack, ACK_WORDS, deadline_ns);
      if (err != ET_MESH_OK) {
        subcomm_close(sub);
        return err;
      }
    }
    return ET_MESH_OK;
  }

  /* Peer sub_rank > 0: connect with monotonic backoff retry. */
  int cfd = -1;
  for (;;) {
    int rem_ms = 0;
    if (remaining_ms_from_deadline(deadline_ns, &rem_ms) != 0) {
      return ET_MESH_ERR_IO;
    }
    if (rem_ms <= 0) {
      return ET_MESH_ERR_TIMEOUT;
    }
    cfd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (cfd < 0) {
      return ET_MESH_ERR_IO;
    }
    if (connect(cfd, (const struct sockaddr *)&addr, sizeof(addr)) == 0) {
      break;
    }
    const int saved = errno;
    (void)close(cfd);
    cfd = -1;
    if (saved == EINTR) {
      continue;
    }
    if (saved == ENOENT || saved == ECONNREFUSED || saved == EAGAIN) {
      const int sleep_ms = rem_ms < 2 ? rem_ms : 2;
      (void)poll(NULL, 0u, sleep_ms);
      continue;
    }
    return ET_MESH_ERR_IO;
  }

  const uint64_t hello[HELLO_WORDS] = {
      MESH_HELLO_MAGIC,        (uint64_t)ET_MESH_ABI_MAJOR,
      (uint64_t)ET_MESH_ABI_MINOR, (uint64_t)dp_size,
      (uint64_t)tp_size,       (uint64_t)subgroup_kind,
      (uint64_t)subgroup_id,   (uint64_t)sub->sub_rank};
  et_mesh_status_t st = send_words(cfd, hello, HELLO_WORDS, deadline_ns);
  if (st != ET_MESH_OK) {
    (void)close(cfd);
    return st;
  }

  uint64_t ack[ACK_WORDS];
  st = recv_words(cfd, ack, ACK_WORDS, deadline_ns);
  if (st != ET_MESH_OK) {
    (void)close(cfd);
    return st;
  }
  if (ack[0] != MESH_HELLO_MAGIC || ack[1] != (uint64_t)ET_MESH_OK) {
    (void)close(cfd);
    return ET_MESH_ERR_MISMATCH;
  }
  sub->fd[0] = cfd;
  return ET_MESH_OK;
}

/* --- 1. Topology & Lifecycle Public API --- */

static int valid_axis_size(uint32_t s) {
  return s == 1u || s == 2u || s == 4u;
}

ET_MESH_API et_mesh_status_t et_mesh_validate_topology(
    uint32_t dp_size, uint32_t tp_size, uint32_t global_rank,
    et_mesh_coords_v1 *out_coords) {
  if (!valid_axis_size(dp_size) || !valid_axis_size(tp_size)) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  const uint32_t world_size = dp_size * tp_size;
  if (world_size == 0u || world_size > ET_MESH_MAX_WORLD_SIZE) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if (global_rank >= world_size) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if (out_coords != NULL) {
    out_coords->global_rank = global_rank;
    out_coords->world_size = world_size;
    out_coords->dp_size = dp_size;
    out_coords->tp_size = tp_size;
    out_coords->dp_rank = global_rank / tp_size;
    out_coords->tp_rank = global_rank % tp_size;
  }
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_open_unix(
    uint32_t abi_major, uint32_t abi_minor, const char *sock_dir,
    uint32_t global_rank, uint32_t dp_size, uint32_t tp_size, int timeout_ms,
    et_mesh_t **out_mesh) {
  if (out_mesh == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  *out_mesh = NULL;
  if (abi_major != ET_MESH_ABI_MAJOR || abi_minor > ET_MESH_ABI_MINOR) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  et_mesh_coords_v1 coords;
  et_mesh_status_t st =
      et_mesh_validate_topology(dp_size, tp_size, global_rank, &coords);
  if (st != ET_MESH_OK) {
    return st;
  }
  uint64_t deadline_ns = 0u;
  if (compute_deadline_ns(timeout_ms, &deadline_ns) != 0) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if ((dp_size > 1u || tp_size > 1u) &&
      (sock_dir == NULL || sock_dir[0] == '\0')) {
    return ET_MESH_ERR_INVALID_ARG;
  }

  et_mesh_t *mesh = (et_mesh_t *)calloc(1u, sizeof(*mesh));
  if (mesh == NULL) {
    return ET_MESH_ERR_ALLOC;
  }
  mesh->coords = coords;
  subcomm_init_empty(&mesh->tp_group, coords.tp_rank, coords.tp_size);
  subcomm_init_empty(&mesh->dp_group, coords.dp_rank, coords.dp_size);

  /* Phase 1: Establish TP subgroups (ranks sharing dp_rank). */
  st = subcomm_open_unix(&mesh->tp_group, sock_dir, dp_size, tp_size,
                         SUBGROUP_TP, coords.dp_rank, deadline_ns);
  if (st != ET_MESH_OK) {
    et_mesh_destroy(mesh);
    return st;
  }

  /* Phase 2: Establish orthogonal DP subgroups (ranks sharing tp_rank). */
  st = subcomm_open_unix(&mesh->dp_group, sock_dir, dp_size, tp_size,
                         SUBGROUP_DP, coords.tp_rank, deadline_ns);
  if (st != ET_MESH_OK) {
    et_mesh_destroy(mesh);
    return st;
  }

  *out_mesh = mesh;
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_create_fds(
    uint32_t abi_major, uint32_t abi_minor, uint32_t global_rank,
    uint32_t dp_size, uint32_t tp_size, const int *tp_fds, size_t tp_fd_count,
    const int *dp_fds, size_t dp_fd_count, et_mesh_t **out_mesh) {
  if (out_mesh == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  *out_mesh = NULL;
  if (abi_major != ET_MESH_ABI_MAJOR || abi_minor > ET_MESH_ABI_MINOR) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  et_mesh_coords_v1 coords;
  et_mesh_status_t st =
      et_mesh_validate_topology(dp_size, tp_size, global_rank, &coords);
  if (st != ET_MESH_OK) {
    return st;
  }

  et_mesh_t *mesh = (et_mesh_t *)calloc(1u, sizeof(*mesh));
  if (mesh == NULL) {
    return ET_MESH_ERR_ALLOC;
  }
  mesh->coords = coords;
  subcomm_init_empty(&mesh->tp_group, coords.tp_rank, coords.tp_size);
  subcomm_init_empty(&mesh->dp_group, coords.dp_rank, coords.dp_size);

  st = subcomm_set_fds(&mesh->tp_group, tp_fds, tp_fd_count);
  if (st != ET_MESH_OK) {
    et_mesh_destroy(mesh);
    return st;
  }
  st = subcomm_set_fds(&mesh->dp_group, dp_fds, dp_fd_count);
  if (st != ET_MESH_OK) {
    et_mesh_destroy(mesh);
    return st;
  }

  *out_mesh = mesh;
  return ET_MESH_OK;
}

ET_MESH_API void et_mesh_destroy(et_mesh_t *mesh) {
  if (mesh == NULL) {
    return;
  }
  subcomm_close(&mesh->tp_group);
  subcomm_close(&mesh->dp_group);
  free(mesh);
}

ET_MESH_API et_mesh_status_t et_mesh_get_coords(const et_mesh_t *mesh,
                                                et_mesh_coords_v1 *out_coords) {
  if (mesh == NULL || out_coords == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  *out_coords = mesh->coords;
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_get_stats(
    const et_mesh_t *mesh, et_mesh_comm_stats_v1 *out_stats) {
  if (mesh == NULL || out_stats == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  *out_stats = mesh->stats;
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_reset_stats(et_mesh_t *mesh) {
  if (mesh == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  memset(&mesh->stats, 0, sizeof(mesh->stats));
  return ET_MESH_OK;
}

/* --- 2. Subgroup Collective Engine --- */

static et_mesh_status_t subcomm_fail_fatal(et_subcomm_t *sub,
                                           et_mesh_status_t err) {
  if (sub->fatal == ET_MESH_OK) {
    sub->fatal = err;
  }
  return sub->fatal;
}

static et_mesh_status_t subcomm_run_tensor_op(
    et_subcomm_t *sub, uint32_t op_kind, const void *sendbuf, void *recvbuf,
    size_t send_count, uint64_t tag, int timeout_ms, uint64_t *bytes_sent_acc,
    uint64_t *bytes_recv_acc, uint64_t *calls_acc) {
  if (sub == NULL || sendbuf == NULL || recvbuf == NULL || send_count == 0u) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if (sub->fatal != ET_MESH_OK) {
    return sub->fatal;
  }
  uint64_t deadline_ns = 0u;
  if (compute_deadline_ns(timeout_ms, &deadline_ns) != 0) {
    return ET_MESH_ERR_INVALID_ARG;
  }

  const int is_f64 =
      (op_kind == OP_ALLREDUCE_SUM_F64 || op_kind == OP_ALLREDUCE_AVG_F64);
  const int is_allgather = (op_kind == OP_ALLGATHER_F32);
  const size_t elem_size = is_f64 ? sizeof(double) : sizeof(float);

  uint64_t send_bytes_u64 = 0u;
  if (mul_u64_overflow((uint64_t)send_count, (uint64_t)elem_size,
                       &send_bytes_u64) ||
      send_bytes_u64 > ET_MESH_MAX_BUFFER_BYTES) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  uint64_t recv_bytes_u64 = send_bytes_u64;
  if (is_allgather) {
    if (mul_u64_overflow(send_bytes_u64, (uint64_t)sub->sub_size,
                         &recv_bytes_u64) ||
        recv_bytes_u64 > ET_MESH_MAX_BUFFER_BYTES) {
      return ET_MESH_ERR_INVALID_ARG;
    }
  }
  const size_t send_bytes = (size_t)send_bytes_u64;
  const size_t recv_bytes = (size_t)recv_bytes_u64;

  /* Aliasing rules: AllGather must be disjoint; AllReduce allows exact in-place. */
  if (is_allgather) {
    if (ranges_overlap(sendbuf, send_bytes, recvbuf, recv_bytes)) {
      return ET_MESH_ERR_ALIAS;
    }
  } else if (sendbuf != recvbuf &&
             ranges_overlap(sendbuf, send_bytes, recvbuf, recv_bytes)) {
    return ET_MESH_ERR_ALIAS;
  }

  const uint64_t local_nf =
      is_f64 ? (uint64_t)has_nonfinite_f64((const double *)sendbuf, send_count)
             : (uint64_t)has_nonfinite_f32((const float *)sendbuf, send_count);

  /* Subgroup size 1: zero sockets, zero IPC calls, local canonical copy. */
  if (sub->sub_size == 1u) {
    if (local_nf != 0u) {
      return ET_MESH_ERR_NONFINITE;
    }
    if (is_f64) {
      const double *src = (const double *)sendbuf;
      double *dst = (double *)recvbuf;
      for (size_t i = 0u; i < send_count; ++i) {
        dst[i] = canonical_f64(src[i]);
      }
    } else {
      const float *src = (const float *)sendbuf;
      float *dst = (float *)recvbuf;
      for (size_t i = 0u; i < send_count; ++i) {
        dst[i] = canonical_f32(src[i]);
      }
    }
    ++sub->sequence;
    return ET_MESH_OK;
  }

  const uint64_t seq = sub->sequence;

  if (sub->sub_rank == 0u) {
    /* Phase 1: Receive peer headers and validate strict agreement. */
    int mismatch = 0;
    uint64_t any_nf = local_nf;
    for (uint32_t r = 1u; r < sub->sub_size; ++r) {
      uint64_t req[REQ_WORDS];
      et_mesh_status_t st = recv_words(sub->fd[r], req, REQ_WORDS, deadline_ns);
      if (st != ET_MESH_OK) {
        return subcomm_fail_fatal(sub, st);
      }
      if (req[0] != MESH_WIRE_MAGIC || req[1] != seq || req[2] != tag ||
          req[3] != (uint64_t)op_kind || req[4] != (uint64_t)send_count) {
        mismatch = 1;
      } else if (req[5] != 0u) {
        any_nf = 1u;
      }
    }

    if (mismatch) {
      const uint64_t rep[REP_WORDS] = {
          (uint64_t)PHASE_FATAL, (uint64_t)(int64_t)ET_MESH_ERR_MISMATCH, 0u};
      for (uint32_t r = 1u; r < sub->sub_size; ++r) {
        (void)send_words(sub->fd[r], rep, REP_WORDS, deadline_ns);
      }
      return subcomm_fail_fatal(sub, ET_MESH_ERR_MISMATCH);
    }

    if (any_nf != 0u) {
      const uint64_t rep[REP_WORDS] = {
          (uint64_t)PHASE_ABORT_NONFINITE,
          (uint64_t)(int64_t)ET_MESH_ERR_NONFINITE, 0u};
      for (uint32_t r = 1u; r < sub->sub_size; ++r) {
        et_mesh_status_t st =
            send_words(sub->fd[r], rep, REP_WORDS, deadline_ns);
        if (st != ET_MESH_OK) {
          return subcomm_fail_fatal(sub, st);
        }
      }
      ++sub->sequence;
      return ET_MESH_ERR_NONFINITE;
    }

    /* Grant payload phase. */
    const uint64_t grant[REP_WORDS] = {(uint64_t)PHASE_GRANT,
                                       (uint64_t)ET_MESH_OK, 0u};
    for (uint32_t r = 1u; r < sub->sub_size; ++r) {
      et_mesh_status_t st =
          send_words(sub->fd[r], grant, REP_WORDS, deadline_ns);
      if (st != ET_MESH_OK) {
        return subcomm_fail_fatal(sub, st);
      }
    }

    /* Allocate temporary gather/result buffers (leader never mutates recvbuf on error). */
    uint8_t *peer_buf = (uint8_t *)malloc((size_t)sub->sub_size * send_bytes);
    uint8_t *result_buf = (uint8_t *)malloc(recv_bytes);
    if (peer_buf == NULL || result_buf == NULL) {
      free(peer_buf);
      free(result_buf);
      return subcomm_fail_fatal(sub, ET_MESH_ERR_ALLOC);
    }
    memcpy(peer_buf, sendbuf, send_bytes);

    for (uint32_t r = 1u; r < sub->sub_size; ++r) {
      et_mesh_status_t st = recv_all_deadline(
          sub->fd[r], peer_buf + (size_t)r * send_bytes, send_bytes,
          deadline_ns);
      if (st != ET_MESH_OK) {
        free(peer_buf);
        free(result_buf);
        return subcomm_fail_fatal(sub, st);
      }
      *bytes_recv_acc += (uint64_t)send_bytes;
    }

    /* Canonical ascending sub_rank order 0..sub_size-1 reduction in binary64. */
    int overflow_nf = 0;
    if (op_kind == OP_ALLGATHER_F32) {
      const float *src = (const float *)peer_buf;
      float *dst = (float *)result_buf;
      const size_t total = (size_t)sub->sub_size * send_count;
      for (size_t i = 0u; i < total; ++i) {
        if (!isfinite(src[i])) {
          overflow_nf = 1;
          break;
        }
        dst[i] = canonical_f32(src[i]);
      }
    } else if (op_kind == OP_ALLREDUCE_SUM_F32 ||
               op_kind == OP_ALLREDUCE_AVG_F32) {
      float *dst = (float *)result_buf;
      const double inv_s =
          (op_kind == OP_ALLREDUCE_AVG_F32) ? (1.0 / (double)sub->sub_size) : 1.0;
      for (size_t i = 0u; i < send_count; ++i) {
        double acc = 0.0;
        for (uint32_t r = 0u; r < sub->sub_size; ++r) {
          const float *rb = (const float *)(peer_buf + (size_t)r * send_bytes);
          acc += (double)rb[i];
        }
        const double val =
            (op_kind == OP_ALLREDUCE_AVG_F32) ? (acc * inv_s) : acc;
        const float out_v = (float)val;
        if (!isfinite(val) || !isfinite(out_v)) {
          overflow_nf = 1;
          break;
        }
        dst[i] = canonical_f32(out_v);
      }
    } else {
      /* OP_ALLREDUCE_SUM_F64 or OP_ALLREDUCE_AVG_F64 */
      double *dst = (double *)result_buf;
      for (size_t i = 0u; i < send_count; ++i) {
        double acc = 0.0;
        for (uint32_t r = 0u; r < sub->sub_size; ++r) {
          const double *rb = (const double *)(peer_buf + (size_t)r * send_bytes);
          acc += rb[i];
        }
        const double val = (op_kind == OP_ALLREDUCE_AVG_F64)
                               ? (acc / (double)sub->sub_size)
                               : acc;
        if (!isfinite(val)) {
          overflow_nf = 1;
          break;
        }
        dst[i] = canonical_f64(val);
      }
    }
    free(peer_buf);

    if (overflow_nf) {
      free(result_buf);
      const uint64_t rep[REP_WORDS] = {
          (uint64_t)PHASE_ABORT_NONFINITE,
          (uint64_t)(int64_t)ET_MESH_ERR_NONFINITE, 0u};
      for (uint32_t r = 1u; r < sub->sub_size; ++r) {
        et_mesh_status_t st =
            send_words(sub->fd[r], rep, REP_WORDS, deadline_ns);
        if (st != ET_MESH_OK) {
          return subcomm_fail_fatal(sub, st);
        }
      }
      ++sub->sequence;
      return ET_MESH_ERR_NONFINITE;
    }

    const uint64_t ok_rep[REP_WORDS] = {(uint64_t)PHASE_RESULT,
                                        (uint64_t)ET_MESH_OK, 0u};
    for (uint32_t r = 1u; r < sub->sub_size; ++r) {
      et_mesh_status_t st =
          send_words(sub->fd[r], ok_rep, REP_WORDS, deadline_ns);
      if (st == ET_MESH_OK) {
        st = send_all_deadline(sub->fd[r], result_buf, recv_bytes, deadline_ns);
      }
      if (st != ET_MESH_OK) {
        free(result_buf);
        return subcomm_fail_fatal(sub, st);
      }
      *bytes_sent_acc += (uint64_t)recv_bytes;
    }

    memcpy(recvbuf, result_buf, recv_bytes);
    free(result_buf);
    ++sub->sequence;
    ++(*calls_acc);
    return ET_MESH_OK;
  }

  /* Peer sub_rank > 0 */
  const uint64_t req[REQ_WORDS] = {
      MESH_WIRE_MAGIC, seq, tag, (uint64_t)op_kind, (uint64_t)send_count,
      local_nf,        0u};
  et_mesh_status_t st = send_words(sub->fd[0], req, REQ_WORDS, deadline_ns);
  if (st != ET_MESH_OK) {
    return subcomm_fail_fatal(sub, st);
  }

  uint64_t rep[REP_WORDS];
  st = recv_words(sub->fd[0], rep, REP_WORDS, deadline_ns);
  if (st != ET_MESH_OK) {
    return subcomm_fail_fatal(sub, st);
  }
  if (rep[0] == (uint64_t)PHASE_ABORT_NONFINITE) {
    ++sub->sequence;
    return ET_MESH_ERR_NONFINITE;
  }
  if (rep[0] != (uint64_t)PHASE_GRANT) {
    const et_mesh_status_t ferr =
        (rep[0] == (uint64_t)PHASE_FATAL)
            ? (et_mesh_status_t)(int64_t)rep[1]
            : ET_MESH_ERR_MISMATCH;
    return subcomm_fail_fatal(sub, ferr);
  }

  st = send_all_deadline(sub->fd[0], sendbuf, send_bytes, deadline_ns);
  if (st != ET_MESH_OK) {
    return subcomm_fail_fatal(sub, st);
  }
  *bytes_sent_acc += (uint64_t)send_bytes;

  st = recv_words(sub->fd[0], rep, REP_WORDS, deadline_ns);
  if (st != ET_MESH_OK) {
    return subcomm_fail_fatal(sub, st);
  }
  if (rep[0] == (uint64_t)PHASE_ABORT_NONFINITE) {
    ++sub->sequence;
    return ET_MESH_ERR_NONFINITE;
  }
  if (rep[0] != (uint64_t)PHASE_RESULT || rep[1] != (uint64_t)ET_MESH_OK) {
    return subcomm_fail_fatal(sub, ET_MESH_ERR_MISMATCH);
  }

  uint8_t *tmp = (uint8_t *)malloc(recv_bytes);
  if (tmp == NULL) {
    return subcomm_fail_fatal(sub, ET_MESH_ERR_ALLOC);
  }
  st = recv_all_deadline(sub->fd[0], tmp, recv_bytes, deadline_ns);
  if (st != ET_MESH_OK) {
    free(tmp);
    return subcomm_fail_fatal(sub, st);
  }
  *bytes_recv_acc += (uint64_t)recv_bytes;
  memcpy(recvbuf, tmp, recv_bytes);
  free(tmp);
  ++sub->sequence;
  ++(*calls_acc);
  return ET_MESH_OK;
}

static et_mesh_status_t subcomm_consensus_or_u64(
    et_subcomm_t *sub, uint64_t local_flags, uint64_t *out_flags, uint64_t tag,
    uint64_t deadline_ns, uint64_t *bytes_sent_acc, uint64_t *bytes_recv_acc,
    uint64_t *calls_acc) {
  if (sub == NULL || out_flags == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if (sub->fatal != ET_MESH_OK) {
    return sub->fatal;
  }
  if (sub->sub_size == 1u) {
    *out_flags = local_flags;
    ++sub->sequence;
    return ET_MESH_OK;
  }

  const uint64_t seq = sub->sequence;
  if (sub->sub_rank == 0u) {
    int mismatch = 0;
    uint64_t acc = local_flags;
    for (uint32_t r = 1u; r < sub->sub_size; ++r) {
      uint64_t req[REQ_WORDS];
      et_mesh_status_t st = recv_words(sub->fd[r], req, REQ_WORDS, deadline_ns);
      if (st != ET_MESH_OK) {
        return subcomm_fail_fatal(sub, st);
      }
      *bytes_recv_acc += (uint64_t)(REQ_WORDS * 8u);
      if (req[0] != MESH_WIRE_MAGIC || req[1] != seq || req[2] != tag ||
          req[3] != (uint64_t)OP_CONSENSUS_OR_U64 || req[4] != 1u) {
        mismatch = 1;
      } else {
        acc |= req[6];
      }
    }
    if (mismatch) {
      const uint64_t rep[REP_WORDS] = {
          (uint64_t)PHASE_FATAL, (uint64_t)(int64_t)ET_MESH_ERR_MISMATCH, 0u};
      for (uint32_t r = 1u; r < sub->sub_size; ++r) {
        (void)send_words(sub->fd[r], rep, REP_WORDS, deadline_ns);
      }
      return subcomm_fail_fatal(sub, ET_MESH_ERR_MISMATCH);
    }
    const uint64_t rep[REP_WORDS] = {(uint64_t)PHASE_RESULT,
                                     (uint64_t)ET_MESH_OK, acc};
    for (uint32_t r = 1u; r < sub->sub_size; ++r) {
      et_mesh_status_t st = send_words(sub->fd[r], rep, REP_WORDS, deadline_ns);
      if (st != ET_MESH_OK) {
        return subcomm_fail_fatal(sub, st);
      }
      *bytes_sent_acc += (uint64_t)(REP_WORDS * 8u);
    }
    *out_flags = acc;
    ++sub->sequence;
    ++(*calls_acc);
    return ET_MESH_OK;
  }

  const uint64_t req[REQ_WORDS] = {MESH_WIRE_MAGIC,
                                   seq,
                                   tag,
                                   (uint64_t)OP_CONSENSUS_OR_U64,
                                   1u,
                                   0u,
                                   local_flags};
  et_mesh_status_t st = send_words(sub->fd[0], req, REQ_WORDS, deadline_ns);
  if (st != ET_MESH_OK) {
    return subcomm_fail_fatal(sub, st);
  }
  *bytes_sent_acc += (uint64_t)(REQ_WORDS * 8u);

  uint64_t rep[REP_WORDS];
  st = recv_words(sub->fd[0], rep, REP_WORDS, deadline_ns);
  if (st != ET_MESH_OK) {
    return subcomm_fail_fatal(sub, st);
  }
  *bytes_recv_acc += (uint64_t)(REP_WORDS * 8u);
  if (rep[0] != (uint64_t)PHASE_RESULT || rep[1] != (uint64_t)ET_MESH_OK) {
    const et_mesh_status_t ferr =
        (rep[0] == (uint64_t)PHASE_FATAL)
            ? (et_mesh_status_t)(int64_t)rep[1]
            : ET_MESH_ERR_MISMATCH;
    return subcomm_fail_fatal(sub, ferr);
  }
  *out_flags = rep[2];
  ++sub->sequence;
  ++(*calls_acc);
  return ET_MESH_OK;
}

/* --- Public Orthogonal Collective Entry Points --- */

ET_MESH_API et_mesh_status_t et_mesh_tp_allreduce_sum_f32(
    et_mesh_t *mesh, const float *sendbuf, float *recvbuf, size_t count,
    uint64_t tag, int timeout_ms) {
  if (mesh == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  return subcomm_run_tensor_op(&mesh->tp_group, OP_ALLREDUCE_SUM_F32, sendbuf,
                               recvbuf, count, tag, timeout_ms,
                               &mesh->stats.tp_bytes_sent,
                               &mesh->stats.tp_bytes_recv,
                               &mesh->stats.tp_allreduce_calls);
}

ET_MESH_API et_mesh_status_t et_mesh_tp_allreduce_sum_f64(
    et_mesh_t *mesh, const double *sendbuf, double *recvbuf, size_t count,
    uint64_t tag, int timeout_ms) {
  if (mesh == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  return subcomm_run_tensor_op(&mesh->tp_group, OP_ALLREDUCE_SUM_F64, sendbuf,
                               recvbuf, count, tag, timeout_ms,
                               &mesh->stats.tp_bytes_sent,
                               &mesh->stats.tp_bytes_recv,
                               &mesh->stats.tp_allreduce_calls);
}

ET_MESH_API et_mesh_status_t et_mesh_tp_allgather_f32(
    et_mesh_t *mesh, const float *sendbuf, size_t send_count, float *recvbuf,
    uint64_t tag, int timeout_ms) {
  if (mesh == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  return subcomm_run_tensor_op(&mesh->tp_group, OP_ALLGATHER_F32, sendbuf,
                               recvbuf, send_count, tag, timeout_ms,
                               &mesh->stats.tp_bytes_sent,
                               &mesh->stats.tp_bytes_recv,
                               &mesh->stats.tp_allgather_calls);
}

ET_MESH_API et_mesh_status_t et_mesh_dp_allreduce_avg_f32(
    et_mesh_t *mesh, const float *sendbuf, float *recvbuf, size_t count,
    uint64_t tag, int timeout_ms) {
  if (mesh == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  return subcomm_run_tensor_op(&mesh->dp_group, OP_ALLREDUCE_AVG_F32, sendbuf,
                               recvbuf, count, tag, timeout_ms,
                               &mesh->stats.dp_bytes_sent,
                               &mesh->stats.dp_bytes_recv,
                               &mesh->stats.dp_allreduce_calls);
}

ET_MESH_API et_mesh_status_t et_mesh_dp_allreduce_avg_f64(
    et_mesh_t *mesh, const double *sendbuf, double *recvbuf, size_t count,
    uint64_t tag, int timeout_ms) {
  if (mesh == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  return subcomm_run_tensor_op(&mesh->dp_group, OP_ALLREDUCE_AVG_F64, sendbuf,
                               recvbuf, count, tag, timeout_ms,
                               &mesh->stats.dp_bytes_sent,
                               &mesh->stats.dp_bytes_recv,
                               &mesh->stats.dp_allreduce_calls);
}

ET_MESH_API et_mesh_status_t et_mesh_consensus_or_u64(
    et_mesh_t *mesh, uint64_t local_flags, uint64_t *out_global_flags,
    uint64_t tag, int timeout_ms) {
  if (mesh == NULL || out_global_flags == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  uint64_t deadline_ns = 0u;
  if (compute_deadline_ns(timeout_ms, &deadline_ns) != 0) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  uint64_t tp_or = 0u;
  et_mesh_status_t st = subcomm_consensus_or_u64(
      &mesh->tp_group, local_flags, &tp_or, tag, deadline_ns,
      &mesh->stats.tp_bytes_sent, &mesh->stats.tp_bytes_recv,
      &mesh->stats.tp_consensus_calls);
  if (st != ET_MESH_OK) {
    return st;
  }
  uint64_t dp_or = 0u;
  st = subcomm_consensus_or_u64(
      &mesh->dp_group, tp_or, &dp_or, tag ^ UINT64_C(0x4450434f4e53454e),
      deadline_ns, &mesh->stats.dp_bytes_sent, &mesh->stats.dp_bytes_recv,
      &mesh->stats.dp_consensus_calls);
  if (st != ET_MESH_OK) {
    return st;
  }
  *out_global_flags = dp_or;
  return ET_MESH_OK;
}

/* --- 3. Model Config, Parameter Layout & Deterministic Data --- */

static uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

static double splitmix64_unit_signed(uint64_t *state) {
  const uint32_t u = (uint32_t)(splitmix64_next(state) >> 40u);
  return ((double)u / 8388608.0) - 1.0;
}

ET_MESH_API et_mesh_status_t et_mesh_validate_configs(
    const et_mesh_model_config_v1 *mcfg, const et_mesh_optim_config_v1 *ocfg,
    uint32_t dp_size, uint32_t tp_size) {
  if (mcfg == NULL || ocfg == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  et_mesh_status_t st = et_mesh_validate_topology(dp_size, tp_size, 0u, NULL);
  if (st != ET_MESH_OK) {
    return st;
  }
  if (mcfg->d_model == 0u || mcfg->d_ff == 0u || mcfg->num_classes == 0u ||
      mcfg->global_batch == 0u || mcfg->reserved != 0u ||
      !isfinite(mcfg->rms_eps) || mcfg->rms_eps <= 0.0f) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if ((mcfg->d_ff % tp_size) != 0u || (mcfg->global_batch % dp_size) != 0u) {
    return ET_MESH_ERR_DIVISIBILITY;
  }
  if (!isfinite(ocfg->lr) || ocfg->lr < 0.0 || !isfinite(ocfg->beta1) ||
      ocfg->beta1 < 0.0 || ocfg->beta1 >= 1.0 || !isfinite(ocfg->beta2) ||
      ocfg->beta2 < 0.0 || ocfg->beta2 >= 1.0 || !isfinite(ocfg->eps) ||
      ocfg->eps <= 0.0 || !isfinite(ocfg->weight_decay) ||
      ocfg->weight_decay < 0.0 || !isfinite(ocfg->max_grad_norm) ||
      ocfg->max_grad_norm < 0.0) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  uint64_t full_p = 0u, rank_p = 0u, rep_p = 0u, sh_p = 0u;
  st = et_mesh_param_counts(mcfg, tp_size, &full_p, &rank_p, &rep_p, &sh_p);
  if (st != ET_MESH_OK) {
    return st;
  }
  uint64_t full_bytes = 0u;
  if (mul_u64_overflow(full_p, 4u, &full_bytes) ||
      full_bytes > ET_MESH_MAX_BUFFER_BYTES) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_param_counts(
    const et_mesh_model_config_v1 *mcfg, uint32_t tp_size,
    uint64_t *out_full_params, uint64_t *out_rank_params,
    uint64_t *out_replicated_params, uint64_t *out_rank_sharded_params) {
  if (mcfg == NULL || !valid_axis_size(tp_size)) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if (mcfg->d_model == 0u || mcfg->d_ff == 0u || mcfg->num_classes == 0u) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if ((mcfg->d_ff % tp_size) != 0u) {
    return ET_MESH_ERR_DIVISIBILITY;
  }
  const uint64_t d = (uint64_t)mcfg->d_model;
  const uint64_t h = (uint64_t)mcfg->d_ff;
  const uint64_t h_loc = h / (uint64_t)tp_size;
  const uint64_t c = (uint64_t)mcfg->num_classes;

  uint64_t cd = 0u, rep = 0u, hd = 0u, full_sh = 0u, rank_sh = 0u, full_tot = 0u,
           rank_tot = 0u;
  if (mul_u64_overflow(c, d, &cd) || add_u64_overflow(d, d, &rep) ||
      add_u64_overflow(rep, cd, &rep) || add_u64_overflow(rep, c, &rep) ||
      mul_u64_overflow(h, d, &hd) || mul_u64_overflow(3u, hd, &full_sh) ||
      mul_u64_overflow(3u, h_loc * d, &rank_sh) ||
      add_u64_overflow(rep, full_sh, &full_tot) ||
      add_u64_overflow(rep, rank_sh, &rank_tot)) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if (out_full_params != NULL) {
    *out_full_params = full_tot;
  }
  if (out_rank_params != NULL) {
    *out_rank_params = rank_tot;
  }
  if (out_replicated_params != NULL) {
    *out_replicated_params = rep;
  }
  if (out_rank_sharded_params != NULL) {
    *out_rank_sharded_params = rank_sh;
  }
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_init_full_params(
    const et_mesh_model_config_v1 *mcfg, uint64_t seed, float *out_full_params,
    size_t full_param_count) {
  if (mcfg == NULL || out_full_params == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  uint64_t expected = 0u;
  et_mesh_status_t st =
      et_mesh_param_counts(mcfg, 1u, &expected, NULL, NULL, NULL);
  if (st != ET_MESH_OK) {
    return st;
  }
  if ((uint64_t)full_param_count != expected) {
    return ET_MESH_ERR_INVALID_ARG;
  }

  const size_t d = (size_t)mcfg->d_model;
  const size_t h = (size_t)mcfg->d_ff;
  const size_t c = (size_t)mcfg->num_classes;
  uint64_t rng = seed ^ UINT64_C(0x45543244494e4954);
  size_t off = 0u;

  /* gamma [d] */
  for (size_t i = 0u; i < d; ++i) {
    out_full_params[off++] =
        canonical_f32((float)(1.0 + 0.1 * splitmix64_unit_signed(&rng)));
  }
  /* b_down [d] */
  for (size_t i = 0u; i < d; ++i) {
    out_full_params[off++] =
        canonical_f32((float)(0.05 * splitmix64_unit_signed(&rng)));
  }
  /* W_head [c * d] */
  for (size_t i = 0u; i < c * d; ++i) {
    out_full_params[off++] =
        canonical_f32((float)(0.25 * splitmix64_unit_signed(&rng)));
  }
  /* b_head [c] */
  for (size_t i = 0u; i < c; ++i) {
    out_full_params[off++] =
        canonical_f32((float)(0.05 * splitmix64_unit_signed(&rng)));
  }
  /* W_gate [h * d], W_up [h * d], W_down [d * h] */
  for (size_t i = 0u; i < 3u * h * d; ++i) {
    out_full_params[off++] =
        canonical_f32((float)(0.25 * splitmix64_unit_signed(&rng)));
  }
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_make_microbatch(
    const et_mesh_model_config_v1 *mcfg, uint32_t dp_rank, uint32_t dp_size,
    uint64_t step_idx, float *out_x, int32_t *out_targets) {
  if (mcfg == NULL || out_x == NULL || out_targets == NULL ||
      !valid_axis_size(dp_size) || dp_rank >= dp_size || mcfg->d_model == 0u ||
      mcfg->num_classes == 0u || mcfg->global_batch == 0u) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if ((mcfg->global_batch % dp_size) != 0u) {
    return ET_MESH_ERR_DIVISIBILITY;
  }
  const uint32_t local_batch = mcfg->global_batch / dp_size;
  const uint32_t d = mcfg->d_model;
  const uint32_t c = mcfg->num_classes;

  for (uint32_t n = 0u; n < local_batch; ++n) {
    const uint32_t sample_idx = dp_rank * local_batch + n;
    const uint32_t target = (uint32_t)(((uint64_t)sample_idx + step_idx) % c);
    out_targets[n] = (int32_t)target;

    uint64_t rng = UINT64_C(0x5450445032303236) ^
                   (step_idx * UINT64_C(0x9e3779b97f4a7c15)) ^
                   ((uint64_t)sample_idx * UINT64_C(0xbf58476d1ce4e5b9));
    for (uint32_t j = 0u; j < d; ++j) {
      const double noise = 0.6 * splitmix64_unit_signed(&rng);
      const double signal = ((j % c) == target) ? 0.75 : -0.15;
      out_x[(size_t)n * d + j] = canonical_f32((float)(noise + signal));
    }
  }
  return ET_MESH_OK;
}

/* --- Parameter Offsets Inside Packed Local Buffer --- */

static size_t off_gamma(void) { return 0u; }
static size_t off_b_down(const et_mesh_engine_t *e) {
  return (size_t)e->mcfg.d_model;
}
static size_t off_w_head(const et_mesh_engine_t *e) {
  return 2u * (size_t)e->mcfg.d_model;
}
static size_t off_b_head(const et_mesh_engine_t *e) {
  return 2u * (size_t)e->mcfg.d_model +
         (size_t)e->mcfg.num_classes * (size_t)e->mcfg.d_model;
}
static size_t off_w_gate(const et_mesh_engine_t *e) {
  return (size_t)e->replicated_params;
}
static size_t off_w_up(const et_mesh_engine_t *e) {
  return (size_t)e->replicated_params +
         (size_t)e->d_ff_local * (size_t)e->mcfg.d_model;
}
static size_t off_w_down(const et_mesh_engine_t *e) {
  return (size_t)e->replicated_params +
         2u * (size_t)e->d_ff_local * (size_t)e->mcfg.d_model;
}

/* --- 4. 2D Hybrid-Parallel Training Engine --- */

ET_MESH_API et_mesh_status_t et_mesh_engine_create(
    et_mesh_t *mesh, const et_mesh_model_config_v1 *mcfg,
    const et_mesh_optim_config_v1 *ocfg, et_mesh_engine_t **out_engine) {
  if (out_engine == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  *out_engine = NULL;
  if (mesh == NULL || mcfg == NULL || ocfg == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  et_mesh_status_t st = et_mesh_validate_configs(
      mcfg, ocfg, mesh->coords.dp_size, mesh->coords.tp_size);
  if (st != ET_MESH_OK) {
    return st;
  }

  et_mesh_engine_t *e = (et_mesh_engine_t *)calloc(1u, sizeof(*e));
  if (e == NULL) {
    return ET_MESH_ERR_ALLOC;
  }
  e->mesh = mesh;
  e->mcfg = *mcfg;
  e->ocfg = *ocfg;
  e->d_ff_local = mcfg->d_ff / mesh->coords.tp_size;
  e->local_batch = mcfg->global_batch / mesh->coords.dp_size;

  st = et_mesh_param_counts(mcfg, mesh->coords.tp_size, &e->full_total_params,
                            &e->rank_total_params, &e->replicated_params,
                            &e->rank_sharded_params);
  if (st != ET_MESH_OK) {
    et_mesh_engine_destroy(e);
    return st;
  }
  e->full_sharded_params =
      e->rank_sharded_params * (uint64_t)mesh->coords.tp_size;

  const size_t rp = (size_t)e->rank_total_params;
  const size_t bd = (size_t)e->local_batch * (size_t)mcfg->d_model;
  const size_t bh = (size_t)e->local_batch * (size_t)e->d_ff_local;
  const size_t bc = (size_t)e->local_batch * (size_t)mcfg->num_classes;

  e->params = (float *)calloc(rp, sizeof(float));
  e->grads = (float *)calloc(rp, sizeof(float));
  e->m = (float *)calloc(rp, sizeof(float));
  e->v = (float *)calloc(rp, sizeof(float));
  e->cache_x = (float *)calloc(bd, sizeof(float));
  e->cache_target = (int32_t *)calloc((size_t)e->local_batch, sizeof(int32_t));
  e->cache_rrms = (double *)calloc((size_t)e->local_batch, sizeof(double));
  e->cache_u = (double *)calloc(bd, sizeof(double));
  e->cache_gate = (double *)calloc(bh, sizeof(double));
  e->cache_up = (double *)calloc(bh, sizeof(double));
  e->cache_act = (double *)calloc(bh, sizeof(double));
  e->cache_h = (double *)calloc(bd, sizeof(double));
  e->cache_probs = (double *)calloc(bc, sizeof(double));
  e->scratch_bd_send = (float *)calloc(bd, sizeof(float));
  e->scratch_bd_recv = (float *)calloc(bd, sizeof(float));

  if (e->params == NULL || e->grads == NULL || e->m == NULL || e->v == NULL ||
      e->cache_x == NULL || e->cache_target == NULL || e->cache_rrms == NULL ||
      e->cache_u == NULL || e->cache_gate == NULL || e->cache_up == NULL ||
      e->cache_act == NULL || e->cache_h == NULL || e->cache_probs == NULL ||
      e->scratch_bd_send == NULL || e->scratch_bd_recv == NULL) {
    et_mesh_engine_destroy(e);
    return ET_MESH_ERR_ALLOC;
  }

  *out_engine = e;
  return ET_MESH_OK;
}

ET_MESH_API void et_mesh_engine_destroy(et_mesh_engine_t *engine) {
  if (engine == NULL) {
    return;
  }
  free(engine->params);
  free(engine->grads);
  free(engine->m);
  free(engine->v);
  free(engine->cache_x);
  free(engine->cache_target);
  free(engine->cache_rrms);
  free(engine->cache_u);
  free(engine->cache_gate);
  free(engine->cache_up);
  free(engine->cache_act);
  free(engine->cache_h);
  free(engine->cache_probs);
  free(engine->scratch_bd_send);
  free(engine->scratch_bd_recv);
  free(engine);
}

ET_MESH_API et_mesh_status_t et_mesh_engine_load_full_params(
    et_mesh_engine_t *engine, const float *full_params,
    size_t full_param_count) {
  if (engine == NULL || full_params == NULL ||
      (uint64_t)full_param_count != engine->full_total_params) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if (has_nonfinite_f32(full_params, full_param_count)) {
    return ET_MESH_ERR_NONFINITE;
  }

  const size_t d = (size_t)engine->mcfg.d_model;
  const size_t h = (size_t)engine->mcfg.d_ff;
  const size_t h_loc = (size_t)engine->d_ff_local;
  const size_t tp_r = (size_t)engine->mesh->coords.tp_rank;
  const size_t rep = (size_t)engine->replicated_params;

  /* 1. Copy replicated weights [gamma, b_down, W_head, b_head]. */
  for (size_t i = 0u; i < rep; ++i) {
    engine->params[i] = canonical_f32(full_params[i]);
  }

  /* 2. Slice ColumnParallel W_gate_r [h_loc, d] from W_gate [h, d]. */
  const float *full_gate = full_params + rep;
  float *loc_gate = engine->params + off_w_gate(engine);
  for (size_t k = 0u; k < h_loc; ++k) {
    for (size_t j = 0u; j < d; ++j) {
      loc_gate[k * d + j] =
          canonical_f32(full_gate[(tp_r * h_loc + k) * d + j]);
    }
  }

  /* 3. Slice ColumnParallel W_up_r [h_loc, d] from W_up [h, d]. */
  const float *full_up = full_gate + h * d;
  float *loc_up = engine->params + off_w_up(engine);
  for (size_t k = 0u; k < h_loc; ++k) {
    for (size_t j = 0u; j < d; ++j) {
      loc_up[k * d + j] = canonical_f32(full_up[(tp_r * h_loc + k) * d + j]);
    }
  }

  /* 4. Slice RowParallel W_down_r [d, h_loc] from W_down [d, h]. */
  const float *full_down = full_up + h * d;
  float *loc_down = engine->params + off_w_down(engine);
  for (size_t j = 0u; j < d; ++j) {
    for (size_t k = 0u; k < h_loc; ++k) {
      loc_down[j * h_loc + k] =
          canonical_f32(full_down[j * h + tp_r * h_loc + k]);
    }
  }

  memset(engine->grads, 0, (size_t)engine->rank_total_params * sizeof(float));
  memset(engine->m, 0, (size_t)engine->rank_total_params * sizeof(float));
  memset(engine->v, 0, (size_t)engine->rank_total_params * sizeof(float));
  engine->step = 0u;
  engine->has_forward_cache = 0;
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_engine_get_memory_report(
    const et_mesh_engine_t *engine, et_mesh_memory_report_v1 *out_report) {
  if (engine == NULL || out_report == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  out_report->dp_size = engine->mesh->coords.dp_size;
  out_report->tp_size = engine->mesh->coords.tp_size;
  out_report->dp_rank = engine->mesh->coords.dp_rank;
  out_report->tp_rank = engine->mesh->coords.tp_rank;
  out_report->full_sharded_params = engine->full_sharded_params;
  out_report->rank_sharded_params = engine->rank_sharded_params;
  out_report->replicated_params = engine->replicated_params;
  out_report->full_total_params = engine->full_total_params;
  out_report->rank_total_params = engine->rank_total_params;
  out_report->full_sharded_weight_bytes = engine->full_sharded_params * 4u;
  out_report->rank_sharded_weight_bytes = engine->rank_sharded_params * 4u;
  out_report->replicated_weight_bytes = engine->replicated_params * 4u;
  out_report->rank_total_weight_bytes = engine->rank_total_params * 4u;
  out_report->rank_optimizer_state_bytes = engine->rank_total_params * 8u;
  out_report->tp_sharding_ratio =
      (double)engine->full_sharded_params / (double)engine->rank_sharded_params;
  return ET_MESH_OK;
}

ET_MESH_API uint64_t et_mesh_engine_get_step(const et_mesh_engine_t *engine) {
  return engine == NULL ? 0u : engine->step;
}

/* Forward pass: 1 TP AllReduceSum across tp_group; ZERO DP communication. */
ET_MESH_API et_mesh_status_t et_mesh_engine_forward(
    et_mesh_engine_t *engine, const float *x_local,
    const int32_t *targets_local, uint64_t tag_base, int timeout_ms,
    double *out_local_loss) {
  if (engine == NULL || x_local == NULL || targets_local == NULL ||
      out_local_loss == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  const size_t b = (size_t)engine->local_batch;
  const size_t d = (size_t)engine->mcfg.d_model;
  const size_t h_loc = (size_t)engine->d_ff_local;
  const size_t c = (size_t)engine->mcfg.num_classes;

  for (size_t n = 0u; n < b; ++n) {
    if (targets_local[n] < 0 || (uint32_t)targets_local[n] >= engine->mcfg.num_classes) {
      return ET_MESH_ERR_INVALID_ARG;
    }
  }

  engine->has_forward_cache = 0;
  int local_nf = has_nonfinite_f32(x_local, b * d);

  const float *gamma = engine->params + off_gamma();
  const float *b_down = engine->params + off_b_down(engine);
  const float *w_head = engine->params + off_w_head(engine);
  const float *b_head = engine->params + off_b_head(engine);
  const float *w_gate = engine->params + off_w_gate(engine);
  const float *w_up = engine->params + off_w_up(engine);
  const float *w_down = engine->params + off_w_down(engine);

  if (!local_nf) {
    memcpy(engine->cache_x, x_local, b * d * sizeof(float));
    memcpy(engine->cache_target, targets_local, b * sizeof(int32_t));

    /* 1. RMSNorm + ColumnParallel SwiGLU + local RowParallel partial sum. */
    for (size_t n = 0u; n < b; ++n) {
      const float *xn = x_local + n * d;
      double ms = 0.0;
      for (size_t j = 0u; j < d; ++j) {
        const double xj = (double)xn[j];
        ms += xj * xj;
      }
      ms /= (double)d;
      const double rrms = 1.0 / sqrt(ms + (double)engine->mcfg.rms_eps);
      engine->cache_rrms[n] = rrms;
      double *un = engine->cache_u + n * d;
      for (size_t j = 0u; j < d; ++j) {
        un[j] = (double)xn[j] * rrms * (double)gamma[j];
      }

      double *gn = engine->cache_gate + n * h_loc;
      double *pn = engine->cache_up + n * h_loc;
      double *an = engine->cache_act + n * h_loc;
      for (size_t k = 0u; k < h_loc; ++k) {
        double gs = 0.0;
        double ps = 0.0;
        const float *wg_k = w_gate + k * d;
        const float *wu_k = w_up + k * d;
        for (size_t j = 0u; j < d; ++j) {
          gs += un[j] * (double)wg_k[j];
          ps += un[j] * (double)wu_k[j];
        }
        gn[k] = gs;
        pn[k] = ps;
        const double sig = 1.0 / (1.0 + exp(-gs));
        an[k] = (gs * sig) * ps;
      }

      float *z_part = engine->scratch_bd_send + n * d;
      for (size_t j = 0u; j < d; ++j) {
        double zs = 0.0;
        const float *wd_j = w_down + j * h_loc;
        for (size_t k = 0u; k < h_loc; ++k) {
          zs += an[k] * (double)wd_j[k];
        }
        if (!isfinite(zs) || !isfinite((float)zs)) {
          local_nf = 1;
        }
        z_part[j] = canonical_f32((float)zs);
      }
    }
  }

  /* If local_nf occurred, poison scratch_bd_send[0] so TP AllReduce aborts cleanly across tp_group. */
  if (local_nf) {
    engine->scratch_bd_send[0] = bits_to_f32(UINT32_C(0x7fc00000));
  }

  /* 2. Exactly 1 TP AllReduceSum across tp_group; ZERO DP communication! */
  et_mesh_status_t st = et_mesh_tp_allreduce_sum_f32(
      engine->mesh, engine->scratch_bd_send, engine->scratch_bd_recv, b * d,
      tag_base, timeout_ms);
  if (st != ET_MESH_OK) {
    return st;
  }

  /* 3. Add b_down ONCE post-TP-AllReduce + residual -> Linear Head -> Cross-Entropy. */
  double total_loss = 0.0;
  for (size_t n = 0u; n < b; ++n) {
    const float *xn = x_local + n * d;
    const float *zn = engine->scratch_bd_recv + n * d;
    double *hn = engine->cache_h + n * d;
    for (size_t j = 0u; j < d; ++j) {
      hn[j] = (double)xn[j] + (double)zn[j] + (double)b_down[j];
    }

    double *probs = engine->cache_probs + n * c;
    double max_logit = -INFINITY;
    for (size_t cls = 0u; cls < c; ++cls) {
      double lg = (double)b_head[cls];
      const float *wh_c = w_head + cls * d;
      for (size_t j = 0u; j < d; ++j) {
        lg += hn[j] * (double)wh_c[j];
      }
      if (!isfinite(lg)) {
        return ET_MESH_ERR_NONFINITE;
      }
      probs[cls] = lg;
      if (lg > max_logit) {
        max_logit = lg;
      }
    }

    double sum_exp = 0.0;
    for (size_t cls = 0u; cls < c; ++cls) {
      probs[cls] = exp(probs[cls] - max_logit);
      sum_exp += probs[cls];
    }
    if (!isfinite(sum_exp) || sum_exp <= 0.0) {
      return ET_MESH_ERR_NONFINITE;
    }
    const double lse = max_logit + log(sum_exp);
    const size_t y = (size_t)targets_local[n];
    double target_logit = (double)b_head[y];
    const float *wh_y = w_head + y * d;
    for (size_t j = 0u; j < d; ++j) {
      target_logit += hn[j] * (double)wh_y[j];
    }
    total_loss += (lse - target_logit);

    for (size_t cls = 0u; cls < c; ++cls) {
      probs[cls] /= sum_exp;
    }
  }

  const double local_loss = total_loss / (double)b;
  if (!isfinite(local_loss)) {
    return ET_MESH_ERR_NONFINITE;
  }
  engine->cache_local_loss = local_loss;
  engine->has_forward_cache = 1;
  *out_local_loss = local_loss;
  return ET_MESH_OK;
}

/* Analytic local backward pass: 1 TP AllReduceSum for dU; ZERO DP comm. */
ET_MESH_API et_mesh_status_t et_mesh_engine_backward(
    et_mesh_engine_t *engine, uint64_t tag_base, int timeout_ms,
    float *out_dx_local) {
  if (engine == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if (!engine->has_forward_cache) {
    return ET_MESH_ERR_STATE;
  }

  const size_t b = (size_t)engine->local_batch;
  const size_t d = (size_t)engine->mcfg.d_model;
  const size_t h_loc = (size_t)engine->d_ff_local;
  const size_t c = (size_t)engine->mcfg.num_classes;
  const size_t rp = (size_t)engine->rank_total_params;

  double *g_acc = (double *)calloc(rp, sizeof(double));
  double *dh_buf = (double *)calloc(b * d, sizeof(double));
  if (g_acc == NULL || dh_buf == NULL) {
    free(g_acc);
    free(dh_buf);
    return ET_MESH_ERR_ALLOC;
  }

  const float *gamma = engine->params + off_gamma();
  const float *w_head = engine->params + off_w_head(engine);
  const float *w_gate = engine->params + off_w_gate(engine);
  const float *w_up = engine->params + off_w_up(engine);
  const float *w_down = engine->params + off_w_down(engine);

  double *dgamma = g_acc + off_gamma();
  double *db_down = g_acc + off_b_down(engine);
  double *dw_head = g_acc + off_w_head(engine);
  double *db_head = g_acc + off_b_head(engine);
  double *dw_gate = g_acc + off_w_gate(engine);
  double *dw_up = g_acc + off_w_up(engine);
  double *dw_down = g_acc + off_w_down(engine);

  const double inv_b = 1.0 / (double)b;
  int local_nf = 0;

  for (size_t n = 0u; n < b; ++n) {
    const double *probs = engine->cache_probs + n * c;
    const double *hn = engine->cache_h + n * d;
    const double *un = engine->cache_u + n * d;
    const double *gn = engine->cache_gate + n * h_loc;
    const double *pn = engine->cache_up + n * h_loc;
    const double *an = engine->cache_act + n * h_loc;
    const size_t y = (size_t)engine->cache_target[n];
    double *dh = dh_buf + n * d;

    /* 1. Cross-Entropy + Linear Head backward. */
    for (size_t cls = 0u; cls < c; ++cls) {
      const double dl = (probs[cls] - (cls == y ? 1.0 : 0.0)) * inv_b;
      db_head[cls] += dl;
      const float *wh_c = w_head + cls * d;
      double *dwh_c = dw_head + cls * d;
      for (size_t j = 0u; j < d; ++j) {
        dwh_c[j] += dl * hn[j];
        dh[j] += dl * (double)wh_c[j];
      }
    }

    /* 2. b_down & RowParallel W_down_r backward (dZ == dH). */
    for (size_t j = 0u; j < d; ++j) {
      db_down[j] += dh[j];
    }

    float *du_part = engine->scratch_bd_send + n * d;
    double du_acc[256];
    double *du_tmp = (d <= 256u) ? du_acc : (double *)calloc(d, sizeof(double));
    if (du_tmp == NULL) {
      free(g_acc);
      free(dh_buf);
      return ET_MESH_ERR_ALLOC;
    }
    if (d <= 256u) {
      memset(du_acc, 0, d * sizeof(double));
    }

    for (size_t k = 0u; k < h_loc; ++k) {
      double da = 0.0;
      for (size_t j = 0u; j < d; ++j) {
        const double dz = dh[j];
        dw_down[j * h_loc + k] += dz * an[k];
        da += dz * (double)w_down[j * h_loc + k];
      }

      /* 3. SwiGLU backward: A = SiLU(G) * P. */
      const double g_val = gn[k];
      const double p_val = pn[k];
      const double sig = 1.0 / (1.0 + exp(-g_val));
      const double silu = g_val * sig;
      const double dsilu = sig * (1.0 + g_val * (1.0 - sig));
      const double dp = da * silu;
      const double dg = da * p_val * dsilu;

      /* 4. ColumnParallel W_gate_r, W_up_r backward + fused partial dU. */
      const float *wg_k = w_gate + k * d;
      const float *wu_k = w_up + k * d;
      double *dwg_k = dw_gate + k * d;
      double *dwu_k = dw_up + k * d;
      for (size_t j = 0u; j < d; ++j) {
        dwg_k[j] += dg * un[j];
        dwu_k[j] += dp * un[j];
        du_tmp[j] += dg * (double)wg_k[j] + dp * (double)wu_k[j];
      }
    }

    for (size_t j = 0u; j < d; ++j) {
      if (!isfinite(du_tmp[j]) || !isfinite((float)du_tmp[j])) {
        local_nf = 1;
      }
      du_part[j] = canonical_f32((float)du_tmp[j]);
    }
    if (du_tmp != du_acc) {
      free(du_tmp);
    }
  }

  if (local_nf) {
    engine->scratch_bd_send[0] = bits_to_f32(UINT32_C(0x7fc00000));
  }

  /* 5. Exactly 1 TP AllReduceSum across tp_group for dU! */
  et_mesh_status_t st = et_mesh_tp_allreduce_sum_f32(
      engine->mesh, engine->scratch_bd_send, engine->scratch_bd_recv, b * d,
      tag_base, timeout_ms);
  if (st != ET_MESH_OK) {
    free(g_acc);
    free(dh_buf);
    return st;
  }

  /* 6. RMSNorm backward + residual dX. */
  for (size_t n = 0u; n < b; ++n) {
    const float *xn = engine->cache_x + n * d;
    const float *du = engine->scratch_bd_recv + n * d;
    const double *dh = dh_buf + n * d;
    const double rrms = engine->cache_rrms[n];
    const double rrms2 = rrms * rrms;

    double dot = 0.0;
    for (size_t j = 0u; j < d; ++j) {
      const double xj = (double)xn[j];
      const double duj = (double)du[j];
      dgamma[j] += duj * xj * rrms;
      dot += duj * (double)gamma[j] * xj;
    }
    const double coeff = dot / (double)d;
    if (out_dx_local != NULL) {
      float *dxn = out_dx_local + n * d;
      for (size_t j = 0u; j < d; ++j) {
        const double xj = (double)xn[j];
        const double duj = (double)du[j];
        const double dx_norm =
            rrms * (duj * (double)gamma[j] - xj * rrms2 * coeff);
        const double dx_tot = dh[j] + dx_norm;
        if (!isfinite(dx_tot) || !isfinite((float)dx_tot)) {
          local_nf = 1;
        }
        dxn[j] = canonical_f32((float)dx_tot);
      }
    }
  }

  for (size_t i = 0u; i < rp; ++i) {
    const float gv = (float)g_acc[i];
    if (!isfinite(g_acc[i]) || !isfinite(gv)) {
      local_nf = 1;
    }
    engine->grads[i] = canonical_f32(gv);
  }

  free(g_acc);
  free(dh_buf);
  return local_nf ? ET_MESH_ERR_NONFINITE : ET_MESH_OK;
}

static et_mesh_status_t engine_train_step_internal(
    et_mesh_engine_t *engine, const float *x_local,
    const int32_t *targets_local, int inject_nf, uint32_t target_global_rank,
    size_t grad_index, uint32_t bad_f32_bits, uint64_t tag_base, int timeout_ms,
    et_mesh_step_info_v1 *out_info) {
  if (engine == NULL || x_local == NULL || targets_local == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  if (inject_nf && grad_index >= (size_t)engine->rank_total_params) {
    return ET_MESH_ERR_INVALID_ARG;
  }

  double local_loss = 0.0;
  et_mesh_status_t fwd_st = et_mesh_engine_forward(
      engine, x_local, targets_local, tag_base + 0u, timeout_ms, &local_loss);
  if (fwd_st != ET_MESH_OK && fwd_st != ET_MESH_ERR_NONFINITE) {
    return fwd_st;
  }

  et_mesh_status_t bwd_st = fwd_st;
  if (fwd_st == ET_MESH_OK) {
    bwd_st = et_mesh_engine_backward(engine, tag_base + 1u, timeout_ms, NULL);
    if (bwd_st != ET_MESH_OK && bwd_st != ET_MESH_ERR_NONFINITE) {
      return bwd_st;
    }
  }

  /* Optional single-rank non-finite fault injection for 2D consensus verification. */
  if (inject_nf && engine->mesh->coords.global_rank == target_global_rank) {
    engine->grads[grad_index] = bits_to_f32(bad_f32_bits);
  }

  const size_t rp = (size_t)engine->rank_total_params;
  const uint64_t local_nf =
      (fwd_st != ET_MESH_OK || bwd_st != ET_MESH_OK || !isfinite(local_loss) ||
       has_nonfinite_f32(engine->grads, rp))
          ? 1u
          : 0u;

  /* 2D Mesh-Wide Consensus #1 (tp_group -> dp_group): zero split-brain! */
  uint64_t global_nf = 0u;
  et_mesh_status_t st = et_mesh_consensus_or_u64(
      engine->mesh, local_nf, &global_nf, tag_base + 2u, timeout_ms);
  if (st != ET_MESH_OK) {
    return st;
  }
  if (global_nf != 0u) {
    return ET_MESH_ERR_NONFINITE;
  }

  /* DP AllReduceAvg on ALL local parameter gradients across dp_group!
   * Matching tp_rank replicas receive bitwise-identical averaged gradients. */
  st = et_mesh_dp_allreduce_avg_f32(engine->mesh, engine->grads, engine->grads,
                                    rp, tag_base + 3u, timeout_ms);
  if (st != ET_MESH_OK && st != ET_MESH_ERR_NONFINITE) {
    return st;
  }
  const int dp_grad_nf = (st == ET_MESH_ERR_NONFINITE);

  /* DP AllReduceAvg_f64 for exact global mean loss across dp_group. */
  double global_loss = local_loss;
  if (!dp_grad_nf) {
    st = et_mesh_dp_allreduce_avg_f64(engine->mesh, &local_loss, &global_loss,
                                      1u, tag_base + 4u, timeout_ms);
    if (st != ET_MESH_OK && st != ET_MESH_ERR_NONFINITE) {
      return st;
    }
  }
  const int dp_loss_nf = (st == ET_MESH_ERR_NONFINITE);

  /* Global L2 norm across the 2D mesh:
   * - Sum squared norms of TP-sharded gradients across tp_group in binary64.
   * - Add squared norm of replicated gradients once (already identical across tp_group). */
  const size_t rep = (size_t)engine->replicated_params;
  double rep_sq = 0.0;
  for (size_t i = 0u; i < rep; ++i) {
    const double g = (double)engine->grads[i];
    rep_sq += g * g;
  }
  double sh_sq_local = 0.0;
  for (size_t i = rep; i < rp; ++i) {
    const double g = (double)engine->grads[i];
    sh_sq_local += g * g;
  }

  double sh_sq_full = sh_sq_local;
  int norm_nf = (dp_grad_nf || dp_loss_nf || !isfinite(rep_sq) ||
                 !isfinite(sh_sq_local));
  if (norm_nf) {
    sh_sq_local = bits_to_f64(UINT64_C(0x7ff8000000000000));
  }
  st = et_mesh_tp_allreduce_sum_f64(engine->mesh, &sh_sq_local, &sh_sq_full, 1u,
                                    tag_base + 5u, timeout_ms);
  if (st != ET_MESH_OK && st != ET_MESH_ERR_NONFINITE) {
    return st;
  }
  if (st == ET_MESH_ERR_NONFINITE) {
    norm_nf = 1;
  }

  const double total_sq = sh_sq_full + rep_sq;
  const double grad_norm = norm_nf ? NAN : sqrt(total_sq);
  const uint64_t post_nf =
      (norm_nf || !isfinite(total_sq) || !isfinite(grad_norm) ||
       !isfinite(global_loss))
          ? 1u
          : 0u;

  /* 2D Mesh-Wide Consensus #2 before mutating weights/moments/step. */
  uint64_t post_global_nf = 0u;
  st = et_mesh_consensus_or_u64(engine->mesh, post_nf, &post_global_nf,
                                tag_base + 6u, timeout_ms);
  if (st != ET_MESH_OK) {
    return st;
  }
  if (post_global_nf != 0u) {
    return ET_MESH_ERR_NONFINITE;
  }

  /* Global L2 gradient clipping. */
  double clip_scale = 1.0;
  uint32_t clipped = 0u;
  if (engine->ocfg.max_grad_norm > 0.0 &&
      grad_norm > engine->ocfg.max_grad_norm) {
    clip_scale = engine->ocfg.max_grad_norm / grad_norm;
    clipped = 1u;
  }

  /* Compute candidate AdamW update in scratch buffers for 100% atomic commit. */
  float *new_p = (float *)malloc(rp * sizeof(float));
  float *new_m = (float *)malloc(rp * sizeof(float));
  float *new_v = (float *)malloc(rp * sizeof(float));
  if (new_p == NULL || new_m == NULL || new_v == NULL) {
    free(new_p);
    free(new_m);
    free(new_v);
    return ET_MESH_ERR_ALLOC;
  }

  const uint64_t next_step = engine->step + 1u;
  const double b1 = engine->ocfg.beta1;
  const double b2 = engine->ocfg.beta2;
  const double lr = engine->ocfg.lr;
  const double eps = engine->ocfg.eps;
  const double wd = engine->ocfg.weight_decay;
  const double bc1 = 1.0 - pow(b1, (double)next_step);
  const double bc2 = 1.0 - pow(b2, (double)next_step);
  const double decay_factor = 1.0 - lr * wd;

  int update_nf = 0;
  for (size_t i = 0u; i < rp; ++i) {
    const double g = (double)engine->grads[i] * clip_scale;
    const double m_new = b1 * (double)engine->m[i] + (1.0 - b1) * g;
    const double v_new = b2 * (double)engine->v[i] + (1.0 - b2) * g * g;
    const double m_hat = m_new / bc1;
    const double v_hat = v_new / bc2;
    const double w_new =
        (double)engine->params[i] * decay_factor - lr * m_hat / (sqrt(v_hat) + eps);
    const float pf = (float)w_new;
    const float mf = (float)m_new;
    const float vf = (float)v_new;
    if (!isfinite(pf) || !isfinite(mf) || !isfinite(vf) || vf < 0.0f) {
      update_nf = 1;
      break;
    }
    new_p[i] = canonical_f32(pf);
    new_m[i] = canonical_f32(mf);
    new_v[i] = canonical_f32(vf);
  }

  uint64_t upd_global_nf = 0u;
  st = et_mesh_consensus_or_u64(engine->mesh, (uint64_t)update_nf,
                                &upd_global_nf, tag_base + 7u, timeout_ms);
  if (st != ET_MESH_OK || upd_global_nf != 0u) {
    free(new_p);
    free(new_m);
    free(new_v);
    return (st != ET_MESH_OK) ? st : ET_MESH_ERR_NONFINITE;
  }

  memcpy(engine->params, new_p, rp * sizeof(float));
  memcpy(engine->m, new_m, rp * sizeof(float));
  memcpy(engine->v, new_v, rp * sizeof(float));
  free(new_p);
  free(new_m);
  free(new_v);
  engine->step = next_step;

  if (out_info != NULL) {
    out_info->step = next_step;
    out_info->local_loss = local_loss;
    out_info->global_loss = global_loss;
    out_info->global_grad_norm = grad_norm;
    out_info->clip_scale = clip_scale;
    out_info->clipped = clipped;
    out_info->reserved = 0u;
  }
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_engine_train_step(
    et_mesh_engine_t *engine, const float *x_local,
    const int32_t *targets_local, uint64_t tag_base, int timeout_ms,
    et_mesh_step_info_v1 *out_info) {
  return engine_train_step_internal(engine, x_local, targets_local, 0, 0u, 0u,
                                    0u, tag_base, timeout_ms, out_info);
}

ET_MESH_API et_mesh_status_t et_mesh_engine_train_step_inject_nonfinite(
    et_mesh_engine_t *engine, const float *x_local,
    const int32_t *targets_local, uint32_t target_global_rank,
    size_t grad_index, uint32_t bad_f32_bits, uint64_t tag_base, int timeout_ms,
    et_mesh_step_info_v1 *out_info) {
  return engine_train_step_internal(engine, x_local, targets_local, 1,
                                    target_global_rank, grad_index,
                                    bad_f32_bits, tag_base, timeout_ms,
                                    out_info);
}

ET_MESH_API et_mesh_status_t et_mesh_engine_gather_full_params(
    et_mesh_engine_t *engine, float *out_full_params, size_t full_param_count,
    uint64_t tag, int timeout_ms) {
  if (engine == NULL || out_full_params == NULL ||
      (uint64_t)full_param_count != engine->full_total_params) {
    return ET_MESH_ERR_INVALID_ARG;
  }

  const size_t d = (size_t)engine->mcfg.d_model;
  const size_t h = (size_t)engine->mcfg.d_ff;
  const size_t h_loc = (size_t)engine->d_ff_local;
  const size_t tp_s = (size_t)engine->mesh->coords.tp_size;
  const size_t rep = (size_t)engine->replicated_params;
  const size_t sh_loc = (size_t)engine->rank_sharded_params;

  float *gathered = (float *)malloc(tp_s * sh_loc * sizeof(float));
  if (gathered == NULL) {
    return ET_MESH_ERR_ALLOC;
  }
  et_mesh_status_t st = et_mesh_tp_allgather_f32(
      engine->mesh, engine->params + rep, sh_loc, gathered, tag, timeout_ms);
  if (st != ET_MESH_OK) {
    free(gathered);
    return st;
  }

  /* 1. Copy replicated weights. */
  memcpy(out_full_params, engine->params, rep * sizeof(float));

  /* 2. Unpack each tp_rank's [W_gate_r, W_up_r, W_down_r] into dense layout. */
  float *full_gate = out_full_params + rep;
  float *full_up = full_gate + h * d;
  float *full_down = full_up + h * d;

  for (size_t r = 0u; r < tp_s; ++r) {
    const float *rb = gathered + r * sh_loc;
    const float *rg = rb;
    const float *ru = rg + h_loc * d;
    const float *rd = ru + h_loc * d;

    memcpy(full_gate + r * h_loc * d, rg, h_loc * d * sizeof(float));
    memcpy(full_up + r * h_loc * d, ru, h_loc * d * sizeof(float));
    for (size_t j = 0u; j < d; ++j) {
      memcpy(full_down + j * h + r * h_loc, rd + j * h_loc,
             h_loc * sizeof(float));
    }
  }

  free(gathered);
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_engine_get_local_params(
    const et_mesh_engine_t *engine, float *out_params,
    size_t local_param_count) {
  if (engine == NULL || out_params == NULL ||
      (uint64_t)local_param_count != engine->rank_total_params) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  memcpy(out_params, engine->params, local_param_count * sizeof(float));
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_engine_get_local_grads(
    const et_mesh_engine_t *engine, float *out_grads,
    size_t local_param_count) {
  if (engine == NULL || out_grads == NULL ||
      (uint64_t)local_param_count != engine->rank_total_params) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  memcpy(out_grads, engine->grads, local_param_count * sizeof(float));
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_engine_get_local_moments(
    const et_mesh_engine_t *engine, float *out_m, float *out_v,
    size_t local_param_count) {
  if (engine == NULL || out_m == NULL || out_v == NULL ||
      (uint64_t)local_param_count != engine->rank_total_params) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  memcpy(out_m, engine->m, local_param_count * sizeof(float));
  memcpy(out_v, engine->v, local_param_count * sizeof(float));
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_engine_local_state_sha256(
    const et_mesh_engine_t *engine, uint8_t out_digest[32]) {
  if (engine == NULL || out_digest == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  mesh_sha256_ctx ctx;
  sha256_init(&ctx);
  uint8_t step_le[8];
  put_u64_le(step_le, engine->step);
  sha256_update(&ctx, step_le, sizeof(step_le));

  const size_t rp = (size_t)engine->rank_total_params;
  uint8_t word[4];
  for (size_t i = 0u; i < rp; ++i) {
    put_u32_le(word, f32_to_bits(canonical_f32(engine->params[i])));
    sha256_update(&ctx, word, 4u);
  }
  for (size_t i = 0u; i < rp; ++i) {
    put_u32_le(word, f32_to_bits(canonical_f32(engine->m[i])));
    sha256_update(&ctx, word, 4u);
  }
  for (size_t i = 0u; i < rp; ++i) {
    put_u32_le(word, f32_to_bits(canonical_f32(engine->v[i])));
    sha256_update(&ctx, word, 4u);
  }
  sha256_final(&ctx, out_digest);
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_engine_gather_full_params_sha256(
    et_mesh_engine_t *engine, uint64_t tag, int timeout_ms,
    uint8_t out_digest[32]) {
  if (engine == NULL || out_digest == NULL) {
    return ET_MESH_ERR_INVALID_ARG;
  }
  const size_t fp = (size_t)engine->full_total_params;
  float *full = (float *)malloc(fp * sizeof(float));
  if (full == NULL) {
    return ET_MESH_ERR_ALLOC;
  }
  et_mesh_status_t st =
      et_mesh_engine_gather_full_params(engine, full, fp, tag, timeout_ms);
  if (st != ET_MESH_OK) {
    free(full);
    return st;
  }
  mesh_sha256_ctx ctx;
  sha256_init(&ctx);
  uint8_t word[4];
  for (size_t i = 0u; i < fp; ++i) {
    put_u32_le(word, f32_to_bits(canonical_f32(full[i])));
    sha256_update(&ctx, word, 4u);
  }
  sha256_final(&ctx, out_digest);
  free(full);
  return ET_MESH_OK;
}

/* --- 5. Versioned SHA-256 Checksummed Per-Shard Checkpoint (ET2D1000 v1.0) --- */

static int write_all_fd(int fd, const uint8_t *buf, size_t len) {
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

static int read_all_fd(int fd, uint8_t *buf, size_t len) {
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = read(fd, buf + off, len - off);
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

static void fsync_parent_dir(const char *path) {
  const char *slash = strrchr(path, '/');
  char dirbuf[PATH_MAX];
  const char *dir = ".";
  if (slash != NULL) {
    const size_t dlen = (slash == path) ? 1u : (size_t)(slash - path);
    if (dlen < sizeof(dirbuf)) {
      memcpy(dirbuf, path, dlen);
      dirbuf[dlen] = '\0';
      dir = dirbuf;
    }
  }
  const int dfd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dfd >= 0) {
    (void)fsync(dfd);
    (void)close(dfd);
  }
}

ET_MESH_API et_mesh_status_t et_mesh_engine_save_checkpoint(
    const et_mesh_engine_t *engine, const char *path) {
  if (engine == NULL || path == NULL || path[0] == '\0') {
    return ET_MESH_ERR_INVALID_ARG;
  }

  /* Refuse existing symlink or non-regular destination path. */
  struct stat st_dest;
  if (lstat(path, &st_dest) == 0) {
    if (S_ISLNK(st_dest.st_mode) || !S_ISREG(st_dest.st_mode)) {
      return ET_MESH_ERR_IO;
    }
  } else if (errno != ENOENT) {
    return ET_MESH_ERR_IO;
  }

  const size_t rp = (size_t)engine->rank_total_params;
  const uint64_t payload_bytes_u64 = (uint64_t)rp * 3u * 4u;
  const size_t payload_bytes = (size_t)payload_bytes_u64;
  uint8_t *payload = (uint8_t *)malloc(payload_bytes);
  if (payload == NULL) {
    return ET_MESH_ERR_ALLOC;
  }

  size_t off = 0u;
  for (size_t i = 0u; i < rp; ++i) {
    put_u32_le(payload + off, f32_to_bits(canonical_f32(engine->params[i])));
    off += 4u;
  }
  for (size_t i = 0u; i < rp; ++i) {
    put_u32_le(payload + off, f32_to_bits(canonical_f32(engine->m[i])));
    off += 4u;
  }
  for (size_t i = 0u; i < rp; ++i) {
    put_u32_le(payload + off, f32_to_bits(canonical_f32(engine->v[i])));
    off += 4u;
  }

  uint8_t header[ET_MESH_CKPT_HEADER_BYTES];
  memset(header, 0, sizeof(header));
  memcpy(header + 0u, ET_MESH_CKPT_MAGIC, ET_MESH_CKPT_MAGIC_BYTES);
  put_u32_le(header + 8u, ET_MESH_CKPT_VERSION_MAJOR);
  put_u32_le(header + 12u, ET_MESH_CKPT_VERSION_MINOR);
  put_u32_le(header + 16u, engine->mesh->coords.dp_size);
  put_u32_le(header + 20u, engine->mesh->coords.tp_size);
  put_u32_le(header + 24u, engine->mesh->coords.dp_rank);
  put_u32_le(header + 28u, engine->mesh->coords.tp_rank);
  put_u32_le(header + 32u, engine->mcfg.d_model);
  put_u32_le(header + 36u, engine->mcfg.d_ff);
  put_u32_le(header + 40u, engine->mcfg.num_classes);
  put_u32_le(header + 44u, engine->mcfg.global_batch);
  put_u32_le(header + 48u, f32_to_bits(engine->mcfg.rms_eps));
  put_u32_le(header + 52u, 0u);
  put_u64_le(header + 56u, engine->step);
  put_u64_le(header + 64u, engine->rank_total_params);
  put_u64_le(header + 72u, f64_to_bits(engine->ocfg.lr));
  put_u64_le(header + 80u, f64_to_bits(engine->ocfg.beta1));
  put_u64_le(header + 88u, f64_to_bits(engine->ocfg.beta2));
  put_u64_le(header + 96u, f64_to_bits(engine->ocfg.eps));
  put_u64_le(header + 104u, f64_to_bits(engine->ocfg.weight_decay));
  put_u64_le(header + 112u, f64_to_bits(engine->ocfg.max_grad_norm));
  put_u64_le(header + 120u, payload_bytes_u64);

  mesh_sha256_ctx sha;
  sha256_init(&sha);
  sha256_update(&sha, header, 128u);
  sha256_update(&sha, payload, payload_bytes);
  sha256_final(&sha, header + 128u);

  /* Atomic 0600 sibling temporary file + fsync + rename. */
  char tmp_path[PATH_MAX];
  int fd = -1;
  for (int attempt = 0; attempt < 32; ++attempt) {
    uint64_t rnd = 0u;
    if (getrandom(&rnd, sizeof(rnd), 0u) != (ssize_t)sizeof(rnd)) {
      rnd = (uint64_t)getpid() ^ ((uint64_t)attempt * UINT64_C(0x9e3779b97f4a7c15));
    }
    const int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%016llx", path,
                           (unsigned long long)rnd);
    if (n <= 0 || (size_t)n >= sizeof(tmp_path)) {
      free(payload);
      return ET_MESH_ERR_INVALID_ARG;
    }
    fd = open(tmp_path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
              0600);
    if (fd >= 0) {
      break;
    }
    if (errno != EEXIST) {
      free(payload);
      return ET_MESH_ERR_IO;
    }
  }
  if (fd < 0) {
    free(payload);
    return ET_MESH_ERR_IO;
  }

  et_mesh_status_t st = ET_MESH_OK;
  if (write_all_fd(fd, header, sizeof(header)) != 0 ||
      write_all_fd(fd, payload, payload_bytes) != 0 || fsync(fd) != 0) {
    st = ET_MESH_ERR_IO;
  }
  if (close(fd) != 0 && st == ET_MESH_OK) {
    st = ET_MESH_ERR_IO;
  }
  free(payload);

  if (st != ET_MESH_OK) {
    (void)unlink(tmp_path);
    return st;
  }

  /* Re-check destination is not a symlink immediately before rename. */
  if (lstat(path, &st_dest) == 0 &&
      (S_ISLNK(st_dest.st_mode) || !S_ISREG(st_dest.st_mode))) {
    (void)unlink(tmp_path);
    return ET_MESH_ERR_IO;
  }

  if (rename(tmp_path, path) != 0) {
    (void)unlink(tmp_path);
    return ET_MESH_ERR_IO;
  }
  fsync_parent_dir(path);
  return ET_MESH_OK;
}

ET_MESH_API et_mesh_status_t et_mesh_engine_load_checkpoint(
    et_mesh_engine_t *engine, const char *path) {
  if (engine == NULL || path == NULL || path[0] == '\0') {
    return ET_MESH_ERR_INVALID_ARG;
  }

  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) {
    return ET_MESH_ERR_IO;
  }

  struct stat st_file;
  if (fstat(fd, &st_file) != 0 || !S_ISREG(st_file.st_mode)) {
    (void)close(fd);
    return ET_MESH_ERR_IO;
  }
  if (st_file.st_size < (off_t)ET_MESH_CKPT_HEADER_BYTES) {
    (void)close(fd);
    return ET_MESH_ERR_TRUNCATED;
  }

  uint8_t header[ET_MESH_CKPT_HEADER_BYTES];
  if (read_all_fd(fd, header, sizeof(header)) != 0) {
    (void)close(fd);
    return ET_MESH_ERR_IO;
  }

  if (memcmp(header + 0u, ET_MESH_CKPT_MAGIC, ET_MESH_CKPT_MAGIC_BYTES) != 0) {
    (void)close(fd);
    return ET_MESH_ERR_BAD_MAGIC;
  }
  const uint32_t ver_maj = get_u32_le(header + 8u);
  const uint32_t ver_min = get_u32_le(header + 12u);
  if (ver_maj != ET_MESH_CKPT_VERSION_MAJOR ||
      ver_min != ET_MESH_CKPT_VERSION_MINOR) {
    (void)close(fd);
    return ET_MESH_ERR_VERSION_MISMATCH;
  }

  const uint64_t payload_bytes_u64 = get_u64_le(header + 120u);
  if (payload_bytes_u64 > (uint64_t)ET_MESH_MAX_BUFFER_BYTES * 3u ||
      (uint64_t)st_file.st_size !=
          (uint64_t)ET_MESH_CKPT_HEADER_BYTES + payload_bytes_u64) {
    (void)close(fd);
    return ET_MESH_ERR_TRUNCATED;
  }

  const size_t payload_bytes = (size_t)payload_bytes_u64;
  uint8_t *payload = (uint8_t *)malloc(payload_bytes > 0u ? payload_bytes : 1u);
  if (payload == NULL) {
    (void)close(fd);
    return ET_MESH_ERR_ALLOC;
  }
  if (payload_bytes > 0u && read_all_fd(fd, payload, payload_bytes) != 0) {
    free(payload);
    (void)close(fd);
    return ET_MESH_ERR_IO;
  }
  (void)close(fd);

  /* Verify SHA-256 checksum over header[0..128) || payload. */
  uint8_t computed[32];
  mesh_sha256_ctx sha;
  sha256_init(&sha);
  sha256_update(&sha, header, 128u);
  if (payload_bytes > 0u) {
    sha256_update(&sha, payload, payload_bytes);
  }
  sha256_final(&sha, computed);
  if (memcmp(computed, header + 128u, 32u) != 0) {
    free(payload);
    return ET_MESH_ERR_CHECKSUM_MISMATCH;
  }

  /* Verify 2D mesh coordinates. */
  const uint32_t f_dp_size = get_u32_le(header + 16u);
  const uint32_t f_tp_size = get_u32_le(header + 20u);
  const uint32_t f_dp_rank = get_u32_le(header + 24u);
  const uint32_t f_tp_rank = get_u32_le(header + 28u);
  if (f_dp_size != engine->mesh->coords.dp_size ||
      f_tp_size != engine->mesh->coords.tp_size ||
      f_dp_rank != engine->mesh->coords.dp_rank ||
      f_tp_rank != engine->mesh->coords.tp_rank) {
    free(payload);
    return ET_MESH_ERR_MESH_MISMATCH;
  }

  /* Verify model & optimizer config exact match. */
  const uint32_t f_d_model = get_u32_le(header + 32u);
  const uint32_t f_d_ff = get_u32_le(header + 36u);
  const uint32_t f_num_classes = get_u32_le(header + 40u);
  const uint32_t f_global_batch = get_u32_le(header + 44u);
  const uint32_t f_rms_eps_bits = get_u32_le(header + 48u);
  const uint32_t f_reserved = get_u32_le(header + 52u);
  const uint64_t f_step = get_u64_le(header + 56u);
  const uint64_t f_local_params = get_u64_le(header + 64u);
  const uint64_t f_lr_bits = get_u64_le(header + 72u);
  const uint64_t f_b1_bits = get_u64_le(header + 80u);
  const uint64_t f_b2_bits = get_u64_le(header + 88u);
  const uint64_t f_eps_bits = get_u64_le(header + 96u);
  const uint64_t f_wd_bits = get_u64_le(header + 104u);
  const uint64_t f_mgn_bits = get_u64_le(header + 112u);

  if (f_d_model != engine->mcfg.d_model || f_d_ff != engine->mcfg.d_ff ||
      f_num_classes != engine->mcfg.num_classes ||
      f_global_batch != engine->mcfg.global_batch ||
      f_rms_eps_bits != f32_to_bits(engine->mcfg.rms_eps) ||
      f_reserved != 0u || f_local_params != engine->rank_total_params ||
      payload_bytes_u64 != engine->rank_total_params * 3u * 4u ||
      f_lr_bits != f64_to_bits(engine->ocfg.lr) ||
      f_b1_bits != f64_to_bits(engine->ocfg.beta1) ||
      f_b2_bits != f64_to_bits(engine->ocfg.beta2) ||
      f_eps_bits != f64_to_bits(engine->ocfg.eps) ||
      f_wd_bits != f64_to_bits(engine->ocfg.weight_decay) ||
      f_mgn_bits != f64_to_bits(engine->ocfg.max_grad_norm)) {
    free(payload);
    return ET_MESH_ERR_CONFIG_MISMATCH;
  }

  /* Decode into temporary arrays and validate finiteness + v >= 0 before commit. */
  const size_t rp = (size_t)engine->rank_total_params;
  float *tmp_p = (float *)malloc(rp * sizeof(float));
  float *tmp_m = (float *)malloc(rp * sizeof(float));
  float *tmp_v = (float *)malloc(rp * sizeof(float));
  if (tmp_p == NULL || tmp_m == NULL || tmp_v == NULL) {
    free(payload);
    free(tmp_p);
    free(tmp_m);
    free(tmp_v);
    return ET_MESH_ERR_ALLOC;
  }

  size_t off = 0u;
  int bad = 0;
  for (size_t i = 0u; i < rp; ++i) {
    const float pv = bits_to_f32(get_u32_le(payload + off));
    off += 4u;
    if (!isfinite(pv)) {
      bad = 1;
      break;
    }
    tmp_p[i] = canonical_f32(pv);
  }
  for (size_t i = 0u; !bad && i < rp; ++i) {
    const float mv = bits_to_f32(get_u32_le(payload + off));
    off += 4u;
    if (!isfinite(mv)) {
      bad = 1;
      break;
    }
    tmp_m[i] = canonical_f32(mv);
  }
  for (size_t i = 0u; !bad && i < rp; ++i) {
    const float vv = bits_to_f32(get_u32_le(payload + off));
    off += 4u;
    if (!isfinite(vv) || vv < 0.0f) {
      bad = 1;
      break;
    }
    tmp_v[i] = canonical_f32(vv);
  }
  free(payload);

  if (bad) {
    free(tmp_p);
    free(tmp_m);
    free(tmp_v);
    return ET_MESH_ERR_NONFINITE;
  }

  /* Atomic state replacement. */
  memcpy(engine->params, tmp_p, rp * sizeof(float));
  memcpy(engine->m, tmp_m, rp * sizeof(float));
  memcpy(engine->v, tmp_v, rp * sizeof(float));
  memset(engine->grads, 0, rp * sizeof(float));
  engine->step = f_step;
  engine->has_forward_cache = 0;

  free(tmp_p);
  free(tmp_m);
  free(tmp_v);
  return ET_MESH_OK;
}
