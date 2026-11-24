#define _GNU_SOURCE
#include "eshkol_transformer/fsdp_zero_abi.h"

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
               "FSDP/ZeRO engine requires IEEE-754 binary32");
_Static_assert(sizeof(double) == 8u && DBL_MANT_DIG == 53,
               "FSDP/ZeRO engine requires IEEE-754 binary64");

#define FSDP_HELLO_MAGIC UINT64_C(0x4554465348454c4f) /* "ETFSHELO" */
#define FSDP_WIRE_MAGIC UINT64_C(0x4554465344500101)  /* "ETFSDP\x01\x01" */

enum {
  HELLO_WORDS = 5,
  ACK_WORDS = 2,
  REQ_WORDS = 9,
  REP_WORDS = 3
};

enum fsdp_op_kind {
  KIND_ALLGATHER_F32 = 1,
  KIND_REDUCE_SCATTER_AVG_F32 = 2,
  KIND_ALLREDUCE_AVG_F32 = 3,
  KIND_ALLREDUCE_SUM_F64 = 4
};

enum fsdp_phase {
  PHASE_GRANT = 1,
  PHASE_RESULT = 2,
  PHASE_COMMIT = 3,
  PHASE_ABORT = 4
};

struct et_fsdp_comm {
  uint32_t rank;
  uint32_t world_size;
  uint64_t sequence;
  et_fsdp_status_t fatal;
  int fd[ET_FSDP_MAX_WORLD_SIZE];
  et_fsdp_comm_stats_v1 stats;
};

struct et_fsdp_engine {
  et_fsdp_comm_t *comm;
  et_fsdp_config_v1 cfg;
  uint64_t padded_params;
  uint64_t shard_len;
  uint64_t step;
  uint64_t ephemeral_peak_bytes;

  /* Persistent buffers allocated strictly per ZeRO stage:
   * - Stage 1: params_full[padded], grads_full[padded], m_shard[S], v_shard[S]
   * - Stage 2: params_full[padded], grad_shard[S],      m_shard[S], v_shard[S]
   * - Stage 3: params_shard[S],     grad_shard[S],      m_shard[S], v_shard[S]
   */
  float *params_full;
  float *params_shard;
  float *grads_full;
  float *grad_shard;
  float *m_shard;
  float *v_shard;
};

/* --- Bit / Endian / FP helpers --- */

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

static void put_u16_le(uint8_t *dst, uint16_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
}

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

static uint16_t get_u16_le(const uint8_t *src) {
  return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8u));
}

static uint32_t get_u32_le(const uint8_t *src) {
  return ((uint32_t)src[0]) | ((uint32_t)src[1] << 8u) |
         ((uint32_t)src[2] << 16u) | ((uint32_t)src[3] << 24u);
}

static uint64_t get_u64_le(const uint8_t *src) {
  uint64_t v = 0u;
  for (size_t i = 0u; i < 8u; ++i) {
    v |= ((uint64_t)src[i] << (8u * i));
  }
  return v;
}

static double ipow_f64(double base, uint64_t exp) {
  double result = 1.0;
  double factor = base;
  while (exp > 0u) {
    if ((exp & UINT64_C(1)) != 0u) {
      result *= factor;
    }
    exp >>= 1u;
    if (exp > 0u) {
      factor *= factor;
    }
  }
  return result;
}

static int valid_world_size(uint32_t w) {
  return w == 1u || w == 2u || w == 4u;
}

static int valid_stage(uint32_t stage) {
  return stage == ET_ZERO_STAGE_1 || stage == ET_ZERO_STAGE_2 ||
         stage == ET_ZERO_STAGE_3_FSDP;
}

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  if (a_bytes == 0u || b_bytes == 0u) {
    return 0;
  }
  const uintptr_t la = (uintptr_t)a;
  const uintptr_t lb = (uintptr_t)b;
  if (la > UINTPTR_MAX - a_bytes || lb > UINTPTR_MAX - b_bytes) {
    return 1;
  }
  return la < lb + b_bytes && lb < la + a_bytes;
}

/* --- Standalone FIPS 180-4 SHA-256 --- */

typedef struct fsdp_sha256_ctx {
  uint32_t state[8];
  uint64_t bitlen;
  uint8_t buf[64];
  size_t buflen;
} fsdp_sha256_ctx;

static uint32_t rotr32(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32u - n));
}

static void sha256_transform(fsdp_sha256_ctx *ctx, const uint8_t block[64]) {
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

static void sha256_init(fsdp_sha256_ctx *ctx) {
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

static void sha256_update(fsdp_sha256_ctx *ctx, const uint8_t *data,
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

static void sha256_final(fsdp_sha256_ctx *ctx, uint8_t digest[32]) {
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

ET_FSDP_API et_fsdp_status_t et_fsdp_sha256_bytes(const uint8_t *data,
                                                  size_t len,
                                                  uint8_t out_digest[32]) {
  if (out_digest == NULL || (len > 0u && data == NULL)) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  fsdp_sha256_ctx ctx;
  sha256_init(&ctx);
  if (len > 0u) {
    sha256_update(&ctx, data, len);
  }
  sha256_final(&ctx, out_digest);
  return ET_FSDP_OK;
}

/* --- Monotonic clock and nonblocking AF_UNIX socket transport --- */

static int64_t now_ms(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return -1;
  }
  return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void sleep_ms_bounded(int64_t ms) {
  if (ms <= 0) {
    return;
  }
  struct timespec req;
  req.tv_sec = (time_t)(ms / 1000);
  req.tv_nsec = (long)((ms % 1000) * 1000000);
  while (nanosleep(&req, &req) != 0 && errno == EINTR) {
  }
}

static int retryable_errno(int err) {
  return err == EINTR || err == EAGAIN || err == EWOULDBLOCK;
}

static et_fsdp_status_t check_unix_stream_fd(int fd, struct stat *st) {
  struct sockaddr_storage local;
  struct sockaddr_storage peer;
  socklen_t len = sizeof(local);
  int sock_type = 0;

  if (fd < 0 || fstat(fd, st) != 0) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  if (getsockname(fd, (struct sockaddr *)&local, &len) != 0 ||
      local.ss_family != AF_UNIX) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  len = sizeof(peer);
  if (getpeername(fd, (struct sockaddr *)&peer, &len) != 0 ||
      peer.ss_family != AF_UNIX) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  len = sizeof(sock_type);
  if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &sock_type, &len) != 0 ||
      sock_type != SOCK_STREAM) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  return ET_FSDP_OK;
}

static et_fsdp_status_t io_single_fd_deadline(int fd, void *buf, size_t len,
                                              int writing, int64_t deadline) {
  uint8_t *cursor = (uint8_t *)buf;
  while (len > 0u) {
    const int64_t now = now_ms();
    if (now < 0) {
      return ET_FSDP_ERR_IO;
    }
    if (now >= deadline) {
      return ET_FSDP_ERR_TIMEOUT;
    }
    const int64_t rem = deadline - now;
    const int poll_ms = rem > INT_MAX ? INT_MAX : (int)rem;
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = (short)((writing ? POLLOUT : POLLIN) | POLLRDHUP);
    pfd.revents = 0;

    const int rc = poll(&pfd, 1u, poll_ms);
    if (rc < 0 && errno == EINTR) {
      continue;
    }
    if (rc < 0) {
      return ET_FSDP_ERR_IO;
    }
    if (rc == 0) {
      continue;
    }
    if ((pfd.revents & POLLNVAL) != 0) {
      return ET_FSDP_ERR_IO;
    }
    if (!writing && (pfd.revents & POLLIN) != 0) {
      /* Drain buffered bytes before interpreting hangup as EOF. */
    } else if ((pfd.revents & (POLLERR | POLLHUP | POLLRDHUP)) != 0) {
      return ET_FSDP_ERR_PEER_DISCONNECTED;
    }
    const ssize_t n =
        writing ? send(fd, cursor, len, MSG_DONTWAIT | MSG_NOSIGNAL)
                : recv(fd, cursor, len, MSG_DONTWAIT);
    if (n < 0 && retryable_errno(errno)) {
      continue;
    }
    if (n == 0 ||
        (n < 0 && (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN))) {
      return ET_FSDP_ERR_PEER_DISCONNECTED;
    }
    if (n < 0) {
      return ET_FSDP_ERR_IO;
    }
    cursor += (size_t)n;
    len -= (size_t)n;
  }
  return ET_FSDP_OK;
}

static int build_rendezvous_addr(const char *sock_dir,
                                 struct sockaddr_un *addr) {
  if (sock_dir == NULL || sock_dir[0] == '\0') {
    return 0;
  }
  struct stat st;
  if (stat(sock_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
    return 0;
  }
  memset(addr, 0, sizeof(*addr));
  addr->sun_family = AF_UNIX;
  const int n = snprintf(addr->sun_path, sizeof(addr->sun_path),
                         "%s/fsdp_r0.sock", sock_dir);
  if (n <= 0 || (size_t)n >= sizeof(addr->sun_path)) {
    return 0;
  }
  return 1;
}

static void best_effort_send_ack(int fd, et_fsdp_status_t status) {
  if (fd < 0) {
    return;
  }
  const uint64_t ack[ACK_WORDS] = {FSDP_HELLO_MAGIC, (uint64_t)(int64_t)status};
  (void)send(fd, ack, sizeof(ack), MSG_DONTWAIT | MSG_NOSIGNAL);
}

struct pending_hello {
  int fd;
  uint8_t buf[HELLO_WORDS * sizeof(uint64_t)];
  size_t got;
};

static et_fsdp_status_t open_coordinator_unix(const struct sockaddr_un *addr,
                                              uint32_t world_size,
                                              int64_t deadline,
                                              int peer_fds[ET_FSDP_MAX_WORLD_SIZE]) {
  const int lfd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (lfd < 0) {
    return ET_FSDP_ERR_IO;
  }
  if (unlink(addr->sun_path) != 0 && errno != ENOENT) {
    close(lfd);
    return ET_FSDP_ERR_IO;
  }
  if (bind(lfd, (const struct sockaddr *)addr, sizeof(*addr)) != 0 ||
      listen(lfd, (int)world_size * 2) != 0) {
    close(lfd);
    (void)unlink(addr->sun_path);
    return ET_FSDP_ERR_IO;
  }

  struct pending_hello pending[ET_FSDP_MAX_WORLD_SIZE * 2u];
  for (size_t i = 0u; i < ET_FSDP_MAX_WORLD_SIZE * 2u; ++i) {
    pending[i].fd = -1;
    pending[i].got = 0u;
  }

  uint32_t connected = 0u;
  et_fsdp_status_t status = ET_FSDP_OK;

  while (status == ET_FSDP_OK && connected < world_size - 1u) {
    struct pollfd pfds[1u + ET_FSDP_MAX_WORLD_SIZE * 3u];
    int pkind[1u + ET_FSDP_MAX_WORLD_SIZE * 3u];
    size_t pslot[1u + ET_FSDP_MAX_WORLD_SIZE * 3u];
    nfds_t nfds = 0u;

    pfds[nfds].fd = lfd;
    pfds[nfds].events = POLLIN;
    pfds[nfds].revents = 0;
    pkind[nfds] = 0;
    pslot[nfds] = 0u;
    ++nfds;

    for (size_t i = 0u; i < ET_FSDP_MAX_WORLD_SIZE * 2u; ++i) {
      if (pending[i].fd >= 0) {
        pfds[nfds].fd = pending[i].fd;
        pfds[nfds].events = POLLIN | POLLRDHUP;
        pfds[nfds].revents = 0;
        pkind[nfds] = 1;
        pslot[nfds] = i;
        ++nfds;
      }
    }
    for (uint32_t r = 1u; r < world_size; ++r) {
      if (peer_fds[r] >= 0) {
        pfds[nfds].fd = peer_fds[r];
        pfds[nfds].events = POLLRDHUP;
        pfds[nfds].revents = 0;
        pkind[nfds] = 2;
        pslot[nfds] = r;
        ++nfds;
      }
    }

    const int64_t now = now_ms();
    if (now < 0) {
      status = ET_FSDP_ERR_IO;
      break;
    }
    if (now >= deadline) {
      status = ET_FSDP_ERR_TIMEOUT;
      break;
    }
    const int64_t rem = deadline - now;
    const int poll_ms = rem > INT_MAX ? INT_MAX : (int)rem;
    const int rc = poll(pfds, nfds, poll_ms);
    if (rc < 0 && errno == EINTR) {
      continue;
    }
    if (rc < 0) {
      status = ET_FSDP_ERR_IO;
      break;
    }
    if (rc == 0) {
      continue;
    }

    for (nfds_t i = 0u; i < nfds && status == ET_FSDP_OK; ++i) {
      if (pfds[i].revents == 0) {
        continue;
      }
      if (pkind[i] == 0) {
        for (;;) {
          const int cfd = accept4(lfd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
          if (cfd < 0) {
            if (retryable_errno(errno)) {
              break;
            }
            status = ET_FSDP_ERR_IO;
            break;
          }
          size_t free_slot = ET_FSDP_MAX_WORLD_SIZE * 2u;
          for (size_t s = 0u; s < ET_FSDP_MAX_WORLD_SIZE * 2u; ++s) {
            if (pending[s].fd < 0) {
              free_slot = s;
              break;
            }
          }
          if (free_slot == ET_FSDP_MAX_WORLD_SIZE * 2u) {
            best_effort_send_ack(cfd, ET_FSDP_ERR_MISMATCH);
            close(cfd);
            status = ET_FSDP_ERR_MISMATCH;
            break;
          }
          pending[free_slot].fd = cfd;
          pending[free_slot].got = 0u;
        }
      } else if (pkind[i] == 1) {
        struct pending_hello *ph = &pending[pslot[i]];
        if ((pfds[i].revents & POLLIN) != 0) {
          const ssize_t n = recv(ph->fd, ph->buf + ph->got,
                                 sizeof(ph->buf) - ph->got, MSG_DONTWAIT);
          if (n > 0) {
            ph->got += (size_t)n;
            if (ph->got == sizeof(ph->buf)) {
              uint64_t words[HELLO_WORDS];
              memcpy(words, ph->buf, sizeof(words));
              if (words[0] != FSDP_HELLO_MAGIC ||
                  words[1] != ET_FSDP_ABI_MAJOR ||
                  words[2] > ET_FSDP_ABI_MINOR ||
                  words[4] != (uint64_t)world_size || words[3] == 0u ||
                  words[3] >= (uint64_t)world_size ||
                  peer_fds[words[3]] >= 0) {
                best_effort_send_ack(ph->fd, ET_FSDP_ERR_MISMATCH);
                close(ph->fd);
                ph->fd = -1;
                ph->got = 0u;
                status = ET_FSDP_ERR_MISMATCH;
              } else {
                peer_fds[words[3]] = ph->fd;
                ph->fd = -1;
                ph->got = 0u;
                ++connected;
              }
            }
          } else if (n == 0 ||
                     (n < 0 && !retryable_errno(errno))) {
            close(ph->fd);
            ph->fd = -1;
            ph->got = 0u;
            status = ET_FSDP_ERR_PEER_DISCONNECTED;
          }
        } else if ((pfds[i].revents & (POLLERR | POLLHUP | POLLRDHUP)) != 0) {
          close(ph->fd);
          ph->fd = -1;
          ph->got = 0u;
          status = ET_FSDP_ERR_PEER_DISCONNECTED;
        }
      } else if (pkind[i] == 2) {
        if ((pfds[i].revents & (POLLERR | POLLHUP | POLLRDHUP | POLLNVAL)) != 0) {
          status = ET_FSDP_ERR_PEER_DISCONNECTED;
        }
      }
    }
  }

  for (size_t i = 0u; i < ET_FSDP_MAX_WORLD_SIZE * 2u; ++i) {
    if (pending[i].fd >= 0) {
      best_effort_send_ack(pending[i].fd,
                           status == ET_FSDP_OK ? ET_FSDP_ERR_MISMATCH : status);
      close(pending[i].fd);
      pending[i].fd = -1;
    }
  }

  close(lfd);
  (void)unlink(addr->sun_path);

  if (status == ET_FSDP_OK) {
    uint64_t ack[ACK_WORDS] = {FSDP_HELLO_MAGIC, (uint64_t)(int64_t)ET_FSDP_OK};
    for (uint32_t r = 1u; r < world_size; ++r) {
      status = io_single_fd_deadline(peer_fds[r], ack, sizeof(ack), 1, deadline);
      if (status != ET_FSDP_OK) {
        break;
      }
    }
  }

  if (status != ET_FSDP_OK) {
    for (uint32_t r = 1u; r < world_size; ++r) {
      if (peer_fds[r] >= 0) {
        best_effort_send_ack(peer_fds[r], status);
        close(peer_fds[r]);
        peer_fds[r] = -1;
      }
    }
  }
  return status;
}

static et_fsdp_status_t open_worker_unix(const struct sockaddr_un *addr,
                                         uint32_t abi_major, uint32_t abi_minor,
                                         uint32_t rank, uint32_t world_size,
                                         int64_t deadline, int *out_fd) {
  *out_fd = -1;
  int fd = -1;
  for (;;) {
    const int64_t now = now_ms();
    if (now < 0) {
      return ET_FSDP_ERR_IO;
    }
    if (now >= deadline) {
      return ET_FSDP_ERR_TIMEOUT;
    }
    fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) {
      return ET_FSDP_ERR_IO;
    }
    if (connect(fd, (const struct sockaddr *)addr, sizeof(*addr)) == 0) {
      break;
    }
    const int err = errno;
    if (err == EINPROGRESS) {
      struct pollfd pfd = {fd, POLLOUT | POLLERR, 0};
      const int64_t rem = deadline - now;
      const int rc = poll(&pfd, 1u, rem > 50 ? 50 : (int)rem);
      if (rc > 0) {
        int so_err = 0;
        socklen_t slen = sizeof(so_err);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &slen) == 0 &&
            so_err == 0) {
          break;
        }
      }
      close(fd);
      fd = -1;
      sleep_ms_bounded(2);
      continue;
    }
    close(fd);
    fd = -1;
    if (err == ENOENT || err == ECONNREFUSED || err == EAGAIN || err == EINTR) {
      const int64_t rem = deadline - now;
      sleep_ms_bounded(rem > 2 ? 2 : rem);
      continue;
    }
    return ET_FSDP_ERR_IO;
  }

  uint64_t hello[HELLO_WORDS] = {FSDP_HELLO_MAGIC, abi_major, abi_minor, rank,
                                 world_size};
  et_fsdp_status_t st =
      io_single_fd_deadline(fd, hello, sizeof(hello), 1, deadline);
  if (st != ET_FSDP_OK) {
    close(fd);
    return st;
  }
  uint64_t ack[ACK_WORDS] = {0u, 0u};
  st = io_single_fd_deadline(fd, ack, sizeof(ack), 0, deadline);
  if (st != ET_FSDP_OK) {
    close(fd);
    return st;
  }
  if (ack[0] != FSDP_HELLO_MAGIC) {
    close(fd);
    return ET_FSDP_ERR_MISMATCH;
  }
  const et_fsdp_status_t remote_st = (et_fsdp_status_t)(int64_t)ack[1];
  if (remote_st != ET_FSDP_OK) {
    close(fd);
    return remote_st;
  }
  *out_fd = fd;
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_comm_create_fds(
    uint32_t abi_major, uint32_t abi_minor, uint32_t rank, uint32_t world_size,
    const int *fds, size_t fd_count, et_fsdp_comm_t **out_comm) {
  if (out_comm == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  *out_comm = NULL;
  if (abi_major != ET_FSDP_ABI_MAJOR || abi_minor > ET_FSDP_ABI_MINOR ||
      !valid_world_size(world_size) || rank >= world_size) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  const size_t required = (world_size == 1u) ? 0u : (rank == 0u ? world_size - 1u : 1u);
  if (fd_count != required || (required > 0u && fds == NULL)) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  et_fsdp_comm_t *comm = (et_fsdp_comm_t *)calloc(1u, sizeof(*comm));
  if (comm == NULL) {
    return ET_FSDP_ERR_ALLOC;
  }
  comm->rank = rank;
  comm->world_size = world_size;
  comm->fatal = ET_FSDP_OK;
  for (uint32_t i = 0u; i < ET_FSDP_MAX_WORLD_SIZE; ++i) {
    comm->fd[i] = -1;
  }

  struct stat ids[ET_FSDP_MAX_WORLD_SIZE];
  et_fsdp_status_t st = ET_FSDP_OK;
  for (size_t i = 0u; i < required; ++i) {
    int dup_fd;
    do {
      dup_fd = fcntl(fds[i], F_DUPFD_CLOEXEC, 0);
    } while (dup_fd < 0 && errno == EINTR);
    if (dup_fd < 0) {
      st = (errno == EBADF) ? ET_FSDP_ERR_INVALID_ARG : ET_FSDP_ERR_IO;
      break;
    }
    const uint32_t slot = (rank == 0u) ? (uint32_t)(i + 1u) : 0u;
    comm->fd[slot] = dup_fd;
    st = check_unix_stream_fd(dup_fd, &ids[i]);
    for (size_t j = 0u; st == ET_FSDP_OK && j < i; ++j) {
      if (ids[i].st_dev == ids[j].st_dev && ids[i].st_ino == ids[j].st_ino) {
        st = ET_FSDP_ERR_INVALID_ARG;
      }
    }
    if (st != ET_FSDP_OK) {
      break;
    }
  }

  if (st != ET_FSDP_OK) {
    et_fsdp_comm_destroy(comm);
    return st;
  }
  *out_comm = comm;
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_comm_open_unix(
    uint32_t abi_major, uint32_t abi_minor, const char *sock_dir, uint32_t rank,
    uint32_t world_size, int timeout_ms, et_fsdp_comm_t **out_comm) {
  if (out_comm == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  *out_comm = NULL;
  if (abi_major != ET_FSDP_ABI_MAJOR || abi_minor > ET_FSDP_ABI_MINOR ||
      !valid_world_size(world_size) || rank >= world_size || timeout_ms <= 0) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  if (world_size == 1u) {
    return et_fsdp_comm_create_fds(abi_major, abi_minor, 0u, 1u, NULL, 0u,
                                   out_comm);
  }

  struct sockaddr_un addr;
  if (!build_rendezvous_addr(sock_dir, &addr)) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  const int64_t start = now_ms();
  if (start < 0 || start > INT64_MAX - (int64_t)timeout_ms) {
    return ET_FSDP_ERR_IO;
  }
  const int64_t deadline = start + (int64_t)timeout_ms;

  int peer_fds[ET_FSDP_MAX_WORLD_SIZE];
  for (uint32_t i = 0u; i < ET_FSDP_MAX_WORLD_SIZE; ++i) {
    peer_fds[i] = -1;
  }

  et_fsdp_status_t st = ET_FSDP_OK;
  if (rank == 0u) {
    st = open_coordinator_unix(&addr, world_size, deadline, peer_fds);
    if (st == ET_FSDP_OK) {
      st = et_fsdp_comm_create_fds(abi_major, abi_minor, 0u, world_size,
                                   &peer_fds[1], world_size - 1u, out_comm);
    }
    for (uint32_t r = 1u; r < world_size; ++r) {
      if (peer_fds[r] >= 0) {
        close(peer_fds[r]);
      }
    }
  } else {
    int fd = -1;
    st = open_worker_unix(&addr, abi_major, abi_minor, rank, world_size,
                          deadline, &fd);
    if (st == ET_FSDP_OK) {
      st = et_fsdp_comm_create_fds(abi_major, abi_minor, rank, world_size, &fd,
                                   1u, out_comm);
    }
    if (fd >= 0) {
      close(fd);
    }
  }
  return st;
}

ET_FSDP_API void et_fsdp_comm_destroy(et_fsdp_comm_t *comm) {
  if (comm == NULL) {
    return;
  }
  for (uint32_t i = 0u; i < ET_FSDP_MAX_WORLD_SIZE; ++i) {
    if (comm->fd[i] >= 0) {
      close(comm->fd[i]);
      comm->fd[i] = -1;
    }
  }
  free(comm);
}

ET_FSDP_API uint32_t et_fsdp_comm_rank(const et_fsdp_comm_t *comm) {
  return comm == NULL ? UINT32_MAX : comm->rank;
}

ET_FSDP_API uint32_t et_fsdp_comm_world_size(const et_fsdp_comm_t *comm) {
  return comm == NULL ? 0u : comm->world_size;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_comm_get_stats(
    const et_fsdp_comm_t *comm, et_fsdp_comm_stats_v1 *out_stats) {
  if (comm == NULL || out_stats == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  *out_stats = comm->stats;
  return ET_FSDP_OK;
}

/* --- Collective star-topology protocol engine --- */

struct xfer_ctx {
  et_fsdp_comm_t *comm;
  int64_t deadline;
  uint32_t released_mask;
  uint32_t incomplete_mask;
};

static et_fsdp_status_t wait_peer_ready(const struct xfer_ctx *x, uint32_t peer,
                                        short event, int watch_all) {
  struct pollfd pfds[ET_FSDP_MAX_WORLD_SIZE];
  nfds_t n = 0u;
  const short hangup = POLLHUP | POLLRDHUP;

  pfds[n].fd = x->comm->fd[peer];
  pfds[n].events = (short)(event | POLLRDHUP);
  pfds[n].revents = 0;
  ++n;

  if (watch_all && x->comm->rank == 0u) {
    for (uint32_t r = 1u; r < x->comm->world_size; ++r) {
      if (r != peer && (x->released_mask & (1u << r)) == 0u) {
        pfds[n].fd = x->comm->fd[r];
        pfds[n].events = POLLRDHUP;
        pfds[n].revents = 0;
        ++n;
      }
    }
  }

  for (;;) {
    const int64_t now = now_ms();
    if (now < 0) {
      return ET_FSDP_ERR_IO;
    }
    if (now >= x->deadline) {
      return ET_FSDP_ERR_TIMEOUT;
    }
    const int64_t rem = x->deadline - now;
    const int poll_ms = rem > INT_MAX ? INT_MAX : (int)rem;
    const int rc = poll(pfds, n, poll_ms);
    if (rc < 0 && errno == EINTR) {
      continue;
    }
    if (rc < 0) {
      return ET_FSDP_ERR_IO;
    }
    if (rc == 0) {
      continue;
    }
    for (nfds_t i = 0u; i < n; ++i) {
      if ((pfds[i].revents & POLLNVAL) != 0) {
        return ET_FSDP_ERR_IO;
      }
      if (i == 0u && event == POLLIN && (pfds[i].revents & POLLIN) != 0) {
        continue;
      }
      if ((pfds[i].revents & (hangup | POLLERR)) != 0) {
        return ET_FSDP_ERR_PEER_DISCONNECTED;
      }
    }
    if ((pfds[0].revents & event) != 0) {
      return ET_FSDP_OK;
    }
  }
}

static et_fsdp_status_t xfer_bytes(struct xfer_ctx *x, uint32_t peer,
                                   void *buf, size_t len, int writing,
                                   int watch_all) {
  uint8_t *cursor = (uint8_t *)buf;
  const int fd = x->comm->fd[peer];
  while (len > 0u) {
    const et_fsdp_status_t wst =
        wait_peer_ready(x, peer, writing ? POLLOUT : POLLIN, watch_all);
    if (wst != ET_FSDP_OK) {
      if (writing && cursor != (uint8_t *)buf) {
        x->incomplete_mask |= (1u << peer);
      }
      return wst;
    }
    const ssize_t n =
        writing ? send(fd, cursor, len, MSG_DONTWAIT | MSG_NOSIGNAL)
                : recv(fd, cursor, len, MSG_DONTWAIT);
    if (n < 0 && retryable_errno(errno)) {
      continue;
    }
    if (n <= 0 && writing && cursor != (uint8_t *)buf) {
      x->incomplete_mask |= (1u << peer);
    }
    if (n == 0 ||
        (n < 0 && (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN))) {
      return ET_FSDP_ERR_PEER_DISCONNECTED;
    }
    if (n < 0) {
      return ET_FSDP_ERR_IO;
    }
    cursor += (size_t)n;
    len -= (size_t)n;
  }
  return ET_FSDP_OK;
}

static et_fsdp_status_t send_reply_frame(struct xfer_ctx *x, uint32_t peer,
                                         enum fsdp_phase phase,
                                         et_fsdp_status_t status,
                                         int watch_all) {
  uint64_t words[REP_WORDS] = {x->comm->sequence, (uint64_t)phase,
                               (uint64_t)(int64_t)status};
  return xfer_bytes(x, peer, words, sizeof(words), 1, watch_all);
}

static et_fsdp_status_t recv_reply_frame(struct xfer_ctx *x,
                                         enum fsdp_phase expected_phase,
                                         enum fsdp_phase *out_phase,
                                         et_fsdp_status_t *out_status) {
  uint64_t words[REP_WORDS] = {0u, 0u, 0u};
  const et_fsdp_status_t st = xfer_bytes(x, 0u, words, sizeof(words), 0, 0);
  if (st != ET_FSDP_OK) {
    return st;
  }
  if (words[0] != x->comm->sequence) {
    return ET_FSDP_ERR_MISMATCH;
  }
  if (words[1] != (uint64_t)expected_phase && words[1] != (uint64_t)PHASE_ABORT) {
    return ET_FSDP_ERR_MISMATCH;
  }
  const et_fsdp_status_t rstat = (et_fsdp_status_t)(int64_t)words[2];
  if ((words[1] == (uint64_t)PHASE_ABORT) == (rstat == ET_FSDP_OK)) {
    return ET_FSDP_ERR_MISMATCH;
  }
  *out_phase = (enum fsdp_phase)words[1];
  *out_status = rstat;
  return ET_FSDP_OK;
}

static void fail_comm_fatal(struct xfer_ctx *x, et_fsdp_status_t fatal_err) {
  et_fsdp_comm_t *comm = x->comm;
  if (comm->fatal == ET_FSDP_OK) {
    comm->fatal = fatal_err;
  }
  if (comm->rank == 0u) {
    uint64_t words[REP_WORDS] = {comm->sequence, (uint64_t)PHASE_ABORT,
                                 (uint64_t)(int64_t)fatal_err};
    for (uint32_t r = 1u; r < comm->world_size; ++r) {
      if (comm->fd[r] >= 0) {
        if ((x->incomplete_mask & (1u << r)) == 0u &&
            (x->released_mask & (1u << r)) == 0u) {
          (void)send(comm->fd[r], words, sizeof(words),
                     MSG_DONTWAIT | MSG_NOSIGNAL);
        }
        (void)shutdown(comm->fd[r], SHUT_RDWR);
      }
    }
  } else if (comm->fd[0] >= 0) {
    (void)shutdown(comm->fd[0], SHUT_RDWR);
  }
}

static int is_recoverable_consensus(et_fsdp_status_t st) {
  return st == ET_FSDP_OK || st == ET_FSDP_ERR_INVALID_ARG ||
         st == ET_FSDP_ERR_NONFINITE || st == ET_FSDP_ERR_ALLOC;
}

static et_fsdp_status_t merge_local_status(et_fsdp_status_t a,
                                           et_fsdp_status_t b) {
  if (a == ET_FSDP_ERR_INVALID_ARG || b == ET_FSDP_ERR_INVALID_ARG) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  if (a == ET_FSDP_ERR_NONFINITE || b == ET_FSDP_ERR_NONFINITE) {
    return ET_FSDP_ERR_NONFINITE;
  }
  if (a == ET_FSDP_ERR_ALLOC || b == ET_FSDP_ERR_ALLOC) {
    return ET_FSDP_ERR_ALLOC;
  }
  return a != ET_FSDP_OK ? a : b;
}

static int check_all_finite_f32(const float *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!isfinite(buf[i])) {
      return 0;
    }
  }
  return 1;
}

static int check_all_finite_f64(const double *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!isfinite(buf[i])) {
      return 0;
    }
  }
  return 1;
}

static et_fsdp_status_t broadcast_recoverable_abort(struct xfer_ctx *x,
                                                    et_fsdp_status_t err) {
  for (uint32_t r = 1u; r < x->comm->world_size; ++r) {
    const et_fsdp_status_t st = send_reply_frame(x, r, PHASE_ABORT, err, 0);
    if (st != ET_FSDP_OK) {
      fail_comm_fatal(x, st);
      return st;
    }
  }
  x->comm->sequence += 1u;
  return err;
}

static et_fsdp_status_t run_collective(et_fsdp_comm_t *comm,
                                       enum fsdp_op_kind kind, uint64_t tag,
                                       size_t count, const void *sendbuf,
                                       void *recvbuf, int timeout_ms) {
  if (comm == NULL || timeout_ms <= 0) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  if (comm->fatal != ET_FSDP_OK) {
    return comm->fatal;
  }

  const uint32_t W = comm->world_size;
  const uint32_t rank = comm->rank;

  /* Local argument & finiteness validation (routed through preflight when W > 1). */
  et_fsdp_status_t local_st = ET_FSDP_OK;
  size_t send_bytes = 0u;
  size_t recv_bytes = 0u;

  if (count == 0u || sendbuf == NULL || recvbuf == NULL) {
    local_st = ET_FSDP_ERR_INVALID_ARG;
  } else {
    switch (kind) {
    case KIND_ALLGATHER_F32:
      if (count > ET_FSDP_MAX_BUFFER_BYTES / (sizeof(float) * (size_t)W)) {
        local_st = ET_FSDP_ERR_INVALID_ARG;
      } else {
        send_bytes = count * sizeof(float);
        recv_bytes = count * (size_t)W * sizeof(float);
        if (spans_overlap(sendbuf, send_bytes, recvbuf, recv_bytes)) {
          local_st = ET_FSDP_ERR_INVALID_ARG;
        }
      }
      break;
    case KIND_REDUCE_SCATTER_AVG_F32:
      if (count > ET_FSDP_MAX_BUFFER_BYTES / (sizeof(float) * (size_t)W)) {
        local_st = ET_FSDP_ERR_INVALID_ARG;
      } else {
        send_bytes = count * (size_t)W * sizeof(float);
        recv_bytes = count * sizeof(float);
        if (spans_overlap(sendbuf, send_bytes, recvbuf, recv_bytes)) {
          local_st = ET_FSDP_ERR_INVALID_ARG;
        } else if (!check_all_finite_f32((const float *)sendbuf,
                                         count * (size_t)W)) {
          local_st = ET_FSDP_ERR_NONFINITE;
        }
      }
      break;
    case KIND_ALLREDUCE_AVG_F32:
      if (count > ET_FSDP_MAX_BUFFER_BYTES / sizeof(float)) {
        local_st = ET_FSDP_ERR_INVALID_ARG;
      } else {
        send_bytes = count * sizeof(float);
        recv_bytes = count * sizeof(float);
        if (sendbuf != recvbuf &&
            spans_overlap(sendbuf, send_bytes, recvbuf, recv_bytes)) {
          local_st = ET_FSDP_ERR_INVALID_ARG;
        } else if (!check_all_finite_f32((const float *)sendbuf, count)) {
          local_st = ET_FSDP_ERR_NONFINITE;
        }
      }
      break;
    case KIND_ALLREDUCE_SUM_F64:
      if (count > ET_FSDP_MAX_BUFFER_BYTES / sizeof(double)) {
        local_st = ET_FSDP_ERR_INVALID_ARG;
      } else {
        send_bytes = count * sizeof(double);
        recv_bytes = count * sizeof(double);
        if (sendbuf != recvbuf &&
            spans_overlap(sendbuf, send_bytes, recvbuf, recv_bytes)) {
          local_st = ET_FSDP_ERR_INVALID_ARG;
        } else if (!check_all_finite_f64((const double *)sendbuf, count)) {
          local_st = ET_FSDP_ERR_NONFINITE;
        }
      }
      break;
    default:
      local_st = ET_FSDP_ERR_INVALID_ARG;
      break;
    }
  }

  /* Fast-path for W == 1: exact same semantics, zero sockets. */
  if (W == 1u) {
    if (local_st != ET_FSDP_OK) {
      return local_st;
    }
    if (sendbuf != recvbuf) {
      memcpy(recvbuf, sendbuf, recv_bytes);
    }
    comm->sequence += 1u;
    if (kind == KIND_ALLGATHER_F32) {
      comm->stats.allgather_calls += 1u;
    } else if (kind == KIND_REDUCE_SCATTER_AVG_F32) {
      comm->stats.reduce_scatter_calls += 1u;
    } else if (kind == KIND_ALLREDUCE_AVG_F32) {
      comm->stats.allreduce_f32_calls += 1u;
    } else if (kind == KIND_ALLREDUCE_SUM_F64) {
      comm->stats.allreduce_f64_calls += 1u;
    }
    return ET_FSDP_OK;
  }

  const int64_t start = now_ms();
  if (start < 0 || start > INT64_MAX - (int64_t)timeout_ms) {
    return ET_FSDP_ERR_IO;
  }
  struct xfer_ctx x;
  x.comm = comm;
  x.deadline = start + (int64_t)timeout_ms;
  x.released_mask = 0u;
  x.incomplete_mask = 0u;

  /* Worker path (rank > 0). */
  if (rank != 0u) {
    uint64_t req[REQ_WORDS] = {FSDP_WIRE_MAGIC,
                               comm->sequence,
                               rank,
                               W,
                               (uint64_t)kind,
                               tag,
                               (uint64_t)count,
                               (uint64_t)(int64_t)local_st,
                               0u};
    et_fsdp_status_t st = xfer_bytes(&x, 0u, req, sizeof(req), 1, 0);
    if (st != ET_FSDP_OK) {
      fail_comm_fatal(&x, st);
      return st;
    }

    enum fsdp_phase ph = PHASE_ABORT;
    et_fsdp_status_t rstat = ET_FSDP_OK;
    st = recv_reply_frame(&x, PHASE_GRANT, &ph, &rstat);
    if (st != ET_FSDP_OK) {
      fail_comm_fatal(&x, st);
      return st;
    }
    if (ph == PHASE_ABORT) {
      if (rstat == ET_FSDP_ERR_MISMATCH || !is_recoverable_consensus(rstat)) {
        fail_comm_fatal(&x, rstat);
      } else {
        comm->sequence += 1u;
      }
      return rstat;
    }

    /* Granted: send worker payload. */
    st = xfer_bytes(&x, 0u, (void *)sendbuf, send_bytes, 1, 0);
    if (st != ET_FSDP_OK) {
      fail_comm_fatal(&x, st);
      return st;
    }
    comm->stats.payload_bytes_sent += (uint64_t)send_bytes;

    /* Receive RESULT (or ABORT if reduction overflowed). */
    st = recv_reply_frame(&x, PHASE_RESULT, &ph, &rstat);
    if (st != ET_FSDP_OK) {
      fail_comm_fatal(&x, st);
      return st;
    }
    if (ph == PHASE_ABORT) {
      if (!is_recoverable_consensus(rstat)) {
        fail_comm_fatal(&x, rstat);
      } else {
        comm->sequence += 1u;
      }
      return rstat;
    }

    uint8_t *stage = (uint8_t *)malloc(recv_bytes);
    if (stage == NULL) {
      fail_comm_fatal(&x, ET_FSDP_ERR_ALLOC);
      return ET_FSDP_ERR_ALLOC;
    }
    st = xfer_bytes(&x, 0u, stage, recv_bytes, 0, 0);
    if (st != ET_FSDP_OK) {
      free(stage);
      fail_comm_fatal(&x, st);
      return st;
    }
    comm->stats.payload_bytes_recv += (uint64_t)recv_bytes;

    /* Acknowledge staged result. */
    uint64_t ack = comm->sequence;
    st = xfer_bytes(&x, 0u, &ack, sizeof(ack), 1, 0);
    if (st != ET_FSDP_OK) {
      free(stage);
      fail_comm_fatal(&x, st);
      return st;
    }

    /* Wait for COMMIT before publishing to caller buffer. */
    st = recv_reply_frame(&x, PHASE_COMMIT, &ph, &rstat);
    if (st != ET_FSDP_OK) {
      free(stage);
      fail_comm_fatal(&x, st);
      return st;
    }
    if (ph == PHASE_ABORT) {
      free(stage);
      if (!is_recoverable_consensus(rstat)) {
        fail_comm_fatal(&x, rstat);
      } else {
        comm->sequence += 1u;
      }
      return rstat;
    }

    memcpy(recvbuf, stage, recv_bytes);
    free(stage);
    comm->sequence += 1u;
    if (kind == KIND_ALLGATHER_F32) {
      comm->stats.allgather_calls += 1u;
    } else if (kind == KIND_REDUCE_SCATTER_AVG_F32) {
      comm->stats.reduce_scatter_calls += 1u;
    } else if (kind == KIND_ALLREDUCE_AVG_F32) {
      comm->stats.allreduce_f32_calls += 1u;
    } else if (kind == KIND_ALLREDUCE_SUM_F64) {
      comm->stats.allreduce_f64_calls += 1u;
    }
    return ET_FSDP_OK;
  }

  /* Coordinator path (rank == 0). */
  et_fsdp_status_t consensus = local_st;
  int mismatch = 0;
  for (uint32_t r = 1u; r < W; ++r) {
    uint64_t incoming[REQ_WORDS];
    const et_fsdp_status_t st =
        xfer_bytes(&x, r, incoming, sizeof(incoming), 0, 1);
    if (st != ET_FSDP_OK) {
      fail_comm_fatal(&x, st);
      return st;
    }
    const et_fsdp_status_t peer_local = (et_fsdp_status_t)(int64_t)incoming[7];
    if (incoming[0] != FSDP_WIRE_MAGIC || incoming[1] != comm->sequence ||
        incoming[2] != (uint64_t)r || incoming[3] != (uint64_t)W ||
        incoming[4] != (uint64_t)kind || incoming[5] != tag ||
        !is_recoverable_consensus(peer_local) || incoming[8] != 0u) {
      mismatch = 1;
    } else if (local_st != ET_FSDP_ERR_INVALID_ARG &&
               peer_local != ET_FSDP_ERR_INVALID_ARG &&
               incoming[6] != (uint64_t)count) {
      mismatch = 1;
    }
    consensus = merge_local_status(consensus, peer_local);
  }

  if (mismatch) {
    fail_comm_fatal(&x, ET_FSDP_ERR_MISMATCH);
    return ET_FSDP_ERR_MISMATCH;
  }
  if (consensus != ET_FSDP_OK) {
    return broadcast_recoverable_abort(&x, consensus);
  }

  /* Allocate coordinator staging buffers before granting workers. */
  uint8_t *all_inputs = (uint8_t *)malloc(send_bytes * (size_t)W);
  uint8_t *result_full = NULL;
  size_t result_full_bytes = 0u;
  if (kind == KIND_ALLGATHER_F32) {
    result_full_bytes = count * (size_t)W * sizeof(float);
  } else if (kind == KIND_REDUCE_SCATTER_AVG_F32) {
    result_full_bytes = count * (size_t)W * sizeof(float);
  } else if (kind == KIND_ALLREDUCE_AVG_F32) {
    result_full_bytes = count * sizeof(float);
  } else {
    result_full_bytes = count * sizeof(double);
  }
  result_full = (uint8_t *)malloc(result_full_bytes);

  if (all_inputs == NULL || result_full == NULL) {
    free(all_inputs);
    free(result_full);
    return broadcast_recoverable_abort(&x, ET_FSDP_ERR_ALLOC);
  }

  /* Copy rank 0's local input into slot 0. */
  memcpy(all_inputs, sendbuf, send_bytes);

  /* Send GRANT to all workers and receive their payloads in rank order. */
  for (uint32_t r = 1u; r < W; ++r) {
    et_fsdp_status_t st = send_reply_frame(&x, r, PHASE_GRANT, ET_FSDP_OK, 1);
    if (st != ET_FSDP_OK) {
      free(all_inputs);
      free(result_full);
      fail_comm_fatal(&x, st);
      return st;
    }
    st = xfer_bytes(&x, r, all_inputs + (size_t)r * send_bytes, send_bytes, 0,
                    1);
    if (st != ET_FSDP_OK) {
      free(all_inputs);
      free(result_full);
      fail_comm_fatal(&x, st);
      return st;
    }
    comm->stats.payload_bytes_recv += (uint64_t)send_bytes;
  }

  /* Canonical ascending rank-order computation (binary64 accumulation). */
  et_fsdp_status_t compute_st = ET_FSDP_OK;
  if (kind == KIND_ALLGATHER_F32) {
    memcpy(result_full, all_inputs, result_full_bytes);
  } else if (kind == KIND_REDUCE_SCATTER_AVG_F32) {
    const size_t total_elems = count * (size_t)W;
    float *out_f = (float *)result_full;
    for (size_t i = 0u; i < total_elems; ++i) {
      double acc = 0.0;
      for (uint32_t r = 0u; r < W; ++r) {
        const float *in_r = (const float *)(all_inputs + (size_t)r * send_bytes);
        acc += (double)in_r[i];
      }
      const double mean = acc / (double)W;
      const float mf = canonical_f32((float)mean);
      if (!isfinite(acc) || !isfinite(mean) || !isfinite(mf)) {
        compute_st = ET_FSDP_ERR_NONFINITE;
        break;
      }
      out_f[i] = mf;
    }
  } else if (kind == KIND_ALLREDUCE_AVG_F32) {
    float *out_f = (float *)result_full;
    for (size_t i = 0u; i < count; ++i) {
      double acc = 0.0;
      for (uint32_t r = 0u; r < W; ++r) {
        const float *in_r = (const float *)(all_inputs + (size_t)r * send_bytes);
        acc += (double)in_r[i];
      }
      const double mean = acc / (double)W;
      const float mf = canonical_f32((float)mean);
      if (!isfinite(acc) || !isfinite(mean) || !isfinite(mf)) {
        compute_st = ET_FSDP_ERR_NONFINITE;
        break;
      }
      out_f[i] = mf;
    }
  } else if (kind == KIND_ALLREDUCE_SUM_F64) {
    double *out_d = (double *)result_full;
    for (size_t i = 0u; i < count; ++i) {
      double acc = 0.0;
      for (uint32_t r = 0u; r < W; ++r) {
        const double *in_r =
            (const double *)(all_inputs + (size_t)r * send_bytes);
        acc += in_r[i];
      }
      if (!isfinite(acc)) {
        compute_st = ET_FSDP_ERR_NONFINITE;
        break;
      }
      out_d[i] = (acc == 0.0) ? 0.0 : acc;
    }
  }

  free(all_inputs);
  all_inputs = NULL;

  if (compute_st != ET_FSDP_OK) {
    free(result_full);
    return broadcast_recoverable_abort(&x, compute_st);
  }

  /* Send RESULT + rank r's slice to each worker and collect ACKs. */
  for (uint32_t r = 1u; r < W; ++r) {
    et_fsdp_status_t st = send_reply_frame(&x, r, PHASE_RESULT, ET_FSDP_OK, 1);
    if (st != ET_FSDP_OK) {
      free(result_full);
      fail_comm_fatal(&x, st);
      return st;
    }
    uint8_t *slice_ptr =
        (kind == KIND_REDUCE_SCATTER_AVG_F32)
            ? (result_full + (size_t)r * recv_bytes)
            : result_full;
    st = xfer_bytes(&x, r, slice_ptr, recv_bytes, 1, 1);
    if (st != ET_FSDP_OK) {
      free(result_full);
      fail_comm_fatal(&x, st);
      return st;
    }
    comm->stats.payload_bytes_sent += (uint64_t)recv_bytes;
  }

  for (uint32_t r = 1u; r < W; ++r) {
    uint64_t ack = 0u;
    const et_fsdp_status_t st = xfer_bytes(&x, r, &ack, sizeof(ack), 0, 1);
    if (st != ET_FSDP_OK) {
      free(result_full);
      fail_comm_fatal(&x, st);
      return st;
    }
    if (ack != comm->sequence) {
      free(result_full);
      fail_comm_fatal(&x, ET_FSDP_ERR_MISMATCH);
      return ET_FSDP_ERR_MISMATCH;
    }
  }

  /* Commit to all workers, then publish locally on rank 0. */
  for (uint32_t r = 1u; r < W; ++r) {
    const et_fsdp_status_t st =
        send_reply_frame(&x, r, PHASE_COMMIT, ET_FSDP_OK, 1);
    if (st != ET_FSDP_OK) {
      free(result_full);
      fail_comm_fatal(&x, st);
      return st;
    }
    x.released_mask |= (1u << r);
  }

  memcpy(recvbuf, result_full, recv_bytes);
  free(result_full);
  comm->sequence += 1u;

  if (kind == KIND_ALLGATHER_F32) {
    comm->stats.allgather_calls += 1u;
  } else if (kind == KIND_REDUCE_SCATTER_AVG_F32) {
    comm->stats.reduce_scatter_calls += 1u;
  } else if (kind == KIND_ALLREDUCE_AVG_F32) {
    comm->stats.allreduce_f32_calls += 1u;
  } else if (kind == KIND_ALLREDUCE_SUM_F64) {
    comm->stats.allreduce_f64_calls += 1u;
  }
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_allgather_f32(
    et_fsdp_comm_t *comm, const float *send_shard, size_t shard_len,
    float *recv_full, uint64_t tag, int timeout_ms) {
  return run_collective(comm, KIND_ALLGATHER_F32, tag, shard_len, send_shard,
                        recv_full, timeout_ms);
}

ET_FSDP_API et_fsdp_status_t et_fsdp_reduce_scatter_avg_f32(
    et_fsdp_comm_t *comm, const float *send_full, float *recv_shard,
    size_t shard_len, uint64_t tag, int timeout_ms) {
  return run_collective(comm, KIND_REDUCE_SCATTER_AVG_F32, tag, shard_len,
                        send_full, recv_shard, timeout_ms);
}

ET_FSDP_API et_fsdp_status_t et_fsdp_allreduce_avg_f32(
    et_fsdp_comm_t *comm, const float *send_full, float *recv_full,
    size_t count, uint64_t tag, int timeout_ms) {
  return run_collective(comm, KIND_ALLREDUCE_AVG_F32, tag, count, send_full,
                        recv_full, timeout_ms);
}

ET_FSDP_API et_fsdp_status_t et_fsdp_allreduce_sum_f64(
    et_fsdp_comm_t *comm, const double *send_vals, double *recv_vals,
    size_t count, uint64_t tag, int timeout_ms) {
  return run_collective(comm, KIND_ALLREDUCE_SUM_F64, tag, count, send_vals,
                        recv_vals, timeout_ms);
}

/* --- ZeRO-1 / ZeRO-2 / ZeRO-3 (FSDP) Sharded Engine --- */

ET_FSDP_API et_fsdp_status_t et_fsdp_compute_padded_len(
    uint64_t num_params, uint32_t world_size, uint64_t *out_padded_len,
    uint64_t *out_shard_len) {
  if (num_params == 0u || num_params > ET_FSDP_MAX_PARAMS ||
      !valid_world_size(world_size) || out_padded_len == NULL ||
      out_shard_len == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  const uint64_t w = (uint64_t)world_size;
  const uint64_t shard = (num_params + w - 1u) / w;
  *out_shard_len = shard;
  *out_padded_len = shard * w;
  return ET_FSDP_OK;
}

static et_fsdp_status_t validate_config(const et_fsdp_config_v1 *cfg) {
  if (cfg == NULL || cfg->reserved != 0u) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  if (!valid_stage(cfg->stage)) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  if (cfg->num_params == 0u || cfg->num_params > ET_FSDP_MAX_PARAMS) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  if (!isfinite(cfg->lr) || !isfinite(cfg->beta1) || !isfinite(cfg->beta2) ||
      !isfinite(cfg->eps) || !isfinite(cfg->weight_decay) ||
      !isfinite(cfg->max_grad_norm)) {
    return ET_FSDP_ERR_NONFINITE;
  }
  if (cfg->lr < 0.0 || cfg->beta1 < 0.0 || cfg->beta1 >= 1.0 ||
      cfg->beta2 < 0.0 || cfg->beta2 >= 1.0 || cfg->eps <= 0.0 ||
      cfg->weight_decay < 0.0 || cfg->max_grad_norm < 0.0) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_create(
    et_fsdp_comm_t *comm, const et_fsdp_config_v1 *cfg,
    et_fsdp_engine_t **out_engine) {
  if (out_engine == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  *out_engine = NULL;
  if (comm == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  et_fsdp_status_t st = validate_config(cfg);
  if (st != ET_FSDP_OK) {
    return st;
  }

  uint64_t padded = 0u;
  uint64_t shard = 0u;
  st = et_fsdp_compute_padded_len(cfg->num_params, comm->world_size, &padded,
                                  &shard);
  if (st != ET_FSDP_OK) {
    return st;
  }

  et_fsdp_engine_t *eng = (et_fsdp_engine_t *)calloc(1u, sizeof(*eng));
  if (eng == NULL) {
    return ET_FSDP_ERR_ALLOC;
  }
  eng->comm = comm;
  eng->cfg = *cfg;
  eng->padded_params = padded;
  eng->shard_len = shard;
  eng->step = 0u;
  eng->ephemeral_peak_bytes = 0u;

  /* Allocate ONLY the persistent buffers mandated by the selected ZeRO stage. */
  eng->m_shard = (float *)calloc((size_t)shard, sizeof(float));
  eng->v_shard = (float *)calloc((size_t)shard, sizeof(float));
  if (eng->m_shard == NULL || eng->v_shard == NULL) {
    et_fsdp_engine_destroy(eng);
    return ET_FSDP_ERR_ALLOC;
  }

  if (cfg->stage == ET_ZERO_STAGE_1) {
    eng->params_full = (float *)calloc((size_t)padded, sizeof(float));
    eng->grads_full = (float *)calloc((size_t)padded, sizeof(float));
    if (eng->params_full == NULL || eng->grads_full == NULL) {
      et_fsdp_engine_destroy(eng);
      return ET_FSDP_ERR_ALLOC;
    }
  } else if (cfg->stage == ET_ZERO_STAGE_2) {
    eng->params_full = (float *)calloc((size_t)padded, sizeof(float));
    eng->grad_shard = (float *)calloc((size_t)shard, sizeof(float));
    if (eng->params_full == NULL || eng->grad_shard == NULL) {
      et_fsdp_engine_destroy(eng);
      return ET_FSDP_ERR_ALLOC;
    }
  } else {
    /* ET_ZERO_STAGE_3_FSDP: NO persistent params_full, NO persistent grads_full! */
    eng->params_shard = (float *)calloc((size_t)shard, sizeof(float));
    eng->grad_shard = (float *)calloc((size_t)shard, sizeof(float));
    if (eng->params_shard == NULL || eng->grad_shard == NULL) {
      et_fsdp_engine_destroy(eng);
      return ET_FSDP_ERR_ALLOC;
    }
  }

  *out_engine = eng;
  return ET_FSDP_OK;
}

ET_FSDP_API void et_fsdp_engine_destroy(et_fsdp_engine_t *engine) {
  if (engine == NULL) {
    return;
  }
  free(engine->params_full);
  free(engine->params_shard);
  free(engine->grads_full);
  free(engine->grad_shard);
  free(engine->m_shard);
  free(engine->v_shard);
  free(engine);
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_load_initial_params(
    et_fsdp_engine_t *engine, const float *full_params, size_t num_params) {
  if (engine == NULL || full_params == NULL ||
      (uint64_t)num_params != engine->cfg.num_params) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  if (!check_all_finite_f32(full_params, num_params)) {
    return ET_FSDP_ERR_NONFINITE;
  }

  const uint64_t shard = engine->shard_len;
  const uint64_t padded = engine->padded_params;
  const uint64_t base = (uint64_t)engine->comm->rank * shard;

  if (engine->cfg.stage == ET_ZERO_STAGE_3_FSDP) {
    for (uint64_t i = 0u; i < shard; ++i) {
      const uint64_t gidx = base + i;
      engine->params_shard[i] =
          (gidx < (uint64_t)num_params) ? canonical_f32(full_params[gidx]) : 0.0f;
    }
  } else {
    for (uint64_t i = 0u; i < (uint64_t)num_params; ++i) {
      engine->params_full[i] = canonical_f32(full_params[i]);
    }
    for (uint64_t i = (uint64_t)num_params; i < padded; ++i) {
      engine->params_full[i] = 0.0f;
    }
  }

  memset(engine->m_shard, 0, (size_t)shard * sizeof(float));
  memset(engine->v_shard, 0, (size_t)shard * sizeof(float));
  if (engine->grads_full != NULL) {
    memset(engine->grads_full, 0, (size_t)padded * sizeof(float));
  }
  if (engine->grad_shard != NULL) {
    memset(engine->grad_shard, 0, (size_t)shard * sizeof(float));
  }
  engine->step = 0u;
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_get_memory_report(
    const et_fsdp_engine_t *engine, et_fsdp_memory_report_v1 *out_report) {
  if (engine == NULL || out_report == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  memset(out_report, 0, sizeof(*out_report));
  const uint64_t f32 = sizeof(float);
  const uint64_t padded = engine->padded_params;
  const uint64_t shard = engine->shard_len;

  const uint64_t param_bytes =
      (engine->params_full != NULL) ? (padded * f32) : (shard * f32);
  const uint64_t grad_bytes =
      (engine->grads_full != NULL) ? (padded * f32) : (shard * f32);
  const uint64_t opt_bytes = 2u * shard * f32;
  const uint64_t total_bytes = param_bytes + grad_bytes + opt_bytes;
  const uint64_t unsharded_opt_bytes = 2u * padded * f32;
  const uint64_t unsharded_total = 4u * padded * f32;

  out_report->stage = engine->cfg.stage;
  out_report->rank = engine->comm->rank;
  out_report->world_size = engine->comm->world_size;
  out_report->reserved = 0u;
  out_report->num_params = engine->cfg.num_params;
  out_report->padded_params = padded;
  out_report->shard_len = shard;
  out_report->persistent_param_bytes = param_bytes;
  out_report->persistent_grad_bytes = grad_bytes;
  out_report->persistent_opt_bytes = opt_bytes;
  out_report->persistent_total_bytes = total_bytes;
  out_report->unsharded_baseline_bytes = unsharded_total;
  out_report->ephemeral_peak_bytes = engine->ephemeral_peak_bytes;
  out_report->optimizer_reduction_ratio =
      (double)unsharded_opt_bytes / (double)opt_bytes;
  out_report->total_persistent_reduction_ratio =
      (double)unsharded_total / (double)total_bytes;
  return ET_FSDP_OK;
}

ET_FSDP_API uint64_t et_fsdp_engine_get_step(const et_fsdp_engine_t *engine) {
  return engine == NULL ? 0u : engine->step;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_poison_ephemeral(float *buf,
                                                             size_t count) {
  if (buf == NULL || count == 0u) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  const float poison = bits_to_f32(ET_FSDP_POISON_F32_BITS);
  for (size_t i = 0u; i < count; ++i) {
    buf[i] = poison;
  }
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_gather_params(
    et_fsdp_engine_t *engine, float *out_full_params, size_t out_count,
    uint64_t tag, int timeout_ms) {
  if (engine == NULL || out_full_params == NULL || timeout_ms <= 0) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  if ((uint64_t)out_count != engine->cfg.num_params &&
      (uint64_t)out_count != engine->padded_params) {
    return ET_FSDP_ERR_INVALID_ARG;
  }

  if (engine->cfg.stage != ET_ZERO_STAGE_3_FSDP) {
    memcpy(out_full_params, engine->params_full, out_count * sizeof(float));
    return ET_FSDP_OK;
  }

  /* Stage 3 FSDP: materialize ephemeral full params via AllGather, then poison
   * and release the internal ephemeral buffer immediately! */
  const size_t padded_sz = (size_t)engine->padded_params;
  float *ephemeral = (float *)malloc(padded_sz * sizeof(float));
  if (ephemeral == NULL) {
    return ET_FSDP_ERR_ALLOC;
  }
  const uint64_t eph_bytes = (uint64_t)padded_sz * sizeof(float);
  if (eph_bytes > engine->ephemeral_peak_bytes) {
    engine->ephemeral_peak_bytes = eph_bytes;
  }

  const et_fsdp_status_t st = et_fsdp_allgather_f32(
      engine->comm, engine->params_shard, (size_t)engine->shard_len, ephemeral,
      tag, timeout_ms);
  if (st == ET_FSDP_OK) {
    memcpy(out_full_params, ephemeral, out_count * sizeof(float));
  }
  (void)et_fsdp_engine_poison_ephemeral(ephemeral, padded_sz);
  free(ephemeral);
  return st;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_step(
    et_fsdp_engine_t *engine, const float *raw_full_grads, size_t num_params,
    uint64_t tag, int timeout_ms, et_fsdp_step_info_v1 *out_info) {
  if (engine == NULL || raw_full_grads == NULL ||
      (uint64_t)num_params != engine->cfg.num_params || timeout_ms <= 0) {
    return ET_FSDP_ERR_INVALID_ARG;
  }

  const uint64_t padded = engine->padded_params;
  const uint64_t shard = engine->shard_len;
  const uint64_t base = (uint64_t)engine->comm->rank * shard;

  /* Prepare padded gradient buffer with trailing pad slots strictly 0.0f. */
  float *padded_grads = (float *)malloc((size_t)padded * sizeof(float));
  if (padded_grads == NULL) {
    return ET_FSDP_ERR_ALLOC;
  }
  memcpy(padded_grads, raw_full_grads, num_params * sizeof(float));
  for (uint64_t i = (uint64_t)num_params; i < padded; ++i) {
    padded_grads[i] = 0.0f;
  }

  /* Stage-specific gradient synchronization into staging buffer (atomic on
   * coordinated non-finite abort: live grad/param/m/v untouched on error!). */
  float *staged_grad_full = NULL;
  float *staged_grad_shard = (float *)malloc((size_t)shard * sizeof(float));
  if (staged_grad_shard == NULL) {
    free(padded_grads);
    return ET_FSDP_ERR_ALLOC;
  }

  et_fsdp_status_t st = ET_FSDP_OK;
  if (engine->cfg.stage == ET_ZERO_STAGE_1) {
    staged_grad_full = (float *)malloc((size_t)padded * sizeof(float));
    if (staged_grad_full == NULL) {
      free(padded_grads);
      free(staged_grad_shard);
      return ET_FSDP_ERR_ALLOC;
    }
    st = et_fsdp_allreduce_avg_f32(engine->comm, padded_grads, staged_grad_full,
                                   (size_t)padded, tag, timeout_ms);
    if (st == ET_FSDP_OK) {
      memcpy(staged_grad_shard, staged_grad_full + base,
             (size_t)shard * sizeof(float));
    }
  } else {
    st = et_fsdp_reduce_scatter_avg_f32(engine->comm, padded_grads,
                                        staged_grad_shard, (size_t)shard, tag,
                                        timeout_ms);
  }
  free(padded_grads);
  padded_grads = NULL;

  if (st != ET_FSDP_OK) {
    free(staged_grad_full);
    free(staged_grad_shard);
    return st;
  }

  /* Strictly zero any trailing pad slots on this rank's owned gradient shard. */
  for (uint64_t i = 0u; i < shard; ++i) {
    if (base + i >= engine->cfg.num_params) {
      staged_grad_shard[i] = 0.0f;
    }
  }

  /* Exact global L2 gradient norm via AllReduceSum_f64 of shard squared norms. */
  double local_sq_norm = 0.0;
  for (uint64_t i = 0u; i < shard; ++i) {
    const double g = (double)staged_grad_shard[i];
    local_sq_norm += g * g;
  }
  double global_sq_norm = 0.0;
  st = et_fsdp_allreduce_sum_f64(engine->comm, &local_sq_norm, &global_sq_norm,
                                 1u, tag + UINT64_C(0x10000000), timeout_ms);
  if (st != ET_FSDP_OK) {
    free(staged_grad_full);
    free(staged_grad_shard);
    return st;
  }

  const double global_norm = sqrt(global_sq_norm);
  double clip_scale = 1.0;
  uint32_t clipped = 0u;
  if (engine->cfg.max_grad_norm > 0.0 &&
      global_norm > engine->cfg.max_grad_norm) {
    clip_scale = engine->cfg.max_grad_norm / global_norm;
    clipped = 1u;
  }

  /* Stage shard-local decoupled AdamW update before committing live state. */
  float *next_p = (float *)malloc((size_t)shard * sizeof(float));
  float *next_m = (float *)malloc((size_t)shard * sizeof(float));
  float *next_v = (float *)malloc((size_t)shard * sizeof(float));
  if (next_p == NULL || next_m == NULL || next_v == NULL) {
    free(staged_grad_full);
    free(staged_grad_shard);
    free(next_p);
    free(next_m);
    free(next_v);
    return ET_FSDP_ERR_ALLOC;
  }

  const float *cur_p_shard = (engine->cfg.stage == ET_ZERO_STAGE_3_FSDP)
                                 ? engine->params_shard
                                 : (engine->params_full + base);
  const uint64_t next_step = engine->step + 1u;
  const double lr = engine->cfg.lr;
  const double b1 = engine->cfg.beta1;
  const double b2 = engine->cfg.beta2;
  const double eps = engine->cfg.eps;
  const double wd = engine->cfg.weight_decay;
  const double bc1 = 1.0 - ipow_f64(b1, next_step);
  const double bc2 = 1.0 - ipow_f64(b2, next_step);

  et_fsdp_status_t adam_st = ET_FSDP_OK;
  if (!isfinite(global_norm) || !isfinite(clip_scale) || bc1 <= 0.0 ||
      bc2 <= 0.0) {
    adam_st = ET_FSDP_ERR_NONFINITE;
  } else {
    for (uint64_t i = 0u; i < shard; ++i) {
      if (base + i >= engine->cfg.num_params) {
        next_p[i] = 0.0f;
        next_m[i] = 0.0f;
        next_v[i] = 0.0f;
        continue;
      }
      const double p = (double)cur_p_shard[i];
      const double g = (double)staged_grad_shard[i] * clip_scale;
      const double m = b1 * (double)engine->m_shard[i] + (1.0 - b1) * g;
      const double v = b2 * (double)engine->v_shard[i] + (1.0 - b2) * g * g;
      const double m_hat = m / bc1;
      const double v_hat = v / bc2;
      const double p_decayed = p * (1.0 - lr * wd);
      const double p_new = p_decayed - lr * (m_hat / (sqrt(v_hat) + eps));
      const float fp = canonical_f32((float)p_new);
      const float fm = canonical_f32((float)m);
      const float fv = canonical_f32((float)v);
      if (!isfinite(p_new) || !isfinite(m) || !isfinite(v) || !isfinite(fp) ||
          !isfinite(fm) || !isfinite(fv) || fv < 0.0f) {
        adam_st = ET_FSDP_ERR_NONFINITE;
        break;
      }
      next_p[i] = fp;
      next_m[i] = fm;
      next_v[i] = fv;
    }
  }

  if (adam_st != ET_FSDP_OK) {
    free(staged_grad_full);
    free(staged_grad_shard);
    free(next_p);
    free(next_m);
    free(next_v);
    return adam_st;
  }

  /* For Stage 1 & 2, synchronize updated parameter shards back into persistent
   * params_full via AllGather BEFORE committing live state! */
  if (engine->cfg.stage != ET_ZERO_STAGE_3_FSDP) {
    float *next_full = (float *)malloc((size_t)padded * sizeof(float));
    if (next_full == NULL) {
      free(staged_grad_full);
      free(staged_grad_shard);
      free(next_p);
      free(next_m);
      free(next_v);
      return ET_FSDP_ERR_ALLOC;
    }
    st = et_fsdp_allgather_f32(engine->comm, next_p, (size_t)shard, next_full,
                               tag + UINT64_C(0x20000000), timeout_ms);
    if (st != ET_FSDP_OK) {
      free(next_full);
      free(staged_grad_full);
      free(staged_grad_shard);
      free(next_p);
      free(next_m);
      free(next_v);
      return st;
    }
    memcpy(engine->params_full, next_full, (size_t)padded * sizeof(float));
    free(next_full);
    if (engine->grads_full != NULL) {
      memcpy(engine->grads_full, staged_grad_full,
             (size_t)padded * sizeof(float));
    }
    if (engine->grad_shard != NULL) {
      memcpy(engine->grad_shard, staged_grad_shard,
             (size_t)shard * sizeof(float));
    }
  } else {
    /* Stage 3 FSDP: commit ONLY shard-local params_shard, grad_shard, m, v! */
    memcpy(engine->params_shard, next_p, (size_t)shard * sizeof(float));
    memcpy(engine->grad_shard, staged_grad_shard,
           (size_t)shard * sizeof(float));
  }

  memcpy(engine->m_shard, next_m, (size_t)shard * sizeof(float));
  memcpy(engine->v_shard, next_v, (size_t)shard * sizeof(float));
  engine->step = next_step;

  free(staged_grad_full);
  free(staged_grad_shard);
  free(next_p);
  free(next_m);
  free(next_v);

  if (out_info != NULL) {
    out_info->step = next_step;
    out_info->global_grad_norm = global_norm;
    out_info->clip_scale = clip_scale;
    out_info->clipped = clipped;
    out_info->reserved = 0u;
  }
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_get_param_shard(
    const et_fsdp_engine_t *engine, float *out_shard, size_t shard_len) {
  if (engine == NULL || out_shard == NULL ||
      (uint64_t)shard_len != engine->shard_len) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  const float *src =
      (engine->cfg.stage == ET_ZERO_STAGE_3_FSDP)
          ? engine->params_shard
          : (engine->params_full +
             (size_t)engine->comm->rank * (size_t)engine->shard_len);
  memcpy(out_shard, src, shard_len * sizeof(float));
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_get_moments_shard(
    const et_fsdp_engine_t *engine, float *out_m, float *out_v,
    size_t shard_len) {
  if (engine == NULL || out_m == NULL || out_v == NULL ||
      (uint64_t)shard_len != engine->shard_len) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  memcpy(out_m, engine->m_shard, shard_len * sizeof(float));
  memcpy(out_v, engine->v_shard, shard_len * sizeof(float));
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_get_grad_shard(
    const et_fsdp_engine_t *engine, float *out_grad, size_t shard_len) {
  if (engine == NULL || out_grad == NULL ||
      (uint64_t)shard_len != engine->shard_len) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  const float *src =
      (engine->grad_shard != NULL)
          ? engine->grad_shard
          : (engine->grads_full +
             (size_t)engine->comm->rank * (size_t)engine->shard_len);
  memcpy(out_grad, src, shard_len * sizeof(float));
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_gather_full_params_sha256(
    et_fsdp_engine_t *engine, uint64_t tag, int timeout_ms,
    uint8_t out_digest[32]) {
  if (engine == NULL || out_digest == NULL || timeout_ms <= 0) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  const size_t n = (size_t)engine->cfg.num_params;
  float *full = (float *)malloc(n * sizeof(float));
  if (full == NULL) {
    return ET_FSDP_ERR_ALLOC;
  }
  const et_fsdp_status_t st =
      et_fsdp_engine_gather_params(engine, full, n, tag, timeout_ms);
  if (st != ET_FSDP_OK) {
    free(full);
    return st;
  }
  uint8_t *le_bytes = (uint8_t *)malloc(n * 4u);
  if (le_bytes == NULL) {
    free(full);
    return ET_FSDP_ERR_ALLOC;
  }
  for (size_t i = 0u; i < n; ++i) {
    put_u32_le(le_bytes + i * 4u, f32_to_bits(canonical_f32(full[i])));
  }
  const et_fsdp_status_t hst = et_fsdp_sha256_bytes(le_bytes, n * 4u, out_digest);
  (void)et_fsdp_engine_poison_ephemeral(full, n);
  free(le_bytes);
  free(full);
  return hst;
}

/* --- Versioned, SHA-256 Checksummed Atomic 0600 Checkpoint Save & O_NOFOLLOW Load --- */

struct path_parts {
  char *parent;
  char *base;
};

static void free_path_parts(struct path_parts *p) {
  free(p->parent);
  free(p->base);
  p->parent = NULL;
  p->base = NULL;
}

static int split_path(const char *path, struct path_parts *p) {
  memset(p, 0, sizeof(*p));
  if (path == NULL || path[0] == '\0') {
    return -1;
  }
  const char *slash = strrchr(path, '/');
  const char *base = (slash == NULL) ? path : (slash + 1);
  const size_t blen = strlen(base);
  if (blen == 0u || (blen == 1u && base[0] == '.') ||
      (blen == 2u && base[0] == '.' && base[1] == '.')) {
    return -1;
  }
  p->base = (char *)malloc(blen + 1u);
  if (p->base == NULL) {
    return -1;
  }
  memcpy(p->base, base, blen + 1u);
  if (slash == NULL) {
    p->parent = (char *)malloc(2u);
    if (p->parent != NULL) {
      memcpy(p->parent, ".", 2u);
    }
  } else if (slash == path) {
    p->parent = (char *)malloc(2u);
    if (p->parent != NULL) {
      memcpy(p->parent, "/", 2u);
    }
  } else {
    size_t plen = (size_t)(slash - path);
    while (plen > 1u && path[plen - 1u] == '/') {
      --plen;
    }
    p->parent = (char *)malloc(plen + 1u);
    if (p->parent != NULL) {
      memcpy(p->parent, path, plen);
      p->parent[plen] = '\0';
    }
  }
  if (p->parent == NULL) {
    free_path_parts(p);
    return -1;
  }
  return 0;
}

static int write_all_fd(int fd, const uint8_t *data, size_t len) {
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = write(fd, data + off, len - off);
    if (n > 0) {
      off += (size_t)n;
    } else if (n == 0) {
      errno = EIO;
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

static int read_all_fd(int fd, uint8_t *dst, size_t len) {
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = read(fd, dst + off, len - off);
    if (n > 0) {
      off += (size_t)n;
    } else if (n == 0) {
      errno = ENODATA;
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

static int fsync_fd_retry(int fd) {
  int rc;
  do {
    rc = fsync(fd);
  } while (rc < 0 && errno == EINTR);
  return rc;
}

static int fill_random_16(uint8_t out[16]) {
  size_t off = 0u;
  while (off < 16u) {
    const ssize_t n = getrandom(out + off, 16u - off, 0u);
    if (n > 0) {
      off += (size_t)n;
    } else if (n == 0) {
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

static et_fsdp_status_t atomic_write_0600(const char *path, const uint8_t *data,
                                          size_t len) {
  struct path_parts parts;
  if (split_path(path, &parts) != 0) {
    return ET_FSDP_ERR_INVALID_ARG;
  }

  int dir_fd;
  do {
    dir_fd = open(parts.parent, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  } while (dir_fd < 0 && errno == EINTR);
  if (dir_fd < 0) {
    free_path_parts(&parts);
    return ET_FSDP_ERR_IO;
  }

  /* Refuse existing symlink or non-regular destination before writing. */
  struct stat dst_st;
  if (fstatat(dir_fd, parts.base, &dst_st, AT_SYMLINK_NOFOLLOW) == 0) {
    if (!S_ISREG(dst_st.st_mode)) {
      close(dir_fd);
      free_path_parts(&parts);
      return ET_FSDP_ERR_IO;
    }
  } else if (errno != ENOENT) {
    close(dir_fd);
    free_path_parts(&parts);
    return ET_FSDP_ERR_IO;
  }

  int tmp_fd = -1;
  char tmp_name[48];
  static const char hex[] = "0123456789abcdef";
  for (int attempt = 0; attempt < 16; ++attempt) {
    uint8_t rnd[16];
    if (fill_random_16(rnd) != 0) {
      close(dir_fd);
      free_path_parts(&parts);
      return ET_FSDP_ERR_IO;
    }
    memcpy(tmp_name, ".etfs-", 6u);
    for (size_t i = 0u; i < 16u; ++i) {
      tmp_name[6u + i * 2u] = hex[rnd[i] >> 4u];
      tmp_name[7u + i * 2u] = hex[rnd[i] & 0x0fu];
    }
    memcpy(tmp_name + 38u, ".tmp", 5u);
    do {
      tmp_fd = openat(dir_fd, tmp_name,
                      O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                      (mode_t)0600);
    } while (tmp_fd < 0 && errno == EINTR);
    if (tmp_fd >= 0) {
      break;
    }
    if (errno != EEXIST) {
      close(dir_fd);
      free_path_parts(&parts);
      return ET_FSDP_ERR_IO;
    }
  }
  if (tmp_fd < 0) {
    close(dir_fd);
    free_path_parts(&parts);
    return ET_FSDP_ERR_IO;
  }

  et_fsdp_status_t st = ET_FSDP_OK;
  if (fchmod(tmp_fd, (mode_t)0600) != 0 ||
      write_all_fd(tmp_fd, data, len) != 0 || fsync_fd_retry(tmp_fd) != 0) {
    st = ET_FSDP_ERR_IO;
  }
  if (close(tmp_fd) != 0 && st == ET_FSDP_OK) {
    st = ET_FSDP_ERR_IO;
  }

  if (st == ET_FSDP_OK) {
    /* Re-verify destination is not a symlink right before renameat. */
    if (fstatat(dir_fd, parts.base, &dst_st, AT_SYMLINK_NOFOLLOW) == 0 &&
        !S_ISREG(dst_st.st_mode)) {
      st = ET_FSDP_ERR_IO;
    } else if (renameat(dir_fd, tmp_name, dir_fd, parts.base) != 0 ||
               fsync_fd_retry(dir_fd) != 0) {
      st = ET_FSDP_ERR_IO;
    }
  }
  if (st != ET_FSDP_OK) {
    (void)unlinkat(dir_fd, tmp_name, 0);
  }

  close(dir_fd);
  free_path_parts(&parts);
  return st;
}

static uint64_t expected_ckpt_payload_bytes(const et_fsdp_engine_t *engine) {
  const uint64_t param_count = (engine->cfg.stage == ET_ZERO_STAGE_3_FSDP)
                                   ? engine->shard_len
                                   : engine->padded_params;
  return (param_count + 2u * engine->shard_len) * (uint64_t)sizeof(float);
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_save_checkpoint(
    const et_fsdp_engine_t *engine, const char *path) {
  if (engine == NULL || path == NULL || path[0] == '\0') {
    return ET_FSDP_ERR_INVALID_ARG;
  }

  const uint64_t payload_bytes = expected_ckpt_payload_bytes(engine);
  const uint64_t total_bytes =
      (uint64_t)ET_FSDP_CKPT_HEADER_BYTES + payload_bytes +
      (uint64_t)ET_FSDP_CKPT_CHECKSUM_BYTES;
  uint8_t *buf = (uint8_t *)calloc(1u, (size_t)total_bytes);
  if (buf == NULL) {
    return ET_FSDP_ERR_ALLOC;
  }

  /* Fixed 128-byte little-endian header. */
  memcpy(buf + 0u, ET_FSDP_CKPT_MAGIC, ET_FSDP_CKPT_MAGIC_BYTES);
  put_u16_le(buf + 8u, (uint16_t)ET_FSDP_CKPT_VERSION_MAJOR);
  put_u16_le(buf + 10u, (uint16_t)ET_FSDP_CKPT_VERSION_MINOR);
  put_u32_le(buf + 12u, engine->cfg.stage);
  put_u32_le(buf + 16u, engine->comm->rank);
  put_u32_le(buf + 20u, engine->comm->world_size);
  put_u32_le(buf + 24u, 0u);
  put_u32_le(buf + 28u, 0u);
  put_u64_le(buf + 32u, engine->cfg.num_params);
  put_u64_le(buf + 40u, engine->padded_params);
  put_u64_le(buf + 48u, engine->shard_len);
  put_u64_le(buf + 56u, engine->step);
  put_u64_le(buf + 64u, f64_to_bits(engine->cfg.lr));
  put_u64_le(buf + 72u, f64_to_bits(engine->cfg.beta1));
  put_u64_le(buf + 80u, f64_to_bits(engine->cfg.beta2));
  put_u64_le(buf + 88u, f64_to_bits(engine->cfg.eps));
  put_u64_le(buf + 96u, f64_to_bits(engine->cfg.weight_decay));
  put_u64_le(buf + 104u, f64_to_bits(engine->cfg.max_grad_norm));
  put_u64_le(buf + 112u, payload_bytes);
  put_u64_le(buf + 120u, 0u);

  /* Serialize parameter slice (or full for stage 1/2), m_shard, v_shard. */
  size_t off = ET_FSDP_CKPT_HEADER_BYTES;
  const float *p_src = (engine->cfg.stage == ET_ZERO_STAGE_3_FSDP)
                           ? engine->params_shard
                           : engine->params_full;
  const uint64_t p_cnt = (engine->cfg.stage == ET_ZERO_STAGE_3_FSDP)
                             ? engine->shard_len
                             : engine->padded_params;
  for (uint64_t i = 0u; i < p_cnt; ++i) {
    put_u32_le(buf + off, f32_to_bits(canonical_f32(p_src[i])));
    off += 4u;
  }
  for (uint64_t i = 0u; i < engine->shard_len; ++i) {
    put_u32_le(buf + off, f32_to_bits(canonical_f32(engine->m_shard[i])));
    off += 4u;
  }
  for (uint64_t i = 0u; i < engine->shard_len; ++i) {
    put_u32_le(buf + off, f32_to_bits(canonical_f32(engine->v_shard[i])));
    off += 4u;
  }

  /* Append 32-byte SHA-256 over header + payload. */
  uint8_t digest[32];
  (void)et_fsdp_sha256_bytes(buf, off, digest);
  memcpy(buf + off, digest, ET_FSDP_CKPT_CHECKSUM_BYTES);

  const et_fsdp_status_t st = atomic_write_0600(path, buf, (size_t)total_bytes);
  free(buf);
  return st;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_load_checkpoint(
    et_fsdp_engine_t *engine, const char *path) {
  if (engine == NULL || path == NULL || path[0] == '\0') {
    return ET_FSDP_ERR_INVALID_ARG;
  }

  int fd;
  do {
    fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return ET_FSDP_ERR_IO;
  }

  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    close(fd);
    return ET_FSDP_ERR_IO;
  }

  const uint64_t min_bytes =
      (uint64_t)ET_FSDP_CKPT_HEADER_BYTES + (uint64_t)ET_FSDP_CKPT_CHECKSUM_BYTES;
  if (st.st_size < 0 || (uint64_t)st.st_size < min_bytes ||
      (uint64_t)st.st_size > ET_FSDP_MAX_BUFFER_BYTES) {
    close(fd);
    return (st.st_size >= 0 && (uint64_t)st.st_size < min_bytes)
               ? ET_FSDP_ERR_TRUNCATED
               : ET_FSDP_ERR_IO;
  }

  const size_t file_len = (size_t)st.st_size;
  uint8_t *buf = (uint8_t *)malloc(file_len);
  if (buf == NULL) {
    close(fd);
    return ET_FSDP_ERR_ALLOC;
  }
  if (read_all_fd(fd, buf, file_len) != 0) {
    free(buf);
    close(fd);
    return ET_FSDP_ERR_IO;
  }
  /* Reject trailing bytes if file grew concurrently. */
  uint8_t extra = 0u;
  const ssize_t tail_rc = read(fd, &extra, 1u);
  close(fd);
  if (tail_rc != 0) {
    free(buf);
    return ET_FSDP_ERR_TRUNCATED;
  }

  /* 1. Magic and version checks. */
  if (memcmp(buf, ET_FSDP_CKPT_MAGIC, ET_FSDP_CKPT_MAGIC_BYTES) != 0) {
    free(buf);
    return ET_FSDP_ERR_BAD_MAGIC;
  }
  const uint16_t vmaj = get_u16_le(buf + 8u);
  const uint16_t vmin = get_u16_le(buf + 10u);
  if (vmaj != (uint16_t)ET_FSDP_CKPT_VERSION_MAJOR ||
      vmin != (uint16_t)ET_FSDP_CKPT_VERSION_MINOR) {
    free(buf);
    return ET_FSDP_ERR_VERSION_MISMATCH;
  }

  /* 2. Exact length & SHA-256 verification before trusting payload fields. */
  const uint64_t hdr_payload = get_u64_le(buf + 112u);
  const uint64_t expected_file_len = (uint64_t)ET_FSDP_CKPT_HEADER_BYTES +
                                     hdr_payload +
                                     (uint64_t)ET_FSDP_CKPT_CHECKSUM_BYTES;
  if (hdr_payload > ET_FSDP_MAX_BUFFER_BYTES ||
      expected_file_len != (uint64_t)file_len) {
    free(buf);
    return ET_FSDP_ERR_TRUNCATED;
  }

  uint8_t actual_digest[32];
  (void)et_fsdp_sha256_bytes(buf, file_len - ET_FSDP_CKPT_CHECKSUM_BYTES,
                             actual_digest);
  if (memcmp(actual_digest, buf + file_len - ET_FSDP_CKPT_CHECKSUM_BYTES,
             ET_FSDP_CKPT_CHECKSUM_BYTES) != 0) {
    free(buf);
    return ET_FSDP_ERR_CHECKSUM_MISMATCH;
  }

  /* 3. Verify topology, stage, rank, world_size, shape, and AdamW config match. */
  const uint32_t stage = get_u32_le(buf + 12u);
  const uint32_t rank = get_u32_le(buf + 16u);
  const uint32_t world = get_u32_le(buf + 20u);
  const uint32_t flags = get_u32_le(buf + 24u);
  const uint32_t res = get_u32_le(buf + 28u);
  const uint64_t num_params = get_u64_le(buf + 32u);
  const uint64_t padded_params = get_u64_le(buf + 40u);
  const uint64_t shard_len = get_u64_le(buf + 48u);
  const uint64_t step = get_u64_le(buf + 56u);
  const double lr = bits_to_f64(get_u64_le(buf + 64u));
  const double b1 = bits_to_f64(get_u64_le(buf + 72u));
  const double b2 = bits_to_f64(get_u64_le(buf + 80u));
  const double eps = bits_to_f64(get_u64_le(buf + 88u));
  const double wd = bits_to_f64(get_u64_le(buf + 96u));
  const double max_gn = bits_to_f64(get_u64_le(buf + 104u));
  const uint64_t res2 = get_u64_le(buf + 120u);

  if (flags != 0u || res != 0u || res2 != 0u || stage != engine->cfg.stage ||
      rank != engine->comm->rank || world != engine->comm->world_size ||
      num_params != engine->cfg.num_params ||
      padded_params != engine->padded_params || shard_len != engine->shard_len ||
      hdr_payload != expected_ckpt_payload_bytes(engine) ||
      f64_to_bits(lr) != f64_to_bits(engine->cfg.lr) ||
      f64_to_bits(b1) != f64_to_bits(engine->cfg.beta1) ||
      f64_to_bits(b2) != f64_to_bits(engine->cfg.beta2) ||
      f64_to_bits(eps) != f64_to_bits(engine->cfg.eps) ||
      f64_to_bits(wd) != f64_to_bits(engine->cfg.weight_decay) ||
      f64_to_bits(max_gn) != f64_to_bits(engine->cfg.max_grad_norm)) {
    free(buf);
    return ET_FSDP_ERR_CONFIG_MISMATCH;
  }

  /* 4. Stage and validate all float payloads atomically before committing. */
  const uint64_t p_cnt = (engine->cfg.stage == ET_ZERO_STAGE_3_FSDP)
                             ? engine->shard_len
                             : engine->padded_params;
  float *tmp_p = (float *)malloc((size_t)p_cnt * sizeof(float));
  float *tmp_m = (float *)malloc((size_t)shard_len * sizeof(float));
  float *tmp_v = (float *)malloc((size_t)shard_len * sizeof(float));
  if (tmp_p == NULL || tmp_m == NULL || tmp_v == NULL) {
    free(buf);
    free(tmp_p);
    free(tmp_m);
    free(tmp_v);
    return ET_FSDP_ERR_ALLOC;
  }

  size_t off = ET_FSDP_CKPT_HEADER_BYTES;
  const uint64_t base = (uint64_t)engine->comm->rank * shard_len;
  et_fsdp_status_t val_st = ET_FSDP_OK;

  for (uint64_t i = 0u; i < p_cnt; ++i) {
    const float v = bits_to_f32(get_u32_le(buf + off));
    off += 4u;
    const uint64_t gidx =
        (engine->cfg.stage == ET_ZERO_STAGE_3_FSDP) ? (base + i) : i;
    if (!isfinite(v) || (gidx >= num_params && v != 0.0f)) {
      val_st = ET_FSDP_ERR_NONFINITE;
      break;
    }
    tmp_p[i] = canonical_f32(v);
  }
  for (uint64_t i = 0u; val_st == ET_FSDP_OK && i < shard_len; ++i) {
    const float v = bits_to_f32(get_u32_le(buf + off));
    off += 4u;
    if (!isfinite(v) || (base + i >= num_params && v != 0.0f)) {
      val_st = ET_FSDP_ERR_NONFINITE;
      break;
    }
    tmp_m[i] = canonical_f32(v);
  }
  for (uint64_t i = 0u; val_st == ET_FSDP_OK && i < shard_len; ++i) {
    const float v = bits_to_f32(get_u32_le(buf + off));
    off += 4u;
    if (!isfinite(v) || v < 0.0f || (base + i >= num_params && v != 0.0f)) {
      val_st = ET_FSDP_ERR_NONFINITE;
      break;
    }
    tmp_v[i] = canonical_f32(v);
  }

  free(buf);
  if (val_st != ET_FSDP_OK) {
    free(tmp_p);
    free(tmp_m);
    free(tmp_v);
    return val_st;
  }

  /* Commit validated state atomically. */
  if (engine->cfg.stage == ET_ZERO_STAGE_3_FSDP) {
    memcpy(engine->params_shard, tmp_p, (size_t)p_cnt * sizeof(float));
  } else {
    memcpy(engine->params_full, tmp_p, (size_t)p_cnt * sizeof(float));
  }
  memcpy(engine->m_shard, tmp_m, (size_t)shard_len * sizeof(float));
  memcpy(engine->v_shard, tmp_v, (size_t)shard_len * sizeof(float));
  if (engine->grads_full != NULL) {
    memset(engine->grads_full, 0, (size_t)padded_params * sizeof(float));
  }
  if (engine->grad_shard != NULL) {
    memset(engine->grad_shard, 0, (size_t)shard_len * sizeof(float));
  }
  engine->step = step;

  free(tmp_p);
  free(tmp_m);
  free(tmp_v);
  return ET_FSDP_OK;
}

/* --- Deterministic 2-Layer Tanh MLP + Cross-Entropy Workload --- */

static uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

static float splitmix64_sym_f32(uint64_t *state, double scale) {
  const uint64_t r = splitmix64_next(state);
  const uint32_t u = (uint32_t)(r >> 40u); /* 24-bit integer in [0, 2^24-1] */
  const double v = (((double)u / 8388608.0) - 1.0) * scale;
  return canonical_f32((float)v);
}

ET_FSDP_API et_fsdp_status_t et_fsdp_mlp_num_params(size_t in_dim,
                                                    size_t hidden_dim,
                                                    size_t num_classes,
                                                    uint64_t *out_num_params) {
  if (in_dim == 0u || hidden_dim == 0u || num_classes < 2u ||
      out_num_params == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  if (in_dim > 4096u || hidden_dim > 4096u || num_classes > 4096u) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  const uint64_t w1 = (uint64_t)in_dim * (uint64_t)hidden_dim;
  const uint64_t b1 = (uint64_t)hidden_dim;
  const uint64_t w2 = (uint64_t)hidden_dim * (uint64_t)num_classes;
  const uint64_t b2 = (uint64_t)num_classes;
  const uint64_t total = w1 + b1 + w2 + b2;
  if (total == 0u || total > ET_FSDP_MAX_PARAMS) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  *out_num_params = total;
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_mlp_init_params(
    size_t in_dim, size_t hidden_dim, size_t num_classes, uint64_t seed,
    float *out_params, size_t num_params) {
  uint64_t expected = 0u;
  const et_fsdp_status_t st =
      et_fsdp_mlp_num_params(in_dim, hidden_dim, num_classes, &expected);
  if (st != ET_FSDP_OK) {
    return st;
  }
  if (out_params == NULL || (uint64_t)num_params != expected) {
    return ET_FSDP_ERR_INVALID_ARG;
  }

  uint64_t rng = seed ^ UINT64_C(0xa5a5a5a55a5a5a5a);
  const size_t w1_len = in_dim * hidden_dim;
  const size_t b1_len = hidden_dim;
  const size_t w2_len = hidden_dim * num_classes;
  const size_t b2_len = num_classes;
  const double s1 = 0.5 / sqrt((double)in_dim);
  const double s2 = 0.5 / sqrt((double)hidden_dim);

  size_t pos = 0u;
  for (size_t i = 0u; i < w1_len; ++i) {
    out_params[pos++] = splitmix64_sym_f32(&rng, s1);
  }
  for (size_t i = 0u; i < b1_len; ++i) {
    out_params[pos++] = splitmix64_sym_f32(&rng, 0.05);
  }
  for (size_t i = 0u; i < w2_len; ++i) {
    out_params[pos++] = splitmix64_sym_f32(&rng, s2);
  }
  for (size_t i = 0u; i < b2_len; ++i) {
    out_params[pos++] = splitmix64_sym_f32(&rng, 0.05);
  }
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_mlp_make_batch(
    size_t in_dim, size_t num_classes, size_t global_batch, uint32_t rank,
    uint32_t world_size, uint64_t step_idx, float *out_x, int32_t *out_targets) {
  if (in_dim == 0u || num_classes < 2u || global_batch == 0u ||
      !valid_world_size(world_size) || rank >= world_size ||
      (global_batch % (size_t)world_size) != 0u || out_x == NULL ||
      out_targets == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }
  /* Fixed learnable structured global batch (step_idx unused for batch
   * identity so full-batch AdamW loss decreases strictly across all steps,
   * while rank r receives exact shard [r*local_batch .. (r+1)*local_batch)). */
  (void)step_idx;
  const size_t local_batch = global_batch / (size_t)world_size;
  const size_t start_sample = (size_t)rank * local_batch;

  for (size_t lb = 0u; lb < local_batch; ++lb) {
    const size_t s = start_sample + lb;
    const int32_t cls = (int32_t)(s % num_classes);
    out_targets[lb] = cls;

    uint64_t rng = UINT64_C(0x517cc1b727220a95) ^
                   ((uint64_t)(s + 1u) * UINT64_C(0xbf58476d1ce4e5b9));
    for (size_t j = 0u; j < in_dim; ++j) {
      const double proto =
          0.9 * cos(0.75 * (double)(cls + 1) * (double)(j + 1));
      const double jitter = (double)splitmix64_sym_f32(&rng, 0.15);
      out_x[lb * in_dim + j] = canonical_f32((float)(proto + jitter));
    }
  }
  return ET_FSDP_OK;
}

ET_FSDP_API et_fsdp_status_t et_fsdp_mlp_forward_backward(
    size_t in_dim, size_t hidden_dim, size_t num_classes, size_t batch_size,
    const float *params, const float *x, const int32_t *targets,
    double *out_loss, float *out_grads) {
  uint64_t num_params_u64 = 0u;
  const et_fsdp_status_t st =
      et_fsdp_mlp_num_params(in_dim, hidden_dim, num_classes, &num_params_u64);
  if (st != ET_FSDP_OK) {
    return st;
  }
  if (batch_size == 0u || params == NULL || x == NULL || targets == NULL ||
      out_loss == NULL) {
    return ET_FSDP_ERR_INVALID_ARG;
  }

  const size_t num_params = (size_t)num_params_u64;
  if (!check_all_finite_f32(params, num_params) ||
      !check_all_finite_f32(x, batch_size * in_dim)) {
    return ET_FSDP_ERR_NONFINITE;
  }
  for (size_t n = 0u; n < batch_size; ++n) {
    if (targets[n] < 0 || (size_t)targets[n] >= num_classes) {
      return ET_FSDP_ERR_INVALID_ARG;
    }
  }

  const float *w1 = params;
  const float *b1 = w1 + in_dim * hidden_dim;
  const float *w2 = b1 + hidden_dim;
  const float *b2 = w2 + hidden_dim * num_classes;

  double *h = (double *)malloc(hidden_dim * sizeof(double));
  double *logits = (double *)malloc(num_classes * sizeof(double));
  double *probs = (double *)malloc(num_classes * sizeof(double));
  double *dh = (double *)malloc(hidden_dim * sizeof(double));
  double *g_acc = (out_grads != NULL)
                      ? (double *)calloc(num_params, sizeof(double))
                      : NULL;

  if (h == NULL || logits == NULL || probs == NULL || dh == NULL ||
      (out_grads != NULL && g_acc == NULL)) {
    free(h);
    free(logits);
    free(probs);
    free(dh);
    free(g_acc);
    return ET_FSDP_ERR_ALLOC;
  }

  double *gw1 = g_acc;
  double *gb1 = (g_acc != NULL) ? (gw1 + in_dim * hidden_dim) : NULL;
  double *gw2 = (g_acc != NULL) ? (gb1 + hidden_dim) : NULL;
  double *gb2 = (g_acc != NULL) ? (gw2 + hidden_dim * num_classes) : NULL;

  double total_loss = 0.0;
  const double inv_b = 1.0 / (double)batch_size;

  for (size_t n = 0u; n < batch_size; ++n) {
    const float *xn = x + n * in_dim;
    const size_t y = (size_t)targets[n];

    /* Layer 1: h = tanh(x * W1 + b1) */
    for (size_t j = 0u; j < hidden_dim; ++j) {
      double acc = (double)b1[j];
      for (size_t i = 0u; i < in_dim; ++i) {
        acc += (double)xn[i] * (double)w1[i * hidden_dim + j];
      }
      h[j] = tanh(acc);
    }

    /* Layer 2: logits = h * W2 + b2 */
    double max_logit = -INFINITY;
    for (size_t c = 0u; c < num_classes; ++c) {
      double acc = (double)b2[c];
      for (size_t j = 0u; j < hidden_dim; ++j) {
        acc += h[j] * (double)w2[j * num_classes + c];
      }
      logits[c] = acc;
      if (acc > max_logit) {
        max_logit = acc;
      }
    }

    /* Stable log-softmax & cross-entropy. */
    double sum_exp = 0.0;
    for (size_t c = 0u; c < num_classes; ++c) {
      probs[c] = exp(logits[c] - max_logit);
      sum_exp += probs[c];
    }
    const double log_z = max_logit + log(sum_exp);
    total_loss += (log_z - logits[y]);

    if (g_acc != NULL) {
      for (size_t c = 0u; c < num_classes; ++c) {
        probs[c] /= sum_exp;
      }
      /* dlogits = (probs - one_hot(y)) / B */
      for (size_t j = 0u; j < hidden_dim; ++j) {
        dh[j] = 0.0;
      }
      for (size_t c = 0u; c < num_classes; ++c) {
        const double dl = (probs[c] - ((c == y) ? 1.0 : 0.0)) * inv_b;
        gb2[c] += dl;
        for (size_t j = 0u; j < hidden_dim; ++j) {
          gw2[j * num_classes + c] += h[j] * dl;
          dh[j] += dl * (double)w2[j * num_classes + c];
        }
      }
      for (size_t j = 0u; j < hidden_dim; ++j) {
        const double dh_pre = dh[j] * (1.0 - h[j] * h[j]);
        gb1[j] += dh_pre;
        for (size_t i = 0u; i < in_dim; ++i) {
          gw1[i * hidden_dim + j] += (double)xn[i] * dh_pre;
        }
      }
    }
  }

  const double mean_loss = total_loss * inv_b;
  et_fsdp_status_t ret = ET_FSDP_OK;
  if (!isfinite(mean_loss)) {
    ret = ET_FSDP_ERR_NONFINITE;
  } else if (out_grads != NULL) {
    for (size_t i = 0u; i < num_params; ++i) {
      const float g = canonical_f32((float)g_acc[i]);
      if (!isfinite(g_acc[i]) || !isfinite(g)) {
        ret = ET_FSDP_ERR_NONFINITE;
        break;
      }
      out_grads[i] = g;
    }
  }

  if (ret == ET_FSDP_OK) {
    *out_loss = mean_loss;
  }

  free(h);
  free(logits);
  free(probs);
  free(dh);
  free(g_acc);
  return ret;
}
