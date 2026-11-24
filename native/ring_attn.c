#define _GNU_SOURCE
#include "eshkol_transformer/ring_attn_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   FLT_MAX_EXP == 128 && FLT_EVAL_METHOD == 0,
               "Ring Attention requires IEEE-754 binary32 float");
_Static_assert(sizeof(double) == 8 && DBL_MANT_DIG == 53 &&
                   DBL_MAX_EXP == 1024,
               "Ring Attention requires IEEE-754 binary64 double");

#define RING_WIRE_MAGIC UINT64_C(0x455452494E474131)

struct et_ring_attn_comm {
  uint32_t rank;
  uint32_t world_size;
  int send_fd;
  int recv_fd;
  et_ring_attn_status_t fatal;
  uint64_t total_bytes_sent;
  uint64_t total_bytes_received;
};

typedef struct ring_wire_header {
  uint64_t magic;
  uint32_t phase;
  uint32_t round;
  uint32_t kv_rank;
  uint32_t sender_rank;
  uint64_t payload_bytes;
} ring_wire_header;

typedef struct child_reply_header {
  int32_t status;
  uint32_t reserved;
  et_ring_attn_stats_v1 stats;
} child_reply_header;

typedef struct span_desc {
  const void *ptr;
  size_t bytes;
} span_desc;

static int retryable_errno(int err) {
  return err == EINTR || err == EAGAIN || err == EWOULDBLOCK;
}

static int64_t monotonic_ms(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    return -1;
  return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static int valid_world_size(uint32_t w) {
  return w == 1u || w == 2u || w == 4u;
}

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  if (p == NULL || alignment == 0u)
    return 0;
  if (((uintptr_t)p % alignment) != 0u)
    return 0;
  if ((uintptr_t)p > UINTPTR_MAX - bytes)
    return 0;
  return 1;
}

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  if (a_bytes == 0u || b_bytes == 0u)
    return 0;
  const uintptr_t a0 = (uintptr_t)a;
  const uintptr_t a1 = a0 + a_bytes;
  const uintptr_t b0 = (uintptr_t)b;
  const uintptr_t b1 = b0 + b_bytes;
  return a0 < b1 && b0 < a1;
}

static int check_no_output_aliasing(const span_desc *inputs, size_t n_inputs,
                                    const span_desc *outputs,
                                    size_t n_outputs) {
  for (size_t i = 0; i < n_outputs; ++i) {
    for (size_t j = i + 1u; j < n_outputs; ++j) {
      if (spans_overlap(outputs[i].ptr, outputs[i].bytes, outputs[j].ptr,
                        outputs[j].bytes))
        return 0;
    }
    for (size_t k = 0; k < n_inputs; ++k) {
      if (spans_overlap(outputs[i].ptr, outputs[i].bytes, inputs[k].ptr,
                        inputs[k].bytes))
        return 0;
    }
  }
  return 1;
}

static int all_f32_finite(const float *buf, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(buf[i]))
      return 0;
  }
  return 1;
}

static int all_f64_finite(const double *buf, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(buf[i]))
      return 0;
  }
  return 1;
}

static int round_f64_to_f32_checked(const double *src, float *dst,
                                    size_t count) {
  for (size_t i = 0; i < count; ++i) {
    const double v = src[i];
    if (!isfinite(v) || v > (double)FLT_MAX || v < -(double)FLT_MAX)
      return 0;
    const float f = (float)v;
    if (!isfinite(f))
      return 0;
    dst[i] = f;
  }
  return 1;
}

static et_ring_attn_status_t parse_config(const void *cfg_ptr,
                                          et_ring_attn_config_v1 *cfg_out,
                                          double *scale_out) {
  if (!aligned_span(cfg_ptr, _Alignof(et_ring_attn_config_v1),
                    sizeof(et_ring_attn_config_v1)))
    return ET_RING_ATTN_ERR_INVALID_ARG;
  et_ring_attn_config_v1 cfg;
  memcpy(&cfg, cfg_ptr, sizeof(cfg));
  if (cfg.struct_size != sizeof(et_ring_attn_config_v1))
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (!valid_world_size(cfg.world_size))
    return ET_RING_ATTN_ERR_UNSUPPORTED;
  if (cfg.causal > 1u)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (cfg.timeout_ms <= 0 || cfg.timeout_ms > ET_RING_ATTN_MAX_TIMEOUT_MS)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (cfg.batch < ET_RING_ATTN_MIN_BATCH ||
      cfg.batch > ET_RING_ATTN_MAX_BATCH ||
      cfg.seq_len < ET_RING_ATTN_MIN_SEQ_LEN ||
      cfg.seq_len > ET_RING_ATTN_MAX_SEQ_LEN ||
      cfg.num_heads < ET_RING_ATTN_MIN_HEADS ||
      cfg.num_heads > ET_RING_ATTN_MAX_HEADS ||
      cfg.num_kv_heads < ET_RING_ATTN_MIN_HEADS ||
      cfg.num_kv_heads > cfg.num_heads ||
      cfg.head_dim < ET_RING_ATTN_MIN_HEAD_DIM ||
      cfg.head_dim > ET_RING_ATTN_MAX_HEAD_DIM)
    return ET_RING_ATTN_ERR_SHAPE_MISMATCH;
  if ((cfg.seq_len % (size_t)cfg.world_size) != 0u)
    return ET_RING_ATTN_ERR_SHAPE_MISMATCH;
  if ((cfg.num_heads % cfg.num_kv_heads) != 0u)
    return ET_RING_ATTN_ERR_SHAPE_MISMATCH;
  if (!isfinite(cfg.scale) || cfg.scale < 0.0)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  const double eff_scale =
      (cfg.scale == 0.0) ? (1.0 / sqrt((double)cfg.head_dim)) : cfg.scale;
  if (!isfinite(eff_scale) || eff_scale <= 0.0)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (cfg_out != NULL)
    *cfg_out = cfg;
  if (scale_out != NULL)
    *scale_out = eff_scale;
  return ET_RING_ATTN_OK;
}

et_ring_attn_status_t et_ring_attn_validate_config_v1(
    const void *cfg_ptr, double *effective_scale_out) {
  return parse_config(cfg_ptr, NULL, effective_scale_out);
}

static et_ring_attn_status_t check_unix_stream_socket(int fd,
                                                      struct stat *st_out) {
  struct sockaddr_storage local, peer;
  socklen_t len = sizeof(local);
  int sock_type = 0;
  if (fd < 0 || fstat(fd, st_out) != 0)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (getsockname(fd, (struct sockaddr *)&local, &len) != 0 ||
      local.ss_family != AF_UNIX)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  len = sizeof(peer);
  if (getpeername(fd, (struct sockaddr *)&peer, &len) != 0 ||
      peer.ss_family != AF_UNIX)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  len = sizeof(sock_type);
  if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &sock_type, &len) != 0 ||
      sock_type != SOCK_STREAM)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  return ET_RING_ATTN_OK;
}

et_ring_attn_status_t et_ring_attn_comm_create_v1(
    uint32_t abi_major, uint32_t abi_minor, uint32_t rank, uint32_t world_size,
    int send_fd, int recv_fd, et_ring_attn_comm_t **out) {
  if (out == NULL)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  *out = NULL;
  if (abi_major != ET_RING_ATTN_ABI_MAJOR ||
      abi_minor > ET_RING_ATTN_ABI_MINOR)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (!valid_world_size(world_size))
    return ET_RING_ATTN_ERR_UNSUPPORTED;
  if (rank >= world_size)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (world_size == 1u) {
    if (send_fd != -1 || recv_fd != -1)
      return ET_RING_ATTN_ERR_INVALID_ARG;
  } else {
    if (send_fd < 0 || recv_fd < 0)
      return ET_RING_ATTN_ERR_INVALID_ARG;
  }

  et_ring_attn_comm_t *comm = calloc(1u, sizeof(*comm));
  if (comm == NULL)
    return ET_RING_ATTN_ERR_ALLOC;
  comm->rank = rank;
  comm->world_size = world_size;
  comm->send_fd = -1;
  comm->recv_fd = -1;
  comm->fatal = ET_RING_ATTN_OK;

  if (world_size > 1u) {
    struct stat st_send, st_recv;
    et_ring_attn_status_t st = check_unix_stream_socket(send_fd, &st_send);
    if (st == ET_RING_ATTN_OK)
      st = check_unix_stream_socket(recv_fd, &st_recv);
    if (st == ET_RING_ATTN_OK && world_size > 2u &&
        st_send.st_dev == st_recv.st_dev && st_send.st_ino == st_recv.st_ino)
      st = ET_RING_ATTN_ERR_INVALID_ARG;
    if (st != ET_RING_ATTN_OK) {
      free(comm);
      return st;
    }

    int dup_send = -1;
    do {
      dup_send = fcntl(send_fd, F_DUPFD_CLOEXEC, 0);
    } while (dup_send < 0 && errno == EINTR);
    if (dup_send < 0) {
      const int err = errno;
      free(comm);
      return err == EBADF ? ET_RING_ATTN_ERR_INVALID_ARG : ET_RING_ATTN_ERR_IO;
    }
    comm->send_fd = dup_send;

    int dup_recv = -1;
    do {
      dup_recv = fcntl(recv_fd, F_DUPFD_CLOEXEC, 0);
    } while (dup_recv < 0 && errno == EINTR);
    if (dup_recv < 0) {
      const int err = errno;
      close(comm->send_fd);
      free(comm);
      return err == EBADF ? ET_RING_ATTN_ERR_INVALID_ARG : ET_RING_ATTN_ERR_IO;
    }
    comm->recv_fd = dup_recv;
  }

  *out = comm;
  return ET_RING_ATTN_OK;
}

void et_ring_attn_comm_destroy_v1(et_ring_attn_comm_t *comm) {
  if (comm == NULL)
    return;
  if (comm->send_fd >= 0)
    close(comm->send_fd);
  if (comm->recv_fd >= 0)
    close(comm->recv_fd);
  free(comm);
}

/*
 * Simultaneous non-blocking send on send_fd and recv on recv_fd using poll(),
 * bounded by a shared monotonic deadline in milliseconds.
 */
static et_ring_attn_status_t duplex_io_deadline(int send_fd,
                                                const unsigned char *sbuf,
                                                size_t slen, int recv_fd,
                                                unsigned char *rbuf,
                                                size_t rlen,
                                                int64_t deadline_ms) {
  size_t sent = 0u;
  size_t recvd = 0u;
  while (sent < slen || recvd < rlen) {
    struct pollfd pfds[2];
    nfds_t nfds = 0u;
    int s_idx = -1;
    int r_idx = -1;
    if (sent < slen) {
      s_idx = (int)nfds;
      pfds[nfds++] = (struct pollfd){send_fd, POLLOUT, 0};
    }
    if (recvd < rlen) {
      r_idx = (int)nfds;
      pfds[nfds++] = (struct pollfd){recv_fd, POLLIN | POLLRDHUP, 0};
    }

    const int64_t now = monotonic_ms();
    if (now < 0)
      return ET_RING_ATTN_ERR_IO;
    if (now >= deadline_ms)
      return ET_RING_ATTN_ERR_TIMEOUT;
    const int64_t rem = deadline_ms - now;
    const int wait_ms = rem > 60000 ? 60000 : (int)rem;
    const int ready = poll(pfds, nfds, wait_ms);
    if (ready < 0) {
      if (errno == EINTR)
        continue;
      return ET_RING_ATTN_ERR_IO;
    }
    if (ready == 0)
      continue;

    if (s_idx >= 0) {
      const short rev = pfds[s_idx].revents;
      if (rev & POLLNVAL)
        return ET_RING_ATTN_ERR_IO;
      if (rev & (POLLERR | POLLHUP))
        return ET_RING_ATTN_ERR_PEER_DISCONNECTED;
      if (rev & POLLOUT) {
        const ssize_t n =
            send(send_fd, sbuf + sent, slen - sent, MSG_DONTWAIT | MSG_NOSIGNAL);
        if (n > 0) {
          sent += (size_t)n;
        } else if (n == 0 ||
                   (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN)) {
          return ET_RING_ATTN_ERR_PEER_DISCONNECTED;
        } else if (!retryable_errno(errno)) {
          return ET_RING_ATTN_ERR_IO;
        }
      }
    }

    if (r_idx >= 0) {
      const short rev = pfds[r_idx].revents;
      if (rev & POLLNVAL)
        return ET_RING_ATTN_ERR_IO;
      if (rev & POLLIN) {
        const ssize_t n =
            recv(recv_fd, rbuf + recvd, rlen - recvd, MSG_DONTWAIT);
        if (n > 0) {
          recvd += (size_t)n;
        } else if (n == 0 ||
                   (errno == ECONNRESET || errno == EPIPE || errno == ENOTCONN)) {
          return ET_RING_ATTN_ERR_PEER_DISCONNECTED;
        } else if (!retryable_errno(errno)) {
          return ET_RING_ATTN_ERR_IO;
        }
      } else if (rev & (POLLERR | POLLHUP | POLLRDHUP)) {
        return ET_RING_ATTN_ERR_PEER_DISCONNECTED;
      }
    }
  }
  return ET_RING_ATTN_OK;
}

static et_ring_attn_status_t send_all_deadline(int fd,
                                               const unsigned char *buf,
                                               size_t len,
                                               int64_t deadline_ms) {
  return duplex_io_deadline(fd, buf, len, -1, NULL, 0u, deadline_ms);
}

static et_ring_attn_status_t recv_all_deadline(int fd, unsigned char *buf,
                                               size_t len,
                                               int64_t deadline_ms) {
  return duplex_io_deadline(-1, NULL, 0u, fd, buf, len, deadline_ms);
}

static et_ring_attn_status_t ring_exchange_deadline(
    et_ring_attn_comm_t *comm, const void *send_buf, void *recv_buf,
    size_t payload_bytes, uint32_t phase, uint32_t round,
    uint32_t send_kv_rank, uint32_t expected_recv_kv_rank,
    int64_t deadline_ms) {
  if (comm == NULL)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (comm->fatal != ET_RING_ATTN_OK)
    return comm->fatal;
  if (comm->world_size == 1u)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (payload_bytes == 0u || send_buf == NULL || recv_buf == NULL)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (spans_overlap(send_buf, payload_bytes, recv_buf, payload_bytes))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const ring_wire_header shdr = {
      RING_WIRE_MAGIC, phase, round, send_kv_rank, comm->rank,
      (uint64_t)payload_bytes};
  ring_wire_header rhdr;
  memset(&rhdr, 0, sizeof(rhdr));

  et_ring_attn_status_t st = duplex_io_deadline(
      comm->send_fd, (const unsigned char *)&shdr, sizeof(shdr), comm->recv_fd,
      (unsigned char *)&rhdr, sizeof(rhdr), deadline_ms);
  if (st != ET_RING_ATTN_OK) {
    comm->fatal = st;
    return st;
  }

  const uint32_t expected_sender =
      (comm->rank + comm->world_size - 1u) % comm->world_size;
  if (rhdr.magic != RING_WIRE_MAGIC || rhdr.phase != phase ||
      rhdr.round != round || rhdr.kv_rank != expected_recv_kv_rank ||
      rhdr.sender_rank != expected_sender ||
      rhdr.payload_bytes != (uint64_t)payload_bytes) {
    comm->fatal = ET_RING_ATTN_ERR_PROTOCOL;
    return ET_RING_ATTN_ERR_PROTOCOL;
  }

  st = duplex_io_deadline(comm->send_fd, (const unsigned char *)send_buf,
                          payload_bytes, comm->recv_fd,
                          (unsigned char *)recv_buf, payload_bytes,
                          deadline_ms);
  if (st != ET_RING_ATTN_OK) {
    comm->fatal = st;
    return st;
  }

  comm->total_bytes_sent += (uint64_t)payload_bytes;
  comm->total_bytes_received += (uint64_t)payload_bytes;
  return ET_RING_ATTN_OK;
}

et_ring_attn_status_t et_ring_attn_ring_exchange_v1(
    et_ring_attn_comm_t *comm, const void *send_buf, void *recv_buf,
    size_t payload_bytes, uint32_t phase, uint32_t round,
    uint32_t send_kv_rank, uint32_t expected_recv_kv_rank, int32_t timeout_ms) {
  if (timeout_ms <= 0 || timeout_ms > ET_RING_ATTN_MAX_TIMEOUT_MS)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  const int64_t now = monotonic_ms();
  if (now < 0)
    return ET_RING_ATTN_ERR_IO;
  return ring_exchange_deadline(comm, send_buf, recv_buf, payload_bytes, phase,
                                round, send_kv_rank, expected_recv_kv_rank,
                                now + (int64_t)timeout_ms);
}

/* ============================================================================
 * Online Softmax Block-Step Forward & Analytic Block-Step Backward (binary64)
 * ============================================================================ */

static void fwd_block_kernel(const et_ring_attn_config_v1 *cfg, double scale,
                             uint32_t q_rank, uint32_t kv_rank,
                             const float *q_shard, const float *k_block,
                             const float *v_block, double *m_state,
                             double *l_state, double *o_acc,
                             et_ring_attn_stats_v1 *stats) {
  const size_t N = cfg->batch;
  const size_t T_loc = cfg->seq_len / (size_t)cfg->world_size;
  const size_t H = cfg->num_heads;
  const size_t Hkv = cfg->num_kv_heads;
  const size_t Dh = cfg->head_dim;
  const size_t group = H / Hkv;

  if (stats != NULL)
    stats->total_block_steps += 1u;

  if (cfg->causal && kv_rank > q_rank) {
    if (stats != NULL)
      stats->skipped_causal_blocks += 1u;
    return;
  }

  if (stats != NULL)
    stats->executed_blocks += 1u;

  const int intra_causal = (cfg->causal && kv_rank == q_rank);

  for (size_t n = 0; n < N; ++n) {
    for (size_t qi = 0; qi < T_loc; ++qi) {
      const size_t j_end = intra_causal ? (qi + 1u) : T_loc;
      for (size_t h = 0; h < H; ++h) {
        const size_t hkv = h / group;
        const size_t q_idx = (n * T_loc + qi) * H + h;
        const float *q_vec = q_shard + q_idx * Dh;
        double *o_vec = o_acc + q_idx * Dh;

        double m_blk = -INFINITY;
        for (size_t kj = 0; kj < j_end; ++kj) {
          const float *k_vec =
              k_block + (((n * T_loc + kj) * Hkv + hkv) * Dh);
          double dot = 0.0;
          for (size_t d = 0; d < Dh; ++d)
            dot += (double)q_vec[d] * (double)k_vec[d];
          const double s = dot * scale;
          if (s > m_blk)
            m_blk = s;
        }

        const double m_old = m_state[q_idx];
        const double l_old = l_state[q_idx];
        const double m_new =
            (l_old == 0.0) ? m_blk : (m_old > m_blk ? m_old : m_blk);
        const double alpha = (l_old == 0.0) ? 0.0 : exp(m_old - m_new);

        for (size_t d = 0; d < Dh; ++d)
          o_vec[d] *= alpha;

        double l_blk = 0.0;
        for (size_t kj = 0; kj < j_end; ++kj) {
          const size_t kv_base = ((n * T_loc + kj) * Hkv + hkv) * Dh;
          const float *k_vec = k_block + kv_base;
          const float *v_vec = v_block + kv_base;
          double dot = 0.0;
          for (size_t d = 0; d < Dh; ++d)
            dot += (double)q_vec[d] * (double)k_vec[d];
          const double w = exp(dot * scale - m_new);
          l_blk += w;
          for (size_t d = 0; d < Dh; ++d)
            o_vec[d] += w * (double)v_vec[d];
        }

        m_state[q_idx] = m_new;
        l_state[q_idx] = l_old * alpha + l_blk;
      }
    }
  }
}

static void bwd_block_kernel(const et_ring_attn_config_v1 *cfg, double scale,
                             uint32_t q_rank, uint32_t kv_rank,
                             const float *q_shard, const float *do_shard,
                             const double *lse_shard, const double *d_vec,
                             const float *k_block, const float *v_block,
                             double *dq_acc, double *dk_acc, double *dv_acc,
                             et_ring_attn_stats_v1 *stats) {
  const size_t N = cfg->batch;
  const size_t T_loc = cfg->seq_len / (size_t)cfg->world_size;
  const size_t H = cfg->num_heads;
  const size_t Hkv = cfg->num_kv_heads;
  const size_t Dh = cfg->head_dim;
  const size_t group = H / Hkv;

  if (stats != NULL)
    stats->total_block_steps += 1u;

  if (cfg->causal && kv_rank > q_rank) {
    if (stats != NULL)
      stats->skipped_causal_blocks += 1u;
    return;
  }

  if (stats != NULL)
    stats->executed_blocks += 1u;

  const int intra_causal = (cfg->causal && kv_rank == q_rank);

  for (size_t n = 0; n < N; ++n) {
    for (size_t qi = 0; qi < T_loc; ++qi) {
      const size_t j_end = intra_causal ? (qi + 1u) : T_loc;
      for (size_t h = 0; h < H; ++h) {
        const size_t hkv = h / group;
        const size_t q_idx = (n * T_loc + qi) * H + h;
        const float *q_vec = q_shard + q_idx * Dh;
        const float *do_vec = do_shard + q_idx * Dh;
        double *dq_vec = dq_acc + q_idx * Dh;
        const double lse = lse_shard[q_idx];
        const double d_i = d_vec[q_idx];

        for (size_t kj = 0; kj < j_end; ++kj) {
          const size_t kv_base = ((n * T_loc + kj) * Hkv + hkv) * Dh;
          const float *k_vec = k_block + kv_base;
          const float *v_vec = v_block + kv_base;
          double *dk_vec = dk_acc + kv_base;
          double *dv_vec = dv_acc + kv_base;

          double dot_qk = 0.0;
          double dp = 0.0;
          for (size_t d = 0; d < Dh; ++d) {
            dot_qk += (double)q_vec[d] * (double)k_vec[d];
            dp += (double)do_vec[d] * (double)v_vec[d];
          }
          const double p = exp(dot_qk * scale - lse);
          const double ds = p * (dp - d_i);
          const double ddot = ds * scale;

          for (size_t d = 0; d < Dh; ++d) {
            dq_vec[d] += ddot * (double)k_vec[d];
            dk_vec[d] += ddot * (double)q_vec[d];
            dv_vec[d] += p * (double)do_vec[d];
          }
        }
      }
    }
  }
}

et_ring_attn_status_t et_ring_attn_fwd_block_step_v1(
    const void *cfg_ptr, uint32_t q_rank, uint32_t kv_rank,
    const void *q_shard, const void *k_block, const void *v_block,
    void *m_state, void *l_state, void *o_acc,
    et_ring_attn_stats_v1 *stats_accum) {
  et_ring_attn_config_v1 cfg;
  double scale = 0.0;
  et_ring_attn_status_t st = parse_config(cfg_ptr, &cfg, &scale);
  if (st != ET_RING_ATTN_OK)
    return st;
  if (q_rank >= cfg.world_size || kv_rank >= cfg.world_size)
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const size_t T_loc = cfg.seq_len / (size_t)cfg.world_size;
  const size_t q_elems = cfg.batch * T_loc * cfg.num_heads * cfg.head_dim;
  const size_t kv_elems = cfg.batch * T_loc * cfg.num_kv_heads * cfg.head_dim;
  const size_t lse_elems = cfg.batch * T_loc * cfg.num_heads;
  const size_t q_bytes = q_elems * sizeof(float);
  const size_t kv_bytes = kv_elems * sizeof(float);
  const size_t lse_bytes = lse_elems * sizeof(double);
  const size_t oacc_bytes = q_elems * sizeof(double);

  if (!aligned_span(q_shard, _Alignof(float), q_bytes) ||
      !aligned_span(k_block, _Alignof(float), kv_bytes) ||
      !aligned_span(v_block, _Alignof(float), kv_bytes) ||
      !aligned_span(m_state, _Alignof(double), lse_bytes) ||
      !aligned_span(l_state, _Alignof(double), lse_bytes) ||
      !aligned_span(o_acc, _Alignof(double), oacc_bytes))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const span_desc ins[] = {
      {q_shard, q_bytes}, {k_block, kv_bytes}, {v_block, kv_bytes}};
  const span_desc outs[] = {
      {m_state, lse_bytes}, {l_state, lse_bytes}, {o_acc, oacc_bytes}};
  if (!check_no_output_aliasing(ins, 3u, outs, 3u))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  if (!all_f32_finite((const float *)q_shard, q_elems) ||
      !all_f32_finite((const float *)k_block, kv_elems) ||
      !all_f32_finite((const float *)v_block, kv_elems))
    return ET_RING_ATTN_ERR_NONFINITE;

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0)
    return ET_RING_ATTN_ERR_IO;
  fesetround(FE_TONEAREST);

  fwd_block_kernel(&cfg, scale, q_rank, kv_rank, (const float *)q_shard,
                   (const float *)k_block, (const float *)v_block,
                   (double *)m_state, (double *)l_state, (double *)o_acc,
                   stats_accum);

  fesetenv(&saved_env);
  return ET_RING_ATTN_OK;
}

et_ring_attn_status_t et_ring_attn_fwd_finalize_v1(
    const void *cfg_ptr, const void *m_state, const void *l_state,
    const void *o_acc, void *o_shard, void *lse_shard) {
  et_ring_attn_config_v1 cfg;
  et_ring_attn_status_t st = parse_config(cfg_ptr, &cfg, NULL);
  if (st != ET_RING_ATTN_OK)
    return st;

  const size_t T_loc = cfg.seq_len / (size_t)cfg.world_size;
  const size_t q_elems = cfg.batch * T_loc * cfg.num_heads * cfg.head_dim;
  const size_t lse_elems = cfg.batch * T_loc * cfg.num_heads;
  const size_t o_bytes = q_elems * sizeof(float);
  const size_t lse_bytes = lse_elems * sizeof(double);
  const size_t oacc_bytes = q_elems * sizeof(double);

  if (!aligned_span(m_state, _Alignof(double), lse_bytes) ||
      !aligned_span(l_state, _Alignof(double), lse_bytes) ||
      !aligned_span(o_acc, _Alignof(double), oacc_bytes) ||
      !aligned_span(o_shard, _Alignof(float), o_bytes) ||
      !aligned_span(lse_shard, _Alignof(double), lse_bytes))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const span_desc ins[] = {
      {m_state, lse_bytes}, {l_state, lse_bytes}, {o_acc, oacc_bytes}};
  const span_desc outs[] = {{o_shard, o_bytes}, {lse_shard, lse_bytes}};
  if (!check_no_output_aliasing(ins, 3u, outs, 2u))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  float *tmp_o = malloc(o_bytes);
  double *tmp_lse = malloc(lse_bytes);
  if (tmp_o == NULL || tmp_lse == NULL) {
    free(tmp_o);
    free(tmp_lse);
    return ET_RING_ATTN_ERR_ALLOC;
  }

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0) {
    free(tmp_o);
    free(tmp_lse);
    return ET_RING_ATTN_ERR_IO;
  }
  fesetround(FE_TONEAREST);

  const double *ms = (const double *)m_state;
  const double *ls = (const double *)l_state;
  const double *oa = (const double *)o_acc;
  int ok = 1;
  for (size_t idx = 0; idx < lse_elems && ok; ++idx) {
    if (!isfinite(ms[idx]) || !isfinite(ls[idx]) || ls[idx] <= 0.0) {
      ok = 0;
      break;
    }
    const double lse_val = ms[idx] + log(ls[idx]);
    if (!isfinite(lse_val)) {
      ok = 0;
      break;
    }
    tmp_lse[idx] = lse_val;
    const double inv_l = 1.0 / ls[idx];
    for (size_t d = 0; d < cfg.head_dim; ++d) {
      const double val = oa[idx * cfg.head_dim + d] * inv_l;
      if (!isfinite(val) || val > (double)FLT_MAX || val < -(double)FLT_MAX) {
        ok = 0;
        break;
      }
      const float f = (float)val;
      if (!isfinite(f)) {
        ok = 0;
        break;
      }
      tmp_o[idx * cfg.head_dim + d] = f;
    }
  }

  fesetenv(&saved_env);

  if (!ok) {
    free(tmp_o);
    free(tmp_lse);
    return ET_RING_ATTN_ERR_NONFINITE;
  }

  memcpy(o_shard, tmp_o, o_bytes);
  memcpy(lse_shard, tmp_lse, lse_bytes);
  free(tmp_o);
  free(tmp_lse);
  return ET_RING_ATTN_OK;
}

et_ring_attn_status_t et_ring_attn_bwd_block_step_v1(
    const void *cfg_ptr, uint32_t q_rank, uint32_t kv_rank,
    const void *q_shard, const void *do_shard, const void *lse_shard,
    const void *d_vec, const void *k_block, const void *v_block,
    void *dq_acc, void *dk_acc, void *dv_acc,
    et_ring_attn_stats_v1 *stats_accum) {
  et_ring_attn_config_v1 cfg;
  double scale = 0.0;
  et_ring_attn_status_t st = parse_config(cfg_ptr, &cfg, &scale);
  if (st != ET_RING_ATTN_OK)
    return st;
  if (q_rank >= cfg.world_size || kv_rank >= cfg.world_size)
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const size_t T_loc = cfg.seq_len / (size_t)cfg.world_size;
  const size_t q_elems = cfg.batch * T_loc * cfg.num_heads * cfg.head_dim;
  const size_t kv_elems = cfg.batch * T_loc * cfg.num_kv_heads * cfg.head_dim;
  const size_t lse_elems = cfg.batch * T_loc * cfg.num_heads;
  const size_t q_bytes = q_elems * sizeof(float);
  const size_t kv_bytes = kv_elems * sizeof(float);
  const size_t lse_bytes = lse_elems * sizeof(double);
  const size_t dq_bytes = q_elems * sizeof(double);
  const size_t dkv_bytes = kv_elems * sizeof(double);

  if (!aligned_span(q_shard, _Alignof(float), q_bytes) ||
      !aligned_span(do_shard, _Alignof(float), q_bytes) ||
      !aligned_span(lse_shard, _Alignof(double), lse_bytes) ||
      !aligned_span(d_vec, _Alignof(double), lse_bytes) ||
      !aligned_span(k_block, _Alignof(float), kv_bytes) ||
      !aligned_span(v_block, _Alignof(float), kv_bytes) ||
      !aligned_span(dq_acc, _Alignof(double), dq_bytes) ||
      !aligned_span(dk_acc, _Alignof(double), dkv_bytes) ||
      !aligned_span(dv_acc, _Alignof(double), dkv_bytes))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const span_desc ins[] = {{q_shard, q_bytes},     {do_shard, q_bytes},
                           {lse_shard, lse_bytes}, {d_vec, lse_bytes},
                           {k_block, kv_bytes},    {v_block, kv_bytes}};
  const span_desc outs[] = {
      {dq_acc, dq_bytes}, {dk_acc, dkv_bytes}, {dv_acc, dkv_bytes}};
  if (!check_no_output_aliasing(ins, 6u, outs, 3u))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  if (!all_f32_finite((const float *)q_shard, q_elems) ||
      !all_f32_finite((const float *)do_shard, q_elems) ||
      !all_f64_finite((const double *)lse_shard, lse_elems) ||
      !all_f64_finite((const double *)d_vec, lse_elems) ||
      !all_f32_finite((const float *)k_block, kv_elems) ||
      !all_f32_finite((const float *)v_block, kv_elems))
    return ET_RING_ATTN_ERR_NONFINITE;

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0)
    return ET_RING_ATTN_ERR_IO;
  fesetround(FE_TONEAREST);

  bwd_block_kernel(&cfg, scale, q_rank, kv_rank, (const float *)q_shard,
                   (const float *)do_shard, (const double *)lse_shard,
                   (const double *)d_vec, (const float *)k_block,
                   (const float *)v_block, (double *)dq_acc, (double *)dk_acc,
                   (double *)dv_acc, stats_accum);

  fesetenv(&saved_env);
  return ET_RING_ATTN_OK;
}

/* ============================================================================
 * Per-Rank Distributed Ring Attention Forward & Backward over AF_UNIX Ring
 * ============================================================================ */

et_ring_attn_status_t et_ring_attn_forward_v1(
    et_ring_attn_comm_t *comm, const void *cfg_ptr, const void *q_shard,
    const void *k_shard, const void *v_shard, void *o_shard, void *lse_shard,
    et_ring_attn_stats_v1 *stats_out) {
  if (comm == NULL)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (comm->fatal != ET_RING_ATTN_OK)
    return comm->fatal;

  et_ring_attn_config_v1 cfg;
  double scale = 0.0;
  et_ring_attn_status_t st = parse_config(cfg_ptr, &cfg, &scale);
  if (st != ET_RING_ATTN_OK)
    return st;
  if (cfg.world_size != comm->world_size)
    return ET_RING_ATTN_ERR_SHAPE_MISMATCH;

  const size_t T_loc = cfg.seq_len / (size_t)cfg.world_size;
  const size_t q_elems = cfg.batch * T_loc * cfg.num_heads * cfg.head_dim;
  const size_t kv_elems = cfg.batch * T_loc * cfg.num_kv_heads * cfg.head_dim;
  const size_t lse_elems = cfg.batch * T_loc * cfg.num_heads;
  const size_t q_bytes = q_elems * sizeof(float);
  const size_t kv_bytes = kv_elems * sizeof(float);
  const size_t lse_bytes = lse_elems * sizeof(double);

  if (!aligned_span(q_shard, _Alignof(float), q_bytes) ||
      !aligned_span(k_shard, _Alignof(float), kv_bytes) ||
      !aligned_span(v_shard, _Alignof(float), kv_bytes) ||
      !aligned_span(o_shard, _Alignof(float), q_bytes) ||
      !aligned_span(lse_shard, _Alignof(double), lse_bytes))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const span_desc ins[] = {
      {q_shard, q_bytes}, {k_shard, kv_bytes}, {v_shard, kv_bytes}};
  const span_desc outs[] = {{o_shard, q_bytes}, {lse_shard, lse_bytes}};
  if (!check_no_output_aliasing(ins, 3u, outs, 2u))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  if (!all_f32_finite((const float *)q_shard, q_elems) ||
      !all_f32_finite((const float *)k_shard, kv_elems) ||
      !all_f32_finite((const float *)v_shard, kv_elems))
    return ET_RING_ATTN_ERR_NONFINITE;

  const int64_t start_ms = monotonic_ms();
  if (start_ms < 0)
    return ET_RING_ATTN_ERR_IO;
  const int64_t deadline_ms = start_ms + (int64_t)cfg.timeout_ms;

  const size_t pkt_bytes = 2u * kv_bytes;
  double *m_state = malloc(lse_bytes);
  double *l_state = calloc(lse_elems, sizeof(double));
  double *o_acc = calloc(q_elems, sizeof(double));
  unsigned char *cur_kv = malloc(pkt_bytes);
  unsigned char *next_kv = (cfg.world_size > 1u) ? malloc(pkt_bytes) : NULL;
  if (m_state == NULL || l_state == NULL || o_acc == NULL || cur_kv == NULL ||
      (cfg.world_size > 1u && next_kv == NULL)) {
    free(m_state);
    free(l_state);
    free(o_acc);
    free(cur_kv);
    free(next_kv);
    return ET_RING_ATTN_ERR_ALLOC;
  }

  for (size_t i = 0; i < lse_elems; ++i)
    m_state[i] = -INFINITY;
  memcpy(cur_kv, k_shard, kv_bytes);
  memcpy(cur_kv + kv_bytes, v_shard, kv_bytes);

  et_ring_attn_stats_v1 stats;
  memset(&stats, 0, sizeof(stats));
  const uint64_t sent_before = comm->total_bytes_sent;
  const uint64_t recv_before = comm->total_bytes_received;

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0) {
    free(m_state);
    free(l_state);
    free(o_acc);
    free(cur_kv);
    free(next_kv);
    return ET_RING_ATTN_ERR_IO;
  }
  fesetround(FE_TONEAREST);

  uint32_t cur_kv_rank = comm->rank;
  st = ET_RING_ATTN_OK;
  for (uint32_t step = 0u; step < cfg.world_size; ++step) {
    const float *k_cur = (const float *)cur_kv;
    const float *v_cur = (const float *)(cur_kv + kv_bytes);
    if (!all_f32_finite(k_cur, kv_elems) || !all_f32_finite(v_cur, kv_elems)) {
      st = ET_RING_ATTN_ERR_NONFINITE;
      break;
    }

    fwd_block_kernel(&cfg, scale, comm->rank, cur_kv_rank,
                     (const float *)q_shard, k_cur, v_cur, m_state, l_state,
                     o_acc, &stats);

    if (cfg.world_size > 1u && step + 1u < cfg.world_size) {
      const uint32_t expect_kv_rank =
          (comm->rank + cfg.world_size - (step + 1u)) % cfg.world_size;
      st = ring_exchange_deadline(comm, cur_kv, next_kv, pkt_bytes,
                                  ET_RING_ATTN_PHASE_FORWARD, step,
                                  cur_kv_rank, expect_kv_rank, deadline_ms);
      if (st != ET_RING_ATTN_OK)
        break;
      unsigned char *tmp = cur_kv;
      cur_kv = next_kv;
      next_kv = tmp;
      cur_kv_rank = expect_kv_rank;
      stats.ring_rounds += 1u;
    }
  }

  fesetenv(&saved_env);

  if (st == ET_RING_ATTN_OK) {
    st = et_ring_attn_fwd_finalize_v1(&cfg, m_state, l_state, o_acc, o_shard,
                                      lse_shard);
  }
  if (st == ET_RING_ATTN_OK && stats_out != NULL) {
    stats.bytes_sent = comm->total_bytes_sent - sent_before;
    stats.bytes_received = comm->total_bytes_received - recv_before;
    *stats_out = stats;
  }

  free(m_state);
  free(l_state);
  free(o_acc);
  free(cur_kv);
  free(next_kv);
  return st;
}

et_ring_attn_status_t et_ring_attn_backward_v1(
    et_ring_attn_comm_t *comm, const void *cfg_ptr, const void *q_shard,
    const void *k_shard, const void *v_shard, const void *o_shard,
    const void *lse_shard, const void *do_shard, void *dq_shard, void *dk_shard,
    void *dv_shard, et_ring_attn_stats_v1 *stats_out) {
  if (comm == NULL)
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (comm->fatal != ET_RING_ATTN_OK)
    return comm->fatal;

  et_ring_attn_config_v1 cfg;
  double scale = 0.0;
  et_ring_attn_status_t st = parse_config(cfg_ptr, &cfg, &scale);
  if (st != ET_RING_ATTN_OK)
    return st;
  if (cfg.world_size != comm->world_size)
    return ET_RING_ATTN_ERR_SHAPE_MISMATCH;

  const size_t T_loc = cfg.seq_len / (size_t)cfg.world_size;
  const size_t q_elems = cfg.batch * T_loc * cfg.num_heads * cfg.head_dim;
  const size_t kv_elems = cfg.batch * T_loc * cfg.num_kv_heads * cfg.head_dim;
  const size_t lse_elems = cfg.batch * T_loc * cfg.num_heads;
  const size_t q_bytes = q_elems * sizeof(float);
  const size_t kv_bytes = kv_elems * sizeof(float);
  const size_t lse_bytes = lse_elems * sizeof(double);
  const size_t dkv_f64_bytes = kv_elems * sizeof(double);

  if (!aligned_span(q_shard, _Alignof(float), q_bytes) ||
      !aligned_span(k_shard, _Alignof(float), kv_bytes) ||
      !aligned_span(v_shard, _Alignof(float), kv_bytes) ||
      !aligned_span(o_shard, _Alignof(float), q_bytes) ||
      !aligned_span(lse_shard, _Alignof(double), lse_bytes) ||
      !aligned_span(do_shard, _Alignof(float), q_bytes) ||
      !aligned_span(dq_shard, _Alignof(float), q_bytes) ||
      !aligned_span(dk_shard, _Alignof(float), kv_bytes) ||
      !aligned_span(dv_shard, _Alignof(float), kv_bytes))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const span_desc ins[] = {{q_shard, q_bytes},     {k_shard, kv_bytes},
                           {v_shard, kv_bytes},    {o_shard, q_bytes},
                           {lse_shard, lse_bytes}, {do_shard, q_bytes}};
  const span_desc outs[] = {
      {dq_shard, q_bytes}, {dk_shard, kv_bytes}, {dv_shard, kv_bytes}};
  if (!check_no_output_aliasing(ins, 6u, outs, 3u))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  if (!all_f32_finite((const float *)q_shard, q_elems) ||
      !all_f32_finite((const float *)k_shard, kv_elems) ||
      !all_f32_finite((const float *)v_shard, kv_elems) ||
      !all_f32_finite((const float *)o_shard, q_elems) ||
      !all_f64_finite((const double *)lse_shard, lse_elems) ||
      !all_f32_finite((const float *)do_shard, q_elems))
    return ET_RING_ATTN_ERR_NONFINITE;

  const int64_t start_ms = monotonic_ms();
  if (start_ms < 0)
    return ET_RING_ATTN_ERR_IO;
  const int64_t deadline_ms = start_ms + (int64_t)cfg.timeout_ms;

  /* Circulate packed [dK_f64, dV_f64, K_f32, V_f32] so doubles stay 8-byte aligned. */
  const size_t full_pkt_bytes = 2u * dkv_f64_bytes + 2u * kv_bytes;
  const size_t final_pkt_bytes = 2u * dkv_f64_bytes;

  double *d_vec = malloc(lse_bytes);
  double *dq_acc = calloc(q_elems, sizeof(double));
  unsigned char *cur_pkt = calloc(1u, full_pkt_bytes);
  unsigned char *next_pkt =
      (cfg.world_size > 1u) ? calloc(1u, full_pkt_bytes) : NULL;
  float *tmp_dq = malloc(q_bytes);
  float *tmp_dk = malloc(kv_bytes);
  float *tmp_dv = malloc(kv_bytes);

  if (d_vec == NULL || dq_acc == NULL || cur_pkt == NULL ||
      (cfg.world_size > 1u && next_pkt == NULL) || tmp_dq == NULL ||
      tmp_dk == NULL || tmp_dv == NULL) {
    free(d_vec);
    free(dq_acc);
    free(cur_pkt);
    free(next_pkt);
    free(tmp_dq);
    free(tmp_dk);
    free(tmp_dv);
    return ET_RING_ATTN_ERR_ALLOC;
  }

  /* Populate initial circulating packet with K_r, V_r and zeroed dK, dV. */
  memcpy(cur_pkt + 2u * dkv_f64_bytes, k_shard, kv_bytes);
  memcpy(cur_pkt + 2u * dkv_f64_bytes + kv_bytes, v_shard, kv_bytes);

  et_ring_attn_stats_v1 stats;
  memset(&stats, 0, sizeof(stats));
  const uint64_t sent_before = comm->total_bytes_sent;
  const uint64_t recv_before = comm->total_bytes_received;

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0) {
    free(d_vec);
    free(dq_acc);
    free(cur_pkt);
    free(next_pkt);
    free(tmp_dq);
    free(tmp_dk);
    free(tmp_dv);
    return ET_RING_ATTN_ERR_IO;
  }
  fesetround(FE_TONEAREST);

  /* Precompute D_i = dot(dO_i, O_i) in binary64. */
  const float *o_f32 = (const float *)o_shard;
  const float *do_f32 = (const float *)do_shard;
  for (size_t idx = 0; idx < lse_elems; ++idx) {
    double acc = 0.0;
    for (size_t d = 0; d < cfg.head_dim; ++d)
      acc += (double)do_f32[idx * cfg.head_dim + d] *
             (double)o_f32[idx * cfg.head_dim + d];
    d_vec[idx] = acc;
  }

  uint32_t cur_kv_rank = comm->rank;
  st = ET_RING_ATTN_OK;
  for (uint32_t step = 0u; step < cfg.world_size; ++step) {
    double *dk_cur = (double *)cur_pkt;
    double *dv_cur = (double *)(cur_pkt + dkv_f64_bytes);
    const float *k_cur = (const float *)(cur_pkt + 2u * dkv_f64_bytes);
    const float *v_cur =
        (const float *)(cur_pkt + 2u * dkv_f64_bytes + kv_bytes);

    if (!all_f64_finite(dk_cur, kv_elems) ||
        !all_f64_finite(dv_cur, kv_elems) ||
        !all_f32_finite(k_cur, kv_elems) || !all_f32_finite(v_cur, kv_elems)) {
      st = ET_RING_ATTN_ERR_NONFINITE;
      break;
    }

    bwd_block_kernel(&cfg, scale, comm->rank, cur_kv_rank,
                     (const float *)q_shard, do_f32, (const double *)lse_shard,
                     d_vec, k_cur, v_cur, dq_acc, dk_cur, dv_cur, &stats);

    if (cfg.world_size > 1u) {
      const int is_final = (step + 1u == cfg.world_size);
      const uint32_t expect_kv_rank =
          (comm->rank + cfg.world_size - ((step + 1u) % cfg.world_size)) %
          cfg.world_size;
      const size_t xfer_bytes = is_final ? final_pkt_bytes : full_pkt_bytes;
      const uint32_t phase = is_final ? ET_RING_ATTN_PHASE_BACKWARD_FINAL
                                      : ET_RING_ATTN_PHASE_BACKWARD;
      st = ring_exchange_deadline(comm, cur_pkt, next_pkt, xfer_bytes, phase,
                                  step, cur_kv_rank, expect_kv_rank,
                                  deadline_ms);
      if (st != ET_RING_ATTN_OK)
        break;
      unsigned char *tmp = cur_pkt;
      cur_pkt = next_pkt;
      next_pkt = tmp;
      cur_kv_rank = expect_kv_rank;
      stats.ring_rounds += 1u;
    }
  }

  if (st == ET_RING_ATTN_OK) {
    if (cur_kv_rank != comm->rank) {
      st = ET_RING_ATTN_ERR_PROTOCOL;
    } else {
      const double *final_dk = (const double *)cur_pkt;
      const double *final_dv = (const double *)(cur_pkt + dkv_f64_bytes);
      if (!round_f64_to_f32_checked(dq_acc, tmp_dq, q_elems) ||
          !round_f64_to_f32_checked(final_dk, tmp_dk, kv_elems) ||
          !round_f64_to_f32_checked(final_dv, tmp_dv, kv_elems)) {
        st = ET_RING_ATTN_ERR_NONFINITE;
      }
    }
  }

  fesetenv(&saved_env);

  if (st == ET_RING_ATTN_OK) {
    memcpy(dq_shard, tmp_dq, q_bytes);
    memcpy(dk_shard, tmp_dk, kv_bytes);
    memcpy(dv_shard, tmp_dv, kv_bytes);
    if (stats_out != NULL) {
      stats.bytes_sent = comm->total_bytes_sent - sent_before;
      stats.bytes_received = comm->total_bytes_received - recv_before;
      *stats_out = stats;
    }
  }

  free(d_vec);
  free(dq_acc);
  free(cur_pkt);
  free(next_pkt);
  free(tmp_dq);
  free(tmp_dk);
  free(tmp_dv);
  return st;
}

/* ============================================================================
 * Shard Scatter / Gather Helpers for Full [N, T, ...] <-> Local [N, T_loc, ...]
 * ============================================================================ */

static void extract_shard_bytes(const unsigned char *full, unsigned char *shard,
                                size_t batch, size_t seq_len, size_t t_local,
                                uint32_t rank, size_t token_stride_bytes) {
  const size_t full_batch_bytes = seq_len * token_stride_bytes;
  const size_t loc_batch_bytes = t_local * token_stride_bytes;
  const size_t rank_offset_bytes = (size_t)rank * loc_batch_bytes;
  for (size_t n = 0; n < batch; ++n) {
    memcpy(shard + n * loc_batch_bytes,
           full + n * full_batch_bytes + rank_offset_bytes, loc_batch_bytes);
  }
}

static void insert_shard_bytes(unsigned char *full, const unsigned char *shard,
                               size_t batch, size_t seq_len, size_t t_local,
                               uint32_t rank, size_t token_stride_bytes) {
  const size_t full_batch_bytes = seq_len * token_stride_bytes;
  const size_t loc_batch_bytes = t_local * token_stride_bytes;
  const size_t rank_offset_bytes = (size_t)rank * loc_batch_bytes;
  for (size_t n = 0; n < batch; ++n) {
    memcpy(full + n * full_batch_bytes + rank_offset_bytes,
           shard + n * loc_batch_bytes, loc_batch_bytes);
  }
}

/* ============================================================================
 * Multi-Process Fork + AF_UNIX Ring Orchestration (Zero Shared Memory)
 * ============================================================================ */

static void close_fd_array(int fds[][2], uint32_t count) {
  for (uint32_t i = 0u; i < count; ++i) {
    if (fds[i][0] >= 0) {
      close(fds[i][0]);
      fds[i][0] = -1;
    }
    if (fds[i][1] >= 0) {
      close(fds[i][1]);
      fds[i][1] = -1;
    }
  }
}

static void reap_children(pid_t pids[ET_RING_ATTN_MAX_WORLD_SIZE],
                          uint32_t count, int force_kill) {
  if (force_kill) {
    for (uint32_t r = 0u; r < count; ++r) {
      if (pids[r] > 0)
        kill(pids[r], SIGKILL);
    }
  }
  for (uint32_t r = 0u; r < count; ++r) {
    if (pids[r] > 0) {
      int wstatus = 0;
      while (waitpid(pids[r], &wstatus, 0) < 0 && errno == EINTR) {
      }
      pids[r] = -1;
    }
  }
}

et_ring_attn_status_t et_ring_attn_multiprocess_forward_v1(
    const void *cfg_ptr, const void *q_full, const void *k_full,
    const void *v_full, void *o_full, void *lse_full,
    et_ring_attn_stats_v1 *agg_stats_out) {
  et_ring_attn_config_v1 cfg;
  et_ring_attn_status_t st = parse_config(cfg_ptr, &cfg, NULL);
  if (st != ET_RING_ATTN_OK)
    return st;

  const size_t q_elems = cfg.batch * cfg.seq_len * cfg.num_heads * cfg.head_dim;
  const size_t kv_elems =
      cfg.batch * cfg.seq_len * cfg.num_kv_heads * cfg.head_dim;
  const size_t lse_elems = cfg.batch * cfg.seq_len * cfg.num_heads;
  const size_t q_bytes = q_elems * sizeof(float);
  const size_t kv_bytes = kv_elems * sizeof(float);
  const size_t lse_bytes = lse_elems * sizeof(double);

  if (!aligned_span(q_full, _Alignof(float), q_bytes) ||
      !aligned_span(k_full, _Alignof(float), kv_bytes) ||
      !aligned_span(v_full, _Alignof(float), kv_bytes) ||
      !aligned_span(o_full, _Alignof(float), q_bytes) ||
      !aligned_span(lse_full, _Alignof(double), lse_bytes))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const span_desc ins[] = {
      {q_full, q_bytes}, {k_full, kv_bytes}, {v_full, kv_bytes}};
  const span_desc outs[] = {{o_full, q_bytes}, {lse_full, lse_bytes}};
  if (!check_no_output_aliasing(ins, 3u, outs, 2u))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  if (!all_f32_finite((const float *)q_full, q_elems) ||
      !all_f32_finite((const float *)k_full, kv_elems) ||
      !all_f32_finite((const float *)v_full, kv_elems))
    return ET_RING_ATTN_ERR_NONFINITE;

  if (cfg.world_size == 1u) {
    et_ring_attn_comm_t *comm = NULL;
    st = et_ring_attn_comm_create_v1(ET_RING_ATTN_ABI_MAJOR,
                                     ET_RING_ATTN_ABI_MINOR, 0u, 1u, -1, -1,
                                     &comm);
    if (st != ET_RING_ATTN_OK)
      return st;
    st = et_ring_attn_forward_v1(comm, &cfg, q_full, k_full, v_full, o_full,
                                 lse_full, agg_stats_out);
    et_ring_attn_comm_destroy_v1(comm);
    return st;
  }

  const uint32_t W = cfg.world_size;
  const size_t T_loc = cfg.seq_len / (size_t)W;
  const size_t q_loc_bytes =
      cfg.batch * T_loc * cfg.num_heads * cfg.head_dim * sizeof(float);
  const size_t kv_loc_bytes =
      cfg.batch * T_loc * cfg.num_kv_heads * cfg.head_dim * sizeof(float);
  const size_t lse_loc_bytes =
      cfg.batch * T_loc * cfg.num_heads * sizeof(double);

  int edge_fds[ET_RING_ATTN_MAX_WORLD_SIZE][2];
  int res_fds[ET_RING_ATTN_MAX_WORLD_SIZE][2];
  pid_t pids[ET_RING_ATTN_MAX_WORLD_SIZE];
  for (uint32_t r = 0u; r < ET_RING_ATTN_MAX_WORLD_SIZE; ++r) {
    edge_fds[r][0] = edge_fds[r][1] = -1;
    res_fds[r][0] = res_fds[r][1] = -1;
    pids[r] = -1;
  }

  for (uint32_t r = 0u; r < W; ++r) {
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, edge_fds[r]) != 0 ||
        socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, res_fds[r]) != 0) {
      close_fd_array(edge_fds, W);
      close_fd_array(res_fds, W);
      return ET_RING_ATTN_ERR_IO;
    }
  }

  const int64_t start_ms = monotonic_ms();
  if (start_ms < 0) {
    close_fd_array(edge_fds, W);
    close_fd_array(res_fds, W);
    return ET_RING_ATTN_ERR_IO;
  }
  const int64_t deadline_ms = start_ms + (int64_t)cfg.timeout_ms;

  for (uint32_t r = 0u; r < W; ++r) {
    const pid_t pid = fork();
    if (pid < 0) {
      close_fd_array(edge_fds, W);
      close_fd_array(res_fds, W);
      reap_children(pids, W, 1);
      return ET_RING_ATTN_ERR_IO;
    }
    if (pid == 0) {
      /* Child rank r: keep only send_fd, recv_fd, and parent reply fd. */
      const uint32_t prev_rank = (r + W - 1u) % W;
      const int send_fd = edge_fds[r][0];
      const int recv_fd = edge_fds[prev_rank][1];
      const int reply_fd = res_fds[r][1];

      for (uint32_t i = 0u; i < W; ++i) {
        if (edge_fds[i][0] != send_fd && edge_fds[i][0] >= 0)
          close(edge_fds[i][0]);
        if (edge_fds[i][1] != recv_fd && edge_fds[i][1] >= 0)
          close(edge_fds[i][1]);
        if (res_fds[i][0] >= 0)
          close(res_fds[i][0]);
        if (res_fds[i][1] != reply_fd && res_fds[i][1] >= 0)
          close(res_fds[i][1]);
      }

      unsigned char *q_loc = malloc(q_loc_bytes);
      unsigned char *k_loc = malloc(kv_loc_bytes);
      unsigned char *v_loc = malloc(kv_loc_bytes);
      unsigned char *o_loc = malloc(q_loc_bytes);
      unsigned char *lse_loc = malloc(lse_loc_bytes);
      child_reply_header rep;
      memset(&rep, 0, sizeof(rep));

      if (q_loc == NULL || k_loc == NULL || v_loc == NULL || o_loc == NULL ||
          lse_loc == NULL) {
        rep.status = ET_RING_ATTN_ERR_ALLOC;
      } else {
        extract_shard_bytes((const unsigned char *)q_full, q_loc, cfg.batch,
                            cfg.seq_len, T_loc, r,
                            cfg.num_heads * cfg.head_dim * sizeof(float));
        extract_shard_bytes((const unsigned char *)k_full, k_loc, cfg.batch,
                            cfg.seq_len, T_loc, r,
                            cfg.num_kv_heads * cfg.head_dim * sizeof(float));
        extract_shard_bytes((const unsigned char *)v_full, v_loc, cfg.batch,
                            cfg.seq_len, T_loc, r,
                            cfg.num_kv_heads * cfg.head_dim * sizeof(float));

        et_ring_attn_comm_t *comm = NULL;
        rep.status = et_ring_attn_comm_create_v1(ET_RING_ATTN_ABI_MAJOR,
                                                 ET_RING_ATTN_ABI_MINOR, r, W,
                                                 send_fd, recv_fd, &comm);
        close(send_fd);
        close(recv_fd);
        if (rep.status == ET_RING_ATTN_OK) {
          rep.status = et_ring_attn_forward_v1(comm, &cfg, q_loc, k_loc, v_loc,
                                               o_loc, lse_loc, &rep.stats);
          et_ring_attn_comm_destroy_v1(comm);
        }
      }

      const int64_t child_deadline = deadline_ms + 5000;
      if (send_all_deadline(reply_fd, (const unsigned char *)&rep, sizeof(rep),
                            child_deadline) == ET_RING_ATTN_OK &&
          rep.status == ET_RING_ATTN_OK) {
        (void)send_all_deadline(reply_fd, o_loc, q_loc_bytes, child_deadline);
        (void)send_all_deadline(reply_fd, lse_loc, lse_loc_bytes,
                                child_deadline);
      }

      close(reply_fd);
      free(q_loc);
      free(k_loc);
      free(v_loc);
      free(o_loc);
      free(lse_loc);
      _exit(0);
    }
    pids[r] = pid;
  }

  /* Parent closes all ring edges and child reply ends immediately. */
  close_fd_array(edge_fds, W);
  for (uint32_t r = 0u; r < W; ++r) {
    close(res_fds[r][1]);
    res_fds[r][1] = -1;
  }

  unsigned char *stage_o = malloc(q_bytes);
  unsigned char *stage_lse = malloc(lse_bytes);
  unsigned char *rx_o = malloc(q_loc_bytes);
  unsigned char *rx_lse = malloc(lse_loc_bytes);
  et_ring_attn_stats_v1 agg;
  memset(&agg, 0, sizeof(agg));

  if (stage_o == NULL || stage_lse == NULL || rx_o == NULL || rx_lse == NULL) {
    st = ET_RING_ATTN_ERR_ALLOC;
  } else {
    st = ET_RING_ATTN_OK;
    for (uint32_t r = 0u; r < W; ++r) {
      child_reply_header rep;
      memset(&rep, 0, sizeof(rep));
      et_ring_attn_status_t rst = recv_all_deadline(
          res_fds[r][0], (unsigned char *)&rep, sizeof(rep), deadline_ms);
      if (rst != ET_RING_ATTN_OK) {
        st = rst;
        break;
      }
      if (rep.status != ET_RING_ATTN_OK) {
        st = rep.status;
        break;
      }
      rst = recv_all_deadline(res_fds[r][0], rx_o, q_loc_bytes, deadline_ms);
      if (rst != ET_RING_ATTN_OK) {
        st = rst;
        break;
      }
      rst =
          recv_all_deadline(res_fds[r][0], rx_lse, lse_loc_bytes, deadline_ms);
      if (rst != ET_RING_ATTN_OK) {
        st = rst;
        break;
      }
      insert_shard_bytes(stage_o, rx_o, cfg.batch, cfg.seq_len, T_loc, r,
                         cfg.num_heads * cfg.head_dim * sizeof(float));
      insert_shard_bytes(stage_lse, rx_lse, cfg.batch, cfg.seq_len, T_loc, r,
                         cfg.num_heads * sizeof(double));
      agg.total_block_steps += rep.stats.total_block_steps;
      agg.executed_blocks += rep.stats.executed_blocks;
      agg.skipped_causal_blocks += rep.stats.skipped_causal_blocks;
      agg.bytes_sent += rep.stats.bytes_sent;
      agg.bytes_received += rep.stats.bytes_received;
      agg.ring_rounds = rep.stats.ring_rounds;
    }
  }

  close_fd_array(res_fds, W);
  reap_children(pids, W, st != ET_RING_ATTN_OK);

  if (st == ET_RING_ATTN_OK) {
    memcpy(o_full, stage_o, q_bytes);
    memcpy(lse_full, stage_lse, lse_bytes);
    if (agg_stats_out != NULL)
      *agg_stats_out = agg;
  }

  free(stage_o);
  free(stage_lse);
  free(rx_o);
  free(rx_lse);
  return st;
}

et_ring_attn_status_t et_ring_attn_multiprocess_backward_v1(
    const void *cfg_ptr, const void *q_full, const void *k_full,
    const void *v_full, const void *o_full, const void *lse_full,
    const void *do_full, void *dq_full, void *dk_full, void *dv_full,
    et_ring_attn_stats_v1 *agg_stats_out) {
  et_ring_attn_config_v1 cfg;
  et_ring_attn_status_t st = parse_config(cfg_ptr, &cfg, NULL);
  if (st != ET_RING_ATTN_OK)
    return st;

  const size_t q_elems = cfg.batch * cfg.seq_len * cfg.num_heads * cfg.head_dim;
  const size_t kv_elems =
      cfg.batch * cfg.seq_len * cfg.num_kv_heads * cfg.head_dim;
  const size_t lse_elems = cfg.batch * cfg.seq_len * cfg.num_heads;
  const size_t q_bytes = q_elems * sizeof(float);
  const size_t kv_bytes = kv_elems * sizeof(float);
  const size_t lse_bytes = lse_elems * sizeof(double);

  if (!aligned_span(q_full, _Alignof(float), q_bytes) ||
      !aligned_span(k_full, _Alignof(float), kv_bytes) ||
      !aligned_span(v_full, _Alignof(float), kv_bytes) ||
      !aligned_span(o_full, _Alignof(float), q_bytes) ||
      !aligned_span(lse_full, _Alignof(double), lse_bytes) ||
      !aligned_span(do_full, _Alignof(float), q_bytes) ||
      !aligned_span(dq_full, _Alignof(float), q_bytes) ||
      !aligned_span(dk_full, _Alignof(float), kv_bytes) ||
      !aligned_span(dv_full, _Alignof(float), kv_bytes))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const span_desc ins[] = {{q_full, q_bytes},     {k_full, kv_bytes},
                           {v_full, kv_bytes},    {o_full, q_bytes},
                           {lse_full, lse_bytes}, {do_full, q_bytes}};
  const span_desc outs[] = {
      {dq_full, q_bytes}, {dk_full, kv_bytes}, {dv_full, kv_bytes}};
  if (!check_no_output_aliasing(ins, 6u, outs, 3u))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  if (!all_f32_finite((const float *)q_full, q_elems) ||
      !all_f32_finite((const float *)k_full, kv_elems) ||
      !all_f32_finite((const float *)v_full, kv_elems) ||
      !all_f32_finite((const float *)o_full, q_elems) ||
      !all_f64_finite((const double *)lse_full, lse_elems) ||
      !all_f32_finite((const float *)do_full, q_elems))
    return ET_RING_ATTN_ERR_NONFINITE;

  if (cfg.world_size == 1u) {
    et_ring_attn_comm_t *comm = NULL;
    st = et_ring_attn_comm_create_v1(ET_RING_ATTN_ABI_MAJOR,
                                     ET_RING_ATTN_ABI_MINOR, 0u, 1u, -1, -1,
                                     &comm);
    if (st != ET_RING_ATTN_OK)
      return st;
    st = et_ring_attn_backward_v1(comm, &cfg, q_full, k_full, v_full, o_full,
                                  lse_full, do_full, dq_full, dk_full, dv_full,
                                  agg_stats_out);
    et_ring_attn_comm_destroy_v1(comm);
    return st;
  }

  const uint32_t W = cfg.world_size;
  const size_t T_loc = cfg.seq_len / (size_t)W;
  const size_t q_loc_bytes =
      cfg.batch * T_loc * cfg.num_heads * cfg.head_dim * sizeof(float);
  const size_t kv_loc_bytes =
      cfg.batch * T_loc * cfg.num_kv_heads * cfg.head_dim * sizeof(float);
  const size_t lse_loc_bytes =
      cfg.batch * T_loc * cfg.num_heads * sizeof(double);

  int edge_fds[ET_RING_ATTN_MAX_WORLD_SIZE][2];
  int res_fds[ET_RING_ATTN_MAX_WORLD_SIZE][2];
  pid_t pids[ET_RING_ATTN_MAX_WORLD_SIZE];
  for (uint32_t r = 0u; r < ET_RING_ATTN_MAX_WORLD_SIZE; ++r) {
    edge_fds[r][0] = edge_fds[r][1] = -1;
    res_fds[r][0] = res_fds[r][1] = -1;
    pids[r] = -1;
  }

  for (uint32_t r = 0u; r < W; ++r) {
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, edge_fds[r]) != 0 ||
        socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, res_fds[r]) != 0) {
      close_fd_array(edge_fds, W);
      close_fd_array(res_fds, W);
      return ET_RING_ATTN_ERR_IO;
    }
  }

  const int64_t start_ms = monotonic_ms();
  if (start_ms < 0) {
    close_fd_array(edge_fds, W);
    close_fd_array(res_fds, W);
    return ET_RING_ATTN_ERR_IO;
  }
  const int64_t deadline_ms = start_ms + (int64_t)cfg.timeout_ms;

  for (uint32_t r = 0u; r < W; ++r) {
    const pid_t pid = fork();
    if (pid < 0) {
      close_fd_array(edge_fds, W);
      close_fd_array(res_fds, W);
      reap_children(pids, W, 1);
      return ET_RING_ATTN_ERR_IO;
    }
    if (pid == 0) {
      const uint32_t prev_rank = (r + W - 1u) % W;
      const int send_fd = edge_fds[r][0];
      const int recv_fd = edge_fds[prev_rank][1];
      const int reply_fd = res_fds[r][1];

      for (uint32_t i = 0u; i < W; ++i) {
        if (edge_fds[i][0] != send_fd && edge_fds[i][0] >= 0)
          close(edge_fds[i][0]);
        if (edge_fds[i][1] != recv_fd && edge_fds[i][1] >= 0)
          close(edge_fds[i][1]);
        if (res_fds[i][0] >= 0)
          close(res_fds[i][0]);
        if (res_fds[i][1] != reply_fd && res_fds[i][1] >= 0)
          close(res_fds[i][1]);
      }

      unsigned char *q_loc = malloc(q_loc_bytes);
      unsigned char *k_loc = malloc(kv_loc_bytes);
      unsigned char *v_loc = malloc(kv_loc_bytes);
      unsigned char *o_loc = malloc(q_loc_bytes);
      unsigned char *lse_loc = malloc(lse_loc_bytes);
      unsigned char *do_loc = malloc(q_loc_bytes);
      unsigned char *dq_loc = malloc(q_loc_bytes);
      unsigned char *dk_loc = malloc(kv_loc_bytes);
      unsigned char *dv_loc = malloc(kv_loc_bytes);
      child_reply_header rep;
      memset(&rep, 0, sizeof(rep));

      if (q_loc == NULL || k_loc == NULL || v_loc == NULL || o_loc == NULL ||
          lse_loc == NULL || do_loc == NULL || dq_loc == NULL ||
          dk_loc == NULL || dv_loc == NULL) {
        rep.status = ET_RING_ATTN_ERR_ALLOC;
      } else {
        const size_t q_tok_bytes = cfg.num_heads * cfg.head_dim * sizeof(float);
        const size_t kv_tok_bytes =
            cfg.num_kv_heads * cfg.head_dim * sizeof(float);
        const size_t lse_tok_bytes = cfg.num_heads * sizeof(double);

        extract_shard_bytes((const unsigned char *)q_full, q_loc, cfg.batch,
                            cfg.seq_len, T_loc, r, q_tok_bytes);
        extract_shard_bytes((const unsigned char *)k_full, k_loc, cfg.batch,
                            cfg.seq_len, T_loc, r, kv_tok_bytes);
        extract_shard_bytes((const unsigned char *)v_full, v_loc, cfg.batch,
                            cfg.seq_len, T_loc, r, kv_tok_bytes);
        extract_shard_bytes((const unsigned char *)o_full, o_loc, cfg.batch,
                            cfg.seq_len, T_loc, r, q_tok_bytes);
        extract_shard_bytes((const unsigned char *)lse_full, lse_loc, cfg.batch,
                            cfg.seq_len, T_loc, r, lse_tok_bytes);
        extract_shard_bytes((const unsigned char *)do_full, do_loc, cfg.batch,
                            cfg.seq_len, T_loc, r, q_tok_bytes);

        et_ring_attn_comm_t *comm = NULL;
        rep.status = et_ring_attn_comm_create_v1(ET_RING_ATTN_ABI_MAJOR,
                                                 ET_RING_ATTN_ABI_MINOR, r, W,
                                                 send_fd, recv_fd, &comm);
        close(send_fd);
        close(recv_fd);
        if (rep.status == ET_RING_ATTN_OK) {
          rep.status = et_ring_attn_backward_v1(
              comm, &cfg, q_loc, k_loc, v_loc, o_loc, lse_loc, do_loc, dq_loc,
              dk_loc, dv_loc, &rep.stats);
          et_ring_attn_comm_destroy_v1(comm);
        }
      }

      const int64_t child_deadline = deadline_ms + 5000;
      if (send_all_deadline(reply_fd, (const unsigned char *)&rep, sizeof(rep),
                            child_deadline) == ET_RING_ATTN_OK &&
          rep.status == ET_RING_ATTN_OK) {
        (void)send_all_deadline(reply_fd, dq_loc, q_loc_bytes, child_deadline);
        (void)send_all_deadline(reply_fd, dk_loc, kv_loc_bytes, child_deadline);
        (void)send_all_deadline(reply_fd, dv_loc, kv_loc_bytes, child_deadline);
      }

      close(reply_fd);
      free(q_loc);
      free(k_loc);
      free(v_loc);
      free(o_loc);
      free(lse_loc);
      free(do_loc);
      free(dq_loc);
      free(dk_loc);
      free(dv_loc);
      _exit(0);
    }
    pids[r] = pid;
  }

  close_fd_array(edge_fds, W);
  for (uint32_t r = 0u; r < W; ++r) {
    close(res_fds[r][1]);
    res_fds[r][1] = -1;
  }

  unsigned char *stage_dq = malloc(q_bytes);
  unsigned char *stage_dk = malloc(kv_bytes);
  unsigned char *stage_dv = malloc(kv_bytes);
  unsigned char *rx_dq = malloc(q_loc_bytes);
  unsigned char *rx_dk = malloc(kv_loc_bytes);
  unsigned char *rx_dv = malloc(kv_loc_bytes);
  et_ring_attn_stats_v1 agg;
  memset(&agg, 0, sizeof(agg));

  if (stage_dq == NULL || stage_dk == NULL || stage_dv == NULL ||
      rx_dq == NULL || rx_dk == NULL || rx_dv == NULL) {
    st = ET_RING_ATTN_ERR_ALLOC;
  } else {
    st = ET_RING_ATTN_OK;
    for (uint32_t r = 0u; r < W; ++r) {
      child_reply_header rep;
      memset(&rep, 0, sizeof(rep));
      et_ring_attn_status_t rst = recv_all_deadline(
          res_fds[r][0], (unsigned char *)&rep, sizeof(rep), deadline_ms);
      if (rst != ET_RING_ATTN_OK) {
        st = rst;
        break;
      }
      if (rep.status != ET_RING_ATTN_OK) {
        st = rep.status;
        break;
      }
      rst = recv_all_deadline(res_fds[r][0], rx_dq, q_loc_bytes, deadline_ms);
      if (rst != ET_RING_ATTN_OK) {
        st = rst;
        break;
      }
      rst = recv_all_deadline(res_fds[r][0], rx_dk, kv_loc_bytes, deadline_ms);
      if (rst != ET_RING_ATTN_OK) {
        st = rst;
        break;
      }
      rst = recv_all_deadline(res_fds[r][0], rx_dv, kv_loc_bytes, deadline_ms);
      if (rst != ET_RING_ATTN_OK) {
        st = rst;
        break;
      }
      insert_shard_bytes(stage_dq, rx_dq, cfg.batch, cfg.seq_len, T_loc, r,
                         cfg.num_heads * cfg.head_dim * sizeof(float));
      insert_shard_bytes(stage_dk, rx_dk, cfg.batch, cfg.seq_len, T_loc, r,
                         cfg.num_kv_heads * cfg.head_dim * sizeof(float));
      insert_shard_bytes(stage_dv, rx_dv, cfg.batch, cfg.seq_len, T_loc, r,
                         cfg.num_kv_heads * cfg.head_dim * sizeof(float));
      agg.total_block_steps += rep.stats.total_block_steps;
      agg.executed_blocks += rep.stats.executed_blocks;
      agg.skipped_causal_blocks += rep.stats.skipped_causal_blocks;
      agg.bytes_sent += rep.stats.bytes_sent;
      agg.bytes_received += rep.stats.bytes_received;
      agg.ring_rounds = rep.stats.ring_rounds;
    }
  }

  close_fd_array(res_fds, W);
  reap_children(pids, W, st != ET_RING_ATTN_OK);

  if (st == ET_RING_ATTN_OK) {
    memcpy(dq_full, stage_dq, q_bytes);
    memcpy(dk_full, stage_dk, kv_bytes);
    memcpy(dv_full, stage_dv, kv_bytes);
    if (agg_stats_out != NULL)
      *agg_stats_out = agg;
  }

  free(stage_dq);
  free(stage_dk);
  free(stage_dv);
  free(rx_dq);
  free(rx_dk);
  free(rx_dv);
  return st;
}

/* ============================================================================
 * Single-Rank Dense Binary64 Attention Reference (Full [N, T, H, Dh])
 * ============================================================================ */

et_ring_attn_status_t et_ring_attn_dense_ref_forward_v1(
    const void *cfg_ptr, const void *q_full, const void *k_full,
    const void *v_full, void *o_full, void *lse_full) {
  et_ring_attn_config_v1 cfg;
  double scale = 0.0;
  et_ring_attn_status_t st = parse_config(cfg_ptr, &cfg, &scale);
  if (st != ET_RING_ATTN_OK)
    return st;

  const size_t N = cfg.batch;
  const size_t T = cfg.seq_len;
  const size_t H = cfg.num_heads;
  const size_t Hkv = cfg.num_kv_heads;
  const size_t Dh = cfg.head_dim;
  const size_t group = H / Hkv;

  const size_t q_elems = N * T * H * Dh;
  const size_t kv_elems = N * T * Hkv * Dh;
  const size_t lse_elems = N * T * H;
  const size_t q_bytes = q_elems * sizeof(float);
  const size_t kv_bytes = kv_elems * sizeof(float);
  const size_t lse_bytes = lse_elems * sizeof(double);

  if (!aligned_span(q_full, _Alignof(float), q_bytes) ||
      !aligned_span(k_full, _Alignof(float), kv_bytes) ||
      !aligned_span(v_full, _Alignof(float), kv_bytes) ||
      !aligned_span(o_full, _Alignof(float), q_bytes) ||
      !aligned_span(lse_full, _Alignof(double), lse_bytes))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const span_desc ins[] = {
      {q_full, q_bytes}, {k_full, kv_bytes}, {v_full, kv_bytes}};
  const span_desc outs[] = {{o_full, q_bytes}, {lse_full, lse_bytes}};
  if (!check_no_output_aliasing(ins, 3u, outs, 2u))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const float *q = (const float *)q_full;
  const float *k = (const float *)k_full;
  const float *v = (const float *)v_full;
  if (!all_f32_finite(q, q_elems) || !all_f32_finite(k, kv_elems) ||
      !all_f32_finite(v, kv_elems))
    return ET_RING_ATTN_ERR_NONFINITE;

  double *scores = malloc(T * sizeof(double));
  double *o_f64 = calloc(q_elems, sizeof(double));
  double *tmp_lse = malloc(lse_bytes);
  float *tmp_o = malloc(q_bytes);
  if (scores == NULL || o_f64 == NULL || tmp_lse == NULL || tmp_o == NULL) {
    free(scores);
    free(o_f64);
    free(tmp_lse);
    free(tmp_o);
    return ET_RING_ATTN_ERR_ALLOC;
  }

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0) {
    free(scores);
    free(o_f64);
    free(tmp_lse);
    free(tmp_o);
    return ET_RING_ATTN_ERR_IO;
  }
  fesetround(FE_TONEAREST);

  int ok = 1;
  for (size_t n = 0; n < N && ok; ++n) {
    for (size_t tq = 0; tq < T && ok; ++tq) {
      const size_t j_end = cfg.causal ? (tq + 1u) : T;
      for (size_t h = 0; h < H && ok; ++h) {
        const size_t hkv = h / group;
        const size_t q_idx = (n * T + tq) * H + h;
        const float *q_vec = q + q_idx * Dh;
        double *o_vec = o_f64 + q_idx * Dh;

        double m = -INFINITY;
        for (size_t tk = 0; tk < j_end; ++tk) {
          const float *k_vec = k + (((n * T + tk) * Hkv + hkv) * Dh);
          double dot = 0.0;
          for (size_t d = 0; d < Dh; ++d)
            dot += (double)q_vec[d] * (double)k_vec[d];
          const double s = dot * scale;
          scores[tk] = s;
          if (s > m)
            m = s;
        }
        double sum_exp = 0.0;
        for (size_t tk = 0; tk < j_end; ++tk)
          sum_exp += exp(scores[tk] - m);
        const double lse = m + log(sum_exp);
        if (!isfinite(lse)) {
          ok = 0;
          break;
        }
        tmp_lse[q_idx] = lse;

        for (size_t tk = 0; tk < j_end; ++tk) {
          const double p = exp(scores[tk] - lse);
          const float *v_vec = v + (((n * T + tk) * Hkv + hkv) * Dh);
          for (size_t d = 0; d < Dh; ++d)
            o_vec[d] += p * (double)v_vec[d];
        }
      }
    }
  }

  if (ok && !round_f64_to_f32_checked(o_f64, tmp_o, q_elems))
    ok = 0;

  fesetenv(&saved_env);

  if (!ok) {
    free(scores);
    free(o_f64);
    free(tmp_lse);
    free(tmp_o);
    return ET_RING_ATTN_ERR_NONFINITE;
  }

  memcpy(o_full, tmp_o, q_bytes);
  memcpy(lse_full, tmp_lse, lse_bytes);
  free(scores);
  free(o_f64);
  free(tmp_lse);
  free(tmp_o);
  return ET_RING_ATTN_OK;
}

et_ring_attn_status_t et_ring_attn_dense_ref_backward_v1(
    const void *cfg_ptr, const void *q_full, const void *k_full,
    const void *v_full, const void *o_full, const void *lse_full,
    const void *do_full, void *dq_full, void *dk_full, void *dv_full) {
  et_ring_attn_config_v1 cfg;
  double scale = 0.0;
  et_ring_attn_status_t st = parse_config(cfg_ptr, &cfg, &scale);
  if (st != ET_RING_ATTN_OK)
    return st;

  const size_t N = cfg.batch;
  const size_t T = cfg.seq_len;
  const size_t H = cfg.num_heads;
  const size_t Hkv = cfg.num_kv_heads;
  const size_t Dh = cfg.head_dim;
  const size_t group = H / Hkv;

  const size_t q_elems = N * T * H * Dh;
  const size_t kv_elems = N * T * Hkv * Dh;
  const size_t lse_elems = N * T * H;
  const size_t q_bytes = q_elems * sizeof(float);
  const size_t kv_bytes = kv_elems * sizeof(float);
  const size_t lse_bytes = lse_elems * sizeof(double);

  if (!aligned_span(q_full, _Alignof(float), q_bytes) ||
      !aligned_span(k_full, _Alignof(float), kv_bytes) ||
      !aligned_span(v_full, _Alignof(float), kv_bytes) ||
      !aligned_span(o_full, _Alignof(float), q_bytes) ||
      !aligned_span(lse_full, _Alignof(double), lse_bytes) ||
      !aligned_span(do_full, _Alignof(float), q_bytes) ||
      !aligned_span(dq_full, _Alignof(float), q_bytes) ||
      !aligned_span(dk_full, _Alignof(float), kv_bytes) ||
      !aligned_span(dv_full, _Alignof(float), kv_bytes))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const span_desc ins[] = {{q_full, q_bytes},     {k_full, kv_bytes},
                           {v_full, kv_bytes},    {o_full, q_bytes},
                           {lse_full, lse_bytes}, {do_full, q_bytes}};
  const span_desc outs[] = {
      {dq_full, q_bytes}, {dk_full, kv_bytes}, {dv_full, kv_bytes}};
  if (!check_no_output_aliasing(ins, 6u, outs, 3u))
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const float *q = (const float *)q_full;
  const float *k = (const float *)k_full;
  const float *v = (const float *)v_full;
  const float *o = (const float *)o_full;
  const double *lse = (const double *)lse_full;
  const float *do_f = (const float *)do_full;

  if (!all_f32_finite(q, q_elems) || !all_f32_finite(k, kv_elems) ||
      !all_f32_finite(v, kv_elems) || !all_f32_finite(o, q_elems) ||
      !all_f64_finite(lse, lse_elems) || !all_f32_finite(do_f, q_elems))
    return ET_RING_ATTN_ERR_NONFINITE;

  double *dq_f64 = calloc(q_elems, sizeof(double));
  double *dk_f64 = calloc(kv_elems, sizeof(double));
  double *dv_f64 = calloc(kv_elems, sizeof(double));
  float *tmp_dq = malloc(q_bytes);
  float *tmp_dk = malloc(kv_bytes);
  float *tmp_dv = malloc(kv_bytes);
  if (dq_f64 == NULL || dk_f64 == NULL || dv_f64 == NULL || tmp_dq == NULL ||
      tmp_dk == NULL || tmp_dv == NULL) {
    free(dq_f64);
    free(dk_f64);
    free(dv_f64);
    free(tmp_dq);
    free(tmp_dk);
    free(tmp_dv);
    return ET_RING_ATTN_ERR_ALLOC;
  }

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0) {
    free(dq_f64);
    free(dk_f64);
    free(dv_f64);
    free(tmp_dq);
    free(tmp_dk);
    free(tmp_dv);
    return ET_RING_ATTN_ERR_IO;
  }
  fesetround(FE_TONEAREST);

  for (size_t n = 0; n < N; ++n) {
    for (size_t tq = 0; tq < T; ++tq) {
      const size_t j_end = cfg.causal ? (tq + 1u) : T;
      for (size_t h = 0; h < H; ++h) {
        const size_t hkv = h / group;
        const size_t q_idx = (n * T + tq) * H + h;
        const float *q_vec = q + q_idx * Dh;
        const float *o_vec = o + q_idx * Dh;
        const float *do_vec = do_f + q_idx * Dh;
        double *dq_vec = dq_f64 + q_idx * Dh;
        const double lse_i = lse[q_idx];

        double d_i = 0.0;
        for (size_t d = 0; d < Dh; ++d)
          d_i += (double)do_vec[d] * (double)o_vec[d];

        for (size_t tk = 0; tk < j_end; ++tk) {
          const size_t kv_base = ((n * T + tk) * Hkv + hkv) * Dh;
          const float *k_vec = k + kv_base;
          const float *v_vec = v + kv_base;
          double *dk_vec = dk_f64 + kv_base;
          double *dv_vec = dv_f64 + kv_base;

          double dot_qk = 0.0;
          double dp = 0.0;
          for (size_t d = 0; d < Dh; ++d) {
            dot_qk += (double)q_vec[d] * (double)k_vec[d];
            dp += (double)do_vec[d] * (double)v_vec[d];
          }
          const double p = exp(dot_qk * scale - lse_i);
          const double ds = p * (dp - d_i);
          const double ddot = ds * scale;

          for (size_t d = 0; d < Dh; ++d) {
            dq_vec[d] += ddot * (double)k_vec[d];
            dk_vec[d] += ddot * (double)q_vec[d];
            dv_vec[d] += p * (double)do_vec[d];
          }
        }
      }
    }
  }

  int ok = round_f64_to_f32_checked(dq_f64, tmp_dq, q_elems) &&
           round_f64_to_f32_checked(dk_f64, tmp_dk, kv_elems) &&
           round_f64_to_f32_checked(dv_f64, tmp_dv, kv_elems);

  fesetenv(&saved_env);

  if (!ok) {
    free(dq_f64);
    free(dk_f64);
    free(dv_f64);
    free(tmp_dq);
    free(tmp_dk);
    free(tmp_dv);
    return ET_RING_ATTN_ERR_NONFINITE;
  }

  memcpy(dq_full, tmp_dq, q_bytes);
  memcpy(dk_full, tmp_dk, kv_bytes);
  memcpy(dv_full, tmp_dv, kv_bytes);
  free(dq_f64);
  free(dk_f64);
  free(dv_f64);
  free(tmp_dq);
  free(tmp_dk);
  free(tmp_dv);
  return ET_RING_ATTN_OK;
}
