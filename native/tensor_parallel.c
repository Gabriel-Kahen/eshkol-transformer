#define _GNU_SOURCE
#include "eshkol_transformer/tensor_parallel_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <poll.h>
#include <stdalign.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   FLT_MAX_EXP == 128 && FLT_EVAL_METHOD == 0,
               "TP engine requires IEEE-754 binary32 float");

#define ET_TP_WIRE_MAGIC UINT64_C(0x4554545031440100)
#define ET_TP_REQ_WORDS 8u
#define ET_TP_REP_WORDS 3u

enum { ET_TP_KIND_ALLREDUCE_SUM = 1u, ET_TP_KIND_ALLGATHER = 2u };
enum { ET_TP_PHASE_GRANT = 1u, ET_TP_PHASE_RESULT = 2u, ET_TP_PHASE_COMMIT = 3u,
       ET_TP_PHASE_ABORT = 4u };

struct et_tp_group {
  uint32_t rank;
  uint32_t world_size;
  int32_t default_timeout_ms;
  uint64_t seq;
  et_tp_status_t fatal_status;
  int fd[ET_TP_MAX_WORLD_SIZE];
  et_tp_collective_stats_t stats;
};

/* =========================================================================
 * Numeric, Memory, Alignment, Aliasing, and Overflow Helpers
 * ========================================================================= */

static int tp_world_valid(uint32_t world_size) {
  return world_size == 1u || world_size == 2u || world_size == 4u;
}

static int tp_mul_size(size_t a, size_t b, size_t *out) {
  if (a != 0u && b > SIZE_MAX / a)
    return 0;
  *out = a * b;
  return 1;
}

static int tp_aligned_f32(const void *ptr) {
  return ptr != NULL && (((uintptr_t)ptr % alignof(float)) == 0u);
}

static int tp_spans_overlap(const void *a, size_t a_bytes, const void *b,
                            size_t b_bytes) {
  if (a == NULL || b == NULL || a_bytes == 0u || b_bytes == 0u)
    return 0;
  const uintptr_t a0 = (uintptr_t)a;
  const uintptr_t b0 = (uintptr_t)b;
  if (a0 <= b0)
    return (b0 - a0) < a_bytes;
  return (a0 - b0) < b_bytes;
}

static et_tp_status_t tp_check_f32_buf(const float *ptr, size_t count,
                                       int check_finite) {
  if (ptr == NULL)
    return ET_TP_ERR_INVALID_ARG;
  if (!tp_aligned_f32(ptr))
    return ET_TP_ERR_ALIGNMENT;
  if (check_finite) {
    for (size_t i = 0u; i < count; ++i) {
      if (!isfinite(ptr[i]))
        return ET_TP_ERR_NONFINITE;
    }
  }
  return ET_TP_OK;
}

static et_tp_status_t tp_check_f32_out(float *ptr) {
  if (ptr == NULL)
    return ET_TP_ERR_INVALID_ARG;
  if (!tp_aligned_f32(ptr))
    return ET_TP_ERR_ALIGNMENT;
  return ET_TP_OK;
}

static double tp_sigmoid_f64(double x) {
  if (x >= 0.0) {
    return 1.0 / (1.0 + exp(-x));
  }
  const double ex = exp(x);
  return ex / (1.0 + ex);
}

static int tp_retryable(int err) {
  return err == EINTR || err == EAGAIN || err == EWOULDBLOCK;
}

static int64_t tp_now_ms(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    return -1;
  return (int64_t)ts.tv_sec * 1000 + (int64_t)(ts.tv_nsec / 1000000L);
}

static int tp_resolve_timeout(const et_tp_group_t *group, int32_t timeout_ms) {
  if (timeout_ms > 0)
    return (int)timeout_ms;
  if (timeout_ms == 0 && group != NULL && group->default_timeout_ms > 0)
    return (int)group->default_timeout_ms;
  return -1;
}

/* =========================================================================
 * AF_UNIX Socket Collective Implementation
 * ========================================================================= */

static et_tp_status_t tp_validate_unix_stream_fd(int fd, struct stat *st) {
  struct sockaddr_storage local_addr;
  struct sockaddr_storage peer_addr;
  socklen_t len = (socklen_t)sizeof(local_addr);
  int sock_type = 0;

  if (fd < 0 || fstat(fd, st) != 0)
    return ET_TP_ERR_INVALID_ARG;
  if (getsockname(fd, (struct sockaddr *)&local_addr, &len) != 0 ||
      local_addr.ss_family != AF_UNIX)
    return ET_TP_ERR_INVALID_ARG;
  len = (socklen_t)sizeof(peer_addr);
  if (getpeername(fd, (struct sockaddr *)&peer_addr, &len) != 0 ||
      peer_addr.ss_family != AF_UNIX)
    return ET_TP_ERR_INVALID_ARG;
  len = (socklen_t)sizeof(sock_type);
  if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &sock_type, &len) != 0 ||
      sock_type != SOCK_STREAM)
    return ET_TP_ERR_INVALID_ARG;
  return ET_TP_OK;
}

et_tp_status_t et_tp_group_create(uint32_t abi_major, uint32_t abi_minor,
                                  uint32_t rank, uint32_t world_size,
                                  const int *fds, size_t fd_count,
                                  int32_t default_timeout_ms,
                                  et_tp_group_t **out_group) {
  if (out_group == NULL)
    return ET_TP_ERR_INVALID_ARG;
  *out_group = NULL;

  if (abi_major != ET_TP_ABI_MAJOR || abi_minor > ET_TP_ABI_MINOR ||
      !tp_world_valid(world_size) || rank >= world_size ||
      default_timeout_ms <= 0)
    return ET_TP_ERR_INVALID_ARG;

  const size_t expected_fds =
      (world_size == 1u) ? 0u : ((rank == 0u) ? (size_t)(world_size - 1u) : 1u);
  if (fd_count != expected_fds || (expected_fds > 0u && fds == NULL))
    return ET_TP_ERR_INVALID_ARG;

  et_tp_group_t *group = (et_tp_group_t *)calloc(1u, sizeof(*group));
  if (group == NULL)
    return ET_TP_ERR_ALLOC;

  group->rank = rank;
  group->world_size = world_size;
  group->default_timeout_ms = default_timeout_ms;
  group->seq = 0u;
  group->fatal_status = ET_TP_OK;
  for (uint32_t i = 0u; i < ET_TP_MAX_WORLD_SIZE; ++i)
    group->fd[i] = -1;

  struct stat ids[ET_TP_MAX_WORLD_SIZE];
  et_tp_status_t st = ET_TP_OK;
  for (size_t i = 0u; i < expected_fds; ++i) {
    int dup_fd = -1;
    do {
      dup_fd = fcntl(fds[i], F_DUPFD_CLOEXEC, 0);
    } while (dup_fd < 0 && errno == EINTR);
    if (dup_fd < 0) {
      st = (errno == EBADF) ? ET_TP_ERR_INVALID_ARG : ET_TP_ERR_IO;
      break;
    }
    const uint32_t slot = (rank == 0u) ? (uint32_t)(i + 1u) : 0u;
    group->fd[slot] = dup_fd;

    const int flags = fcntl(dup_fd, F_GETFL, 0);
    if (flags < 0 || fcntl(dup_fd, F_SETFL, flags | O_NONBLOCK) != 0) {
      st = ET_TP_ERR_IO;
      break;
    }

    st = tp_validate_unix_stream_fd(dup_fd, &ids[i]);
    for (size_t j = 0u; st == ET_TP_OK && j < i; ++j) {
      if (ids[i].st_dev == ids[j].st_dev && ids[i].st_ino == ids[j].st_ino)
        st = ET_TP_ERR_INVALID_ARG;
    }
    if (st != ET_TP_OK)
      break;
  }

  if (st != ET_TP_OK) {
    et_tp_group_destroy(group);
    return st;
  }

  *out_group = group;
  return ET_TP_OK;
}

void et_tp_group_destroy(et_tp_group_t *group) {
  if (group == NULL)
    return;
  for (uint32_t i = 0u; i < ET_TP_MAX_WORLD_SIZE; ++i) {
    if (group->fd[i] >= 0) {
      close(group->fd[i]);
      group->fd[i] = -1;
    }
  }
  free(group);
}

et_tp_status_t et_tp_group_rank(const et_tp_group_t *group,
                                uint32_t *out_rank) {
  if (group == NULL || out_rank == NULL)
    return ET_TP_ERR_INVALID_ARG;
  *out_rank = group->rank;
  return ET_TP_OK;
}

et_tp_status_t et_tp_group_world_size(const et_tp_group_t *group,
                                      uint32_t *out_world_size) {
  if (group == NULL || out_world_size == NULL)
    return ET_TP_ERR_INVALID_ARG;
  *out_world_size = group->world_size;
  return ET_TP_OK;
}

et_tp_status_t et_tp_group_get_stats(const et_tp_group_t *group,
                                     et_tp_collective_stats_t *out_stats) {
  if (group == NULL || out_stats == NULL)
    return ET_TP_ERR_INVALID_ARG;
  *out_stats = group->stats;
  return ET_TP_OK;
}

et_tp_status_t et_tp_group_reset_stats(et_tp_group_t *group) {
  if (group == NULL)
    return ET_TP_ERR_INVALID_ARG;
  memset(&group->stats, 0, sizeof(group->stats));
  return ET_TP_OK;
}

et_tp_status_t et_tp_group_set_timeout_ms(et_tp_group_t *group,
                                          int32_t timeout_ms) {
  if (group == NULL || timeout_ms <= 0)
    return ET_TP_ERR_INVALID_ARG;
  group->default_timeout_ms = timeout_ms;
  return ET_TP_OK;
}

static void tp_shutdown_all(et_tp_group_t *group) {
  for (uint32_t i = 0u; i < group->world_size; ++i) {
    if (group->fd[i] >= 0)
      (void)shutdown(group->fd[i], SHUT_RDWR);
  }
}

static et_tp_status_t tp_wait_fd(et_tp_group_t *group, uint32_t peer,
                                 short events, int64_t deadline,
                                 int watch_all_peers) {
  for (;;) {
    const int64_t now = tp_now_ms();
    if (now < 0)
      return ET_TP_ERR_IO;
    const int64_t rem = deadline - now;
    if (rem <= 0)
      return ET_TP_ERR_TIMEOUT;

    struct pollfd pfds[ET_TP_MAX_WORLD_SIZE];
    uint32_t peers[ET_TP_MAX_WORLD_SIZE];
    nfds_t nfds = 0u;

    if (watch_all_peers && group->rank == 0u) {
      for (uint32_t r = 1u; r < group->world_size; ++r) {
        if (group->fd[r] < 0)
          continue;
        pfds[nfds].fd = group->fd[r];
        pfds[nfds].events = (r == peer) ? events : (POLLERR | POLLHUP);
        pfds[nfds].revents = 0;
        peers[nfds] = r;
        ++nfds;
      }
    } else {
      pfds[0].fd = group->fd[peer];
      pfds[0].events = events;
      pfds[0].revents = 0;
      peers[0] = peer;
      nfds = 1u;
    }

    const int poll_ms = (rem > 60000) ? 60000 : (int)rem;
    const int rc = poll(pfds, nfds, poll_ms);
    if (rc < 0) {
      if (errno == EINTR)
        continue;
      return ET_TP_ERR_IO;
    }
    if (rc == 0)
      continue;

    for (nfds_t i = 0u; i < nfds; ++i) {
      if (peers[i] == peer) {
        if ((pfds[i].revents & (events | POLLERR | POLLHUP)) != 0)
          return ET_TP_OK;
      } else if ((pfds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        return ET_TP_ERR_PEER_DISCONNECTED;
      }
    }
  }
}

static et_tp_status_t tp_send_all(et_tp_group_t *group, uint32_t peer,
                                  const void *buf, size_t bytes,
                                  int64_t deadline) {
  const unsigned char *p = (const unsigned char *)buf;
  size_t done = 0u;
  while (done < bytes) {
    const ssize_t n =
        send(group->fd[peer], p + done, bytes - done, MSG_NOSIGNAL);
    if (n > 0) {
      done += (size_t)n;
      continue;
    }
    if (n == 0)
      return ET_TP_ERR_PEER_DISCONNECTED;
    if (tp_retryable(errno)) {
      const et_tp_status_t w =
          tp_wait_fd(group, peer, POLLOUT, deadline, group->rank == 0u);
      if (w != ET_TP_OK)
        return w;
      continue;
    }
    if (errno == EPIPE || errno == ECONNRESET)
      return ET_TP_ERR_PEER_DISCONNECTED;
    return ET_TP_ERR_IO;
  }
  return ET_TP_OK;
}

static et_tp_status_t tp_recv_all(et_tp_group_t *group, uint32_t peer,
                                  void *buf, size_t bytes, int64_t deadline) {
  unsigned char *p = (unsigned char *)buf;
  size_t done = 0u;
  while (done < bytes) {
    const ssize_t n = recv(group->fd[peer], p + done, bytes - done, 0);
    if (n > 0) {
      done += (size_t)n;
      continue;
    }
    if (n == 0)
      return ET_TP_ERR_PEER_DISCONNECTED;
    if (tp_retryable(errno)) {
      const et_tp_status_t w =
          tp_wait_fd(group, peer, POLLIN, deadline, group->rank == 0u);
      if (w != ET_TP_OK)
        return w;
      continue;
    }
    if (errno == ECONNRESET)
      return ET_TP_ERR_PEER_DISCONNECTED;
    return ET_TP_ERR_IO;
  }
  return ET_TP_OK;
}

static void tp_broadcast_abort(et_tp_group_t *group, uint64_t seq,
                               et_tp_status_t status, int64_t deadline) {
  uint64_t rep[ET_TP_REP_WORDS];
  rep[0] = seq;
  rep[1] = ET_TP_PHASE_ABORT;
  rep[2] = (uint64_t)(int64_t)status;
  for (uint32_t r = 1u; r < group->world_size; ++r) {
    if (group->fd[r] >= 0)
      (void)tp_send_all(group, r, rep, sizeof(rep), deadline);
  }
}

static et_tp_status_t tp_collective_run(et_tp_group_t *group, uint32_t kind,
                                        const float *sendbuf, float *recvbuf,
                                        size_t send_count, uint64_t tag,
                                        int32_t timeout_ms) {
  if (group == NULL)
    return ET_TP_ERR_INVALID_ARG;
  const int resolved_ms = tp_resolve_timeout(group, timeout_ms);
  if (resolved_ms <= 0)
    return ET_TP_ERR_INVALID_ARG;
  if (group->fatal_status != ET_TP_OK)
    return group->fatal_status;

  if (send_count == 0u)
    return ET_TP_ERR_INVALID_ARG;

  size_t send_bytes = 0u;
  size_t recv_count = 0u;
  size_t recv_bytes = 0u;
  if (!tp_mul_size(send_count, sizeof(float), &send_bytes))
    return ET_TP_ERR_OVERFLOW;
  if (kind == ET_TP_KIND_ALLGATHER) {
    if (!tp_mul_size(send_count, (size_t)group->world_size, &recv_count) ||
        !tp_mul_size(recv_count, sizeof(float), &recv_bytes))
      return ET_TP_ERR_OVERFLOW;
  } else {
    recv_count = send_count;
    recv_bytes = send_bytes;
  }
  if (recv_bytes > ET_TP_MAX_COLLECTIVE_BYTES)
    return ET_TP_ERR_OVERFLOW;

  if (sendbuf == NULL || recvbuf == NULL)
    return ET_TP_ERR_INVALID_ARG;
  if (!tp_aligned_f32(sendbuf) || !tp_aligned_f32(recvbuf))
    return ET_TP_ERR_ALIGNMENT;

  if (kind == ET_TP_KIND_ALLGATHER) {
    if (tp_spans_overlap(sendbuf, send_bytes, recvbuf, recv_bytes))
      return ET_TP_ERR_ALIAS;
  } else {
    if (sendbuf != recvbuf &&
        tp_spans_overlap(sendbuf, send_bytes, recvbuf, recv_bytes))
      return ET_TP_ERR_ALIAS;
  }

  et_tp_status_t local_status = ET_TP_OK;
  for (size_t i = 0u; i < send_count; ++i) {
    if (!isfinite(sendbuf[i])) {
      local_status = ET_TP_ERR_NONFINITE;
      break;
    }
  }

  /* Single-rank fast path: no sockets, still validates finiteness & stats. */
  if (group->world_size == 1u) {
    if (local_status != ET_TP_OK)
      return local_status;
    if (sendbuf != recvbuf)
      memcpy(recvbuf, sendbuf, send_bytes);
    ++group->seq;
    if (kind == ET_TP_KIND_ALLREDUCE_SUM) {
      ++group->stats.allreduce_calls;
      group->stats.allreduce_elements += (uint64_t)send_count;
      group->stats.allreduce_bytes += (uint64_t)send_bytes;
    } else {
      ++group->stats.allgather_calls;
      group->stats.allgather_elements += (uint64_t)recv_count;
      group->stats.allgather_bytes += (uint64_t)recv_bytes;
    }
    return ET_TP_OK;
  }

  const int64_t start = tp_now_ms();
  if (start < 0)
    return ET_TP_ERR_IO;
  const int64_t deadline = start + (int64_t)resolved_ms;
  const uint64_t seq = group->seq;

  if (group->rank == 0u) {
    et_tp_status_t soft_err = local_status;
    et_tp_status_t hard_err = ET_TP_OK;

    /* 1. Receive requests from all peers 1..W-1. */
    for (uint32_t r = 1u; r < group->world_size; ++r) {
      uint64_t req[ET_TP_REQ_WORDS];
      const et_tp_status_t rc =
          tp_recv_all(group, r, req, sizeof(req), deadline);
      if (rc != ET_TP_OK) {
        hard_err = rc;
        break;
      }
      if (req[0] != ET_TP_WIRE_MAGIC || req[1] != seq || req[2] != r ||
          req[3] != group->world_size || req[4] != kind || req[5] != tag ||
          req[6] != (uint64_t)send_count) {
        hard_err = ET_TP_ERR_MISMATCH;
        break;
      }
      const et_tp_status_t peer_st = (et_tp_status_t)(int64_t)req[7];
      if (peer_st != ET_TP_OK && soft_err == ET_TP_OK)
        soft_err = peer_st;
    }

    if (hard_err != ET_TP_OK) {
      tp_broadcast_abort(group, seq, hard_err, deadline);
      tp_shutdown_all(group);
      group->fatal_status = hard_err;
      return hard_err;
    }

    if (soft_err != ET_TP_OK) {
      tp_broadcast_abort(group, seq, soft_err, deadline);
      ++group->seq;
      return soft_err;
    }

    /* 2. Allocate staging buffers. */
    float *result_stage = (float *)malloc(recv_bytes);
    double *acc_stage = NULL;
    float *peer_stage = (float *)malloc(send_bytes);
    if (kind == ET_TP_KIND_ALLREDUCE_SUM)
      acc_stage = (double *)malloc(send_count * sizeof(double));
    if (result_stage == NULL || peer_stage == NULL ||
        (kind == ET_TP_KIND_ALLREDUCE_SUM && acc_stage == NULL)) {
      free(result_stage);
      free(acc_stage);
      free(peer_stage);
      tp_broadcast_abort(group, seq, ET_TP_ERR_ALLOC, deadline);
      ++group->seq;
      return ET_TP_ERR_ALLOC;
    }

    if (kind == ET_TP_KIND_ALLREDUCE_SUM) {
      for (size_t i = 0u; i < send_count; ++i)
        acc_stage[i] = (double)sendbuf[i];
    } else {
      memcpy(result_stage, sendbuf, send_bytes);
    }

    /* 3. Grant & read payloads in canonical ascending rank order 1..W-1. */
    for (uint32_t r = 1u; r < group->world_size; ++r) {
      uint64_t grant[ET_TP_REP_WORDS] = {seq, ET_TP_PHASE_GRANT,
                                         (uint64_t)(int64_t)ET_TP_OK};
      hard_err = tp_send_all(group, r, grant, sizeof(grant), deadline);
      if (hard_err != ET_TP_OK)
        break;
      hard_err = tp_recv_all(group, r, peer_stage, send_bytes, deadline);
      if (hard_err != ET_TP_OK)
        break;
      if (kind == ET_TP_KIND_ALLREDUCE_SUM) {
        for (size_t i = 0u; i < send_count; ++i) {
          if (!isfinite(peer_stage[i]))
            soft_err = ET_TP_ERR_NONFINITE;
          else
            acc_stage[i] += (double)peer_stage[i];
        }
      } else {
        memcpy(result_stage + (size_t)r * send_count, peer_stage, send_bytes);
      }
    }

    if (hard_err == ET_TP_OK && soft_err == ET_TP_OK &&
        kind == ET_TP_KIND_ALLREDUCE_SUM) {
      for (size_t i = 0u; i < send_count; ++i) {
        if (!isfinite(acc_stage[i])) {
          soft_err = ET_TP_ERR_NONFINITE;
          break;
        }
        const float narrowed = (float)acc_stage[i];
        if (!isfinite(narrowed)) {
          soft_err = ET_TP_ERR_NONFINITE;
          break;
        }
        result_stage[i] = narrowed;
      }
    }

    free(acc_stage);
    free(peer_stage);

    if (hard_err != ET_TP_OK) {
      free(result_stage);
      tp_broadcast_abort(group, seq, hard_err, deadline);
      tp_shutdown_all(group);
      group->fatal_status = hard_err;
      return hard_err;
    }

    if (soft_err != ET_TP_OK) {
      free(result_stage);
      tp_broadcast_abort(group, seq, soft_err, deadline);
      ++group->seq;
      return soft_err;
    }

    /* 4. Broadcast RESULT + payload to all workers. */
    for (uint32_t r = 1u; r < group->world_size; ++r) {
      uint64_t res_hdr[ET_TP_REP_WORDS] = {seq, ET_TP_PHASE_RESULT,
                                           (uint64_t)(int64_t)ET_TP_OK};
      hard_err = tp_send_all(group, r, res_hdr, sizeof(res_hdr), deadline);
      if (hard_err != ET_TP_OK)
        break;
      hard_err = tp_send_all(group, r, result_stage, recv_bytes, deadline);
      if (hard_err != ET_TP_OK)
        break;
    }

    /* 5. Collect ACKs from all workers. */
    for (uint32_t r = 1u; hard_err == ET_TP_OK && r < group->world_size; ++r) {
      uint64_t ack = 0u;
      hard_err = tp_recv_all(group, r, &ack, sizeof(ack), deadline);
      if (hard_err == ET_TP_OK && ack != seq)
        hard_err = ET_TP_ERR_MISMATCH;
    }

    if (hard_err != ET_TP_OK) {
      free(result_stage);
      tp_shutdown_all(group);
      group->fatal_status = hard_err;
      return hard_err;
    }

    /* 6. Send COMMIT to all workers, then publish locally. */
    for (uint32_t r = 1u; r < group->world_size; ++r) {
      uint64_t commit[ET_TP_REP_WORDS] = {seq, ET_TP_PHASE_COMMIT,
                                          (uint64_t)(int64_t)ET_TP_OK};
      hard_err = tp_send_all(group, r, commit, sizeof(commit), deadline);
      if (hard_err != ET_TP_OK)
        break;
    }

    if (hard_err != ET_TP_OK) {
      free(result_stage);
      tp_shutdown_all(group);
      group->fatal_status = hard_err;
      return hard_err;
    }

    memcpy(recvbuf, result_stage, recv_bytes);
    free(result_stage);
    ++group->seq;
  } else {
    /* Worker rank r > 0 */
    uint64_t req[ET_TP_REQ_WORDS];
    req[0] = ET_TP_WIRE_MAGIC;
    req[1] = seq;
    req[2] = group->rank;
    req[3] = group->world_size;
    req[4] = kind;
    req[5] = tag;
    req[6] = (uint64_t)send_count;
    req[7] = (uint64_t)(int64_t)local_status;

    et_tp_status_t st = tp_send_all(group, 0u, req, sizeof(req), deadline);
    if (st != ET_TP_OK) {
      tp_shutdown_all(group);
      group->fatal_status = st;
      return st;
    }

    uint64_t rep[ET_TP_REP_WORDS];
    st = tp_recv_all(group, 0u, rep, sizeof(rep), deadline);
    if (st != ET_TP_OK) {
      tp_shutdown_all(group);
      group->fatal_status = st;
      return st;
    }
    if (rep[0] != seq) {
      tp_shutdown_all(group);
      group->fatal_status = ET_TP_ERR_MISMATCH;
      return ET_TP_ERR_MISMATCH;
    }
    if (rep[1] == ET_TP_PHASE_ABORT) {
      const et_tp_status_t abort_st = (et_tp_status_t)(int64_t)rep[2];
      if (abort_st == ET_TP_ERR_NONFINITE || abort_st == ET_TP_ERR_ALLOC) {
        ++group->seq;
        return abort_st;
      }
      tp_shutdown_all(group);
      group->fatal_status = (abort_st == ET_TP_OK) ? ET_TP_ERR_IO : abort_st;
      return group->fatal_status;
    }
    if (rep[1] != ET_TP_PHASE_GRANT) {
      tp_shutdown_all(group);
      group->fatal_status = ET_TP_ERR_MISMATCH;
      return ET_TP_ERR_MISMATCH;
    }

    st = tp_send_all(group, 0u, sendbuf, send_bytes, deadline);
    if (st != ET_TP_OK) {
      tp_shutdown_all(group);
      group->fatal_status = st;
      return st;
    }

    st = tp_recv_all(group, 0u, rep, sizeof(rep), deadline);
    if (st != ET_TP_OK) {
      tp_shutdown_all(group);
      group->fatal_status = st;
      return st;
    }
    if (rep[0] != seq) {
      tp_shutdown_all(group);
      group->fatal_status = ET_TP_ERR_MISMATCH;
      return ET_TP_ERR_MISMATCH;
    }
    if (rep[1] == ET_TP_PHASE_ABORT) {
      const et_tp_status_t abort_st = (et_tp_status_t)(int64_t)rep[2];
      if (abort_st == ET_TP_ERR_NONFINITE || abort_st == ET_TP_ERR_ALLOC) {
        ++group->seq;
        return abort_st;
      }
      tp_shutdown_all(group);
      group->fatal_status = (abort_st == ET_TP_OK) ? ET_TP_ERR_IO : abort_st;
      return group->fatal_status;
    }
    if (rep[1] != ET_TP_PHASE_RESULT) {
      tp_shutdown_all(group);
      group->fatal_status = ET_TP_ERR_MISMATCH;
      return ET_TP_ERR_MISMATCH;
    }

    float *result_stage = (float *)malloc(recv_bytes);
    if (result_stage == NULL) {
      tp_shutdown_all(group);
      group->fatal_status = ET_TP_ERR_ALLOC;
      return ET_TP_ERR_ALLOC;
    }

    st = tp_recv_all(group, 0u, result_stage, recv_bytes, deadline);
    if (st != ET_TP_OK) {
      free(result_stage);
      tp_shutdown_all(group);
      group->fatal_status = st;
      return st;
    }

    const uint64_t ack = seq;
    st = tp_send_all(group, 0u, &ack, sizeof(ack), deadline);
    if (st != ET_TP_OK) {
      free(result_stage);
      tp_shutdown_all(group);
      group->fatal_status = st;
      return st;
    }

    st = tp_recv_all(group, 0u, rep, sizeof(rep), deadline);
    if (st != ET_TP_OK || rep[0] != seq || rep[1] != ET_TP_PHASE_COMMIT) {
      free(result_stage);
      tp_shutdown_all(group);
      group->fatal_status = (st != ET_TP_OK) ? st : ET_TP_ERR_MISMATCH;
      return group->fatal_status;
    }

    memcpy(recvbuf, result_stage, recv_bytes);
    free(result_stage);
    ++group->seq;
  }

  if (kind == ET_TP_KIND_ALLREDUCE_SUM) {
    ++group->stats.allreduce_calls;
    ++group->stats.ipc_allreduce_calls;
    group->stats.allreduce_elements += (uint64_t)send_count;
    group->stats.allreduce_bytes += (uint64_t)send_bytes;
  } else {
    ++group->stats.allgather_calls;
    ++group->stats.ipc_allgather_calls;
    group->stats.allgather_elements += (uint64_t)recv_count;
    group->stats.allgather_bytes += (uint64_t)recv_bytes;
  }
  return ET_TP_OK;
}

et_tp_status_t et_tp_allreduce_sum_f32(et_tp_group_t *group,
                                       const float *sendbuf, float *recvbuf,
                                       size_t count, uint64_t tag,
                                       int32_t timeout_ms) {
  return tp_collective_run(group, ET_TP_KIND_ALLREDUCE_SUM, sendbuf, recvbuf,
                           count, tag, timeout_ms);
}

et_tp_status_t et_tp_allgather_f32(et_tp_group_t *group, const float *sendbuf,
                                   size_t send_count, float *recvbuf,
                                   uint64_t tag, int32_t timeout_ms) {
  return tp_collective_run(group, ET_TP_KIND_ALLGATHER, sendbuf, recvbuf,
                           send_count, tag, timeout_ms);
}

/* =========================================================================
 * Internal Dense Matmul Helpers (Binary64 Inner Accumulation)
 * ========================================================================= */

/* Y [M, N] = X [M, K] @ W^T ([N, K]^T) + optional bias b [N] */
static et_tp_status_t tp_gemm_nt(size_t m, size_t n, size_t k, const float *x,
                                 const float *w, const float *b, float *y) {
  for (size_t i = 0u; i < m; ++i) {
    const float *xi = x + i * k;
    float *yi = y + i * n;
    for (size_t j = 0u; j < n; ++j) {
      const float *wj = w + j * k;
      double acc = (b != NULL) ? (double)b[j] : 0.0;
      for (size_t p = 0u; p < k; ++p)
        acc += (double)xi[p] * (double)wj[p];
      const float val = (float)acc;
      if (!isfinite(val))
        return ET_TP_ERR_NONFINITE;
      yi[j] = val;
    }
  }
  return ET_TP_OK;
}

/* dW [N, K] = dY^T ([M, N]^T) @ X [M, K] */
static et_tp_status_t tp_gemm_tn(size_t m, size_t n, size_t k, const float *dy,
                                 const float *x, float *dw) {
  for (size_t j = 0u; j < n; ++j) {
    float *dwj = dw + j * k;
    for (size_t p = 0u; p < k; ++p) {
      double acc = 0.0;
      for (size_t i = 0u; i < m; ++i)
        acc += (double)dy[i * n + j] * (double)x[i * k + p];
      const float val = (float)acc;
      if (!isfinite(val))
        return ET_TP_ERR_NONFINITE;
      dwj[p] = val;
    }
  }
  return ET_TP_OK;
}

/* dX [M, K] = dY [M, N] @ W [N, K] (if accumulate==1, adds into double acc) */
static et_tp_status_t tp_gemm_nn_f64_acc(size_t m, size_t n, size_t k,
                                         const float *dy, const float *w,
                                         double *dx_acc, int accumulate) {
  for (size_t i = 0u; i < m; ++i) {
    const float *dyi = dy + i * n;
    double *dxi = dx_acc + i * k;
    for (size_t p = 0u; p < k; ++p) {
      double acc = accumulate ? dxi[p] : 0.0;
      for (size_t j = 0u; j < n; ++j)
        acc += (double)dyi[j] * (double)w[j * k + p];
      if (!isfinite(acc))
        return ET_TP_ERR_NONFINITE;
      dxi[p] = acc;
    }
  }
  return ET_TP_OK;
}

static et_tp_status_t tp_narrow_f64_to_f32(const double *src, float *dst,
                                           size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    const float v = (float)src[i];
    if (!isfinite(v))
      return ET_TP_ERR_NONFINITE;
    dst[i] = v;
  }
  return ET_TP_OK;
}

/* =========================================================================
 * 1. ColumnParallelLinear
 * ========================================================================= */

et_tp_status_t et_tp_column_linear_forward_v1(
    et_tp_group_t *group, uint32_t rows, uint32_t in_features,
    uint32_t out_features, const float *x, const float *w_r, const float *b_r,
    float *y_r) {
  if (group == NULL)
    return ET_TP_ERR_INVALID_ARG;
  if (rows == 0u || in_features == 0u || out_features == 0u)
    return ET_TP_ERR_SHAPE;
  if ((out_features % group->world_size) != 0u)
    return ET_TP_ERR_DIVISIBILITY;

  const size_t out_local = (size_t)(out_features / group->world_size);
  size_t x_cnt = 0u, w_cnt = 0u, y_cnt = 0u;
  size_t x_bytes = 0u, w_bytes = 0u, b_bytes = 0u, y_bytes = 0u;
  if (!tp_mul_size(rows, in_features, &x_cnt) ||
      !tp_mul_size(out_local, in_features, &w_cnt) ||
      !tp_mul_size(rows, out_local, &y_cnt) ||
      !tp_mul_size(x_cnt, sizeof(float), &x_bytes) ||
      !tp_mul_size(w_cnt, sizeof(float), &w_bytes) ||
      !tp_mul_size(out_local, sizeof(float), &b_bytes) ||
      !tp_mul_size(y_cnt, sizeof(float), &y_bytes))
    return ET_TP_ERR_OVERFLOW;

  et_tp_status_t st = tp_check_f32_buf(x, x_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_r, w_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  if (b_r != NULL) {
    st = tp_check_f32_buf(b_r, out_local, 1);
    if (st != ET_TP_OK)
      return st;
  }
  st = tp_check_f32_out(y_r);
  if (st != ET_TP_OK)
    return st;

  if (tp_spans_overlap(y_r, y_bytes, x, x_bytes) ||
      tp_spans_overlap(y_r, y_bytes, w_r, w_bytes) ||
      (b_r != NULL && tp_spans_overlap(y_r, y_bytes, b_r, b_bytes)))
    return ET_TP_ERR_ALIAS;

  return tp_gemm_nt(rows, out_local, in_features, x, w_r, b_r, y_r);
}

et_tp_status_t et_tp_column_linear_backward_v1(
    et_tp_group_t *group, uint32_t rows, uint32_t in_features,
    uint32_t out_features, const float *x, const float *w_r, const float *dy_r,
    float *dx, float *dw_r, float *db_r, uint64_t tag, int32_t timeout_ms) {
  if (group == NULL)
    return ET_TP_ERR_INVALID_ARG;
  if (rows == 0u || in_features == 0u || out_features == 0u)
    return ET_TP_ERR_SHAPE;
  if ((out_features % group->world_size) != 0u)
    return ET_TP_ERR_DIVISIBILITY;

  const size_t out_local = (size_t)(out_features / group->world_size);
  size_t x_cnt = 0u, w_cnt = 0u, y_cnt = 0u;
  size_t x_bytes = 0u, w_bytes = 0u, y_bytes = 0u, b_bytes = 0u;
  if (!tp_mul_size(rows, in_features, &x_cnt) ||
      !tp_mul_size(out_local, in_features, &w_cnt) ||
      !tp_mul_size(rows, out_local, &y_cnt) ||
      !tp_mul_size(x_cnt, sizeof(float), &x_bytes) ||
      !tp_mul_size(w_cnt, sizeof(float), &w_bytes) ||
      !tp_mul_size(y_cnt, sizeof(float), &y_bytes) ||
      !tp_mul_size(out_local, sizeof(float), &b_bytes))
    return ET_TP_ERR_OVERFLOW;

  et_tp_status_t st = tp_check_f32_buf(x, x_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_r, w_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(dy_r, y_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dx);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dw_r);
  if (st != ET_TP_OK)
    return st;
  if (db_r != NULL) {
    st = tp_check_f32_out(db_r);
    if (st != ET_TP_OK)
      return st;
  }

  /* Disjointness between outputs and inputs/each other. */
  if (tp_spans_overlap(dx, x_bytes, x, x_bytes) ||
      tp_spans_overlap(dx, x_bytes, w_r, w_bytes) ||
      tp_spans_overlap(dx, x_bytes, dy_r, y_bytes) ||
      tp_spans_overlap(dw_r, w_bytes, x, x_bytes) ||
      tp_spans_overlap(dw_r, w_bytes, w_r, w_bytes) ||
      tp_spans_overlap(dw_r, w_bytes, dy_r, y_bytes) ||
      tp_spans_overlap(dx, x_bytes, dw_r, w_bytes))
    return ET_TP_ERR_ALIAS;
  if (db_r != NULL && (tp_spans_overlap(db_r, b_bytes, x, x_bytes) ||
                       tp_spans_overlap(db_r, b_bytes, w_r, w_bytes) ||
                       tp_spans_overlap(db_r, b_bytes, dy_r, y_bytes) ||
                       tp_spans_overlap(db_r, b_bytes, dx, x_bytes) ||
                       tp_spans_overlap(db_r, b_bytes, dw_r, w_bytes)))
    return ET_TP_ERR_ALIAS;

  double *dx_acc = (double *)malloc(x_cnt * sizeof(double));
  float *dx_local = (float *)malloc(x_bytes);
  if (dx_acc == NULL || dx_local == NULL) {
    free(dx_acc);
    free(dx_local);
    return ET_TP_ERR_ALLOC;
  }

  st = tp_gemm_nn_f64_acc(rows, out_local, in_features, dy_r, w_r, dx_acc, 0);
  if (st == ET_TP_OK)
    st = tp_narrow_f64_to_f32(dx_acc, dx_local, x_cnt);
  free(dx_acc);
  if (st != ET_TP_OK) {
    free(dx_local);
    return st;
  }

  st = et_tp_allreduce_sum_f32(group, dx_local, dx, x_cnt, tag, timeout_ms);
  free(dx_local);
  if (st != ET_TP_OK)
    return st;

  st = tp_gemm_tn(rows, out_local, in_features, dy_r, x, dw_r);
  if (st != ET_TP_OK)
    return st;

  if (db_r != NULL) {
    for (size_t j = 0u; j < out_local; ++j) {
      double acc = 0.0;
      for (size_t i = 0u; i < rows; ++i)
        acc += (double)dy_r[i * out_local + j];
      const float val = (float)acc;
      if (!isfinite(val))
        return ET_TP_ERR_NONFINITE;
      db_r[j] = val;
    }
  }
  return ET_TP_OK;
}

/* =========================================================================
 * 2. RowParallelLinear
 * ========================================================================= */

et_tp_status_t et_tp_row_linear_forward_v1(
    et_tp_group_t *group, uint32_t rows, uint32_t in_features,
    uint32_t out_features, const float *x_r, const float *w_r, const float *b,
    float *y, uint64_t tag, int32_t timeout_ms) {
  if (group == NULL)
    return ET_TP_ERR_INVALID_ARG;
  if (rows == 0u || in_features == 0u || out_features == 0u)
    return ET_TP_ERR_SHAPE;
  if ((in_features % group->world_size) != 0u)
    return ET_TP_ERR_DIVISIBILITY;

  const size_t in_local = (size_t)(in_features / group->world_size);
  size_t x_cnt = 0u, w_cnt = 0u, y_cnt = 0u;
  size_t x_bytes = 0u, w_bytes = 0u, b_bytes = 0u, y_bytes = 0u;
  if (!tp_mul_size(rows, in_local, &x_cnt) ||
      !tp_mul_size(out_features, in_local, &w_cnt) ||
      !tp_mul_size(rows, out_features, &y_cnt) ||
      !tp_mul_size(x_cnt, sizeof(float), &x_bytes) ||
      !tp_mul_size(w_cnt, sizeof(float), &w_bytes) ||
      !tp_mul_size(out_features, sizeof(float), &b_bytes) ||
      !tp_mul_size(y_cnt, sizeof(float), &y_bytes))
    return ET_TP_ERR_OVERFLOW;

  et_tp_status_t st = tp_check_f32_buf(x_r, x_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_r, w_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  if (b != NULL) {
    st = tp_check_f32_buf(b, out_features, 1);
    if (st != ET_TP_OK)
      return st;
  }
  st = tp_check_f32_out(y);
  if (st != ET_TP_OK)
    return st;

  if (tp_spans_overlap(y, y_bytes, x_r, x_bytes) ||
      tp_spans_overlap(y, y_bytes, w_r, w_bytes) ||
      (b != NULL && tp_spans_overlap(y, y_bytes, b, b_bytes)))
    return ET_TP_ERR_ALIAS;

  float *y_partial = (float *)malloc(y_bytes);
  if (y_partial == NULL)
    return ET_TP_ERR_ALLOC;

  st = tp_gemm_nt(rows, out_features, in_local, x_r, w_r, NULL, y_partial);
  if (st == ET_TP_OK)
    st = et_tp_allreduce_sum_f32(group, y_partial, y, y_cnt, tag, timeout_ms);
  free(y_partial);
  if (st != ET_TP_OK)
    return st;

  if (b != NULL) {
    for (size_t i = 0u; i < rows; ++i) {
      float *yi = y + i * out_features;
      for (size_t j = 0u; j < out_features; ++j) {
        const float val = (float)((double)yi[j] + (double)b[j]);
        if (!isfinite(val))
          return ET_TP_ERR_NONFINITE;
        yi[j] = val;
      }
    }
  }
  return ET_TP_OK;
}

et_tp_status_t et_tp_row_linear_backward_v1(
    et_tp_group_t *group, uint32_t rows, uint32_t in_features,
    uint32_t out_features, const float *x_r, const float *w_r, const float *dy,
    float *dx_r, float *dw_r, float *db) {
  if (group == NULL)
    return ET_TP_ERR_INVALID_ARG;
  if (rows == 0u || in_features == 0u || out_features == 0u)
    return ET_TP_ERR_SHAPE;
  if ((in_features % group->world_size) != 0u)
    return ET_TP_ERR_DIVISIBILITY;

  const size_t in_local = (size_t)(in_features / group->world_size);
  size_t x_cnt = 0u, w_cnt = 0u, y_cnt = 0u;
  size_t x_bytes = 0u, w_bytes = 0u, y_bytes = 0u, b_bytes = 0u;
  if (!tp_mul_size(rows, in_local, &x_cnt) ||
      !tp_mul_size(out_features, in_local, &w_cnt) ||
      !tp_mul_size(rows, out_features, &y_cnt) ||
      !tp_mul_size(x_cnt, sizeof(float), &x_bytes) ||
      !tp_mul_size(w_cnt, sizeof(float), &w_bytes) ||
      !tp_mul_size(y_cnt, sizeof(float), &y_bytes) ||
      !tp_mul_size(out_features, sizeof(float), &b_bytes))
    return ET_TP_ERR_OVERFLOW;

  et_tp_status_t st = tp_check_f32_buf(x_r, x_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_r, w_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(dy, y_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dx_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dw_r);
  if (st != ET_TP_OK)
    return st;
  if (db != NULL) {
    st = tp_check_f32_out(db);
    if (st != ET_TP_OK)
      return st;
  }

  if (tp_spans_overlap(dx_r, x_bytes, x_r, x_bytes) ||
      tp_spans_overlap(dx_r, x_bytes, w_r, w_bytes) ||
      tp_spans_overlap(dx_r, x_bytes, dy, y_bytes) ||
      tp_spans_overlap(dw_r, w_bytes, x_r, x_bytes) ||
      tp_spans_overlap(dw_r, w_bytes, w_r, w_bytes) ||
      tp_spans_overlap(dw_r, w_bytes, dy, y_bytes) ||
      tp_spans_overlap(dx_r, x_bytes, dw_r, w_bytes))
    return ET_TP_ERR_ALIAS;
  if (db != NULL && (tp_spans_overlap(db, b_bytes, x_r, x_bytes) ||
                     tp_spans_overlap(db, b_bytes, w_r, w_bytes) ||
                     tp_spans_overlap(db, b_bytes, dy, y_bytes) ||
                     tp_spans_overlap(db, b_bytes, dx_r, x_bytes) ||
                     tp_spans_overlap(db, b_bytes, dw_r, w_bytes)))
    return ET_TP_ERR_ALIAS;

  double *dx_acc = (double *)malloc(x_cnt * sizeof(double));
  if (dx_acc == NULL)
    return ET_TP_ERR_ALLOC;

  st = tp_gemm_nn_f64_acc(rows, out_features, in_local, dy, w_r, dx_acc, 0);
  if (st == ET_TP_OK)
    st = tp_narrow_f64_to_f32(dx_acc, dx_r, x_cnt);
  free(dx_acc);
  if (st != ET_TP_OK)
    return st;

  st = tp_gemm_tn(rows, out_features, in_local, dy, x_r, dw_r);
  if (st != ET_TP_OK)
    return st;

  if (db != NULL) {
    for (size_t j = 0u; j < out_features; ++j) {
      double acc = 0.0;
      for (size_t i = 0u; i < rows; ++i)
        acc += (double)dy[i * out_features + j];
      const float val = (float)acc;
      if (!isfinite(val))
        return ET_TP_ERR_NONFINITE;
      db[j] = val;
    }
  }
  return ET_TP_OK;
}

/* =========================================================================
 * 3. TP-Sharded SwiGLU MLP Block (1 Fwd AllReduce, 1 Bwd AllReduce)
 * ========================================================================= */

et_tp_status_t et_tp_swiglu_validate_config(const et_tp_swiglu_config_t *cfg,
                                            uint32_t world_size) {
  if (cfg == NULL || !tp_world_valid(world_size))
    return ET_TP_ERR_INVALID_ARG;
  if (cfg->tokens == 0u || cfg->d_model == 0u || cfg->d_ff == 0u)
    return ET_TP_ERR_SHAPE;
  if ((cfg->d_ff % world_size) != 0u)
    return ET_TP_ERR_DIVISIBILITY;
  return ET_TP_OK;
}

et_tp_status_t et_tp_swiglu_forward_v1(
    et_tp_group_t *group, const et_tp_swiglu_config_t *cfg, const float *x,
    const float *w_gate_r, const float *w_up_r, const float *w_down_r,
    float *gate_cache_r, float *up_cache_r, float *y, uint64_t tag,
    int32_t timeout_ms) {
  if (group == NULL)
    return ET_TP_ERR_INVALID_ARG;
  et_tp_status_t st = et_tp_swiglu_validate_config(cfg, group->world_size);
  if (st != ET_TP_OK)
    return st;

  const size_t t = cfg->tokens;
  const size_t d = cfg->d_model;
  const size_t f_loc = (size_t)(cfg->d_ff / group->world_size);
  size_t xd_cnt = 0u, wf_cnt = 0u, tf_cnt = 0u;
  size_t xd_bytes = 0u, wf_bytes = 0u, tf_bytes = 0u;
  if (!tp_mul_size(t, d, &xd_cnt) || !tp_mul_size(f_loc, d, &wf_cnt) ||
      !tp_mul_size(t, f_loc, &tf_cnt) ||
      !tp_mul_size(xd_cnt, sizeof(float), &xd_bytes) ||
      !tp_mul_size(wf_cnt, sizeof(float), &wf_bytes) ||
      !tp_mul_size(tf_cnt, sizeof(float), &tf_bytes))
    return ET_TP_ERR_OVERFLOW;

  st = tp_check_f32_buf(x, xd_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_gate_r, wf_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_up_r, wf_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_down_r, wf_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(gate_cache_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(up_cache_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(y);
  if (st != ET_TP_OK)
    return st;

  if (tp_spans_overlap(gate_cache_r, tf_bytes, up_cache_r, tf_bytes) ||
      tp_spans_overlap(gate_cache_r, tf_bytes, y, xd_bytes) ||
      tp_spans_overlap(up_cache_r, tf_bytes, y, xd_bytes) ||
      tp_spans_overlap(y, xd_bytes, x, xd_bytes) ||
      tp_spans_overlap(y, xd_bytes, w_gate_r, wf_bytes) ||
      tp_spans_overlap(y, xd_bytes, w_up_r, wf_bytes) ||
      tp_spans_overlap(y, xd_bytes, w_down_r, wf_bytes) ||
      tp_spans_overlap(gate_cache_r, tf_bytes, x, xd_bytes) ||
      tp_spans_overlap(up_cache_r, tf_bytes, x, xd_bytes))
    return ET_TP_ERR_ALIAS;

  st = tp_gemm_nt(t, f_loc, d, x, w_gate_r, NULL, gate_cache_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_gemm_nt(t, f_loc, d, x, w_up_r, NULL, up_cache_r);
  if (st != ET_TP_OK)
    return st;

  float *act_r = (float *)malloc(tf_bytes);
  float *y_partial = (float *)malloc(xd_bytes);
  if (act_r == NULL || y_partial == NULL) {
    free(act_r);
    free(y_partial);
    return ET_TP_ERR_ALLOC;
  }

  for (size_t i = 0u; i < tf_cnt; ++i) {
    const double g = (double)gate_cache_r[i];
    const double u = (double)up_cache_r[i];
    const double silu = g * tp_sigmoid_f64(g);
    const float val = (float)(silu * u);
    if (!isfinite(val)) {
      free(act_r);
      free(y_partial);
      return ET_TP_ERR_NONFINITE;
    }
    act_r[i] = val;
  }

  st = tp_gemm_nt(t, d, f_loc, act_r, w_down_r, NULL, y_partial);
  free(act_r);
  if (st == ET_TP_OK)
    st = et_tp_allreduce_sum_f32(group, y_partial, y, xd_cnt, tag, timeout_ms);
  free(y_partial);
  return st;
}

et_tp_status_t et_tp_swiglu_backward_v1(
    et_tp_group_t *group, const et_tp_swiglu_config_t *cfg, const float *x,
    const float *w_gate_r, const float *w_up_r, const float *w_down_r,
    const float *gate_cache_r, const float *up_cache_r, const float *dy,
    float *dx, float *dw_gate_r, float *dw_up_r, float *dw_down_r, uint64_t tag,
    int32_t timeout_ms) {
  if (group == NULL)
    return ET_TP_ERR_INVALID_ARG;
  et_tp_status_t st = et_tp_swiglu_validate_config(cfg, group->world_size);
  if (st != ET_TP_OK)
    return st;

  const size_t t = cfg->tokens;
  const size_t d = cfg->d_model;
  const size_t f_loc = (size_t)(cfg->d_ff / group->world_size);
  size_t xd_cnt = 0u, wf_cnt = 0u, tf_cnt = 0u;
  size_t xd_bytes = 0u, wf_bytes = 0u, tf_bytes = 0u;
  if (!tp_mul_size(t, d, &xd_cnt) || !tp_mul_size(f_loc, d, &wf_cnt) ||
      !tp_mul_size(t, f_loc, &tf_cnt) ||
      !tp_mul_size(xd_cnt, sizeof(float), &xd_bytes) ||
      !tp_mul_size(wf_cnt, sizeof(float), &wf_bytes) ||
      !tp_mul_size(tf_cnt, sizeof(float), &tf_bytes))
    return ET_TP_ERR_OVERFLOW;

  st = tp_check_f32_buf(x, xd_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_gate_r, wf_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_up_r, wf_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_down_r, wf_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(gate_cache_r, tf_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(up_cache_r, tf_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(dy, xd_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dx);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dw_gate_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dw_up_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dw_down_r);
  if (st != ET_TP_OK)
    return st;

  if (tp_spans_overlap(dx, xd_bytes, x, xd_bytes) ||
      tp_spans_overlap(dx, xd_bytes, dy, xd_bytes) ||
      tp_spans_overlap(dx, xd_bytes, dw_gate_r, wf_bytes) ||
      tp_spans_overlap(dx, xd_bytes, dw_up_r, wf_bytes) ||
      tp_spans_overlap(dx, xd_bytes, dw_down_r, wf_bytes) ||
      tp_spans_overlap(dw_gate_r, wf_bytes, dw_up_r, wf_bytes) ||
      tp_spans_overlap(dw_gate_r, wf_bytes, dw_down_r, wf_bytes) ||
      tp_spans_overlap(dw_up_r, wf_bytes, dw_down_r, wf_bytes))
    return ET_TP_ERR_ALIAS;

  float *act_r = (float *)malloc(tf_bytes);
  double *dact_acc = (double *)malloc(tf_cnt * sizeof(double));
  float *dgate_r = (float *)malloc(tf_bytes);
  float *dup_r = (float *)malloc(tf_bytes);
  double *dx_acc = (double *)malloc(xd_cnt * sizeof(double));
  float *dx_local = (float *)malloc(xd_bytes);
  if (act_r == NULL || dact_acc == NULL || dgate_r == NULL || dup_r == NULL ||
      dx_acc == NULL || dx_local == NULL) {
    free(act_r);
    free(dact_acc);
    free(dgate_r);
    free(dup_r);
    free(dx_acc);
    free(dx_local);
    return ET_TP_ERR_ALLOC;
  }

  for (size_t i = 0u; i < tf_cnt; ++i) {
    const double g = (double)gate_cache_r[i];
    const double u = (double)up_cache_r[i];
    const float val = (float)(g * tp_sigmoid_f64(g) * u);
    if (!isfinite(val)) {
      st = ET_TP_ERR_NONFINITE;
      goto cleanup;
    }
    act_r[i] = val;
  }

  /* RowParallel(w_down_r) backward (0 AllReduces). */
  st = tp_gemm_tn(t, d, f_loc, dy, act_r, dw_down_r);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_gemm_nn_f64_acc(t, d, f_loc, dy, w_down_r, dact_acc, 0);
  if (st != ET_TP_OK)
    goto cleanup;

  /* Local SwiGLU elementwise backward. */
  for (size_t i = 0u; i < tf_cnt; ++i) {
    const double g = (double)gate_cache_r[i];
    const double u = (double)up_cache_r[i];
    const double da = dact_acc[i];
    const double sig = tp_sigmoid_f64(g);
    const double silu = g * sig;
    const double dsilu = sig * (1.0 + g * (1.0 - sig));
    const float dg = (float)(da * u * dsilu);
    const float du = (float)(da * silu);
    if (!isfinite(dg) || !isfinite(du)) {
      st = ET_TP_ERR_NONFINITE;
      goto cleanup;
    }
    dgate_r[i] = dg;
    dup_r[i] = du;
  }

  /* Local weight grads for w_gate_r and w_up_r. */
  st = tp_gemm_tn(t, f_loc, d, dgate_r, x, dw_gate_r);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_gemm_tn(t, f_loc, d, dup_r, x, dw_up_r);
  if (st != ET_TP_OK)
    goto cleanup;

  /* Fused dX_local = dgate_r @ w_gate_r + dup_r @ w_up_r, then 1 AllReduce! */
  st = tp_gemm_nn_f64_acc(t, f_loc, d, dgate_r, w_gate_r, dx_acc, 0);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_gemm_nn_f64_acc(t, f_loc, d, dup_r, w_up_r, dx_acc, 1);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_narrow_f64_to_f32(dx_acc, dx_local, xd_cnt);
  if (st != ET_TP_OK)
    goto cleanup;

  st = et_tp_allreduce_sum_f32(group, dx_local, dx, xd_cnt, tag, timeout_ms);

cleanup:
  free(act_r);
  free(dact_acc);
  free(dgate_r);
  free(dup_r);
  free(dx_acc);
  free(dx_local);
  return st;
}

/* =========================================================================
 * 4. TP-Sharded Causal GQA Attention Block (1 Fwd AllReduce, 1 Bwd AllReduce)
 * ========================================================================= */

et_tp_status_t et_tp_gqa_validate_config(const et_tp_gqa_config_t *cfg,
                                         uint32_t world_size) {
  if (cfg == NULL || !tp_world_valid(world_size))
    return ET_TP_ERR_INVALID_ARG;
  if (cfg->batch == 0u || cfg->seq_len == 0u || cfg->d_model == 0u ||
      cfg->num_q_heads == 0u || cfg->num_kv_heads == 0u || cfg->head_dim == 0u)
    return ET_TP_ERR_SHAPE;
  if ((uint64_t)cfg->num_q_heads * (uint64_t)cfg->head_dim !=
      (uint64_t)cfg->d_model)
    return ET_TP_ERR_SHAPE;
  if ((cfg->num_q_heads % world_size) != 0u ||
      (cfg->num_kv_heads % world_size) != 0u ||
      (cfg->num_q_heads % cfg->num_kv_heads) != 0u)
    return ET_TP_ERR_DIVISIBILITY;
  return ET_TP_OK;
}

et_tp_status_t et_tp_gqa_forward_v1(
    et_tp_group_t *group, const et_tp_gqa_config_t *cfg, const float *x,
    const float *w_q_r, const float *w_k_r, const float *w_v_r,
    const float *w_o_r, float *q_cache_r, float *k_cache_r, float *v_cache_r,
    float *attn_cache_r, float *y, uint64_t tag, int32_t timeout_ms) {
  if (group == NULL)
    return ET_TP_ERR_INVALID_ARG;
  et_tp_status_t st = et_tp_gqa_validate_config(cfg, group->world_size);
  if (st != ET_TP_OK)
    return st;

  const size_t B = cfg->batch;
  const size_t T = cfg->seq_len;
  const size_t D = cfg->d_model;
  const size_t Hq_loc = (size_t)(cfg->num_q_heads / group->world_size);
  const size_t Hkv_loc = (size_t)(cfg->num_kv_heads / group->world_size);
  const size_t Dh = cfg->head_dim;
  const size_t group_ratio = Hq_loc / Hkv_loc;
  const size_t q_dim_loc = Hq_loc * Dh;
  const size_t kv_dim_loc = Hkv_loc * Dh;

  size_t M = 0u, xd_cnt = 0u, q_cnt = 0u, kv_cnt = 0u;
  size_t wq_cnt = 0u, wkv_cnt = 0u;
  size_t xd_bytes = 0u, q_bytes = 0u, kv_bytes = 0u, wq_bytes = 0u,
         wkv_bytes = 0u;
  if (!tp_mul_size(B, T, &M) || !tp_mul_size(M, D, &xd_cnt) ||
      !tp_mul_size(M, q_dim_loc, &q_cnt) ||
      !tp_mul_size(M, kv_dim_loc, &kv_cnt) ||
      !tp_mul_size(q_dim_loc, D, &wq_cnt) ||
      !tp_mul_size(kv_dim_loc, D, &wkv_cnt) ||
      !tp_mul_size(xd_cnt, sizeof(float), &xd_bytes) ||
      !tp_mul_size(q_cnt, sizeof(float), &q_bytes) ||
      !tp_mul_size(kv_cnt, sizeof(float), &kv_bytes) ||
      !tp_mul_size(wq_cnt, sizeof(float), &wq_bytes) ||
      !tp_mul_size(wkv_cnt, sizeof(float), &wkv_bytes))
    return ET_TP_ERR_OVERFLOW;

  st = tp_check_f32_buf(x, xd_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_q_r, wq_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_k_r, wkv_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_v_r, wkv_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_o_r, wq_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(q_cache_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(k_cache_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(v_cache_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(attn_cache_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(y);
  if (st != ET_TP_OK)
    return st;

  if (tp_spans_overlap(y, xd_bytes, x, xd_bytes) ||
      tp_spans_overlap(y, xd_bytes, w_q_r, wq_bytes) ||
      tp_spans_overlap(y, xd_bytes, w_k_r, wkv_bytes) ||
      tp_spans_overlap(y, xd_bytes, w_v_r, wkv_bytes) ||
      tp_spans_overlap(y, xd_bytes, w_o_r, wq_bytes) ||
      tp_spans_overlap(q_cache_r, q_bytes, k_cache_r, kv_bytes) ||
      tp_spans_overlap(q_cache_r, q_bytes, v_cache_r, kv_bytes) ||
      tp_spans_overlap(q_cache_r, q_bytes, attn_cache_r, q_bytes) ||
      tp_spans_overlap(k_cache_r, kv_bytes, v_cache_r, kv_bytes) ||
      tp_spans_overlap(attn_cache_r, q_bytes, y, xd_bytes))
    return ET_TP_ERR_ALIAS;

  /* 1. Local head-sharded Q, K, V projections (0 AllReduces). */
  st = tp_gemm_nt(M, q_dim_loc, D, x, w_q_r, NULL, q_cache_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_gemm_nt(M, kv_dim_loc, D, x, w_k_r, NULL, k_cache_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_gemm_nt(M, kv_dim_loc, D, x, w_v_r, NULL, v_cache_r);
  if (st != ET_TP_OK)
    return st;

  /* 2. Local Causal GQA SDPA (0 AllReduces). */
  double *scores = (double *)malloc(T * sizeof(double));
  double *probs = (double *)malloc(T * sizeof(double));
  float *y_partial = (float *)malloc(xd_bytes);
  if (scores == NULL || probs == NULL || y_partial == NULL) {
    free(scores);
    free(probs);
    free(y_partial);
    return ET_TP_ERR_ALLOC;
  }

  const double scale = 1.0 / sqrt((double)Dh);
  for (size_t b = 0u; b < B; ++b) {
    for (size_t hq = 0u; hq < Hq_loc; ++hq) {
      const size_t hkv = hq / group_ratio;
      for (size_t i = 0u; i < T; ++i) {
        const float *q_vec =
            q_cache_r + (b * T + i) * q_dim_loc + hq * Dh;
        double max_s = -INFINITY;
        for (size_t j = 0u; j <= i; ++j) {
          const float *k_vec =
              k_cache_r + (b * T + j) * kv_dim_loc + hkv * Dh;
          double dot = 0.0;
          for (size_t d = 0u; d < Dh; ++d)
            dot += (double)q_vec[d] * (double)k_vec[d];
          scores[j] = dot * scale;
          if (scores[j] > max_s)
            max_s = scores[j];
        }
        double denom = 0.0;
        for (size_t j = 0u; j <= i; ++j) {
          probs[j] = exp(scores[j] - max_s);
          denom += probs[j];
        }
        if (!(denom > 0.0) || !isfinite(denom)) {
          st = ET_TP_ERR_NONFINITE;
          goto cleanup;
        }
        for (size_t j = 0u; j <= i; ++j)
          probs[j] /= denom;

        float *out_vec =
            attn_cache_r + (b * T + i) * q_dim_loc + hq * Dh;
        for (size_t d = 0u; d < Dh; ++d) {
          double acc = 0.0;
          for (size_t j = 0u; j <= i; ++j) {
            const float *v_vec =
                v_cache_r + (b * T + j) * kv_dim_loc + hkv * Dh;
            acc += probs[j] * (double)v_vec[d];
          }
          const float val = (float)acc;
          if (!isfinite(val)) {
            st = ET_TP_ERR_NONFINITE;
            goto cleanup;
          }
          out_vec[d] = val;
        }
      }
    }
  }

  /* 3. RowParallel(w_o_r): local matmul + 1 AllReduce! */
  st = tp_gemm_nt(M, D, q_dim_loc, attn_cache_r, w_o_r, NULL, y_partial);
  if (st == ET_TP_OK)
    st = et_tp_allreduce_sum_f32(group, y_partial, y, xd_cnt, tag, timeout_ms);

cleanup:
  free(scores);
  free(probs);
  free(y_partial);
  return st;
}

et_tp_status_t et_tp_gqa_backward_v1(
    et_tp_group_t *group, const et_tp_gqa_config_t *cfg, const float *x,
    const float *w_q_r, const float *w_k_r, const float *w_v_r,
    const float *w_o_r, const float *q_cache_r, const float *k_cache_r,
    const float *v_cache_r, const float *attn_cache_r, const float *dy,
    float *dx, float *dw_q_r, float *dw_k_r, float *dw_v_r, float *dw_o_r,
    uint64_t tag, int32_t timeout_ms) {
  if (group == NULL)
    return ET_TP_ERR_INVALID_ARG;
  et_tp_status_t st = et_tp_gqa_validate_config(cfg, group->world_size);
  if (st != ET_TP_OK)
    return st;

  const size_t B = cfg->batch;
  const size_t T = cfg->seq_len;
  const size_t D = cfg->d_model;
  const size_t Hq_loc = (size_t)(cfg->num_q_heads / group->world_size);
  const size_t Hkv_loc = (size_t)(cfg->num_kv_heads / group->world_size);
  const size_t Dh = cfg->head_dim;
  const size_t group_ratio = Hq_loc / Hkv_loc;
  const size_t q_dim_loc = Hq_loc * Dh;
  const size_t kv_dim_loc = Hkv_loc * Dh;

  size_t M = 0u, xd_cnt = 0u, q_cnt = 0u, kv_cnt = 0u;
  size_t wq_cnt = 0u, wkv_cnt = 0u;
  size_t xd_bytes = 0u, q_bytes = 0u, kv_bytes = 0u, wq_bytes = 0u,
         wkv_bytes = 0u;
  if (!tp_mul_size(B, T, &M) || !tp_mul_size(M, D, &xd_cnt) ||
      !tp_mul_size(M, q_dim_loc, &q_cnt) ||
      !tp_mul_size(M, kv_dim_loc, &kv_cnt) ||
      !tp_mul_size(q_dim_loc, D, &wq_cnt) ||
      !tp_mul_size(kv_dim_loc, D, &wkv_cnt) ||
      !tp_mul_size(xd_cnt, sizeof(float), &xd_bytes) ||
      !tp_mul_size(q_cnt, sizeof(float), &q_bytes) ||
      !tp_mul_size(kv_cnt, sizeof(float), &kv_bytes) ||
      !tp_mul_size(wq_cnt, sizeof(float), &wq_bytes) ||
      !tp_mul_size(wkv_cnt, sizeof(float), &wkv_bytes))
    return ET_TP_ERR_OVERFLOW;

  st = tp_check_f32_buf(x, xd_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_q_r, wq_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_k_r, wkv_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_v_r, wkv_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(w_o_r, wq_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(q_cache_r, q_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(k_cache_r, kv_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(v_cache_r, kv_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(attn_cache_r, q_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(dy, xd_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dx);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dw_q_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dw_k_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dw_v_r);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dw_o_r);
  if (st != ET_TP_OK)
    return st;

  if (tp_spans_overlap(dx, xd_bytes, x, xd_bytes) ||
      tp_spans_overlap(dx, xd_bytes, dy, xd_bytes) ||
      tp_spans_overlap(dw_q_r, wq_bytes, dw_o_r, wq_bytes) ||
      tp_spans_overlap(dw_k_r, wkv_bytes, dw_v_r, wkv_bytes) ||
      tp_spans_overlap(dw_q_r, wq_bytes, dw_k_r, wkv_bytes))
    return ET_TP_ERR_ALIAS;

  double *dattn_acc = (double *)malloc(q_cnt * sizeof(double));
  double *dq_acc = (double *)calloc(q_cnt, sizeof(double));
  double *dk_acc = (double *)calloc(kv_cnt, sizeof(double));
  double *dv_acc = (double *)calloc(kv_cnt, sizeof(double));
  float *dq_r = (float *)malloc(q_bytes);
  float *dk_r = (float *)malloc(kv_bytes);
  float *dv_r = (float *)malloc(kv_bytes);
  double *scores = (double *)malloc(T * sizeof(double));
  double *probs = (double *)malloc(T * sizeof(double));
  double *dp = (double *)malloc(T * sizeof(double));
  double *dx_acc = (double *)malloc(xd_cnt * sizeof(double));
  float *dx_local = (float *)malloc(xd_bytes);
  if (dattn_acc == NULL || dq_acc == NULL || dk_acc == NULL || dv_acc == NULL ||
      dq_r == NULL || dk_r == NULL || dv_r == NULL || scores == NULL ||
      probs == NULL || dp == NULL || dx_acc == NULL || dx_local == NULL) {
    st = ET_TP_ERR_ALLOC;
    goto cleanup;
  }

  /* 1. RowParallel(w_o_r) backward (0 AllReduces). */
  st = tp_gemm_tn(M, D, q_dim_loc, dy, attn_cache_r, dw_o_r);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_gemm_nn_f64_acc(M, D, q_dim_loc, dy, w_o_r, dattn_acc, 0);
  if (st != ET_TP_OK)
    goto cleanup;

  /* 2. Local Causal GQA SDPA backward in binary64. */
  const double scale = 1.0 / sqrt((double)Dh);
  for (size_t b = 0u; b < B; ++b) {
    for (size_t hq = 0u; hq < Hq_loc; ++hq) {
      const size_t hkv = hq / group_ratio;
      for (size_t i = 0u; i < T; ++i) {
        const float *q_vec =
            q_cache_r + (b * T + i) * q_dim_loc + hq * Dh;
        const double *do_vec =
            dattn_acc + (b * T + i) * q_dim_loc + hq * Dh;
        double max_s = -INFINITY;
        for (size_t j = 0u; j <= i; ++j) {
          const float *k_vec =
              k_cache_r + (b * T + j) * kv_dim_loc + hkv * Dh;
          double dot = 0.0;
          for (size_t d = 0u; d < Dh; ++d)
            dot += (double)q_vec[d] * (double)k_vec[d];
          scores[j] = dot * scale;
          if (scores[j] > max_s)
            max_s = scores[j];
        }
        double denom = 0.0;
        for (size_t j = 0u; j <= i; ++j) {
          probs[j] = exp(scores[j] - max_s);
          denom += probs[j];
        }
        if (!(denom > 0.0) || !isfinite(denom)) {
          st = ET_TP_ERR_NONFINITE;
          goto cleanup;
        }
        for (size_t j = 0u; j <= i; ++j)
          probs[j] /= denom;

        double p_dot_dp = 0.0;
        for (size_t j = 0u; j <= i; ++j) {
          const float *v_vec =
              v_cache_r + (b * T + j) * kv_dim_loc + hkv * Dh;
          double *dv_vec =
              dv_acc + (b * T + j) * kv_dim_loc + hkv * Dh;
          double acc_dp = 0.0;
          for (size_t d = 0u; d < Dh; ++d) {
            acc_dp += do_vec[d] * (double)v_vec[d];
            dv_vec[d] += probs[j] * do_vec[d];
          }
          dp[j] = acc_dp;
          p_dot_dp += probs[j] * acc_dp;
        }

        double *dq_vec = dq_acc + (b * T + i) * q_dim_loc + hq * Dh;
        for (size_t j = 0u; j <= i; ++j) {
          const double ds = probs[j] * (dp[j] - p_dot_dp) * scale;
          const float *k_vec =
              k_cache_r + (b * T + j) * kv_dim_loc + hkv * Dh;
          double *dk_vec =
              dk_acc + (b * T + j) * kv_dim_loc + hkv * Dh;
          for (size_t d = 0u; d < Dh; ++d) {
            dq_vec[d] += ds * (double)k_vec[d];
            dk_vec[d] += ds * (double)q_vec[d];
          }
        }
      }
    }
  }

  st = tp_narrow_f64_to_f32(dq_acc, dq_r, q_cnt);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_narrow_f64_to_f32(dk_acc, dk_r, kv_cnt);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_narrow_f64_to_f32(dv_acc, dv_r, kv_cnt);
  if (st != ET_TP_OK)
    goto cleanup;

  /* 3. Local weight grads for w_q_r, w_k_r, w_v_r. */
  st = tp_gemm_tn(M, q_dim_loc, D, dq_r, x, dw_q_r);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_gemm_tn(M, kv_dim_loc, D, dk_r, x, dw_k_r);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_gemm_tn(M, kv_dim_loc, D, dv_r, x, dw_v_r);
  if (st != ET_TP_OK)
    goto cleanup;

  /* 4. Fused dX_local = dQ_r @ w_q_r + dK_r @ w_k_r + dV_r @ w_v_r + 1 AllReduce! */
  st = tp_gemm_nn_f64_acc(M, q_dim_loc, D, dq_r, w_q_r, dx_acc, 0);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_gemm_nn_f64_acc(M, kv_dim_loc, D, dk_r, w_k_r, dx_acc, 1);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_gemm_nn_f64_acc(M, kv_dim_loc, D, dv_r, w_v_r, dx_acc, 1);
  if (st != ET_TP_OK)
    goto cleanup;
  st = tp_narrow_f64_to_f32(dx_acc, dx_local, xd_cnt);
  if (st != ET_TP_OK)
    goto cleanup;

  st = et_tp_allreduce_sum_f32(group, dx_local, dx, xd_cnt, tag, timeout_ms);

cleanup:
  free(dattn_acc);
  free(dq_acc);
  free(dk_acc);
  free(dv_acc);
  free(dq_r);
  free(dk_r);
  free(dv_r);
  free(scores);
  free(probs);
  free(dp);
  free(dx_acc);
  free(dx_local);
  return st;
}

/* =========================================================================
 * 5. RMSNorm & Full TP Transformer Block (2 Fwd + 2 Bwd AllReduces)
 * ========================================================================= */

et_tp_status_t et_tp_rmsnorm_forward_v1(uint32_t rows, uint32_t d_model,
                                        float eps, const float *x,
                                        const float *gamma, float *y) {
  if (rows == 0u || d_model == 0u)
    return ET_TP_ERR_SHAPE;
  if (!isfinite(eps) || !(eps > 0.0f))
    return ET_TP_ERR_INVALID_ARG;

  size_t xd_cnt = 0u, xd_bytes = 0u, g_bytes = 0u;
  if (!tp_mul_size(rows, d_model, &xd_cnt) ||
      !tp_mul_size(xd_cnt, sizeof(float), &xd_bytes) ||
      !tp_mul_size(d_model, sizeof(float), &g_bytes))
    return ET_TP_ERR_OVERFLOW;

  et_tp_status_t st = tp_check_f32_buf(x, xd_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(gamma, d_model, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(y);
  if (st != ET_TP_OK)
    return st;

  if (tp_spans_overlap(y, xd_bytes, x, xd_bytes) ||
      tp_spans_overlap(y, xd_bytes, gamma, g_bytes))
    return ET_TP_ERR_ALIAS;

  const double inv_d = 1.0 / (double)d_model;
  for (size_t i = 0u; i < rows; ++i) {
    const float *xi = x + i * d_model;
    float *yi = y + i * d_model;
    double sq = 0.0;
    for (size_t d = 0u; d < d_model; ++d)
      sq += (double)xi[d] * (double)xi[d];
    const double inv_rms = 1.0 / sqrt(sq * inv_d + (double)eps);
    for (size_t d = 0u; d < d_model; ++d) {
      const float val = (float)((double)xi[d] * inv_rms * (double)gamma[d]);
      if (!isfinite(val))
        return ET_TP_ERR_NONFINITE;
      yi[d] = val;
    }
  }
  return ET_TP_OK;
}

et_tp_status_t et_tp_rmsnorm_backward_v1(uint32_t rows, uint32_t d_model,
                                         float eps, const float *x,
                                         const float *gamma, const float *dy,
                                         float *dx, float *dgamma) {
  if (rows == 0u || d_model == 0u)
    return ET_TP_ERR_SHAPE;
  if (!isfinite(eps) || !(eps > 0.0f))
    return ET_TP_ERR_INVALID_ARG;

  size_t xd_cnt = 0u, xd_bytes = 0u, g_bytes = 0u;
  if (!tp_mul_size(rows, d_model, &xd_cnt) ||
      !tp_mul_size(xd_cnt, sizeof(float), &xd_bytes) ||
      !tp_mul_size(d_model, sizeof(float), &g_bytes))
    return ET_TP_ERR_OVERFLOW;

  et_tp_status_t st = tp_check_f32_buf(x, xd_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(gamma, d_model, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_buf(dy, xd_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dx);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dgamma);
  if (st != ET_TP_OK)
    return st;

  if (tp_spans_overlap(dx, xd_bytes, x, xd_bytes) ||
      tp_spans_overlap(dx, xd_bytes, dy, xd_bytes) ||
      tp_spans_overlap(dx, xd_bytes, dgamma, g_bytes) ||
      tp_spans_overlap(dgamma, g_bytes, gamma, g_bytes))
    return ET_TP_ERR_ALIAS;

  double *dg_acc = (double *)calloc(d_model, sizeof(double));
  if (dg_acc == NULL)
    return ET_TP_ERR_ALLOC;

  const double inv_d = 1.0 / (double)d_model;
  for (size_t i = 0u; i < rows; ++i) {
    const float *xi = x + i * d_model;
    const float *dyi = dy + i * d_model;
    float *dxi = dx + i * d_model;
    double sq = 0.0;
    for (size_t d = 0u; d < d_model; ++d)
      sq += (double)xi[d] * (double)xi[d];
    const double mean_sq = sq * inv_d + (double)eps;
    const double inv_rms = 1.0 / sqrt(mean_sq);
    const double inv_rms2 = 1.0 / mean_sq;

    double dot = 0.0;
    for (size_t d = 0u; d < d_model; ++d) {
      const double xv = (double)xi[d];
      const double dyv = (double)dyi[d];
      const double gv = (double)gamma[d];
      dg_acc[d] += dyv * xv * inv_rms;
      dot += dyv * gv * xv;
    }
    const double scale = dot * inv_d * inv_rms2;
    for (size_t d = 0u; d < d_model; ++d) {
      const double g_in = (double)dyi[d] * (double)gamma[d];
      const float val = (float)(inv_rms * (g_in - (double)xi[d] * scale));
      if (!isfinite(val)) {
        free(dg_acc);
        return ET_TP_ERR_NONFINITE;
      }
      dxi[d] = val;
    }
  }

  st = tp_narrow_f64_to_f32(dg_acc, dgamma, d_model);
  free(dg_acc);
  return st;
}

et_tp_status_t et_tp_block_validate_config(const et_tp_block_config_t *cfg,
                                           uint32_t world_size) {
  if (cfg == NULL || !tp_world_valid(world_size))
    return ET_TP_ERR_INVALID_ARG;
  if (!isfinite(cfg->rms_eps) || !(cfg->rms_eps > 0.0f))
    return ET_TP_ERR_INVALID_ARG;

  et_tp_gqa_config_t gqa = {cfg->batch,        cfg->seq_len,
                            cfg->d_model,      cfg->num_q_heads,
                            cfg->num_kv_heads, cfg->head_dim};
  et_tp_status_t st = et_tp_gqa_validate_config(&gqa, world_size);
  if (st != ET_TP_OK)
    return st;

  size_t m = 0u;
  if (!tp_mul_size(cfg->batch, cfg->seq_len, &m) || m > UINT32_MAX)
    return ET_TP_ERR_OVERFLOW;

  et_tp_swiglu_config_t swiglu = {(uint32_t)m, cfg->d_model, cfg->d_ff};
  return et_tp_swiglu_validate_config(&swiglu, world_size);
}

et_tp_status_t et_tp_block_memory_stats_v1(
    const et_tp_block_config_t *cfg, uint32_t world_size,
    et_tp_block_memory_stats_t *out_stats) {
  if (out_stats == NULL)
    return ET_TP_ERR_INVALID_ARG;
  et_tp_status_t st = et_tp_block_validate_config(cfg, world_size);
  if (st != ET_TP_OK)
    return st;

  const uint64_t D = cfg->d_model;
  const uint64_t F = cfg->d_ff;
  const uint64_t q_dim = (uint64_t)cfg->num_q_heads * (uint64_t)cfg->head_dim;
  const uint64_t kv_dim = (uint64_t)cfg->num_kv_heads * (uint64_t)cfg->head_dim;

  /* Sharded matrices: w_q, w_k, w_v, w_o, w_gate, w_up, w_down */
  const uint64_t full_sharded_elems =
      (q_dim * D) + (kv_dim * D) + (kv_dim * D) + (D * q_dim) +
      (F * D) + (F * D) + (D * F);
  const uint64_t full_sharded_bytes = full_sharded_elems * sizeof(float);
  const uint64_t rank_sharded_bytes = full_sharded_bytes / (uint64_t)world_size;
  const uint64_t replicated_bytes = 2u * D * sizeof(float);

  out_stats->full_sharded_weight_bytes = full_sharded_bytes;
  out_stats->rank_sharded_weight_bytes = rank_sharded_bytes;
  out_stats->replicated_weight_bytes = replicated_bytes;
  out_stats->rank_total_weight_bytes = rank_sharded_bytes + replicated_bytes;
  return ET_TP_OK;
}

static void tp_slice_col_shard(const float *full, float *shard, size_t rows,
                               size_t full_cols, uint32_t rank,
                               uint32_t world_size) {
  const size_t local_cols = full_cols / world_size;
  const size_t col_off = (size_t)rank * local_cols;
  for (size_t i = 0u; i < rows; ++i) {
    memcpy(shard + i * local_cols, full + i * full_cols + col_off,
           local_cols * sizeof(float));
  }
}

et_tp_status_t et_tp_block_shard_weights_v1(
    const et_tp_block_config_t *cfg, uint32_t rank, uint32_t world_size,
    const et_tp_block_weights_t *full_w, et_tp_block_weights_t *rank_w) {
  et_tp_status_t st = et_tp_block_validate_config(cfg, world_size);
  if (st != ET_TP_OK)
    return st;
  if (rank >= world_size || full_w == NULL || rank_w == NULL)
    return ET_TP_ERR_INVALID_ARG;

  if (full_w->gamma1 == NULL || full_w->w_q == NULL || full_w->w_k == NULL ||
      full_w->w_v == NULL || full_w->w_o == NULL || full_w->gamma2 == NULL ||
      full_w->w_gate == NULL || full_w->w_up == NULL ||
      full_w->w_down == NULL || rank_w->gamma1 == NULL ||
      rank_w->w_q == NULL || rank_w->w_k == NULL || rank_w->w_v == NULL ||
      rank_w->w_o == NULL || rank_w->gamma2 == NULL ||
      rank_w->w_gate == NULL || rank_w->w_up == NULL || rank_w->w_down == NULL)
    return ET_TP_ERR_INVALID_ARG;

  const size_t D = cfg->d_model;
  const size_t F = cfg->d_ff;
  const size_t F_loc = F / world_size;
  const size_t q_dim = (size_t)cfg->num_q_heads * (size_t)cfg->head_dim;
  const size_t q_loc = q_dim / world_size;
  const size_t kv_dim = (size_t)cfg->num_kv_heads * (size_t)cfg->head_dim;
  const size_t kv_loc = kv_dim / world_size;

  memcpy(rank_w->gamma1, full_w->gamma1, D * sizeof(float));
  memcpy(rank_w->gamma2, full_w->gamma2, D * sizeof(float));

  /* Row-sharded (ColumnParallel): contiguous row blocks */
  memcpy(rank_w->w_q, full_w->w_q + (size_t)rank * q_loc * D,
         q_loc * D * sizeof(float));
  memcpy(rank_w->w_k, full_w->w_k + (size_t)rank * kv_loc * D,
         kv_loc * D * sizeof(float));
  memcpy(rank_w->w_v, full_w->w_v + (size_t)rank * kv_loc * D,
         kv_loc * D * sizeof(float));
  memcpy(rank_w->w_gate, full_w->w_gate + (size_t)rank * F_loc * D,
         F_loc * D * sizeof(float));
  memcpy(rank_w->w_up, full_w->w_up + (size_t)rank * F_loc * D,
         F_loc * D * sizeof(float));

  /* Column-sharded (RowParallel): per-row column slices */
  tp_slice_col_shard(full_w->w_o, rank_w->w_o, D, q_dim, rank, world_size);
  tp_slice_col_shard(full_w->w_down, rank_w->w_down, D, F, rank, world_size);
  return ET_TP_OK;
}

static et_tp_status_t tp_gather_col_sharded_matrix(
    et_tp_group_t *group, const float *shard, float *full, size_t rows,
    size_t full_cols, uint64_t tag, int32_t timeout_ms) {
  const size_t W = group->world_size;
  const size_t local_cols = full_cols / W;
  const size_t shard_elems = rows * local_cols;
  const size_t full_elems = rows * full_cols;

  float *gathered = (float *)malloc(full_elems * sizeof(float));
  if (gathered == NULL)
    return ET_TP_ERR_ALLOC;

  const et_tp_status_t st =
      et_tp_allgather_f32(group, shard, shard_elems, gathered, tag, timeout_ms);
  if (st == ET_TP_OK) {
    for (size_t r = 0u; r < W; ++r) {
      const float *src = gathered + r * shard_elems;
      for (size_t i = 0u; i < rows; ++i) {
        memcpy(full + i * full_cols + r * local_cols, src + i * local_cols,
               local_cols * sizeof(float));
      }
    }
  }
  free(gathered);
  return st;
}

et_tp_status_t et_tp_block_gather_weights_v1(
    et_tp_group_t *group, const et_tp_block_config_t *cfg,
    const et_tp_block_weights_t *rank_w, et_tp_block_weights_t *full_w,
    uint64_t tag_base, int32_t timeout_ms) {
  if (group == NULL || rank_w == NULL || full_w == NULL)
    return ET_TP_ERR_INVALID_ARG;
  et_tp_status_t st = et_tp_block_validate_config(cfg, group->world_size);
  if (st != ET_TP_OK)
    return st;

  if (rank_w->gamma1 == NULL || rank_w->w_q == NULL || rank_w->w_k == NULL ||
      rank_w->w_v == NULL || rank_w->w_o == NULL || rank_w->gamma2 == NULL ||
      rank_w->w_gate == NULL || rank_w->w_up == NULL ||
      rank_w->w_down == NULL || full_w->gamma1 == NULL ||
      full_w->w_q == NULL || full_w->w_k == NULL || full_w->w_v == NULL ||
      full_w->w_o == NULL || full_w->gamma2 == NULL ||
      full_w->w_gate == NULL || full_w->w_up == NULL || full_w->w_down == NULL)
    return ET_TP_ERR_INVALID_ARG;

  const size_t W = group->world_size;
  const size_t D = cfg->d_model;
  const size_t F = cfg->d_ff;
  const size_t F_loc = F / W;
  const size_t q_dim = (size_t)cfg->num_q_heads * (size_t)cfg->head_dim;
  const size_t q_loc = q_dim / W;
  const size_t kv_dim = (size_t)cfg->num_kv_heads * (size_t)cfg->head_dim;
  const size_t kv_loc = kv_dim / W;

  memcpy(full_w->gamma1, rank_w->gamma1, D * sizeof(float));
  memcpy(full_w->gamma2, rank_w->gamma2, D * sizeof(float));

  st = et_tp_allgather_f32(group, rank_w->w_q, q_loc * D, full_w->w_q,
                           tag_base + 0u, timeout_ms);
  if (st != ET_TP_OK)
    return st;
  st = et_tp_allgather_f32(group, rank_w->w_k, kv_loc * D, full_w->w_k,
                           tag_base + 1u, timeout_ms);
  if (st != ET_TP_OK)
    return st;
  st = et_tp_allgather_f32(group, rank_w->w_v, kv_loc * D, full_w->w_v,
                           tag_base + 2u, timeout_ms);
  if (st != ET_TP_OK)
    return st;
  st = tp_gather_col_sharded_matrix(group, rank_w->w_o, full_w->w_o, D, q_dim,
                                    tag_base + 3u, timeout_ms);
  if (st != ET_TP_OK)
    return st;
  st = et_tp_allgather_f32(group, rank_w->w_gate, F_loc * D, full_w->w_gate,
                           tag_base + 4u, timeout_ms);
  if (st != ET_TP_OK)
    return st;
  st = et_tp_allgather_f32(group, rank_w->w_up, F_loc * D, full_w->w_up,
                           tag_base + 5u, timeout_ms);
  if (st != ET_TP_OK)
    return st;
  return tp_gather_col_sharded_matrix(group, rank_w->w_down, full_w->w_down, D,
                                      F, tag_base + 6u, timeout_ms);
}

et_tp_status_t et_tp_block_forward_v1(
    et_tp_group_t *group, const et_tp_block_config_t *cfg,
    const et_tp_block_weights_t *w, const float *x, float *y,
    et_tp_block_cache_t *cache, uint64_t tag_base, int32_t timeout_ms) {
  if (group == NULL || w == NULL || cache == NULL)
    return ET_TP_ERR_INVALID_ARG;
  et_tp_status_t st = et_tp_block_validate_config(cfg, group->world_size);
  if (st != ET_TP_OK)
    return st;

  const uint32_t M = cfg->batch * cfg->seq_len;
  const size_t xd_cnt = (size_t)M * (size_t)cfg->d_model;
  const size_t xd_bytes = xd_cnt * sizeof(float);

  st = tp_check_f32_buf(x, xd_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(y);
  if (st != ET_TP_OK)
    return st;
  if (cache->x == NULL || cache->norm1 == NULL || cache->q_r == NULL ||
      cache->k_r == NULL || cache->v_r == NULL || cache->attn_r == NULL ||
      cache->h1 == NULL || cache->norm2 == NULL || cache->gate_r == NULL ||
      cache->up_r == NULL)
    return ET_TP_ERR_INVALID_ARG;
  if (tp_spans_overlap(y, xd_bytes, x, xd_bytes))
    return ET_TP_ERR_ALIAS;

  memcpy(cache->x, x, xd_bytes);

  /* 1. RMSNorm 1 */
  st = et_tp_rmsnorm_forward_v1(M, cfg->d_model, cfg->rms_eps, cache->x,
                                w->gamma1, cache->norm1);
  if (st != ET_TP_OK)
    return st;

  /* 2. TP-GQA Attention (1 AllReduce) */
  float *attn_out = (float *)malloc(xd_bytes);
  float *mlp_out = (float *)malloc(xd_bytes);
  if (attn_out == NULL || mlp_out == NULL) {
    free(attn_out);
    free(mlp_out);
    return ET_TP_ERR_ALLOC;
  }

  const et_tp_gqa_config_t gqa_cfg = {cfg->batch,        cfg->seq_len,
                                      cfg->d_model,      cfg->num_q_heads,
                                      cfg->num_kv_heads, cfg->head_dim};
  st = et_tp_gqa_forward_v1(group, &gqa_cfg, cache->norm1, w->w_q, w->w_k,
                            w->w_v, w->w_o, cache->q_r, cache->k_r, cache->v_r,
                            cache->attn_r, attn_out, tag_base + 0u, timeout_ms);
  if (st != ET_TP_OK)
    goto cleanup;

  /* 3. Residual 1 */
  for (size_t i = 0u; i < xd_cnt; ++i) {
    const float v = (float)((double)cache->x[i] + (double)attn_out[i]);
    if (!isfinite(v)) {
      st = ET_TP_ERR_NONFINITE;
      goto cleanup;
    }
    cache->h1[i] = v;
  }

  /* 4. RMSNorm 2 */
  st = et_tp_rmsnorm_forward_v1(M, cfg->d_model, cfg->rms_eps, cache->h1,
                                w->gamma2, cache->norm2);
  if (st != ET_TP_OK)
    goto cleanup;

  /* 5. TP-SwiGLU MLP (1 AllReduce) */
  const et_tp_swiglu_config_t swiglu_cfg = {M, cfg->d_model, cfg->d_ff};
  st = et_tp_swiglu_forward_v1(group, &swiglu_cfg, cache->norm2, w->w_gate,
                               w->w_up, w->w_down, cache->gate_r, cache->up_r,
                               mlp_out, tag_base + 1u, timeout_ms);
  if (st != ET_TP_OK)
    goto cleanup;

  /* 6. Residual 2 */
  for (size_t i = 0u; i < xd_cnt; ++i) {
    const float v = (float)((double)cache->h1[i] + (double)mlp_out[i]);
    if (!isfinite(v)) {
      st = ET_TP_ERR_NONFINITE;
      goto cleanup;
    }
    y[i] = v;
  }

cleanup:
  free(attn_out);
  free(mlp_out);
  return st;
}

et_tp_status_t et_tp_block_backward_v1(
    et_tp_group_t *group, const et_tp_block_config_t *cfg,
    const et_tp_block_weights_t *w, const et_tp_block_cache_t *cache,
    const float *dy, float *dx, et_tp_block_grads_t *grads, uint64_t tag_base,
    int32_t timeout_ms) {
  if (group == NULL || w == NULL || cache == NULL || grads == NULL)
    return ET_TP_ERR_INVALID_ARG;
  et_tp_status_t st = et_tp_block_validate_config(cfg, group->world_size);
  if (st != ET_TP_OK)
    return st;

  const uint32_t M = cfg->batch * cfg->seq_len;
  const size_t xd_cnt = (size_t)M * (size_t)cfg->d_model;
  const size_t xd_bytes = xd_cnt * sizeof(float);

  st = tp_check_f32_buf(dy, xd_cnt, 1);
  if (st != ET_TP_OK)
    return st;
  st = tp_check_f32_out(dx);
  if (st != ET_TP_OK)
    return st;
  if (tp_spans_overlap(dx, xd_bytes, dy, xd_bytes))
    return ET_TP_ERR_ALIAS;

  float *dnorm2 = (float *)malloc(xd_bytes);
  float *dh1_mlp = (float *)malloc(xd_bytes);
  float *dh1 = (float *)malloc(xd_bytes);
  float *dnorm1 = (float *)malloc(xd_bytes);
  float *dx_attn = (float *)malloc(xd_bytes);
  if (dnorm2 == NULL || dh1_mlp == NULL || dh1 == NULL || dnorm1 == NULL ||
      dx_attn == NULL) {
    st = ET_TP_ERR_ALLOC;
    goto cleanup;
  }

  /* 1. TP-SwiGLU backward (1 AllReduce) */
  const et_tp_swiglu_config_t swiglu_cfg = {M, cfg->d_model, cfg->d_ff};
  st = et_tp_swiglu_backward_v1(group, &swiglu_cfg, cache->norm2, w->w_gate,
                                w->w_up, w->w_down, cache->gate_r, cache->up_r,
                                dy, dnorm2, grads->dw_gate, grads->dw_up,
                                grads->dw_down, tag_base + 0u, timeout_ms);
  if (st != ET_TP_OK)
    goto cleanup;

  /* 2. RMSNorm 2 backward */
  st = et_tp_rmsnorm_backward_v1(M, cfg->d_model, cfg->rms_eps, cache->h1,
                                 w->gamma2, dnorm2, dh1_mlp, grads->dgamma2);
  if (st != ET_TP_OK)
    goto cleanup;

  /* 3. Residual 2 backward: dh1 = dy + dh1_mlp */
  for (size_t i = 0u; i < xd_cnt; ++i) {
    const float v = (float)((double)dy[i] + (double)dh1_mlp[i]);
    if (!isfinite(v)) {
      st = ET_TP_ERR_NONFINITE;
      goto cleanup;
    }
    dh1[i] = v;
  }

  /* 4. TP-GQA backward (1 AllReduce) */
  const et_tp_gqa_config_t gqa_cfg = {cfg->batch,        cfg->seq_len,
                                      cfg->d_model,      cfg->num_q_heads,
                                      cfg->num_kv_heads, cfg->head_dim};
  st = et_tp_gqa_backward_v1(group, &gqa_cfg, cache->norm1, w->w_q, w->w_k,
                             w->w_v, w->w_o, cache->q_r, cache->k_r, cache->v_r,
                             cache->attn_r, dh1, dnorm1, grads->dw_q,
                             grads->dw_k, grads->dw_v, grads->dw_o,
                             tag_base + 1u, timeout_ms);
  if (st != ET_TP_OK)
    goto cleanup;

  /* 5. RMSNorm 1 backward */
  st = et_tp_rmsnorm_backward_v1(M, cfg->d_model, cfg->rms_eps, cache->x,
                                 w->gamma1, dnorm1, dx_attn, grads->dgamma1);
  if (st != ET_TP_OK)
    goto cleanup;

  /* 6. Residual 1 backward: dx = dh1 + dx_attn */
  for (size_t i = 0u; i < xd_cnt; ++i) {
    const float v = (float)((double)dh1[i] + (double)dx_attn[i]);
    if (!isfinite(v)) {
      st = ET_TP_ERR_NONFINITE;
      goto cleanup;
    }
    dx[i] = v;
  }

cleanup:
  free(dnorm2);
  free(dh1_mlp);
  free(dh1);
  free(dnorm1);
  free(dx_attn);
  return st;
}
