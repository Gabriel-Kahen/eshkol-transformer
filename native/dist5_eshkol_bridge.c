#define _GNU_SOURCE
#include "dist5_eshkol_bridge.h"

#include "eshkol_transformer/kernel_abi.h"
#include "eshkol_transformer/mod5_primitives_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON

_Static_assert(sizeof(float) == 4u && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   FLT_MAX_EXP == 128 && FLT_EVAL_METHOD == 0,
               "DIST5 AOT bridge requires IEEE-754 binary32");

#define HELLO_MAGIC UINT64_C(0x45544435414f5401)
enum { HELLO_WORDS = 4, ACK_WORDS = 2 };
enum { TOKENS = 8, WIDTH = 16, HIDDEN = 32, HEADS = 4, HEAD_WIDTH = 4 };
enum { MAX_F32_COUNT = ET_DIST5_MAX_BUFFER_BYTES / sizeof(float) };

struct et_dist5_aot_handle {
  et_dist5_comm_t *comm;
  et_kernel_runtime *mod5_runtime;
  uint32_t rank;
  uint32_t world_size;
};

static const et_kernel_provider_v1 *resolve_mod5(void *context,
                                                 const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mod5_kernel_provider_v1()
             : NULL;
}

static int64_t now_ms(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    return -1;
  return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static void sleep_ms_bounded(int64_t ms) {
  if (ms <= 0)
    return;
  struct timespec req;
  req.tv_sec = (time_t)(ms / 1000);
  req.tv_nsec = (long)((ms % 1000) * 1000000);
  while (nanosleep(&req, &req) != 0 && errno == EINTR) {
  }
}

static int retryable_errno(int err) {
  return err == EINTR || err == EAGAIN || err == EWOULDBLOCK;
}

static int valid_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && ((uintptr_t)p % alignment) == 0u &&
         (uintptr_t)p <= UINTPTR_MAX - bytes;
}

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  if (a_bytes == 0u || b_bytes == 0u)
    return 0;
  const uintptr_t left = (uintptr_t)a;
  const uintptr_t right = (uintptr_t)b;
  if (left > UINTPTR_MAX - a_bytes || right > UINTPTR_MAX - b_bytes)
    return 1;
  return left < right + b_bytes && right < left + a_bytes;
}

static int valid_f32_count(int64_t count) {
  return count > 0 && (uint64_t)count <= (uint64_t)MAX_F32_COUNT;
}

static int valid_f32_span(const void *p, int64_t count) {
  if (!valid_f32_count(count))
    return 0;
  return valid_span(p, _Alignof(float), (size_t)count * sizeof(float));
}

static int valid_f32_index(const void *buf, int64_t capacity, int64_t index) {
  if (capacity <= 0 || index < 0 || index >= capacity)
    return 0;
  if ((uint64_t)capacity > (uint64_t)(SIZE_MAX / sizeof(float)))
    return 0;
  return valid_span(buf, _Alignof(float), (size_t)capacity * sizeof(float));
}

static float f32_from_double(double value) {
  if (isnan(value))
    return NAN;
  if (value > (double)FLT_MAX)
    return INFINITY;
  if (value < -(double)FLT_MAX)
    return -INFINITY;
  return (float)value;
}

static int32_t io_all_deadline(int fd, void *buf, size_t len, int writing,
                               int64_t deadline) {
  unsigned char *cursor = (unsigned char *)buf;
  while (len != 0u) {
    const int64_t now = now_ms();
    if (now < 0)
      return ET_DIST5_ERR_IO;
    if (now >= deadline)
      return ET_DIST5_ERR_TIMEOUT;
    const int64_t rem = deadline - now;
    const int poll_ms = rem > INT_MAX ? INT_MAX : (int)rem;
    struct pollfd pfd = {fd, (short)((writing ? POLLOUT : POLLIN) | POLLRDHUP),
                         0};
    const int rc = poll(&pfd, 1, poll_ms);
    if (rc < 0 && errno == EINTR)
      continue;
    if (rc < 0)
      return ET_DIST5_ERR_IO;
    if (rc == 0)
      continue;
    if (pfd.revents & POLLNVAL)
      return ET_DIST5_ERR_IO;
    if (!writing && (pfd.revents & POLLIN)) {
      /* Read buffered bytes before treating POLLHUP/POLLRDHUP as EOF. */
    } else if (pfd.revents & (POLLERR | POLLHUP | POLLRDHUP)) {
      return ET_DIST5_ERR_PEER_DISCONNECTED;
    }
    const ssize_t n =
        writing ? send(fd, cursor, len, MSG_DONTWAIT | MSG_NOSIGNAL)
                : recv(fd, cursor, len, MSG_DONTWAIT);
    if (n < 0 && retryable_errno(errno))
      continue;
    if (n == 0 ||
        (n < 0 && (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN)))
      return ET_DIST5_ERR_PEER_DISCONNECTED;
    if (n < 0)
      return ET_DIST5_ERR_IO;
    cursor += (size_t)n;
    len -= (size_t)n;
  }
  return ET_DIST5_OK;
}

static int build_sock_addr(const char *sock_dir, struct sockaddr_un *addr) {
  if (sock_dir == NULL || sock_dir[0] == '\0')
    return 0;
  struct stat st;
  if (stat(sock_dir, &st) != 0 || !S_ISDIR(st.st_mode))
    return 0;
  memset(addr, 0, sizeof(*addr));
  addr->sun_family = AF_UNIX;
  const int n =
      snprintf(addr->sun_path, sizeof(addr->sun_path), "%s/dist5_r0.sock",
               sock_dir);
  if (n <= 0 || (size_t)n >= sizeof(addr->sun_path))
    return 0;
  return 1;
}

struct pending_peer {
  int fd;
  unsigned char buf[HELLO_WORDS * sizeof(uint64_t)];
  size_t got;
};

static void best_effort_ack(int fd, int32_t status) {
  if (fd < 0)
    return;
  const uint64_t ack[ACK_WORDS] = {HELLO_MAGIC, (uint64_t)(uint32_t)status};
  (void)send(fd, ack, sizeof(ack), MSG_DONTWAIT | MSG_NOSIGNAL);
}

static int32_t open_coordinator_sockets(const struct sockaddr_un *addr,
                                        uint32_t world_size, int64_t deadline,
                                        int *peer_fds) {
  int lfd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (lfd < 0)
    return ET_DIST5_ERR_IO;

  if (unlink(addr->sun_path) != 0 && errno != ENOENT) {
    close(lfd);
    return ET_DIST5_ERR_IO;
  }
  if (bind(lfd, (const struct sockaddr *)addr, sizeof(*addr)) != 0 ||
      listen(lfd, (int)world_size) != 0) {
    close(lfd);
    (void)unlink(addr->sun_path);
    return ET_DIST5_ERR_IO;
  }

  struct pending_peer pending[ET_DIST5_MAX_WORLD_SIZE];
  for (size_t i = 0; i < ET_DIST5_MAX_WORLD_SIZE; ++i) {
    pending[i].fd = -1;
    pending[i].got = 0u;
  }

  uint32_t connected = 0u;
  int32_t status = ET_DIST5_OK;

  while (status == ET_DIST5_OK && connected < world_size - 1u) {
    struct pollfd pfds[1u + 2u * ET_DIST5_MAX_WORLD_SIZE];
    int pfd_kind[1u + 2u * ET_DIST5_MAX_WORLD_SIZE];
    size_t pfd_slot[1u + 2u * ET_DIST5_MAX_WORLD_SIZE];
    nfds_t n = 0u;

    pfds[n] = (struct pollfd){lfd, POLLIN, 0};
    pfd_kind[n] = 0;
    pfd_slot[n] = 0u;
    ++n;

    for (size_t i = 0; i < ET_DIST5_MAX_WORLD_SIZE; ++i) {
      if (pending[i].fd >= 0) {
        pfds[n] = (struct pollfd){pending[i].fd, POLLIN | POLLRDHUP, 0};
        pfd_kind[n] = 1;
        pfd_slot[n] = i;
        ++n;
      }
    }
    for (uint32_t r = 1u; r < world_size; ++r) {
      if (peer_fds[r] >= 0) {
        pfds[n] = (struct pollfd){peer_fds[r], POLLRDHUP, 0};
        pfd_kind[n] = 2;
        pfd_slot[n] = r;
        ++n;
      }
    }

    const int64_t now = now_ms();
    if (now < 0) {
      status = ET_DIST5_ERR_IO;
      break;
    }
    if (now >= deadline) {
      status = ET_DIST5_ERR_TIMEOUT;
      break;
    }
    const int64_t rem = deadline - now;
    const int poll_ms = rem > INT_MAX ? INT_MAX : (int)rem;
    const int rc = poll(pfds, n, poll_ms);
    if (rc < 0 && errno == EINTR)
      continue;
    if (rc < 0) {
      status = ET_DIST5_ERR_IO;
      break;
    }
    if (rc == 0)
      continue;

    for (nfds_t idx = 0; idx < n && status == ET_DIST5_OK; ++idx) {
      const short rev = pfds[idx].revents;
      if (rev == 0)
        continue;
      if (rev & POLLNVAL) {
        status = ET_DIST5_ERR_IO;
        break;
      }
      if (pfd_kind[idx] == 0) {
        if (rev & POLLERR) {
          status = ET_DIST5_ERR_IO;
          break;
        }
        for (;;) {
          const int cfd = accept4(lfd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
          if (cfd < 0) {
            if (errno == EINTR)
              continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK)
              break;
            status = ET_DIST5_ERR_IO;
            break;
          }
          size_t free_slot = ET_DIST5_MAX_WORLD_SIZE;
          for (size_t i = 0; i < ET_DIST5_MAX_WORLD_SIZE; ++i) {
            if (pending[i].fd < 0) {
              free_slot = i;
              break;
            }
          }
          if (free_slot == ET_DIST5_MAX_WORLD_SIZE) {
            best_effort_ack(cfd, ET_DIST5_ERR_MISMATCH);
            close(cfd);
            status = ET_DIST5_ERR_MISMATCH;
            break;
          }
          pending[free_slot].fd = cfd;
          pending[free_slot].got = 0u;
        }
      } else if (pfd_kind[idx] == 1) {
        const size_t slot = pfd_slot[idx];
        struct pending_peer *pp = &pending[slot];
        if (rev & POLLIN) {
          const size_t need = sizeof(pp->buf) - pp->got;
          const ssize_t got =
              recv(pp->fd, pp->buf + pp->got, need, MSG_DONTWAIT);
          if (got < 0 && !retryable_errno(errno)) {
            status = ET_DIST5_ERR_PEER_DISCONNECTED;
            break;
          }
          if (got == 0) {
            status = ET_DIST5_ERR_PEER_DISCONNECTED;
            break;
          }
          if (got > 0) {
            pp->got += (size_t)got;
            if (pp->got == sizeof(pp->buf)) {
              uint64_t hello[HELLO_WORDS];
              memcpy(hello, pp->buf, sizeof(hello));
              const uint64_t peer_rank = hello[1];
              if (hello[0] != HELLO_MAGIC ||
                  hello[2] != (uint64_t)world_size ||
                  hello[3] != (uint64_t)ET_DIST5_ABI_MAJOR ||
                  peer_rank == 0u || peer_rank >= (uint64_t)world_size ||
                  peer_fds[peer_rank] >= 0) {
                best_effort_ack(pp->fd, ET_DIST5_ERR_MISMATCH);
                close(pp->fd);
                pp->fd = -1;
                pp->got = 0u;
                status = ET_DIST5_ERR_MISMATCH;
                break;
              }
              peer_fds[peer_rank] = pp->fd;
              pp->fd = -1;
              pp->got = 0u;
              ++connected;
            }
          }
        } else if (rev & (POLLERR | POLLHUP | POLLRDHUP)) {
          status = ET_DIST5_ERR_PEER_DISCONNECTED;
          break;
        }
      } else {
        if (rev & (POLLERR | POLLHUP | POLLRDHUP)) {
          status = ET_DIST5_ERR_PEER_DISCONNECTED;
          break;
        }
      }
    }
  }

  /* Reject any stray pending connections beyond world_size - 1. */
  for (size_t i = 0; i < ET_DIST5_MAX_WORLD_SIZE; ++i) {
    if (pending[i].fd >= 0) {
      best_effort_ack(pending[i].fd,
                      status == ET_DIST5_OK ? ET_DIST5_ERR_MISMATCH : status);
      close(pending[i].fd);
      pending[i].fd = -1;
    }
  }

  close(lfd);
  (void)unlink(addr->sun_path);

  if (status != ET_DIST5_OK) {
    for (uint32_t r = 1u; r < world_size; ++r) {
      if (peer_fds[r] >= 0) {
        best_effort_ack(peer_fds[r], status);
        close(peer_fds[r]);
        peer_fds[r] = -1;
      }
    }
    return status;
  }

  uint64_t ack[ACK_WORDS] = {HELLO_MAGIC, (uint64_t)ET_DIST5_OK};
  for (uint32_t r = 1u; r < world_size; ++r) {
    status = io_all_deadline(peer_fds[r], ack, sizeof(ack), 1, deadline);
    if (status != ET_DIST5_OK) {
      for (uint32_t k = 1u; k < world_size; ++k) {
        if (peer_fds[k] >= 0) {
          close(peer_fds[k]);
          peer_fds[k] = -1;
        }
      }
      return status;
    }
  }
  return ET_DIST5_OK;
}

static int32_t open_worker_socket(const struct sockaddr_un *addr, uint32_t rank,
                                  uint32_t world_size, int64_t deadline,
                                  int *out_fd) {
  *out_fd = -1;
  int fd = -1;

  for (;;) {
    const int64_t now = now_ms();
    if (now < 0)
      return ET_DIST5_ERR_IO;
    if (now >= deadline)
      return ET_DIST5_ERR_TIMEOUT;

    fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0)
      return ET_DIST5_ERR_IO;

    const int rc = connect(fd, (const struct sockaddr *)addr, sizeof(*addr));
    if (rc == 0)
      break;

    const int err = errno;
    if (err == EINPROGRESS) {
      const int64_t t = now_ms();
      if (t < 0) {
        close(fd);
        return ET_DIST5_ERR_IO;
      }
      if (t >= deadline) {
        close(fd);
        return ET_DIST5_ERR_TIMEOUT;
      }
      const int64_t rem = deadline - t;
      struct pollfd pfd = {fd, POLLOUT, 0};
      const int prc = poll(&pfd, 1, rem > INT_MAX ? INT_MAX : (int)rem);
      if (prc <= 0) {
        close(fd);
        if (prc < 0 && errno != EINTR)
          return ET_DIST5_ERR_IO;
        continue;
      }
      int so_err = 0;
      socklen_t slen = sizeof(so_err);
      if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &slen) != 0) {
        close(fd);
        return ET_DIST5_ERR_IO;
      }
      if (so_err == 0)
        break;
      close(fd);
      fd = -1;
      if (so_err != ENOENT && so_err != ECONNREFUSED && so_err != EAGAIN)
        return ET_DIST5_ERR_IO;
    } else {
      close(fd);
      fd = -1;
      if (err != ENOENT && err != ECONNREFUSED && err != EAGAIN && err != EINTR)
        return ET_DIST5_ERR_IO;
    }

    const int64_t after = now_ms();
    if (after < 0)
      return ET_DIST5_ERR_IO;
    if (after >= deadline)
      return ET_DIST5_ERR_TIMEOUT;
    const int64_t rem = deadline - after;
    sleep_ms_bounded(rem < 5 ? rem : 5);
  }

  uint64_t hello[HELLO_WORDS] = {HELLO_MAGIC, (uint64_t)rank,
                                 (uint64_t)world_size,
                                 (uint64_t)ET_DIST5_ABI_MAJOR};
  int32_t status = io_all_deadline(fd, hello, sizeof(hello), 1, deadline);
  if (status != ET_DIST5_OK) {
    close(fd);
    return status;
  }

  uint64_t ack[ACK_WORDS] = {0u, 0u};
  status = io_all_deadline(fd, ack, sizeof(ack), 0, deadline);
  if (status != ET_DIST5_OK) {
    close(fd);
    return status;
  }
  if (ack[0] != HELLO_MAGIC || ack[1] > (uint64_t)ET_DIST5_ERR_ALLOC) {
    close(fd);
    return ET_DIST5_ERR_MISMATCH;
  }
  if (ack[1] != (uint64_t)ET_DIST5_OK) {
    close(fd);
    return (int32_t)ack[1];
  }

  *out_fd = fd;
  return ET_DIST5_OK;
}

int32_t et_dist5_aot_open(const char *sock_dir, int64_t rank,
                          int64_t world_size, int64_t timeout_ms,
                          et_dist5_aot_handle_t **out) {
  if (out == NULL)
    return ET_DIST5_ERR_INVALID_ARG;
  *out = NULL;
  if (world_size <= 0 || world_size > (int64_t)ET_DIST5_MAX_WORLD_SIZE ||
      rank < 0 || rank >= world_size || timeout_ms <= 0 ||
      timeout_ms > (int64_t)INT_MAX)
    return ET_DIST5_ERR_INVALID_ARG;

  struct sockaddr_un addr;
  if (world_size > 1 && !build_sock_addr(sock_dir, &addr))
    return ET_DIST5_ERR_INVALID_ARG;

  et_kernel_runtime *mod5_runtime = NULL;
  et_kernel_error kerr;
  if (et_kernel_runtime_discover(resolve_mod5, NULL, &mod5_runtime, &kerr) !=
          0 ||
      mod5_runtime == NULL)
    return ET_DIST5_ERR_ALLOC;

  et_dist5_comm_t *comm = NULL;
  int32_t status = ET_DIST5_OK;

  if (world_size == 1) {
    status = et_dist5_comm_create(ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR, 0u,
                                  1u, NULL, 0u, &comm);
  } else {
    const int64_t now = now_ms();
    if (now < 0) {
      et_kernel_runtime_destroy(mod5_runtime);
      return ET_DIST5_ERR_IO;
    }
    const int64_t deadline = now + timeout_ms;
    if (rank == 0) {
      int peer_fds[ET_DIST5_MAX_WORLD_SIZE];
      for (size_t i = 0; i < ET_DIST5_MAX_WORLD_SIZE; ++i)
        peer_fds[i] = -1;
      status = open_coordinator_sockets(&addr, (uint32_t)world_size, deadline,
                                        peer_fds);
      if (status == ET_DIST5_OK) {
        status = et_dist5_comm_create(ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR,
                                      0u, (uint32_t)world_size, &peer_fds[1],
                                      (size_t)(world_size - 1), &comm);
      }
      for (uint32_t r = 1u; r < (uint32_t)world_size; ++r) {
        if (peer_fds[r] >= 0)
          close(peer_fds[r]);
      }
    } else {
      int fd = -1;
      status = open_worker_socket(&addr, (uint32_t)rank, (uint32_t)world_size,
                                  deadline, &fd);
      if (status == ET_DIST5_OK) {
        status = et_dist5_comm_create(ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR,
                                      (uint32_t)rank, (uint32_t)world_size, &fd,
                                      1u, &comm);
      }
      if (fd >= 0)
        close(fd);
    }
  }

  if (status != ET_DIST5_OK) {
    et_kernel_runtime_destroy(mod5_runtime);
    return status;
  }

  et_dist5_aot_handle_t *handle =
      (et_dist5_aot_handle_t *)calloc(1u, sizeof(*handle));
  if (handle == NULL) {
    et_dist5_comm_destroy(comm);
    et_kernel_runtime_destroy(mod5_runtime);
    return ET_DIST5_ERR_ALLOC;
  }
  handle->comm = comm;
  handle->mod5_runtime = mod5_runtime;
  handle->rank = (uint32_t)rank;
  handle->world_size = (uint32_t)world_size;
  *out = handle;
  return ET_DIST5_OK;
}

void et_dist5_aot_close(et_dist5_aot_handle_t *handle) {
  if (handle == NULL)
    return;
  et_dist5_comm_destroy(handle->comm);
  et_kernel_runtime_destroy(handle->mod5_runtime);
  free(handle);
}

int64_t et_dist5_aot_rank(const et_dist5_aot_handle_t *handle) {
  return handle == NULL ? -1 : (int64_t)handle->rank;
}

int64_t et_dist5_aot_world_size(const et_dist5_aot_handle_t *handle) {
  return handle == NULL ? -1 : (int64_t)handle->world_size;
}

static int valid_call_header(const et_dist5_aot_handle_t *handle, int64_t tag,
                             int64_t timeout_ms) {
  return handle != NULL && handle->comm != NULL && tag >= 0 &&
         timeout_ms > 0 && timeout_ms <= (int64_t)INT_MAX;
}

int32_t et_dist5_aot_barrier(et_dist5_aot_handle_t *handle, int64_t tag,
                             int64_t timeout_ms) {
  if (!valid_call_header(handle, tag, timeout_ms))
    return ET_DIST5_ERR_INVALID_ARG;
  return et_dist5_barrier(handle->comm, (uint64_t)tag, (int)timeout_ms);
}

int32_t et_dist5_aot_broadcast_f32(et_dist5_aot_handle_t *handle, void *buf,
                                   int64_t count, int64_t root_rank,
                                   int64_t tag, int64_t timeout_ms) {
  if (!valid_call_header(handle, tag, timeout_ms))
    return ET_DIST5_ERR_INVALID_ARG;
  const int ok = valid_f32_span(buf, count) && root_rank >= 0 &&
                 root_rank < (int64_t)handle->world_size;
  return et_dist5_broadcast(
      handle->comm, ok ? buf : NULL, ok ? (size_t)count : 0u, ET_DIST5_F32,
      ok ? (uint32_t)root_rank : UINT32_MAX, (uint64_t)tag, (int)timeout_ms);
}

int32_t et_dist5_aot_allreduce_f32(et_dist5_aot_handle_t *handle,
                                   const void *sendbuf, void *recvbuf,
                                   int64_t count, int32_t op, int64_t tag,
                                   int64_t timeout_ms) {
  if (!valid_call_header(handle, tag, timeout_ms))
    return ET_DIST5_ERR_INVALID_ARG;
  const int ok = valid_f32_span(sendbuf, count) &&
                 valid_f32_span(recvbuf, count) && op >= ET_DIST5_SUM &&
                 op <= ET_DIST5_MAX;
  return et_dist5_allreduce(handle->comm, ok ? sendbuf : NULL,
                            ok ? recvbuf : NULL, ok ? (size_t)count : 0u,
                            ET_DIST5_F32, ok ? (et_dist5_op_t)op : 0u,
                            (uint64_t)tag, (int)timeout_ms);
}

int32_t et_dist5_aot_allgather_f32(et_dist5_aot_handle_t *handle,
                                   const void *sendbuf, int64_t send_count,
                                   void *recvbuf, int64_t tag,
                                   int64_t timeout_ms) {
  if (!valid_call_header(handle, tag, timeout_ms))
    return ET_DIST5_ERR_INVALID_ARG;
  const int count_ok =
      send_count > 0 &&
      (uint64_t)send_count <=
          (uint64_t)(MAX_F32_COUNT / handle->world_size);
  const int ok = count_ok && valid_f32_span(sendbuf, send_count) &&
                 valid_f32_span(recvbuf,
                                send_count * (int64_t)handle->world_size);
  return et_dist5_allgather(handle->comm, ok ? sendbuf : NULL,
                            ok ? (size_t)send_count : 0u, ok ? recvbuf : NULL,
                            ET_DIST5_F32, (uint64_t)tag, (int)timeout_ms);
}

int32_t et_dist5_aot_reduce_scatter_f32(et_dist5_aot_handle_t *handle,
                                        const void *sendbuf, void *recvbuf,
                                        int64_t recv_count, int32_t op,
                                        int64_t tag, int64_t timeout_ms) {
  if (!valid_call_header(handle, tag, timeout_ms))
    return ET_DIST5_ERR_INVALID_ARG;
  const int count_ok =
      recv_count > 0 &&
      (uint64_t)recv_count <=
          (uint64_t)(MAX_F32_COUNT / handle->world_size);
  const int ok = count_ok &&
                 valid_f32_span(sendbuf,
                                recv_count * (int64_t)handle->world_size) &&
                 valid_f32_span(recvbuf, recv_count) && op >= ET_DIST5_SUM &&
                 op <= ET_DIST5_MAX;
  return et_dist5_reduce_scatter(handle->comm, ok ? sendbuf : NULL,
                                 ok ? recvbuf : NULL,
                                 ok ? (size_t)recv_count : 0u, ET_DIST5_F32,
                                 ok ? (et_dist5_op_t)op : 0u, (uint64_t)tag,
                                 (int)timeout_ms);
}

int32_t et_dist5_aot_sync_gradients_f32(et_dist5_aot_handle_t *handle,
                                        const void *local, void *mean,
                                        int64_t count, int64_t tag,
                                        int64_t timeout_ms) {
  if (!valid_call_header(handle, tag, timeout_ms))
    return ET_DIST5_ERR_INVALID_ARG;
  const int ok = valid_f32_span(local, count) && valid_f32_span(mean, count);
  return et_dist5_sync_gradients_f32(
      handle->comm, ok ? (const float *)local : NULL, ok ? (float *)mean : NULL,
      ok ? (size_t)count : 0u, (uint64_t)tag, (int)timeout_ms);
}

void *et_dist5_aot_alloc_f32(int64_t count) {
  if (!valid_f32_count(count))
    return NULL;
  return calloc((size_t)count, sizeof(float));
}

void et_dist5_aot_free_f32(void *buf) { free(buf); }

int32_t et_dist5_aot_store_f32(void *buf, int64_t capacity, int64_t index,
                               double value) {
  if (!valid_f32_index(buf, capacity, index))
    return ET_DIST5_ERR_INVALID_ARG;
  const float f = f32_from_double(value);
  memcpy((unsigned char *)buf + (size_t)index * sizeof(float), &f, sizeof(f));
  return ET_DIST5_OK;
}

double et_dist5_aot_load_f32(const void *buf, int64_t capacity, int64_t index) {
  if (!valid_f32_index(buf, capacity, index))
    return NAN;
  float f = 0.0f;
  memcpy(&f, (const unsigned char *)buf + (size_t)index * sizeof(float),
         sizeof(f));
  return (double)f;
}

int32_t et_dist5_aot_store_f32_bits(void *buf, int64_t capacity, int64_t index,
                                    int64_t bits) {
  if (!valid_f32_index(buf, capacity, index) || bits < 0 ||
      bits > INT64_C(0xffffffff))
    return ET_DIST5_ERR_INVALID_ARG;
  const uint32_t u = (uint32_t)bits;
  memcpy((unsigned char *)buf + (size_t)index * sizeof(uint32_t), &u,
         sizeof(u));
  return ET_DIST5_OK;
}

int64_t et_dist5_aot_load_f32_bits(const void *buf, int64_t capacity,
                                   int64_t index) {
  if (!valid_f32_index(buf, capacity, index))
    return -1;
  uint32_t u = 0u;
  memcpy(&u, (const unsigned char *)buf + (size_t)index * sizeof(uint32_t),
         sizeof(u));
  return (int64_t)u;
}

int32_t et_dist5_aot_equal_bits_f32(const void *a, const void *b,
                                    int64_t count) {
  if (!valid_f32_span(a, count) || !valid_f32_span(b, count))
    return -1;
  return memcmp(a, b, (size_t)count * sizeof(float)) == 0 ? 1 : 0;
}

/* Self-contained FIPS 180-4 SHA-256 over raw byte spans. */
static uint32_t rotr32(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32u - n));
}

static void sha256_compress(uint32_t state[8], const unsigned char block[64]) {
  static const uint32_t k[64] = {
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
  for (size_t i = 0; i < 16u; ++i) {
    w[i] = ((uint32_t)block[4u * i] << 24) |
           ((uint32_t)block[4u * i + 1u] << 16) |
           ((uint32_t)block[4u * i + 2u] << 8) |
           (uint32_t)block[4u * i + 3u];
  }
  for (size_t i = 16u; i < 64u; ++i) {
    const uint32_t s0 =
        rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^ (w[i - 15u] >> 3);
    const uint32_t s1 =
        rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^ (w[i - 2u] >> 10);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }

  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
  for (size_t i = 0; i < 64u; ++i) {
    const uint32_t S1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t temp1 = h + S1 + ch + k[i] + w[i];
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
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

int32_t et_dist5_aot_sha256_hex_f32(const void *buf, int64_t count,
                                    char *hex_out, int64_t hex_capacity) {
  if (!valid_f32_span(buf, count) || hex_out == NULL || hex_capacity < 65)
    return ET_DIST5_ERR_INVALID_ARG;
  const size_t total_bytes = (size_t)count * sizeof(float);
  if (!valid_span(hex_out, 1u, (size_t)hex_capacity) ||
      spans_overlap(buf, total_bytes, hex_out, 65u))
    return ET_DIST5_ERR_INVALID_ARG;

  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  const unsigned char *bytes = (const unsigned char *)buf;
  size_t offset = 0u;
  while (total_bytes - offset >= 64u) {
    sha256_compress(state, bytes + offset);
    offset += 64u;
  }

  unsigned char tail[128];
  memset(tail, 0, sizeof(tail));
  const size_t rem = total_bytes - offset;
  if (rem > 0u)
    memcpy(tail, bytes + offset, rem);
  tail[rem] = 0x80u;
  const size_t pad_blocks = rem >= 56u ? 2u : 1u;
  const size_t len_pos = pad_blocks * 64u - 8u;
  const uint64_t bit_len = (uint64_t)total_bytes * 8u;
  for (size_t i = 0; i < 8u; ++i)
    tail[len_pos + i] = (unsigned char)((bit_len >> (56u - 8u * i)) & 0xffu);
  sha256_compress(state, tail);
  if (pad_blocks == 2u)
    sha256_compress(state, tail + 64u);

  static const char hex[] = "0123456789abcdef";
  for (size_t i = 0; i < 8u; ++i) {
    const uint32_t v = state[i];
    hex_out[8u * i + 0u] = hex[(v >> 28) & 0xfu];
    hex_out[8u * i + 1u] = hex[(v >> 24) & 0xfu];
    hex_out[8u * i + 2u] = hex[(v >> 20) & 0xfu];
    hex_out[8u * i + 3u] = hex[(v >> 16) & 0xfu];
    hex_out[8u * i + 4u] = hex[(v >> 12) & 0xfu];
    hex_out[8u * i + 5u] = hex[(v >> 8) & 0xfu];
    hex_out[8u * i + 6u] = hex[(v >> 4) & 0xfu];
    hex_out[8u * i + 7u] = hex[v & 0xfu];
  }
  hex_out[64] = '\0';
  return ET_DIST5_OK;
}

int32_t et_dist5_aot_read_byte(const void *buf, int64_t capacity,
                               int64_t index) {
  if (buf == NULL || capacity <= 0 || index < 0 || index >= capacity)
    return -1;
  if (!valid_span(buf, 1u, (size_t)capacity))
    return -1;
  return (int32_t)((const unsigned char *)buf)[(size_t)index];
}

static int finite_f32_buf(const void *buf, size_t count) {
  const unsigned char *bytes = (const unsigned char *)buf;
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits = 0u;
    memcpy(&bits, bytes + i * sizeof(bits), sizeof(bits));
    if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
      return 0;
  }
  return 1;
}

int32_t et_dist5_aot_f32_add_inplace(void *dst, const void *src,
                                     int64_t count) {
  if (!valid_f32_span(dst, count) || !valid_f32_span(src, count))
    return ET_DIST5_ERR_INVALID_ARG;
  const size_t n = (size_t)count;
  const size_t bytes = n * sizeof(float);
  if (dst != src && spans_overlap(dst, bytes, src, bytes))
    return ET_DIST5_ERR_INVALID_ARG;
  if (!finite_f32_buf(dst, n) || !finite_f32_buf(src, n))
    return ET_DIST5_ERR_NONFINITE;

  fenv_t saved;
  if (fegetenv(&saved) != 0)
    return ET_DIST5_ERR_IO;
  int32_t status = ET_DIST5_ERR_IO;
  if (fesetenv(FE_DFL_ENV) == 0) {
    status = ET_DIST5_OK;
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    /* Preflight sums so dst stays untouched if any element overflows. */
    for (size_t i = 0; i < n; ++i) {
      float a = 0.0f, b = 0.0f;
      memcpy(&a, d + i * sizeof(float), sizeof(float));
      memcpy(&b, s + i * sizeof(float), sizeof(float));
      if (!isfinite(a + b)) {
        status = ET_DIST5_ERR_NONFINITE;
        break;
      }
    }
    if (status == ET_DIST5_OK) {
      for (size_t i = 0; i < n; ++i) {
        float a = 0.0f, b = 0.0f;
        memcpy(&a, d + i * sizeof(float), sizeof(float));
        memcpy(&b, s + i * sizeof(float), sizeof(float));
        const float sum = a + b;
        memcpy(d + i * sizeof(float), &sum, sizeof(float));
      }
    }
  }
  return fesetenv(&saved) == 0 ? status : ET_DIST5_ERR_IO;
}

int32_t et_dist5_aot_f32_div_scalar_inplace(void *buf, int64_t count,
                                            double divisor) {
  if (!valid_f32_span(buf, count))
    return ET_DIST5_ERR_INVALID_ARG;
  const float div = f32_from_double(divisor);
  if (!isfinite(div) || div == 0.0f)
    return ET_DIST5_ERR_INVALID_ARG;
  const size_t n = (size_t)count;
  if (!finite_f32_buf(buf, n))
    return ET_DIST5_ERR_NONFINITE;

  fenv_t saved;
  if (fegetenv(&saved) != 0)
    return ET_DIST5_ERR_IO;
  int32_t status = ET_DIST5_ERR_IO;
  if (fesetenv(FE_DFL_ENV) == 0) {
    status = ET_DIST5_OK;
    unsigned char *p = (unsigned char *)buf;
    for (size_t i = 0; i < n; ++i) {
      float v = 0.0f;
      memcpy(&v, p + i * sizeof(float), sizeof(float));
      if (!isfinite(v / div)) {
        status = ET_DIST5_ERR_NONFINITE;
        break;
      }
    }
    if (status == ET_DIST5_OK) {
      for (size_t i = 0; i < n; ++i) {
        float v = 0.0f;
        memcpy(&v, p + i * sizeof(float), sizeof(float));
        const float q = v / div;
        memcpy(p + i * sizeof(float), &q, sizeof(float));
      }
    }
  }
  return fesetenv(&saved) == 0 ? status : ET_DIST5_ERR_IO;
}

/* MOD5 K1 kernel dispatch helpers. */
static et_kernel_tensor_view_v1 make_view(void *data, const char *dtype,
                                          size_t element_bytes, size_t rank,
                                          const uint64_t *shape) {
  size_t bytes = element_bytes;
  for (size_t i = 0; i < rank; ++i)
    bytes *= shape[i];
  return (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1),
                                    data,
                                    bytes,
                                    dtype,
                                    "cpu",
                                    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
                                    0u,
                                    rank,
                                    shape};
}

static et_kernel_tensor_view_v1 f32_view(void *data, size_t rank,
                                         const uint64_t *shape) {
  return make_view(data, "f32", sizeof(float), rank, shape);
}

static int32_t dispatch_mod5(const et_dist5_aot_handle_t *handle,
                             const char *cap, const char *operation,
                             size_t rank, const uint64_t *shape, size_t ni,
                             const et_kernel_tensor_view_v1 *in, size_t no,
                             et_kernel_tensor_view_v1 *out) {
  if (handle == NULL || handle->mod5_runtime == NULL)
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const et_kernel_request_v1 request = {
      sizeof(request), operation, "f32", "cpu", rank, shape, 1u, {0}};
  const size_t stride = sizeof(*in);
  const et_kernel_call_v1 call = {sizeof(call), cap,         &request, ni,
                                  stride,       ni * stride, in,       no,
                                  stride,       no * stride, out};
  et_kernel_error error;
  return et_kernel_runtime_dispatch(handle->mod5_runtime, &call, &error);
}

static int32_t mod5_rms(const et_dist5_aot_handle_t *handle, int backward,
                        void *x, void *gamma, void *epsilon, void *dy,
                        void *out, void *dgamma) {
  const size_t x_bytes = (size_t)TOKENS * WIDTH * sizeof(float);
  const size_t g_bytes = (size_t)WIDTH * sizeof(float);
  if (!valid_span(x, _Alignof(float), x_bytes) ||
      !valid_span(gamma, _Alignof(float), g_bytes) ||
      !valid_span(epsilon, _Alignof(float), sizeof(float)) ||
      !valid_span(out, _Alignof(float), x_bytes))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  if (backward && (!valid_span(dy, _Alignof(float), x_bytes) ||
                   !valid_span(dgamma, _Alignof(float), g_bytes)))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;

  const uint64_t shape[] = {1u, TOKENS, WIDTH};
  const et_kernel_tensor_view_v1 in[] = {
      f32_view(x, 3u, shape), f32_view(gamma, 1u, shape + 2u),
      f32_view(epsilon, 0u, NULL), f32_view(dy, 3u, shape)};
  et_kernel_tensor_view_v1 outputs[] = {f32_view(out, 3u, shape),
                                        f32_view(dgamma, 1u, shape + 2u)};
  return dispatch_mod5(handle, "kernel.rmsnorm",
                       backward ? "rmsnorm.backward" : "rmsnorm.forward", 3u,
                       shape, backward ? 4u : 3u, in, backward ? 2u : 1u,
                       outputs);
}

int32_t et_dist5_aot_mod5_rms_forward(const et_dist5_aot_handle_t *handle,
                                      void *x, void *gamma, void *epsilon,
                                      void *y) {
  return mod5_rms(handle, 0, x, gamma, epsilon, NULL, y, NULL);
}

int32_t et_dist5_aot_mod5_rms_backward(const et_dist5_aot_handle_t *handle,
                                       void *x, void *gamma, void *epsilon,
                                       void *dy, void *dx, void *dgamma) {
  return mod5_rms(handle, 1, x, gamma, epsilon, dy, dx, dgamma);
}

static int32_t mod5_swiglu(const et_dist5_aot_handle_t *handle, int backward,
                           void *gate, void *up, void *dy, void *out,
                           void *dup) {
  const size_t bytes = (size_t)TOKENS * HIDDEN * sizeof(float);
  if (!valid_span(gate, _Alignof(float), bytes) ||
      !valid_span(up, _Alignof(float), bytes) ||
      !valid_span(out, _Alignof(float), bytes))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  if (backward && (!valid_span(dy, _Alignof(float), bytes) ||
                   !valid_span(dup, _Alignof(float), bytes)))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;

  const uint64_t shape[] = {1u, TOKENS, HIDDEN};
  const et_kernel_tensor_view_v1 in[] = {f32_view(gate, 3u, shape),
                                         f32_view(up, 3u, shape),
                                         f32_view(dy, 3u, shape)};
  et_kernel_tensor_view_v1 outputs[] = {f32_view(out, 3u, shape),
                                        f32_view(dup, 3u, shape)};
  return dispatch_mod5(handle, "kernel.swiglu",
                       backward ? "swiglu.backward" : "swiglu.forward", 3u,
                       shape, backward ? 3u : 2u, in, backward ? 2u : 1u,
                       outputs);
}

int32_t et_dist5_aot_mod5_swiglu_forward(const et_dist5_aot_handle_t *handle,
                                         void *gate, void *up, void *y) {
  return mod5_swiglu(handle, 0, gate, up, NULL, y, NULL);
}

int32_t et_dist5_aot_mod5_swiglu_backward(const et_dist5_aot_handle_t *handle,
                                          void *gate, void *up, void *dy,
                                          void *dgate, void *dup) {
  return mod5_swiglu(handle, 1, gate, up, dy, dgate, dup);
}

static int32_t mod5_attention(const et_dist5_aot_handle_t *handle, int backward,
                              int64_t kv_heads, void *q, void *k, void *v,
                              void *qpos, void *kpos, void *mask, void *slopes,
                              int64_t window, void *dy, void *out, void *dk,
                              void *dv) {
  if (kv_heads != 1 && kv_heads != 2)
    return ET_KERNEL_ERROR_UNSUPPORTED;
  const size_t q_bytes = (size_t)HEADS * TOKENS * HEAD_WIDTH * sizeof(float);
  const size_t kv_bytes =
      (size_t)kv_heads * TOKENS * HEAD_WIDTH * sizeof(float);
  const size_t pos_bytes = (size_t)TOKENS * sizeof(int64_t);
  const size_t mask_bytes = (size_t)TOKENS * TOKENS * sizeof(uint8_t);
  const size_t slope_bytes = (size_t)HEADS * sizeof(float);
  if (!valid_span(q, _Alignof(float), q_bytes) ||
      !valid_span(k, _Alignof(float), kv_bytes) ||
      !valid_span(v, _Alignof(float), kv_bytes) ||
      !valid_span(qpos, _Alignof(int64_t), pos_bytes) ||
      !valid_span(kpos, _Alignof(int64_t), pos_bytes) ||
      !valid_span(mask, 1u, mask_bytes) ||
      !valid_span(slopes, _Alignof(float), slope_bytes) ||
      !valid_span(out, _Alignof(float), q_bytes))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  if (backward && (!valid_span(dy, _Alignof(float), q_bytes) ||
                   !valid_span(dk, _Alignof(float), kv_bytes) ||
                   !valid_span(dv, _Alignof(float), kv_bytes)))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;

  const uint64_t shape[] = {1u,     HEADS,  (uint64_t)kv_heads,
                            TOKENS, TOKENS, HEAD_WIDTH};
  const uint64_t q_shape[] = {1u, HEADS, TOKENS, HEAD_WIDTH};
  const uint64_t kv_shape[] = {1u, (uint64_t)kv_heads, TOKENS, HEAD_WIDTH};
  const uint64_t pos_shape[] = {1u, TOKENS};
  const uint64_t mask_shape[] = {1u, TOKENS, TOKENS};
  const uint64_t slope_shape[] = {HEADS};
  const et_kernel_tensor_view_v1 in[] = {
      f32_view(q, 4u, q_shape),
      f32_view(k, 4u, kv_shape),
      f32_view(v, 4u, kv_shape),
      make_view(qpos, "i64", sizeof(int64_t), 2u, pos_shape),
      make_view(kpos, "i64", sizeof(int64_t), 2u, pos_shape),
      make_view(mask, "bool", sizeof(uint8_t), 3u, mask_shape),
      f32_view(slopes, 1u, slope_shape),
      make_view(&window, "i64", sizeof(window), 0u, NULL),
      f32_view(dy, 4u, q_shape)};
  et_kernel_tensor_view_v1 outputs[] = {f32_view(out, 4u, q_shape),
                                        f32_view(dk, 4u, kv_shape),
                                        f32_view(dv, 4u, kv_shape)};
  return dispatch_mod5(
      handle, "kernel.variant-attention",
      backward ? "variant-attention.backward" : "variant-attention.forward", 6u,
      shape, backward ? 9u : 8u, in, backward ? 3u : 1u, outputs);
}

int32_t et_dist5_aot_mod5_attention_forward(const et_dist5_aot_handle_t *handle,
                                            int64_t kv_heads, void *q, void *k,
                                            void *v, void *qpos, void *kpos,
                                            void *mask, void *slopes,
                                            int64_t window, void *y) {
  return mod5_attention(handle, 0, kv_heads, q, k, v, qpos, kpos, mask, slopes,
                        window, NULL, y, NULL, NULL);
}

int32_t et_dist5_aot_mod5_attention_backward(
    const et_dist5_aot_handle_t *handle, int64_t kv_heads, void *q, void *k,
    void *v, void *qpos, void *kpos, void *mask, void *slopes, int64_t window,
    void *dy, void *dq, void *dk, void *dv) {
  return mod5_attention(handle, 1, kv_heads, q, k, v, qpos, kpos, mask, slopes,
                        window, dy, dq, dk, dv);
}
