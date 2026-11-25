#define _GNU_SOURCE
#include "eshkol_transformer/dist5_collectives_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   FLT_MAX_EXP == 128 && FLT_EVAL_METHOD == 0,
               "DIST5 requires binary32 floats");

enum kind {
  BARRIER = 1,
  BROADCAST,
  ALLREDUCE,
  ALLGATHER,
  REDUCE_SCATTER,
  SYNC
};
enum phase { GRANT = 1, RESULT, COMMIT, ABORT };
enum { REQUEST_WORDS = 11, REPLY_WORDS = 3 };
#define WIRE_MAGIC UINT64_C(0x4554444953540501)

struct et_dist5_comm {
  uint32_t rank, world;
  uint64_t sequence;
  et_dist5_status_t fatal;
  int fd[ET_DIST5_MAX_WORLD_SIZE];
};

struct call {
  enum kind kind;
  uint64_t tag;
  size_t count;
  et_dist5_dtype_t dtype;
  et_dist5_op_t op;
  uint32_t root;
  const void *send;
  void *recv;
};

struct exchange {
  et_dist5_comm_t *comm;
  const struct call *call;
  int64_t deadline;
  size_t input, output, total;
  unsigned char *stage, *scratch;
  et_dist5_status_t local;
  uint64_t released;
  uint64_t incomplete;
};

static int retryable(int error) {
  return error == EINTR || error == EAGAIN || error == EWOULDBLOCK;
}

static int64_t now_ms(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    return -1;
  return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static et_dist5_status_t check_socket(int fd, struct stat *identity) {
  struct sockaddr_storage local, peer;
  socklen_t length = sizeof(local);
  int type;
  if (fd < 0 || fstat(fd, identity) != 0 ||
      getsockname(fd, (struct sockaddr *)&local, &length) != 0 ||
      local.ss_family != AF_UNIX)
    return ET_DIST5_ERR_INVALID_ARG;
  length = sizeof(peer);
  if (getpeername(fd, (struct sockaddr *)&peer, &length) != 0 ||
      peer.ss_family != AF_UNIX)
    return ET_DIST5_ERR_INVALID_ARG;
  length = sizeof(type);
  if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &length) != 0 ||
      type != SOCK_STREAM)
    return ET_DIST5_ERR_INVALID_ARG;
  return ET_DIST5_OK;
}

et_dist5_status_t et_dist5_comm_create(uint32_t abi_major, uint32_t abi_minor,
                                       uint32_t rank, uint32_t world_size,
                                       const int *fds, size_t fd_count,
                                       et_dist5_comm_t **out) {
  if (out == NULL)
    return ET_DIST5_ERR_INVALID_ARG;
  *out = NULL;
  if (abi_major != ET_DIST5_ABI_MAJOR || abi_minor > ET_DIST5_ABI_MINOR ||
      world_size == 0 || world_size > ET_DIST5_MAX_WORLD_SIZE ||
      rank >= world_size)
    return ET_DIST5_ERR_INVALID_ARG;
  const size_t required = rank == 0 ? world_size - 1u : 1u;
  if (fd_count != required || (required != 0 && fds == NULL))
    return ET_DIST5_ERR_INVALID_ARG;
  et_dist5_comm_t *comm = calloc(1, sizeof(*comm));
  if (comm == NULL)
    return ET_DIST5_ERR_ALLOC;
  comm->rank = rank;
  comm->world = world_size;
  for (uint32_t i = 0; i < world_size; ++i)
    comm->fd[i] = -1;
  struct stat identities[ET_DIST5_MAX_WORLD_SIZE];
  et_dist5_status_t status = ET_DIST5_OK;
  for (size_t i = 0; i < required; ++i) {
    int fd;
    do {
      fd = fcntl(fds[i], F_DUPFD_CLOEXEC, 0);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
      status = errno == EBADF ? ET_DIST5_ERR_INVALID_ARG : ET_DIST5_ERR_IO;
      break;
    }
    comm->fd[rank == 0 ? i + 1 : 0] = fd;
    status = check_socket(fd, &identities[i]);
    for (size_t j = 0; status == ET_DIST5_OK && j < i; ++j) {
      if (identities[i].st_dev == identities[j].st_dev &&
          identities[i].st_ino == identities[j].st_ino)
        status = ET_DIST5_ERR_INVALID_ARG;
    }
    if (status != ET_DIST5_OK)
      break;
  }
  if (status != ET_DIST5_OK) {
    et_dist5_comm_destroy(comm);
    return status;
  }
  *out = comm;
  return ET_DIST5_OK;
}

void et_dist5_comm_destroy(et_dist5_comm_t *comm) {
  if (comm == NULL)
    return;
  for (uint32_t i = 0; i < comm->world; ++i) {
    if (comm->fd[i] >= 0)
      close(comm->fd[i]);
  }
  free(comm);
}

static uint64_t rank_bit(uint32_t rank) { return UINT64_C(1) << rank; }

/* Watch the entire star during receipt, including when the expected peer
 * stalls. */
static et_dist5_status_t wait_ready(const struct exchange *x, uint32_t peer,
                                    short event, int watch_all) {
  struct pollfd pollfds[ET_DIST5_MAX_WORLD_SIZE];
  nfds_t n = 1;
  const short hangup = POLLHUP | POLLRDHUP;
  pollfds[0] = (struct pollfd){x->comm->fd[peer], event | POLLRDHUP, 0};
  if (watch_all && x->comm->rank == 0) {
    for (uint32_t r = 1; r < x->comm->world; ++r) {
      if (r != peer && !(x->released & rank_bit(r)))
        pollfds[n++] = (struct pollfd){x->comm->fd[r], POLLRDHUP, 0};
    }
  }
  for (;;) {
    const int64_t now = now_ms();
    if (now < 0)
      return ET_DIST5_ERR_IO;
    if (now >= x->deadline)
      return ET_DIST5_ERR_TIMEOUT;
    const int ready = poll(pollfds, n, (int)(x->deadline - now));
    if (ready < 0 && errno == EINTR)
      continue;
    if (ready < 0)
      return ET_DIST5_ERR_IO;
    if (ready == 0)
      continue;
    for (nfds_t i = 0; i < n; ++i) {
      if (pollfds[i].revents & POLLNVAL)
        return ET_DIST5_ERR_IO;
      /* A worker may have a complete final reply buffered before EOF. */
      if (i == 0 && x->comm->rank != 0 && event == POLLIN &&
          (pollfds[i].revents & POLLIN))
        continue;
      if (pollfds[i].revents & (hangup | POLLERR))
        return ET_DIST5_ERR_PEER_DISCONNECTED;
    }
    if (pollfds[0].revents & event)
      return ET_DIST5_OK;
  }
}

static et_dist5_status_t transfer(struct exchange *x, uint32_t peer,
                                  void *buffer, size_t size, int writing,
                                  int watch_all) {
  unsigned char *next = buffer;
  const int fd = x->comm->fd[peer];
  while (size != 0) {
    et_dist5_status_t status =
        wait_ready(x, peer, writing ? POLLOUT : POLLIN, watch_all);
    if (status != ET_DIST5_OK) {
      if (writing && next != buffer)
        x->incomplete |= rank_bit(peer);
      return status;
    }
    const ssize_t n = writing
                          ? send(fd, next, size, MSG_DONTWAIT | MSG_NOSIGNAL)
                          : recv(fd, next, size, MSG_DONTWAIT);
    if (n < 0 && retryable(errno))
      continue;
    if (n <= 0 && writing && next != buffer)
      x->incomplete |= rank_bit(peer);
    if (n == 0 ||
        (n < 0 && (errno == EPIPE || errno == ECONNRESET || errno == ENOTCONN)))
      return ET_DIST5_ERR_PEER_DISCONNECTED;
    if (n < 0)
      return ET_DIST5_ERR_IO;
    next += n;
    size -= (size_t)n;
  }
  return ET_DIST5_OK;
}

static void request(const struct exchange *x, uint32_t rank, uint64_t *words) {
  const struct call *c = x->call;
  const uint64_t fields[REQUEST_WORDS] = {
      WIRE_MAGIC,     x->comm->sequence, rank,
      x->comm->world, c->kind,           c->tag,
      c->count,       c->dtype,          c->op,
      c->root,        (uint64_t)x->local};
  memcpy(words, fields, sizeof(fields));
}

static et_dist5_status_t send_reply(struct exchange *x, uint32_t peer,
                                    enum phase phase, et_dist5_status_t status,
                                    int watch_all) {
  uint64_t words[REPLY_WORDS] = {x->comm->sequence, phase, (uint64_t)status};
  return transfer(x, peer, words, sizeof(words), 1, watch_all);
}

static et_dist5_status_t receive_reply(struct exchange *x, enum phase phase) {
  uint64_t words[REPLY_WORDS];
  const et_dist5_status_t status = transfer(x, 0, words, sizeof(words), 0, 0);
  if (status != ET_DIST5_OK)
    return status;
  if (words[0] != x->comm->sequence ||
      (words[1] != (uint64_t)phase && words[1] != ABORT) ||
      words[2] > ET_DIST5_ERR_ALLOC ||
      ((words[1] == ABORT) == (words[2] == ET_DIST5_OK)))
    return ET_DIST5_ERR_MISMATCH;
  return (et_dist5_status_t)words[2];
}

static int valid_local_status(uint64_t status) {
  return status == ET_DIST5_OK || status == ET_DIST5_ERR_INVALID_ARG ||
         status == ET_DIST5_ERR_NONFINITE || status == ET_DIST5_ERR_ALLOC;
}

/* Consistent priority even when more than one rank rejects its operand. */
static et_dist5_status_t merge_status(et_dist5_status_t a,
                                      et_dist5_status_t b) {
  if (a == ET_DIST5_ERR_INVALID_ARG || b == ET_DIST5_ERR_INVALID_ARG)
    return ET_DIST5_ERR_INVALID_ARG;
  if (a == ET_DIST5_ERR_NONFINITE || b == ET_DIST5_ERR_NONFINITE)
    return ET_DIST5_ERR_NONFINITE;
  return a != ET_DIST5_OK ? a : b;
}

static et_dist5_status_t check_requests(struct exchange *x) {
  et_dist5_status_t consensus = x->local;
  int mismatch = 0;
  for (uint32_t r = 1; r < x->comm->world; ++r) {
    uint64_t expected[REQUEST_WORDS], incoming[REQUEST_WORDS];
    const et_dist5_status_t status =
        transfer(x, r, incoming, sizeof(incoming), 0, 1);
    if (status != ET_DIST5_OK)
      return status;
    request(x, r, expected);
    if (memcmp(incoming, expected, 4 * sizeof(uint64_t)) != 0 ||
        !valid_local_status(incoming[REQUEST_WORDS - 1])) {
      mismatch = 1;
      continue;
    }
    const et_dist5_status_t local =
        (et_dist5_status_t)incoming[REQUEST_WORDS - 1];
    if (incoming[4] != expected[4] || incoming[5] != expected[5])
      mismatch = 1;
    else if (x->local != ET_DIST5_ERR_INVALID_ARG &&
             local != ET_DIST5_ERR_INVALID_ARG &&
             memcmp(incoming + 6, expected + 6, 4 * sizeof(uint64_t)) != 0)
      mismatch = 1;
    consensus = merge_status(consensus, local);
  }
  return mismatch ? ET_DIST5_ERR_MISMATCH : consensus;
}

static int has_input(const struct exchange *x, uint32_t rank) {
  return x->input != 0 && (x->call->kind != BROADCAST || rank == x->call->root);
}

static int finite_f32(const void *data, size_t count) {
  const unsigned char *bytes = data;
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits;
    memcpy(&bits, bytes + i * sizeof(bits), sizeof(bits));
    if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
      return 0;
  }
  return 1;
}

static et_dist5_status_t accumulate(struct exchange *x, const void *data,
                                    uint32_t rank) {
  const size_t size = x->input;
  if (x->call->kind == ALLGATHER || x->call->kind == BROADCAST) {
    const size_t offset = x->call->kind == ALLGATHER ? rank * size : 0;
    memcpy(x->stage + offset, data, size);
    return ET_DIST5_OK;
  }
  const et_dist5_op_t op = x->call->op;
  if (x->call->dtype == ET_DIST5_F32) {
    const size_t count = size / sizeof(float);
    if (!finite_f32(data, count))
      return ET_DIST5_ERR_NONFINITE;
    if (rank == 0) {
      memcpy(x->stage, data, size);
      return ET_DIST5_OK;
    }
    const unsigned char *input = data;
    /* The seed may have been copied from a caller's byte array. */
    for (size_t i = 0; i < count; ++i) {
      const size_t offset = i * sizeof(float);
      float accumulated, value;
      memcpy(&accumulated, x->stage + offset, sizeof(accumulated));
      memcpy(&value, input + offset, sizeof(value));
      if (op == ET_DIST5_SUM || op == ET_DIST5_MEAN) {
        accumulated += value;
        if (!isfinite(accumulated))
          return ET_DIST5_ERR_NONFINITE;
      } else if ((op == ET_DIST5_MIN && value < accumulated) ||
                 (op == ET_DIST5_MAX && value > accumulated)) {
        accumulated = value;
      }
      memcpy(x->stage + offset, &accumulated, sizeof(accumulated));
    }
  } else {
    if (rank == 0) {
      memcpy(x->stage, data, size);
      return ET_DIST5_OK;
    }
    const unsigned char *input = data;
    for (size_t i = 0; i < size / sizeof(int64_t); ++i) {
      const size_t offset = i * sizeof(int64_t);
      int64_t accumulated, value;
      memcpy(&accumulated, x->stage + offset, sizeof(accumulated));
      memcpy(&value, input + offset, sizeof(value));
      if (op == ET_DIST5_SUM) {
        if ((value > 0 && accumulated > INT64_MAX - value) ||
            (value < 0 && accumulated < INT64_MIN - value))
          return ET_DIST5_ERR_INVALID_ARG;
        accumulated += value;
      } else if ((op == ET_DIST5_MIN && value < accumulated) ||
                 (op == ET_DIST5_MAX && value > accumulated)) {
        accumulated = value;
      }
      memcpy(x->stage + offset, &accumulated, sizeof(accumulated));
    }
  }
  return ET_DIST5_OK;
}

static et_dist5_status_t gather_inputs(struct exchange *x) {
  for (uint32_t r = 0; r < x->comm->world; ++r) {
    if (!has_input(x, r))
      continue;
    const void *data = x->call->send;
    if (r != 0) {
      et_dist5_status_t status = send_reply(x, r, GRANT, ET_DIST5_OK, 1);
      if (status == ET_DIST5_OK)
        status = transfer(x, r, x->scratch, x->input, 0, 1);
      if (status != ET_DIST5_OK)
        return status;
      data = x->scratch;
    }
    const et_dist5_status_t status = accumulate(x, data, r);
    if (status != ET_DIST5_OK)
      return status;
  }
  if (x->call->op == ET_DIST5_MEAN) {
    for (size_t offset = 0; offset < x->total; offset += sizeof(float)) {
      float value;
      memcpy(&value, x->stage + offset, sizeof(value));
      value /= (float)x->comm->world;
      memcpy(x->stage + offset, &value, sizeof(value));
    }
  }
  return ET_DIST5_OK;
}

static et_dist5_status_t compute(struct exchange *x) {
  if (x->call->op == 0 || x->call->dtype != ET_DIST5_F32)
    return gather_inputs(x);
  fenv_t saved;
  if (fegetenv(&saved) != 0)
    return ET_DIST5_ERR_IO;
  et_dist5_status_t status = ET_DIST5_ERR_IO;
  if (fesetenv(FE_DFL_ENV) == 0)
    status = gather_inputs(x);
  return fesetenv(&saved) == 0 ? status : ET_DIST5_ERR_IO;
}

static int recoverable(et_dist5_status_t status) {
  return status == ET_DIST5_ERR_INVALID_ARG ||
         status == ET_DIST5_ERR_NONFINITE || status == ET_DIST5_ERR_ALLOC;
}

static et_dist5_status_t fan_out_status(struct exchange *x,
                                        et_dist5_status_t status) {
  et_dist5_status_t failure = ET_DIST5_OK;
  for (uint32_t r = 1; r < x->comm->world; ++r) {
    const et_dist5_status_t sent = send_reply(x, r, ABORT, status, 0);
    if (failure == ET_DIST5_OK)
      failure = sent;
    if (sent == ET_DIST5_OK)
      x->released |= rank_bit(r);
  }
  return failure == ET_DIST5_OK ? status : failure;
}

static et_dist5_status_t coordinate(struct exchange *x) {
  et_dist5_status_t status = check_requests(x);
  if (status == ET_DIST5_OK)
    status = compute(x);
  if (recoverable(status) || status == ET_DIST5_ERR_MISMATCH)
    return fan_out_status(x, status);
  if (status != ET_DIST5_OK)
    return status;

  for (uint32_t r = 1; r < x->comm->world; ++r) {
    status = send_reply(x, r, RESULT, ET_DIST5_OK, 1);
    if (status != ET_DIST5_OK)
      return status;
    if (x->output != 0) {
      const size_t offset = x->call->kind == REDUCE_SCATTER ? r * x->output : 0;
      status = transfer(x, r, x->stage + offset, x->output, 1, 1);
      if (status != ET_DIST5_OK) {
        x->incomplete |= rank_bit(r);
        return status;
      }
    }
  }
  for (uint32_t r = 1; r < x->comm->world; ++r) {
    uint64_t ready;
    status = transfer(x, r, &ready, sizeof(ready), 0, 1);
    if (status != ET_DIST5_OK)
      return status;
    if (ready != x->comm->sequence)
      return fan_out_status(x, ET_DIST5_ERR_MISMATCH);
  }
  for (uint32_t r = 1; r < x->comm->world; ++r) {
    status = send_reply(x, r, COMMIT, ET_DIST5_OK, 1);
    if (status != ET_DIST5_OK)
      return status;
    x->released |= rank_bit(r);
  }
  return ET_DIST5_OK;
}

static et_dist5_status_t participate(struct exchange *x) {
  uint64_t words[REQUEST_WORDS];
  request(x, x->comm->rank, words);
  et_dist5_status_t status = transfer(x, 0, words, sizeof(words), 1, 0);
  if (status != ET_DIST5_OK)
    return status;
  if (x->local == ET_DIST5_OK && has_input(x, x->comm->rank)) {
    status = receive_reply(x, GRANT);
    if (status != ET_DIST5_OK)
      return status;
    /* transfer only passes this buffer to send(). */
    status = transfer(x, 0, (void *)x->call->send, x->input, 1, 0);
    if (status != ET_DIST5_OK)
      return status;
  }
  status = receive_reply(x, RESULT);
  if (status != ET_DIST5_OK)
    return status;
  if (x->local != ET_DIST5_OK)
    return ET_DIST5_ERR_MISMATCH;
  status = transfer(x, 0, x->stage, x->output, 0, 0);
  if (status != ET_DIST5_OK)
    return status;
  uint64_t ready = x->comm->sequence;
  status = transfer(x, 0, &ready, sizeof(ready), 1, 0);
  return status == ET_DIST5_OK ? receive_reply(x, COMMIT) : status;
}

static void fail_transport(struct exchange *x, et_dist5_status_t status) {
  x->comm->fatal = status;
  for (uint32_t r = 0; r < x->comm->world; ++r) {
    const int fd = x->comm->fd[r];
    if (fd < 0)
      continue;
    if (x->comm->rank == 0 && !((x->released | x->incomplete) & rank_bit(r))) {
      const uint64_t words[REPLY_WORDS] = {x->comm->sequence, ABORT,
                                           (uint64_t)status};
      (void)send(fd, words, sizeof(words), MSG_DONTWAIT | MSG_NOSIGNAL);
    }
    (void)shutdown(fd, SHUT_RDWR);
  }
}

static int invalid_span(const void *buffer, size_t bytes) {
  return buffer == NULL || (uintptr_t)buffer > UINTPTR_MAX - bytes;
}

static int overlaps(const void *send, size_t input, const void *recv,
                    size_t output) {
  const uintptr_t a = (uintptr_t)send, b = (uintptr_t)recv;
  return a < b + output && b < a + input;
}

static et_dist5_status_t prepare(struct exchange *x) {
  const struct call *c = x->call;
  if (c->kind == BARRIER)
    return ET_DIST5_OK;
  const int reduce =
      c->kind == ALLREDUCE || c->kind == REDUCE_SCATTER || c->kind == SYNC;
  const size_t element =
      c->dtype == ET_DIST5_F32 ? sizeof(float) : sizeof(int64_t);
  const size_t factor =
      c->kind == ALLGATHER || c->kind == REDUCE_SCATTER ? x->comm->world : 1;
  if ((c->dtype != ET_DIST5_F32 && c->dtype != ET_DIST5_I64) ||
      (reduce && (c->op < ET_DIST5_SUM || c->op > ET_DIST5_MAX ||
                  (c->dtype == ET_DIST5_I64 && c->op == ET_DIST5_MEAN))) ||
      c->root >= x->comm->world || c->count == 0 ||
      c->count > ET_DIST5_MAX_BUFFER_BYTES / element / factor)
    return ET_DIST5_ERR_INVALID_ARG;

  const size_t bytes = c->count * element;
  x->total = bytes * factor;
  x->input = c->kind == REDUCE_SCATTER ? x->total : bytes;
  x->output = c->kind == ALLGATHER ? x->total : bytes;
  if (invalid_span(c->recv, x->output) ||
      (has_input(x, x->comm->rank) && invalid_span(c->send, x->input)))
    return ET_DIST5_ERR_INVALID_ARG;
  const int in_place =
      c->kind == BROADCAST ||
      ((c->kind == ALLREDUCE || c->kind == SYNC) && c->send == c->recv);
  if (!in_place && overlaps(c->send, x->input, c->recv, x->output))
    return ET_DIST5_ERR_INVALID_ARG;
  if (c->kind == SYNC && !finite_f32(c->send, c->count))
    return ET_DIST5_ERR_NONFINITE;

  const int coordinator = x->comm->rank == 0;
  const int needs_scratch = coordinator && x->comm->world > 1 &&
                            (c->kind != BROADCAST || c->root != 0);
  x->stage = malloc(coordinator ? x->total : x->output);
  if (needs_scratch)
    x->scratch = malloc(x->input);
  if (x->stage == NULL || (needs_scratch && x->scratch == NULL))
    return ET_DIST5_ERR_ALLOC;
  return ET_DIST5_OK;
}

static et_dist5_status_t execute(et_dist5_comm_t *comm, const struct call *call,
                                 int timeout_ms) {
  if (comm == NULL || timeout_ms <= 0)
    return ET_DIST5_ERR_INVALID_ARG;
  if (comm->fatal != ET_DIST5_OK)
    return comm->fatal;
  const int64_t now = now_ms();
  if (now < 0 || comm->sequence == UINT64_MAX)
    return ET_DIST5_ERR_IO;
  struct exchange x = {
      .comm = comm, .call = call, .deadline = now + timeout_ms};
  x.local = prepare(&x);
  et_dist5_status_t status = comm->rank == 0 ? coordinate(&x) : participate(&x);
  if (status == ET_DIST5_OK && comm->world == 1) {
    const int64_t end = now_ms();
    if (end < 0 || end >= x.deadline)
      status = end < 0 ? ET_DIST5_ERR_IO : ET_DIST5_ERR_TIMEOUT;
  }
  if (status == ET_DIST5_OK) {
    if (x.output != 0)
      memcpy(call->recv, x.stage, x.output);
  } else if (!recoverable(status)) {
    fail_transport(&x, status);
  }
  ++comm->sequence;
  free(x.scratch);
  free(x.stage);
  return status;
}

et_dist5_status_t et_dist5_barrier(et_dist5_comm_t *comm, uint64_t tag,
                                   int timeout_ms) {
  const struct call call = {.kind = BARRIER, .tag = tag};
  return execute(comm, &call, timeout_ms);
}

et_dist5_status_t et_dist5_broadcast(et_dist5_comm_t *comm, void *buf,
                                     size_t count, et_dist5_dtype_t dtype,
                                     uint32_t root_rank, uint64_t tag,
                                     int timeout_ms) {
  const struct call call = {BROADCAST, tag,       count, dtype,
                            0,         root_rank, buf,   buf};
  return execute(comm, &call, timeout_ms);
}

et_dist5_status_t et_dist5_allreduce(et_dist5_comm_t *comm, const void *sendbuf,
                                     void *recvbuf, size_t count,
                                     et_dist5_dtype_t dtype, et_dist5_op_t op,
                                     uint64_t tag, int timeout_ms) {
  const struct call call = {ALLREDUCE, tag, count,   dtype,
                            op,        0,   sendbuf, recvbuf};
  return execute(comm, &call, timeout_ms);
}

et_dist5_status_t et_dist5_allgather(et_dist5_comm_t *comm, const void *sendbuf,
                                     size_t send_count, void *recvbuf,
                                     et_dist5_dtype_t dtype, uint64_t tag,
                                     int timeout_ms) {
  const struct call call = {ALLGATHER, tag, send_count, dtype,
                            0,         0,   sendbuf,    recvbuf};
  return execute(comm, &call, timeout_ms);
}

et_dist5_status_t et_dist5_reduce_scatter(et_dist5_comm_t *comm,
                                          const void *sendbuf, void *recvbuf,
                                          size_t recv_count,
                                          et_dist5_dtype_t dtype,
                                          et_dist5_op_t op, uint64_t tag,
                                          int timeout_ms) {
  const struct call call = {REDUCE_SCATTER, tag,    recv_count, dtype, op, 0,
                            sendbuf,        recvbuf};
  return execute(comm, &call, timeout_ms);
}

et_dist5_status_t et_dist5_sync_gradients_f32(et_dist5_comm_t *comm,
                                              const float *local, float *mean,
                                              size_t count, uint64_t tag,
                                              int timeout_ms) {
  const struct call call = {SYNC,          tag, count, ET_DIST5_F32,
                            ET_DIST5_MEAN, 0,   local, mean};
  return execute(comm, &call, timeout_ms);
}
