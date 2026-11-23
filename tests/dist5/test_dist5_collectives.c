#define _POSIX_C_SOURCE 200809L
#include "../mod5/mod5_test.h"
#include "eshkol_transformer/dist5_collectives_abi.h"

#include <dirent.h>
#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON

enum { MAX_RANKS = 4, TIMEOUT = 2000 };
static unsigned process_groups;

#define STATUS(call, expected)                                                 \
  do {                                                                         \
    const et_dist5_status_t observed = (call);                                 \
    if (observed != (expected)) {                                              \
      fprintf(stderr, "%s:%d: %s returned %d (expected %d)\n", __FILE__,       \
              __LINE__, #call, observed, (expected));                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static et_dist5_status_t make_comm(unsigned rank, unsigned world,
                                   const int *fds, size_t count,
                                   et_dist5_comm_t **comm) {
  return et_dist5_comm_create(ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR, rank,
                              world, fds, count, comm);
}

static void sleep_ms(long ms) {
  struct timespec delay = {ms / 1000, ms % 1000 * 1000000};
  while (nanosleep(&delay, &delay) < 0)
    CHECK(errno == EINTR);
}

static int64_t monotonic_ms(void) {
  struct timespec ts;
  CHECK(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
  return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void move_bytes(int fd, void *buffer, size_t size, int writing) {
  unsigned char *p = buffer;
  while (size != 0) {
    const ssize_t n = writing ? write(fd, p, size) : read(fd, p, size);
    if (n < 0 && errno == EINTR)
      continue;
    CHECK(n > 0);
    p += n;
    size -= (size_t)n;
  }
}

typedef void (*rank_test)(unsigned rank, unsigned world, const int *fds,
                          size_t count, unsigned variant, int report_fd);

static void run_group(unsigned world, rank_test test, unsigned variant,
                      void *report, size_t report_size, unsigned killed_rank) {
  int pairs[MAX_RANKS - 1][2], trace[2] = {-1, -1};
  pid_t children[MAX_RANKS];
  CHECK(fflush(NULL) == 0);
  if (report != NULL)
    CHECK(pipe(trace) == 0);
  for (unsigned r = 1; r < world; ++r) {
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, pairs[r - 1]) == 0);
    const int capacity = 1024;
    for (unsigned end = 0; end < 2; ++end)
      CHECK(setsockopt(pairs[r - 1][end], SOL_SOCKET, SO_SNDBUF, &capacity,
                       sizeof(capacity)) == 0);
  }
  for (unsigned rank = 0; rank < world; ++rank) {
    children[rank] = fork();
    CHECK(children[rank] >= 0);
    if (children[rank] != 0)
      continue;
    alarm(8);
    if (report != NULL) {
      CHECK(close(trace[0]) == 0);
      if (rank != 0)
        CHECK(close(trace[1]) == 0);
    }
    int fds[MAX_RANKS - 1];
    size_t count = 0;
    for (unsigned r = 1; r < world; ++r) {
      if (rank == 0)
        fds[count++] = pairs[r - 1][0];
      else
        CHECK(close(pairs[r - 1][0]) == 0);
      if (rank == r)
        fds[count++] = pairs[r - 1][1];
      else
        CHECK(close(pairs[r - 1][1]) == 0);
    }
    test(rank, world, fds, count, variant, rank == 0 ? trace[1] : -1);
    exit(EXIT_SUCCESS);
  }
  for (unsigned r = 1; r < world; ++r) {
    CHECK(close(pairs[r - 1][0]) == 0);
    CHECK(close(pairs[r - 1][1]) == 0);
  }
  if (report != NULL) {
    CHECK(close(trace[1]) == 0);
    move_bytes(trace[0], report, report_size, 0);
    CHECK(close(trace[0]) == 0);
  }
  int all_passed = 1;
  for (unsigned rank = 0; rank < world; ++rank) {
    int status;
    pid_t observed;
    do {
      observed = waitpid(children[rank], &status, 0);
    } while (observed < 0 && errno == EINTR);
    CHECK(observed == children[rank]);
    const int passed = rank == killed_rank
                           ? WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL
                           : WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (!passed)
      fprintf(stderr, "rank %u/%u failed (case %u, wait status %#x)\n", rank,
              world, variant, status);
    all_passed &= passed;
  }
  CHECK(all_passed);
  ++process_groups;
}

static et_dist5_comm_t *open_rank(unsigned rank, unsigned world, const int *fds,
                                  size_t count) {
  et_dist5_comm_t *comm;
  STATUS(make_comm(rank, world, count == 0 ? NULL : fds, count, &comm),
         ET_DIST5_OK);
  for (size_t i = 0; i < count; ++i)
    CHECK(close(fds[i]) == 0);
  return comm;
}

static void collectives(unsigned rank, unsigned world, const int *fds,
                        size_t count, unsigned variant, int report_fd) {
  (void)report_fd;
  et_dist5_comm_t *comm = open_rank(rank, world, fds, count);
  sleep_ms((long)(variant == 0 ? rank : world - rank) * 3);
  STATUS(et_dist5_barrier(comm, UINT64_MAX, TIMEOUT), ET_DIST5_OK);

  enum { LARGE = 16384 };
  int64_t large[LARGE];
  for (unsigned root = 0; root < world; ++root) {
    memset(large, 0xa5, sizeof(large));
    if (rank == root) {
      for (size_t i = 0; i < ARRAY_COUNT(large); ++i)
        large[i] = (int64_t)(17u * i + root);
    }
    STATUS(et_dist5_broadcast(comm, large, LARGE, ET_DIST5_I64, root, root,
                              TIMEOUT),
           ET_DIST5_OK);
    for (size_t i = 0; i < ARRAY_COUNT(large); ++i)
      CHECK(large[i] == (int64_t)(17u * i + root));
  }

  const float cancellation[] = {0x1p24f, 1.0f, -0x1p24f, 3.0f};
  const float local[] = {cancellation[rank], (float)rank - 1.0f,
                         rank == 0 ? -0.0f : 0.0f};
  const float sum = world == 4 ? 3.0f : 0x1p24f;
  const float ramp = (float)(world * (world - 1)) / 2.0f - (float)world;
  const float high = world == 1 ? -1.0f : (float)world - 2.0f;
  const float expected[][3] = {
      {sum, ramp, world == 1 ? -0.0f : 0.0f},
      {sum / (float)world, ramp / (float)world, world == 1 ? -0.0f : 0.0f},
      {world == 4 ? -0x1p24f : cancellation[world - 1], -1.0f, -0.0f},
      {0x1p24f, high, -0.0f}};
  /* An upward-rounded rank-0 sum of 2^24 + 1 would produce different bits. */
  CHECK(fesetround(FE_UPWARD) == 0);
  for (unsigned op = ET_DIST5_SUM; op <= ET_DIST5_MAX; ++op) {
    unsigned char reduced[sizeof(local)];
    memcpy(reduced, local, sizeof(reduced));
    STATUS(et_dist5_allreduce(comm, reduced, reduced, 3, ET_DIST5_F32, op, 12,
                              TIMEOUT),
           ET_DIST5_OK);
    CHECK(memcmp(reduced, expected[op - ET_DIST5_SUM], sizeof(reduced)) == 0);
    CHECK(fegetround() == FE_UPWARD);
  }
  CHECK(fesetround(FE_TONEAREST) == 0);

  const int64_t integers[] = {(int64_t)rank + 1, -((int64_t)rank + 1)};
  const int64_t triangular = (int64_t)world * (world + 1) / 2;
  const int64_t integer_answers[][2] = {
      {triangular, -triangular}, {1, -(int64_t)world}, {(int64_t)world, -1}};
  const unsigned integer_ops[] = {ET_DIST5_SUM, ET_DIST5_MIN, ET_DIST5_MAX};
  unsigned char packed[sizeof(integers)];
  memcpy(packed, integers, sizeof(packed));
  for (size_t i = 0; i < ARRAY_COUNT(integer_ops); ++i) {
    unsigned char reduced[sizeof(integers)];
    STATUS(et_dist5_allreduce(comm, packed, reduced, 2, ET_DIST5_I64,
                              integer_ops[i], 11, TIMEOUT),
           ET_DIST5_OK);
    CHECK(memcmp(reduced, integer_answers[i], sizeof(reduced)) == 0);
  }

  int64_t gathered[MAX_RANKS * 2];
  STATUS(et_dist5_allgather(comm, integers, 2, gathered, ET_DIST5_I64, 11,
                            TIMEOUT),
         ET_DIST5_OK);
  for (unsigned r = 0; r < world; ++r) {
    CHECK(gathered[r * 2] == (int64_t)r + 1);
    CHECK(gathered[r * 2 + 1] == -((int64_t)r + 1));
  }

  int64_t to_scatter[MAX_RANKS * 2], shard[2];
  float to_scatter_f32[MAX_RANKS * 2], shard_f32[2];
  for (unsigned r = 0; r < world; ++r) {
    to_scatter[r * 2] = 7 * (int64_t)r + integers[0];
    to_scatter[r * 2 + 1] = -7 * (int64_t)r + integers[1];
    to_scatter_f32[r * 2] = cancellation[rank];
    to_scatter_f32[r * 2 + 1] = (float)r - (float)rank;
  }
  STATUS(et_dist5_reduce_scatter(comm, to_scatter, shard, 2, ET_DIST5_I64,
                                 ET_DIST5_SUM, 11, TIMEOUT),
         ET_DIST5_OK);
  const int64_t offset = 7 * (int64_t)rank * world + triangular;
  CHECK(shard[0] == offset && shard[1] == -offset);
  STATUS(et_dist5_reduce_scatter(comm, to_scatter_f32, shard_f32, 2,
                                 ET_DIST5_F32, ET_DIST5_MEAN, 11, TIMEOUT),
         ET_DIST5_OK);
  const float expected_shard[] = {sum / (float)world,
                                  (float)rank - (float)(world - 1) / 2.0f};
  CHECK(memcmp(shard_f32, expected_shard, sizeof(shard_f32)) == 0);
  STATUS(et_dist5_barrier(comm, UINT64_MAX, TIMEOUT), ET_DIST5_OK);
  et_dist5_comm_destroy(comm);
}

static size_t count_open_fds(void) {
  DIR *dir = opendir("/proc/self/fd");
  CHECK(dir != NULL);
  size_t count = 0;
  struct dirent *entry;
  errno = 0;
  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_name[0] != '.')
      ++count;
  }
  CHECK(errno == 0 && count > 0);
  CHECK(closedir(dir) == 0);
  return count - 1; /* Exclude the directory descriptor. */
}

#define INVALID_BOOTSTRAP(call)                                                \
  do {                                                                        \
    const size_t before = count_open_fds();                                    \
    STATUS(call, ET_DIST5_ERR_INVALID_ARG);                                    \
    CHECK(count_open_fds() == before);                                         \
  } while (0)

static void malformed_bootstrap(void) {
  et_dist5_comm_t *comm = NULL;
  int pair[2], dgram[2], other[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0);
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, dgram) == 0);
  CHECK(pipe(other) == 0);
  INVALID_BOOTSTRAP(make_comm(0, 0, NULL, 0, &comm));
  INVALID_BOOTSTRAP(make_comm(1, 1, NULL, 0, &comm));
  INVALID_BOOTSTRAP(make_comm(0, ET_DIST5_MAX_WORLD_SIZE + 1, NULL, 0, &comm));
  INVALID_BOOTSTRAP(make_comm(0, 2, NULL, 1, &comm));
  INVALID_BOOTSTRAP(make_comm(0, 2, pair, 0, &comm));
  INVALID_BOOTSTRAP(make_comm(0, 1, NULL, 0, NULL));
  INVALID_BOOTSTRAP(et_dist5_comm_create(2, 0, 0, 1, NULL, 0, &comm));
  INVALID_BOOTSTRAP(et_dist5_comm_create(1, 1, 0, 1, NULL, 0, &comm));
  INVALID_BOOTSTRAP(make_comm(0, 2, dgram, 1, &comm));
  INVALID_BOOTSTRAP(make_comm(0, 2, other, 1, &comm));
  const int dup_fd = dup(pair[0]);
  CHECK(dup_fd >= 0);
  const int dup_fds[] = {pair[0], dup_fd};
  INVALID_BOOTSTRAP(make_comm(0, 3, dup_fds, 2, &comm));
  CHECK(close(dup_fd) == 0);
  INVALID_BOOTSTRAP(make_comm(0, 2, &dup_fd, 1, &comm));
  const int bad_fds[] = {pair[0], -1};
  INVALID_BOOTSTRAP(make_comm(0, 3, bad_fds, 2, &comm));
  CHECK(comm == NULL);
  STATUS(et_dist5_barrier(NULL, 0, TIMEOUT), ET_DIST5_ERR_INVALID_ARG);
  et_dist5_comm_destroy(NULL);
  for (unsigned i = 0; i < 2; ++i) {
    CHECK(close(pair[i]) == 0);
    CHECK(close(dgram[i]) == 0);
    CHECK(close(other[i]) == 0);
  }
}
#undef INVALID_BOOTSTRAP

static void malformed_calls(unsigned rank, unsigned world, const int *fds,
                            size_t count, unsigned variant, int report_fd) {
  (void)variant;
  (void)report_fd;
  et_dist5_comm_t *comm = open_rank(rank, world, fds, count);
  const int64_t input[] = {11, 22, 33, 44};
  int64_t out[] = {5, 6, 7, 8, 9};
  const int64_t sentinel[] = {5, 6, 7, 8, 9};
  const int bad = rank == world - 1;
  STATUS(et_dist5_barrier(comm, 0, 0), ET_DIST5_ERR_INVALID_ARG);
  STATUS(et_dist5_broadcast(comm, bad ? NULL : out, 1, ET_DIST5_I64, 0, 0,
                            TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  STATUS(et_dist5_broadcast(comm, out, 1, ET_DIST5_I64, bad ? world : 0, 0,
                            TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  STATUS(et_dist5_allreduce(comm, bad ? NULL : input, out, 1, ET_DIST5_I64,
                            ET_DIST5_SUM, 0, TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  STATUS(et_dist5_allreduce(comm, input, bad ? NULL : out, 1, ET_DIST5_I64,
                            ET_DIST5_SUM, 0, TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  STATUS(et_dist5_allreduce(comm, input, out, bad ? 0 : 1, ET_DIST5_I64,
                            ET_DIST5_SUM, 0, TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  STATUS(et_dist5_allreduce(comm, input, out, bad ? SIZE_MAX : 1, ET_DIST5_I64,
                            ET_DIST5_SUM, 0, TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  STATUS(et_dist5_allreduce(comm, input, out, 1, ET_DIST5_I64,
                            bad ? ET_DIST5_MEAN : ET_DIST5_MIN, 0, TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  STATUS(et_dist5_allgather(comm, bad ? out : input, 1, out, ET_DIST5_I64, 0,
                            TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  STATUS(et_dist5_allgather(comm, bad ? out + 1 : input, 1, out, ET_DIST5_I64,
                            0, TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  STATUS(et_dist5_allgather(
             comm, input, bad ? ET_DIST5_MAX_BUFFER_BYTES / sizeof(int64_t) : 1,
             out, ET_DIST5_I64, 0, TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  STATUS(et_dist5_reduce_scatter(comm, bad ? out : input, out, 1, ET_DIST5_I64,
                                 ET_DIST5_SUM, 0, TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  const int64_t big = rank == 0 ? INT64_MAX : 1;
  STATUS(et_dist5_allreduce(comm, &big, out, 1, ET_DIST5_I64, ET_DIST5_SUM, 0,
                            TIMEOUT),
         ET_DIST5_ERR_INVALID_ARG);
  const float overflow = FLT_MAX;
  STATUS(et_dist5_allreduce(comm, &overflow, out, 1, ET_DIST5_F32, ET_DIST5_SUM,
                            0, TIMEOUT),
         ET_DIST5_ERR_NONFINITE);
  CHECK(memcmp(out, sentinel, sizeof(out)) == 0);
  STATUS(et_dist5_barrier(comm, 0, TIMEOUT), ET_DIST5_OK);
  et_dist5_comm_destroy(comm);
}

enum fault {
  TAG_MISMATCH,
  COUNT_MISMATCH,
  DTYPE_MISMATCH,
  OP_MISMATCH,
  KIND_MISMATCH,
  ROOT_MISMATCH,
  INVALID_KIND_MISMATCH,
  INVALID_TAG_MISMATCH,
  CRASH_INPUT,
  CRASH_READY,
  CRASH_COORDINATOR,
  STALL_METADATA,
  STALL_INPUT,
  STALL_READY,
  TRICKLE_INPUT
};

static void mismatch(unsigned rank, unsigned world, const int *fds,
                     size_t count, unsigned variant, int report_fd) {
  (void)report_fd;
  et_dist5_comm_t *comm = open_rank(rank, world, fds, count);
  const int bad = rank == world - 1;
  const uint64_t tag =
      bad && (variant == TAG_MISMATCH || variant == INVALID_TAG_MISMATCH) ? 9 : 8;
  const size_t n = bad && variant == COUNT_MISMATCH ? 2 : 1;
  const unsigned dtype =
      bad && variant == DTYPE_MISMATCH ? ET_DIST5_F32 : ET_DIST5_I64;
  const unsigned op =
      bad && variant == OP_MISMATCH ? ET_DIST5_MIN : ET_DIST5_SUM;
  int64_t in[] = {7, 8}, out[] = {14, 15}, sentinel[] = {14, 15};
  if ((bad && variant == KIND_MISMATCH) ||
      (rank == 0 && variant == INVALID_KIND_MISMATCH)) {
    STATUS(et_dist5_barrier(comm, tag, TIMEOUT), ET_DIST5_ERR_MISMATCH);
  } else if (variant == ROOT_MISMATCH) {
    STATUS(et_dist5_broadcast(comm, out, n, dtype, (unsigned)bad, tag, TIMEOUT),
           ET_DIST5_ERR_MISMATCH);
  } else {
    const int invalid = (bad && variant == INVALID_KIND_MISMATCH) ||
                        (rank == 0 && variant == INVALID_TAG_MISMATCH);
    STATUS(et_dist5_allreduce(comm, in, invalid ? NULL : out, n, dtype, op, tag,
                              TIMEOUT),
           ET_DIST5_ERR_MISMATCH);
  }
  CHECK(memcmp(out, sentinel, sizeof(out)) == 0);
  STATUS(et_dist5_barrier(comm, 0, TIMEOUT), ET_DIST5_ERR_MISMATCH);
  et_dist5_comm_destroy(comm);
}

/* A real peer participates up to the failure point using the documented wire.
 */
static void fault_worker(unsigned rank, unsigned world, int fd,
                         unsigned variant) {
  uint64_t request[] = {UINT64_C(0x4554444953540501),
                        0,
                        rank,
                        world,
                        3,
                        10,
                        2,
                        ET_DIST5_I64,
                        ET_DIST5_SUM,
                        0,
                        ET_DIST5_OK};
  uint64_t reply[3];
  int64_t data[] = {10, 11};
  move_bytes(fd, request, sizeof(request), 1);
  move_bytes(fd, reply, sizeof(reply), 0);
  CHECK(reply[0] == 0 && reply[1] == 1 && reply[2] == ET_DIST5_OK);
  if (variant == CRASH_INPUT || variant == STALL_INPUT) {
    move_bytes(fd, data, 1, 1);
    if (variant == STALL_INPUT)
      sleep_ms(650);
    _exit(0);
  }
  if (variant == TRICKLE_INPUT) {
    /* Wait for the coordinator's reply between one-byte writes. */
    const unsigned char *payload = (const unsigned char *)data;
    size_t sent = 0;
    struct pollfd response = {fd, POLLIN, 0};
    while (sent < sizeof(data)) {
      const ssize_t n = send(fd, payload + sent, 1, MSG_NOSIGNAL);
      if (n < 0 && (errno == EPIPE || errno == ECONNRESET))
        break;
      CHECK(n == 1);
      ++sent;
      const int ready = poll(&response, 1, 35);
      CHECK(ready >= 0);
      if (ready != 0)
        break;
    }
    move_bytes(fd, reply, sizeof(reply), 0);
    CHECK(reply[0] == 0 && reply[1] == 4 && reply[2] == ET_DIST5_ERR_TIMEOUT);
    CHECK(sent >= 2 && sent < sizeof(data));
    _exit(0);
  }
  move_bytes(fd, data, sizeof(data), 1);
  move_bytes(fd, reply, sizeof(reply), 0);
  CHECK(reply[0] == 0 && reply[1] == 2 && reply[2] == ET_DIST5_OK);
  move_bytes(fd, data, sizeof(data), 0);
  if (variant == STALL_READY) {
    sleep_ms(650);
    _exit(0);
  }
  CHECK(raise(SIGKILL) == 0);
}

static void transfer_fault(unsigned rank, unsigned world, const int *fds,
                           size_t count, unsigned variant, int report_fd) {
  (void)report_fd;
  if (rank == 2 &&
      (variant == CRASH_INPUT || variant == CRASH_READY ||
       variant == STALL_INPUT || variant == STALL_READY ||
       variant == TRICKLE_INPUT))
    fault_worker(rank, world, fds[0], variant);
  if (rank == 0 && variant == CRASH_COORDINATOR) {
    uint64_t request[11];
    move_bytes(fds[0], request, sizeof(request), 0);
    CHECK(request[0] == UINT64_C(0x4554444953540501));
    CHECK(raise(SIGKILL) == 0);
  }

  et_dist5_comm_t *comm = open_rank(rank, world, fds, count);
  if (variant == STALL_METADATA && rank == 1) {
    sleep_ms(650);
  } else {
    const int stalled = variant >= STALL_METADATA;
    const int64_t start = monotonic_ms();
    const int timeout = stalled ? (rank == 0 ? 220 : 1300) : TIMEOUT;
    const et_dist5_status_t expected =
        stalled ? ET_DIST5_ERR_TIMEOUT : ET_DIST5_ERR_PEER_DISCONNECTED;
    int64_t in[] = {2, 3}, out[] = {4, 5}, sentinel[] = {4, 5};
    STATUS(et_dist5_allreduce(comm, in, out, 2, ET_DIST5_I64, ET_DIST5_SUM, 10,
                              timeout),
           expected);
    CHECK(memcmp(out, sentinel, sizeof(out)) == 0);
    const int64_t elapsed = monotonic_ms() - start;
    CHECK(elapsed < 1000);
    if (stalled && rank == 0)
      CHECK(elapsed >= 190);
    STATUS(et_dist5_barrier(comm, 11, TIMEOUT), expected);
  }
  et_dist5_comm_destroy(comm);
}

enum {
  HEADS = 2,
  HEAD_WIDTH = 2,
  WIDTH = HEADS * HEAD_WIDTH,
  TOKENS = 2,
  ELEMENTS = WIDTH * TOKENS,
  KV_ELEMENTS = TOKENS * HEAD_WIDTH,
  MICROS = 8,
  STEPS = 16
};
enum parameter {
  GAMMA1,
  QUERY,
  KEY,
  VALUE,
  OUT,
  GAMMA2,
  GATE,
  UP,
  DOWN,
  PARAMS
};
enum { WEIGHTS = PARAMS * WIDTH };
static const uint64_t ntd[] = {1, TOKENS, WIDTH};

static void rms(const et_kernel_runtime *runtime, float *x, float *gamma,
                float *dy, float *y, float *dgamma) {
  const uint64_t shape[] = {WIDTH};
  float epsilon = 0.125f;
  const et_kernel_request_v1 request = {sizeof(request),
                                        dy ? "rmsnorm.backward"
                                           : "rmsnorm.forward",
                                        "f32",
                                        "cpu",
                                        ARRAY_COUNT(ntd),
                                        ntd,
                                        1,
                                        {0}};
  const et_kernel_tensor_view_v1 in[] = {
      test_view(x, "f32", 3, ntd), test_view(gamma, "f32", 1, shape),
      test_view(&epsilon, "f32", 0, NULL), test_view(dy, "f32", 3, ntd)};
  et_kernel_tensor_view_v1 out[] = {test_view(y, "f32", 3, ntd),
                                    test_view(dgamma, "f32", 1, shape)};
  const et_kernel_call_v1 call =
      test_call("kernel.rmsnorm", &request, dy ? 4 : 3, in, dy ? 2 : 1, out);
  dispatch(runtime, &call);
}

static void glu(const et_kernel_runtime *runtime, float *gate, float *up,
                float *dy, float *y, float *dup) {
  const et_kernel_request_v1 request = {sizeof(request),
                                        dy ? "swiglu.backward"
                                           : "swiglu.forward",
                                        "f32",
                                        "cpu",
                                        ARRAY_COUNT(ntd),
                                        ntd,
                                        1,
                                        {0}};
  const et_kernel_tensor_view_v1 in[] = {test_view(gate, "f32", 3, ntd),
                                         test_view(up, "f32", 3, ntd),
                                         test_view(dy, "f32", 3, ntd)};
  et_kernel_tensor_view_v1 out[] = {test_view(y, "f32", 3, ntd),
                                    test_view(dup, "f32", 3, ntd)};
  const et_kernel_call_v1 call =
      test_call("kernel.swiglu", &request, dy ? 3 : 2, in, dy ? 2 : 1, out);
  dispatch(runtime, &call);
}

static void attend(const et_kernel_runtime *runtime, const float *q,
                   const float *k, const float *v, const float *dy, float *y,
                   float *dk, float *dv) {
  const uint64_t shape[] = {1, HEADS, 1, TOKENS, TOKENS, HEAD_WIDTH};
  const uint64_t head[] = {1, HEADS, TOKENS, HEAD_WIDTH};
  const uint64_t kv[] = {1, 1, TOKENS, HEAD_WIDTH};
  const uint64_t pos[] = {1, TOKENS}, mask_shape[] = {1, TOKENS, TOKENS};
  const uint64_t slope_shape[] = {HEADS};
  int64_t qpos[] = {0, 1}, kpos[] = {0, 1}, window = TOKENS;
  uint8_t mask[] = {1, 1, 1, 1};
  float slopes[] = {0.125f, 0.25f};
  float qh[ELEMENTS], dyh[ELEMENTS], yh[ELEMENTS];
  float kh[KV_ELEMENTS], vh[KV_ELEMENTS], dkh[KV_ELEMENTS], dvh[KV_ELEMENTS];
  for (size_t t = 0; t < TOKENS; ++t) {
    for (size_t d = 0; d < HEAD_WIDTH; ++d) {
      const size_t first = t * WIDTH + d, second = first + HEAD_WIDTH;
      const size_t key = t * HEAD_WIDTH + d;
      kh[key] = 0.5f * (k[first] + k[second]);
      vh[key] = 0.5f * (v[first] + v[second]);
      for (size_t h = 0; h < HEADS; ++h) {
        const size_t i = (h * TOKENS + t) * HEAD_WIDTH + d;
        qh[i] = q[first + h * HEAD_WIDTH];
        if (dy != NULL)
          dyh[i] = dy[first + h * HEAD_WIDTH];
      }
    }
  }
  const et_kernel_request_v1 request = {sizeof(request),
                                        dy ? "variant-attention.backward"
                                           : "variant-attention.forward",
                                        "f32",
                                        "cpu",
                                        ARRAY_COUNT(shape),
                                        shape,
                                        1,
                                        {0}};
  const et_kernel_tensor_view_v1 in[] = {
      test_view(qh, "f32", 4, head),
      test_view(kh, "f32", 4, kv),
      test_view(vh, "f32", 4, kv),
      test_view(qpos, "i64", 2, pos),
      test_view(kpos, "i64", 2, pos),
      test_view(mask, "bool", 3, mask_shape),
      test_view(slopes, "f32", 1, slope_shape),
      test_view(&window, "i64", 0, NULL),
      test_view(dyh, "f32", 4, head)};
  et_kernel_tensor_view_v1 out[] = {test_view(yh, "f32", 4, head),
                                    test_view(dkh, "f32", 4, kv),
                                    test_view(dvh, "f32", 4, kv)};
  const et_kernel_call_v1 call = test_call("kernel.variant-attention", &request,
                                           dy ? 9 : 8, in, dy ? 3 : 1, out);
  dispatch(runtime, &call);
  for (size_t t = 0; t < TOKENS; ++t) {
    for (size_t h = 0; h < HEADS; ++h) {
      for (size_t d = 0; d < HEAD_WIDTH; ++d) {
        const size_t i = (h * TOKENS + t) * HEAD_WIDTH + d;
        const size_t j = t * WIDTH + h * HEAD_WIDTH + d;
        y[j] = yh[i];
        if (dy != NULL) {
          dk[j] = 0.5f * dkh[t * HEAD_WIDTH + d];
          dv[j] = 0.5f * dvh[t * HEAD_WIDTH + d];
        }
      }
    }
  }
}

static float *parameter(float *weights, enum parameter which) {
  return weights + which * WIDTH;
}

/* Test-only diagonal projections keep the model small enough to
 * finite-difference. */
static void project(const float *x, const float *w, float *y) {
  for (size_t i = 0; i < ELEMENTS; ++i)
    y[i] = x[i] * w[i % WIDTH];
}

static void project_backward(const float *x, const float *w, const float *dy,
                             float *dx, float *dw) {
  for (size_t i = 0; i < ELEMENTS; ++i) {
    dx[i] += dy[i] * w[i % WIDTH];
    dw[i % WIDTH] += dy[i] * x[i];
  }
}

static float microbatch(const et_kernel_runtime *runtime, float *w, unsigned id,
                        float *grad) {
  float x[ELEMENTS], target[ELEMENTS];
  float a[ELEMENTS], q[ELEMENTS], k[ELEMENTS], v[ELEMENTS], attention[ELEMENTS];
  float residual[ELEMENTS], b[ELEMENTS], gate[ELEMENTS], up[ELEMENTS];
  float hidden[ELEMENTS], y[ELEMENTS], dy[ELEMENTS];
  for (size_t i = 0; i < ELEMENTS; ++i) {
    const int seed = (int)((id * 11u + i * 5u + i / 2u) % 23u) - 11;
    x[i] = (float)seed * 0x1p-4f;
    target[i] = 0.5f * x[i] + 0.125f * x[i % WIDTH];
  }
  rms(runtime, x, parameter(w, GAMMA1), NULL, a, NULL);
  project(a, parameter(w, QUERY), q);
  project(a, parameter(w, KEY), k);
  project(a, parameter(w, VALUE), v);
  attend(runtime, q, k, v, NULL, attention, NULL, NULL);
  project(attention, parameter(w, OUT), residual);
  for (size_t i = 0; i < ELEMENTS; ++i)
    residual[i] += x[i];
  rms(runtime, residual, parameter(w, GAMMA2), NULL, b, NULL);
  project(b, parameter(w, GATE), gate);
  project(b, parameter(w, UP), up);
  glu(runtime, gate, up, NULL, hidden, NULL);
  project(hidden, parameter(w, DOWN), y);
  float loss = 0.0f;
  for (size_t i = 0; i < ELEMENTS; ++i) {
    const float delta = y[i] + residual[i] - target[i];
    loss += delta * delta;
    dy[i] = delta / (float)ELEMENTS;
  }
  if (grad == NULL)
    return loss / (2.0f * ELEMENTS);

  memset(grad, 0, WEIGHTS * sizeof(float));
  float dhidden[ELEMENTS] = {0}, dgate[ELEMENTS], dup[ELEMENTS];
  float db[ELEMENTS] = {0}, dresidual[ELEMENTS], dattn[ELEMENTS] = {0};
  float dq[ELEMENTS], dk[ELEMENTS], dv[ELEMENTS], da[ELEMENTS] = {0};
  float dx[ELEMENTS];
  project_backward(hidden, parameter(w, DOWN), dy, dhidden,
                   parameter(grad, DOWN));
  glu(runtime, gate, up, dhidden, dgate, dup);
  project_backward(b, parameter(w, GATE), dgate, db, parameter(grad, GATE));
  project_backward(b, parameter(w, UP), dup, db, parameter(grad, UP));
  rms(runtime, residual, parameter(w, GAMMA2), db, dresidual,
      parameter(grad, GAMMA2));
  for (size_t i = 0; i < ELEMENTS; ++i)
    dresidual[i] += dy[i];
  project_backward(attention, parameter(w, OUT), dresidual, dattn,
                   parameter(grad, OUT));
  attend(runtime, q, k, v, dattn, dq, dk, dv);
  project_backward(a, parameter(w, QUERY), dq, da, parameter(grad, QUERY));
  project_backward(a, parameter(w, KEY), dk, da, parameter(grad, KEY));
  project_backward(a, parameter(w, VALUE), dv, da, parameter(grad, VALUE));
  rms(runtime, x, parameter(w, GAMMA1), da, dx, parameter(grad, GAMMA1));
  return loss / (2.0f * ELEMENTS);
}

static float batch(const et_kernel_runtime *runtime, float *w, unsigned begin,
                   unsigned count, float *grad) {
  if (grad != NULL)
    memset(grad, 0, WEIGHTS * sizeof(float));
  float loss = 0.0f;
  for (unsigned id = begin; id < begin + count; ++id) {
    float one[WEIGHTS];
    loss += microbatch(runtime, w, id, grad == NULL ? NULL : one);
    if (grad != NULL) {
      for (size_t i = 0; i < WEIGHTS; ++i)
        grad[i] += one[i];
    }
  }
  if (grad != NULL) {
    for (size_t i = 0; i < WEIGHTS; ++i)
      grad[i] /= (float)count;
  }
  return loss / (float)count;
}

static void serial_rank_order(const et_kernel_runtime *runtime, float *w,
                              unsigned world, float *grad) {
  const unsigned count = MICROS / world;
  for (unsigned r = 0; r < world; ++r) {
    float local[WEIGHTS];
    (void)batch(runtime, w, r * count, count, local);
    if (r == 0) {
      memcpy(grad, local, sizeof(local));
    } else {
      for (size_t i = 0; i < WEIGHTS; ++i)
        grad[i] += local[i];
    }
  }
  for (size_t i = 0; i < WEIGHTS; ++i)
    grad[i] /= (float)world;
}

static void check_block_gradient(const et_kernel_runtime *runtime, float *w) {
  float grad[WEIGHTS];
  (void)batch(runtime, w, 0, MICROS, grad);
  for (size_t i = 0; i < WEIGHTS; ++i) {
    const float start = w[i], increment = 0x1p-8f;
    w[i] = start + increment;
    const float plus = batch(runtime, w, 0, MICROS, NULL);
    w[i] = start - increment;
    const float minus = batch(runtime, w, 0, MICROS, NULL);
    w[i] = start;
    check_close(grad[i], ((double)plus - minus) / (2.0 * increment), 2e-6,
                0.02);
  }
}

struct training_trace {
  float loss[STEPS + 1];
  float weights[STEPS][WEIGHTS];
};

static void training(unsigned rank, unsigned world, const int *fds,
                     size_t count, unsigned variant, int report_fd) {
  et_dist5_comm_t *comm = open_rank(rank, world, fds, count);
  et_kernel_runtime *runtime = discover_mod5();
  float w[WEIGHTS], serial[WEIGHTS];
  for (size_t i = 0; i < WEIGHTS; ++i)
    w[i] = rank == 0 ? 0.5f + (float)(i % 7u) * 0x1p-4f : 0.0f;
  STATUS(et_dist5_broadcast(comm, w, WEIGHTS, ET_DIST5_F32, 0, 0, TIMEOUT),
         ET_DIST5_OK);
  memcpy(serial, w, sizeof(w));
  if (rank == 0 && world == 1 && variant == 0)
    check_block_gradient(runtime, w);
  struct training_trace trace;
  const unsigned local_count = MICROS / world;
  for (unsigned step = 0; step <= STEPS; ++step) {
    sleep_ms((long)(variant == 0 ? rank : world - rank));
    float local_grad[WEIGHTS], mean[WEIGHTS], ordered[WEIGHTS],
        reference[WEIGHTS];
    const float local_loss =
        batch(runtime, w, rank * local_count, local_count, local_grad);
    STATUS(et_dist5_allreduce(comm, &local_loss, &trace.loss[step], 1,
                              ET_DIST5_F32, ET_DIST5_MEAN, 1, TIMEOUT),
           ET_DIST5_OK);
    CHECK(isfinite(trace.loss[step]));
    if (step != 0)
      CHECK(trace.loss[step] < trace.loss[step - 1]);
    if (step == STEPS)
      break;

    if (step == 2 || step == 4) {
      float before[WEIGHTS], injected[WEIGHTS];
      memcpy(before, w, sizeof(before));
      memcpy(injected, local_grad, sizeof(injected));
      if (rank == world - 1)
        injected[3] = step == 2 ? NAN : INFINITY;
      et_dist5_status_t sync =
          et_dist5_sync_gradients_f32(comm, injected, w, WEIGHTS, 2, TIMEOUT);
      CHECK(sync == ET_DIST5_ERR_NONFINITE);
      CHECK(memcmp(w, before, sizeof(w)) == 0);
    }

    STATUS(et_dist5_sync_gradients_f32(comm, local_grad, mean, WEIGHTS, 2,
                                       TIMEOUT),
           ET_DIST5_OK);
    serial_rank_order(runtime, serial, world, ordered);
    CHECK(memcmp(mean, ordered, sizeof(mean)) == 0);
    const float flat_loss = batch(runtime, w, 0, MICROS, reference);
    check_close(trace.loss[step], flat_loss, 2e-7, 1e-7);
    for (size_t i = 0; i < WEIGHTS; ++i) {
      check_close(mean[i], reference[i], 2e-8, 2e-7);
      w[i] -= mean[i];
      serial[i] -= ordered[i];
    }
    CHECK(memcmp(w, serial, sizeof(w)) == 0);
    memcpy(trace.weights[step], w, sizeof(w));
    float all[MAX_RANKS][WEIGHTS];
    STATUS(et_dist5_allgather(comm, w, WEIGHTS, all, ET_DIST5_F32, 3, TIMEOUT),
           ET_DIST5_OK);
    for (unsigned r = 0; r < world; ++r)
      CHECK(memcmp(w, all[r], sizeof(w)) == 0);
  }
  CHECK(trace.loss[STEPS] < 0.7f * trace.loss[0]);
  if (rank == 0) {
    move_bytes(report_fd, &trace, sizeof(trace), 1);
    CHECK(close(report_fd) == 0);
  }
  et_kernel_runtime_destroy(runtime);
  et_dist5_comm_destroy(comm);
}

int main(void) {
  alarm(45);
  malformed_bootstrap();
  for (unsigned world = 1; world <= MAX_RANKS; world *= 2) {
    run_group(world, collectives, 0, NULL, 0, MAX_RANKS);
    run_group(world, collectives, 1, NULL, 0, MAX_RANKS);
  }
  puts("DIST5 collectives: PASS (6 process groups, W=1/2/4, opposite start "
       "order)");
  run_group(MAX_RANKS, malformed_calls, 0, NULL, 0, MAX_RANKS);
  for (unsigned variant = TAG_MISMATCH; variant <= ROOT_MISMATCH; ++variant)
    run_group(MAX_RANKS, mismatch, variant, NULL, 0, MAX_RANKS);
  run_group(2, mismatch, INVALID_KIND_MISMATCH, NULL, 0, MAX_RANKS);
  run_group(2, mismatch, INVALID_TAG_MISMATCH, NULL, 0, MAX_RANKS);
  run_group(MAX_RANKS, transfer_fault, CRASH_INPUT, NULL, 0, MAX_RANKS);
  run_group(MAX_RANKS, transfer_fault, CRASH_READY, NULL, 0, 2);
  run_group(MAX_RANKS, transfer_fault, CRASH_COORDINATOR, NULL, 0, 0);
  for (unsigned variant = STALL_METADATA; variant <= TRICKLE_INPUT; ++variant)
    run_group(MAX_RANKS, transfer_fault, variant, NULL, 0, MAX_RANKS);
  puts("DIST5 failures: PASS (bootstrap checks and 16 process groups)");

  struct training_trace single;
  for (unsigned world = 1; world <= MAX_RANKS; world *= 2) {
    struct training_trace first, repeat;
    run_group(world, training, 0, &first, sizeof(first), MAX_RANKS);
    run_group(world, training, 1, &repeat, sizeof(repeat), MAX_RANKS);
    CHECK(memcmp(&first, &repeat, sizeof(first)) == 0);
    if (world == 1) {
      single = first;
    } else {
      double max_delta = 0.0;
      for (size_t step = 0; step <= STEPS; ++step) {
        check_close(first.loss[step], single.loss[step], 2e-7, 1e-7);
        if (step == STEPS)
          break;
        for (size_t i = 0; i < WEIGHTS; ++i) {
          check_close(first.weights[step][i], single.weights[step][i], 2e-7, 0);
          const double delta =
              fabs((double)first.weights[step][i] - single.weights[step][i]);
          max_delta = fmax(max_delta, delta);
        }
      }
      printf("DIST5 W=%u versus W=1 maximum weight delta: %.9g\n", world,
             max_delta);
    }
    printf("DIST5 training W=%u: PASS (%u steps, loss %.9g -> %.9g, "
           "bitwise repeated and rank-ordered)\n",
           world, (unsigned)STEPS, first.loss[0], first.loss[STEPS]);
  }
  printf("DIST5 suite: PASS (%u process groups, %u block finite differences)\n",
         process_groups, (unsigned)WEIGHTS);
  return 0;
}
