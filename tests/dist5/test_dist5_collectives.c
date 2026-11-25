#define _POSIX_C_SOURCE 200809L
#include "eshkol_transformer/dist5_collectives_abi.h"

#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON

#define CHECK(condition) assert(condition)
#define ARRAY_COUNT(array) (sizeof(array) / sizeof((array)[0]))

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
  const float gradients[] = {(float)rank + 1.0f, (float)rank * 2.0f};
  float mean[] = {-7.0f, -7.0f};
  const float expected_mean[] = {(float)(world + 1) / 2.0f,
                                 (float)(world - 1)};
  STATUS(et_dist5_sync_gradients_f32(comm, gradients, mean, 2, 12, TIMEOUT),
         ET_DIST5_OK);
  CHECK(memcmp(mean, expected_mean, sizeof(mean)) == 0);
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
  const float gradient = bad ? NAN : (float)rank;
  float mean = -7.0f;
  STATUS(et_dist5_sync_gradients_f32(comm, &gradient, &mean, 1, 0, TIMEOUT),
         ET_DIST5_ERR_NONFINITE);
  CHECK(mean == -7.0f);
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

  printf("DIST5 suite: PASS (%u process groups)\n", process_groups);
  return 0;
}
