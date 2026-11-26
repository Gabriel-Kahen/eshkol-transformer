#define _POSIX_C_SOURCE 200809L
#define main mod5_training_smoke_main
#include "../mod5/test_mod5_training_smoke.c"
#undef main

#include "eshkol_transformer/dist5_collectives_abi.h"

#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

enum { WORLD = 2, SYNC_TIMEOUT_MS = 2000, TRAIN_STEPS = 3 };
static const size_t KV_HEADS = 2;
static const float LEARNING_RATE = 0.25f;

static void initial_weights(weights *w) {
  memset(w, 0, sizeof(*w));
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, KV_HEADS); ++i) {
      const int seed = (int)((p * 29u + i * 11u + i / 5u) % 37u) - 18;
      w->data[p][i] = (p >= GAMMA1 ? 1.0f : 0.0f) + (float)seed * 0x1p-7f;
    }
}

static void local_batch(unsigned rank, float *x, float *target) {
  for (size_t i = 0; i < ELEMENTS; ++i) {
    const float base = (float)((int)((i * 7u + i / 5u) % 31u) - 15) * 0x1p-5f;
    x[i] = base + (float)rank * (float)((int)(i % 5u) - 2) * 0x1p-6f;
    target[i] = 0.65f * base +
                (float)((int)((i * 3u) % 13u) - 6) * 0x1p-6f +
                (float)rank * (float)((int)(i % 3u) - 1) * 0x1p-5f;
  }
}

static void serial_mean(const et_kernel_runtime *runtime, const weights *w,
                        weights *mean) {
  weights pieces[WORLD];
  float x[ELEMENTS], target[ELEMENTS], dx[ELEMENTS];
  for (unsigned rank = 0; rank < WORLD; ++rank) {
    local_batch(rank, x, target);
    weights copy = *w;
    CHECK(isfinite(block_loss(runtime, KV_HEADS, &copy, x, target,
                              &pieces[rank], dx)));
  }
  memset(mean, 0, sizeof(*mean));
  double difference = 0.0;
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, KV_HEADS); ++i) {
      const float a = pieces[0].data[p][i];
      const float b = pieces[1].data[p][i];
      CHECK(isfinite(a) && isfinite(b));
      mean->data[p][i] = (a + b) * 0.5f;
      difference += fabs((double)a - b);
    }
  CHECK(difference > 0.001);
}

static void check_weights(const weights *actual, const weights *expected) {
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, KV_HEADS); ++i)
      check_close(actual->data[p][i], expected->data[p][i], 1e-7, 1e-6);
}

static void update(weights *w, const weights *gradient) {
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, KV_HEADS); ++i)
      w->data[p][i] -= LEARNING_RATE * gradient->data[p][i];
}

static void rank_run(unsigned rank, int fd) {
  alarm(8);
  et_dist5_comm_t *comm = NULL;
  CHECK(et_dist5_comm_create(ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR, rank,
                             WORLD, &fd, 1, &comm) == ET_DIST5_OK);
  CHECK(close(fd) == 0);
  et_kernel_runtime *runtime = discover_mod5();
  weights model;
  initial_weights(&model);
  float x[ELEMENTS], target[ELEMENTS], dx[ELEMENTS];
  local_batch(rank, x, target);

  for (unsigned step = 0; step < TRAIN_STEPS + 1u; ++step) {
    weights reference_gradient, local_gradient, synchronized, expected = model;
    serial_mean(runtime, &model, &reference_gradient);
    CHECK(isfinite(block_loss(runtime, KV_HEADS, &model, x, target,
                              &local_gradient, dx)));
    for (size_t i = 0; i < ELEMENTS; ++i)
      CHECK(isfinite(dx[i]));
    for (size_t p = 0; p < PARAMETERS; ++p)
      for (size_t i = 0; i < WIDTH * WIDTH; ++i)
        synchronized.data[p][i] = -17.0f;

    if (step == 2u && rank == 1u)
      local_gradient.data[WQ][0] = NAN;
    const et_dist5_status_t status = et_dist5_sync_gradients_f32(
        comm, &local_gradient.data[0][0], &synchronized.data[0][0],
        PARAMETERS * WIDTH * WIDTH, step, SYNC_TIMEOUT_MS);
    if (step == 2u) {
      CHECK(status == ET_DIST5_ERR_NONFINITE);
      CHECK(memcmp(&model, &expected, sizeof(model)) == 0);
      for (size_t p = 0; p < PARAMETERS; ++p)
        for (size_t i = 0; i < WIDTH * WIDTH; ++i)
          CHECK(synchronized.data[p][i] == -17.0f);
      CHECK(et_dist5_barrier(comm, 99, SYNC_TIMEOUT_MS) == ET_DIST5_OK);
      continue;
    }
    CHECK(status == ET_DIST5_OK);
    for (size_t p = 0; p < PARAMETERS; ++p)
      for (size_t i = 0; i < param_size(p, KV_HEADS); ++i)
        check_close(synchronized.data[p][i],
                    reference_gradient.data[p][i], 1e-7, 1e-6);
    update(&expected, &reference_gradient);
    update(&model, &synchronized);
    check_weights(&model, &expected);
  }
  CHECK(et_dist5_barrier(comm, 100, SYNC_TIMEOUT_MS) == ET_DIST5_OK);
  et_kernel_runtime_destroy(runtime);
  et_dist5_comm_destroy(comm);
}

int main(void) {
  int fds[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
  pid_t children[WORLD] = {0};
  for (unsigned rank = 0; rank < WORLD; ++rank) {
    children[rank] = fork();
    if (children[rank] < 0) {
      for (unsigned previous = 0; previous < rank; ++previous) {
        (void)kill(children[previous], SIGKILL);
        (void)waitpid(children[previous], NULL, 0);
      }
      CHECK(children[rank] >= 0);
    }
    if (children[rank] != 0)
      continue;
    CHECK(close(fds[1u - rank]) == 0);
    rank_run(rank, fds[rank]);
    exit(EXIT_SUCCESS);
  }
  CHECK(close(fds[0]) == 0);
  CHECK(close(fds[1]) == 0);
  int passed = 1;
  int active[WORLD] = {1, 1};
  for (unsigned done = 0; done < WORLD; ++done) {
    int status;
    pid_t child;
    do {
      child = waitpid(-1, &status, 0);
    } while (child < 0 && errno == EINTR);
    CHECK(child > 0);
    for (unsigned rank = 0; rank < WORLD; ++rank)
      if (children[rank] == child)
        active[rank] = 0;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      fprintf(stderr, "rank process %ld failed (status %#x)\n", (long)child,
              status);
      passed = 0;
      for (unsigned rank = 0; rank < WORLD; ++rank)
        if (active[rank])
          (void)kill(children[rank], SIGKILL);
    }
  }
  CHECK(passed);
  puts("DIST5/MOD5 actual-gradient SGD parity: PASS (2 ranks, 3 updates, "
       "coordinated rejection and recovery)");
  return 0;
}
