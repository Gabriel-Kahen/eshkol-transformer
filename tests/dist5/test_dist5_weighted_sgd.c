#define _POSIX_C_SOURCE 200809L
#define main mod5_training_smoke_main
#include "../mod5/test_mod5_training_smoke.c"
#undef main

#include "eshkol_transformer/dist5_collectives_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

enum { WORLD = 2, TIMEOUT_MS = 2000, PARAMS = PARAMETERS * WIDTH * WIDTH };
static const size_t KV_HEADS = 2;
static const float RATE = 0.25f;
static const unsigned masks[][WORLD] = {{0x15, 0x3e}, {0x3f, 0},
                                        {0, 0x3f}, {0x20, 0x37}};

static void initial_weights(weights *model) {
  memset(model, 0, sizeof(*model));
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, KV_HEADS); ++i) {
      const int seed = (int)((p * 29u + i * 11u + i / 5u) % 37u) - 18;
      model->data[p][i] = (p >= GAMMA1 ? 1.0f : 0.0f) + (float)seed * 0x1p-7f;
    }
}

static void batch(unsigned rank, float *x, float *target) {
  for (size_t i = 0; i < ELEMENTS; ++i) {
    const float base = (float)((int)((i * 7u + i / 5u) % 31u) - 15) * 0x1p-5f;
    x[i] = base + (float)rank * (float)((int)(i % 5u) - 2) * 0x1p-6f;
    target[i] = 0.65f * base +
                (float)((int)((i * 3u) % 13u) - 6) * 0x1p-6f +
                (float)rank * (float)((int)(i % 3u) - 1) * 0x1p-5f;
  }
}

static unsigned effective_tokens(unsigned mask) {
  unsigned count = 0;
  for (size_t token = 0; token < ROWS; ++token)
    count += (mask >> token) & 1u;
  return count;
}

static double serial_objective(const et_kernel_runtime *runtime, weights *model,
                               const unsigned *mask) {
  double sum = 0.0;
  unsigned count = 0;
  for (unsigned rank = 0; rank < WORLD; ++rank) {
    float x[ELEMENTS], target[ELEMENTS];
    activations a;
    batch(rank, x, target);
    block_forward(runtime, KV_HEADS, model, x, &a);
    for (size_t token = 0; token < ROWS; ++token) {
      if (((mask[rank] >> token) & 1u) == 0)
        continue;
      ++count;
      for (size_t d = 0; d < WIDTH; ++d) {
        const size_t i = token * WIDTH + d;
        const double delta = (double)a.output[i] - target[i];
        sum += 0.5 * delta * delta;
      }
    }
  }
  CHECK(count > 0);
  return sum / (count * WIDTH);
}

/* Excluded loss rows supply exactly zero upstream derivative. The fixture's
 * denominator stays fixed at all rows until the caller forms its mean. */
static double masked_gradient(const et_kernel_runtime *runtime, weights *model,
                              unsigned rank, unsigned mask, weights *gradient) {
  float x[ELEMENTS], target[ELEMENTS], dx[ELEMENTS];
  activations a;
  batch(rank, x, target);
  block_forward(runtime, KV_HEADS, model, x, &a);
  for (size_t token = 0; token < ROWS; ++token)
    if (((mask >> token) & 1u) == 0)
      memcpy(target + token * WIDTH, a.output + token * WIDTH,
             WIDTH * sizeof(float));
  const double loss =
      block_loss(runtime, KV_HEADS, model, x, target, gradient, dx);
  CHECK(isfinite(loss));
  for (size_t i = 0; i < ELEMENTS; ++i)
    CHECK(isfinite(dx[i]));
  return loss;
}

static void check_serial_derivative(const et_kernel_runtime *runtime,
                                    weights *model, const unsigned *mask,
                                    const float *expected_mean) {
  for (size_t p = 0; p < PARAMETERS; ++p) {
    const size_t i = param_size(p, KV_HEADS) / 2;
    float *value = &model->data[p][i];
    const float original = *value, step = 0x1p-9f;
    *value = original + step;
    const double plus = serial_objective(runtime, model, mask);
    *value = original - step;
    const double minus = serial_objective(runtime, model, mask);
    *value = original;
    check_close(expected_mean[1 + p * WIDTH * WIDTH + i],
                (plus - minus) / (2.0 * step), 2e-7, 0.004);
  }
}

static void training(et_dist5_comm_t *comm, unsigned rank) {
  et_kernel_runtime *runtime = discover_mod5();
  weights model;
  initial_weights(&model);
  weights serial = model;
  float parameters[PARAMS];
  _Static_assert(sizeof(parameters) == sizeof(model), "flat parameter layout");
  memcpy(parameters, &model, sizeof(parameters));

  for (size_t step = 0; step < ARRAY_COUNT(masks); ++step) {
    weights local, raw[WORLD];
    const unsigned local_count = effective_tokens(masks[step][rank]);
    const unsigned total = effective_tokens(masks[step][0]) +
                           effective_tokens(masks[step][1]);
    const double before = serial_objective(runtime, &serial, masks[step]);
    for (unsigned r = 0; r < WORLD; ++r)
      (void)masked_gradient(runtime, &serial, r, masks[step][r], &raw[r]);
    float expected[PARAMS + 1] = {(float)before};
    float packed[PARAMS + 1] = {0};
    if (local_count != 0)
      packed[0] = (float)(masked_gradient(runtime, &model, rank,
                                          masks[step][rank], &local) *
                          ROWS / local_count);
    for (size_t p = 0; p < PARAMETERS; ++p)
      for (size_t i = 0; i < param_size(p, KV_HEADS); ++i) {
        const size_t index = 1 + p * WIDTH * WIDTH + i;
        expected[index] =
            (float)(((double)raw[0].data[p][i] + raw[1].data[p][i]) * ROWS /
                    total);
        if (local_count != 0)
          packed[index] = local.data[p][i] * ((float)ROWS / local_count);
      }
    if (step == 0)
      check_serial_derivative(runtime, &serial, masks[step], expected);

    float mean[PARAMS + 1];
    int64_t global_count = -1;
    if (rank != 0)
      CHECK(fesetround(FE_UPWARD) == 0);
    const et_dist5_status_t status = et_dist5_weighted_sgd_f32(
        comm, local_count, local_count == 0 ? NULL : packed, PARAMS, RATE,
        parameters, mean, &global_count, 10, TIMEOUT_MS);
    CHECK(fegetround() == (rank == 0 ? FE_TONEAREST : FE_UPWARD));
    CHECK(fesetround(FE_TONEAREST) == 0);
    CHECK(status == ET_DIST5_OK && global_count == total);
    for (size_t i = 0; i <= PARAMS; ++i)
      check_close(mean[i], expected[i], 1e-7, 1e-6);
    memcpy(&model, parameters, sizeof(model));
    for (size_t p = 0; p < PARAMETERS; ++p)
      for (size_t i = 0; i < WIDTH * WIDTH; ++i) {
        serial.data[p][i] -= RATE * expected[1 + p * WIDTH * WIDTH + i];
        check_close(model.data[p][i], serial.data[p][i], 1e-7, 1e-6);
      }
    CHECK(serial_objective(runtime, &model, masks[step]) < before);
    float replica[PARAMS];
    memcpy(replica, parameters, sizeof(replica));
    CHECK(et_dist5_broadcast(comm, replica, PARAMS, ET_DIST5_F32, 0, 11,
                             TIMEOUT_MS) == ET_DIST5_OK);
    CHECK(memcmp(replica, parameters, sizeof(replica)) == 0);
  }
  et_kernel_runtime_destroy(runtime);
}

enum rejection {
  NEGATIVE_COUNT, EMPTY_WORLD, COUNT_OVERFLOW, NULL_INPUT, ZERO_WIDTH,
  PARTIAL_ALIAS, NONFINITE_OBJECTIVE, NONFINITE_GRADIENT, NONFINITE_PARAMETER,
  INVALID_RATE, UPDATE_OVERFLOW, REJECTION_CASES
};

static void rejection_cases(et_dist5_comm_t *comm, unsigned rank) {
  for (enum rejection test = NEGATIVE_COUNT; test < REJECTION_CASES; ++test) {
    int64_t local_count = 1, total = -7;
    float parameter = 0.0f, mean[] = {17.0f, -19.0f, 23.0f};
    float packed[] = {3.0f, 0.5f};
    const float *input = packed;
    size_t count = 1;
    float rate = RATE;
    et_dist5_status_t expected = ET_DIST5_ERR_INVALID_ARG;
    if (test == EMPTY_WORLD)
      local_count = 0;
    if (test >= NONFINITE_OBJECTIVE && test != INVALID_RATE)
      expected = ET_DIST5_ERR_NONFINITE;
    if (test == UPDATE_OVERFLOW) {
      /* Only rank 1 overflows after the common gradient has been reduced. */
      local_count = rank == 0 ? 1 : 0;
      packed[1] = FLT_MAX;
      rate = 1.0f;
    }
    if (rank == 1) {
      switch (test) {
      case NEGATIVE_COUNT: local_count = -1; break;
      case COUNT_OVERFLOW: local_count = INT64_MAX; break;
      case NULL_INPUT: input = NULL; break;
      case ZERO_WIDTH: count = 0; break;
      case PARTIAL_ALIAS: input = mean + 1; break;
      case NONFINITE_OBJECTIVE: packed[0] = NAN; break;
      case NONFINITE_GRADIENT: packed[1] = INFINITY; break;
      case NONFINITE_PARAMETER: parameter = -INFINITY; break;
      case INVALID_RATE: rate = -0.0f; break;
      case UPDATE_OVERFLOW: parameter = -FLT_MAX; break;
      default: break;
      }
    }
    const float original = parameter;
    CHECK(et_dist5_weighted_sgd_f32(comm, local_count, input, count, rate,
                                    &parameter, mean, &total, 12, TIMEOUT_MS) ==
          expected);
    CHECK(memcmp(&parameter, &original, sizeof(parameter)) == 0);
    CHECK(mean[0] == 17.0f && mean[1] == -19.0f && mean[2] == 23.0f && total == -7);

    /* Recover through another weighted step using an exact, very large count,
     * a gradient that would overflow if multiplied by count, and in-place IO. */
    float next = 0.0f, in_place[] = {2.0f, FLT_MAX};
    const int64_t valid_count = rank == 0 ? INT64_MAX - 3 : 3;
    CHECK(et_dist5_weighted_sgd_f32(
              comm, valid_count, in_place, 1, 0x1p-126f, &next, in_place,
              &total, 12, TIMEOUT_MS) == ET_DIST5_OK);
    CHECK(total == INT64_MAX && in_place[0] == 2.0f && in_place[1] == FLT_MAX);
    CHECK(next == -0x1.fffffep1f);
  }
}

static void mismatch(et_dist5_comm_t *comm, unsigned rank, int wrong_kind) {
  float parameter = 2.0f, input[] = {1.0f, 3.0f}, mean[] = {-1.0f, -1.0f};
  int64_t total = -1;
  const et_dist5_status_t status = wrong_kind && rank == 1
      ? et_dist5_barrier(comm, 33, TIMEOUT_MS)
      : et_dist5_weighted_sgd_f32(comm, 1, input, 1, rank == 0 ? 0.5f : 1.0f,
                                  &parameter, mean, &total, 33, TIMEOUT_MS);
  CHECK(status == ET_DIST5_ERR_MISMATCH);
  CHECK(parameter == 2.0f && mean[0] == -1.0f && mean[1] == -1.0f && total == -1);
  CHECK(et_dist5_barrier(comm, 34, TIMEOUT_MS) == ET_DIST5_ERR_MISMATCH);
}

enum group { TRAINING, MISMATCH_RATE, MISMATCH_KIND };

static void rank_run(enum group group, unsigned rank, int fd) {
  alarm(10);
  et_dist5_comm_t *comm = NULL;
  const uint32_t minor =
      group == MISMATCH_KIND && rank == 1 ? 0 : ET_DIST5_ABI_MINOR;
  CHECK(et_dist5_comm_create(ET_DIST5_ABI_MAJOR, minor, rank, WORLD, &fd, 1,
                             &comm) == ET_DIST5_OK);
  CHECK(close(fd) == 0);
  if (group == TRAINING) {
    training(comm, rank);
    rejection_cases(comm, rank);
  } else {
    mismatch(comm, rank, group == MISMATCH_KIND);
  }
  et_dist5_comm_destroy(comm);
}

static void pair(enum group group) {
  int fds[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
  pid_t child[WORLD] = {0};
  for (unsigned rank = 0; rank < WORLD; ++rank) {
    child[rank] = fork();
    if (child[rank] < 0) {
      for (unsigned prior = 0; prior < rank; ++prior) {
        (void)kill(child[prior], SIGKILL);
        (void)waitpid(child[prior], NULL, 0);
      }
      CHECK(child[rank] >= 0);
    }
    if (child[rank] == 0) {
      CHECK(close(fds[1u - rank]) == 0);
      rank_run(group, rank, fds[rank]);
      exit(EXIT_SUCCESS);
    }
  }
  CHECK(close(fds[0]) == 0);
  CHECK(close(fds[1]) == 0);
  int passed = 1;
  for (unsigned rank = 0; rank < WORLD; ++rank) {
    int status;
    pid_t waited;
    do {
      waited = waitpid(child[rank], &status, 0);
    } while (waited < 0 && errno == EINTR);
    CHECK(waited == child[rank]);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      fprintf(stderr, "weighted step group %d rank %u failed (status %#x)\n",
              group, rank, status);
      passed = 0;
    }
  }
  CHECK(passed);
}

int main(void) {
  pair(TRAINING);
  pair(MISMATCH_RATE);
  pair(MISMATCH_KIND);
  puts("DIST5 weighted SGD: PASS (4 MOD5 updates, 9 finite differences, "
       "11 rejection/recovery cases and 2 protocol mismatches)");
  return 0;
}
