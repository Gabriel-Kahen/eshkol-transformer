#define _GNU_SOURCE

#include "eshkol_transformer/wave4_composed_trainer_abi.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define STATUS(name) ET_WAVE4_COMPOSED_##name
#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define CHECK(expression)                                                      \
  do {                                                                         \
    if (!(expression)) {                                                       \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expression);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#define CHECK_STATUS(expression, expected)                                     \
  do {                                                                         \
    const et_wave4_composed_status actual = (expression);                      \
    if (actual != (expected))                                                  \
      fprintf(stderr, "actual=%u expected=%u\n", actual,                       \
              (unsigned)(expected));                                           \
    CHECK(actual == (expected));                                               \
  } while (0)

enum {
  MAX_RANKS = 4,
  X = 8,
  PER_LAYER = 104,
  MAX_P = 8 * PER_LAYER,
  STEPS = 7,
  FAULT_STEP = 2,
  TIMEOUT_MS = 10000
};

typedef struct {
  float master[MAX_P], momentum[MAX_P];
  et_mp4_loss_scaler_v1 scaler;
} snapshot;

typedef struct {
  snapshot states[STEPS];
  double losses[STEPS];
  size_t activation_peak, workspace_bytes;
  et_eval5_manifest manifest;
} report;

typedef struct {
  et_wave4_composed_config_v1 config;
  et_wave4_composed_memory_v1 plan;
  et_wave4_composed_state_v1 state;
  et_wave4_composed_batch_v1 batch;
  et_wave4_composed_workspace_v1 workspace;
  float master[MAX_P], momentum[MAX_P], input[X], target[X];
} replica;

static const et_wave4_composed_scaler_config_v1 scaler = {.scale = 64.0f,
                                                          .growth_factor = 2.0f,
                                                          .backoff_factor =
                                                              0.5f,
                                                          .min_scale = 32.0f,
                                                          .max_scale = 128.0f,
                                                          .growth_interval = 5};

static float sample(uint32_t *seed) {
  *seed = *seed * UINT32_C(1664525) + UINT32_C(1013904223);
  return (float)(*seed >> 16) * 0x1p-15f - 1.0f;
}

static void initialize(replica *local, uint32_t layers,
                       et_wave4_composed_dtype dtype, et_ac4_policy policy,
                       uint32_t rank) {
  *local =
      (replica){.config = {.ac4 = {.shape = {layers, 1, 2, 2, 1, 2},
                                   .mode = ET_AC4_FORWARD_BACKWARD,
                                   .policy = policy,
                                   .checkpoint_every =
                                       policy == ET_AC4_POLICY_EVERY_K ? 2 : 0,
                                   .epsilon = 0.125f,
                                   .dropout_rate = 0.03125f,
                                   .seed = 412775 + rank},
                           .dtype = dtype,
                           .learning_rate = 0.22f,
                           .momentum = 0.5f}};
  CHECK_STATUS(et_wave4_composed_plan_v1(&local->config, &local->plan),
               STATUS(OK));
  const size_t p = local->plan.ac4.parameter_floats;
  CHECK(p == layers * PER_LAYER && local->plan.ac4.activation_floats == X);
  CHECK(local->plan.narrow_parameter_bytes * 2 == p * sizeof(float));
  local->state = (et_wave4_composed_state_v1){.parameter_count = p,
                                              .master = local->master,
                                              .momentum = local->momentum};
  local->batch =
      (et_wave4_composed_batch_v1){{local->input, X}, {local->target, X}};
  local->workspace = (et_wave4_composed_workspace_v1){
      malloc(local->plan.workspace_bytes), local->plan.workspace_bytes};
  CHECK(local->workspace.data != NULL);

  uint32_t rng = 241;
  for (size_t i = 0; i < p; ++i) {
    local->master[i] = rank == 0 ? 0.20f * sample(&rng) : -7.0f;
    local->momentum[i] = 1.0f;
  }
  if (rank == 0)
    for (uint32_t layer = 0; layer < layers; ++layer)
      for (size_t i = 0; i < 4; ++i) {
        local->master[layer * PER_LAYER + i] = 1.0f;
        local->master[layer * PER_LAYER + 52 + i] = 1.0f;
      }

  rng = 61;
  for (size_t i = 0; i < X; ++i) {
    const float common = sample(&rng);
    const float offset = sample(&rng);
    local->input[i] = 0.75f * common + (float)rank * 0.025f * offset;
    local->target[i] = 0.70f * local->input[i] - 0.20f + 0.15f * sample(&rng);
  }
}

static void save(const replica *local, snapshot *out) {
  const size_t bytes = local->state.parameter_count * sizeof(float);
  memcpy(out->master, local->master, bytes);
  memcpy(out->momentum, local->momentum, bytes);
  out->scaler = local->state.scaler;
}

static void check_weights(const replica *local, const snapshot *prior) {
  const size_t bytes = local->state.parameter_count * sizeof(float);
  CHECK(memcmp(local->master, prior->master, bytes) == 0);
  CHECK(memcmp(local->momentum, prior->momentum, bytes) == 0);
}

static void check_same_state(const snapshot *a, const snapshot *b, size_t p) {
  CHECK(memcmp(a->master, b->master, p * sizeof(float)) == 0);
  CHECK(memcmp(a->momentum, b->momentum, p * sizeof(float)) == 0);
  CHECK(memcmp(&a->scaler, &b->scaler, sizeof(a->scaler)) == 0);
}

/* Reference BF16 storage rounding gives AC4 the same widened operands while
 * choosing the input fault. A clean scaled gradient must reach the finite
 * F32 interval that rounds to BF16 infinity, even for the BF16 lane. */
static float reference_bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  bits = (bits + UINT32_C(0x7fff) + ((bits >> 16) & 1)) & UINT32_C(0xffff0000);
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static float largest_reference_gradient(const replica *local,
                                        const float *target) {
  float weights[MAX_P], gradients[MAX_P], output[X], d_input[X];
  const size_t p = local->state.parameter_count;
  for (size_t i = 0; i < p; ++i)
    weights[i] = reference_bf16(local->master[i]);

  float *scratch = malloc(local->plan.ac4.workspace_bytes);
  CHECK(scratch != NULL);
  const et_ac4_io_v1 io = {local->batch.input,
                           {target, X},
                           {weights, p},
                           {output, X},
                           {d_input, X},
                           {gradients, p},
                           scratch,
                           local->plan.ac4.workspace_bytes};
  et_ac4_result_v1 result;
  CHECK(et_ac4_run_v1(&local->config.ac4, &io, &result) == ET_AC4_OK);
  free(scratch);
  float largest = 0.0f;
  for (size_t i = 0; i < p; ++i)
    largest = fmaxf(largest, fabsf(gradients[i]));
  return largest;
}

static void make_overflow_target(const replica *local, float *target) {
  for (size_t i = 0; i < X; ++i)
    target[i] = 0x1p20f;
  if (local->config.dtype == ET_WAVE4_COMPOSED_F16)
    return;

  const float probe = largest_reference_gradient(local, target);
  CHECK(probe > 0.0f);
  const double desired = 0x1.ff8p127 / (double)local->state.scaler.scale;
  const double amplitude = desired / (double)probe * (double)target[0];
  CHECK(amplitude < FLT_MAX);
  for (size_t i = 0; i < X; ++i)
    target[i] = (float)amplitude;
  const float scaled =
      largest_reference_gradient(local, target) * local->state.scaler.scale;
  CHECK(isfinite(scaled) && scaled >= 0x1.ffp127f);
}

static void digest(const void *data, size_t bytes,
                   uint8_t out[ET_EVAL5_SHA256_BYTES]) {
  int input[2], output[2];
  CHECK(pipe(input) == 0 && pipe(output) == 0);
  const pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    if (dup2(input[0], STDIN_FILENO) < 0 ||
        dup2(output[1], STDOUT_FILENO) < 0 || close(input[0]) != 0 ||
        close(input[1]) != 0 || close(output[0]) != 0 || close(output[1]) != 0)
      _exit(126);
    execlp("sha256sum", "sha256sum", (char *)NULL);
    _exit(127);
  }
  CHECK(close(input[0]) == 0 && close(output[1]) == 0);
  CHECK(write(input[1], data, bytes) == (ssize_t)bytes && close(input[1]) == 0);
  char hex[80];
  const ssize_t n = read(output[0], hex, sizeof(hex) - 1);
  CHECK(n >= 64 && close(output[0]) == 0);
  hex[n] = '\0';
  int status;
  CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
        WEXITSTATUS(status) == 0);
  for (size_t i = 0; i < ET_EVAL5_SHA256_BYTES; ++i) {
    unsigned value;
    CHECK(sscanf(hex + 2 * i, "%2x", &value) == 1);
    out[i] = (uint8_t)value;
  }
}

static void evaluate(replica *local, const char *path, et_eval5_manifest *out) {
  static const uint8_t train[] = "red fox";
  static const uint8_t held_out[] = "sky blue";
  static const uint8_t vocabulary[] =
      "four fixture classes: red green blue black";
  static const uint32_t ids[] = {0, 2};
  const float input[X] = {0.35f,  -0.55f, 0.41f,  0.28f,
                          -0.21f, 0.49f,  -0.22f, 0.13f};
  const float target[X] = {0};
  et_eval5_manifest identity = {
      .version_major = ET_EVAL5_VERSION_MAJOR,
      .version_minor = ET_EVAL5_VERSION_MINOR,
      .suite = "wave4-composed",
      .task = "fixture-logits",
      .split = "held-out",
      .max_samples = 1,
      .contamination = {ET_EVAL5_NGRAM_BYTES, 2, 1, 0.5, 0.0}};
  digest(train, sizeof(train) - 1, identity.train_dataset_sha256);
  digest(held_out, sizeof(held_out) - 1, identity.eval_dataset_sha256);
  digest(vocabulary, sizeof(vocabulary) - 1, identity.tokenizer_sha256);
  digest(local->master, local->state.parameter_count * sizeof(float),
         identity.checkpoint_sha256);
  const et_wave4_composed_eval_v1 eval = {
      {{input, X}, {target, X}},
      {train, sizeof(train) - 1, NULL, 0},
      {held_out, sizeof(held_out) - 1, NULL, 0},
      ids,
      COUNT(ids),
      path,
      &identity};

  CHECK_STATUS(et_wave4_composed_evaluate_v1(&local->config, &local->state,
                                             &eval, local->workspace, out),
               STATUS(OK));
  et_eval5_manifest reloaded;
  CHECK(et_eval5_manifest_read_file(path, &reloaded) == ET_EVAL5_OK);
  CHECK(reloaded.lm.token_count == 2 &&
        reloaded.lm.byte_count == sizeof(held_out) - 1);
  CHECK(reloaded.lm.perplexity > 1.0 &&
        reloaded.audit.contaminated_documents == 0);
  CHECK(memcmp(reloaded.checkpoint_sha256, identity.checkpoint_sha256,
               ET_EVAL5_SHA256_BYTES) == 0);
  CHECK(memcmp(reloaded.manifest_sha256, out->manifest_sha256,
               ET_EVAL5_SHA256_BYTES) == 0);

  FILE *file = fopen(path, "r+");
  CHECK(file != NULL);
  char bytes[ET_EVAL5_MANIFEST_MAX_BYTES + 1];
  const size_t n = fread(bytes, 1, sizeof(bytes) - 1, file);
  CHECK(n > 0 && !ferror(file));
  bytes[n] = 0;
  char *changed = strstr(bytes, "split=held-out");
  CHECK(changed != NULL);
  CHECK(fseek(file, changed - bytes + (long)strlen("split="), SEEK_SET) == 0);
  CHECK(fputc('w', file) == 'w' && fclose(file) == 0);
  CHECK(et_eval5_manifest_read_file(path, &reloaded) == ET_EVAL5_ERR_CHECKSUM);
}

static void train(uint32_t rank, uint32_t world, uint32_t layers,
                  et_wave4_composed_dtype dtype, et_ac4_policy policy,
                  et_dist5_comm_t *comm, report *observed, const char *path) {
  replica local;
  initialize(&local, layers, dtype, policy, rank);
  CHECK_STATUS(et_wave4_composed_bootstrap_v1(&local.config, comm, TIMEOUT_MS,
                                              &scaler, &local.state,
                                              local.workspace),
               STATUS(OK));
  CHECK(local.master[0] == 1.0f && local.momentum[0] == 0.0f &&
        local.state.scaler.scale == 64.0f);

  if (world > 1) {
    snapshot before = {0};
    save(&local, &before);
    et_wave4_composed_workspace_v1 scratch = local.workspace;
    if (rank == world - 1)
      --scratch.bytes;
    et_wave4_composed_result_v1 denied = {.loss = -1.0};
    CHECK_STATUS(et_wave4_composed_step_v1(&local.config, comm, TIMEOUT_MS,
                                           &local.state, &local.batch, scratch,
                                           &denied),
                 STATUS(WORKSPACE_TOO_SMALL));
    CHECK(denied.loss == -1.0);
    check_weights(&local, &before);
    CHECK(memcmp(&local.state.scaler, &before.scaler, sizeof(before.scaler)) ==
          0);
  }

  const size_t p = local.state.parameter_count;
  for (size_t step = 0; step < STEPS; ++step) {
    local.config.ac4.step = step;
    snapshot before = {0};
    save(&local, &before);
    const int inject = step == FAULT_STEP && rank == world - 1;
    float fault[X];
    et_wave4_composed_batch_v1 batch = local.batch;
    if (inject) {
      make_overflow_target(&local, fault);
      batch.target.data = fault;
    }
    et_wave4_composed_result_v1 result;
    CHECK_STATUS(et_wave4_composed_step_v1(&local.config, comm, TIMEOUT_MS,
                                           &local.state, &batch,
                                           local.workspace, &result),
                 step == FAULT_STEP ? STATUS(OVERFLOW_BACKOFF) : STATUS(OK));

    CHECK(result.local_overflow == (uint32_t)inject);
    CHECK(result.memory.ac4.activation_peak_bytes ==
          local.plan.ac4.activation_peak_bytes);
    CHECK(result.memory.workspace_bytes == local.workspace.bytes);
    CHECK(result.memory.narrow_parameter_bytes == p * sizeof(uint16_t));
    if (step == FAULT_STEP) {
      check_weights(&local, &before);
      CHECK(before.scaler.scale == 64.0f);
      CHECK(local.state.scaler.scale == 32.0f &&
            local.state.scaler.clean_steps == 0);
      save(&local, &before);
      if (inject)
        make_overflow_target(&local, fault);
      et_wave4_composed_result_v1 rejected;
      memset(&rejected, 0x53, sizeof(rejected));
      unsigned char unchanged[sizeof(rejected)];
      memcpy(unchanged, &rejected, sizeof(unchanged));
      CHECK_STATUS(et_wave4_composed_step_v1(&local.config, comm, TIMEOUT_MS,
                                             &local.state, &batch,
                                             local.workspace, &rejected),
                   STATUS(UNDERFLOW_FAULT));
      check_weights(&local, &before);
      CHECK(memcmp(&local.state.scaler, &before.scaler,
                   sizeof(before.scaler)) == 0);
      CHECK(memcmp(&rejected, unchanged, sizeof(rejected)) == 0);
    } else {
      CHECK(memcmp(local.master, before.master, p * sizeof(float)) != 0);
      CHECK(local.state.scaler.scale == before.scaler.scale);
      CHECK(local.state.scaler.clean_steps == before.scaler.clean_steps + 1);
    }
    save(&local, &observed->states[step]);
    observed->losses[step] = result.loss;
    observed->activation_peak = result.memory.ac4.activation_peak_bytes;
    observed->workspace_bytes = local.workspace.bytes;
  }

  if (rank == 0)
    evaluate(&local, path, &observed->manifest);

  /* After the other ranks close their sockets, rank zero must surface the
   * transport failure and preserve the state from the last accepted step. */
  if (world > 1 && rank == 0) {
    et_wave4_composed_result_v1 untouched = {.loss = -1.0};
    CHECK_STATUS(et_wave4_composed_step_v1(&local.config, comm, TIMEOUT_MS,
                                           &local.state, &local.batch,
                                           local.workspace, &untouched),
                 STATUS(DIST5_ERROR_BASE) + ET_DIST5_ERR_PEER_DISCONNECTED);
    check_weights(&local, &observed->states[STEPS - 1]);
    CHECK(memcmp(&local.state.scaler, &observed->states[STEPS - 1].scaler,
                 sizeof(local.state.scaler)) == 0);
    CHECK(untouched.loss == -1.0);
  }
  free(local.workspace.data);
}

static void fork_world(uint32_t world, uint32_t layers,
                       et_wave4_composed_dtype dtype, et_ac4_policy policy,
                       report *shared, const char *path) {
  int pairs[MAX_RANKS - 1][2];
  pid_t children[MAX_RANKS];
  for (uint32_t i = 0; i + 1 < world; ++i)
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, pairs[i]) == 0);

  for (uint32_t rank = 0; rank < world; ++rank) {
    children[rank] = fork();
    CHECK(children[rank] >= 0);
    if (children[rank] != 0)
      continue;

    int fds[MAX_RANKS - 1];
    size_t count = 0;
    for (uint32_t peer = 1; peer < world; ++peer) {
      if (rank == 0)
        fds[count++] = pairs[peer - 1][0];
      else if (rank == peer)
        fds[count++] = pairs[peer - 1][1];
    }
    et_dist5_comm_t *comm = NULL;
    CHECK(et_dist5_comm_create(ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR, rank,
                               world, fds, count, &comm) == ET_DIST5_OK);
    for (uint32_t i = 0; i + 1 < world; ++i) {
      CHECK(close(pairs[i][0]) == 0);
      CHECK(close(pairs[i][1]) == 0);
    }
    train(rank, world, layers, dtype, policy, comm, &shared[rank], path);
    et_dist5_comm_destroy(comm);
    exit(EXIT_SUCCESS); /* Run LeakSanitizer in every training process. */
  }
  for (uint32_t i = 0; i + 1 < world; ++i) {
    CHECK(close(pairs[i][0]) == 0);
    CHECK(close(pairs[i][1]) == 0);
  }

  for (uint32_t rank = 0; rank < world; ++rank) {
    int status;
    CHECK(waitpid(children[rank], &status, 0) == children[rank]);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
  CHECK(unlink(path) == 0);
}

static void verify_matrix(const char *path) {
  const size_t bytes = 2 * MAX_RANKS * sizeof(report);
  report *none = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  CHECK(none != MAP_FAILED);
  report *sqrt = none + MAX_RANKS;
  for (uint32_t layers = 4; layers <= 8; layers *= 2) {
    for (et_wave4_composed_dtype dtype = ET_WAVE4_COMPOSED_F16;
         dtype <= ET_WAVE4_COMPOSED_BF16; ++dtype) {
      for (uint32_t world = 1; world <= MAX_RANKS; world *= 2) {
        fork_world(world, layers, dtype, ET_AC4_POLICY_NONE, none, path);
        fork_world(world, layers, dtype, ET_AC4_POLICY_SQRT_SEGMENTS, sqrt,
                   path);
        const size_t p = layers * PER_LAYER;
        double mean[STEPS] = {0};
        for (uint32_t rank = 0; rank < world; ++rank) {
          for (size_t s = 0; s < STEPS; ++s) {
            check_same_state(&none[rank].states[s], &none[0].states[s], p);
            check_same_state(&sqrt[rank].states[s], &none[rank].states[s], p);
            CHECK(memcmp(&none[rank].losses[s], &sqrt[rank].losses[s],
                         sizeof(double)) == 0);
            mean[s] += none[rank].losses[s] / (double)world;
          }
          CHECK(sqrt[rank].activation_peak < none[rank].activation_peak);
          CHECK(sqrt[rank].workspace_bytes < none[rank].workspace_bytes);
        }
        CHECK(memcmp(none[0].manifest.manifest_sha256,
                     sqrt[0].manifest.manifest_sha256,
                     ET_EVAL5_SHA256_BYTES) == 0);
        printf("L=%u W=%u %s activation=%zu/%zu workspace=%zu/%zu losses",
               layers, world, dtype == ET_WAVE4_COMPOSED_F16 ? "f16" : "bf16",
               none[0].activation_peak, sqrt[0].activation_peak,
               none[0].workspace_bytes, sqrt[0].workspace_bytes);
        for (size_t s = 0; s < STEPS; ++s)
          printf(" %.17g", mean[s]);
        printf(" eval_nll=%.17g\n", none[0].manifest.lm.mean_nll);
        fflush(stdout);
        double previous = INFINITY;
        for (size_t s = 0; s < STEPS; ++s) {
          if (s == FAULT_STEP)
            continue;
          CHECK(mean[s] < previous);
          previous = mean[s];
        }
      }
    }
  }
  CHECK(munmap(none, bytes) == 0);
}

static void expect_error(replica *local, et_dist5_comm_t *comm,
                         const et_wave4_composed_batch_v1 *batch,
                         et_wave4_composed_workspace_v1 workspace,
                         et_wave4_composed_status expected) {
  snapshot before = {0};
  save(local, &before);
  et_wave4_composed_result_v1 result;
  memset(&result, 0x71, sizeof(result));
  unsigned char unchanged[sizeof(result)];
  memcpy(unchanged, &result, sizeof(result));
  CHECK_STATUS(et_wave4_composed_step_v1(&local->config, comm, TIMEOUT_MS,
                                         &local->state, batch, workspace,
                                         &result),
               expected);
  check_weights(local, &before);
  CHECK(memcmp(&local->state.scaler, &before.scaler, sizeof(before.scaler)) ==
        0);
  CHECK(memcmp(&result, unchanged, sizeof(result)) == 0);
}

static void verify_errors_and_optional_policies(void) {
  replica local;
  initialize(&local, 4, ET_WAVE4_COMPOSED_F16, ET_AC4_POLICY_NONE, 0);
  et_dist5_comm_t *comm = NULL;
  CHECK(et_dist5_comm_create(ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR, 0, 1, NULL,
                             0, &comm) == ET_DIST5_OK);

  et_wave4_composed_scaler_config_v1 invalid = scaler;
  invalid.min_scale = 0;
  snapshot pre_bootstrap = {0};
  save(&local, &pre_bootstrap);
  CHECK_STATUS(et_wave4_composed_bootstrap_v1(&local.config, comm, TIMEOUT_MS,
                                              &invalid, &local.state,
                                              local.workspace),
               STATUS(INVALID_ARGUMENT));
  check_weights(&local, &pre_bootstrap);
  CHECK(memcmp(&local.state.scaler, &pre_bootstrap.scaler,
               sizeof(pre_bootstrap.scaler)) == 0);
  CHECK_STATUS(et_wave4_composed_bootstrap_v1(&local.config, comm, TIMEOUT_MS,
                                              &scaler, &local.state,
                                              local.workspace),
               STATUS(OK));

  local.config.ac4.shape.layers = 5;
  expect_error(&local, comm, &local.batch, local.workspace,
               STATUS(INVALID_ARGUMENT));
  local.config.ac4.shape.layers = 4;
  local.config.ac4.activation_limit_bytes =
      local.plan.ac4.activation_peak_bytes - 1;
  expect_error(&local, comm, &local.batch, local.workspace,
               STATUS(ACTIVATION_LIMIT));
  local.config.ac4.activation_limit_bytes = 0;

  const unsigned csr = _mm_getcsr();
  _mm_setcsr(csr | 0x8040u);
  expect_error(&local, comm, &local.batch, local.workspace,
               STATUS(UNSUPPORTED_ENVIRONMENT));
  CHECK(_mm_getcsr() == (csr | 0x8040u));
  _mm_setcsr(csr);

  expect_error(&local, comm, NULL, local.workspace, STATUS(INVALID_ARGUMENT));
  et_wave4_composed_batch_v1 batch = local.batch;
  batch.target.data = NULL;
  expect_error(&local, comm, &batch, local.workspace, STATUS(INVALID_ARGUMENT));
  batch.target.data = batch.input.data;
  expect_error(&local, comm, &batch, local.workspace, STATUS(ALIASING));
  batch = local.batch;
  batch.input.data = local.master + local.state.parameter_count - 1;
  expect_error(&local, comm, &batch, local.workspace, STATUS(ALIASING));
  et_wave4_composed_workspace_v1 short_workspace = local.workspace;
  --short_workspace.bytes;
  expect_error(&local, comm, &local.batch, short_workspace,
               STATUS(WORKSPACE_TOO_SMALL));
  local.master[0] = FLT_MAX;
  expect_error(&local, comm, &local.batch, local.workspace,
               STATUS(NUMERIC_ERROR));
  local.master[0] = 1.0f;
  local.state.scaler.clean_steps = scaler.growth_interval;
  expect_error(&local, comm, &local.batch, local.workspace,
               STATUS(INVALID_ARGUMENT));
  local.state.scaler.clean_steps = 0;

  snapshot baseline;
  double loss = 0;
  for (et_ac4_policy policy = ET_AC4_POLICY_NONE;
       policy <= ET_AC4_POLICY_EVERY_K; ++policy) {
    memcpy(local.master, pre_bootstrap.master,
           local.state.parameter_count * sizeof(float));
    local.config.ac4.policy = policy;
    local.config.ac4.checkpoint_every = policy == ET_AC4_POLICY_EVERY_K ? 2 : 0;
    CHECK_STATUS(et_wave4_composed_bootstrap_v1(&local.config, comm, TIMEOUT_MS,
                                                &scaler, &local.state,
                                                local.workspace),
                 STATUS(OK));
    et_wave4_composed_result_v1 result;
    CHECK_STATUS(et_wave4_composed_step_v1(&local.config, comm, TIMEOUT_MS,
                                           &local.state, &local.batch,
                                           local.workspace, &result),
                 STATUS(OK));
    if (policy == ET_AC4_POLICY_NONE) {
      save(&local, &baseline);
      loss = result.loss;
    } else {
      snapshot observed;
      save(&local, &observed);
      check_same_state(&observed, &baseline, local.state.parameter_count);
      CHECK(memcmp(&result.loss, &loss, sizeof(loss)) == 0);
    }
  }
  et_dist5_comm_destroy(comm);
  free(local.workspace.data);
}

int main(void) {
  verify_errors_and_optional_policies();
  char directory[] = "/tmp/eshkol-wave4-composed.XXXXXX";
  CHECK(mkdtemp(directory) != NULL);
  char path[sizeof(directory) + sizeof("/fixture.eval5")];
  CHECK(snprintf(path, sizeof(path), "%s/fixture.eval5", directory) > 0);
  verify_matrix(path);
  CHECK(rmdir(directory) == 0);
  puts("Wave 4/5 composed process matrix, faults, and manifest: PASS");
  return EXIT_SUCCESS;
}
