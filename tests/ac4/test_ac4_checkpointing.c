#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/ac4_checkpointing_abi.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

enum { MAX_X = 48, MAX_P = ET_AC4_MAX_LAYERS * 464, POLICIES = 4, STEPS = 8 };
typedef struct fixture {
  size_t x, p, per_layer;
  float input[MAX_X], target[MAX_X], parameters[MAX_P];
} fixture;

typedef struct run {
  et_ac4_config_v1 config;
  et_ac4_memory_v1 plan;
  et_ac4_result_v1 result;
  et_ac4_io_v1 io;
  float output[MAX_X], dx[MAX_X], grad[MAX_P];
  float *workspace;
} run;

static const et_ac4_shape_v1 shapes[] = {{4, 1, 2, 2, 1, 2},
                                         {2, 2, 3, 4, 1, 2},
                                         {3, 2, 3, 4, 2, 2},
                                         {2, 2, 3, 2, 2, 2}};
static const et_ac4_policy policies[] = {
    ET_AC4_POLICY_NONE, ET_AC4_POLICY_EVERY_LAYER, ET_AC4_POLICY_SQRT_SEGMENTS,
    ET_AC4_POLICY_EVERY_K};
static const char *const policy_names[] = {"none", "layer", "sqrt", "every-k"};

static et_ac4_config_v1 config_for(et_ac4_shape_v1 shape, et_ac4_policy policy,
                                   float rate, uint32_t k) {
  return (et_ac4_config_v1){.shape = shape,
                            .mode = ET_AC4_FORWARD_BACKWARD,
                            .policy = policy,
                            .checkpoint_every = k,
                            .epsilon = 0.125f,
                            .dropout_rate = rate,
                            .seed = UINT64_C(0x570efdb000311fb0),
                            .step = 0};
}

static float sample(uint32_t *state) {
  *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
  return (float)(*state >> 16) * 0x1p-15f - 1.0f;
}

static void fill_fixture(fixture *data, et_ac4_shape_v1 shape) {
  const size_t width = shape.query_heads * shape.head_dim;
  const size_t kv_width = shape.kv_heads * shape.head_dim;
  const size_t gamma2 = width + 2 * width * width + 2 * kv_width * width;
  const size_t per_layer = gamma2 + width + 3 * width * width;
  data->x = shape.batch * shape.tokens * width;
  data->p = shape.layers * per_layer;
  data->per_layer = per_layer;
  CHECK(data->x <= MAX_X && data->p <= MAX_P);
  uint32_t state = 41;
  for (size_t i = 0; i < data->x; ++i) {
    data->input[i] = 0.65f * sample(&state);
    data->target[i] = 0.62f * data->input[i] + 0.27f * sample(&state);
  }
  for (size_t i = 0; i < data->p; ++i)
    data->parameters[i] = 0.28f * sample(&state);
  for (uint32_t layer = 0; layer < shape.layers; ++layer) {
    float *parameters = data->parameters + layer * per_layer;
    for (size_t i = 0; i < width; ++i) {
      parameters[i] = 0.92f + 0.16f * sample(&state);
      parameters[gamma2 + i] = 0.87f + 0.16f * sample(&state);
    }
  }
}

static void bind_inputs(run *trial, const fixture *data) {
  trial->io.input = (et_ac4_const_span_v1){data->input, data->x};
  trial->io.target = (et_ac4_const_span_v1){data->target, data->x};
  trial->io.parameters = (et_ac4_const_span_v1){data->parameters, data->p};
}

static void init_run(run *trial, et_ac4_config_v1 config, const fixture *data) {
  trial->config = config;
  CHECK(et_ac4_plan_v1(&config, &trial->plan) == ET_AC4_OK);
  CHECK(trial->plan.activation_floats == data->x);
  CHECK(trial->plan.parameter_floats == data->p);
  CHECK(trial->plan.parameter_floats_per_layer == data->per_layer);
  CHECK(trial->plan.workspace_bytes ==
        trial->plan.activation_peak_bytes + trial->plan.gradient_staging_bytes);
  trial->workspace = malloc(trial->plan.workspace_bytes);
  CHECK(trial->workspace != NULL);
  trial->io = (et_ac4_io_v1){.output = {trial->output, data->x},
                             .workspace = trial->workspace,
                             .workspace_bytes = trial->plan.workspace_bytes};
  if (config.mode == ET_AC4_FORWARD_BACKWARD) {
    trial->io.d_input = (et_ac4_span_v1){trial->dx, data->x};
    trial->io.d_parameters = (et_ac4_span_v1){trial->grad, data->p};
  }
  bind_inputs(trial, data);
}

static void close_run(run *trial) { free(trial->workspace); }

static void execute(run *trial) {
  const et_ac4_status status =
      et_ac4_run_v1(&trial->config, &trial->io, &trial->result);
  if (status != ET_AC4_OK)
    fprintf(stderr, "run status=%u policy=%u L=%u step=%llu\n", status,
            trial->config.policy, trial->config.shape.layers,
            (unsigned long long)trial->config.step);
  CHECK(status == ET_AC4_OK);
}

static void check_stats(const run *trial) {
  const et_ac4_memory_v1 *p = &trial->plan;
  const et_ac4_memory_v1 *m = &trial->result.memory;
  CHECK(m->activation_floats == p->activation_floats);
  CHECK(m->parameter_floats_per_layer == p->parameter_floats_per_layer);
  CHECK(m->parameter_floats == p->parameter_floats);
  CHECK(m->activation_peak_bytes == p->activation_peak_bytes);
  CHECK(m->gradient_staging_bytes == p->gradient_staging_bytes);
  CHECK(m->workspace_bytes == p->workspace_bytes);
  CHECK(m->retained_checkpoints == p->retained_checkpoints);
  CHECK(m->retained_tapes == p->retained_tapes);
  CHECK(m->rematerialized_layers == p->rematerialized_layers);
}

static void check_forward_parity(const run *expected, const run *actual) {
  const size_t bytes = expected->io.output.count * sizeof(float);
  CHECK(memcmp(&expected->result.loss, &actual->result.loss, sizeof(double)) ==
        0);
  CHECK(memcmp(expected->output, actual->output, bytes) == 0);
}

static void check_parity(const run *expected, const run *actual) {
  check_forward_parity(expected, actual);
  CHECK(memcmp(expected->dx, actual->dx,
               actual->io.d_input.count * sizeof(float)) == 0);
  CHECK(memcmp(expected->grad, actual->grad,
               actual->io.d_parameters.count * sizeof(float)) == 0);
}

static void update(fixture *data, const run *trial) {
  for (size_t i = 0; i < data->p; ++i)
    data->parameters[i] -= 0.035f * trial->grad[i];
}

static void train_parity(void) {
  const uint32_t depths[] = {4, 8, 16};
  const float rates[] = {0.0f, 0.2f};
  for (size_t d = 0; d < COUNT(depths); ++d) {
    et_ac4_shape_v1 shape = shapes[2];
    shape.layers = depths[d];
    for (size_t r = 0; r < COUNT(rates); ++r) {
      fixture data[POLICIES];
      run trials[POLICIES];
      fixture evaluation;
      fill_fixture(&evaluation, shape);
      et_ac4_config_v1 eval_config =
          config_for(shape, ET_AC4_POLICY_NONE, 0, 0);
      eval_config.mode = ET_AC4_FORWARD_ONLY;
      run eval;
      init_run(&eval, eval_config, &evaluation);
      for (size_t p = 0; p < COUNT(policies); ++p) {
        data[p] = evaluation;
        init_run(
            &trials[p],
            config_for(shape, policies[p], rates[r], p == 3 ? shape.layers : 0),
            &data[p]);
        trials[p].config.activation_limit_bytes =
            trials[p].plan.activation_peak_bytes;
      }
      execute(&eval);
      const double initial = eval.result.loss;
      double previous = initial;
      bind_inputs(&eval, &data[0]);
      for (uint64_t step = 0; step < STEPS; ++step) {
        for (size_t p = 0; p < COUNT(policies); ++p) {
          trials[p].config.step = step;
          execute(&trials[p]);
          check_stats(&trials[p]);
          if (p != 0)
            check_parity(&trials[0], &trials[p]);
          update(&data[p], &trials[p]);
          CHECK(memcmp(data[0].parameters, data[p].parameters,
                       data[p].p * sizeof(float)) == 0);
        }
        eval.config.step = step;
        eval.config.dropout_rate = rates[r];
        execute(&eval);
        CHECK(eval.result.loss < trials[0].result.loss);
        eval.config.dropout_rate = 0.0f;
        execute(&eval);
        CHECK(eval.result.loss < previous);
        previous = eval.result.loss;
      }
      printf("AC4 train L=%u dropout=%.3f steps=%d eval_loss=%.9g->%.9g "
             "parity=PASS\n",
             shape.layers, rates[r], STEPS, initial, previous);
      for (size_t p = 0; p < COUNT(policies); ++p)
        close_run(&trials[p]);
      close_run(&eval);
    }
  }
}

static void schedules(void) {
  const uint32_t depths[] = {4, 8, 16};
  const size_t expected[][POLICIES] = {{8832, 3648, 3456, 3264},
                                       {16512, 4416, 3840, 3264},
                                       {31872, 5952, 4224, 3264}};
  const uint32_t replays[][POLICIES] = {
      {0, 4, 6, 10}, {0, 8, 13, 36}, {0, 16, 28, 136}};
  for (size_t i = 0; i < COUNT(depths); ++i) {
    et_ac4_shape_v1 shape = shapes[2];
    shape.layers = depths[i];
    fixture data;
    fill_fixture(&data, shape);
    for (size_t p = 0; p < COUNT(policies); ++p) {
      run trial;
      init_run(&trial,
               config_for(shape, policies[p], 0.25f, p == 3 ? shape.layers : 0),
               &data);
      execute(&trial);
      check_stats(&trial);
      CHECK(trial.result.memory.activation_peak_bytes == expected[i][p]);
      CHECK(trial.result.memory.rematerialized_layers == replays[i][p]);
      close_run(&trial);
    }
    CHECK(expected[i][0] > expected[i][1] && expected[i][1] > expected[i][2] &&
          expected[i][2] > expected[i][3]);
  }
  CHECK(expected[2][0] - expected[0][0] >
        4 * (expected[2][1] - expected[0][1]));
  CHECK(expected[2][1] - expected[0][1] > expected[2][2] - expected[0][2]);
  CHECK(expected[2][3] == expected[0][3]);

  /* Exercise ragged segments, every legal depth and K, and loss-only parity. */
  unsigned cases = 0;
  for (uint32_t layers = 1; layers <= ET_AC4_MAX_LAYERS; ++layers) {
    et_ac4_shape_v1 shape = shapes[0];
    shape.layers = layers;
    fixture data;
    fill_fixture(&data, shape);
    run baseline;
    init_run(&baseline, config_for(shape, ET_AC4_POLICY_NONE, 0.3f, 0), &data);
    execute(&baseline);
    for (size_t p = 0; p < COUNT(policies); ++p) {
      et_ac4_config_v1 config =
          config_for(shape, policies[p], 0.3f, p == 3 ? layers : 0);
      config.mode = ET_AC4_FORWARD_ONLY;
      run trial;
      init_run(&trial, config, &data);
      execute(&trial);
      check_stats(&trial);
      check_forward_parity(&baseline, &trial);
      CHECK(trial.plan.gradient_staging_bytes == 0);
      CHECK(trial.plan.rematerialized_layers == 0);
      close_run(&trial);
    }
    for (uint32_t k = 1; k <= layers; ++k) {
      run trial;
      init_run(&trial, config_for(shape, ET_AC4_POLICY_EVERY_K, 0.3f, k),
               &data);
      execute(&trial);
      check_stats(&trial);
      check_parity(&baseline, &trial);
      CHECK(trial.result.memory.retained_checkpoints == (layers - 1) / k);
      const uint32_t tail = layers % k;
      const uint32_t replay_count =
          (layers / k) * k * (k + 1) / 2 + tail * (tail + 1) / 2;
      CHECK(trial.result.memory.rematerialized_layers == replay_count);
      close_run(&trial);
      ++cases;
    }
    run sqrt_trial;
    init_run(&sqrt_trial,
             config_for(shape, ET_AC4_POLICY_SQRT_SEGMENTS, 0.3f, 0), &data);
    execute(&sqrt_trial);
    check_stats(&sqrt_trial);
    check_parity(&baseline, &sqrt_trial);
    close_run(&sqrt_trial);
    close_run(&baseline);
  }
  printf("AC4 exact arena peaks and schedules: PASS (%u EVERY_K, 16 SQRT, 64 "
         "forward-only cases)\n",
         cases);
}

static void derivatives(void) {
  const float rates[] = {0, 0.25f};
  size_t checked = 0;
  double max_absolute = 0.0;
  for (size_t s = 0; s < COUNT(shapes); ++s) {
    fixture data;
    fill_fixture(&data, shapes[s]);
    for (size_t r = 0; r < COUNT(rates); ++r) {
      et_ac4_config_v1 cfg =
          config_for(shapes[s], ET_AC4_POLICY_SQRT_SEGMENTS, rates[r], 0);
      cfg.step = 19;
      run analytic;
      init_run(&analytic, cfg, &data);
      execute(&analytic);
      cfg.mode = ET_AC4_FORWARD_ONLY;
      run numeric;
      init_run(&numeric, cfg, &data);
      execute(&numeric);
      check_forward_parity(&analytic, &numeric);
      for (size_t i = 0; i < data.p + data.x; ++i) {
        float *slot =
            i < data.p ? &data.parameters[i] : &data.input[i - data.p];
        const float value = *slot;
        const float hi = value + 0x1p-10f;
        const float lo = value - 0x1p-10f;
        *slot = hi;
        execute(&numeric);
        const double plus = numeric.result.loss;
        *slot = lo;
        execute(&numeric);
        const double minus = numeric.result.loss;
        *slot = value;
        const double estimate = (plus - minus) / ((double)hi - lo);
        const double gradient =
            i < data.p ? analytic.grad[i] : analytic.dx[i - data.p];
        const double error = fabs(estimate - gradient);
        if (error > max_absolute)
          max_absolute = error;
        if (error > 7e-6 + 0.003 * fabs(gradient))
          fprintf(stderr,
                  "derivative shape=%zu dropout=%.3f i=%zu analytic=%.9g "
                  "numeric=%.9g\n",
                  s, rates[r], i, gradient, estimate);
        CHECK(error <= 7e-6 + 0.003 * fabs(gradient));
        ++checked;
      }
      close_run(&numeric);
      close_run(&analytic);
    }
  }
  printf("AC4 central differences: PASS (%zu derivatives, max absolute error "
         "%.9g)\n",
         checked, max_absolute);
}

static void mask_reference(void) {
  enum {
    D = 4,
    V = 2,
    WV = D + D * D + V * D,
    WO = WV + V * D,
    GAMMA2 = WO + D * D,
    WGATE = GAMMA2 + D,
    WUP = WGATE + D * D,
    WDOWN = WUP + D * D
  };
  et_ac4_shape_v1 shape = shapes[0];
  shape.layers = 1;
  fixture data;
  fill_fixture(&data, shape);
  for (size_t i = 0; i < data.x; ++i)
    data.input[i] = i % D == 0 ? 0.0f : 1.0f;
  et_ac4_config_v1 cfg = config_for(shape, ET_AC4_POLICY_NONE, 0, 0);
  cfg.mode = ET_AC4_FORWARD_ONLY;
  run trial;
  init_run(&trial, cfg, &data);

  /* Independent SplitMix64 counter goldens for feature zero of both tokens.
   * Turn on one branch at a time and route it into the zero residual feature.
   */
  const uint64_t steps[] = {0, 5};
  const int keep[2][2][2] = {{{1, 1}, {1, 0}}, {{0, 1}, {1, 1}}};
  for (size_t branch = 0; branch < COUNT(keep); ++branch) {
    memset(data.parameters, 0, data.p * sizeof(float));
    for (size_t d = 0; d < D; ++d) {
      data.parameters[d] = 1.0f;
      data.parameters[GAMMA2 + d] = 1.0f;
    }
    if (branch == 0) {
      data.parameters[WV + 1] = 1.0f;
      data.parameters[WO] = 1.0f;
    } else {
      data.parameters[WGATE + 1] = 1.0f;
      data.parameters[WUP + 2] = 1.0f;
      data.parameters[WDOWN] = 1.0f;
    }
    trial.config.dropout_rate = 0.0f;
    execute(&trial);
    const float values[] = {trial.output[0], trial.output[D]};
    CHECK(values[0] > 0.0f && values[1] > 0.0f);
    trial.config.dropout_rate = 0.5f;
    for (size_t i = 0; i < COUNT(steps); ++i) {
      trial.config.step = steps[i];
      execute(&trial);
      for (size_t t = 0; t < COUNT(values); ++t) {
        const float expected = keep[branch][i][t] ? 2.0f * values[t] : 0.0f;
        CHECK(memcmp(&expected, &trial.output[t * D], sizeof(expected)) == 0);
      }
    }
  }
  close_run(&trial);
}

static void masks_and_causality(void) {
  mask_reference();
  et_ac4_shape_v1 shape = shapes[2];
  fixture data;
  fill_fixture(&data, shape);
  run reference, changed;
  const et_ac4_config_v1 cfg =
      config_for(shape, ET_AC4_POLICY_EVERY_K, 0.25f, 2);
  init_run(&reference, cfg, &data);
  init_run(&changed, cfg, &data);
  execute(&reference);
  execute(&changed);
  check_parity(&reference, &changed);
  changed.config.step = UINT64_MAX;
  execute(&changed);
  CHECK(memcmp(reference.output, changed.output, data.x * sizeof(float)) != 0);
  changed.config = cfg;
  changed.config.seed ^= UINT64_C(1);
  execute(&changed);
  CHECK(memcmp(reference.output, changed.output, data.x * sizeof(float)) != 0);

  changed.config = cfg;
  changed.config.dropout_rate = 0;
  reference.config.dropout_rate = 0;
  execute(&reference);
  const size_t width = shape.query_heads * shape.head_dim;
  const size_t batch = shape.tokens * width;
  for (size_t i = width; i < batch; ++i)
    data.input[i] += 0.3f;
  execute(&changed);
  CHECK(memcmp(reference.output, changed.output, width * sizeof(float)) == 0);
  CHECK(memcmp(reference.output + batch, changed.output + batch,
               batch * sizeof(float)) == 0);
  CHECK(memcmp(reference.output + width, changed.output + width,
               (batch - width) * sizeof(float)) != 0);
  close_run(&changed);
  close_run(&reference);
  puts("AC4 RNG goldens and seed/step replay, causal order and batch "
       "isolation: PASS");
}

static unsigned rejected;
static void expect_rejected(run *trial, const et_ac4_config_v1 *config,
                            const et_ac4_io_v1 *io, et_ac4_result_v1 *result,
                            et_ac4_status expected, int scratch_untouched) {
  memset(trial->output, 0xa1, sizeof(trial->output));
  memset(trial->dx, 0xa2, sizeof(trial->dx));
  memset(trial->grad, 0xa3, sizeof(trial->grad));
  memset(&trial->result, 0xa4, sizeof(trial->result));
  memset(trial->workspace, 0xa5, trial->plan.workspace_bytes);
  CHECK(et_ac4_run_v1(config, io, result) == expected);
  const unsigned char *buffers[] = {
      (const unsigned char *)trial->output, (const unsigned char *)trial->dx,
      (const unsigned char *)trial->grad, (const unsigned char *)&trial->result,
      (const unsigned char *)trial->workspace};
  const size_t sizes[] = {sizeof(trial->output), sizeof(trial->dx),
                          sizeof(trial->grad), sizeof(trial->result),
                          trial->plan.workspace_bytes};
  size_t changed_scratch = 0;
  for (size_t b = 0; b < COUNT(buffers); ++b) {
    for (size_t i = 0; i < sizes[b]; ++i) {
      if (b == 4 && !scratch_untouched)
        changed_scratch += buffers[b][i] != 0xa1 + b;
      else
        CHECK(buffers[b][i] == 0xa1 + b);
    }
  }
  if (!scratch_untouched)
    CHECK(changed_scratch != 0);
  ++rejected;
}

static void errors(void) {
  const et_ac4_shape_v1 shape = shapes[0];
  fixture data;
  fill_fixture(&data, shape);
  run trial;
  const et_ac4_config_v1 valid = config_for(shape, ET_AC4_POLICY_EVERY_K, 0, 3);
  init_run(&trial, valid, &data);

  et_ac4_config_v1 cfg;
  et_ac4_io_v1 io;
#define BAD_CONFIG(field, value, status)                                       \
  do {                                                                         \
    cfg = valid;                                                               \
    cfg.field = (value);                                                       \
    expect_rejected(&trial, &cfg, &trial.io, &trial.result, status, 1);        \
  } while (0)
#define BAD_IO(field, value, status)                                           \
  do {                                                                         \
    io = trial.io;                                                             \
    io.field = (value);                                                        \
    expect_rejected(&trial, &valid, &io, &trial.result, status, 1);            \
  } while (0)

  BAD_CONFIG(shape.layers, 0, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(shape.layers, ET_AC4_MAX_LAYERS + 1, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(shape.tokens, 0, ET_AC4_SHAPE_MISMATCH);
  BAD_CONFIG(shape.kv_heads, 3, ET_AC4_SHAPE_MISMATCH);
  BAD_CONFIG(shape.head_dim, UINT32_MAX, ET_AC4_UNSUPPORTED);
  BAD_CONFIG(shape.tokens, 4, ET_AC4_UNSUPPORTED);
  BAD_CONFIG(policy, UINT32_MAX, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(policy, ET_AC4_POLICY_NONE, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(mode, UINT32_MAX, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(checkpoint_every, 0, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(checkpoint_every, shape.layers + 1, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(epsilon, 0, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(epsilon, INFINITY, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(dropout_rate, -0.1f, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(dropout_rate, 1, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(dropout_rate, NAN, ET_AC4_INVALID_ARGUMENT);
  BAD_CONFIG(activation_limit_bytes, trial.plan.activation_peak_bytes - 1,
             ET_AC4_ACTIVATION_LIMIT);
  CHECK(et_ac4_plan_v1(&cfg, &trial.plan) == ET_AC4_OK);

  BAD_IO(input.count, 0, ET_AC4_SHAPE_MISMATCH);
  BAD_IO(target.count, data.x + 1, ET_AC4_SHAPE_MISMATCH);
  BAD_IO(parameters.count, SIZE_MAX, ET_AC4_SHAPE_MISMATCH);
  BAD_IO(output.count, data.x - 1, ET_AC4_SHAPE_MISMATCH);
  BAD_IO(d_input.count, data.x - 1, ET_AC4_SHAPE_MISMATCH);
  BAD_IO(d_parameters.count, 0, ET_AC4_SHAPE_MISMATCH);
  BAD_IO(input.data, NULL, ET_AC4_INVALID_ARGUMENT);
  BAD_IO(target.data, NULL, ET_AC4_INVALID_ARGUMENT);
  BAD_IO(parameters.data, NULL, ET_AC4_INVALID_ARGUMENT);
  BAD_IO(output.data, NULL, ET_AC4_INVALID_ARGUMENT);
  BAD_IO(d_input.data, NULL, ET_AC4_INVALID_ARGUMENT);
  BAD_IO(d_parameters.data, NULL, ET_AC4_INVALID_ARGUMENT);
  BAD_IO(workspace, NULL, ET_AC4_INVALID_ARGUMENT);
  BAD_IO(workspace, (float *)((unsigned char *)trial.workspace + 1),
         ET_AC4_INVALID_ARGUMENT);
  BAD_IO(workspace_bytes, SIZE_MAX, ET_AC4_INVALID_ARGUMENT);
  BAD_IO(workspace_bytes, trial.plan.workspace_bytes - 1,
         ET_AC4_WORKSPACE_TOO_SMALL);
  BAD_IO(workspace_bytes, 0, ET_AC4_WORKSPACE_TOO_SMALL);
  BAD_IO(target.data, data.input + 1, ET_AC4_ALIASING);
  BAD_IO(output.data, data.parameters + 1, ET_AC4_ALIASING);
  BAD_IO(d_input.data, trial.output + 1, ET_AC4_ALIASING);
  BAD_IO(d_parameters.data, trial.workspace, ET_AC4_ALIASING);
  BAD_IO(workspace, (float *)(void *)&trial.result, ET_AC4_ALIASING);
  io = trial.io;
  io.output.data = (float *)(void *)&trial.config;
  expect_rejected(&trial, &trial.config, &io, &trial.result, ET_AC4_ALIASING,
                  1);

#undef BAD_IO
#undef BAD_CONFIG
  expect_rejected(&trial, NULL, &trial.io, &trial.result,
                  ET_AC4_INVALID_ARGUMENT, 1);
  expect_rejected(&trial, &valid, NULL, &trial.result, ET_AC4_INVALID_ARGUMENT,
                  1);
  expect_rejected(&trial, &valid, &trial.io, NULL, ET_AC4_INVALID_ARGUMENT, 1);
  expect_rejected(&trial, &valid, &trial.io,
                  (et_ac4_result_v1 *)(void *)&trial.io, ET_AC4_ALIASING, 1);

  cfg = valid;
  cfg.mode = ET_AC4_FORWARD_ONLY;
  io = trial.io;
  io.d_input.count = 0;
  io.d_parameters = (et_ac4_span_v1){0};
  expect_rejected(&trial, &cfg, &io, &trial.result, ET_AC4_INVALID_ARGUMENT, 1);

  et_ac4_memory_v1 plan;
  memset(&plan, 0xb7, sizeof(plan));
  cfg = valid;
  cfg.shape.layers = 0;
  CHECK(et_ac4_plan_v1(&cfg, &plan) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_plan_v1(NULL, &plan) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_plan_v1(&valid, NULL) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_plan_v1(&valid, (et_ac4_memory_v1 *)(void *)&valid) ==
        ET_AC4_ALIASING);
  for (size_t i = 0; i < sizeof(plan); ++i)
    CHECK(((unsigned char *)&plan)[i] == 0xb7);

  float saved = data.input[0];
  data.input[0] = NAN;
  expect_rejected(&trial, &valid, &trial.io, &trial.result,
                  ET_AC4_NUMERIC_ERROR, 1);
  data.input[0] = saved;
  saved = data.target[1];
  data.target[1] = -INFINITY;
  expect_rejected(&trial, &valid, &trial.io, &trial.result,
                  ET_AC4_NUMERIC_ERROR, 1);
  data.target[1] = saved;
  saved = data.parameters[data.p - 1];
  data.parameters[data.p - 1] = INFINITY;
  expect_rejected(&trial, &valid, &trial.io, &trial.result,
                  ET_AC4_NUMERIC_ERROR, 1);
  data.parameters[data.p - 1] = saved;
  CHECK(fesetround(FE_DOWNWARD) == 0);
  expect_rejected(&trial, &valid, &trial.io, &trial.result, ET_AC4_UNSUPPORTED,
                  1);
  CHECK(fegetround() == FE_DOWNWARD);
  CHECK(fesetround(FE_TONEAREST) == 0);

  /* Subnormal epsilon must validate via integer bits without trapping when
   * denormals are unmasked or being flushed to zero when DAZ is enabled. */
  et_ac4_config_v1 subnormal_cfg = valid;
  subnormal_cfg.epsilon = FLT_TRUE_MIN;
  const unsigned int orig_mxcsr = _mm_getcsr();
  /* Unmask denormal exception (clear bit 8 = 0x0100). */
  _mm_setcsr(orig_mxcsr & ~0x0100u);
  et_ac4_memory_v1 subnormal_plan;
  CHECK(et_ac4_plan_v1(&subnormal_cfg, &subnormal_plan) == ET_AC4_OK);
  expect_rejected(&trial, &subnormal_cfg, &trial.io, &trial.result,
                  ET_AC4_UNSUPPORTED, 1);
  CHECK(_mm_getcsr() == (orig_mxcsr & ~0x0100u));
  /* Enable DAZ (set bit 6 = 0x0040). */
  _mm_setcsr(orig_mxcsr | 0x0040u);
  CHECK(et_ac4_plan_v1(&subnormal_cfg, &subnormal_plan) == ET_AC4_OK);
  expect_rejected(&trial, &subnormal_cfg, &trial.io, &trial.result,
                  ET_AC4_UNSUPPORTED, 1);
  CHECK(_mm_getcsr() == (orig_mxcsr | 0x0040u));
  _mm_setcsr(orig_mxcsr);

  /* Start with identity layers, then overflow a late forward projection. */
  for (size_t i = 0; i < data.x; ++i) {
    data.input[i] = 1.0f;
    data.target[i] = -16.0f;
  }
  memset(data.parameters, 0, data.p * sizeof(float));
  for (uint32_t l = 0; l < shape.layers; ++l)
    for (size_t d = 0; d < 4; ++d)
      data.parameters[l * data.per_layer + d] = 1.0f;
  float *last = data.parameters + (shape.layers - 1) * data.per_layer;
  for (size_t d = 0; d < 4; ++d)
    last[4 + d] = FLT_MAX;
  expect_rejected(&trial, &valid, &trial.io, &trial.result,
                  ET_AC4_NUMERIC_ERROR, 0);
  memset(last + 4, 0, 4 * sizeof(float));

  /* Zero hidden activations let forward succeed with huge first-layer Wdown.
   * Backward fails in that layer after later layers have staged gradients. */
  const size_t wdown = data.per_layer - 4 * 4;
  data.parameters[wdown] = FLT_MAX;
  run forward_only;
  cfg = valid;
  cfg.mode = ET_AC4_FORWARD_ONLY;
  init_run(&forward_only, cfg, &data);
  execute(&forward_only);
  close_run(&forward_only);
  expect_rejected(&trial, &valid, &trial.io, &trial.result,
                  ET_AC4_NUMERIC_ERROR, 0);
  data.parameters[wdown] = 0;

  execute(&trial);
  check_stats(&trial);
  run clean;
  init_run(&clean, valid, &data);
  execute(&clean);
  check_parity(&clean, &trial);
  close_run(&clean);
  close_run(&trial);
  printf("AC4 rejection and failure atomicity: PASS (%u rejected runs, 4 plan "
         "errors)\n",
         rejected);
}

static double seconds(void) {
  struct timespec ts;
  CHECK(clock_gettime(CLOCK_MONOTONIC, &ts) == 0);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int compare_double(const void *a, const void *b) {
  const double x = *(const double *)a;
  const double y = *(const double *)b;
  return (x > y) - (x < y);
}

static void benchmark(void) {
  enum { SAMPLES = 7, REPETITIONS = 120 };
  const uint32_t depths[] = {4, 8, 16};
  printf("AC4 benchmark GQA dropout=0.2 clock=MONOTONIC samples=%d "
         "repetitions=%d\n",
         SAMPLES, REPETITIONS);
  for (size_t i = 0; i < COUNT(depths); ++i) {
    et_ac4_shape_v1 shape = shapes[2];
    shape.layers = depths[i];
    fixture data;
    fill_fixture(&data, shape);
    run trials[POLICIES];
    double times[POLICIES][SAMPLES];
    for (size_t j = 0; j < COUNT(policies); ++j) {
      init_run(&trials[j],
               config_for(shape, policies[j], 0.2f, j == 3 ? shape.layers : 0),
               &data);
      for (unsigned warmup = 0; warmup < 8; ++warmup)
        execute(&trials[j]);
      check_stats(&trials[j]);
    }
    for (unsigned sample_index = 0; sample_index < SAMPLES; ++sample_index) {
      for (size_t j = 0; j < COUNT(policies); ++j) {
        const size_t p = (j + sample_index) % COUNT(policies);
        const double start = seconds();
        for (unsigned rep = 0; rep < REPETITIONS; ++rep)
          execute(&trials[p]);
        times[p][sample_index] = (seconds() - start) * 1000000 / REPETITIONS;
      }
    }
    for (size_t p = 0; p < COUNT(policies); ++p)
      qsort(times[p], SAMPLES, sizeof(double), compare_double);
    const double baseline = times[0][SAMPLES / 2];
    const size_t baseline_peak = trials[0].result.memory.activation_peak_bytes;
    for (size_t p = 0; p < COUNT(policies); ++p) {
      const et_ac4_memory_v1 *m = &trials[p].result.memory;
      const double us = times[p][SAMPLES / 2];
      printf("AC4 bench L=%u policy=%s k=%u planned=%zu peak=%zu staging=%zu "
             "workspace=%zu "
             "saved_pct=%.2f replay=%u us=%.3f ratio=%.3f\n",
             shape.layers, policy_names[p], trials[p].config.checkpoint_every,
             trials[p].plan.activation_peak_bytes, m->activation_peak_bytes,
             m->gradient_staging_bytes, m->workspace_bytes,
             100.0 * (double)(baseline_peak - m->activation_peak_bytes) /
                 (double)baseline_peak,
             m->rematerialized_layers, us, us / baseline);
      close_run(&trials[p]);
    }
  }
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--benchmark") == 0) {
    benchmark();
    return EXIT_SUCCESS;
  }
  CHECK(argc == 1);
  train_parity();
  schedules();
  derivatives();
  masks_and_causality();
  errors();
  puts("AC4 native suite: PASS");
  return EXIT_SUCCESS;
}
