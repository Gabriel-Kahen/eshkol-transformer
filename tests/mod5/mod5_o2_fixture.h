#ifndef ESHKOL_TEST_MOD5_O2_FIXTURE_H
#define ESHKOL_TEST_MOD5_O2_FIXTURE_H

/* Include after the native MOD5 training fixture, which owns its geometry. */
#include "o2_optimizer_internal.h"

enum {
  UPDATE_COUNT = 6,
  MAX_PARAMETER_ELEMENTS = WIDTH * WIDTH,
  MOMENT_COUNT = 2,
  OPTION_COUNT = ET_O2_OPTION_WEIGHT_DECAY + 1
};

typedef struct parameter_shape {
  size_t rank;
  uint64_t extents[2];
} parameter_shape;

typedef struct live_model {
  et_f32_parameter *parameters[PARAMETERS];
  et_o2_optimizer *optimizer;
} live_model;

typedef struct state_bits {
  uint32_t values[PARAMETERS][MAX_PARAMETER_ELEMENTS];
  uint32_t moments[PARAMETERS][MOMENT_COUNT][MAX_PARAMETER_ELEMENTS];
  uint32_t options[PARAMETERS][OPTION_COUNT];
  uint64_t completed;
  uint32_t schedule_factor;
} state_bits;

static const et_o2_state_reconstruct_config_v1 run_config = {
    .struct_size = sizeof(run_config),
    .clip_kind = ET_O2_CLIP_NONE,
    .schedule_kind = ET_O2_SCHEDULE_LINEAR,
    .minimum_ratio_bits = 0x3e800000u, /* 0.25 */
    .warmup_updates = 4,
    .total_updates = UPDATE_COUNT,
    .parameter_count = PARAMETERS,
};
static const uint32_t run_options[OPTION_COUNT] = {
    0x3c800000u, /* learning rate 0.015625 */
    0x3f600000u, /* beta1 0.875 */
    0x3f780000u, /* beta2 0.96875 */
    0x35800000u, /* epsilon 2^-20 */
    0x3c800000u, /* weight decay 0.015625 */
};

static const uint32_t fresh_options[OPTION_COUNT] = {
    0x00000000u, /* learning rate 0 */
    0x3f000000u, /* beta1 0.5 */
    0x3f400000u, /* beta2 0.75 */
    0x37800000u, /* epsilon 2^-16 */
    0x00000000u, /* weight decay 0 */
};

static parameter_shape model_shape(size_t parameter, size_t kv_heads) {
  if (parameter >= GAMMA1)
    return (parameter_shape){1, {WIDTH, 0}};
  return (parameter_shape){2, {param_size(parameter, kv_heads) / WIDTH, WIDTH}};
}

static et_f32_tensor *new_tensor(parameter_shape shape, const uint32_t *bits,
                                 size_t count) {
  et_f32_tensor *tensor = NULL;
  CHECK(et_f32_tensor_create_v1(shape.rank, shape.extents, &tensor, NULL) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, bits, count, NULL) == 0);
  return tensor;
}

static live_model create_model(const state_bits *state, size_t kv_heads,
                               int fresh_optimizer, int wrong_shape) {
  live_model model = {0};
  et_o2_optimizer_builder *builder = NULL;
  const uint32_t schedule =
      fresh_optimizer ? ET_O2_SCHEDULE_CONSTANT : run_config.schedule_kind;
  CHECK(et_o2_optimizer_builder_create_v1(
            PARAMETERS, run_config.clip_kind, run_config.clip_max_bits,
            schedule, fresh_optimizer ? 0 : run_config.warmup_updates,
            fresh_optimizer ? 0 : run_config.total_updates,
            fresh_optimizer ? 0x3f800000u : run_config.minimum_ratio_bits,
            &builder, NULL) == 0);
  for (size_t p = 0; p < PARAMETERS; ++p) {
    parameter_shape shape = model_shape(p, kv_heads);
    if (wrong_shape && p == GAMMA2)
      shape = (parameter_shape){2, {1, WIDTH}};
    et_f32_tensor *initial =
        new_tensor(shape, state->values[p], param_size(p, kv_heads));
    CHECK(et_f32_parameter_create_v1(initial, &model.parameters[p], NULL) == 0);
    CHECK(et_f32_tensor_destroy_v1(&initial, NULL) == 0);
    CHECK(et_f32_parameter_bind_identity_v1(model.parameters[p],
                                            model.parameters[p], NULL) == 0);
    uint32_t options[OPTION_COUNT];
    for (size_t option = 0; option < OPTION_COUNT; ++option)
      options[option] = fresh_optimizer
                            ? fresh_options[option]
                            : run_options[option] + (uint32_t)p * 0x4000u;
    CHECK(et_o2_optimizer_builder_set_v1(
              builder, p, model.parameters[p], model.parameters[p],
              options[ET_O2_OPTION_LEARNING_RATE], options[ET_O2_OPTION_BETA1],
              options[ET_O2_OPTION_BETA2], options[ET_O2_OPTION_EPSILON],
              options[ET_O2_OPTION_WEIGHT_DECAY], NULL) == 0);
  }
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &model.optimizer, NULL) ==
        0);
  return model;
}

static void copy_values(const live_model *model, size_t kv_heads,
                        uint32_t values[PARAMETERS][MAX_PARAMETER_ELEMENTS]) {
  for (size_t p = 0; p < PARAMETERS; ++p) {
    et_f32_tensor *clone = NULL;
    CHECK(et_f32_parameter_value_snapshot_v1(model->parameters[p], &clone,
                                             NULL) == 0);
    CHECK(et_f32_tensor_copy_bits_to_v1(clone, values[p],
                                        param_size(p, kv_heads), NULL) == 0);
    CHECK(et_f32_tensor_destroy_v1(&clone, NULL) == 0);
  }
}

static state_bits capture(const live_model *model, size_t kv_heads) {
  state_bits result = {0};
  et_o2_optimizer_state *snapshot = NULL;
  size_t entries = 0;
  copy_values(model, kv_heads, result.values);
  CHECK(et_o2_optimizer_state_snapshot_v1(model->optimizer, &snapshot, NULL) ==
        0);
  CHECK(et_o2_optimizer_state_entry_count_v1(snapshot, &entries, NULL) == 0);
  CHECK(entries == PARAMETERS);
  CHECK(et_o2_optimizer_state_completed_updates_v1(snapshot, &result.completed,
                                                   NULL) == 0);
  CHECK(et_o2_optimizer_schedule_factor_bits_v1(
            model->optimizer, &result.schedule_factor, NULL) == 0);
  for (size_t p = 0; p < PARAMETERS; ++p) {
    for (uint32_t moment = 0; moment < MOMENT_COUNT; ++moment)
      CHECK(et_o2_optimizer_state_copy_moment_bits_v1(
                snapshot, p, moment, result.moments[p][moment],
                param_size(p, kv_heads), NULL) == 0);
    for (uint32_t option = 0; option < OPTION_COUNT; ++option)
      CHECK(et_o2_optimizer_state_option_bits_v1(
                snapshot, p, option, &result.options[p][option], NULL) == 0);
  }
  CHECK(et_o2_optimizer_state_release_v1(snapshot, NULL) == 0);
  return result;
}

static void check_same_state(const state_bits *actual,
                             const state_bits *expected) {
  CHECK(memcmp(actual->values, expected->values, sizeof(actual->values)) == 0);
  CHECK(memcmp(actual->moments, expected->moments, sizeof(actual->moments)) ==
        0);
  CHECK(memcmp(actual->options, expected->options, sizeof(actual->options)) ==
        0);
  CHECK(actual->completed == expected->completed);
  CHECK(actual->schedule_factor == expected->schedule_factor);
}

static void apply_o2_gradient(const live_model *model, size_t kv_heads,
                              const weights *gradient) {
  et_f32_tensor *numerators[PARAMETERS] = {0};
  et_f32_gradient_contribution_v1 contributions[PARAMETERS];
  for (size_t p = 0; p < PARAMETERS; ++p) {
    const size_t count = param_size(p, kv_heads);
    uint32_t words[MAX_PARAMETER_ELEMENTS];
    memcpy(words, gradient->data[p], count * sizeof(words[0]));
    numerators[p] = new_tensor(model_shape(p, kv_heads), words, count);
    contributions[p] = (et_f32_gradient_contribution_v1){
        .struct_size = sizeof(contributions[p]),
        .destination = model->parameters[p],
        .weighted_numerator = numerators[p],
        .expected_ordinal = 0,
    };
  }
  et_f32_gradient_plan *plan = NULL;
  CHECK(et_f32_gradient_plan_prepare_v1(PARAMETERS, contributions, 0x3f800000u,
                                        &plan, NULL) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, NULL) == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, NULL) == 0);
  for (size_t p = 0; p < PARAMETERS; ++p)
    CHECK(et_f32_tensor_destroy_v1(&numerators[p], NULL) == 0);
  CHECK(et_o2_optimizer_step_v1(model->optimizer, NULL) == 0);
  CHECK(et_o2_optimizer_zero_grad_v1(model->optimizer, NULL) == 0);
}

#endif
