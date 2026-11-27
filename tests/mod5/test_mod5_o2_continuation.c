#define main mod5_training_smoke_main
#include "test_mod5_training_smoke.c"
#undef main

#include "mod5_o2_fixture.h"

enum { SPLIT_UPDATE = 2 };

static int32_t reconstruct(const state_bits *bits, size_t kv_heads, int corrupt,
                           et_o2_optimizer_state **state) {
  et_o2_state_reconstruct_config_v1 config = run_config;
  config.completed_updates = bits->completed;
  et_o2_state_reconstruct_builder *builder = NULL;
  CHECK(et_o2_optimizer_state_reconstruct_create_v1(&config, &builder, NULL) ==
        0);
  int32_t status = ET_O2_STATUS_OK;
  for (size_t p = 0; p < PARAMETERS && status == ET_O2_STATUS_OK; ++p) {
    const parameter_shape shape = model_shape(p, kv_heads);
    const uint32_t *options = bits->options[p];
    const et_o2_state_reconstruct_entry_v1 entry = {
        .struct_size = sizeof(entry),
        .rank = shape.rank,
        .shape = shape.extents,
        .exp_avg_bits = bits->moments[p][ET_O2_MOMENT_EXP_AVG],
        .exp_avg_sq_bits = bits->moments[p][ET_O2_MOMENT_EXP_AVG_SQ],
        .element_count = param_size(p, kv_heads),
        .learning_rate_bits = options[ET_O2_OPTION_LEARNING_RATE],
        .beta1_bits = options[ET_O2_OPTION_BETA1],
        .beta2_bits = options[ET_O2_OPTION_BETA2],
        .epsilon_bits = options[ET_O2_OPTION_EPSILON],
        .weight_decay_bits = options[ET_O2_OPTION_WEIGHT_DECAY],
    };
    status = et_o2_optimizer_state_reconstruct_set_v1(builder, p, &entry, NULL);
    if (corrupt)
      CHECK(status ==
            (p == GAMMA2 ? ET_O2_STATUS_CORRUPT_DATA : ET_O2_STATUS_OK));
  }
  if (status == ET_O2_STATUS_OK) {
    CHECK(et_o2_optimizer_state_reconstruct_prepare_v1(builder, NULL) == 0);
    status = et_o2_optimizer_state_reconstruct_commit_v1(&builder, state, NULL);
  }
  CHECK(et_o2_optimizer_state_reconstruct_abort_v1(&builder, NULL) == 0);
  return status;
}

static void train_step(const et_kernel_runtime *runtime,
                       const live_model *model, size_t kv_heads, size_t step) {
  uint32_t values[PARAMETERS][MAX_PARAMETER_ELEMENTS] = {{0}};
  weights current, gradient;
  float input[ELEMENTS], target[ELEMENTS], dx[ELEMENTS];
  copy_values(model, kv_heads, values);
  memcpy(&current.data, values, sizeof(values));
  for (size_t i = 0; i < ELEMENTS; ++i) {
    const int seed = (int)((i * 7u + i / 5u + step * 3u) % 31u) - 15;
    input[i] = (float)seed * 0x1p-5f;
    target[i] = 0.65f * input[i] + (float)((int)((i * 3u) % 13u) - 6) * 0x1p-6f;
  }
  CHECK(isfinite(
      block_loss(runtime, kv_heads, &current, input, target, &gradient, dx)));
  apply_o2_gradient(model, kv_heads, &gradient);
}

static void check_negative_reconstruction(const state_bits *split,
                                          size_t kv_heads) {
  et_o2_test_live_counts_v1 before = {.struct_size = sizeof(before)};
  et_o2_test_live_counts_v1 after = {.struct_size = sizeof(after)};
  et_o2_test_live_counts_snapshot_v1(&before);
  state_bits corrupt = *split;
  corrupt.moments[GAMMA2][ET_O2_MOMENT_EXP_AVG_SQ][WIDTH - 1] = 0x7fc00000u;
  et_o2_optimizer_state *unpublished = NULL;
  CHECK(reconstruct(&corrupt, kv_heads, 1, &unpublished) ==
        ET_O2_STATUS_CORRUPT_DATA);
  CHECK(unpublished == NULL);
  et_o2_test_fail_reconstruct_commit_v1(1);
  CHECK(reconstruct(split, kv_heads, 0, &unpublished) == ET_O2_STATUS_INTERNAL);
  et_o2_test_fail_reconstruct_commit_v1(0);
  CHECK(unpublished == NULL);
  et_o2_test_live_counts_snapshot_v1(&after);
  CHECK(after.reconstruct_builders == before.reconstruct_builders);
  CHECK(after.live_states == before.live_states);
  CHECK(after.live_state_handles == before.live_state_handles);
  CHECK(after.owned_state_clones == before.owned_state_clones);
}

static void check_continuation(const et_kernel_runtime *runtime,
                               size_t kv_heads) {
  state_bits initial = {0};
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, kv_heads); ++i) {
      const int seed = (int)((p * 29u + i * 11u + i / 5u) % 37u) - 18;
      const float value = (p >= GAMMA1 ? 1.0f : 0.0f) + (float)seed * 0x1p-7f;
      memcpy(&initial.values[p][i], &value, sizeof(value));
    }
  const live_model uninterrupted = create_model(&initial, kv_heads, 0, 0);
  for (size_t step = 0; step < SPLIT_UPDATE; ++step)
    train_step(runtime, &uninterrupted, kv_heads, step);
  const state_bits split = capture(&uninterrupted, kv_heads);
  CHECK(split.completed == SPLIT_UPDATE);
  CHECK(split.schedule_factor == 0x3f400000u); /* Next warmup factor 0.75. */
  for (size_t p = 0; p < PARAMETERS; ++p) {
    for (size_t moment = 0; moment < MOMENT_COUNT; ++moment) {
      uint32_t activity = 0;
      for (size_t i = 0; i < param_size(p, kv_heads); ++i)
        activity |= split.moments[p][moment][i] & 0x7fffffffu;
      CHECK(activity != 0);
    }
  }
  check_negative_reconstruction(&split, kv_heads);

  et_o2_optimizer_state *restored = NULL;
  CHECK(reconstruct(&split, kv_heads, 0, &restored) == 0);
  const live_model wrong_shape = create_model(&split, kv_heads, 1, 1);
  const state_bits before = capture(&wrong_shape, kv_heads);
  CHECK(et_o2_optimizer_load_state_v1(wrong_shape.optimizer, restored, NULL) ==
        ET_O2_STATUS_SHAPE_MISMATCH);
  const state_bits after = capture(&wrong_shape, kv_heads);
  check_same_state(&after, &before);

  const live_model resumed = create_model(&split, kv_heads, 1, 0);
  for (size_t p = 0; p < PARAMETERS; ++p)
    CHECK(resumed.parameters[p] != uninterrupted.parameters[p]);
  CHECK(et_o2_optimizer_load_state_v1(resumed.optimizer, restored, NULL) == 0);
  CHECK(et_o2_optimizer_state_release_v1(restored, NULL) == 0);
  const state_bits loaded = capture(&resumed, kv_heads);
  check_same_state(&loaded, &split);

  for (size_t step = SPLIT_UPDATE; step < UPDATE_COUNT; ++step) {
    train_step(runtime, &uninterrupted, kv_heads, step);
    train_step(runtime, &resumed, kv_heads, step);
    const state_bits expected = capture(&uninterrupted, kv_heads);
    const state_bits actual = capture(&resumed, kv_heads);
    CHECK(actual.completed == step + 1);
    check_same_state(&actual, &expected);
  }
  printf("MOD5/O2 %s component continuation: PASS (%d + %d updates, exact "
         "weights, moments, options and schedule; malformed input and "
         "pre-commit rollback)\n",
         kv_heads == 1 ? "MQA" : "GQA", SPLIT_UPDATE,
         UPDATE_COUNT - SPLIT_UPDATE);
}

int main(void) {
  et_kernel_runtime *runtime = discover_mod5();
  check_continuation(runtime, 1);
  check_continuation(runtime, 2);
  et_kernel_runtime_destroy(runtime);
  return 0;
}
