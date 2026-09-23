#define ET_O2_TESTING 1
#define ET_F32_TENSOR_TESTING 1
#define ET_I2_PRIVATE_OWNED_CLONE_MATCH 1
#define ET_TR3_C_O2_RESTORE_NATIVE 1
#define ET_TR3_O2_STEP_CLEAR_NATIVE 1
#include "../../native/o2_optimizer.c"

#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

void et_tr3_c_i2_test_fail_owned_release_v1(int64_t enabled);

typedef struct restore_fixture {
  et_f32_parameter *parameters[ET_TR3_C_MODEL_DESTINATIONS];
  et_f32_tensor *prefix_sources[ET_TR3_C_MODEL_DESTINATIONS];
  et_o2_optimizer *optimizer;
  et_o2_tr3_c_restore_request_v1 request;
} restore_fixture;

static int checks;
static int failures;

#define CHECK(condition)                                                       \
  do {                                                                         \
    ++checks;                                                                  \
    if (!(condition)) {                                                        \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,         \
                    #condition);                                               \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static void expect_error(int32_t result, const et_o2_error_v1 *error,
                         uint32_t category, uint32_t code) {
  CHECK(result == (int32_t)category);
  CHECK(error->category == category);
  CHECK(error->code == code);
  CHECK(strcmp(error->operation, "trainer-load-state!") == 0);
}

static et_f32_tensor *tensor(uint32_t bits) {
  const uint64_t shape[] = {1u};
  et_f32_tensor_error error;
  et_f32_tensor *value = NULL;
  if (et_f32_tensor_create_v1(1u, shape, &value, &error) != 0 ||
      et_f32_tensor_copy_bits_from_v1(value, &bits, 1u, &error) != 0) {
    return NULL;
  }
  return value;
}

static et_f32_tensor *owned_from(const et_f32_tensor *source, uint32_t bits) {
  et_f32_tensor_error error;
  et_f32_tensor *owned = NULL;
  if (et_f32_owned_tensor_clone_v1(source, &owned, &error) != 0 ||
      et_f32_tensor_copy_bits_from_v1(owned, &bits, 1u, &error) != 0) {
    return NULL;
  }
  return owned;
}

static void request_defaults(et_o2_tr3_c_restore_request_v1 *request) {
  memset(request, 0, sizeof(*request));
  request->struct_size = sizeof(*request);
  request->target_completed_updates = 5u;
  request->expected_receiver_completed_updates = 0u;
  request->clip_kind = ET_O2_CLIP_NONE;
  request->schedule_kind = ET_O2_SCHEDULE_LINEAR;
  request->minimum_ratio_bits = UINT32_C(0x3f000000);
  request->warmup_updates = 2u;
  request->total_updates = 10u;
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    request->entries[index].struct_size = sizeof(request->entries[index]);
    request->entries[index].learning_rate_bits = UINT32_C(0x3ca3d70a);
    request->entries[index].beta1_bits = UINT32_C(0x3f4ccccd);
    request->entries[index].beta2_bits = UINT32_C(0x3f7d70a4);
    request->entries[index].epsilon_bits = UINT32_C(0x322bcc77);
    request->entries[index].weight_decay_bits =
        index % 2u == 0u ? UINT32_C(0x3c23d70a) : 0u;
  }
}

static void fixture_init(restore_fixture *fixture) {
  et_o2_optimizer_builder *builder = NULL;
  et_o2_error_v1 o2_error;
  et_f32_tensor_error i2_error;
  memset(fixture, 0, sizeof(*fixture));
  request_defaults(&fixture->request);
  CHECK(et_o2_optimizer_builder_create_v1(
            ET_TR3_C_MODEL_DESTINATIONS, ET_O2_CLIP_NONE, 0u,
            ET_O2_SCHEDULE_CONSTANT, 0u, 0u, UINT32_C(0x3f800000),
            &builder, &o2_error) == 0);
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    et_f32_tensor *initial = tensor(UINT32_C(0x3f000000) + (uint32_t)index);
    CHECK(initial != NULL);
    CHECK(et_f32_parameter_create_v1(initial, &fixture->parameters[index],
                                     &i2_error) == 0);
    CHECK(et_f32_parameter_bind_identity_v1(
              fixture->parameters[index], fixture->parameters[index],
              &i2_error) == 0);
    CHECK(et_o2_optimizer_builder_set_v1(
              builder, index, fixture->parameters[index],
              fixture->parameters[index], UINT32_C(0x3c23d70a),
              UINT32_C(0x3f666666), UINT32_C(0x3f7fbe77),
              UINT32_C(0x322bcc77), 0u, &o2_error) == 0);
    CHECK(et_f32_tensor_destroy_v1(&initial, &i2_error) == 0);
  }
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &fixture->optimizer,
                                          &o2_error) == 0);
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    const et_f32_tensor *value =
        et_f32_parameter_canonical_owner_v1(fixture->parameters[index]);
    fixture->prefix_sources[index] =
        owned_from(value, UINT32_C(0x3f400000) + (uint32_t)index);
    fixture->request.entries[index].exp_avg_owned = owned_from(
        fixture->optimizer->entries[index].exp_avg,
        UINT32_C(0xbf000000) + (uint32_t)(index << 8u));
    fixture->request.entries[index].exp_avg_sq_owned = owned_from(
        fixture->optimizer->entries[index].exp_avg_sq,
        UINT32_C(0x3e800000) + (uint32_t)(index << 8u));
    CHECK(fixture->prefix_sources[index] != NULL);
    CHECK(fixture->request.entries[index].exp_avg_owned != NULL);
    CHECK(fixture->request.entries[index].exp_avg_sq_owned != NULL);
  }
}

static void release_prefix_sources(restore_fixture *fixture) {
  et_f32_tensor_error error;
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    if (fixture->prefix_sources[index] != NULL) {
      CHECK(et_f32_owned_tensor_release_v1(fixture->prefix_sources[index],
                                           &error) == 0);
      fixture->prefix_sources[index] = NULL;
    }
  }
}

static void release_request_stages(restore_fixture *fixture) {
  et_f32_tensor_error error;
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    if (fixture->request.entries[index].exp_avg_owned != NULL) {
      CHECK(et_f32_owned_tensor_release_v1(
                (et_f32_tensor *)fixture->request.entries[index].exp_avg_owned,
                &error) == 0);
      fixture->request.entries[index].exp_avg_owned = NULL;
    }
    if (fixture->request.entries[index].exp_avg_sq_owned != NULL) {
      CHECK(et_f32_owned_tensor_release_v1(
                (et_f32_tensor *)
                    fixture->request.entries[index].exp_avg_sq_owned,
                &error) == 0);
      fixture->request.entries[index].exp_avg_sq_owned = NULL;
    }
  }
}

static void forget_released_stages(restore_fixture *fixture) {
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    fixture->request.entries[index].exp_avg_owned = NULL;
    fixture->request.entries[index].exp_avg_sq_owned = NULL;
  }
}

static et_o2_tr3_c_restore_plan *prepare(restore_fixture *fixture) {
  et_o2_error_v1 error;
  et_o2_tr3_c_restore_plan *plan = NULL;
  CHECK(et_o2_tr3_c_restore_prepare_v1(
            fixture->optimizer, &fixture->request, &plan, &error) == 0);
  CHECK(plan != NULL);
  return plan;
}

static void *prefix_builder(restore_fixture *fixture,
                            et_o2_tr3_c_restore_plan *plan) {
  void *builder = NULL;
  CHECK(et_tr3_c_i2_copy_builder_create_restore42_v1(plan, &builder) == 0);
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    CHECK(et_tr3_c_i2_copy_builder_append_parameter_v1(
              builder, plan, fixture->parameters[index],
              fixture->parameters[index], fixture->prefix_sources[index]) ==
          0);
  }
  return builder;
}

static uint32_t tensor_bits(const et_f32_tensor *tensor_value) {
  et_f32_tensor_error error;
  uint32_t bits = UINT32_MAX;
  CHECK(et_f32_tensor_copy_bits_to_v1(tensor_value, &bits, 1u, &error) == 0);
  return bits;
}

static void test_prepare_negatives(void) {
  restore_fixture fixture;
  et_o2_error_v1 error;
  et_o2_tr3_c_restore_plan *plan = NULL;
  et_o2_tr3_c_restore_request_v1 bad;
  fixture_init(&fixture);

  expect_error(et_o2_tr3_c_restore_prepare_v1(
                   NULL, &fixture.request, &plan, &error),
               &error, ET_O2_STATUS_INVALID_ARGUMENT,
               ET_O2_CODE_NULL_ARGUMENT);
  plan = (et_o2_tr3_c_restore_plan *)(uintptr_t)1u;
  expect_error(et_o2_tr3_c_restore_prepare_v1(
                   fixture.optimizer, &fixture.request, &plan, &error),
               &error, ET_O2_STATUS_INVALID_ARGUMENT,
               ET_O2_CODE_OWNER_CONFLICT);
  plan = NULL;
  bad = fixture.request;
  bad.struct_size--;
  expect_error(et_o2_tr3_c_restore_prepare_v1(fixture.optimizer, &bad, &plan,
                                               &error),
               &error, ET_O2_STATUS_VERSION_MISMATCH,
               ET_O2_CODE_INVALID_OPTION);
  bad = fixture.request;
  bad.entries[3].reserved = 1u;
  expect_error(et_o2_tr3_c_restore_prepare_v1(fixture.optimizer, &bad, &plan,
                                               &error),
               &error, ET_O2_STATUS_VERSION_MISMATCH,
               ET_O2_CODE_INVALID_OPTION);
  bad = fixture.request;
  bad.expected_receiver_completed_updates = (uint64_t)INT64_MAX + 1u;
  expect_error(et_o2_tr3_c_restore_prepare_v1(fixture.optimizer, &bad, &plan,
                                               &error),
               &error, ET_O2_STATUS_INVALID_ARGUMENT,
               ET_O2_CODE_COUNTER_OVERFLOW);
  bad = fixture.request;
  bad.target_completed_updates = (uint64_t)INT64_MAX + 1u;
  expect_error(et_o2_tr3_c_restore_prepare_v1(fixture.optimizer, &bad, &plan,
                                               &error),
               &error, ET_O2_STATUS_CORRUPT_DATA,
               ET_O2_CODE_COUNTER_OVERFLOW);
  bad = fixture.request;
  bad.entries[4].beta1_bits = UINT32_C(0x7fc00000);
  expect_error(et_o2_tr3_c_restore_prepare_v1(fixture.optimizer, &bad, &plan,
                                               &error),
               &error, ET_O2_STATUS_CORRUPT_DATA,
               ET_O2_CODE_INVALID_OPTION);

  bad = fixture.request;
  bad.entries[2].exp_avg_owned = bad.entries[1].exp_avg_owned;
  expect_error(et_o2_tr3_c_restore_prepare_v1(fixture.optimizer, &bad, &plan,
                                               &error),
               &error, ET_O2_STATUS_INVALID_ARGUMENT,
               ET_O2_CODE_INVALID_HANDLE);
  bad = fixture.request;
  CHECK(et_f32_tensor_copy_bits_from_v1(
            (et_f32_tensor *)bad.entries[0].exp_avg_owned,
            (const uint32_t[]){UINT32_C(0x7f800000)}, 1u, NULL) == 0);
  expect_error(et_o2_tr3_c_restore_prepare_v1(fixture.optimizer, &bad, &plan,
                                               &error),
               &error, ET_O2_STATUS_CORRUPT_DATA, ET_O2_CODE_NONFINITE);
  CHECK(et_f32_tensor_copy_bits_from_v1(
            (et_f32_tensor *)bad.entries[0].exp_avg_owned,
            (const uint32_t[]){UINT32_C(0xbf000000)}, 1u, NULL) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(
            (et_f32_tensor *)bad.entries[0].exp_avg_sq_owned,
            (const uint32_t[]){UINT32_C(0xbf800000)}, 1u, NULL) == 0);
  expect_error(et_o2_tr3_c_restore_prepare_v1(fixture.optimizer, &bad, &plan,
                                               &error),
               &error, ET_O2_STATUS_CORRUPT_DATA,
               ET_O2_CODE_INVALID_MOMENT);
  CHECK(et_f32_tensor_copy_bits_from_v1(
            (et_f32_tensor *)bad.entries[0].exp_avg_sq_owned,
            (const uint32_t[]){UINT32_C(0x3e800000)}, 1u, NULL) == 0);

  et_o2_tr3_c_restore_request_v1 snapshot = fixture.request;
  CHECK(et_o2_tr3_c_restore_prepare_v1(
            fixture.optimizer, &fixture.request,
            (et_o2_tr3_c_restore_plan **)(void *)&fixture.request.entries[0]
                .exp_avg_owned,
            &error) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(&snapshot, &fixture.request, sizeof(snapshot)) == 0);
  CHECK(et_o2_tr3_c_restore_prepare_v1(
            fixture.optimizer, &fixture.request, &plan,
            (et_o2_error_v1 *)(void *)&fixture.request) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(&snapshot, &fixture.request, sizeof(snapshot)) == 0);

  et_o2_test_fail_alloc_after_v1(0u);
  expect_error(et_o2_tr3_c_restore_prepare_v1(
                   fixture.optimizer, &fixture.request, &plan, &error),
               &error, ET_O2_STATUS_INTERNAL,
               ET_O2_CODE_ALLOCATION_FAILED);
  CHECK(plan == NULL && fixture.optimizer->busy == 0u &&
        fixture.optimizer->active_operation == NULL);
  et_o2_test_reset_failpoints_v1();

  plan = prepare(&fixture);
  et_o2_trainer_step_clear_plan *step = NULL;
  CHECK(et_o2_trainer_step_clear_prepare_v1(
            fixture.optimizer, 0u, 1u, UINT32_C(0x3f800000), &step,
            &error) == ET_O2_STATUS_INVALID_STATE);
  CHECK(error.category == ET_O2_STATUS_INVALID_STATE &&
        error.code == ET_O2_CODE_ACTIVE_BORROW);
  CHECK(et_o2_tr3_c_restore_abort_v1(plan, &error) == 0);
  forget_released_stages(&fixture);
  release_prefix_sources(&fixture);
}

static void test_append_abort_and_drift(void) {
  restore_fixture fixture;
  et_o2_error_v1 error;
  fixture_init(&fixture);
  et_o2_tr3_c_restore_plan *plan = prepare(&fixture);
  et_o2_tr3_c_restore_plan copied = *plan;
  const size_t allocations = et_o2_successful_allocations;
  void *builder = prefix_builder(&fixture, plan);

  expect_error(et_o2_tr3_c_restore_check_v1(&copied, &error), &error,
               ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_INVALID_HANDLE);
  expect_error(et_o2_tr3_c_restore_check_v1(
                   (et_o2_tr3_c_restore_plan *)(void *)fixture.optimizer,
                   &error),
               &error, ET_O2_STATUS_INVALID_ARGUMENT,
               ET_O2_CODE_INVALID_HANDLE);

  expect_error(et_o2_tr3_c_restore_append_v1(plan, builder, 13u, &error),
               &error, ET_O2_STATUS_INVALID_ARGUMENT,
               ET_O2_CODE_INVALID_OPTION);
  CHECK(et_o2_tr3_c_restore_append_v1(plan, builder, 14u, &error) == 0);
  CHECK(et_o2_tr3_c_restore_check_v1(plan, &error) == 0);
  CHECK(et_o2_successful_allocations == allocations);
  expect_error(et_o2_tr3_c_restore_append_v1(plan, builder, 14u, &error),
               &error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED);
  expect_error(et_o2_tr3_c_restore_abort_v1(plan, &error), &error,
               ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW);

  fixture.optimizer->completed_updates = 1u;
  expect_error(et_o2_tr3_c_restore_check_v1(plan, &error), &error,
               ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_COUNTER_MISMATCH);
  fixture.optimizer->completed_updates = 0u;
  fixture.optimizer->config.clip_kind = ET_O2_CLIP_GLOBAL_L2;
  expect_error(et_o2_tr3_c_restore_check_v1(plan, &error), &error,
               ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_OWNER_CONFLICT);
  fixture.optimizer->config = plan->baseline_config;

  CHECK(et_tr3_c_i2_copy_builder_prepare_restore42_v1(builder, plan) == 0);
  expect_error(et_o2_tr3_c_restore_check_v1(plan, &error), &error,
               ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW);
  CHECK(et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(builder, plan) ==
        0);
  CHECK(et_o2_tr3_c_restore_abort_v1(plan, &error) == 0);
  CHECK(et_o2_successful_allocations == allocations);
  expect_error(et_o2_tr3_c_restore_check_v1(plan, &error), &error,
               ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED);
  CHECK(plan->lifecycle == ET_O2_TR3_C_RESTORE_ABORTED &&
        plan->owner == NULL && plan->i2_builder == NULL);
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    CHECK(plan->stages[index] == NULL &&
          plan->moment_destinations[index] == NULL);
  }
  forget_released_stages(&fixture);
  release_prefix_sources(&fixture);
}

static void test_update_clear_owns_first(void) {
  restore_fixture fixture;
  et_o2_error_v1 error;
  et_o2_trainer_step_clear_plan *step = NULL;
  et_o2_tr3_c_restore_plan *restore = NULL;
  fixture_init(&fixture);
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    const uint32_t gradient = UINT32_C(0x3f000000);
    et_f32_parameter_test_set_gradient_bits_v1(fixture.parameters[index],
                                                &gradient, 1u);
    et_f32_parameter_test_set_metadata_v1(
        fixture.parameters[index], ET_F32_GRADIENT_PRESENT, 1u,
        UINT32_C(0x3f800000));
  }
  CHECK(et_o2_trainer_step_clear_prepare_v1(
            fixture.optimizer, 0u, 1u, UINT32_C(0x3f800000), &step,
            &error) == 0);
  expect_error(et_o2_tr3_c_restore_prepare_v1(
                   fixture.optimizer, &fixture.request, &restore, &error),
               &error, ET_O2_STATUS_INVALID_STATE,
               ET_O2_CODE_ACTIVE_BORROW);
  CHECK(restore == NULL);
  CHECK(et_o2_trainer_step_clear_abort_v1(step, &error) == 0);
  release_request_stages(&fixture);
  release_prefix_sources(&fixture);
}

static void test_commit(void) {
  restore_fixture fixture;
  et_o2_error_v1 error;
  fixture_init(&fixture);
  et_o2_config baseline = fixture.optimizer->config;
  et_o2_tr3_c_restore_plan *plan = prepare(&fixture);
  const size_t allocations = et_o2_successful_allocations;
  void *builder = prefix_builder(&fixture, plan);
  CHECK(et_o2_tr3_c_restore_append_v1(plan, builder, 14u, &error) == 0);
  CHECK(et_o2_tr3_c_restore_check_v1(plan, &error) == 0);
  CHECK(memcmp(&fixture.optimizer->config, &baseline, sizeof(baseline)) == 0);
  CHECK(fixture.optimizer->completed_updates == 0u);
  CHECK(et_tr3_c_i2_copy_builder_prepare_restore42_v1(builder, plan) == 0);
  CHECK(et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(builder, plan) ==
        0);
  expect_error(et_o2_tr3_c_restore_abort_v1(plan, &error), &error,
               ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED);
  et_o2_tr3_c_restore_commit_native_v1(plan);
  CHECK(plan->lifecycle == ET_O2_TR3_C_RESTORE_COMMITTING);
  CHECK(fixture.optimizer->completed_updates == 5u);
  CHECK(fixture.optimizer->config.schedule_kind == ET_O2_SCHEDULE_LINEAR);
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    CHECK(fixture.optimizer->entries[index].options.learning_rate_bits ==
          fixture.request.entries[index].learning_rate_bits);
    CHECK(tensor_bits(fixture.optimizer->entries[index].exp_avg) ==
          tensor_bits(fixture.request.entries[index].exp_avg_owned));
    CHECK(tensor_bits(fixture.optimizer->entries[index].exp_avg_sq) ==
          tensor_bits(fixture.request.entries[index].exp_avg_sq_owned));
  }
  et_o2_tr3_c_restore_finalize_v1(plan);
  CHECK(et_o2_successful_allocations == allocations);
  CHECK(plan->lifecycle == ET_O2_TR3_C_RESTORE_COMMITTED &&
        plan->owner == NULL && plan->i2_builder == NULL);
  CHECK(fixture.optimizer->busy == 0u &&
        fixture.optimizer->active_operation == NULL);
  forget_released_stages(&fixture);
  release_prefix_sources(&fixture);
}

static void test_rewind_commit(void) {
  restore_fixture fixture;
  et_o2_error_v1 error;
  fixture_init(&fixture);
  fixture.optimizer->completed_updates = 7u;
  fixture.request.expected_receiver_completed_updates = 7u;
  fixture.request.target_completed_updates = 2u;
  et_o2_tr3_c_restore_plan *plan = prepare(&fixture);
  void *builder = prefix_builder(&fixture, plan);
  CHECK(et_o2_tr3_c_restore_append_v1(plan, builder, 14u, &error) == 0);
  CHECK(et_o2_tr3_c_restore_check_v1(plan, &error) == 0);
  CHECK(fixture.optimizer->completed_updates == 7u);
  CHECK(et_tr3_c_i2_copy_builder_prepare_restore42_v1(builder, plan) == 0);
  CHECK(et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(builder, plan) ==
        0);
  et_o2_tr3_c_restore_commit_native_v1(plan);
  CHECK(fixture.optimizer->completed_updates == 2u);
  et_o2_tr3_c_restore_finalize_v1(plan);
  CHECK(fixture.optimizer->active_operation == NULL);
  forget_released_stages(&fixture);
  release_prefix_sources(&fixture);
}

static void expect_exit_134(void (*action)(void)) {
  int status = 0;
  const pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    action();
    _Exit(90);
  }
  if (child < 0) {
    return;
  }
  CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 134);
}

static void fail_commit_without_terminal_builder(void) {
  restore_fixture fixture;
  et_o2_error_v1 error;
  fixture_init(&fixture);
  et_o2_tr3_c_restore_plan *plan = prepare(&fixture);
  void *builder = prefix_builder(&fixture, plan);
  if (et_o2_tr3_c_restore_append_v1(plan, builder, 14u, &error) != 0) {
    _Exit(91);
  }
  et_o2_tr3_c_restore_commit_native_v1(plan);
}

static void fail_commit_after_builder_abort(void) {
  restore_fixture fixture;
  et_o2_error_v1 error;
  fixture_init(&fixture);
  et_o2_tr3_c_restore_plan *plan = prepare(&fixture);
  void *builder = prefix_builder(&fixture, plan);
  if (et_o2_tr3_c_restore_append_v1(plan, builder, 14u, &error) != 0 ||
      et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(builder, plan) !=
          0) {
    _Exit(91);
  }
  et_o2_tr3_c_restore_commit_native_v1(plan);
}

static void fail_abort_stage_release(void) {
  restore_fixture fixture;
  et_o2_error_v1 error;
  fixture_init(&fixture);
  et_o2_tr3_c_restore_plan *plan = prepare(&fixture);
  et_tr3_c_i2_test_fail_owned_release_v1(1);
  (void)et_o2_tr3_c_restore_abort_v1(plan, &error);
}

static void test_fail_stop_and_counts(void) {
  et_o2_tr3_c_restore_test_counts_v1 before = {.struct_size = sizeof(before)};
  et_o2_tr3_c_restore_test_counts_v1 after = {.struct_size = sizeof(after)};
  expect_exit_134(fail_commit_without_terminal_builder);
  expect_exit_134(fail_commit_after_builder_abort);
  expect_exit_134(fail_abort_stage_release);
  et_o2_tr3_c_restore_test_counts_snapshot_v1(&before);
  for (size_t cycle = 0u; cycle < 128u; ++cycle) {
    restore_fixture fixture;
    et_o2_error_v1 error;
    fixture_init(&fixture);
    et_o2_tr3_c_restore_plan *plan = prepare(&fixture);
    CHECK(et_o2_tr3_c_restore_abort_v1(plan, &error) == 0);
    forget_released_stages(&fixture);
    release_prefix_sources(&fixture);
  }
  et_o2_tr3_c_restore_test_counts_snapshot_v1(&after);
  CHECK(after.aborted_plans - before.aborted_plans == 128u);
  CHECK(after.retained_control_bytes - before.retained_control_bytes ==
        128u * sizeof(et_o2_tr3_c_restore_plan));
  (void)printf("TR3-C O2 retained-control slope: %zu bytes/restore\n",
               sizeof(et_o2_tr3_c_restore_plan));
}

int main(void) {
  test_prepare_negatives();
  test_append_abort_and_drift();
  test_update_clear_owns_first();
  test_commit();
  test_rewind_commit();
  test_fail_stop_and_counts();
  if (failures != 0) {
    (void)fprintf(stderr, "TR3-C O2 restore FAIL: %d/%d checks\n", failures,
                  checks);
    return 1;
  }
  (void)printf("TR3-C O2 restore PASS: %d checks\n", checks);
  return 0;
}
