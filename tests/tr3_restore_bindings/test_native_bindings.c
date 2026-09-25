#define ET_TR3_C_RESTORE_BINDINGS 1
#define main et_tr3_c_accepted_o2_test_main
#include "../o2/test_tr3_c_o2_restore.c"
#undef main

#include "../../native/tr3_c_restore_bindings.h"

typedef struct request_bytes {
  int64_t length;
  unsigned char payload[ET_TR3_C_RESTORE_REQUEST_BYTES];
} request_bytes;

static void binding_request(request_bytes *bytes,
                            const restore_fixture *fixture) {
  memset(bytes, 0, sizeof(*bytes));
  bytes->length = ET_TR3_C_RESTORE_REQUEST_BYTES;
  CHECK(et_tr3_c_private_restore_request_initialize_v1(
            bytes, (int64_t)fixture->request.target_completed_updates,
            (int64_t)fixture->request.expected_receiver_completed_updates,
            (int64_t)fixture->request.clip_kind,
            (int64_t)fixture->request.clip_max_bits,
            (int64_t)fixture->request.schedule_kind,
            (int64_t)fixture->request.minimum_ratio_bits,
            (int64_t)fixture->request.warmup_updates,
            (int64_t)fixture->request.total_updates) == 0);
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    const et_o2_tr3_c_restore_entry_v1 *entry =
        &fixture->request.entries[index];
    CHECK(et_tr3_c_private_restore_request_append14_v1(
              bytes, (int64_t)entry->learning_rate_bits,
              (int64_t)entry->beta1_bits, (int64_t)entry->beta2_bits,
              (int64_t)entry->epsilon_bits,
              (int64_t)entry->weight_decay_bits,
              (void *)(uintptr_t)entry->exp_avg_owned,
              (void *)(uintptr_t)entry->exp_avg_sq_owned) == 0);
  }
  CHECK(memcmp(bytes->payload, &fixture->request,
               sizeof(fixture->request)) == 0);
}

static void binding_prefix(restore_fixture *fixture, void *plan,
                           void *builder) {
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    CHECK(et_tr3_c_private_i2_restore_append_parameter_v1(
              builder, plan, fixture->parameters[index],
              fixture->parameters[index], fixture->prefix_sources[index]) ==
          0);
  }
}

static void test_binding_abort(void) {
  restore_fixture fixture;
  request_bytes bytes;
  request_bytes malformed;
  void *plan;
  void *builder;

  fixture_init(&fixture);
  binding_request(&bytes, &fixture);
  CHECK(et_tr3_c_private_restore_request_append14_v1(
            &bytes, 0, 0, 0, 0, 0,
            (void *)(uintptr_t)fixture.request.entries[0].exp_avg_owned,
            (void *)(uintptr_t)fixture.request.entries[0].exp_avg_sq_owned) ==
        -1);
  CHECK(et_tr3_c_private_restore_last_category_v1() ==
        ET_O2_STATUS_INVALID_STATE);

  malformed = bytes;
  malformed.length--;
  CHECK(et_tr3_c_private_o2_restore_prepare_v1(fixture.optimizer,
                                                &malformed) == NULL);
  CHECK(et_tr3_c_private_restore_last_category_v1() ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(fixture.optimizer->active_operation == NULL);

  plan = et_tr3_c_private_o2_restore_prepare_v1(fixture.optimizer, &bytes);
  CHECK(plan != NULL);
  CHECK(fixture.optimizer->active_operation == plan);
  builder = et_tr3_c_private_i2_restore_create_v1(plan);
  CHECK(builder != NULL);
  CHECK(et_tr3_c_private_i2_restore_append_parameter_v1(
            builder, fixture.optimizer, fixture.parameters[0],
            fixture.parameters[0], fixture.prefix_sources[0]) == -1);
  CHECK(et_tr3_c_private_restore_last_category_v1() ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  binding_prefix(&fixture, plan, builder);
  CHECK(et_tr3_c_private_o2_restore_append_v1(plan, builder) == 0);
  CHECK(et_tr3_c_private_o2_restore_check_v1(plan) == 0);
  CHECK(et_tr3_c_private_i2_restore_prepare_v1(builder, plan) == 0);
  CHECK(et_tr3_c_private_i2_restore_abort_checked_v1(builder, plan) == 0);
  CHECK(et_tr3_c_private_o2_restore_abort_v1(plan) == 0);
  CHECK(fixture.optimizer->active_operation == NULL);
  forget_released_stages(&fixture);
  release_prefix_sources(&fixture);
}

static void test_binding_commit(void) {
  restore_fixture fixture;
  request_bytes bytes;
  void *plan;
  void *builder;

  fixture_init(&fixture);
  binding_request(&bytes, &fixture);
  plan = et_tr3_c_private_o2_restore_prepare_v1(fixture.optimizer, &bytes);
  CHECK(plan != NULL);
  builder = et_tr3_c_private_i2_restore_create_v1(plan);
  CHECK(builder != NULL);
  binding_prefix(&fixture, plan, builder);
  CHECK(et_tr3_c_private_o2_restore_append_v1(plan, builder) == 0);
  CHECK(et_tr3_c_private_o2_restore_check_v1(plan) == 0);
  CHECK(et_tr3_c_private_i2_restore_prepare_v1(builder, plan) == 0);
  CHECK(et_tr3_c_private_i2_restore_commit_checked_v1(builder, plan) == 0);
  et_tr3_c_private_o2_restore_commit_native_v1(plan);
  CHECK(fixture.optimizer->completed_updates ==
        fixture.request.target_completed_updates);
  et_tr3_c_private_o2_restore_finalize_v1(plan);
  CHECK(fixture.optimizer->active_operation == NULL);
  forget_released_stages(&fixture);
  release_prefix_sources(&fixture);
}

int main(void) {
  test_binding_abort();
  test_binding_commit();
  if (failures != 0) {
    (void)fprintf(stderr, "TR3-C restore bindings FAIL: %d/%d checks\n",
                  failures, checks);
    return 1;
  }
  (void)printf("TR3-C restore bindings PASS: %d checks\n", checks);
  return 0;
}
