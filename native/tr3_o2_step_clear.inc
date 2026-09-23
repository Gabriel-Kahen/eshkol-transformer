#include "tr3_o2_step_clear_internal.h"

#define ET_O2_TR3_STEP_CLEAR_MAGIC UINT64_C(0x45545452334f5343)

enum {
  ET_O2_TR3_STEP_CLEAR_PREPARED = 1,
  ET_O2_TR3_STEP_CLEAR_COMMITTING = 2,
  ET_O2_TR3_STEP_CLEAR_COMMITTED = 3,
  ET_O2_TR3_STEP_CLEAR_ABORTING = 4,
  ET_O2_TR3_STEP_CLEAR_ABORTED = 5
};

enum {
  ET_O2_TR3_TERMINAL_NONE = 0,
  ET_O2_TR3_TERMINAL_COPY_COMMIT = 1,
  ET_O2_TR3_TERMINAL_RESET_COMMIT = 2,
  ET_O2_TR3_TERMINAL_RESET_RELEASE = 3,
  ET_O2_TR3_TERMINAL_COPY_RELEASE = 4,
  ET_O2_TR3_TERMINAL_STAGE_DESTROY = 5
};

#ifdef ET_O2_TESTING
_Static_assert((int)ET_O2_TR3_TEST_TERMINAL_NONE ==
                   (int)ET_O2_TR3_TERMINAL_NONE,
               "TR3-O terminal test IDs changed");
_Static_assert(
    (int)ET_O2_TR3_TEST_TERMINAL_STAGE_DESTROY ==
        (int)ET_O2_TR3_TERMINAL_STAGE_DESTROY,
    "TR3-O terminal test IDs changed");
#endif

struct et_o2_trainer_step_clear_plan {
  uint64_t magic;
  et_o2_trainer_step_clear_plan *registry_next;
  et_o2_optimizer *owner;
  size_t count;
  uint64_t expected_completed_updates;
  uint64_t next_completed_updates;
  uint32_t lifecycle;
  et_f32_tensor **stages;
  et_f32_tensor_copy_assignment_v1 *assignments;
  et_f32_parameter **parameters;
  et_f32_tensor_copy_plan *copy_plan;
  et_f32_gradient_reset_plan *reset_plan;
};

static et_o2_trainer_step_clear_plan *et_o2_tr3_step_clear_plans;

#ifdef ET_O2_TESTING
static size_t et_o2_tr3_allocation_limit = SIZE_MAX;
static size_t et_o2_tr3_successful_allocations;
static int et_o2_tr3_fail_publication;
static uint32_t et_o2_tr3_terminal_site;
static size_t et_o2_tr3_terminal_stage_index = SIZE_MAX;
#endif

static _Noreturn void et_o2_tr3_step_clear_fail_stop(void) { _Exit(134); }

static void *et_o2_tr3_calloc(size_t count, size_t size) {
  void *allocation;
  if (size != 0u && count > SIZE_MAX / size) {
    return NULL;
  }
#ifdef ET_O2_TESTING
  if (et_o2_tr3_successful_allocations >= et_o2_tr3_allocation_limit) {
    return NULL;
  }
#endif
  allocation = calloc(count, size);
#ifdef ET_O2_TESTING
  if (allocation != NULL) {
    et_o2_tr3_successful_allocations++;
  }
#endif
  return allocation;
}

static et_o2_trainer_step_clear_plan *
et_o2_tr3_find_step_clear_plan(const void *candidate) {
  et_o2_trainer_step_clear_plan *cursor = et_o2_tr3_step_clear_plans;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor->magic == ET_O2_TR3_STEP_CLEAR_MAGIC ? cursor : NULL;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static int et_o2_tr3_terminal_fail(uint32_t site, size_t stage_index) {
#ifdef ET_O2_TESTING
  return site != ET_O2_TR3_TERMINAL_NONE &&
         et_o2_tr3_terminal_site == site &&
         (site != ET_O2_TR3_TERMINAL_STAGE_DESTROY ||
          et_o2_tr3_terminal_stage_index == stage_index);
#else
  (void)site;
  (void)stage_index;
  return 0;
#endif
}

static void et_o2_tr3_release_reset(et_o2_trainer_step_clear_plan *plan,
                                    uint32_t terminal_site) {
  if (plan->reset_plan != NULL) {
    if (et_o2_tr3_terminal_fail(terminal_site, SIZE_MAX) ||
        et_f32_gradient_reset_plan_release_v1(&plan->reset_plan, NULL) != 0) {
      et_o2_tr3_step_clear_fail_stop();
    }
  }
}

static void et_o2_tr3_release_copy(et_o2_trainer_step_clear_plan *plan,
                                   uint32_t terminal_site) {
  if (plan->copy_plan != NULL) {
    if (et_o2_tr3_terminal_fail(terminal_site, SIZE_MAX) ||
        et_f32_tensor_copy_plan_release_v1(&plan->copy_plan, NULL) != 0) {
      et_o2_tr3_step_clear_fail_stop();
    }
  }
}

static void et_o2_tr3_destroy_stages(et_o2_trainer_step_clear_plan *plan,
                                     uint32_t terminal_site) {
  size_t index;
  if (plan->stages == NULL) {
    return;
  }
  for (index = 0u; index < plan->count * 3u; ++index) {
    if (plan->stages[index] != NULL &&
        (et_o2_tr3_terminal_fail(terminal_site, index) ||
         et_f32_tensor_destroy_v1(&plan->stages[index], NULL) != 0)) {
      et_o2_tr3_step_clear_fail_stop();
    }
  }
}

static void et_o2_tr3_free_payload(et_o2_trainer_step_clear_plan *plan) {
  free(plan->parameters);
  free(plan->assignments);
  free(plan->stages);
  plan->parameters = NULL;
  plan->assignments = NULL;
  plan->stages = NULL;
}

static void et_o2_tr3_prepare_cleanup(et_o2_trainer_step_clear_plan *plan) {
  if (plan == NULL) {
    return;
  }
  et_o2_tr3_release_reset(plan, ET_O2_TR3_TERMINAL_NONE);
  et_o2_tr3_release_copy(plan, ET_O2_TR3_TERMINAL_NONE);
  et_o2_tr3_destroy_stages(plan, ET_O2_TR3_TERMINAL_NONE);
  et_o2_tr3_free_payload(plan);
  if (plan->owner != NULL &&
      et_o2_active_operation_release(plan->owner, plan) != 0) {
    et_o2_tr3_step_clear_fail_stop();
  }
  free(plan);
}

static int32_t et_o2_tr3_require_prepared(
    et_o2_trainer_step_clear_plan *candidate, const char *operation,
    et_o2_trainer_step_clear_plan **resolved, et_o2_error_v1 *error) {
  et_o2_trainer_step_clear_plan *plan =
      et_o2_tr3_find_step_clear_plan(candidate);
  et_o2_optimizer *owner;
  if (plan == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_HANDLE, operation,
                      "transaction plan is foreign or malformed");
  }
  if (plan->lifecycle != ET_O2_TR3_STEP_CLEAR_PREPARED) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED,
                      operation, "transaction plan is already consumed");
  }
  owner = et_o2_find_optimizer(plan->owner);
  if (owner == NULL || !et_o2_optimizer_provider_valid(owner) ||
      owner->count != plan->count || plan->count == 0u ||
      plan->stages == NULL || plan->assignments == NULL ||
      plan->parameters == NULL || plan->copy_plan == NULL ||
      plan->reset_plan == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                      ET_O2_CODE_PROVIDER_DEFECT, operation,
                      "transaction plan ownership invariant is broken");
  }
  if (!et_o2_active_operation_owns(owner, plan)) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_OWNER_CONFLICT, operation,
                      "transaction plan does not own the optimizer");
  }
  if (owner->completed_updates != plan->expected_completed_updates) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_COUNTER_MISMATCH, operation,
                      "optimizer update counter changed after prepare");
  }
  *resolved = plan;
  return ET_O2_STATUS_OK;
}

int32_t et_o2_trainer_step_clear_prepare_v1(
    et_o2_optimizer *candidate, uint64_t expected_completed_updates,
    uint64_t expected_contribution_count,
    uint32_t expected_normalization_weight_bits,
    et_o2_trainer_step_clear_plan **output, et_o2_error_v1 *error) {
  const char *operation = "optimizer-step-clear-prepare";
  et_o2_optimizer *optimizer;
  et_o2_trainer_step_clear_plan *plan = NULL;
  et_f32_tensor_error i2_error;
  uint32_t norm = 0u;
  uint32_t clip_factor = UINT32_C(0x3f800000);
  uint32_t schedule_factor = 0u;
  size_t index;
  int32_t result;

  if (output == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_NULL_ARGUMENT, operation,
                      "transaction plan output is null");
  }
  if (*output != NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_OPTION, operation,
                      "transaction plan output is not empty");
  }
  if (expected_completed_updates >= (uint64_t)INT64_MAX ||
      expected_contribution_count == 0u ||
      expected_contribution_count > (uint64_t)INT64_MAX ||
      !et_o2_bits_positive_finite(expected_normalization_weight_bits)) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_OPTION, operation,
                      "transaction expectations are out of range");
  }
  optimizer = et_o2_find_optimizer(candidate);
  if (optimizer == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_ARGUMENT,
                      ET_O2_CODE_INVALID_HANDLE, operation,
                      "optimizer is foreign, stale, or invalid");
  }
  if (!et_o2_optimizer_provider_valid(optimizer) || optimizer->count == 0u ||
      optimizer->count > ET_O2_MAX_PARAMETERS || optimizer->entries == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                      ET_O2_CODE_PROVIDER_DEFECT, operation,
                      "optimizer provider invariant is broken");
  }
  if (optimizer->busy != 0u || optimizer->active_operation != NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_ACTIVE_BORROW, operation,
                      "optimizer is already active");
  }
  if (optimizer->completed_updates >= (uint64_t)INT64_MAX) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_COUNTER_OVERFLOW, operation,
                      "completed-update counter would overflow");
  }
  if (optimizer->completed_updates != expected_completed_updates) {
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_COUNTER_MISMATCH, operation,
                      "optimizer update counter differs from expectation");
  }
  if (!et_o2_float_environment()) {
    return et_o2_fail(error, ET_O2_STATUS_DETERMINISM_UNAVAILABLE,
                      ET_O2_CODE_FLOAT_ENVIRONMENT, operation,
                      "binary32 environment is unsupported");
  }

  plan = (et_o2_trainer_step_clear_plan *)et_o2_tr3_calloc(1u, sizeof(*plan));
  if (plan == NULL) {
    return et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                      ET_O2_CODE_ALLOCATION_FAILED, operation,
                      "cannot allocate transaction plan");
  }
  plan->owner = optimizer;
  plan->count = optimizer->count;
  plan->expected_completed_updates = expected_completed_updates;
  plan->next_completed_updates = expected_completed_updates + 1u;
  if (et_o2_active_operation_acquire(optimizer, plan) != 0) {
    free(plan);
    return et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                      ET_O2_CODE_ACTIVE_BORROW, operation,
                      "optimizer operation ownership was not acquired");
  }
  plan->stages = (et_f32_tensor **)et_o2_tr3_calloc(
      plan->count * 3u, sizeof(*plan->stages));
  plan->assignments =
      (et_f32_tensor_copy_assignment_v1 *)et_o2_tr3_calloc(
          plan->count * 3u, sizeof(*plan->assignments));
  plan->parameters = (et_f32_parameter **)et_o2_tr3_calloc(
      plan->count, sizeof(*plan->parameters));
  if (plan->stages == NULL || plan->assignments == NULL ||
      plan->parameters == NULL) {
    result = et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                        ET_O2_CODE_ALLOCATION_FAILED, operation,
                        "cannot allocate transaction tables");
    goto fail;
  }

  for (index = 0u; index < plan->count; ++index) {
    et_f32_gradient_metadata_v1 metadata;
    int32_t finite = 0;
    memset(&metadata, 0, sizeof(metadata));
    metadata.struct_size = sizeof(metadata);
    memset(&i2_error, 0, sizeof(i2_error));
    if (et_f32_parameter_validate_identity_v1(
            optimizer->entries[index].parameter,
            optimizer->entries[index].p1_handle, &i2_error) != 0 ||
        et_f32_parameter_gradient_metadata_v1(
            optimizer->entries[index].parameter, &metadata, &i2_error) != 0) {
      result = et_o2_from_i2(&i2_error, error, operation,
                             ET_O2_STATUS_INVALID_STATE);
      goto fail;
    }
    if (metadata.state != ET_F32_GRADIENT_PRESENT) {
      result = et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                          ET_O2_CODE_GRADIENT_ABSENT, operation,
                          "every bound parameter requires a present gradient");
      goto fail;
    }
    if (metadata.contribution_count != expected_contribution_count ||
        metadata.normalization_weight_bits !=
            expected_normalization_weight_bits) {
      result = et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                          ET_O2_CODE_GRADIENT_METADATA, operation,
                          "gradient metadata differs from expectation");
      goto fail;
    }
    if (et_f32_parameter_gradient_finite_v1(
            optimizer->entries[index].parameter, &finite, &i2_error) != 0 ||
        finite != 1) {
      result = et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                          ET_O2_CODE_NONFINITE, operation,
                          "gradient contains a nonfinite value");
      goto fail;
    }
    plan->parameters[index] = optimizer->entries[index].parameter;
  }

  if ((result = et_o2_schedule_factor(&optimizer->config,
                                      plan->next_completed_updates,
                                      &schedule_factor, error)) != 0 ||
      (result = et_o2_global_norm(optimizer,
                                  expected_normalization_weight_bits, &norm,
                                  error)) != 0) {
    goto fail;
  }
  if (optimizer->config.clip_kind == ET_O2_CLIP_GLOBAL_L2 &&
      et_o2_float_from_bits(norm) >
          et_o2_float_from_bits(optimizer->config.clip_max_bits)) {
    clip_factor = et_o2_div(optimizer->config.clip_max_bits, norm);
    if (!et_o2_bits_nonnegative_finite(clip_factor)) {
      result = et_o2_fail(error, ET_O2_STATUS_INVALID_STATE,
                          ET_O2_CODE_NONFINITE, operation,
                          "gradient clip factor is invalid");
      goto fail;
    }
  }

  for (index = 0u; index < plan->count; ++index) {
    const et_f32_tensor *parameter_value = NULL;
    size_t base = index * 3u;
    memset(&i2_error, 0, sizeof(i2_error));
    result = et_o2_stage_entry(
        &optimizer->entries[index], plan->next_completed_updates,
        schedule_factor, expected_normalization_weight_bits, clip_factor,
        &plan->stages[base], &plan->stages[base + 1u],
        &plan->stages[base + 2u], error);
    if (result != 0 ||
        et_f32_parameter_value_tensor_v1(optimizer->entries[index].parameter,
                                         &parameter_value, &i2_error) != 0) {
      if (result == 0) {
        result = et_o2_from_i2(&i2_error, error, operation,
                               ET_O2_STATUS_INVALID_STATE);
      }
      goto fail;
    }
    plan->assignments[base].struct_size =
        ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
    plan->assignments[base].destination = (et_f32_tensor *)parameter_value;
    plan->assignments[base].source = plan->stages[base];
    plan->assignments[base + 1u].struct_size =
        ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
    plan->assignments[base + 1u].destination =
        optimizer->entries[index].exp_avg;
    plan->assignments[base + 1u].source = plan->stages[base + 1u];
    plan->assignments[base + 2u].struct_size =
        ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
    plan->assignments[base + 2u].destination =
        optimizer->entries[index].exp_avg_sq;
    plan->assignments[base + 2u].source = plan->stages[base + 2u];
  }

  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_tensor_copy_plan_prepare_v1(plan->count * 3u,
                                         plan->assignments, &plan->copy_plan,
                                         &i2_error) != 0) {
    result = et_o2_from_i2(&i2_error, error, operation,
                           ET_O2_STATUS_INVALID_STATE);
    goto fail;
  }
  memset(&i2_error, 0, sizeof(i2_error));
  if (et_f32_gradient_reset_plan_prepare_v1(
          plan->count, plan->parameters, &plan->reset_plan, &i2_error) != 0) {
    result = et_o2_from_i2(&i2_error, error, operation,
                           ET_O2_STATUS_INVALID_STATE);
    goto fail;
  }
#ifdef ET_O2_TESTING
  if (et_o2_tr3_fail_publication) {
    result = et_o2_fail(error, ET_O2_STATUS_INTERNAL,
                        ET_O2_CODE_ALLOCATION_FAILED, operation,
                        "transaction publication was rejected");
    goto fail;
  }
#endif
  plan->magic = ET_O2_TR3_STEP_CLEAR_MAGIC;
  plan->lifecycle = ET_O2_TR3_STEP_CLEAR_PREPARED;
  plan->registry_next = et_o2_tr3_step_clear_plans;
  et_o2_tr3_step_clear_plans = plan;
  *output = plan;
  return et_o2_success(error);

fail:
  et_o2_tr3_prepare_cleanup(plan);
  return result;
}

int32_t et_o2_trainer_step_clear_commit_v1(
    et_o2_trainer_step_clear_plan *candidate, et_o2_error_v1 *error) {
  et_o2_trainer_step_clear_plan *plan = NULL;
  int32_t result = et_o2_tr3_require_prepared(
      candidate, "optimizer-step-clear-commit", &plan, error);
  if (result != 0) {
    return result;
  }
  (void)et_o2_success(error);
  plan->lifecycle = ET_O2_TR3_STEP_CLEAR_COMMITTING;
  if (et_o2_tr3_terminal_fail(ET_O2_TR3_TERMINAL_COPY_COMMIT,
                              SIZE_MAX) ||
      et_f32_tensor_copy_plan_commit_v1(plan->copy_plan, NULL) != 0) {
    et_o2_tr3_step_clear_fail_stop();
  }
  if (et_o2_tr3_terminal_fail(ET_O2_TR3_TERMINAL_RESET_COMMIT,
                              SIZE_MAX) ||
      et_f32_gradient_reset_plan_commit_v1(plan->reset_plan, NULL) != 0) {
    et_o2_tr3_step_clear_fail_stop();
  }
  plan->owner->completed_updates = plan->next_completed_updates;
  et_o2_tr3_release_reset(plan, ET_O2_TR3_TERMINAL_RESET_RELEASE);
  et_o2_tr3_release_copy(plan, ET_O2_TR3_TERMINAL_COPY_RELEASE);
  et_o2_tr3_destroy_stages(plan, ET_O2_TR3_TERMINAL_STAGE_DESTROY);
  et_o2_tr3_free_payload(plan);
  if (et_o2_active_operation_release(plan->owner, plan) != 0) {
    et_o2_tr3_step_clear_fail_stop();
  }
  plan->owner = NULL;
  plan->lifecycle = ET_O2_TR3_STEP_CLEAR_COMMITTED;
  return ET_O2_STATUS_OK;
}

int32_t et_o2_trainer_step_clear_abort_v1(
    et_o2_trainer_step_clear_plan *candidate, et_o2_error_v1 *error) {
  et_o2_trainer_step_clear_plan *plan = NULL;
  int32_t result = et_o2_tr3_require_prepared(
      candidate, "optimizer-step-clear-abort", &plan, error);
  if (result != 0) {
    return result;
  }
  (void)et_o2_success(error);
  plan->lifecycle = ET_O2_TR3_STEP_CLEAR_ABORTING;
  et_o2_tr3_release_reset(plan, ET_O2_TR3_TERMINAL_RESET_RELEASE);
  et_o2_tr3_release_copy(plan, ET_O2_TR3_TERMINAL_COPY_RELEASE);
  et_o2_tr3_destroy_stages(plan, ET_O2_TR3_TERMINAL_STAGE_DESTROY);
  et_o2_tr3_free_payload(plan);
  if (et_o2_active_operation_release(plan->owner, plan) != 0) {
    et_o2_tr3_step_clear_fail_stop();
  }
  plan->owner = NULL;
  plan->lifecycle = ET_O2_TR3_STEP_CLEAR_ABORTED;
  return ET_O2_STATUS_OK;
}

#ifdef ET_O2_TESTING
void et_o2_trainer_step_clear_test_counts_snapshot_v1(
    et_o2_trainer_step_clear_test_counts_v1 *counts) {
  et_o2_trainer_step_clear_test_counts_v1 snapshot = {
      sizeof(snapshot), 0u, 0u, 0u, 0u, 0u};
  et_o2_trainer_step_clear_plan *plan = et_o2_tr3_step_clear_plans;
  while (plan != NULL) {
    if (plan->lifecycle == ET_O2_TR3_STEP_CLEAR_PREPARED ||
        plan->lifecycle == ET_O2_TR3_STEP_CLEAR_COMMITTING ||
        plan->lifecycle == ET_O2_TR3_STEP_CLEAR_ABORTING) {
      size_t index;
      snapshot.live_plans++;
      if (plan->stages != NULL) {
        for (index = 0u; index < plan->count * 3u; ++index) {
          if (plan->stages[index] != NULL) {
            snapshot.live_stages++;
          }
        }
      }
    } else if (plan->lifecycle == ET_O2_TR3_STEP_CLEAR_COMMITTED) {
      snapshot.committed_plans++;
      snapshot.retained_control_bytes += sizeof(*plan);
    } else if (plan->lifecycle == ET_O2_TR3_STEP_CLEAR_ABORTED) {
      snapshot.aborted_plans++;
      snapshot.retained_control_bytes += sizeof(*plan);
    }
    plan = plan->registry_next;
  }
  if (counts != NULL && counts->struct_size == sizeof(*counts)) {
    *counts = snapshot;
  }
}

void et_o2_trainer_step_clear_test_fail_alloc_after_v1(size_t allowed) {
  et_o2_tr3_allocation_limit = allowed;
  et_o2_tr3_successful_allocations = 0u;
}

void et_o2_trainer_step_clear_test_fail_publication_v1(int enabled) {
  et_o2_tr3_fail_publication = enabled != 0;
}

void et_o2_trainer_step_clear_test_fail_terminal_v1(uint32_t site,
                                                     size_t stage_index) {
  et_o2_tr3_terminal_site = site;
  et_o2_tr3_terminal_stage_index = stage_index;
}

void et_o2_trainer_step_clear_test_reset_failpoints_v1(void) {
  et_o2_tr3_allocation_limit = SIZE_MAX;
  et_o2_tr3_successful_allocations = 0u;
  et_o2_tr3_fail_publication = 0;
  et_o2_tr3_terminal_site = ET_O2_TR3_TERMINAL_NONE;
  et_o2_tr3_terminal_stage_index = SIZE_MAX;
}

const et_f32_tensor *et_o2_trainer_step_clear_test_moment_storage_v1(
    const et_o2_optimizer *candidate, size_t index, uint32_t moment_kind) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer(candidate);
  if (optimizer == NULL || index >= optimizer->count ||
      moment_kind > ET_O2_MOMENT_EXP_AVG_SQ) {
    return NULL;
  }
  return moment_kind == ET_O2_MOMENT_EXP_AVG
             ? optimizer->entries[index].exp_avg
             : optimizer->entries[index].exp_avg_sq;
}

const et_f32_tensor *et_o2_trainer_step_clear_test_stage_v1(
    const et_o2_trainer_step_clear_plan *candidate, size_t stage_index) {
  et_o2_trainer_step_clear_plan *plan =
      et_o2_tr3_find_step_clear_plan(candidate);
  if (plan == NULL || plan->lifecycle != ET_O2_TR3_STEP_CLEAR_PREPARED ||
      plan->stages == NULL || stage_index >= plan->count * 3u) {
    return NULL;
  }
  return plan->stages[stage_index];
}
#endif
