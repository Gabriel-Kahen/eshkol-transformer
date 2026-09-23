#include "tr3_c_o2_restore_internal.h"

#define ET_O2_TR3_C_RESTORE_MAGIC UINT64_C(0x4554545233434f52)
#define ET_O2_TR3_C_RESTORE_OPERATION "trainer-load-state!"

enum {
  ET_O2_TR3_C_RESTORE_PREPARED = 1,
  ET_O2_TR3_C_RESTORE_COMMITTING = 2,
  ET_O2_TR3_C_RESTORE_COMMITTED = 3,
  ET_O2_TR3_C_RESTORE_ABORTING = 4,
  ET_O2_TR3_C_RESTORE_ABORTED = 5
};

struct et_o2_tr3_c_restore_plan {
  uint64_t magic;
  et_o2_tr3_c_restore_plan *registry_next;
  et_o2_optimizer *owner;
  uint32_t lifecycle;
  uint64_t expected_receiver_completed_updates;
  uint64_t target_completed_updates;
  et_o2_config baseline_config;
  et_o2_config target_config;
  et_o2_options baseline_options[ET_TR3_C_MODEL_DESTINATIONS];
  et_o2_options target_options[ET_TR3_C_MODEL_DESTINATIONS];
  et_f32_parameter *parameters[ET_TR3_C_MODEL_DESTINATIONS];
  const void *p1_handles[ET_TR3_C_MODEL_DESTINATIONS];
  const et_f32_tensor *value_tensors[ET_TR3_C_MODEL_DESTINATIONS];
  et_f32_tensor *moment_destinations[ET_TR3_C_O2_MOMENT_ASSIGNMENTS];
  const et_f32_tensor *stages[ET_TR3_C_O2_MOMENT_ASSIGNMENTS];
  void *i2_builder;
};

static et_o2_tr3_c_restore_plan *et_o2_tr3_c_restore_plans;

static _Noreturn void et_o2_tr3_c_restore_fail_stop(void) { _Exit(134); }

static int et_o2_tr3_c_config_equal(const et_o2_config *left,
                                    const et_o2_config *right) {
  return left->clip_kind == right->clip_kind &&
         left->clip_max_bits == right->clip_max_bits &&
         left->schedule_kind == right->schedule_kind &&
         left->minimum_ratio_bits == right->minimum_ratio_bits &&
         left->warmup_updates == right->warmup_updates &&
         left->total_updates == right->total_updates;
}

static int et_o2_tr3_c_options_equal(const et_o2_options *left,
                                     const et_o2_options *right) {
  return left->learning_rate_bits == right->learning_rate_bits &&
         left->beta1_bits == right->beta1_bits &&
         left->beta2_bits == right->beta2_bits &&
         left->epsilon_bits == right->epsilon_bits &&
         left->weight_decay_bits == right->weight_decay_bits;
}

static et_o2_tr3_c_restore_plan *
et_o2_tr3_c_find_restore_plan(const void *candidate) {
  et_o2_tr3_c_restore_plan *cursor = et_o2_tr3_c_restore_plans;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor->magic == ET_O2_TR3_C_RESTORE_MAGIC ? cursor : NULL;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static int32_t et_o2_tr3_c_restore_error(
    et_o2_error_v1 *error, uint32_t category, uint32_t code,
    const char *message) {
  return et_o2_fail(error, category, code, ET_O2_TR3_C_RESTORE_OPERATION,
                    message);
}

static int32_t et_o2_tr3_c_restore_from_i2(int32_t result,
                                           et_o2_error_v1 *error) {
  switch (result) {
  case ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT:
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_INVALID_HANDLE,
        "I2 restore operand is malformed or has the wrong kind");
  case ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE:
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW,
        "I2 restore operand is borrowed, pinned, or in the wrong phase");
  case ET_I2_PRIVATE_OWNED_RESULT_SHAPE_MISMATCH:
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_SHAPE_MISMATCH, ET_O2_CODE_INVALID_MOMENT,
        "I2 restore moment shape differs from its destination");
  case ET_I2_PRIVATE_OWNED_RESULT_NONFINITE:
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_CORRUPT_DATA, ET_O2_CODE_NONFINITE,
        "I2 restore stage contains a nonfinite value");
  case ET_I2_PRIVATE_OWNED_RESULT_INVALID_VALUE:
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_CORRUPT_DATA, ET_O2_CODE_INVALID_MOMENT,
        "I2 restore second moment is negative");
  case ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT:
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_OWNER_CONFLICT,
        "I2 restore authority differs from the typed restore plan");
  case ET_I2_PRIVATE_OWNED_RESULT_VERSION_MISMATCH:
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_VERSION_MISMATCH, ET_O2_CODE_INVALID_OPTION,
        "I2 restore record version differs");
  case ET_I2_PRIVATE_OWNED_RESULT_ALLOCATION_FAILED:
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_ALLOCATION_FAILED,
        "I2 restore allocation failed");
  default:
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_DEFECT,
        "I2 restore provider invariant failed");
  }
}

static int et_o2_tr3_c_error_span_valid(const et_o2_error_v1 *error) {
  return error == NULL ||
         et_o2_private_span_valid(error, sizeof(*error),
                                  _Alignof(et_o2_error_v1));
}

static int32_t et_o2_tr3_c_restore_receiver(
    et_o2_optimizer *candidate, et_o2_optimizer **resolved,
    et_o2_error_v1 *error) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer(candidate);
  if (optimizer == NULL) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_INVALID_HANDLE,
        "optimizer is foreign, stale, or cross-kind");
  }
  if (!et_o2_optimizer_provider_valid(optimizer)) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_MISMATCH,
        "optimizer provider invariant differs from O2 2.0");
  }
  if (optimizer->count != ET_TR3_C_MODEL_DESTINATIONS) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_SHAPE_MISMATCH, ET_O2_CODE_MISSING_PARAMETER,
        "optimizer receiver does not contain exactly 14 parameters");
  }
  if (optimizer->entries == NULL || !et_o2_config_valid(&optimizer->config)) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_DEFECT,
        "optimizer receiver storage is malformed");
  }
  if (optimizer->busy != 0u || optimizer->active_operation != NULL) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW,
        "optimizer receiver already owns an active operation");
  }
  *resolved = optimizer;
  return ET_O2_STATUS_OK;
}

static int32_t et_o2_tr3_c_restore_gradient_absent(
    et_f32_parameter *parameter, et_o2_error_v1 *error,
    uint32_t malformed_category, uint32_t malformed_code) {
  et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
  if (et_f32_parameter_gradient_metadata_v1(parameter, &metadata, NULL) != 0) {
    return et_o2_tr3_c_restore_error(
        error, malformed_category, malformed_code,
        "optimizer parameter gradient metadata is malformed");
  }
  if (metadata.state != ET_F32_GRADIENT_ABSENT ||
      metadata.contribution_count != 0u ||
      metadata.normalization_weight_bits != 0u) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_GRADIENT_METADATA,
        "optimizer restore requires absent gradients");
  }
  return ET_O2_STATUS_OK;
}

static void et_o2_tr3_c_restore_i2_request(
    const et_o2_tr3_c_restore_plan *plan,
    et_tr3_c_i2_restore42_append_v1 *request) {
  memset(request, 0, sizeof(*request));
  request->struct_size = sizeof(*request);
  request->restore_authority = plan;
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    request->parameters[index] = plan->parameters[index];
    request->p1_handles[index] = plan->p1_handles[index];
  }
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    request->moment_destinations[index] = plan->moment_destinations[index];
    request->moment_sources[index] = plan->stages[index];
  }
}

static int32_t et_o2_tr3_c_restore_require_prepared(
    et_o2_tr3_c_restore_plan *candidate,
    et_o2_tr3_c_restore_plan **resolved, et_o2_error_v1 *error) {
  et_o2_tr3_c_restore_plan *plan =
      et_o2_tr3_c_find_restore_plan(candidate);
  if (plan == NULL) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_INVALID_HANDLE,
        "restore plan is foreign, stale, copied, or cross-kind");
  }
  if (plan->lifecycle != ET_O2_TR3_C_RESTORE_PREPARED) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED,
        "restore plan is not prepared");
  }
  *resolved = plan;
  return ET_O2_STATUS_OK;
}

static int32_t et_o2_tr3_c_restore_baseline(
    et_o2_tr3_c_restore_plan *plan, int check_stages,
    et_o2_error_v1 *error) {
  et_o2_optimizer *owner = et_o2_find_optimizer(plan->owner);
  if (owner == NULL || owner->entries == NULL) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_DEFECT,
        "restore plan owner storage is malformed");
  }
  if (!et_o2_optimizer_provider_valid(owner)) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_MISMATCH,
        "restore plan owner provider differs from O2 2.0");
  }
  if (owner->count != ET_TR3_C_MODEL_DESTINATIONS) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_SHAPE_MISMATCH, ET_O2_CODE_MISSING_PARAMETER,
        "restore plan owner no longer contains exactly 14 parameters");
  }
  if (!et_o2_active_operation_owns(owner, plan)) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_OWNER_CONFLICT,
        "restore plan no longer owns the optimizer");
  }
  if (owner->completed_updates !=
      plan->expected_receiver_completed_updates) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_COUNTER_MISMATCH,
        "optimizer update counter changed after restore prepare");
  }
  if (!et_o2_tr3_c_config_equal(&owner->config, &plan->baseline_config)) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_OWNER_CONFLICT,
        "optimizer configuration changed after restore prepare");
  }
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    const et_o2_entry *entry = &owner->entries[index];
    const et_f32_tensor *canonical =
        et_f32_parameter_canonical_owner_v1(plan->parameters[index]);
    if (entry->initialized != 1u || entry->parameter != plan->parameters[index] ||
        entry->p1_handle != plan->p1_handles[index] ||
        entry->storage_owner != plan->value_tensors[index] ||
        canonical != plan->value_tensors[index] ||
        entry->exp_avg != plan->moment_destinations[index * 2u] ||
        entry->exp_avg_sq != plan->moment_destinations[index * 2u + 1u] ||
        !et_o2_tr3_c_options_equal(&entry->options,
                                   &plan->baseline_options[index]) ||
        et_f32_parameter_validate_identity_v1(
            plan->parameters[index], plan->p1_handles[index], NULL) != 0) {
      return et_o2_tr3_c_restore_error(
          error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_OWNER_CONFLICT,
          "optimizer identities changed after restore prepare");
    }
    if (et_o2_tr3_c_restore_gradient_absent(
            plan->parameters[index], error, ET_O2_STATUS_INVALID_STATE,
            ET_O2_CODE_OWNER_CONFLICT) != 0) {
      return error != NULL ? (int32_t)error->category
                           : ET_O2_STATUS_INVALID_STATE;
    }
  }
  if (check_stages != 0) {
    for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
      const uint32_t policy =
          index % 2u == 0u ? ET_I2_PRIVATE_OWNED_VALUE_FINITE
                           : ET_I2_PRIVATE_OWNED_VALUE_FINITE_NONNEGATIVE;
      const int32_t match = et_i2_private_owned_clone_match_v1(
          plan->stages[index], plan->moment_destinations[index], policy);
      if (match != 0) {
        return et_o2_tr3_c_restore_from_i2(match, error);
      }
    }
  }
  return ET_O2_STATUS_OK;
}

static int et_o2_tr3_c_restore_sealed_state(
    const et_o2_tr3_c_restore_plan *plan, int target, int check_stages) {
  et_o2_optimizer *owner;
  if (plan == NULL || plan->magic != ET_O2_TR3_C_RESTORE_MAGIC ||
      plan->owner == NULL ||
      (owner = et_o2_find_optimizer(plan->owner)) == NULL ||
      !et_o2_optimizer_provider_valid(owner) ||
      owner->count != ET_TR3_C_MODEL_DESTINATIONS || owner->entries == NULL ||
      !et_o2_active_operation_owns(owner, plan) ||
      owner->completed_updates !=
          (target != 0 ? plan->target_completed_updates
                       : plan->expected_receiver_completed_updates) ||
      !et_o2_tr3_c_config_equal(
          &owner->config,
          target != 0 ? &plan->target_config : &plan->baseline_config)) {
    return 0;
  }
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
    const et_o2_entry *entry = &owner->entries[index];
    if (entry->initialized != 1u || entry->parameter != plan->parameters[index] ||
        entry->p1_handle != plan->p1_handles[index] ||
        entry->storage_owner != plan->value_tensors[index] ||
        et_f32_parameter_canonical_owner_v1(plan->parameters[index]) !=
            plan->value_tensors[index] ||
        et_f32_parameter_validate_identity_v1(
            plan->parameters[index], plan->p1_handles[index], NULL) != 0 ||
        entry->exp_avg != plan->moment_destinations[index * 2u] ||
        entry->exp_avg_sq != plan->moment_destinations[index * 2u + 1u] ||
        !et_o2_tr3_c_options_equal(
            &entry->options, target != 0 ? &plan->target_options[index]
                                         : &plan->baseline_options[index]) ||
        et_f32_parameter_gradient_metadata_v1(
            plan->parameters[index], &metadata, NULL) != 0 ||
        metadata.state != ET_F32_GRADIENT_ABSENT ||
        metadata.contribution_count != 0u ||
        metadata.normalization_weight_bits != 0u) {
      return 0;
    }
  }
  if (check_stages != 0) {
    for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
      const uint32_t policy =
          index % 2u == 0u ? ET_I2_PRIVATE_OWNED_VALUE_FINITE
                           : ET_I2_PRIVATE_OWNED_VALUE_FINITE_NONNEGATIVE;
      if (et_i2_private_owned_clone_match_v1(
              plan->stages[index], plan->moment_destinations[index], policy) !=
          0) {
        return 0;
      }
    }
  }
  return 1;
}

static void et_o2_tr3_c_restore_clear_pointers(
    et_o2_tr3_c_restore_plan *plan) {
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    plan->parameters[index] = NULL;
    plan->p1_handles[index] = NULL;
    plan->value_tensors[index] = NULL;
  }
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    plan->moment_destinations[index] = NULL;
    plan->stages[index] = NULL;
  }
  plan->i2_builder = NULL;
  plan->owner = NULL;
}

int32_t et_o2_tr3_c_restore_prepare_v1(
    et_o2_optimizer *candidate,
    const et_o2_tr3_c_restore_request_v1 *input,
    et_o2_tr3_c_restore_plan **output, et_o2_error_v1 *error) {
  et_o2_tr3_c_restore_request_v1 request;
  et_o2_tr3_c_restore_plan *plan = NULL;
  et_o2_optimizer *optimizer = NULL;
  const et_f32_tensor *values[ET_TR3_C_MODEL_DESTINATIONS];
  const int error_valid = et_o2_tr3_c_error_span_valid(error);
  int aliases_error = 0;
  int32_t result;

  if (input != NULL &&
      !et_o2_private_span_valid(
          input, sizeof(*input), _Alignof(et_o2_tr3_c_restore_request_v1))) {
    return et_o2_tr3_c_restore_error(
        error_valid ? error : NULL, ET_O2_STATUS_INVALID_ARGUMENT,
        ET_O2_CODE_INVALID_HANDLE, "restore request storage is invalid");
  }
  if (output != NULL &&
      !et_o2_private_span_valid(
          output, sizeof(*output), _Alignof(et_o2_tr3_c_restore_plan *))) {
    return et_o2_tr3_c_restore_error(
        error_valid ? error : NULL, ET_O2_STATUS_INVALID_ARGUMENT,
        ET_O2_CODE_INVALID_HANDLE, "restore output storage is invalid");
  }
  if (!error_valid) {
    return ET_O2_STATUS_INVALID_ARGUMENT;
  }
  if (input != NULL && output != NULL &&
      et_o2_private_ranges_overlap(input, sizeof(*input), output,
                                   sizeof(*output))) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_INVALID_HANDLE,
        "restore request and output storage overlap");
  }
  if (error != NULL) {
    aliases_error =
        (input != NULL &&
         et_o2_private_ranges_overlap(input, sizeof(*input), error,
                                      sizeof(*error))) ||
        (output != NULL &&
         et_o2_private_ranges_overlap(output, sizeof(*output), error,
                                      sizeof(*error)));
    if (aliases_error != 0) {
      return ET_O2_STATUS_INVALID_ARGUMENT;
    }
  }
  if (candidate == NULL || input == NULL || output == NULL) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_NULL_ARGUMENT,
        "optimizer, restore request, and plan output are required");
  }
  memcpy(&request, input, sizeof(request));
  if (*output != NULL) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_OWNER_CONFLICT,
        "restore plan output is already owned");
  }
  if (request.struct_size != sizeof(request)) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_VERSION_MISMATCH, ET_O2_CODE_INVALID_OPTION,
        "restore request version differs");
  }
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    if (request.entries[index].struct_size != sizeof(request.entries[index]) ||
        request.entries[index].reserved != 0u) {
      return et_o2_tr3_c_restore_error(
          error, ET_O2_STATUS_VERSION_MISMATCH, ET_O2_CODE_INVALID_OPTION,
          "restore entry version or reserved field differs");
    }
  }
  if (request.expected_receiver_completed_updates > (uint64_t)INT64_MAX) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_COUNTER_OVERFLOW,
        "receiver counter is outside signed-i64 range");
  }
  if (request.target_completed_updates > (uint64_t)INT64_MAX) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_CORRUPT_DATA, ET_O2_CODE_COUNTER_OVERFLOW,
        "target counter is outside signed-i64 range");
  }
  et_o2_config target_config = {
      request.clip_kind, request.clip_max_bits, request.schedule_kind,
      request.minimum_ratio_bits, request.warmup_updates,
      request.total_updates};
  if (!et_o2_config_valid(&target_config)) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_CORRUPT_DATA, ET_O2_CODE_INVALID_OPTION,
        "target optimizer configuration is invalid");
  }
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    const et_o2_options options = {
        request.entries[index].learning_rate_bits,
        request.entries[index].beta1_bits,
        request.entries[index].beta2_bits,
        request.entries[index].epsilon_bits,
        request.entries[index].weight_decay_bits};
    if (!et_o2_options_valid(&options)) {
      return et_o2_tr3_c_restore_error(
          error, ET_O2_STATUS_CORRUPT_DATA, ET_O2_CODE_INVALID_OPTION,
          "target optimizer group options are invalid");
    }
  }
  result = et_o2_tr3_c_restore_receiver(candidate, &optimizer, error);
  if (result != 0) {
    return result;
  }
  if (optimizer->completed_updates !=
      request.expected_receiver_completed_updates) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_COUNTER_MISMATCH,
        "optimizer receiver counter differs from the request");
  }

  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    et_o2_entry *entry = &optimizer->entries[index];
    const et_f32_tensor *value;
    if (entry->initialized != 1u || entry->parameter == NULL ||
        entry->p1_handle == NULL || entry->storage_owner == NULL ||
        entry->exp_avg == NULL || entry->exp_avg_sq == NULL ||
        !et_o2_options_valid(&entry->options) ||
        et_f32_parameter_validate_identity_v1(entry->parameter,
                                              entry->p1_handle, NULL) != 0 ||
        (value = et_f32_parameter_canonical_owner_v1(entry->parameter)) ==
            NULL ||
        value != entry->storage_owner ||
        et_f32_tensor_canonical_owner_v1(entry->exp_avg) != entry->exp_avg ||
        et_f32_tensor_canonical_owner_v1(entry->exp_avg_sq) !=
            entry->exp_avg_sq) {
      result = et_o2_tr3_c_restore_error(
          error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_DEFECT,
          "optimizer receiver entry storage is malformed");
      return result;
    }
    if ((result = et_o2_tr3_c_restore_gradient_absent(
             entry->parameter, error, ET_O2_STATUS_INTERNAL,
             ET_O2_CODE_PROVIDER_DEFECT)) != 0) {
      return result;
    }
    values[index] = value;
    for (size_t previous = 0u; previous < index; ++previous) {
      /* A valid parameter/P1/value identity is a bijection, so a reachable
         duplicate repeats all three.  Keep the P1/value comparisons as
         defensive rejection of an impossible corrupted representation. */
      if (entry->parameter == optimizer->entries[previous].parameter ||
          entry->p1_handle == optimizer->entries[previous].p1_handle ||
          value == values[previous]) {
        return et_o2_tr3_c_restore_error(
            error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_DEFECT,
            "optimizer receiver contains a duplicate parameter identity");
      }
    }
  }
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    const size_t parameter_index = index / 2u;
    const et_f32_tensor *stage = index % 2u == 0u
                                     ? request.entries[parameter_index]
                                           .exp_avg_owned
                                     : request.entries[parameter_index]
                                           .exp_avg_sq_owned;
    et_f32_tensor *destination =
        index % 2u == 0u ? optimizer->entries[parameter_index].exp_avg
                         : optimizer->entries[parameter_index].exp_avg_sq;
    const uint32_t policy =
        index % 2u == 0u ? ET_I2_PRIVATE_OWNED_VALUE_FINITE
                         : ET_I2_PRIVATE_OWNED_VALUE_FINITE_NONNEGATIVE;
    result = et_i2_private_owned_clone_match_v1(stage, destination, policy);
    if (result != 0) {
      return et_o2_tr3_c_restore_from_i2(result, error);
    }
    for (size_t other = 0u; other < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++other) {
      const size_t other_parameter = other / 2u;
      const et_f32_tensor *other_stage =
          other % 2u == 0u
              ? request.entries[other_parameter].exp_avg_owned
              : request.entries[other_parameter].exp_avg_sq_owned;
      et_f32_tensor *other_destination =
          other % 2u == 0u ? optimizer->entries[other_parameter].exp_avg
                           : optimizer->entries[other_parameter].exp_avg_sq;
      if ((index != other &&
           (stage == other_stage || destination == other_destination)) ||
          stage == other_destination) {
        return et_o2_tr3_c_restore_error(
            error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_INVALID_HANDLE,
            "restore moment sources and destinations must be disjoint");
      }
    }
  }

  plan = (et_o2_tr3_c_restore_plan *)et_o2_calloc(1u, sizeof(*plan));
  if (plan == NULL) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_ALLOCATION_FAILED,
        "cannot allocate native restore plan");
  }
  plan->owner = optimizer;
  plan->expected_receiver_completed_updates =
      request.expected_receiver_completed_updates;
  plan->target_completed_updates = request.target_completed_updates;
  plan->baseline_config = optimizer->config;
  plan->target_config = target_config;
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    const et_o2_options target_options = {
        request.entries[index].learning_rate_bits,
        request.entries[index].beta1_bits,
        request.entries[index].beta2_bits,
        request.entries[index].epsilon_bits,
        request.entries[index].weight_decay_bits};
    plan->baseline_options[index] = optimizer->entries[index].options;
    plan->target_options[index] = target_options;
    plan->parameters[index] = optimizer->entries[index].parameter;
    plan->p1_handles[index] = optimizer->entries[index].p1_handle;
    plan->value_tensors[index] = values[index];
    plan->moment_destinations[index * 2u] =
        optimizer->entries[index].exp_avg;
    plan->moment_destinations[index * 2u + 1u] =
        optimizer->entries[index].exp_avg_sq;
    plan->stages[index * 2u] = request.entries[index].exp_avg_owned;
    plan->stages[index * 2u + 1u] =
        request.entries[index].exp_avg_sq_owned;
  }

  plan->magic = ET_O2_TR3_C_RESTORE_MAGIC;
  plan->lifecycle = ET_O2_TR3_C_RESTORE_PREPARED;
  if (et_o2_active_operation_acquire(optimizer, plan) != 0) {
    result = et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW,
        "optimizer restore ownership could not be acquired");
    goto fail;
  }
  plan->registry_next = et_o2_tr3_c_restore_plans;
  et_o2_tr3_c_restore_plans = plan;
  *output = plan;
  return et_o2_success(error);

fail:
  if (plan != NULL && plan->owner != NULL &&
      et_o2_active_operation_owns(plan->owner, plan) &&
      et_o2_active_operation_release(plan->owner, plan) != 0) {
    et_o2_tr3_c_restore_fail_stop();
  }
  free(plan);
  return result;
}

int32_t et_o2_tr3_c_restore_append_v1(
    et_o2_tr3_c_restore_plan *candidate, void *exact_i2_copy_builder,
    size_t first_index, et_o2_error_v1 *error) {
  et_o2_tr3_c_restore_plan *plan = NULL;
  et_tr3_c_i2_restore42_append_v1 request;
  int32_t result =
      et_o2_tr3_c_restore_require_prepared(candidate, &plan, error);
  if (result != 0) {
    return result;
  }
  if (first_index != ET_TR3_C_O2_FIRST_INDEX) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_INVALID_OPTION,
        "restore suffix must begin at fixed index 14");
  }
  if (plan->i2_builder != NULL) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED,
        "restore plan already has an I2 builder witness");
  }
  if ((result = et_o2_tr3_c_restore_baseline(plan, 1, error)) != 0) {
    return result;
  }
  et_o2_tr3_c_restore_i2_request(plan, &request);
  result = et_tr3_c_i2_copy_builder_append_restore42_v1(
      exact_i2_copy_builder, &request);
  if (result != 0) {
    return et_o2_tr3_c_restore_from_i2(result, error);
  }
  plan->i2_builder = exact_i2_copy_builder;
  return et_o2_success(error);
}

int32_t et_o2_tr3_c_restore_check_v1(
    et_o2_tr3_c_restore_plan *candidate, et_o2_error_v1 *error) {
  et_o2_tr3_c_restore_plan *plan = NULL;
  et_tr3_c_i2_restore42_append_v1 request;
  int32_t result =
      et_o2_tr3_c_restore_require_prepared(candidate, &plan, error);
  if (result != 0) {
    return result;
  }
  if ((result = et_o2_tr3_c_restore_baseline(plan, 1, error)) != 0) {
    return result;
  }
  if (plan->i2_builder == NULL) {
    return et_o2_tr3_c_restore_error(
        error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW,
        "restore plan has no complete I2 builder witness");
  }
  et_o2_tr3_c_restore_i2_request(plan, &request);
  result = et_tr3_c_i2_copy_builder_verify_restore42_v1(plan->i2_builder,
                                                        &request);
  return result == 0 ? et_o2_success(error)
                     : et_o2_tr3_c_restore_from_i2(result, error);
}

void et_o2_tr3_c_restore_commit_native_v1(
    et_o2_tr3_c_restore_plan *candidate) {
  et_o2_tr3_c_restore_plan *plan =
      et_o2_tr3_c_find_restore_plan(candidate);
  if (plan == NULL || plan->lifecycle != ET_O2_TR3_C_RESTORE_PREPARED ||
      !et_o2_tr3_c_restore_sealed_state(plan, 0, 0) ||
      plan->i2_builder == NULL ||
      et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
          plan->i2_builder, ET_TR3_C_I2_TERMINAL_COMMITTED) != 0) {
    et_o2_tr3_c_restore_fail_stop();
  }
  plan->lifecycle = ET_O2_TR3_C_RESTORE_COMMITTING;
  plan->owner->config = plan->target_config;
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    plan->owner->entries[index].options = plan->target_options[index];
  }
  plan->owner->completed_updates = plan->target_completed_updates;
}

void et_o2_tr3_c_restore_finalize_v1(
    et_o2_tr3_c_restore_plan *candidate) {
  et_o2_tr3_c_restore_plan *plan =
      et_o2_tr3_c_find_restore_plan(candidate);
  et_o2_optimizer *owner;
  if (plan == NULL || plan->lifecycle != ET_O2_TR3_C_RESTORE_COMMITTING ||
      !et_o2_tr3_c_restore_sealed_state(plan, 1, 1) ||
      plan->i2_builder == NULL ||
      et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
          plan->i2_builder, ET_TR3_C_I2_TERMINAL_COMMITTED) != 0) {
    et_o2_tr3_c_restore_fail_stop();
  }
  owner = plan->owner;
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    et_tr3_c_i2_owned_release_checked_v1(
        (void *)(uintptr_t)plan->stages[index]);
    plan->stages[index] = NULL;
  }
  if (et_o2_active_operation_release(owner, plan) != 0) {
    et_o2_tr3_c_restore_fail_stop();
  }
  et_o2_tr3_c_restore_clear_pointers(plan);
  plan->lifecycle = ET_O2_TR3_C_RESTORE_COMMITTED;
}

int32_t et_o2_tr3_c_restore_abort_v1(
    et_o2_tr3_c_restore_plan *candidate, et_o2_error_v1 *error) {
  et_o2_tr3_c_restore_plan *plan = NULL;
  et_o2_optimizer *owner;
  int32_t result =
      et_o2_tr3_c_restore_require_prepared(candidate, &plan, error);
  if (result != 0) {
    return result;
  }
  if ((result = et_o2_tr3_c_restore_baseline(plan, 1, error)) != 0) {
    return result;
  }
  if (plan->i2_builder != NULL) {
    result = et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
        plan->i2_builder, ET_TR3_C_I2_TERMINAL_ABORTED);
    if (result == ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE) {
      return et_o2_tr3_c_restore_error(
          error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED,
          "restore builder was committed rather than aborted");
    }
    if (result == ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT) {
      et_tr3_c_i2_restore42_append_v1 request;
      et_o2_tr3_c_restore_i2_request(plan, &request);
      result = et_tr3_c_i2_copy_builder_verify_restore42_v1(
          plan->i2_builder, &request);
      if (result == ET_I2_PRIVATE_OWNED_RESULT_OK) {
        return et_o2_tr3_c_restore_error(
            error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW,
            "restore builder is not yet terminal-aborted");
      }
    }
    if (result != 0) {
      return et_o2_tr3_c_restore_from_i2(result, error);
    }
  }
  (void)et_o2_success(error);
  owner = plan->owner;
  plan->lifecycle = ET_O2_TR3_C_RESTORE_ABORTING;
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    et_tr3_c_i2_owned_release_checked_v1(
        (void *)(uintptr_t)plan->stages[index]);
    plan->stages[index] = NULL;
  }
  if (et_o2_active_operation_release(owner, plan) != 0) {
    et_o2_tr3_c_restore_fail_stop();
  }
  et_o2_tr3_c_restore_clear_pointers(plan);
  plan->lifecycle = ET_O2_TR3_C_RESTORE_ABORTED;
  return ET_O2_STATUS_OK;
}

#ifdef ET_O2_TESTING
void et_o2_tr3_c_restore_test_counts_snapshot_v1(
    et_o2_tr3_c_restore_test_counts_v1 *counts) {
  et_o2_tr3_c_restore_test_counts_v1 snapshot = {
      .struct_size = sizeof(snapshot)};
  et_o2_tr3_c_restore_plan *plan = et_o2_tr3_c_restore_plans;
  while (plan != NULL) {
    if (plan->lifecycle == ET_O2_TR3_C_RESTORE_PREPARED ||
        plan->lifecycle == ET_O2_TR3_C_RESTORE_COMMITTING ||
        plan->lifecycle == ET_O2_TR3_C_RESTORE_ABORTING) {
      snapshot.live_plans++;
      for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS;
           ++index) {
        snapshot.live_stages += plan->stages[index] != NULL;
      }
    } else if (plan->lifecycle == ET_O2_TR3_C_RESTORE_COMMITTED) {
      snapshot.committed_plans++;
      snapshot.retained_control_bytes += sizeof(*plan);
    } else if (plan->lifecycle == ET_O2_TR3_C_RESTORE_ABORTED) {
      snapshot.aborted_plans++;
      snapshot.retained_control_bytes += sizeof(*plan);
    }
    plan = plan->registry_next;
  }
  if (counts != NULL && counts->struct_size == sizeof(*counts)) {
    *counts = snapshot;
  }
}
#endif
