#include "tr3_c_restore_bindings.h"

#include "tr3_c_i2_restore_internal.h"
#include "tr3_c_o2_restore_internal.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if !defined(ET_I2_PRIVATE_OWNED_CLONE_MATCH) ||                              \
    !defined(ET_TR3_C_I2_RESTORE_PRIVATE) ||                                 \
    !defined(ET_TR3_C_O2_RESTORE_NATIVE)
#error "TR3-C bindings require the accepted I2/O2 restore features"
#endif

_Static_assert(ET_TR3_C_RESTORE_REQUEST_BYTES ==
                   sizeof(et_o2_tr3_c_restore_request_v1),
               "TR3-C source request carrier size changed");

static _Thread_local int64_t et_tr3_c_restore_last_category;

static int64_t et_tr3_c_restore_fail(int64_t category) {
  et_tr3_c_restore_last_category = category;
  return -1;
}

static int64_t et_tr3_c_restore_o2_status(int32_t status,
                                          const et_o2_error_v1 *error) {
  if (status == ET_O2_STATUS_OK) {
    et_tr3_c_restore_last_category = ET_O2_STATUS_OK;
    return 0;
  }
  et_tr3_c_restore_last_category =
      error != NULL && error->category != ET_O2_STATUS_OK
          ? (int64_t)error->category
          : (int64_t)ET_O2_STATUS_INTERNAL;
  return -1;
}

static int64_t et_tr3_c_restore_i2_status(int32_t status) {
  int64_t category;
  if (status == ET_I2_PRIVATE_OWNED_RESULT_OK) {
    et_tr3_c_restore_last_category = ET_O2_STATUS_OK;
    return 0;
  }
  switch (status) {
  case ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT:
  case ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT:
    category = ET_O2_STATUS_INVALID_ARGUMENT;
    break;
  case ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE:
    category = ET_O2_STATUS_INVALID_STATE;
    break;
  case ET_I2_PRIVATE_OWNED_RESULT_SHAPE_MISMATCH:
    category = ET_O2_STATUS_SHAPE_MISMATCH;
    break;
  case ET_I2_PRIVATE_OWNED_RESULT_NONFINITE:
  case ET_I2_PRIVATE_OWNED_RESULT_INVALID_VALUE:
    category = ET_O2_STATUS_CORRUPT_DATA;
    break;
  case ET_I2_PRIVATE_OWNED_RESULT_VERSION_MISMATCH:
    category = ET_O2_STATUS_VERSION_MISMATCH;
    break;
  case ET_I2_PRIVATE_OWNED_RESULT_ALLOCATION_FAILED:
  case ET_I2_PRIVATE_OWNED_RESULT_INTERNAL:
  default:
    category = ET_O2_STATUS_INTERNAL;
    break;
  }
  return et_tr3_c_restore_fail(category);
}

static et_o2_tr3_c_restore_request_v1 *
et_tr3_c_restore_request(void *opaque) {
  int64_t declared = 0;
  uintptr_t start;
  if (opaque == NULL) {
    return NULL;
  }
  memcpy(&declared, opaque, sizeof(declared));
  start = (uintptr_t)opaque;
  if (declared != ET_TR3_C_RESTORE_REQUEST_BYTES ||
      start > UINTPTR_MAX - sizeof(declared) ||
      (start + sizeof(declared)) %
              _Alignof(et_o2_tr3_c_restore_request_v1) !=
          0u) {
    return NULL;
  }
  return (et_o2_tr3_c_restore_request_v1 *)(start + sizeof(declared));
}

static int et_tr3_c_restore_u32(int64_t value) {
  return value >= 0 && (uint64_t)value <= UINT32_MAX;
}

int64_t et_tr3_c_private_restore_last_category_v1(void) {
  return et_tr3_c_restore_last_category;
}

int64_t et_tr3_c_private_restore_request_initialize_v1(
    void *request_bytes, int64_t target_completed_updates,
    int64_t expected_receiver_completed_updates, int64_t clip_kind,
    int64_t clip_max_bits, int64_t schedule_kind,
    int64_t minimum_ratio_bits, int64_t warmup_updates,
    int64_t total_updates) {
  et_o2_tr3_c_restore_request_v1 *request =
      et_tr3_c_restore_request(request_bytes);
  if (request == NULL || target_completed_updates < 0 ||
      expected_receiver_completed_updates < 0 || warmup_updates < 0 ||
      total_updates < 0 || !et_tr3_c_restore_u32(clip_kind) ||
      !et_tr3_c_restore_u32(clip_max_bits) ||
      !et_tr3_c_restore_u32(schedule_kind) ||
      !et_tr3_c_restore_u32(minimum_ratio_bits)) {
    return et_tr3_c_restore_fail(ET_O2_STATUS_INVALID_ARGUMENT);
  }
  memset(request, 0, sizeof(*request));
  request->struct_size = sizeof(*request);
  request->target_completed_updates = (uint64_t)target_completed_updates;
  request->expected_receiver_completed_updates =
      (uint64_t)expected_receiver_completed_updates;
  request->clip_kind = (uint32_t)clip_kind;
  request->clip_max_bits = (uint32_t)clip_max_bits;
  request->schedule_kind = (uint32_t)schedule_kind;
  request->minimum_ratio_bits = (uint32_t)minimum_ratio_bits;
  request->warmup_updates = (uint64_t)warmup_updates;
  request->total_updates = (uint64_t)total_updates;
  et_tr3_c_restore_last_category = ET_O2_STATUS_OK;
  return 0;
}

int64_t et_tr3_c_private_restore_request_append14_v1(
    void *request_bytes, int64_t learning_rate_bits,
    int64_t beta1_bits, int64_t beta2_bits, int64_t epsilon_bits,
    int64_t weight_decay_bits, void *exp_avg_owned,
    void *exp_avg_sq_owned) {
  et_o2_tr3_c_restore_request_v1 *request =
      et_tr3_c_restore_request(request_bytes);
  et_o2_tr3_c_restore_entry_v1 *entry = NULL;
  size_t index;
  if (request == NULL ||
      request->struct_size != ET_O2_TR3_C_RESTORE_REQUEST_V1_0_SIZE ||
      !et_tr3_c_restore_u32(learning_rate_bits) ||
      !et_tr3_c_restore_u32(beta1_bits) ||
      !et_tr3_c_restore_u32(beta2_bits) ||
      !et_tr3_c_restore_u32(epsilon_bits) ||
      !et_tr3_c_restore_u32(weight_decay_bits) || exp_avg_owned == NULL ||
      exp_avg_sq_owned == NULL) {
    return et_tr3_c_restore_fail(ET_O2_STATUS_INVALID_ARGUMENT);
  }
  for (index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    if (request->entries[index].struct_size == 0u) {
      entry = &request->entries[index];
      break;
    }
    if (request->entries[index].struct_size !=
        ET_O2_TR3_C_RESTORE_ENTRY_V1_0_SIZE) {
      return et_tr3_c_restore_fail(ET_O2_STATUS_INVALID_STATE);
    }
  }
  if (entry == NULL) {
    return et_tr3_c_restore_fail(ET_O2_STATUS_INVALID_STATE);
  }
  entry->struct_size = sizeof(*entry);
  entry->learning_rate_bits = (uint32_t)learning_rate_bits;
  entry->beta1_bits = (uint32_t)beta1_bits;
  entry->beta2_bits = (uint32_t)beta2_bits;
  entry->epsilon_bits = (uint32_t)epsilon_bits;
  entry->weight_decay_bits = (uint32_t)weight_decay_bits;
  entry->exp_avg_owned = (const et_f32_tensor *)exp_avg_owned;
  entry->exp_avg_sq_owned = (const et_f32_tensor *)exp_avg_sq_owned;
  et_tr3_c_restore_last_category = ET_O2_STATUS_OK;
  return 0;
}

void *et_tr3_c_private_o2_restore_prepare_v1(void *optimizer,
                                             void *request_bytes) {
  et_o2_tr3_c_restore_request_v1 *request =
      et_tr3_c_restore_request(request_bytes);
  et_o2_tr3_c_restore_plan *plan = NULL;
  et_o2_error_v1 error;
  int32_t status;
  memset(&error, 0, sizeof(error));
  if (request == NULL) {
    (void)et_tr3_c_restore_fail(ET_O2_STATUS_INVALID_ARGUMENT);
    return NULL;
  }
  status = et_o2_tr3_c_restore_prepare_v1((et_o2_optimizer *)optimizer,
                                          request, &plan, &error);
  if (status != ET_O2_STATUS_OK) {
    (void)et_tr3_c_restore_o2_status(status, &error);
    return NULL;
  }
  et_tr3_c_restore_last_category = ET_O2_STATUS_OK;
  return plan;
}

int64_t et_tr3_c_private_o2_restore_append_v1(void *plan, void *builder) {
  et_o2_error_v1 error;
  memset(&error, 0, sizeof(error));
  const int32_t status = et_o2_tr3_c_restore_append_v1(
      (et_o2_tr3_c_restore_plan *)plan, builder, ET_TR3_C_O2_FIRST_INDEX,
      &error);
  return et_tr3_c_restore_o2_status(status, &error);
}

int64_t et_tr3_c_private_o2_restore_check_v1(void *plan) {
  et_o2_error_v1 error;
  memset(&error, 0, sizeof(error));
  const int32_t status = et_o2_tr3_c_restore_check_v1(
      (et_o2_tr3_c_restore_plan *)plan, &error);
  return et_tr3_c_restore_o2_status(status, &error);
}

void et_tr3_c_private_o2_restore_commit_native_v1(void *plan) {
  et_o2_tr3_c_restore_commit_native_v1((et_o2_tr3_c_restore_plan *)plan);
}

void et_tr3_c_private_o2_restore_finalize_v1(void *plan) {
  et_o2_tr3_c_restore_finalize_v1((et_o2_tr3_c_restore_plan *)plan);
}

int64_t et_tr3_c_private_o2_restore_abort_v1(void *plan) {
  et_o2_error_v1 error;
  memset(&error, 0, sizeof(error));
  const int32_t status = et_o2_tr3_c_restore_abort_v1(
      (et_o2_tr3_c_restore_plan *)plan, &error);
  return et_tr3_c_restore_o2_status(status, &error);
}

void *et_tr3_c_private_i2_restore_create_v1(void *exact_o2_plan) {
  void *builder = NULL;
  const int32_t status = et_tr3_c_i2_copy_builder_create_restore42_v1(
      exact_o2_plan, &builder);
  if (et_tr3_c_restore_i2_status(status) != 0) {
    return NULL;
  }
  return builder;
}

int64_t et_tr3_c_private_i2_restore_append_parameter_v1(
    void *builder, void *exact_o2_plan, void *parameter, void *p1_handle,
    void *owned_source) {
  return et_tr3_c_restore_i2_status(
      et_tr3_c_i2_copy_builder_append_parameter_v1(
          builder, exact_o2_plan, (et_f32_parameter *)parameter, p1_handle,
          (const et_f32_tensor *)owned_source));
}

int64_t et_tr3_c_private_i2_restore_prepare_v1(void *builder,
                                               void *exact_o2_plan) {
  return et_tr3_c_restore_i2_status(
      et_tr3_c_i2_copy_builder_prepare_restore42_v1(builder,
                                                     exact_o2_plan));
}

int64_t et_tr3_c_private_i2_restore_commit_checked_v1(
    void *builder, void *exact_o2_plan) {
  return et_tr3_c_restore_i2_status(
      et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(builder,
                                                            exact_o2_plan));
}

int64_t et_tr3_c_private_i2_restore_abort_checked_v1(
    void *builder, void *exact_o2_plan) {
  return et_tr3_c_restore_i2_status(
      et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(builder,
                                                           exact_o2_plan));
}

void et_tr3_c_private_i2_owned_release_checked_v1(void *owned) {
  et_tr3_c_i2_owned_release_checked_v1(owned);
}
