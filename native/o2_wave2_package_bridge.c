/* O2 is a source-composed successor of I2. Including the reviewed I2 bridge
 * here preserves one P1/provider/owned-carrier identity universe. */
#ifndef ET_O2_NATIVE_HELPERS_ONLY
#include "e1b_error_consumer_bridge.h"
#include "i2_wave2_package_bridge.c"
#endif

#include "o2_optimizer_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static _Thread_local et_o2_error_v1 et_o2_bridge_error;

int64_t et_o2_private_last_error_category_v1(void) {
  return (int64_t)et_o2_bridge_error.category;
}

void *et_o2_private_builder_create_v1(
    int64_t count, int64_t clip_kind, int64_t clip_bits,
    int64_t schedule_kind, int64_t warmup, int64_t total,
    int64_t minimum_ratio_bits) {
  et_o2_optimizer_builder *builder = NULL;
  if (count < 0 || clip_kind < 0 || clip_bits < 0 || schedule_kind < 0 ||
      warmup < 0 || total < 0 || minimum_ratio_bits < 0 ||
      (uint64_t)count > SIZE_MAX || (uint64_t)clip_kind > UINT32_MAX ||
      (uint64_t)clip_bits > UINT32_MAX ||
      (uint64_t)schedule_kind > UINT32_MAX ||
      (uint64_t)minimum_ratio_bits > UINT32_MAX) {
    return NULL;
  }
  if (et_o2_optimizer_builder_create_v1(
          (size_t)count, (uint32_t)clip_kind, (uint32_t)clip_bits,
          (uint32_t)schedule_kind, (uint64_t)warmup, (uint64_t)total,
          (uint32_t)minimum_ratio_bits, &builder, &et_o2_bridge_error) != 0) {
    return NULL;
  }
  return builder;
}

int64_t et_o2_private_builder_set_v1(
    void *builder, int64_t index, void *parameter, void *p1_handle,
    int64_t learning_rate_bits, int64_t beta1_bits, int64_t beta2_bits,
    int64_t epsilon_bits, int64_t weight_decay_bits) {
  if (index < 0 || learning_rate_bits < 0 || beta1_bits < 0 ||
      beta2_bits < 0 || epsilon_bits < 0 || weight_decay_bits < 0 ||
      (uint64_t)index > SIZE_MAX ||
      (uint64_t)learning_rate_bits > UINT32_MAX ||
      (uint64_t)beta1_bits > UINT32_MAX ||
      (uint64_t)beta2_bits > UINT32_MAX ||
      (uint64_t)epsilon_bits > UINT32_MAX ||
      (uint64_t)weight_decay_bits > UINT32_MAX) {
    return -1;
  }
  return (int64_t)et_o2_optimizer_builder_set_v1(
      (et_o2_optimizer_builder *)builder, (size_t)index,
      (et_f32_parameter *)parameter, p1_handle,
      (uint32_t)learning_rate_bits, (uint32_t)beta1_bits,
      (uint32_t)beta2_bits, (uint32_t)epsilon_bits,
      (uint32_t)weight_decay_bits, &et_o2_bridge_error);
}

void *et_o2_private_builder_finish_v1(void *opaque) {
  et_o2_optimizer_builder *builder = (et_o2_optimizer_builder *)opaque;
  et_o2_optimizer *optimizer = NULL;
  if (et_o2_optimizer_builder_finish_v1(
          &builder, &optimizer, &et_o2_bridge_error) != 0) {
    return NULL;
  }
  return optimizer;
}

int64_t et_o2_private_builder_abort_v1(void *opaque) {
  et_o2_optimizer_builder *builder = (et_o2_optimizer_builder *)opaque;
  return (int64_t)et_o2_optimizer_builder_abort_v1(
      &builder, &et_o2_bridge_error);
}

int64_t et_o2_private_optimizer_step_v1(void *optimizer) {
  return (int64_t)et_o2_optimizer_step_v1(
      (et_o2_optimizer *)optimizer, &et_o2_bridge_error);
}

int64_t et_o2_private_optimizer_zero_grad_v1(void *optimizer) {
  return (int64_t)et_o2_optimizer_zero_grad_v1(
      (et_o2_optimizer *)optimizer, &et_o2_bridge_error);
}

int64_t et_o2_private_optimizer_require_absent_gradients_v1(void *optimizer) {
  return (int64_t)et_o2_optimizer_require_absent_gradients_v1(
      (et_o2_optimizer *)optimizer, &et_o2_bridge_error);
}

int64_t et_o2_private_optimizer_completed_updates_v1(void *optimizer) {
  uint64_t count = 0u;
  if (et_o2_optimizer_completed_updates_v1(
          (const et_o2_optimizer *)optimizer, &count,
          &et_o2_bridge_error) != 0 || count > INT64_MAX) {
    return -1;
  }
  return (int64_t)count;
}

int64_t et_o2_private_optimizer_schedule_factor_bits_v1(void *optimizer) {
  uint32_t bits = 0u;
  if (et_o2_optimizer_schedule_factor_bits_v1(
          (const et_o2_optimizer *)optimizer, &bits,
          &et_o2_bridge_error) != 0) {
    return -1;
  }
  return (int64_t)bits;
}

void *et_o2_private_optimizer_state_snapshot_v1(void *optimizer) {
  et_o2_optimizer_state *state = NULL;
  if (et_o2_optimizer_state_snapshot_v1(
          (et_o2_optimizer *)optimizer, &state, &et_o2_bridge_error) != 0) {
    return NULL;
  }
  return state;
}

int64_t et_o2_private_optimizer_load_state_v1(void *optimizer, void *state) {
  return (int64_t)et_o2_optimizer_load_state_v1(
      (et_o2_optimizer *)optimizer, (et_o2_optimizer_state *)state,
      &et_o2_bridge_error);
}

int64_t et_o2_private_optimizer_state_release_v1(void *state) {
  const int32_t status = et_o2_optimizer_state_release_v1(
      (et_o2_optimizer_state *)state, &et_o2_bridge_error);
  et_o2_optimizer_error saved_error;
  uint32_t lifecycle = 0u;
  if (status == 0) {
    return 0;
  }
  saved_error = et_o2_bridge_error;
  if (et_o2_optimizer_state_lifecycle_v1(
          (const et_o2_optimizer_state *)state, &lifecycle,
          &et_o2_bridge_error) != 0) {
    et_o2_bridge_error = saved_error;
    return -3;
  }
  et_o2_bridge_error = saved_error;
  if (lifecycle == ET_O2_OPTIMIZER_STATE_LIVE) {
    return -1;
  }
  if (lifecycle == ET_O2_OPTIMIZER_STATE_DEAD) {
    return -2;
  }
  return -3;
}

#ifdef ET_C2_O2_RECONSTRUCT_BRIDGE
/*
 * These helpers are compiled only into the future source-composed C2 root.
 * Standalone O2 deliberately neither defines nor references this authority.
 */
static int64_t et_c2_o2_bridge_fail(uint32_t category, uint32_t code,
                                    const char *operation,
                                    const char *message) {
  size_t length;
  memset(&et_o2_bridge_error, 0, sizeof(et_o2_bridge_error));
  et_o2_bridge_error.category = category;
  et_o2_bridge_error.code = code;
  length = strlen(operation);
  memcpy(et_o2_bridge_error.operation, operation, length + 1u);
  length = strlen(message);
  if (length >= sizeof(et_o2_bridge_error.message)) {
    length = sizeof(et_o2_bridge_error.message) - 1u;
  }
  memcpy(et_o2_bridge_error.message, message, length);
  et_o2_bridge_error.message[length] = '\0';
  return (int64_t)category;
}

static int et_c2_o2_span_fits(const void *pointer, size_t bytes) {
  return pointer != NULL && (uintptr_t)pointer <= UINTPTR_MAX - bytes;
}

static int et_c2_o2_ranges_overlap(const void *left, size_t left_bytes,
                                   const void *right, size_t right_bytes) {
  const uintptr_t left_start = (uintptr_t)left;
  const uintptr_t right_start = (uintptr_t)right;
  if (left_bytes == 0u || right_bytes == 0u) {
    return 0;
  }
  if (!et_c2_o2_span_fits(left, left_bytes) ||
      !et_c2_o2_span_fits(right, right_bytes)) {
    return 1;
  }
  return left_start < right_start + right_bytes &&
         right_start < left_start + left_bytes;
}

static unsigned char *et_c2_o2_bytevector_payload(void *header,
                                                  size_t expected_bytes,
                                                  size_t alignment) {
  int64_t encoded_bytes;
  size_t span_bytes;
  unsigned char *payload;
  if (expected_bytes > SIZE_MAX - sizeof(encoded_bytes)) {
    return NULL;
  }
  span_bytes = sizeof(encoded_bytes) + expected_bytes;
  if (!et_c2_o2_span_fits(header, span_bytes)) {
    return NULL;
  }
  memcpy(&encoded_bytes, header, sizeof(encoded_bytes));
  if (encoded_bytes < 0 || (uint64_t)encoded_bytes != expected_bytes) {
    return NULL;
  }
  payload = (unsigned char *)header + sizeof(encoded_bytes);
  if (expected_bytes != 0u && alignment > 1u &&
      (uintptr_t)payload % alignment != 0u) {
    return NULL;
  }
  return payload;
}

static uint64_t et_c2_o2_read_u64_le(const unsigned char bytes[8]) {
  return (uint64_t)bytes[0] | ((uint64_t)bytes[1] << 8u) |
         ((uint64_t)bytes[2] << 16u) | ((uint64_t)bytes[3] << 24u) |
         ((uint64_t)bytes[4] << 32u) | ((uint64_t)bytes[5] << 40u) |
         ((uint64_t)bytes[6] << 48u) | ((uint64_t)bytes[7] << 56u);
}

void *et_c2_private_o2_state_reconstruct_create_v1(
    int64_t parameter_count, int64_t clip_kind, int64_t clip_max_bits,
    int64_t schedule_kind, int64_t warmup_updates, int64_t total_updates,
    int64_t minimum_ratio_bits, int64_t completed_updates) {
  et_o2_state_reconstruct_config_v1 config;
  et_o2_state_reconstruct_builder *builder = NULL;
  if (parameter_count <= 0 ||
      (uint64_t)parameter_count > ET_O2_MAX_PARAMETERS || clip_kind < 0 ||
      (uint64_t)clip_kind > UINT32_MAX || clip_max_bits < 0 ||
      (uint64_t)clip_max_bits > UINT32_MAX || schedule_kind < 0 ||
      (uint64_t)schedule_kind > UINT32_MAX || warmup_updates < 0 ||
      total_updates < 0 || minimum_ratio_bits < 0 ||
      (uint64_t)minimum_ratio_bits > UINT32_MAX || completed_updates < 0) {
    (void)et_c2_o2_bridge_fail(ET_O2_STATUS_INVALID_ARGUMENT,
                               ET_O2_CODE_INVALID_OPTION,
                               "optimizer-state-reconstruct",
                               "reconstruction config integers are invalid");
    return NULL;
  }
  memset(&config, 0, sizeof(config));
  config.struct_size = sizeof(config);
  config.clip_kind = (uint32_t)clip_kind;
  config.clip_max_bits = (uint32_t)clip_max_bits;
  config.schedule_kind = (uint32_t)schedule_kind;
  config.minimum_ratio_bits = (uint32_t)minimum_ratio_bits;
  config.warmup_updates = (uint64_t)warmup_updates;
  config.total_updates = (uint64_t)total_updates;
  config.completed_updates = (uint64_t)completed_updates;
  config.parameter_count = (size_t)parameter_count;
  if (et_o2_optimizer_state_reconstruct_create_v1(&config, &builder,
                                                  &et_o2_bridge_error) != 0) {
    return NULL;
  }
  return builder;
}

int64_t et_c2_private_o2_state_reconstruct_set_v1(
    void *opaque, int64_t index, int64_t rank, void *shape_bytevector_header,
    int64_t element_count, void *exp_avg_bytevector_header,
    void *exp_avg_sq_bytevector_header, int64_t learning_rate_bits,
    int64_t beta1_bits, int64_t beta2_bits, int64_t epsilon_bits,
    int64_t weight_decay_bits) {
  et_o2_state_reconstruct_entry_v1 entry;
  uint64_t shape[ET_KERNEL_MAX_RANK];
  unsigned char *shape_bytes;
  unsigned char *avg_bytes;
  unsigned char *sq_bytes;
  size_t shape_size;
  size_t moment_size;
  size_t elements;
  size_t product = 1u;
  size_t dimension;

  if (opaque == NULL || index < 0 || (uint64_t)index > SIZE_MAX || rank < 0 ||
      (uint64_t)rank > ET_KERNEL_MAX_RANK || element_count < 0 ||
      (uint64_t)element_count > SIZE_MAX || learning_rate_bits < 0 ||
      (uint64_t)learning_rate_bits > UINT32_MAX || beta1_bits < 0 ||
      (uint64_t)beta1_bits > UINT32_MAX || beta2_bits < 0 ||
      (uint64_t)beta2_bits > UINT32_MAX || epsilon_bits < 0 ||
      (uint64_t)epsilon_bits > UINT32_MAX || weight_decay_bits < 0 ||
      (uint64_t)weight_decay_bits > UINT32_MAX) {
    return et_c2_o2_bridge_fail(ET_O2_STATUS_INVALID_ARGUMENT,
                                ET_O2_CODE_INVALID_OPTION,
                                "optimizer-state-reconstruct",
                                "reconstruction entry integers are invalid");
  }
  if ((uint64_t)rank > SIZE_MAX / sizeof(uint64_t) ||
      (uint64_t)element_count > SIZE_MAX / sizeof(uint32_t)) {
    return et_c2_o2_bridge_fail(
        ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_INVALID_OPTION,
        "optimizer-state-reconstruct", "reconstruction entry sizes overflow");
  }
  shape_size = (size_t)rank * sizeof(uint64_t);
  elements = (size_t)element_count;
  moment_size = elements * sizeof(uint32_t);
  shape_bytes =
      et_c2_o2_bytevector_payload(shape_bytevector_header, shape_size, 1u);
  avg_bytes = et_c2_o2_bytevector_payload(exp_avg_bytevector_header,
                                          moment_size, _Alignof(uint32_t));
  sq_bytes = et_c2_o2_bytevector_payload(exp_avg_sq_bytevector_header,
                                         moment_size, _Alignof(uint32_t));
  if (shape_bytes == NULL || avg_bytes == NULL || sq_bytes == NULL ||
      et_c2_o2_ranges_overlap(avg_bytes, moment_size, sq_bytes, moment_size)) {
    return et_c2_o2_bridge_fail(ET_O2_STATUS_INVALID_ARGUMENT,
                                ET_O2_CODE_INVALID_HANDLE,
                                "optimizer-state-reconstruct",
                                "reconstruction bytevectors are invalid");
  }
  for (dimension = 0u; dimension < (size_t)rank; ++dimension) {
    const uint64_t extent =
        et_c2_o2_read_u64_le(shape_bytes + dimension * sizeof(uint64_t));
    shape[dimension] = extent;
    if (product != 0u) {
      if (extent > SIZE_MAX ||
          (extent != 0u && product > SIZE_MAX / (size_t)extent)) {
        return et_c2_o2_bridge_fail(ET_O2_STATUS_SHAPE_MISMATCH,
                                    ET_O2_CODE_INVALID_OPTION,
                                    "optimizer-state-reconstruct",
                                    "reconstruction shape product overflows");
      }
      product *= (size_t)extent;
    }
  }
  if (product != elements) {
    return et_c2_o2_bridge_fail(ET_O2_STATUS_SHAPE_MISMATCH,
                                ET_O2_CODE_INVALID_OPTION,
                                "optimizer-state-reconstruct",
                                "reconstruction shape product is inconsistent");
  }
  memset(&entry, 0, sizeof(entry));
  entry.struct_size = sizeof(entry);
  entry.rank = (size_t)rank;
  entry.shape = shape;
  entry.exp_avg_bits = (const uint32_t *)(const void *)avg_bytes;
  entry.exp_avg_sq_bits = (const uint32_t *)(const void *)sq_bytes;
  entry.element_count = elements;
  entry.learning_rate_bits = (uint32_t)learning_rate_bits;
  entry.beta1_bits = (uint32_t)beta1_bits;
  entry.beta2_bits = (uint32_t)beta2_bits;
  entry.epsilon_bits = (uint32_t)epsilon_bits;
  entry.weight_decay_bits = (uint32_t)weight_decay_bits;
  return (int64_t)et_o2_optimizer_state_reconstruct_set_v1(
      (et_o2_state_reconstruct_builder *)opaque, (size_t)index, &entry,
      &et_o2_bridge_error);
}

int64_t et_c2_private_o2_state_reconstruct_prepare_v1(void *builder) {
  return (int64_t)et_o2_optimizer_state_reconstruct_prepare_v1(
      (et_o2_state_reconstruct_builder *)builder, &et_o2_bridge_error);
}

void *et_c2_private_o2_state_reconstruct_commit_v1(void *opaque) {
  et_o2_state_reconstruct_builder *builder =
      (et_o2_state_reconstruct_builder *)opaque;
  et_o2_optimizer_state *state = NULL;
  if (et_o2_optimizer_state_reconstruct_commit_v1(&builder, &state,
                                                  &et_o2_bridge_error) != 0) {
    return NULL;
  }
  return state;
}

int64_t et_c2_private_o2_state_reconstruct_abort_v1(void *opaque) {
  et_o2_state_reconstruct_builder *builder =
      (et_o2_state_reconstruct_builder *)opaque;
  return (int64_t)et_o2_optimizer_state_reconstruct_abort_v1(
      &builder, &et_o2_bridge_error);
}

int64_t et_c2_private_o2_state_copy_moment_bits_v1(
    void *state, int64_t index, int64_t moment_kind,
    void *destination_bytevector_header, int64_t element_count) {
  unsigned char *destination;
  size_t elements;
  size_t bytes;
  if (state == NULL || index < 0 || (uint64_t)index > SIZE_MAX ||
      moment_kind < 0 || (uint64_t)moment_kind > UINT32_MAX ||
      element_count < 0 || (uint64_t)element_count > SIZE_MAX ||
      (uint64_t)element_count > SIZE_MAX / sizeof(uint32_t)) {
    return et_c2_o2_bridge_fail(
        ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_INVALID_OPTION,
        "optimizer-state-copy-moment", "moment copy integers are invalid");
  }
  elements = (size_t)element_count;
  bytes = elements * sizeof(uint32_t);
  destination = et_c2_o2_bytevector_payload(destination_bytevector_header,
                                            bytes, _Alignof(uint32_t));
  if (destination == NULL) {
    return et_c2_o2_bridge_fail(
        ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_INVALID_HANDLE,
        "optimizer-state-copy-moment", "moment copy bytevector is invalid");
  }
  return (int64_t)et_o2_optimizer_state_copy_moment_bits_v1(
      (const et_o2_optimizer_state *)state, (size_t)index,
      (uint32_t)moment_kind, (uint32_t *)(void *)destination, elements,
      &et_o2_bridge_error);
}
#endif

#ifndef ET_O2_NATIVE_HELPERS_ONLY
#define ET_O2_DECLARE_UNARY(name)                                           \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t value)
#define ET_O2_DECLARE_BINARY(name)                                          \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t left,             \
                                     eshkol_tagged_value_t right)

ET_O2_DECLARE_BINARY(et_e1b_private_o2_optimizer_create_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_step_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_zero_grad_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_state_cabi_v1);
ET_O2_DECLARE_BINARY(et_e1b_private_o2_optimizer_load_state_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_state_release_cabi_v1);

#define ET_O2_PUBLIC_UNARY(name, target)                                    \
  void name(void *input, void *output) {                                    \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) = target(*et_e1b_box_value_v1(input));      \
  }

#define ET_O2_PUBLIC_BINARY(name, target)                                   \
  void name(void *left, void *right, void *output) {                        \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) =                                          \
        target(*et_e1b_box_value_v1(left), *et_e1b_box_value_v1(right));     \
  }

ET_O2_PUBLIC_BINARY(et_e1b_public_o2_optimizer_create_v1,
                    et_e1b_private_o2_optimizer_create_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_step_v1,
                   et_e1b_private_o2_optimizer_step_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_zero_grad_v1,
                   et_e1b_private_o2_optimizer_zero_grad_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_state_v1,
                   et_e1b_private_o2_optimizer_state_cabi_v1)
ET_O2_PUBLIC_BINARY(et_e1b_public_o2_optimizer_load_state_v1,
                    et_e1b_private_o2_optimizer_load_state_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_state_release_v1,
                   et_e1b_private_o2_optimizer_state_release_cabi_v1)
#endif
