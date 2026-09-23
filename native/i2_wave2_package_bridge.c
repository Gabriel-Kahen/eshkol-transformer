#include <stddef.h>

/* Wave 2 extends the accepted source-composed Wave 1 bridge. The fixed native
 * I2 helper subset is also compiled without package wrappers by the trusted
 * source-level integration test. */
#ifndef ET_I2_NATIVE_HELPERS_ONLY
#include "e1b_error_consumer_bridge.h"
#include "t1_wave1_package_bridge.c"
#endif

#if defined(ET_TR3_C_I2_RESTORE_PRIVATE) &&                                  \
    !defined(ET_I2_PRIVATE_OWNED_CLONE_MATCH)
#error "TR3-C restore bridge requires the shared owned-clone authorizer"
#endif
#if defined(ET_TR3_C_I2_RESTORE_PRIVATE) &&                                  \
    !defined(ET_TR3_C_I2_CHECKED_BRIDGE)
#define ET_TR3_C_I2_CHECKED_BRIDGE 1
#endif

#include "f32_parameter_internal.h"
#if defined(ET_G3C4_I2_CONSTRUCTION_PRIVATE) && \
    defined(ET_G3C4_NATIVE_OWNER_PRIVATE)
#include "../src/eshkol_transformer/g3c4_model_owner_internal.h"
#endif
#if defined(ET_TR3_C_I2_RESTORE_PRIVATE)
#include "tr3_c_i2_restore_internal.h"
#endif
#if defined(ET_M3T_PACKAGE_BUILD)
#include "eshkol_transformer/m3t_transport.h"
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
  ET_I2_CARRIER_PARAMETER = 1,
  ET_I2_CARRIER_TENSOR = 2,
  ET_I2_CARRIER_OWNED_CLONE = 3,
  ET_I2_CARRIER_PARAMETER_ADMISSION = 4
};

#define ET_I2_COPY_BUILDER_MAGIC UINT64_C(0x4932434f50594231)
#define ET_I2_RESET_BUILDER_MAGIC UINT64_C(0x4932524553455431)
#define ET_I2_DECODE_BUILDER_MAGIC UINT64_C(0x49324445434f4431)

#ifdef ET_TR3_C_I2_RESTORE_PRIVATE
enum {
  ET_TR3_C_I2_RESTORE_PHASE_ORDINARY = 0,
  ET_TR3_C_I2_RESTORE_PHASE_PREFIX = 1,
  ET_TR3_C_I2_RESTORE_PHASE_COMPLETE = 2,
  ET_TR3_C_I2_RESTORE_PHASE_PREPARED = 3,
  ET_TR3_C_I2_RESTORE_PHASE_COMMITTED = 4,
  ET_TR3_C_I2_RESTORE_PHASE_ABORTED = 5
};
#endif

typedef struct et_i2_copy_builder {
  uint64_t magic;
  struct et_i2_copy_builder *registry_next;
  size_t count;
  et_f32_tensor_copy_assignment_v1 *assignments;
  et_f32_tensor_copy_plan *plan;
#ifdef ET_TR3_C_I2_RESTORE_PRIVATE
  const void *tr3_c_restore_authority;
  size_t tr3_c_next_index;
  uint32_t tr3_c_restore_phase;
#endif
} et_i2_copy_builder;

typedef struct et_i2_reset_builder {
  uint64_t magic;
  struct et_i2_reset_builder *registry_next;
  size_t count;
  et_f32_parameter **parameters;
  et_f32_gradient_reset_plan *plan;
} et_i2_reset_builder;

typedef struct et_i2_decode_builder {
  uint64_t magic;
  struct et_i2_decode_builder *registry_next;
  size_t rank;
  uint64_t *shape;
  unsigned char *payload;
  size_t payload_size;
} et_i2_decode_builder;

static _Thread_local et_f32_tensor_error et_i2_last_error;
static et_i2_copy_builder *et_i2_live_copy_builders;
static et_i2_reset_builder *et_i2_live_reset_builders;
static et_i2_decode_builder *et_i2_live_decode_builders;
static et_i2_copy_builder *et_i2_retired_copy_builders;
static et_i2_reset_builder *et_i2_retired_reset_builders;
static et_i2_decode_builder *et_i2_retired_decode_builders;
static size_t et_i2_owned_clone_count;

#ifdef ET_F32_TENSOR_TESTING
static size_t et_i2_allocation_limit = SIZE_MAX;
static size_t et_i2_successful_allocations;

void et_i2_test_fail_alloc_after_v1(size_t allowed) {
  et_i2_allocation_limit = allowed;
  et_i2_successful_allocations = 0u;
}

void et_i2_test_reset_allocator_v1(void) {
  et_i2_allocation_limit = SIZE_MAX;
  et_i2_successful_allocations = 0u;
}

void et_i2_test_live_builder_counts_v1(size_t *copy_count,
                                        size_t *reset_count,
                                        size_t *decode_count) {
  size_t copies = 0u;
  size_t resets = 0u;
  size_t decodes = 0u;
  const et_i2_copy_builder *copy;
  const et_i2_reset_builder *reset;
  const et_i2_decode_builder *decode;
  for (copy = et_i2_live_copy_builders; copy != NULL;
       copy = copy->registry_next) {
    if (copy->magic == ET_I2_COPY_BUILDER_MAGIC) {
      copies++;
    }
  }
  for (reset = et_i2_live_reset_builders; reset != NULL;
       reset = reset->registry_next) {
    if (reset->magic == ET_I2_RESET_BUILDER_MAGIC) {
      resets++;
    }
  }
  for (decode = et_i2_live_decode_builders; decode != NULL;
       decode = decode->registry_next) {
    if (decode->magic == ET_I2_DECODE_BUILDER_MAGIC) {
      decodes++;
    }
  }
  if (copy_count != NULL) {
    *copy_count = copies;
  }
  if (reset_count != NULL) {
    *reset_count = resets;
  }
  if (decode_count != NULL) {
    *decode_count = decodes;
  }
}

void et_i2_test_retired_builder_counts_v1(size_t *copy_count,
                                           size_t *reset_count,
                                           size_t *decode_count,
                                           size_t *retained_control_bytes) {
  size_t copies = 0u;
  size_t resets = 0u;
  size_t decodes = 0u;
  const et_i2_copy_builder *copy;
  const et_i2_reset_builder *reset;
  const et_i2_decode_builder *decode;
  for (copy = et_i2_retired_copy_builders; copy != NULL;
       copy = copy->registry_next) {
    copies++;
  }
  for (reset = et_i2_retired_reset_builders; reset != NULL;
       reset = reset->registry_next) {
    resets++;
  }
  for (decode = et_i2_retired_decode_builders; decode != NULL;
       decode = decode->registry_next) {
    decodes++;
  }
  if (copy_count != NULL) {
    *copy_count = copies;
  }
  if (reset_count != NULL) {
    *reset_count = resets;
  }
  if (decode_count != NULL) {
    *decode_count = decodes;
  }
  if (retained_control_bytes != NULL) {
    *retained_control_bytes = copies * sizeof(*copy) +
                              resets * sizeof(*reset) +
                              decodes * sizeof(*decode);
  }
}

size_t et_i2_test_decode_builder_control_bytes_v1(void) {
  return sizeof(et_i2_decode_builder);
}
#endif

static void *et_i2_system_calloc(size_t count, size_t size) {
  void *allocation;
#ifdef ET_F32_TENSOR_TESTING
  if (et_i2_successful_allocations >= et_i2_allocation_limit) {
    return NULL;
  }
#endif
  allocation = calloc(count, size);
#ifdef ET_F32_TENSOR_TESTING
  if (allocation != NULL) {
    et_i2_successful_allocations++;
  }
#endif
  return allocation;
}

static void *et_i2_system_malloc(size_t size) {
  void *allocation;
#ifdef ET_F32_TENSOR_TESTING
  if (et_i2_successful_allocations >= et_i2_allocation_limit) {
    return NULL;
  }
#endif
  allocation = malloc(size);
#ifdef ET_F32_TENSOR_TESTING
  if (allocation != NULL) {
    et_i2_successful_allocations++;
  }
#endif
  return allocation;
}

static et_i2_copy_builder *et_i2_find_copy_builder(const void *candidate) {
  et_i2_copy_builder *current = et_i2_live_copy_builders;
  while (current != NULL) {
    if ((const void *)current == candidate) {
      return current->magic == ET_I2_COPY_BUILDER_MAGIC ? current : NULL;
    }
    current = current->registry_next;
  }
  return NULL;
}

static et_i2_reset_builder *et_i2_find_reset_builder(const void *candidate) {
  et_i2_reset_builder *current = et_i2_live_reset_builders;
  while (current != NULL) {
    if ((const void *)current == candidate) {
      return current->magic == ET_I2_RESET_BUILDER_MAGIC ? current : NULL;
    }
    current = current->registry_next;
  }
  return NULL;
}

static et_i2_decode_builder *et_i2_find_decode_builder(const void *candidate) {
  et_i2_decode_builder *current = et_i2_live_decode_builders;
  while (current != NULL) {
    if ((const void *)current == candidate) {
      return current->magic == ET_I2_DECODE_BUILDER_MAGIC ? current : NULL;
    }
    current = current->registry_next;
  }
  return NULL;
}

#define ET_I2_DEFINE_UNLINK(name, type, head)                                \
  static void name(type *builder) {                                          \
    type **link = &(head);                                                    \
    while (*link != NULL) {                                                   \
      if (*link == builder) {                                                 \
        *link = builder->registry_next;                                       \
        builder->registry_next = NULL;                                        \
        builder->magic = 0u;                                                  \
        return;                                                               \
      }                                                                       \
      link = &(*link)->registry_next;                                         \
    }                                                                         \
  }

ET_I2_DEFINE_UNLINK(et_i2_unlink_copy_builder, et_i2_copy_builder,
                    et_i2_live_copy_builders)
ET_I2_DEFINE_UNLINK(et_i2_unlink_reset_builder, et_i2_reset_builder,
                    et_i2_live_reset_builders)
ET_I2_DEFINE_UNLINK(et_i2_unlink_decode_builder, et_i2_decode_builder,
                    et_i2_live_decode_builders)

static void et_i2_retire_copy_builder(et_i2_copy_builder *builder) {
  builder->magic = 0u;
  builder->count = 0u;
  builder->assignments = NULL;
  builder->plan = NULL;
#ifdef ET_TR3_C_I2_RESTORE_PRIVATE
  builder->tr3_c_restore_authority = NULL;
  builder->tr3_c_next_index = 0u;
  builder->tr3_c_restore_phase = ET_TR3_C_I2_RESTORE_PHASE_ORDINARY;
#endif
  builder->registry_next = et_i2_retired_copy_builders;
  et_i2_retired_copy_builders = builder;
}

static void et_i2_retire_reset_builder(et_i2_reset_builder *builder) {
  builder->magic = 0u;
  builder->count = 0u;
  builder->parameters = NULL;
  builder->plan = NULL;
  builder->registry_next = et_i2_retired_reset_builders;
  et_i2_retired_reset_builders = builder;
}

static void et_i2_retire_decode_builder(et_i2_decode_builder *builder) {
  builder->magic = 0u;
  builder->rank = 0u;
  builder->shape = NULL;
  builder->payload = NULL;
  builder->payload_size = 0u;
  builder->registry_next = et_i2_retired_decode_builders;
  et_i2_retired_decode_builders = builder;
}

#define ET_I2_DEFINE_SHELL_OVERLAP(name, type, live_head, retired_head)       \
  static int name(const void *storage, size_t bytes) {                        \
    const uintptr_t start = (uintptr_t)storage;                               \
    const uintptr_t end = start + bytes;                                      \
    const type *builder;                                                      \
    for (builder = (live_head); builder != NULL;                              \
         builder = builder->registry_next) {                                  \
      const uintptr_t shell_start = (uintptr_t)builder;                       \
      const uintptr_t shell_end = shell_start + sizeof(*builder);             \
      if (start < shell_end && shell_start < end) {                            \
        return 1;                                                             \
      }                                                                       \
    }                                                                         \
    for (builder = (retired_head); builder != NULL;                           \
         builder = builder->registry_next) {                                  \
      const uintptr_t shell_start = (uintptr_t)builder;                       \
      const uintptr_t shell_end = shell_start + sizeof(*builder);             \
      if (start < shell_end && shell_start < end) {                            \
        return 1;                                                             \
      }                                                                       \
    }                                                                         \
    return 0;                                                                 \
  }

ET_I2_DEFINE_SHELL_OVERLAP(et_i2_copy_builder_shell_overlaps,
                           et_i2_copy_builder, et_i2_live_copy_builders,
                           et_i2_retired_copy_builders)
ET_I2_DEFINE_SHELL_OVERLAP(et_i2_reset_builder_shell_overlaps,
                           et_i2_reset_builder, et_i2_live_reset_builders,
                           et_i2_retired_reset_builders)
ET_I2_DEFINE_SHELL_OVERLAP(et_i2_decode_builder_shell_overlaps,
                           et_i2_decode_builder, et_i2_live_decode_builders,
                           et_i2_retired_decode_builders)

static int et_i2_builder_shell_aliases(const void *storage, size_t bytes) {
  return bytes > 0u &&
         (et_i2_copy_builder_shell_overlaps(storage, bytes) ||
          et_i2_reset_builder_shell_overlaps(storage, bytes) ||
          et_i2_decode_builder_shell_overlaps(storage, bytes));
}

static void et_i2_clear_error(void) {
  et_f32_tensor_error_clear_v1(&et_i2_last_error);
}

static void et_i2_set_bridge_error(et_f32_tensor_error_category category,
                                   et_f32_tensor_error_code code) {
  et_i2_clear_error();
  et_i2_last_error.category = category;
  et_i2_last_error.code = code;
}

static unsigned char *et_i2_bytevector_payload(void *header,
                                                size_t expected_size) {
  int64_t encoded_size = -1;
  size_t total_size;
  if (header == NULL || expected_size > SIZE_MAX - sizeof(encoded_size)) {
    return NULL;
  }
  total_size = sizeof(encoded_size) + expected_size;
  if ((uintptr_t)header > UINTPTR_MAX - total_size ||
      et_i2_builder_shell_aliases(header, total_size)) {
    return NULL;
  }
  memcpy(&encoded_size, header, sizeof(encoded_size));
  if (encoded_size < 0 || (uint64_t)encoded_size != (uint64_t)expected_size) {
    return NULL;
  }
  return (unsigned char *)header + sizeof(encoded_size);
}

static const et_f32_tensor *et_i2_const_tensor(void *carrier, int64_t role,
                                               const void *p1_handle) {
  const et_f32_tensor *tensor = NULL;
  et_i2_clear_error();
  if (carrier == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_NULL_ARGUMENT);
    return NULL;
  }
  if (role == ET_I2_CARRIER_PARAMETER ||
      role == ET_I2_CARRIER_PARAMETER_ADMISSION) {
    et_f32_parameter *const parameter = (et_f32_parameter *)carrier;
    if (et_f32_parameter_is_live_v1(parameter) != 1 ||
        (role == ET_I2_CARRIER_PARAMETER &&
         et_f32_parameter_validate_identity_v1(parameter, p1_handle,
                                               &et_i2_last_error) != 0) ||
        et_f32_parameter_value_tensor_v1(parameter, &tensor,
                                         &et_i2_last_error) != 0) {
      return NULL;
    }
    return tensor;
  }
  if ((role == ET_I2_CARRIER_TENSOR ||
       role == ET_I2_CARRIER_OWNED_CLONE) &&
      et_f32_tensor_is_live_v1((const et_f32_tensor *)carrier) == 1) {
    return (const et_f32_tensor *)carrier;
  }
  et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_F32_TENSOR_CODE_INVALID_HANDLE);
  return NULL;
}

static et_f32_tensor *et_i2_mutable_tensor(void *carrier, int64_t role,
                                           const void *p1_handle) {
  return (et_f32_tensor *)et_i2_const_tensor(carrier, role, p1_handle);
}

int64_t et_i2_private_last_error_category_v1(void) {
  return (int64_t)et_i2_last_error.category;
}

int64_t et_i2_private_last_error_code_v1(void) {
  return (int64_t)et_i2_last_error.code;
}

void *et_i2_private_tensor_create_uniform_v1(int64_t count, int64_t bits) {
  et_f32_tensor *tensor = NULL;
  uint32_t *payload = NULL;
  uint64_t shape[1];
  size_t index;
  if (count < 0 || bits < 0 || (uint64_t)bits > UINT32_MAX) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_SHAPE);
    return NULL;
  }
  shape[0] = (uint64_t)count;
  et_i2_clear_error();
  if (et_f32_tensor_create_v1(1u, shape, &tensor, &et_i2_last_error) != 0) {
    return NULL;
  }
  if (count > 0) {
    payload =
        (uint32_t *)et_i2_system_calloc((size_t)count, sizeof(*payload));
    if (payload == NULL) {
      (void)et_f32_tensor_destroy_v1(&tensor, &et_i2_last_error);
      et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INTERNAL,
                             ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
      return NULL;
    }
    for (index = 0u; index < (size_t)count; ++index) {
      payload[index] = (uint32_t)bits;
    }
    if (et_f32_tensor_copy_bits_from_v1(tensor, payload, (size_t)count,
                                        &et_i2_last_error) != 0) {
      free(payload);
      (void)et_f32_tensor_destroy_v1(&tensor, &et_i2_last_error);
      return NULL;
    }
    free(payload);
  }
  return tensor;
}

void *et_i2_private_parameter_create_v1(void *initial) {
  et_f32_parameter *parameter = NULL;
  et_i2_clear_error();
  if (et_f32_parameter_create_v1((const et_f32_tensor *)initial, &parameter,
                                 &et_i2_last_error) != 0) {
    return NULL;
  }
  return parameter;
}

int64_t et_i2_private_tensor_destroy_v1(void *tensor_value) {
  et_f32_tensor *tensor = (et_f32_tensor *)tensor_value;
  et_i2_clear_error();
  return et_f32_tensor_destroy_v1(&tensor, &et_i2_last_error);
}

int64_t et_i2_private_parameter_bind_v1(void *parameter, void *p1_handle) {
  et_i2_clear_error();
  if (parameter == NULL || p1_handle == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
  if (et_f32_parameter_bind_identity_v1((et_f32_parameter *)parameter,
                                        p1_handle,
                                        &et_i2_last_error) != 0) {
    return -1;
  }
  return 0;
}

/* Fixed M3T construction availability and read-only teardown preflight.
 * NULL/NULL reports availability only, and never admits an owner or parameter.
 * Ordinary aggregates reject before any scope, enrollment, or allocation. */
int64_t et_i2_private_construction_parameter_preflight_v1(
    void *parameter, void *exact_handle) {
#if defined(ET_M3T_PACKAGE_BUILD)
  et_i2_clear_error();
  if (parameter == NULL && exact_handle == NULL) return INT64_C(0);
  return (int64_t)et_m3t_construction_parameter_preflight_internal(
      parameter, exact_handle, &et_i2_last_error);
#else
  (void)parameter;
  (void)exact_handle;
  return INT64_C(-1);
#endif
}

#if defined(ET_G3C4_I2_CONSTRUCTION_PRIVATE)
int64_t et_i2_private_g3c4_construction_available_v1(void) {
#if defined(ET_G3C4_NATIVE_OWNER_PRIVATE)
  return INT64_C(0);
#else
  return INT64_C(-1);
#endif
}

int64_t et_i2_private_g3c4_construction_parameter_preflight_v1(
    void *exact_owner, void *parameter, void *exact_handle) {
#if defined(ET_G3C4_NATIVE_OWNER_PRIVATE)
  et_i2_clear_error();
  return (int64_t)et_g3c4_construction_parameter_preflight_internal(
      exact_owner, parameter, exact_handle, &et_i2_last_error);
#else
  (void)exact_owner;
  (void)parameter;
  (void)exact_handle;
  return INT64_C(-1);
#endif
}
#endif

int64_t et_i2_private_parameter_admission_live_v1(void *parameter) {
  const void *identity = NULL;
  if (et_f32_parameter_is_live_v1((const et_f32_parameter *)parameter) != 1) {
    return 0;
  }
  et_i2_clear_error();
  return et_f32_parameter_identity_v1((const et_f32_parameter *)parameter,
                                      &identity, &et_i2_last_error) != 0
             ? 1
             : 0;
}

int64_t et_i2_private_carrier_live_v1(void *carrier, int64_t role,
                                      void *p1_handle) {
  return et_i2_const_tensor(carrier, role, p1_handle) != NULL ? 1 : 0;
}

int64_t et_i2_private_tensor_rank_v1(void *carrier, int64_t role,
                                     void *p1_handle) {
  const et_f32_tensor *const tensor =
      et_i2_const_tensor(carrier, role, p1_handle);
  size_t rank = 0u;
  if (tensor == NULL ||
      et_f32_tensor_rank_v1(tensor, &rank, &et_i2_last_error) != 0 ||
      rank > INT64_MAX) {
    return -1;
  }
  return (int64_t)rank;
}

int64_t et_i2_private_tensor_extent_v1(void *carrier, int64_t role,
                                       void *p1_handle, int64_t dimension) {
  const et_f32_tensor *const tensor =
      et_i2_const_tensor(carrier, role, p1_handle);
  uint64_t extent = 0u;
  if (tensor == NULL || dimension < 0 ||
      et_f32_tensor_shape_at_v1(tensor, (size_t)dimension, &extent,
                                &et_i2_last_error) != 0 ||
      extent > INT64_MAX) {
    return -1;
  }
  return (int64_t)extent;
}

void *et_i2_private_owned_clone_v1(void *carrier, int64_t role,
                                   void *p1_handle) {
  const et_f32_tensor *const tensor =
      et_i2_const_tensor(carrier, role, p1_handle);
  et_f32_tensor *clone = NULL;
  if (tensor == NULL ||
      et_f32_owned_tensor_clone_v1(tensor, &clone, &et_i2_last_error) != 0) {
    return NULL;
  }
  et_i2_owned_clone_count++;
  return clone;
}

int64_t et_i2_private_owned_release_v1(void *owned) {
  int32_t status;
  et_i2_clear_error();
  status = et_f32_owned_tensor_release_v1((et_f32_tensor *)owned,
                                          &et_i2_last_error);
  if (status == 0 && et_i2_owned_clone_count > 0u) {
    et_i2_owned_clone_count--;
  }
  return status;
}

int64_t et_i2_private_owned_clone_live_count_v1(void) {
  return et_i2_owned_clone_count <= (size_t)INT64_MAX
             ? (int64_t)et_i2_owned_clone_count
             : -1;
}

int64_t et_i2_private_storage_identical_v1(void *left, int64_t left_role,
                                           void *left_handle, void *right,
                                           int64_t right_role,
                                           void *right_handle) {
  const et_f32_tensor *left_tensor;
  const et_f32_tensor *right_tensor;
  if (left_role == ET_I2_CARRIER_PARAMETER ||
      left_role == ET_I2_CARRIER_PARAMETER_ADMISSION) {
    left_tensor = et_f32_parameter_canonical_owner_v1(
        (const et_f32_parameter *)left);
  } else {
    left_tensor = et_f32_tensor_canonical_owner_v1(
        (const et_f32_tensor *)left);
  }
  if (right_role == ET_I2_CARRIER_PARAMETER ||
      right_role == ET_I2_CARRIER_PARAMETER_ADMISSION) {
    right_tensor = et_f32_parameter_canonical_owner_v1(
        (const et_f32_parameter *)right);
  } else {
    right_tensor = et_f32_tensor_canonical_owner_v1(
        (const et_f32_tensor *)right);
  }
  (void)left_handle;
  (void)right_handle;
  if (left_tensor == NULL || right_tensor == NULL) {
    return -1;
  }
  return et_f32_tensor_storage_owner_identical_v1(left_tensor, right_tensor);
}

int64_t et_i2_private_value_equal_v1(void *left, int64_t left_role,
                                     void *left_handle, void *right,
                                     int64_t right_role, void *right_handle) {
  const et_f32_tensor *const left_tensor =
      et_i2_const_tensor(left, left_role, left_handle);
  const et_f32_tensor *right_tensor;
  int32_t equal = 0;
  if (left_tensor == NULL) {
    return -1;
  }
  right_tensor = et_i2_const_tensor(right, right_role, right_handle);
  if (right_tensor == NULL ||
      et_f32_tensor_bits_equal_v1(left_tensor, right_tensor, &equal,
                                  &et_i2_last_error) != 0) {
    return -1;
  }
  return (int64_t)equal;
}

int64_t et_i2_private_borrow_checksum_v1(void *carrier, int64_t role,
                                         void *p1_handle) {
  et_f32_tensor *const tensor =
      et_i2_mutable_tensor(carrier, role, p1_handle);
  et_f32_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  const unsigned char *bytes;
  uint64_t checksum = UINT64_C(1469598103934665603);
  size_t index;
  if (tensor == NULL ||
      et_f32_tensor_borrow_begin_v1(tensor, &borrow, &et_i2_last_error) != 0 ||
      et_f32_tensor_borrow_view_v1(borrow, &view, &et_i2_last_error) != 0) {
    if (borrow != NULL) {
      (void)et_f32_tensor_borrow_end_v1(&borrow, &et_i2_last_error);
    }
    return -1;
  }
  bytes = (const unsigned char *)view->data;
  for (index = 0u; index < view->byte_length; ++index) {
    checksum ^= (uint64_t)bytes[index];
    checksum *= UINT64_C(1099511628211);
  }
  if (et_f32_tensor_borrow_end_v1(&borrow, &et_i2_last_error) != 0) {
    return -1;
  }
  return (int64_t)(checksum & INT64_MAX);
}

int64_t et_i2_private_tensor_byte_length_v1(void *carrier, int64_t role,
                                             void *p1_handle) {
  const et_f32_tensor *const tensor =
      et_i2_const_tensor(carrier, role, p1_handle);
  size_t bytes = 0u;
  if (tensor == NULL ||
      et_f32_tensor_byte_length_v1(tensor, &bytes, &et_i2_last_error) != 0 ||
      bytes > INT64_MAX) {
    return -1;
  }
  return (int64_t)bytes;
}

int64_t et_i2_private_tensor_copy_bytes_v1(void *carrier, int64_t role,
                                            void *p1_handle,
                                            void *destination,
                                            int64_t destination_size) {
  et_f32_tensor *const tensor =
      et_i2_mutable_tensor(carrier, role, p1_handle);
  et_f32_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  const unsigned char *payload;
  int64_t status = -1;
  if (tensor == NULL || destination_size < 0 ||
      (payload = et_i2_bytevector_payload(
           destination, (size_t)destination_size)) == NULL ||
      et_f32_tensor_borrow_begin_v1(tensor, &borrow, &et_i2_last_error) != 0 ||
      et_f32_tensor_borrow_view_v1(borrow, &view, &et_i2_last_error) != 0 ||
      view->byte_length != (size_t)destination_size) {
    goto done;
  }
  memcpy((unsigned char *)payload, view->data, view->byte_length);
  status = 0;
done:
  if (borrow != NULL &&
      et_f32_tensor_borrow_end_v1(&borrow, &et_i2_last_error) != 0) {
    status = -1;
  }
  return status;
}

#ifdef ET_C2_I2_MODEL_COPY
/* C2 calls this only after its lexical I2 carrier preflight has authenticated
 * the owned-clone role under the enclosing P1 state/tensor pin. */
int64_t et_c2_private_i2_state_owned_copy_bytes_v1(
    void *owned_carrier, void *destination_bytevector_header,
    int64_t exact_byte_count) {
  const et_f32_tensor *tensor;
  unsigned char *payload;
  size_t tensor_bytes = 0u;
  size_t requested_bytes;

  et_i2_clear_error();
  if (exact_byte_count < 0 ||
      (uint64_t)exact_byte_count > (uint64_t)SIZE_MAX ||
      (uint64_t)exact_byte_count % sizeof(uint32_t) != 0u) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_BUFFER);
    return -1;
  }
  requested_bytes = (size_t)exact_byte_count;
  tensor = et_i2_const_tensor(owned_carrier, ET_I2_CARRIER_OWNED_CLONE, NULL);
  if (tensor == NULL ||
      et_f32_tensor_byte_length_v1(tensor, &tensor_bytes,
                                   &et_i2_last_error) != 0) {
    return -1;
  }
  if (tensor_bytes != requested_bytes) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                           ET_F32_TENSOR_CODE_BUFFER_SIZE_MISMATCH);
    return -1;
  }
  payload = et_i2_bytevector_payload(destination_bytevector_header,
                                     requested_bytes);
  if (payload == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_BUFFER);
    return -1;
  }
  return et_f32_tensor_copy_bits_to_v1(
             tensor, (uint32_t *)payload,
             requested_bytes / sizeof(uint32_t), &et_i2_last_error) == 0
             ? 0
             : -1;
}
#endif

void *et_i2_private_decode_builder_create_v1(int64_t rank, void *payload,
                                              int64_t payload_size) {
  et_i2_decode_builder *builder;
  const unsigned char *payload_bytes;
  if (rank < 0 || rank > 64 || payload_size < 0 ||
      (payload_bytes = et_i2_bytevector_payload(
           payload, (size_t)(payload_size < 0 ? 0 : payload_size))) == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_SHAPE);
    return NULL;
  }
  builder = (et_i2_decode_builder *)et_i2_system_calloc(1u, sizeof(*builder));
  if (builder != NULL && rank > 0) {
    builder->shape = (uint64_t *)et_i2_system_calloc(
        (size_t)rank, sizeof(*builder->shape));
  }
  if (builder != NULL && payload_size > 0) {
    builder->payload =
        (unsigned char *)et_i2_system_malloc((size_t)payload_size);
  }
  if (builder == NULL || (rank > 0 && builder->shape == NULL) ||
      (payload_size > 0 && builder->payload == NULL)) {
    if (builder != NULL) {
      free(builder->payload);
      free(builder->shape);
    }
    free(builder);
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INTERNAL,
                           ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    return NULL;
  }
  if (payload_size > 0) {
    memcpy(builder->payload, payload_bytes, (size_t)payload_size);
  }
  builder->magic = ET_I2_DECODE_BUILDER_MAGIC;
  builder->registry_next = et_i2_live_decode_builders;
  et_i2_live_decode_builders = builder;
  builder->rank = (size_t)rank;
  builder->payload_size = (size_t)payload_size;
  return builder;
}

int64_t et_i2_private_decode_builder_set_v1(void *opaque, int64_t index,
                                             int64_t extent) {
  et_i2_decode_builder *const builder = et_i2_find_decode_builder(opaque);
  if (builder == NULL || index < 0 || (size_t)index >= builder->rank ||
      extent < 0) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_SHAPE);
    return -1;
  }
  builder->shape[index] = (uint64_t)extent;
  return 0;
}

void *et_i2_private_decode_builder_finish_v1(void *opaque) {
  et_i2_decode_builder *const builder = et_i2_find_decode_builder(opaque);
  et_f32_tensor *temporary = NULL;
  et_f32_tensor *owned = NULL;
  size_t bytes = 0u;
  if (builder == NULL ||
      et_f32_tensor_create_v1(builder->rank, builder->shape, &temporary,
                              &et_i2_last_error) != 0 ||
      et_f32_tensor_byte_length_v1(temporary, &bytes, &et_i2_last_error) != 0 ||
      bytes != builder->payload_size) {
    goto done;
  }
  if (bytes > 0u) {
    if (et_f32_tensor_copy_bits_from_v1(
            temporary, (const uint32_t *)(const void *)builder->payload,
            bytes / 4u,
            &et_i2_last_error) != 0) {
      goto done;
    }
  }
  if (et_f32_owned_tensor_clone_v1(temporary, &owned,
                                   &et_i2_last_error) != 0) {
    owned = NULL;
  }
done:
  if (temporary != NULL) {
    (void)et_f32_tensor_destroy_v1(&temporary, NULL);
  }
  if (owned != NULL) {
    et_i2_owned_clone_count++;
    et_i2_unlink_decode_builder(builder);
    free(builder->payload);
    free(builder->shape);
    et_i2_retire_decode_builder(builder);
  }
  return owned;
}

int64_t et_i2_private_decode_builder_abort_v1(void *opaque) {
  et_i2_decode_builder *const builder = et_i2_find_decode_builder(opaque);
  if (opaque != NULL && builder == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
  if (builder != NULL) {
    et_i2_unlink_decode_builder(builder);
    free(builder->payload);
    free(builder->shape);
    et_i2_retire_decode_builder(builder);
  }
  return 0;
}

void *et_i2_private_copy_builder_create_v1(int64_t count) {
  et_i2_copy_builder *builder;
  if (count <= 0 || count > (int64_t)ET_F32_PARAMETER_MAX_BATCH) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_SHAPE);
    return NULL;
  }
  builder = (et_i2_copy_builder *)et_i2_system_calloc(1u, sizeof(*builder));
  if (builder != NULL) {
    builder->assignments = (et_f32_tensor_copy_assignment_v1 *)
        et_i2_system_calloc((size_t)count, sizeof(*builder->assignments));
  }
  if (builder == NULL || builder->assignments == NULL) {
    free(builder);
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INTERNAL,
                           ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    return NULL;
  }
  builder->count = (size_t)count;
  builder->magic = ET_I2_COPY_BUILDER_MAGIC;
  builder->registry_next = et_i2_live_copy_builders;
  et_i2_live_copy_builders = builder;
  et_i2_clear_error();
  return builder;
}

int64_t et_i2_private_copy_builder_set_v1(void *opaque, int64_t index,
                                          void *destination,
                                          int64_t destination_role,
                                          void *destination_handle,
                                          void *source, int64_t source_role,
                                          void *source_handle) {
  et_i2_copy_builder *const builder = et_i2_find_copy_builder(opaque);
  if (builder == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#ifdef ET_TR3_C_I2_RESTORE_PRIVATE
  if (builder->tr3_c_restore_phase != ET_TR3_C_I2_RESTORE_PHASE_ORDINARY) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#endif
  et_f32_tensor *const destination_tensor =
      et_i2_mutable_tensor(destination, destination_role, destination_handle);
  const et_f32_tensor *const source_tensor =
      et_i2_const_tensor(source, source_role, source_handle);
  if (builder->plan != NULL || index < 0 ||
      (size_t)index >= builder->count || destination_tensor == NULL ||
      source_tensor == NULL) {
    if (destination_tensor != NULL && source_tensor != NULL) {
      et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                             ET_F32_TENSOR_CODE_INVALID_HANDLE);
    }
    return -1;
  }
  builder->assignments[index].struct_size =
      ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
  builder->assignments[index].destination = destination_tensor;
  builder->assignments[index].source = source_tensor;
  return 0;
}

int64_t et_i2_private_copy_builder_prepare_v1(void *opaque) {
  et_i2_copy_builder *const builder = et_i2_find_copy_builder(opaque);
  size_t index;
  if (builder == NULL || builder->plan != NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#ifdef ET_TR3_C_I2_RESTORE_PRIVATE
  if (builder->tr3_c_restore_phase != ET_TR3_C_I2_RESTORE_PHASE_ORDINARY) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#endif
  for (index = 0u; index < builder->count; ++index) {
    if (builder->assignments[index].struct_size !=
        ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE) {
      et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                             ET_F32_TENSOR_CODE_INVALID_HANDLE);
      return -1;
    }
  }
  et_i2_clear_error();
  return et_f32_tensor_copy_plan_prepare_v1(
      builder->count, builder->assignments, &builder->plan,
      &et_i2_last_error);
}

int64_t et_i2_private_copy_builder_commit_v1(void *opaque) {
  et_i2_copy_builder *const builder = et_i2_find_copy_builder(opaque);
  int32_t status;
  if (builder == NULL || builder->plan == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#ifdef ET_TR3_C_I2_RESTORE_PRIVATE
  if (builder->tr3_c_restore_phase != ET_TR3_C_I2_RESTORE_PHASE_ORDINARY) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#endif
  et_i2_clear_error();
  status = et_f32_tensor_copy_plan_commit_v1(builder->plan,
                                             &et_i2_last_error);
  if (status == 0) {
    (void)et_f32_tensor_copy_plan_release_v1(&builder->plan,
                                             &et_i2_last_error);
    et_i2_unlink_copy_builder(builder);
    free(builder->assignments);
    et_i2_retire_copy_builder(builder);
  }
  return status;
}

int64_t et_i2_private_copy_builder_abort_v1(void *opaque) {
  et_i2_copy_builder *const builder = et_i2_find_copy_builder(opaque);
  if (opaque == NULL) {
    return 0;
  }
  if (builder == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#ifdef ET_TR3_C_I2_RESTORE_PRIVATE
  if (builder->tr3_c_restore_phase != ET_TR3_C_I2_RESTORE_PHASE_ORDINARY) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#endif
  if (builder->plan != NULL) {
    (void)et_f32_tensor_copy_plan_release_v1(&builder->plan,
                                             &et_i2_last_error);
  }
  et_i2_unlink_copy_builder(builder);
  free(builder->assignments);
  et_i2_retire_copy_builder(builder);
  return 0;
}

#ifdef ET_TR3_C_I2_CHECKED_BRIDGE
/* These entries are linked only into the source-composed TR3-C package.  The
 * source plan authenticates the parent/builder relation; this translation unit
 * authenticates the exact live child capability before touching it. */
static void et_tr3_c_i2_fail_stop(void) { _Exit(134); }

#ifdef ET_F32_TENSOR_TESTING
static int et_tr3_c_i2_fail_copy_commit;
static int et_tr3_c_i2_fail_copy_release;
static int et_tr3_c_i2_fail_owned_release;

void et_tr3_c_i2_test_fail_copy_commit_v1(int64_t enabled) {
  et_tr3_c_i2_fail_copy_commit = enabled != 0;
}

void et_tr3_c_i2_test_fail_copy_release_v1(int64_t enabled) {
  et_tr3_c_i2_fail_copy_release = enabled != 0;
}

void et_tr3_c_i2_test_fail_owned_release_v1(int64_t enabled) {
  et_tr3_c_i2_fail_owned_release = enabled != 0;
}
#endif

static int32_t et_tr3_c_i2_copy_commit(et_f32_tensor_copy_plan *plan) {
#ifdef ET_F32_TENSOR_TESTING
  if (et_tr3_c_i2_fail_copy_commit) {
    return -1;
  }
#endif
  return et_f32_tensor_copy_plan_commit_v1(plan, &et_i2_last_error);
}

static int32_t et_tr3_c_i2_copy_release(et_f32_tensor_copy_plan **plan) {
#ifdef ET_F32_TENSOR_TESTING
  if (et_tr3_c_i2_fail_copy_release) {
    return -1;
  }
#endif
  return et_f32_tensor_copy_plan_release_v1(plan, &et_i2_last_error);
}

#ifdef ET_TR3_C_I2_RESTORE_PRIVATE
static int et_tr3_c_i2_span_fits(const void *storage, size_t bytes) {
  return storage != NULL && (uintptr_t)storage <= UINTPTR_MAX - bytes;
}

static int et_tr3_c_i2_ranges_overlap(const void *left, size_t left_bytes,
                                      const void *right,
                                      size_t right_bytes) {
  const uintptr_t left_start = (uintptr_t)left;
  const uintptr_t right_start = (uintptr_t)right;
  if (left_bytes == 0u || right_bytes == 0u) {
    return 0;
  }
  if (!et_tr3_c_i2_span_fits(left, left_bytes) ||
      !et_tr3_c_i2_span_fits(right, right_bytes)) {
    return 1;
  }
  return left_start < right_start + right_bytes &&
         right_start < left_start + left_bytes;
}

static int et_tr3_c_i2_copy_assignments_overlap(const void *storage,
                                                size_t bytes) {
  const et_i2_copy_builder *builder;
  for (builder = et_i2_live_copy_builders; builder != NULL;
       builder = builder->registry_next) {
    if (builder->assignments != NULL &&
        builder->count <= SIZE_MAX / sizeof(*builder->assignments) &&
        et_tr3_c_i2_ranges_overlap(
            storage, bytes, builder->assignments,
            builder->count * sizeof(*builder->assignments))) {
      return 1;
    }
  }
  return 0;
}

static int32_t et_tr3_c_i2_restore_builder_live(
    const void *opaque, et_i2_copy_builder **output) {
  et_i2_copy_builder *builder = et_i2_find_copy_builder(opaque);
  if (builder == NULL || builder->tr3_c_restore_phase ==
                             ET_TR3_C_I2_RESTORE_PHASE_ORDINARY) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
  }
  if (builder->count != ET_TR3_C_RESTORE_ASSIGNMENTS ||
      builder->assignments == NULL) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
  }
  *output = builder;
  return ET_I2_PRIVATE_OWNED_RESULT_OK;
}

static int32_t et_tr3_c_i2_restore_authority(
    const et_i2_copy_builder *builder, const void *authority) {
  if (authority == NULL) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
  }
  return builder->tr3_c_restore_authority == authority
             ? ET_I2_PRIVATE_OWNED_RESULT_OK
             : ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT;
}

static int32_t et_tr3_c_i2_parameter_destination(
    et_f32_parameter *parameter, const void *p1_handle,
    et_f32_tensor **destination) {
  const et_f32_tensor *canonical;
  if (parameter == NULL || p1_handle == NULL ||
      et_f32_parameter_is_live_v1(parameter) != 1) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
  }
  if (et_f32_parameter_validate_identity_v1(parameter, p1_handle, NULL) != 0) {
    return ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT;
  }
  canonical = et_f32_parameter_canonical_owner_v1(parameter);
  if (canonical == NULL) {
    return ET_I2_PRIVATE_OWNED_RESULT_INTERNAL;
  }
  *destination = (et_f32_tensor *)canonical;
  return ET_I2_PRIVATE_OWNED_RESULT_OK;
}

static int32_t et_tr3_c_i2_snapshot_restore42_request(
    const et_i2_copy_builder *builder,
    const et_tr3_c_i2_restore42_append_v1 *request,
    et_tr3_c_i2_restore42_append_v1 *snapshot) {
  if (request == NULL ||
      (uintptr_t)request % _Alignof(et_tr3_c_i2_restore42_append_v1) != 0u ||
      !et_tr3_c_i2_span_fits(request, sizeof(*request)) ||
      et_tr3_c_i2_ranges_overlap(request, sizeof(*request), builder,
                                 sizeof(*builder)) ||
      et_tr3_c_i2_ranges_overlap(
          request, sizeof(*request), builder->assignments,
          builder->count * sizeof(*builder->assignments))) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
  }
  memcpy(snapshot, request, sizeof(*snapshot));
  return ET_I2_PRIVATE_OWNED_RESULT_OK;
}

static int32_t et_tr3_c_i2_validate_restore42(
    et_i2_copy_builder *builder,
    const et_tr3_c_i2_restore42_append_v1 *request, int complete) {
  et_f32_tensor *destinations[ET_TR3_C_RESTORE_ASSIGNMENTS];
  const et_f32_tensor *sources[ET_TR3_C_RESTORE_ASSIGNMENTS];
  int32_t result;

  if (request->struct_size != ET_TR3_C_I2_RESTORE42_APPEND_V1_0_SIZE) {
    return ET_I2_PRIVATE_OWNED_RESULT_VERSION_MISMATCH;
  }
  if ((result = et_tr3_c_i2_restore_authority(
           builder, request->restore_authority)) != 0) {
    return result;
  }
  if (builder->plan != NULL ||
      builder->tr3_c_restore_phase !=
          (complete != 0 ? ET_TR3_C_I2_RESTORE_PHASE_COMPLETE
                         : ET_TR3_C_I2_RESTORE_PHASE_PREFIX) ||
      builder->tr3_c_next_index !=
          (complete != 0 ? ET_TR3_C_RESTORE_ASSIGNMENTS
                         : ET_TR3_C_MODEL_DESTINATIONS)) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE;
  }
  for (size_t index = 0u; index < ET_TR3_C_MODEL_DESTINATIONS; ++index) {
    et_f32_tensor *destination = NULL;
    const et_f32_tensor_copy_assignment_v1 *assignment =
        &builder->assignments[index];
    result = et_tr3_c_i2_parameter_destination(
        request->parameters[index], request->p1_handles[index], &destination);
    if (result != 0) {
      return result;
    }
    if (assignment->struct_size !=
            ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE ||
        assignment->destination != destination || assignment->source == NULL) {
      return ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT;
    }
    result = et_i2_private_owned_clone_match_v1(
        assignment->source, destination, ET_I2_PRIVATE_OWNED_VALUE_FINITE);
    if (result != 0) {
      return result;
    }
    destinations[index] = destination;
    sources[index] = assignment->source;
  }
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    const size_t assignment_index = ET_TR3_C_O2_FIRST_INDEX + index;
    const uint32_t policy =
        index % 2u == 0u ? ET_I2_PRIVATE_OWNED_VALUE_FINITE
                         : ET_I2_PRIVATE_OWNED_VALUE_FINITE_NONNEGATIVE;
    const et_f32_tensor_copy_assignment_v1 *assignment =
        &builder->assignments[assignment_index];
    et_f32_tensor *destination = request->moment_destinations[index];
    const et_f32_tensor *source = request->moment_sources[index];
    if (destination == NULL || source == NULL) {
      return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
    }
    if (complete != 0) {
      if (assignment->struct_size !=
              ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE ||
          assignment->destination != destination ||
          assignment->source != source) {
        return ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT;
      }
    } else if (assignment->struct_size != 0u ||
               assignment->destination != NULL || assignment->source != NULL) {
      return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
    }
    result = et_i2_private_owned_clone_match_v1(source, destination, policy);
    if (result != 0) {
      return result;
    }
    destinations[assignment_index] = destination;
    sources[assignment_index] = source;
  }
  for (size_t index = 0u; index < ET_TR3_C_RESTORE_ASSIGNMENTS; ++index) {
    if (destinations[index] == sources[index]) {
      return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
    }
    for (size_t other = 0u; other < ET_TR3_C_RESTORE_ASSIGNMENTS; ++other) {
      if (index != other &&
          (destinations[index] == destinations[other] ||
           sources[index] == sources[other])) {
        return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
      }
      if (destinations[index] == sources[other]) {
        return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
      }
    }
  }
  return ET_I2_PRIVATE_OWNED_RESULT_OK;
}

static int32_t et_tr3_c_i2_translate_copy_prepare(
    int32_t status, const et_f32_tensor_error *error) {
  if (status == 0) {
    return ET_I2_PRIVATE_OWNED_RESULT_OK;
  }
  if (error->code == ET_F32_TENSOR_CODE_ALLOCATION_FAILED) {
    return ET_I2_PRIVATE_OWNED_RESULT_ALLOCATION_FAILED;
  }
  if (error->category == ET_F32_TENSOR_ERROR_INVALID_STATE) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE;
  }
  if (error->category == ET_F32_TENSOR_ERROR_SHAPE_MISMATCH ||
      error->code == ET_F32_TENSOR_CODE_BUFFER_SIZE_MISMATCH ||
      error->code == ET_F32_TENSOR_CODE_INVALID_SHAPE) {
    return ET_I2_PRIVATE_OWNED_RESULT_SHAPE_MISMATCH;
  }
  if (error->category == ET_F32_TENSOR_ERROR_VERSION_MISMATCH ||
      error->code == ET_F32_TENSOR_CODE_ABI_VERSION_MISMATCH) {
    return ET_I2_PRIVATE_OWNED_RESULT_VERSION_MISMATCH;
  }
  return ET_I2_PRIVATE_OWNED_RESULT_INTERNAL;
}

static void et_tr3_c_i2_retire_restore_builder(
    et_i2_copy_builder *builder, uint32_t terminal_phase) {
  et_i2_unlink_copy_builder(builder);
  free(builder->assignments);
  builder->count = 0u;
  builder->assignments = NULL;
  builder->plan = NULL;
  builder->tr3_c_restore_authority = NULL;
  builder->tr3_c_next_index = 0u;
  builder->tr3_c_restore_phase = terminal_phase;
  builder->registry_next = et_i2_retired_copy_builders;
  et_i2_retired_copy_builders = builder;
}

int32_t et_tr3_c_i2_copy_builder_create_restore42_v1(
    const void *restore_authority, void **output) {
  et_i2_copy_builder *builder = NULL;
  if (restore_authority == NULL || output == NULL ||
      (uintptr_t)output % _Alignof(void *) != 0u ||
      !et_tr3_c_i2_span_fits(output, sizeof(*output)) ||
      (const void *)output == restore_authority ||
      et_i2_builder_shell_aliases(output, sizeof(*output)) ||
      et_tr3_c_i2_copy_assignments_overlap(output, sizeof(*output))) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
  }
  if (*output != NULL) {
    return ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT;
  }
  builder = (et_i2_copy_builder *)et_i2_system_calloc(1u, sizeof(*builder));
  if (builder != NULL) {
    builder->assignments = (et_f32_tensor_copy_assignment_v1 *)
        et_i2_system_calloc(ET_TR3_C_RESTORE_ASSIGNMENTS,
                            sizeof(*builder->assignments));
  }
  if (builder == NULL || builder->assignments == NULL) {
    if (builder != NULL) {
      free(builder->assignments);
    }
    free(builder);
    return ET_I2_PRIVATE_OWNED_RESULT_ALLOCATION_FAILED;
  }
  builder->magic = ET_I2_COPY_BUILDER_MAGIC;
  builder->count = ET_TR3_C_RESTORE_ASSIGNMENTS;
  builder->tr3_c_restore_authority = restore_authority;
  builder->tr3_c_restore_phase = ET_TR3_C_I2_RESTORE_PHASE_PREFIX;
  builder->registry_next = et_i2_live_copy_builders;
  et_i2_live_copy_builders = builder;
  *output = builder;
  return ET_I2_PRIVATE_OWNED_RESULT_OK;
}

int32_t et_tr3_c_i2_copy_builder_append_parameter_v1(
    void *opaque, const void *restore_authority, et_f32_parameter *parameter,
    const void *p1_handle, const et_f32_tensor *owned_source) {
  et_i2_copy_builder *builder = NULL;
  et_f32_tensor *destination = NULL;
  int32_t result = et_tr3_c_i2_restore_builder_live(opaque, &builder);
  if (result != 0) {
    return result;
  }
  if ((result = et_tr3_c_i2_restore_authority(builder, restore_authority)) !=
      0) {
    return result;
  }
  if (builder->tr3_c_restore_phase != ET_TR3_C_I2_RESTORE_PHASE_PREFIX ||
      builder->tr3_c_next_index >= ET_TR3_C_MODEL_DESTINATIONS ||
      builder->plan != NULL) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE;
  }
  if ((result = et_tr3_c_i2_parameter_destination(parameter, p1_handle,
                                                   &destination)) != 0) {
    return result;
  }
  if ((result = et_i2_private_owned_clone_match_v1(
           owned_source, destination,
           ET_I2_PRIVATE_OWNED_VALUE_FINITE)) != 0) {
    return result;
  }
  if (owned_source == destination) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
  }
  for (size_t index = 0u; index < builder->tr3_c_next_index; ++index) {
    const et_f32_tensor_copy_assignment_v1 *prior =
        &builder->assignments[index];
    if (prior->destination == destination || prior->source == owned_source ||
        prior->destination == owned_source || prior->source == destination) {
      return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
    }
  }
  builder->assignments[builder->tr3_c_next_index].struct_size =
      ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
  builder->assignments[builder->tr3_c_next_index].destination = destination;
  builder->assignments[builder->tr3_c_next_index].source = owned_source;
  builder->tr3_c_next_index++;
  return ET_I2_PRIVATE_OWNED_RESULT_OK;
}

int32_t et_tr3_c_i2_copy_builder_append_restore42_v1(
    void *opaque, const et_tr3_c_i2_restore42_append_v1 *request) {
  et_i2_copy_builder *builder = NULL;
  et_tr3_c_i2_restore42_append_v1 snapshot;
  int32_t result = et_tr3_c_i2_restore_builder_live(opaque, &builder);
  if (result != 0) {
    return result;
  }
  result = et_tr3_c_i2_snapshot_restore42_request(builder, request, &snapshot);
  if (result != 0) {
    return result;
  }
  result = et_tr3_c_i2_validate_restore42(builder, &snapshot, 0);
  if (result != 0) {
    return result;
  }
  for (size_t index = 0u; index < ET_TR3_C_O2_MOMENT_ASSIGNMENTS; ++index) {
    const size_t assignment_index = ET_TR3_C_O2_FIRST_INDEX + index;
    builder->assignments[assignment_index].struct_size =
        ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE;
    builder->assignments[assignment_index].destination =
        snapshot.moment_destinations[index];
    builder->assignments[assignment_index].source =
        snapshot.moment_sources[index];
  }
  builder->tr3_c_next_index = ET_TR3_C_RESTORE_ASSIGNMENTS;
  builder->tr3_c_restore_phase = ET_TR3_C_I2_RESTORE_PHASE_COMPLETE;
  return ET_I2_PRIVATE_OWNED_RESULT_OK;
}

int32_t et_tr3_c_i2_copy_builder_verify_restore42_v1(
    const void *opaque,
    const et_tr3_c_i2_restore42_append_v1 *request) {
  et_i2_copy_builder *builder = NULL;
  et_tr3_c_i2_restore42_append_v1 snapshot;
  int32_t result = et_tr3_c_i2_restore_builder_live(opaque, &builder);
  if (result != 0) {
    return result;
  }
  result = et_tr3_c_i2_snapshot_restore42_request(builder, request, &snapshot);
  return result == 0
             ? et_tr3_c_i2_validate_restore42(builder, &snapshot, 1)
             : result;
}

int32_t et_tr3_c_i2_copy_builder_prepare_restore42_v1(
    void *opaque, const void *restore_authority) {
  et_i2_copy_builder *builder = NULL;
  et_f32_tensor_copy_plan *plan = NULL;
  et_f32_tensor_error error;
  int32_t status;
  int32_t result = et_tr3_c_i2_restore_builder_live(opaque, &builder);
  if (result != 0) {
    return result;
  }
  if ((result = et_tr3_c_i2_restore_authority(builder, restore_authority)) !=
      0) {
    return result;
  }
  if (builder->tr3_c_restore_phase != ET_TR3_C_I2_RESTORE_PHASE_COMPLETE ||
      builder->tr3_c_next_index != ET_TR3_C_RESTORE_ASSIGNMENTS ||
      builder->plan != NULL) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE;
  }
  memset(&error, 0, sizeof(error));
  status = et_f32_tensor_copy_plan_prepare_v1(
      builder->count, builder->assignments, &plan, &error);
  if (status != 0) {
    if (plan != NULL &&
        et_f32_tensor_copy_plan_release_v1(&plan, NULL) != 0) {
      et_tr3_c_i2_fail_stop();
    }
    return et_tr3_c_i2_translate_copy_prepare(status, &error);
  }
  if (plan == NULL) {
    et_tr3_c_i2_fail_stop();
  }
  builder->plan = plan;
  builder->tr3_c_restore_phase = ET_TR3_C_I2_RESTORE_PHASE_PREPARED;
  return ET_I2_PRIVATE_OWNED_RESULT_OK;
}

int32_t et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(
    void *opaque, const void *restore_authority) {
  et_i2_copy_builder *builder = NULL;
  int32_t status;
  int32_t result = et_tr3_c_i2_restore_builder_live(opaque, &builder);
  if (result != 0) {
    return result;
  }
  if ((result = et_tr3_c_i2_restore_authority(builder, restore_authority)) !=
      0) {
    return result;
  }
  if (builder->tr3_c_restore_phase != ET_TR3_C_I2_RESTORE_PHASE_PREPARED ||
      builder->tr3_c_next_index != ET_TR3_C_RESTORE_ASSIGNMENTS ||
      builder->plan == NULL) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE;
  }
  et_i2_clear_error();
  status = et_tr3_c_i2_copy_commit(builder->plan);
  if (status != 0) {
    et_tr3_c_i2_fail_stop();
  }
  status = et_tr3_c_i2_copy_release(&builder->plan);
  if (status != 0 || builder->plan != NULL) {
    et_tr3_c_i2_fail_stop();
  }
  et_tr3_c_i2_retire_restore_builder(
      builder, ET_TR3_C_I2_RESTORE_PHASE_COMMITTED);
  return ET_I2_PRIVATE_OWNED_RESULT_OK;
}

int32_t et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
    void *opaque, const void *restore_authority) {
  et_i2_copy_builder *builder = NULL;
  int32_t status;
  int32_t result = et_tr3_c_i2_restore_builder_live(opaque, &builder);
  if (result != 0) {
    return result;
  }
  if ((result = et_tr3_c_i2_restore_authority(builder, restore_authority)) !=
      0) {
    return result;
  }
  if (!((builder->tr3_c_restore_phase ==
             ET_TR3_C_I2_RESTORE_PHASE_PREFIX &&
         builder->tr3_c_next_index <= ET_TR3_C_MODEL_DESTINATIONS &&
         builder->plan == NULL) ||
        (builder->tr3_c_restore_phase ==
             ET_TR3_C_I2_RESTORE_PHASE_COMPLETE &&
         builder->tr3_c_next_index == ET_TR3_C_RESTORE_ASSIGNMENTS &&
         builder->plan == NULL) ||
        (builder->tr3_c_restore_phase ==
             ET_TR3_C_I2_RESTORE_PHASE_PREPARED &&
         builder->tr3_c_next_index == ET_TR3_C_RESTORE_ASSIGNMENTS &&
         builder->plan != NULL))) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE;
  }
  if (builder->plan != NULL) {
    et_i2_clear_error();
    status = et_tr3_c_i2_copy_release(&builder->plan);
    if (status != 0 || builder->plan != NULL) {
      et_tr3_c_i2_fail_stop();
    }
  }
  et_tr3_c_i2_retire_restore_builder(builder,
                                     ET_TR3_C_I2_RESTORE_PHASE_ABORTED);
  return ET_I2_PRIVATE_OWNED_RESULT_OK;
}

int32_t et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
    const void *opaque, uint32_t expected_terminal) {
  const et_i2_copy_builder *builder = et_i2_retired_copy_builders;
  uint32_t expected_phase;
  if (expected_terminal == ET_TR3_C_I2_TERMINAL_COMMITTED) {
    expected_phase = ET_TR3_C_I2_RESTORE_PHASE_COMMITTED;
  } else if (expected_terminal == ET_TR3_C_I2_TERMINAL_ABORTED) {
    expected_phase = ET_TR3_C_I2_RESTORE_PHASE_ABORTED;
  } else {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
  }
  while (builder != NULL && (const void *)builder != opaque) {
    builder = builder->registry_next;
  }
  if (builder == NULL ||
      (builder->tr3_c_restore_phase !=
           ET_TR3_C_I2_RESTORE_PHASE_COMMITTED &&
       builder->tr3_c_restore_phase != ET_TR3_C_I2_RESTORE_PHASE_ABORTED)) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT;
  }
  if (builder->tr3_c_restore_phase != expected_phase) {
    return ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE;
  }
  return builder->magic == 0u && builder->count == 0u &&
                 builder->assignments == NULL && builder->plan == NULL &&
                 builder->tr3_c_restore_authority == NULL &&
                 builder->tr3_c_next_index == 0u
             ? ET_I2_PRIVATE_OWNED_RESULT_OK
             : ET_I2_PRIVATE_OWNED_RESULT_INTERNAL;
}
#endif

int64_t et_tr3_c_i2_copy_builder_commit_checked_v1(void *opaque) {
  et_i2_copy_builder *const builder = et_i2_find_copy_builder(opaque);
  int32_t status;

  if (builder == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#ifdef ET_TR3_C_I2_RESTORE_PRIVATE
  if (builder->tr3_c_restore_phase != ET_TR3_C_I2_RESTORE_PHASE_ORDINARY) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#endif
  if (builder->plan == NULL) {
    et_tr3_c_i2_fail_stop();
  }
  et_i2_clear_error();
  status = et_tr3_c_i2_copy_commit(builder->plan);
  if (status != 0) {
    et_tr3_c_i2_fail_stop();
  }
  status = et_tr3_c_i2_copy_release(&builder->plan);
  if (status != 0 || builder->plan != NULL) {
    et_tr3_c_i2_fail_stop();
  }
  et_i2_unlink_copy_builder(builder);
  free(builder->assignments);
  et_i2_retire_copy_builder(builder);
  return 0;
}

int64_t et_tr3_c_i2_copy_builder_abort_checked_v1(void *opaque) {
  et_i2_copy_builder *const builder = et_i2_find_copy_builder(opaque);
  int32_t status;

  if (builder == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#ifdef ET_TR3_C_I2_RESTORE_PRIVATE
  if (builder->tr3_c_restore_phase != ET_TR3_C_I2_RESTORE_PHASE_ORDINARY) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
#endif
  if (builder->plan != NULL) {
    et_i2_clear_error();
    status = et_tr3_c_i2_copy_release(&builder->plan);
    if (status != 0 || builder->plan != NULL) {
      et_tr3_c_i2_fail_stop();
    }
  }
  et_i2_unlink_copy_builder(builder);
  free(builder->assignments);
  et_i2_retire_copy_builder(builder);
  return 0;
}

void et_tr3_c_i2_owned_release_checked_v1(void *opaque) {
  et_f32_tensor *const owned = (et_f32_tensor *)opaque;
  int32_t status;

  if (et_f32_tensor_canonical_owner_v1(owned) != owned) {
    et_tr3_c_i2_fail_stop();
  }
  et_i2_clear_error();
#ifdef ET_F32_TENSOR_TESTING
  if (et_tr3_c_i2_fail_owned_release) {
    et_tr3_c_i2_fail_stop();
  }
#endif
  status = (int32_t)et_i2_private_owned_release_v1(owned);
  if (status != 0) {
    et_tr3_c_i2_fail_stop();
  }
}
#endif

void *et_i2_private_reset_builder_create_v1(int64_t count) {
  et_i2_reset_builder *builder;
  if (count <= 0 || count > (int64_t)ET_F32_PARAMETER_MAX_BATCH) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_SHAPE);
    return NULL;
  }
  builder = (et_i2_reset_builder *)et_i2_system_calloc(1u, sizeof(*builder));
  if (builder != NULL) {
    builder->parameters = (et_f32_parameter **)et_i2_system_calloc(
        (size_t)count, sizeof(*builder->parameters));
  }
  if (builder == NULL || builder->parameters == NULL) {
    free(builder);
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INTERNAL,
                           ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    return NULL;
  }
  builder->count = (size_t)count;
  builder->magic = ET_I2_RESET_BUILDER_MAGIC;
  builder->registry_next = et_i2_live_reset_builders;
  et_i2_live_reset_builders = builder;
  et_i2_clear_error();
  return builder;
}

int64_t et_i2_private_reset_builder_set_v1(void *opaque, int64_t index,
                                           void *carrier) {
  et_i2_reset_builder *const builder = et_i2_find_reset_builder(opaque);
  if (builder == NULL || carrier == NULL || builder->plan != NULL ||
      index < 0 || (size_t)index >= builder->count) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
  builder->parameters[index] = (et_f32_parameter *)carrier;
  return 0;
}

int64_t et_i2_private_reset_builder_prepare_v1(void *opaque) {
  et_i2_reset_builder *const builder = et_i2_find_reset_builder(opaque);
  size_t index;
  if (builder == NULL || builder->plan != NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
  for (index = 0u; index < builder->count; ++index) {
    if (builder->parameters[index] == NULL) {
      et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                             ET_F32_TENSOR_CODE_INVALID_HANDLE);
      return -1;
    }
  }
  et_i2_clear_error();
  return et_f32_gradient_reset_plan_prepare_v1(
      builder->count, builder->parameters, &builder->plan,
      &et_i2_last_error);
}

int64_t et_i2_private_reset_builder_commit_v1(void *opaque) {
  et_i2_reset_builder *const builder = et_i2_find_reset_builder(opaque);
  int32_t status;
  if (builder == NULL || builder->plan == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
  et_i2_clear_error();
  status = et_f32_gradient_reset_plan_commit_v1(builder->plan,
                                                &et_i2_last_error);
  if (status == 0) {
    (void)et_f32_gradient_reset_plan_release_v1(&builder->plan,
                                                &et_i2_last_error);
    et_i2_unlink_reset_builder(builder);
    free(builder->parameters);
    et_i2_retire_reset_builder(builder);
  }
  return status;
}

int64_t et_i2_private_reset_builder_abort_v1(void *opaque) {
  et_i2_reset_builder *const builder = et_i2_find_reset_builder(opaque);
  if (opaque == NULL) {
    return 0;
  }
  if (builder == NULL) {
    et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                           ET_F32_TENSOR_CODE_INVALID_HANDLE);
    return -1;
  }
  if (builder->plan != NULL) {
    (void)et_f32_gradient_reset_plan_release_v1(&builder->plan,
                                                &et_i2_last_error);
  }
  et_i2_unlink_reset_builder(builder);
  free(builder->parameters);
  et_i2_retire_reset_builder(builder);
  return 0;
}

int64_t et_i2_private_parameter_accumulate_uniform_v1(
    void *carrier, void *p1_handle, int64_t bits, int64_t expected_ordinal,
    int64_t weight_bits) {
  et_f32_parameter *const parameter = (et_f32_parameter *)carrier;
  const et_f32_tensor *value = NULL;
  et_f32_tensor *contribution_tensor = NULL;
  et_f32_gradient_plan *plan = NULL;
  et_f32_gradient_contribution_v1 contribution;
  uint32_t *payload = NULL;
  size_t count = 0u;
  size_t index;
  int32_t status = -1;
  et_f32_tensor_error saved_error;
  if (bits < 0 || (uint64_t)bits > UINT32_MAX || expected_ordinal < 0 ||
      weight_bits < 0 || (uint64_t)weight_bits > UINT32_MAX ||
      et_f32_parameter_validate_identity_v1(parameter, p1_handle,
                                            &et_i2_last_error) != 0 ||
      et_f32_parameter_value_tensor_v1(parameter, &value,
                                       &et_i2_last_error) != 0 ||
      et_f32_tensor_clone_v1(value, &contribution_tensor,
                             &et_i2_last_error) != 0 ||
      et_f32_tensor_element_count_v1(contribution_tensor, &count,
                                     &et_i2_last_error) != 0) {
    goto cleanup;
  }
  if (count > 0u) {
    payload = (uint32_t *)et_i2_system_calloc(count, sizeof(*payload));
    if (payload == NULL) {
      et_i2_set_bridge_error(ET_F32_TENSOR_ERROR_INTERNAL,
                             ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
      goto cleanup;
    }
    for (index = 0u; index < count; ++index) {
      payload[index] = (uint32_t)bits;
    }
    if (et_f32_tensor_copy_bits_from_v1(contribution_tensor, payload, count,
                                        &et_i2_last_error) != 0) {
      goto cleanup;
    }
  }
  contribution.struct_size = sizeof(contribution);
  contribution.destination = parameter;
  contribution.weighted_numerator = contribution_tensor;
  contribution.expected_ordinal = (uint64_t)expected_ordinal;
  if (et_f32_gradient_plan_prepare_v1(
          1u, &contribution, (uint32_t)weight_bits, &plan,
          &et_i2_last_error) != 0) {
    goto cleanup;
  }
  status = et_f32_gradient_plan_commit_v1(plan, &et_i2_last_error);

cleanup:
  saved_error = et_i2_last_error;
  free(payload);
  if (plan != NULL) {
    (void)et_f32_gradient_plan_release_v1(&plan, NULL);
  }
  if (contribution_tensor != NULL) {
    (void)et_f32_tensor_destroy_v1(&contribution_tensor, NULL);
  }
  if (status != 0) {
    et_i2_last_error = saved_error;
  }
  return status;
}

int64_t et_i2_private_parameter_gradient_state_v1(void *carrier,
                                                   void *p1_handle) {
  et_f32_gradient_metadata_v1 metadata;
  metadata.struct_size = sizeof(metadata);
  if (et_f32_parameter_validate_identity_v1(
          (const et_f32_parameter *)carrier, p1_handle,
          &et_i2_last_error) != 0 ||
      et_f32_parameter_gradient_metadata_v1(
          (const et_f32_parameter *)carrier, &metadata,
          &et_i2_last_error) != 0) {
    return -1;
  }
  return (int64_t)metadata.state;
}

int64_t et_i2_private_parameter_gradient_count_v1(void *carrier,
                                                   void *p1_handle) {
  et_f32_gradient_metadata_v1 metadata;
  metadata.struct_size = sizeof(metadata);
  if (et_f32_parameter_validate_identity_v1(
          (const et_f32_parameter *)carrier, p1_handle,
          &et_i2_last_error) != 0 ||
      et_f32_parameter_gradient_metadata_v1(
          (const et_f32_parameter *)carrier, &metadata,
          &et_i2_last_error) != 0 ||
      metadata.contribution_count > INT64_MAX) {
    return -1;
  }
  return (int64_t)metadata.contribution_count;
}
