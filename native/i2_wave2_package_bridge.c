#include <stddef.h>

/* Wave 2 extends the accepted source-composed Wave 1 bridge. The fixed native
 * I2 helper subset is also compiled without package wrappers by the trusted
 * source-level integration test. */
#ifndef ET_I2_NATIVE_HELPERS_ONLY
#include "e1b_error_consumer_bridge.h"
#include "t1_wave1_package_bridge.c"
#endif

#include "f32_parameter_internal.h"

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

typedef struct et_i2_copy_builder {
  uint64_t magic;
  struct et_i2_copy_builder *registry_next;
  size_t count;
  et_f32_tensor_copy_assignment_v1 *assignments;
  et_f32_tensor_copy_plan *plan;
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
      sizeof(builder->assignments[index]);
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
  for (index = 0u; index < builder->count; ++index) {
    if (builder->assignments[index].struct_size !=
        sizeof(builder->assignments[index])) {
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
  if (builder->plan != NULL) {
    (void)et_f32_tensor_copy_plan_release_v1(&builder->plan,
                                             &et_i2_last_error);
  }
  et_i2_unlink_copy_builder(builder);
  free(builder->assignments);
  et_i2_retire_copy_builder(builder);
  return 0;
}

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
