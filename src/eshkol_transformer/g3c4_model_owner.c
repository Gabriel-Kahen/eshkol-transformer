#ifndef ET_G3C4_NATIVE_OWNER_PRIVATE
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#endif

#include "g3c4_model_owner_internal.h"
#include "m3t_f32_scoped.h"
#include "eshkol_transformer/i64_tensor.h"
#include "eshkol_transformer/n3k_primitives_abi.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
  ET_G3C4_OWNER_OPEN = 0,
  ET_G3C4_OWNER_PREPARED = 1,
  ET_G3C4_OWNER_SEALED = 2,
  ET_G3C4_OWNER_ABORTED = 3
};

#define ET_G3C4_OWNER_MAGIC UINT64_C(0x473343344f574e31)

typedef struct et_g3c4_model_owner_internal {
  uint64_t magic;
  struct et_g3c4_model_owner_internal *registry_next;
  int state;
  size_t initialized;
  int successor_ready;
  int64_t original[4];
  int64_t successor[4];
  et_f32_parameter *parameters[14];
  const void *handles[14];
  void *active;
} et_g3c4_model_owner_internal;

static _Thread_local et_g3c4_error_state_internal et_g3c4_error_state;
static et_g3c4_model_owner_internal *et_g3c4_owner_registry;
static et_g3c4_model_owner_internal *staged_c4_owner;
static et_kernel_runtime *et_g3c4_initializer_runtime;

static const uint64_t et_g3c4_parameter_shapes[14][2] = {
  {4,4}, {4,4}, {4,4}, {4,4}, {4,8}, {8,4}, {4,0},
  {4,0}, {4,0}, {4,0}, {256,4}, {4,0}, {4,0}, {4,4}
};

#ifdef ET_G3C4_NATIVE_OWNER_TESTING
static size_t et_g3c4_allocation_limit = SIZE_MAX;
static size_t et_g3c4_successful_allocations;
#endif

static int et_g3c4_valid_error_triple(
    int64_t domain, int64_t category, int64_t code) {
  int64_t max_category;
  int64_t max_code;
  switch (domain) {
    case ET_G3C4_DOMAIN_C4:
      max_category = ET_G3C4_DETERMINISM_UNAVAILABLE;
      max_code = ET_G3C4_CODE_READINESS;
      break;
    case ET_G3C4_DOMAIN_K1:
      max_category = ET_KERNEL_ERROR_INTERNAL;
      max_code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      break;
    case ET_G3C4_DOMAIN_I1:
      max_category = ET_I64_TENSOR_ERROR_INTERNAL;
      max_code = ET_I64_TENSOR_CODE_PROVIDER_REJECTED;
      break;
    case ET_G3C4_DOMAIN_I2:
      max_category = ET_F32_TENSOR_ERROR_INTERNAL;
      max_code = ET_F32_TENSOR_CODE_FLOAT_ENVIRONMENT;
      break;
    default:
      return 0;
  }
  if (category < 0 || category > max_category || code < 0 || code > max_code)
    return 0;
  return (category == 0) == (code == 0);
}

void et_g3c4_error_reset_internal(void) {
  memset(&et_g3c4_error_state, 0, sizeof(et_g3c4_error_state));
}

void et_g3c4_error_set_internal(
    int64_t domain, int64_t category, int64_t code) {
  if (!et_g3c4_valid_error_triple(domain, category, code)) {
    et_g3c4_error_state = (et_g3c4_error_state_internal){
      ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT};
    return;
  }
  et_g3c4_error_state =
      (et_g3c4_error_state_internal){domain, category, code};
}

et_g3c4_error_state_internal et_g3c4_error_snapshot_internal(void) {
  return et_g3c4_error_state;
}

void et_g3c4_error_restore_internal(et_g3c4_error_state_internal snapshot) {
  et_g3c4_error_set_internal(snapshot.domain, snapshot.category, snapshot.code);
}

int64_t et_g3c4_private_last_error_domain_v1(void) {
  return et_g3c4_error_state.domain;
}

int64_t et_g3c4_private_last_error_category_v1(void) {
  return et_g3c4_error_state.category;
}

int64_t et_g3c4_private_last_error_code_v1(void) {
  return et_g3c4_error_state.code;
}

static int64_t et_g3c4_fail(int64_t category, int64_t code) {
  et_g3c4_error_set_internal(ET_G3C4_DOMAIN_C4, category, code);
  return category;
}

static int32_t et_g3c4_capture_f32(
    int32_t status, const et_f32_tensor_error *error) {
  if (status != 0) {
    if (error == NULL) {
      et_g3c4_error_set_internal(-1, -1, -1);
    } else {
      et_g3c4_error_set_internal(
          ET_G3C4_DOMAIN_I2, error->category, error->code);
    }
  }
  return status;
}

static int32_t et_g3c4_capture_kernel(
    int32_t status, const et_kernel_error *error) {
  if (status != 0) {
    if (error == NULL) {
      et_g3c4_error_set_internal(-1, -1, -1);
    } else {
      et_g3c4_error_set_internal(
          ET_G3C4_DOMAIN_K1, error->category, error->code);
    }
  }
  return status;
}

static void *et_g3c4_allocate(size_t size) {
#ifdef ET_G3C4_NATIVE_OWNER_TESTING
  if (et_g3c4_successful_allocations >= et_g3c4_allocation_limit) return NULL;
#endif
  void *allocation = calloc(1u, size);
#ifdef ET_G3C4_NATIVE_OWNER_TESTING
  if (allocation != NULL) et_g3c4_successful_allocations++;
#endif
  return allocation;
}

static size_t et_g3c4_parameter_rank(size_t index) {
  return et_g3c4_parameter_shapes[index][1] == 0u ? 1u : 2u;
}

static size_t et_g3c4_parameter_elements(size_t index) {
  return (size_t)et_g3c4_parameter_shapes[index][0] *
         (et_g3c4_parameter_rank(index) == 1u
              ? 1u : (size_t)et_g3c4_parameter_shapes[index][1]);
}

static et_g3c4_model_owner_internal *et_g3c4_admit_owner(
    const void *candidate, int allow_aborted) {
  et_g3c4_model_owner_internal *owner;
  for (owner = et_g3c4_owner_registry;
       owner != NULL && (const void *)owner != candidate;
       owner = owner->registry_next) {}
  if (owner == NULL || owner->magic != ET_G3C4_OWNER_MAGIC) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
  if (owner->state == ET_G3C4_OWNER_ABORTED && !allow_aborted) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
  return owner;
}

static const et_kernel_provider_v1 *et_g3c4_resolve_provider(
    void *context, const char *symbol) {
  return symbol != NULL && strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? (const et_kernel_provider_v1 *)context : NULL;
}

static int et_g3c4_require_initializer_row(
    et_kernel_runtime *runtime, size_t index) {
  et_kernel_request_v1 request = {
    sizeof(request), "n3k.matrix-init.uniform", "f32", "cpu", 2u,
    et_g3c4_parameter_shapes[index], 1u, {0}};
  const et_kernel_capability_v1 *entry = NULL;
  et_kernel_error error;
  return et_g3c4_capture_kernel(
      et_kernel_runtime_capability_require(
          runtime, "n3k.matrix-init", &request,
          &entry, &error), &error);
}

static int et_g3c4_discover_initializer(
    et_kernel_runtime **runtime_output, int *staged_output) {
  et_kernel_runtime *runtime = et_g3c4_initializer_runtime;
  int staged = 0;
  if (runtime == NULL) {
    et_kernel_error error;
    const et_kernel_provider_v1 *provider = et_n3k_kernel_provider_v1();
    if (et_g3c4_capture_kernel(
            et_kernel_runtime_discover(et_g3c4_resolve_provider,
                                       (void *)provider, &runtime, &error),
            &error))
      return (int)et_g3c4_error_state.category;
    staged = 1;
  }
  for (size_t index = 0u; index < 14u; index++) {
    if (et_g3c4_parameter_rank(index) == 2u &&
        et_g3c4_require_initializer_row(runtime, index)) {
      if (staged) et_kernel_runtime_destroy(runtime);
      return (int)et_g3c4_error_state.category;
    }
  }
  *runtime_output = runtime;
  *staged_output = staged;
  return 0;
}

static et_kernel_tensor_view_v1 et_g3c4_view(
    void *data, size_t bytes, const char *dtype,
    size_t rank, const uint64_t *shape) {
  return (et_kernel_tensor_view_v1){
    sizeof(et_kernel_tensor_view_v1), data, bytes, dtype, "cpu",
    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, rank, shape};
}

static int et_g3c4_dispatch_initializer(
    et_g3c4_model_owner_internal *owner, size_t index,
    et_kernel_tensor_view_v1 output) {
  static const uint64_t state_shape[1] = {4u};
  int64_t next[4] = {0, 0, 0, 0};
  et_kernel_tensor_view_v1 inputs[1] = {
    et_g3c4_view(owner->successor, sizeof(owner->successor),
                 "i64", 1u, state_shape)};
  et_kernel_tensor_view_v1 outputs[2] = {
    output,
    et_g3c4_view(next, sizeof(next), "i64", 1u, state_shape)};
  et_kernel_request_v1 request = {
    sizeof(request), "n3k.matrix-init.uniform", "f32", "cpu", 2u,
    et_g3c4_parameter_shapes[index], 1u, {0}};
  et_kernel_call_v1 call = {
    sizeof(call), "n3k.matrix-init", &request,
    1u, sizeof(inputs[0]), sizeof(inputs), inputs,
    2u, sizeof(outputs[0]), sizeof(outputs), outputs};
  et_kernel_error error;
  int32_t status = et_kernel_runtime_dispatch(
      et_g3c4_initializer_runtime, &call, &error);
  if (et_g3c4_capture_kernel(status, &error)) return (int)status;
  memcpy(owner->successor, next, sizeof(next));
  return 0;
}

static et_f32_tensor *et_g3c4_parameter_value(
    et_g3c4_model_owner_internal *owner, size_t index) {
  const et_f32_tensor *value = NULL;
  et_f32_tensor_error error;
  if (et_g3c4_capture_f32(
          et_f32_parameter_value_tensor_v1(
              owner->parameters[index], &value, &error), &error))
    return NULL;
  return (et_f32_tensor *)value;
}

static int et_g3c4_parameter_shape_valid(
    et_g3c4_model_owner_internal *owner, size_t index) {
  et_f32_tensor *value = et_g3c4_parameter_value(owner, index);
  et_f32_tensor_error error;
  size_t rank = 0u;
  size_t elements = 0u;
  size_t bytes = 0u;
  if (value == NULL) return 0;
  if (et_g3c4_capture_f32(
          et_f32_tensor_rank_v1(value, &rank, &error), &error) ||
      rank != et_g3c4_parameter_rank(index))
    return 0;
  for (size_t dimension = 0u; dimension < rank; dimension++) {
    uint64_t extent = 0u;
    if (et_g3c4_capture_f32(
            et_f32_tensor_shape_at_v1(
                value, dimension, &extent, &error), &error) ||
        extent != et_g3c4_parameter_shapes[index][dimension])
      return 0;
  }
  if (et_g3c4_capture_f32(
          et_f32_tensor_element_count_v1(value, &elements, &error), &error) ||
      et_g3c4_capture_f32(
          et_f32_tensor_byte_length_v1(value, &bytes, &error), &error))
    return 0;
  return elements == et_g3c4_parameter_elements(index) &&
         bytes == elements * sizeof(float);
}

static int et_g3c4_parameter_value_finite(
    et_g3c4_model_owner_internal *owner, size_t index) {
  uint32_t bits[1024];
  const size_t count = et_g3c4_parameter_elements(index);
  et_f32_tensor *value = et_g3c4_parameter_value(owner, index);
  et_f32_tensor_error error;
  if (value == NULL || count > sizeof(bits) / sizeof(bits[0]) ||
      et_g3c4_capture_f32(
          et_f32_tensor_copy_bits_to_v1(value, bits, count, &error), &error))
    return 0;
  for (size_t element = 0u; element < count; element++) {
    if ((bits[element] & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
      (void)et_g3c4_fail(
          ET_G3C4_INVALID_STATE, ET_G3C4_CODE_READINESS);
      return 0;
    }
  }
  return 1;
}

static int32_t et_g3c4_preflight_error(
    et_f32_tensor_error *error, const char *message) {
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
    error->category = ET_F32_TENSOR_ERROR_INVALID_STATE;
    error->code = ET_F32_TENSOR_CODE_INVALID_HANDLE;
    (void)snprintf(error->operation, sizeof(error->operation), "%s",
                   "g3c4-construction-parameter-preflight");
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
  et_g3c4_error_set_internal(
      ET_G3C4_DOMAIN_I2, ET_F32_TENSOR_ERROR_INVALID_STATE,
      ET_F32_TENSOR_CODE_INVALID_HANDLE);
  return ET_F32_TENSOR_ERROR_INVALID_STATE;
}

static int32_t et_g3c4_owner_idle_preflight(
    et_g3c4_model_owner_internal *owner, et_f32_tensor_error *error) {
  et_f32_tensor_error local_error;
  et_f32_tensor_error *reported_error = error != NULL ? error : &local_error;
  for (size_t index = 0u; index < 14u; index++) {
    const void *actual = NULL;
    int32_t status;
    for (size_t prior = 0u; prior < index; prior++) {
      if (owner->parameters[prior] == owner->parameters[index] ||
          (owner->handles[index] != NULL &&
           owner->handles[prior] == owner->handles[index]))
        return et_g3c4_preflight_error(
            error, "owner parameter or identity repeats");
    }
    if (!et_f32_parameter_is_live_v1(owner->parameters[index]))
      return et_g3c4_preflight_error(error, "owner parameter is not live");
    status = et_f32_parameter_identity_v1(
        owner->parameters[index], &actual, reported_error);
    if (owner->handles[index] == NULL) {
      if (status == 0 ||
          reported_error->category != ET_F32_TENSOR_ERROR_INVALID_STATE ||
          reported_error->code != ET_F32_TENSOR_CODE_INVALID_HANDLE)
        return et_g3c4_preflight_error(
            error, "unbound owner parameter has an identity");
    } else {
      if (status != 0) {
        (void)et_g3c4_capture_f32(status, reported_error);
        return status;
      }
      if (actual != owner->handles[index])
        return et_g3c4_preflight_error(
            error, "owner parameter identity differs");
      status = et_f32_parameter_validate_identity_v1(
          owner->parameters[index], owner->handles[index], reported_error);
      if (status != 0) {
        (void)et_g3c4_capture_f32(status, reported_error);
        return status;
      }
    }
    status = et_f32_parameter_idle_preflight_internal(
        owner->parameters[index], reported_error);
    if (status != 0) {
      (void)et_g3c4_capture_f32(status, reported_error);
      return status;
    }
  }
  if (error != NULL) memset(error, 0, sizeof(*error));
  return 0;
}

void *et_g3c4_private_model_owner_create_seeded_v1(int64_t seed) {
  et_g3c4_model_owner_internal *owner = NULL;
  et_kernel_runtime *initializer = NULL;
  int staged_initializer = 0;
  et_g3c4_error_reset_internal();
  if (seed < 0) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
    return NULL;
  }
  if (staged_c4_owner != NULL) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
  if (et_g3c4_discover_initializer(
          &initializer, &staged_initializer))
    return NULL;
  owner = (et_g3c4_model_owner_internal *)
      et_g3c4_allocate(sizeof(*owner));
  if (owner == NULL) {
    if (staged_initializer) et_kernel_runtime_destroy(initializer);
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
    return NULL;
  }
  owner->magic = ET_G3C4_OWNER_MAGIC;
  owner->state = ET_G3C4_OWNER_OPEN;
  owner->original[0] = 1;
  owner->original[1] = seed;
  memcpy(owner->successor, owner->original, sizeof(owner->original));
  for (size_t index = 0u; index < 14u; index++) {
    et_f32_tensor *initial = NULL;
    et_f32_tensor_error error;
    const size_t rank = et_g3c4_parameter_rank(index);
    int32_t status = et_f32_tensor_create_v1(
        rank, et_g3c4_parameter_shapes[index], &initial, &error);
    if (status == 0)
      status = et_f32_parameter_create_v1(
          initial, &owner->parameters[index], &error);
    if (initial != NULL) {
      if (status != 0) (void)et_g3c4_capture_f32(status, &error);
      et_g3c4_error_state_internal first = et_g3c4_error_state;
      if (et_f32_tensor_destroy_v1(&initial, NULL) != 0) abort();
      et_g3c4_error_restore_internal(first);
    }
    if (status != 0) {
      et_g3c4_error_state_internal first;
      (void)et_g3c4_capture_f32(status, &error);
      first = et_g3c4_error_state;
      for (size_t prior = 0u; prior <= index; prior++) {
        if (owner->parameters[prior] != NULL)
          if (et_f32_parameter_destroy_v1(
                  &owner->parameters[prior], NULL) != 0)
            abort();
      }
      free(owner);
      if (staged_initializer) et_kernel_runtime_destroy(initializer);
      et_g3c4_error_restore_internal(first);
      return NULL;
    }
  }
  if (staged_initializer) et_g3c4_initializer_runtime = initializer;
  owner->registry_next = et_g3c4_owner_registry;
  et_g3c4_owner_registry = owner;
  staged_c4_owner = owner;
  return owner;
}

void *et_g3c4_private_model_owner_parameter_v1(
    void *candidate, int64_t index) {
  et_g3c4_model_owner_internal *owner;
  et_g3c4_error_reset_internal();
  owner = et_g3c4_admit_owner(candidate, 0);
  if (owner == NULL) return NULL;
  if (index < 0 || index >= 14) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
    return NULL;
  }
  return owner->parameters[index];
}

int64_t et_g3c4_private_model_owner_bind_v1(
    void *candidate, int64_t index, void *exact_handle) {
  et_g3c4_model_owner_internal *owner;
  et_f32_tensor_error error;
  et_g3c4_error_reset_internal();
  owner = et_g3c4_admit_owner(candidate, 0);
  if (owner == NULL) return et_g3c4_error_state.category;
  if (owner != staged_c4_owner || owner->state != ET_G3C4_OWNER_OPEN)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (index < 0 || index >= 14 || exact_handle == NULL)
    return et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
  if (owner->handles[index] != NULL)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  for (size_t prior = 0u; prior < 14u; prior++) {
    if (owner->handles[prior] == exact_handle)
      return et_g3c4_fail(
          ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_TOPOLOGY);
  }
  if (et_g3c4_capture_f32(
          et_f32_parameter_bind_identity_v1(
              owner->parameters[index], exact_handle, &error), &error))
    return et_g3c4_error_state.category;
  owner->handles[index] = exact_handle;
  return 0;
}

int64_t et_g3c4_private_model_owner_initialize_v1(
    void *candidate, int64_t index) {
  et_g3c4_model_owner_internal *owner;
  et_f32_tensor *value;
  et_g3c4_error_reset_internal();
  owner = et_g3c4_admit_owner(candidate, 0);
  if (owner == NULL) return et_g3c4_error_state.category;
  if (owner != staged_c4_owner || owner->state != ET_G3C4_OWNER_OPEN)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (index < 0 || index >= 14)
    return et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
  if ((size_t)index != owner->initialized)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  for (size_t bound = 0u; bound < 14u; bound++) {
    if (owner->handles[bound] == NULL)
      return et_g3c4_fail(
          ET_G3C4_INVALID_STATE, ET_G3C4_CODE_TOPOLOGY);
  }
  value = et_g3c4_parameter_value(owner, (size_t)index);
  if (value == NULL) return et_g3c4_error_state.category;
  if (et_g3c4_parameter_rank((size_t)index) == 1u) {
    uint32_t bits[4];
    const uint32_t bit =
        index == 7 || index == 9 || index == 12
            ? UINT32_C(0x3f800000) : UINT32_C(0);
    et_f32_tensor_error error;
    for (size_t element = 0u; element < 4u; element++) bits[element] = bit;
    if (et_g3c4_capture_f32(
            et_f32_tensor_copy_bits_from_v1(value, bits, 4u, &error), &error))
      return et_g3c4_error_state.category;
  } else {
    et_f32_scoped_guard_internal guard = {0};
    et_f32_tensor_error error;
    int32_t status = et_f32_tensor_scoped_begin_internal(
        value, &guard, &error);
    if (et_g3c4_capture_f32(status, &error))
      return et_g3c4_error_state.category;
    status = et_g3c4_dispatch_initializer(owner, (size_t)index, guard.view);
    if (et_f32_tensor_scoped_end_internal(&guard) != 0) abort();
    if (status != 0) return et_g3c4_error_state.category;
  }
  owner->initialized++;
  if (owner->initialized == 14u) owner->successor_ready = 1;
  return 0;
}

int64_t et_g3c4_private_model_owner_word_v1(
    void *candidate, int64_t index) {
  et_g3c4_model_owner_internal *owner;
  et_g3c4_error_reset_internal();
  owner = et_g3c4_admit_owner(candidate, 0);
  if (owner == NULL) return 0;
  if (index < 0 || index >= 8) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
    return 0;
  }
  if (index >= 4 && !owner->successor_ready) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_READINESS);
    return 0;
  }
  return index < 4 ? owner->original[index] : owner->successor[index - 4];
}

int32_t et_g3c4_construction_parameter_preflight_internal(
    const void *exact_owner, void *parameter,
    const void *exact_handle, et_f32_tensor_error *error) {
  et_g3c4_model_owner_internal *owner;
  size_t found = 14u;
  et_g3c4_error_reset_internal();
  owner = et_g3c4_admit_owner(exact_owner, 0);
  if (owner == NULL || owner != staged_c4_owner ||
      (owner->state != ET_G3C4_OWNER_OPEN &&
       owner->state != ET_G3C4_OWNER_PREPARED))
    return et_g3c4_preflight_error(error, "owner is not the staged C4 owner");
  for (size_t index = 0u; index < 14u; index++) {
    if (owner->parameters[index] == parameter) {
      if (found != 14u)
        return et_g3c4_preflight_error(error, "parameter repeats in owner");
      found = index;
    }
  }
  if (found == 14u || owner->handles[found] != exact_handle)
    return et_g3c4_preflight_error(
        error, "parameter or exact handle differs from owner membership");
  return et_g3c4_owner_idle_preflight(owner, error);
}

int64_t et_g3c4_private_model_owner_prepare_seal_v1(void *candidate) {
  et_g3c4_model_owner_internal *owner;
  et_f32_tensor_error error;
  et_g3c4_error_reset_internal();
  owner = et_g3c4_admit_owner(candidate, 0);
  if (owner == NULL) return et_g3c4_error_state.category;
  if (owner != staged_c4_owner || owner->state != ET_G3C4_OWNER_OPEN)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (owner->initialized != 14u || !owner->successor_ready)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_READINESS);
  if (owner->active != NULL)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (owner->original[0] != 1 || owner->original[1] < 0 ||
      owner->original[2] != 0 || owner->original[3] != 0 ||
      owner->successor[0] != 1 ||
      owner->successor[1] != owner->original[1] ||
      owner->successor[2] != 292 || owner->successor[3] != 0)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_INVARIANT);
  if (et_g3c4_owner_idle_preflight(owner, &error) != 0)
    return et_g3c4_error_state.category;
  for (size_t index = 0u; index < 14u; index++) {
    et_f32_gradient_metadata_v1 metadata = {sizeof(metadata), 0u, 0u, 0u};
    const et_f32_tensor *value = NULL;
    if (!et_g3c4_parameter_shape_valid(owner, index)) {
      if (et_g3c4_error_state.category == 0)
        (void)et_g3c4_fail(
            ET_G3C4_SHAPE_MISMATCH, ET_G3C4_CODE_SHAPE);
      return et_g3c4_error_state.category;
    }
    if (!et_g3c4_parameter_value_finite(owner, index))
      return et_g3c4_error_state.category;
    if (et_g3c4_capture_f32(
            et_f32_parameter_value_tensor_v1(
                owner->parameters[index], &value, &error), &error))
      return et_g3c4_error_state.category;
    if (et_g3c4_capture_f32(
            et_f32_parameter_gradient_metadata_v1(
                owner->parameters[index], &metadata, &error), &error))
      return et_g3c4_error_state.category;
    if (metadata.state != ET_F32_GRADIENT_ABSENT ||
        metadata.contribution_count != 0u ||
        metadata.normalization_weight_bits != 0u)
      return et_g3c4_fail(
          ET_G3C4_INVALID_STATE, ET_G3C4_CODE_READINESS);
    for (size_t prior = 0u; prior < index; prior++) {
      const et_f32_tensor *prior_value = NULL;
      int32_t identical = 0;
      if (et_g3c4_capture_f32(
              et_f32_parameter_value_tensor_v1(
                  owner->parameters[prior], &prior_value, &error), &error) ||
          et_g3c4_capture_f32(
              et_f32_tensor_storage_identical_v1(
                  value, prior_value, &identical, &error), &error))
        return et_g3c4_error_state.category;
      if (owner->parameters[prior] == owner->parameters[index] ||
          owner->handles[prior] == owner->handles[index] || identical)
        return et_g3c4_fail(
            ET_G3C4_INVALID_STATE, ET_G3C4_CODE_TOPOLOGY);
    }
  }
  owner->state = ET_G3C4_OWNER_PREPARED;
  return 0;
}

int64_t et_g3c4_private_model_owner_commit_seal_v1(void *candidate) {
  et_g3c4_model_owner_internal *owner;
  et_g3c4_error_reset_internal();
  owner = et_g3c4_admit_owner(candidate, 0);
  if (owner == NULL) return et_g3c4_error_state.category;
  if (owner != staged_c4_owner || owner->state != ET_G3C4_OWNER_PREPARED)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  owner->state = ET_G3C4_OWNER_SEALED;
  staged_c4_owner = NULL;
  return 0;
}

int64_t et_g3c4_private_model_owner_abort_v1(void *candidate) {
  et_g3c4_model_owner_internal *owner;
  et_f32_tensor_error error;
  et_g3c4_error_reset_internal();
  owner = et_g3c4_admit_owner(candidate, 1);
  if (owner == NULL) return et_g3c4_error_state.category;
  if (owner->state == ET_G3C4_OWNER_ABORTED) return 0;
  if (owner == staged_c4_owner &&
      (owner->state == ET_G3C4_OWNER_OPEN ||
       owner->state == ET_G3C4_OWNER_PREPARED)) {
    if (et_g3c4_owner_idle_preflight(owner, &error) != 0)
      return et_g3c4_error_state.category;
    for (size_t index = 0u; index < 14u; index++) {
      if (et_f32_parameter_destroy_v1(
              &owner->parameters[index], &error) != 0)
        abort();
    }
    memset(owner->handles, 0, sizeof(owner->handles));
    owner->initialized = 0u;
    owner->successor_ready = 0;
    owner->active = NULL;
    owner->state = ET_G3C4_OWNER_ABORTED;
    staged_c4_owner = NULL;
    return 0;
  }
  return et_g3c4_fail(
      ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
}
