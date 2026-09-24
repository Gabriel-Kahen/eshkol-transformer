#ifndef ET_G3C4_NATIVE_OWNER_PRIVATE
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#endif

#include "g3c4_model_owner_internal.h"
#include "m3t_f32_scoped.h"
#include "eshkol_transformer/i64_tensor.h"
#include "eshkol_transformer/n3k_primitives_abi.h"
#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
#include "../../native/t1_i64_shell.h"
#endif

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

#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
static int32_t et_g3c4_capture_i64(
    int32_t status, const et_i64_tensor_error *error) {
  if (status != 0) {
    if (error == NULL) {
      et_g3c4_error_set_internal(-1, -1, -1);
    } else {
      et_g3c4_error_set_internal(
          ET_G3C4_DOMAIN_I1, error->category, error->code);
    }
  }
  return status;
}
#endif

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

#if defined(ET_G3C4_ACTIVE_CALL_PRIVATE) && \
    (!defined(ET_G3C4_CONTEXT_PRIVATE) || \
     !defined(ET_G3C4_NATIVE_PINS_PRIVATE))
#error "ET_G3C4_ACTIVE_CALL_PRIVATE requires context and C4 pins"
#endif

#if defined(ET_G3C4_GENERATOR_PRIVATE) && \
    (!defined(ET_G3C4_CONTEXT_PRIVATE) || \
     !defined(ET_G3C4_NATIVE_PINS_PRIVATE) || \
     !defined(ET_G3C4_ACTIVE_CALL_PRIVATE))
#error "ET_G3C4_GENERATOR_PRIVATE requires context, C4 pins, and active call"
#endif

#if defined(ET_G3C4_PROMPT_T1_BORROW_PRIVATE) && \
    !defined(ET_G3C4_GENERATOR_PRIVATE)
#error "ET_G3C4_PROMPT_T1_BORROW_PRIVATE requires the private generator"
#endif

#if defined(ET_G3C4_PROVIDER_ROUTES_PRIVATE) && \
    !defined(ET_G3C4_GENERATOR_PRIVATE)
#error "ET_G3C4_PROVIDER_ROUTES_PRIVATE requires the private generator"
#endif

#if defined(ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE) && \
    !defined(ET_G3C4_PROVIDER_ROUTES_PRIVATE)
#error "ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE requires provider routes"
#endif

#if defined(ET_G3C4_SAMPLER_TRANSPORT_PRIVATE) && \
    !defined(ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE)
#error "ET_G3C4_SAMPLER_TRANSPORT_PRIVATE requires full-prefix forward"
#endif

#if defined(ET_G3C4_TOKEN_FRAME_PRIVATE) && \
    !defined(ET_G3C4_SAMPLER_TRANSPORT_PRIVATE)
#error "ET_G3C4_TOKEN_FRAME_PRIVATE requires Step 9A"
#endif

#if defined(ET_G3C4_TOKEN_FORWARD_PRIVATE) && \
    !defined(ET_G3C4_TOKEN_FRAME_PRIVATE)
#error "ET_G3C4_TOKEN_FORWARD_PRIVATE requires Step 10A"
#endif

#if defined(ET_G3C4_PREFILL3_PRIVATE) && \
    !defined(ET_G3C4_TOKEN_FORWARD_PRIVATE)
#error "ET_G3C4_PREFILL3_PRIVATE requires Step 11A"
#endif
#if defined(ET_G3C4_PREFILL3_PRIVATE) && \
    !defined(ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
#error "ET_G3C4_PREFILL3_PRIVATE requires the private A2 storage query"
#endif
#if defined(ET_G3C4_LAST_LOGIT_FRAME_PRIVATE) && \
    !defined(ET_G3C4_PREFILL3_PRIVATE)
#error "ET_G3C4_LAST_LOGIT_FRAME_PRIVATE requires Step 12A"
#endif

#ifdef ET_G3C4_PROVIDER_ROUTES_PRIVATE
#include "eshkol_transformer/g3c4_primitives_abi.h"
#include "eshkol_transformer/g3s_sampling_abi.h"
#ifdef ET_G3C4_TOKEN_FORWARD_PRIVATE
#include "eshkol_transformer/g3n_primitives_abi.h"
#include "eshkol_transformer/n2_primitives_abi.h"
#endif
#endif



#ifdef ET_G3C4_CONTEXT_PRIVATE
#include "g3c4_context_internal.h"
#include "eshkol_transformer/a2_kv_cache.h"
#ifdef ET_G3C4_ACTIVE_CALL_PRIVATE
#include "m3_call_pins.h"
#endif

#define ET_G3C4_CONTEXT_MAGIC UINT64_C(0x4733433443545831)
#ifdef ET_G3C4_GENERATOR_PRIVATE
#define ET_G3C4_RNG_MAGIC UINT64_C(0x47334334524e4731)
#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
#define ET_G3C4_INPUT_MAGIC UINT64_C(0x47334334494e5031)
#endif
#endif

enum {
  ET_G3C4_CONTEXT_KIND = 1,
  ET_G3C4_CONTEXT_LIVE = 1,
  ET_G3C4_CONTEXT_DEAD = 2
};

#ifdef ET_G3C4_GENERATOR_PRIVATE
enum {
#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
  ET_G3C4_INPUT_KIND = 2,
  ET_G3C4_RNG_KIND = 8
#else
  ET_G3C4_RNG_KIND = 2
#endif
};

typedef struct et_g3c4_transport_header_internal {
  struct et_g3c4_transport_header_internal *registry_next;
  uint64_t magic;
  uint32_t kind;
  uint32_t state;
  uint32_t busy;
} et_g3c4_transport_header_internal;
#endif

#ifdef ET_G3C4_ACTIVE_CALL_PRIVATE
enum {
  ET_G3C4_CALL_IDLE = 0,
  ET_G3C4_CALL_ACTIVE = 1,
  ET_G3C4_ACQUIRED_PINS = 1,
  ET_G3C4_ACQUIRED_METADATA = 2,
  ET_G3C4_ACQUIRED_BUSY = 4,
  ET_G3C4_ACQUIRED_OWNER = 8,
  ET_G3C4_ACQUIRED_FULL = 15
};
#endif

#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
enum {
  ET_G3C4_TOKEN_FRAME_IDLE = 0,
  ET_G3C4_TOKEN_FRAME_SAMPLED = 1,
  ET_G3C4_TOKEN_FRAME_READY = 2
};
#endif

typedef struct et_g3c4_context_internal {
#ifdef ET_G3C4_GENERATOR_PRIVATE
  et_g3c4_transport_header_internal transport;
#else
  struct et_g3c4_context_internal *registry_next;
  uint64_t magic;
  uint32_t kind;
  uint32_t state;
  uint32_t busy;
#endif
  et_g3c4_model_owner_internal *owner;
  et_a2_kv_cache *cache;
#ifdef ET_G3C4_ACTIVE_CALL_PRIVATE
  et_g3c4_model_pins_internal pins;
  int64_t call_kind;
  int64_t budget;
  uint32_t acquired_mask;
#endif
#ifdef ET_G3C4_GENERATOR_PRIVATE
  uint32_t generator_kind;
  uint32_t generator_ready;
  int64_t generator_policy[6];
  int64_t generator_rng_words[4];
#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
  et_a2_kv_cache_transaction *token_frame_transaction;
  uint32_t token_frame_state;
  int64_t token_frame_candidate;
  int64_t token_frame_successor[4];
#ifdef ET_G3C4_TOKEN_FORWARD_PRIVATE
  int64_t token_frame_position;
#endif
#endif
#ifdef ET_G3C4_PREFILL3_PRIVATE
  uint32_t prefill_binding_ready;
  uint32_t prefill_binding_reserved;
  const void *prefill_binding_identities[14];
  unsigned char prefill_binding_values[4768];
  int64_t prefill_tokens[3];
#endif
#endif
} et_g3c4_context_internal;

#ifdef ET_G3C4_GENERATOR_PRIVATE
typedef struct et_g3c4_rng_internal {
  et_g3c4_transport_header_internal transport;
  int64_t words[4];
} et_g3c4_rng_internal;

#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
typedef struct et_g3c4_input_internal {
  et_g3c4_transport_header_internal transport;
  et_i64_tensor *tensor;
  int64_t length;
} et_g3c4_input_internal;
#endif

static et_g3c4_transport_header_internal *et_g3c4_transport_registry;

#ifdef ET_G3C4_PROVIDER_ROUTES_PRIVATE
typedef struct et_g3c4_route_requirement {
  const char *capability;
  const char *operation;
  size_t rank;
  uint64_t shape[6];
} et_g3c4_route_requirement;

static et_kernel_runtime *et_g3c4_forward_runtime;
static et_kernel_runtime *et_g3c4_sampler_runtime;

static const et_g3c4_route_requirement et_g3c4_forward_routes[] = {
  {"g3c4.embedding-forward", "g3c4.embedding.forward", 4u,
   {1u, 4u, 4u, 4u}},
  {"g3c4.embedding-forward", "g3c4.embedding.forward", 4u,
   {1u, 4u, 256u, 4u}},
  {"g3c4.linear", "g3c4.linear.forward-no-bias", 4u,
   {1u, 4u, 4u, 4u}},
  {"g3c4.linear", "g3c4.linear.forward-no-bias", 4u,
   {1u, 4u, 4u, 8u}},
  {"g3c4.linear", "g3c4.linear.forward-no-bias", 4u,
   {1u, 4u, 8u, 4u}},
  {"g3c4.linear", "g3c4.linear.forward-no-bias", 4u,
   {1u, 4u, 4u, 256u}},
  {"g3c4.layer-norm", "g3c4.layer-norm.forward", 3u, {1u, 4u, 4u}},
  {"g3c4.gelu", "g3c4.gelu.forward", 3u, {1u, 4u, 8u}},
  {"g3c4.residual", "g3c4.residual.forward", 3u, {1u, 4u, 4u}},
  {"g3c4.head-layout", "g3c4.heads.split.forward", 4u,
   {1u, 4u, 2u, 2u}},
  {"g3c4.head-layout", "g3c4.heads.merge.forward", 4u,
   {1u, 4u, 2u, 2u}},
  {"g3c4.causal-attention", "g3c4.causal-attention.forward", 6u,
   {1u, 2u, 2u, 4u, 4u, 2u}},
};

static const et_g3c4_route_requirement et_g3c4_sampler_routes[] = {
  {"g3s.greedy", "g3s.greedy.forward", 2u, {1u, 256u}},
  {"g3s.categorical", "g3s.categorical.forward", 2u, {1u, 256u}},
};

static int et_g3c4_require_routes(
    et_kernel_runtime *runtime,
    const et_g3c4_route_requirement *routes, size_t count) {
  size_t index;
  for (index = 0u; index < count; index++) {
    et_kernel_request_v1 request = {
      sizeof(request), routes[index].operation, "f32", "cpu",
      routes[index].rank, routes[index].shape, 1u, {0}};
    const et_kernel_capability_v1 *entry = NULL;
    et_kernel_error error;
    if (et_g3c4_capture_kernel(
            et_kernel_runtime_capability_require(
                runtime, routes[index].capability, &request, &entry, &error),
            &error) != 0)
      return (int)et_g3c4_error_state.category;
    if (entry == NULL || entry->status != ET_KERNEL_CAPABILITY_VERIFIED ||
        entry->deterministic == 0u)
      return (int)et_g3c4_fail(
          ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
  }
  return 0;
}

static int et_g3c4_discover_route_runtime(
    const et_kernel_provider_v1 *provider, size_t capability_count,
    const et_g3c4_route_requirement *routes, size_t route_count,
    et_kernel_runtime **runtime_output) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  if (provider == NULL || provider->capability_count != capability_count) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return (int)et_g3c4_error_state.category;
  }
  if (et_g3c4_capture_kernel(
          et_kernel_runtime_discover(et_g3c4_resolve_provider,
                                     (void *)provider, &runtime, &error),
          &error) != 0)
    return (int)et_g3c4_error_state.category;
  if (et_g3c4_require_routes(runtime, routes, route_count) != 0) {
    et_kernel_runtime_destroy(runtime);
    return (int)et_g3c4_error_state.category;
  }
  *runtime_output = runtime;
  return 0;
}

static void et_g3c4_abort_staged_routes(
    et_kernel_runtime *forward, et_kernel_runtime *sampler) {
  if (sampler != NULL && sampler != et_g3c4_sampler_runtime)
    et_kernel_runtime_destroy(sampler);
  if (forward != NULL && forward != et_g3c4_forward_runtime)
    et_kernel_runtime_destroy(forward);
}

static int et_g3c4_stage_provider_routes(
    const et_kernel_provider_v1 *forward_provider,
    const et_kernel_provider_v1 *sampler_provider,
    et_kernel_runtime **forward_output,
    et_kernel_runtime **sampler_output) {
  et_kernel_runtime *forward = et_g3c4_forward_runtime;
  et_kernel_runtime *sampler = et_g3c4_sampler_runtime;
  if (forward == NULL &&
      et_g3c4_discover_route_runtime(
          forward_provider, 7u, et_g3c4_forward_routes,
          sizeof(et_g3c4_forward_routes) /
              sizeof(et_g3c4_forward_routes[0]),
          &forward) != 0)
    return (int)et_g3c4_error_state.category;
  if (forward != NULL && forward == et_g3c4_forward_runtime &&
      et_g3c4_require_routes(
          forward, et_g3c4_forward_routes,
          sizeof(et_g3c4_forward_routes) /
              sizeof(et_g3c4_forward_routes[0])) != 0)
    return (int)et_g3c4_error_state.category;
  if (sampler == NULL &&
      et_g3c4_discover_route_runtime(
          sampler_provider, 2u, et_g3c4_sampler_routes,
          sizeof(et_g3c4_sampler_routes) /
              sizeof(et_g3c4_sampler_routes[0]),
          &sampler) != 0) {
    et_g3c4_abort_staged_routes(forward, sampler);
    return (int)et_g3c4_error_state.category;
  }
  if (sampler != NULL && sampler == et_g3c4_sampler_runtime &&
      et_g3c4_require_routes(
          sampler, et_g3c4_sampler_routes,
          sizeof(et_g3c4_sampler_routes) /
              sizeof(et_g3c4_sampler_routes[0])) != 0)
    return (int)et_g3c4_error_state.category;
  *forward_output = forward;
  *sampler_output = sampler;
  return 0;
}
#endif

#define ET_G3C4_CONTEXT_MAGIC_FIELD(context) ((context)->transport.magic)
#define ET_G3C4_CONTEXT_KIND_FIELD(context) ((context)->transport.kind)
#define ET_G3C4_CONTEXT_STATE(context) ((context)->transport.state)
#define ET_G3C4_CONTEXT_BUSY(context) ((context)->transport.busy)
#else
static et_g3c4_context_internal *et_g3c4_context_registry;

#define ET_G3C4_CONTEXT_MAGIC_FIELD(context) ((context)->magic)
#define ET_G3C4_CONTEXT_KIND_FIELD(context) ((context)->kind)
#define ET_G3C4_CONTEXT_STATE(context) ((context)->state)
#define ET_G3C4_CONTEXT_BUSY(context) ((context)->busy)
#endif

#ifdef ET_G3C4_GENERATOR_PRIVATE
static int et_g3c4_valid_generator_policy(
    int64_t mode, int64_t temperature_bits, int64_t k,
    int64_t p_bits, int64_t max_new, int64_t eos);

static int et_g3c4_zero_i64_words(const int64_t *words, size_t count) {
  size_t index;
  for (index = 0u; index < count; index++)
    if (words[index] != 0) return 0;
  return 1;
}

#ifdef ET_G3C4_PREFILL3_PRIVATE
static int et_g3c4_zero_bytes(const void *data, size_t count) {
  const unsigned char *bytes = (const unsigned char *)data;
  size_t index;
  for (index = 0u; index < count; index++)
    if (bytes[index] != 0u) return 0;
  return 1;
}

static int et_g3c4_prefill_binding_valid(
    const et_g3c4_context_internal *context) {
  size_t index;
  if (context->prefill_binding_reserved != 0u ||
      context->prefill_binding_ready > 1u)
    return 0;
  if (context->prefill_binding_ready == 0u)
    return et_g3c4_zero_bytes(
               context->prefill_binding_identities,
               sizeof(context->prefill_binding_identities)) &&
           et_g3c4_zero_bytes(
               context->prefill_binding_values,
               sizeof(context->prefill_binding_values)) &&
           et_g3c4_zero_i64_words(context->prefill_tokens, 3u);
  for (index = 0u; index < 14u; index++)
    if (context->prefill_binding_identities[index] == NULL) return 0;
  for (index = 0u; index < 3u; index++)
    if (context->prefill_tokens[index] < 0 ||
        context->prefill_tokens[index] > 255)
      return 0;
  return 1;
}

static void et_g3c4_prefill_binding_clear(
    et_g3c4_context_internal *context) {
  context->prefill_binding_ready = 0u;
  context->prefill_binding_reserved = 0u;
  memset(context->prefill_binding_identities, 0,
         sizeof(context->prefill_binding_identities));
  memset(context->prefill_binding_values, 0,
         sizeof(context->prefill_binding_values));
  memset(context->prefill_tokens, 0, sizeof(context->prefill_tokens));
}
#endif

#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
static int et_g3c4_token_frame_valid(
    const et_g3c4_context_internal *context) {
  if (context->token_frame_state == ET_G3C4_TOKEN_FRAME_IDLE)
    return context->token_frame_transaction == NULL &&
           context->token_frame_candidate == 0 &&
           et_g3c4_zero_i64_words(context->token_frame_successor, 4u)
#ifdef ET_G3C4_TOKEN_FORWARD_PRIVATE
           && context->token_frame_position == 0
#endif
           ;
  if (context->token_frame_state != ET_G3C4_TOKEN_FRAME_SAMPLED &&
      context->token_frame_state != ET_G3C4_TOKEN_FRAME_READY)
    return 0;
  return context->token_frame_transaction != NULL &&
         context->token_frame_candidate >= 0 &&
         context->token_frame_candidate <= 255 &&
         context->token_frame_successor[0] == 1 &&
         context->token_frame_successor[1] ==
             context->generator_rng_words[1]
#ifdef ET_G3C4_TOKEN_FORWARD_PRIVATE
         && context->token_frame_position >= 0 &&
         context->token_frame_position <= 3
#endif
         ;
}

static int et_g3c4_token_frame_idle(
    const et_g3c4_context_internal *context) {
  return context->token_frame_state == ET_G3C4_TOKEN_FRAME_IDLE &&
         et_g3c4_token_frame_valid(context);
}
#endif

static int et_g3c4_context_subtype_valid(
    const et_g3c4_context_internal *context) {
  if (context->generator_kind == 0u)
    return context->generator_ready == 0u &&
           et_g3c4_zero_i64_words(context->generator_policy, 6u) &&
           et_g3c4_zero_i64_words(context->generator_rng_words, 4u);
  if (context->generator_kind != 1u) return 0;
  if (ET_G3C4_CONTEXT_STATE(context) == ET_G3C4_CONTEXT_DEAD)
    return context->generator_ready == 0u &&
           et_g3c4_zero_i64_words(context->generator_policy, 6u) &&
           et_g3c4_zero_i64_words(context->generator_rng_words, 4u)
#ifdef ET_G3C4_PREFILL3_PRIVATE
           && et_g3c4_prefill_binding_valid(context) &&
           context->prefill_binding_ready == 0u
#endif
           ;
  return context->generator_ready == 1u &&
         et_g3c4_valid_generator_policy(
             context->generator_policy[0], context->generator_policy[1],
             context->generator_policy[2], context->generator_policy[3],
             context->generator_policy[4], context->generator_policy[5]) &&
         context->generator_rng_words[0] == 1 &&
         context->generator_rng_words[1] >= 0
#ifdef ET_G3C4_PREFILL3_PRIVATE
         && et_g3c4_prefill_binding_valid(context)
#endif
#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
         && et_g3c4_token_frame_valid(context)
#endif
         ;
}

static int et_g3c4_transport_record_valid(
    const et_g3c4_transport_header_internal *header) {
  if (header->kind == ET_G3C4_CONTEXT_KIND) {
    return header->magic == ET_G3C4_CONTEXT_MAGIC &&
           (header->state == ET_G3C4_CONTEXT_LIVE ||
            header->state == ET_G3C4_CONTEXT_DEAD) &&
           header->busy <= ET_G3C4_CALL_ACTIVE &&
           et_g3c4_context_subtype_valid(
               (const et_g3c4_context_internal *)header);
  }
  if (header->kind == ET_G3C4_RNG_KIND) {
    const et_g3c4_rng_internal *rng =
        (const et_g3c4_rng_internal *)header;
    return header->magic == ET_G3C4_RNG_MAGIC &&
           (header->state == ET_G3C4_CONTEXT_LIVE ||
            header->state == ET_G3C4_CONTEXT_DEAD) &&
           header->busy == 0u &&
           (header->state == ET_G3C4_CONTEXT_LIVE
                ? rng->words[0] == 1 && rng->words[1] >= 0
                : et_g3c4_zero_i64_words(rng->words, 4u));
  }
#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
  if (header->kind == ET_G3C4_INPUT_KIND) {
    const et_g3c4_input_internal *input =
        (const et_g3c4_input_internal *)header;
    return header->magic == ET_G3C4_INPUT_MAGIC &&
           (header->state == ET_G3C4_CONTEXT_LIVE ||
            header->state == ET_G3C4_CONTEXT_DEAD) &&
           header->busy == 0u &&
           (header->state == ET_G3C4_CONTEXT_LIVE
                ? input->tensor != NULL &&
                  (input->length == 1 || input->length == 2)
                : input->tensor == NULL && input->length == 0);
  }
#endif
  return 0;
}
#endif

#ifdef ET_G3C4_ACTIVE_CALL_PRIVATE
static int et_g3c4_pins_idle(
    const et_g3c4_model_pins_internal *pins);
#endif

#ifdef ET_G3C4_CONTEXT_TESTING
static size_t et_g3c4_context_allocation_limit = SIZE_MAX;
static size_t et_g3c4_context_successful_allocations;
#endif

#if defined(ET_G3C4_ACTIVE_CALL_PRIVATE) && \
    defined(ET_G3C4_ACTIVE_CALL_TESTING)
static size_t et_g3c4_active_call_fail_after = SIZE_MAX;
#endif

static et_g3c4_context_internal *et_g3c4_admit_context(
    const void *candidate) {
#ifdef ET_G3C4_GENERATOR_PRIVATE
  et_g3c4_transport_header_internal *header;
  for (header = et_g3c4_transport_registry;
       header != NULL && (const void *)header != candidate;
       header = header->registry_next) {}
  if (header == NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
  if (!et_g3c4_transport_record_valid(header)) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return NULL;
  }
  if (header->kind != ET_G3C4_CONTEXT_KIND) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
  return (et_g3c4_context_internal *)header;
#else
  et_g3c4_context_internal *context;
  for (context = et_g3c4_context_registry;
       context != NULL && (const void *)context != candidate;
       context = context->registry_next) {}
  if (context == NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
  if (context->magic != ET_G3C4_CONTEXT_MAGIC ||
      context->kind != ET_G3C4_CONTEXT_KIND) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return NULL;
  }
  return context;
#endif
}

static et_g3c4_context_internal *et_g3c4_context_allocate(void) {
#ifdef ET_G3C4_CONTEXT_TESTING
  if (et_g3c4_context_successful_allocations >=
      et_g3c4_context_allocation_limit)
    return NULL;
#endif
  et_g3c4_context_internal *context =
      (et_g3c4_context_internal *)calloc(1u, sizeof(*context));
#ifdef ET_G3C4_CONTEXT_TESTING
  if (context != NULL) et_g3c4_context_successful_allocations++;
#endif
  return context;
}

#ifdef ET_G3C4_GENERATOR_PRIVATE
static et_g3c4_rng_internal *et_g3c4_rng_allocate(void) {
#ifdef ET_G3C4_CONTEXT_TESTING
  if (et_g3c4_context_successful_allocations >=
      et_g3c4_context_allocation_limit)
    return NULL;
#endif
  et_g3c4_rng_internal *rng =
      (et_g3c4_rng_internal *)calloc(1u, sizeof(*rng));
#ifdef ET_G3C4_CONTEXT_TESTING
  if (rng != NULL) et_g3c4_context_successful_allocations++;
#endif
  return rng;
}

#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
static et_g3c4_input_internal *et_g3c4_input_allocate(void) {
#ifdef ET_G3C4_CONTEXT_TESTING
  if (et_g3c4_context_successful_allocations >=
      et_g3c4_context_allocation_limit)
    return NULL;
#endif
  et_g3c4_input_internal *input =
      (et_g3c4_input_internal *)calloc(1u, sizeof(*input));
#ifdef ET_G3C4_CONTEXT_TESTING
  if (input != NULL) et_g3c4_context_successful_allocations++;
#endif
  return input;
}
#endif

static et_g3c4_rng_internal *et_g3c4_admit_rng(
    const void *candidate, int allow_dead) {
  et_g3c4_transport_header_internal *header;
  et_g3c4_rng_internal *rng;
  for (header = et_g3c4_transport_registry;
       header != NULL && (const void *)header != candidate;
       header = header->registry_next) {}
  if (header == NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
  if (!et_g3c4_transport_record_valid(header)) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return NULL;
  }
  if (header->kind != ET_G3C4_RNG_KIND) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
  rng = (et_g3c4_rng_internal *)header;
  if (rng->transport.state == ET_G3C4_CONTEXT_DEAD) {
    if (rng->words[0] != 0 || rng->words[1] != 0 ||
        rng->words[2] != 0 || rng->words[3] != 0) {
      (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
      return NULL;
    }
    if (!allow_dead) {
      (void)et_g3c4_fail(
          ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
      return NULL;
    }
  } else if (rng->words[0] != 1 || rng->words[1] < 0) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return NULL;
  }
  return rng;
}

#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
static et_g3c4_input_internal *et_g3c4_admit_input(
    const void *candidate, int allow_dead) {
  et_g3c4_transport_header_internal *header;
  et_g3c4_input_internal *input;
  for (header = et_g3c4_transport_registry;
       header != NULL && (const void *)header != candidate;
       header = header->registry_next) {}
  if (header == NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
  if (!et_g3c4_transport_record_valid(header)) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return NULL;
  }
  if (header->kind != ET_G3C4_INPUT_KIND) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
  input = (et_g3c4_input_internal *)header;
  if (input->transport.state == ET_G3C4_CONTEXT_DEAD && !allow_dead) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
  return input;
}
#endif

static void et_g3c4_enroll_context(et_g3c4_context_internal *context) {
  context->transport.registry_next = et_g3c4_transport_registry;
  et_g3c4_transport_registry = &context->transport;
}

static void et_g3c4_enroll_rng(et_g3c4_rng_internal *rng) {
  rng->transport.registry_next = et_g3c4_transport_registry;
  et_g3c4_transport_registry = &rng->transport;
}

#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
static void et_g3c4_enroll_input(et_g3c4_input_internal *input) {
  input->transport.registry_next = et_g3c4_transport_registry;
  et_g3c4_transport_registry = &input->transport;
}
#endif

_Static_assert(sizeof(et_g3c4_transport_header_internal) == 32u,
               "G3-C4 transport header must be 32 bytes");
_Static_assert(offsetof(et_g3c4_context_internal, transport) == 0u,
               "G3-C4 context header must be first");
#ifdef ET_G3C4_PREFILL3_PRIVATE
_Static_assert(sizeof(et_g3c4_context_internal) == 6496u,
               "G3-C4 prefill3 context must be 6496 bytes");
#elif defined(ET_G3C4_TOKEN_FORWARD_PRIVATE)
_Static_assert(sizeof(et_g3c4_context_internal) == 1584u,
               "G3-C4 token-forward context must be 1584 bytes");
#elif defined(ET_G3C4_TOKEN_FRAME_PRIVATE)
_Static_assert(sizeof(et_g3c4_context_internal) == 1576u,
               "G3-C4 token-frame context must be 1576 bytes");
#else
_Static_assert(sizeof(et_g3c4_context_internal) == 1520u,
               "G3-C4 generator context must be 1520 bytes");
#endif
_Static_assert(sizeof(et_g3c4_rng_internal) == 64u,
               "G3-C4 RNG record must be 64 bytes");
#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
_Static_assert(sizeof(et_g3c4_input_internal) == 48u,
               "G3-C4 input record must be 48 bytes");
_Static_assert(offsetof(et_g3c4_input_internal, transport) == 0u,
               "G3-C4 input header must be first");
#endif
#endif

void *et_g3c4_private_context_create_v1(void *candidate) {
  et_g3c4_model_owner_internal *owner;
  et_g3c4_context_internal *context;
  et_a2_kv_cache *cache = NULL;
  et_kernel_error error;
  et_g3c4_error_reset_internal();
  owner = et_g3c4_admit_owner(candidate, 0);
  if (owner == NULL) return NULL;
  if (owner->state != ET_G3C4_OWNER_SEALED || owner->active != NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
  context = et_g3c4_context_allocate();
  if (context == NULL) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
    return NULL;
  }
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_create_v1(
              1u, 1u, 2u, 4u, 2u, &cache, &error),
          &error) != 0) {
    et_g3c4_error_state_internal first = et_g3c4_error_snapshot_internal();
    free(context);
    et_g3c4_error_restore_internal(first);
    return NULL;
  }
  ET_G3C4_CONTEXT_MAGIC_FIELD(context) = ET_G3C4_CONTEXT_MAGIC;
  ET_G3C4_CONTEXT_KIND_FIELD(context) = ET_G3C4_CONTEXT_KIND;
  ET_G3C4_CONTEXT_STATE(context) = ET_G3C4_CONTEXT_LIVE;
  ET_G3C4_CONTEXT_BUSY(context) = 0u;
  context->owner = owner;
  context->cache = cache;
#ifdef ET_G3C4_GENERATOR_PRIVATE
  et_g3c4_enroll_context(context);
#else
  context->registry_next = et_g3c4_context_registry;
  et_g3c4_context_registry = context;
#endif
  return context;
}

int64_t et_g3c4_private_context_close_v1(void *candidate) {
  et_g3c4_context_internal *context;
  et_g3c4_model_owner_internal *owner;
  et_kernel_error error;
  et_g3c4_error_reset_internal();
  context = et_g3c4_admit_context(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
#ifdef ET_G3C4_GENERATOR_PRIVATE
  if (context->generator_kind != 0u)
    return et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
#endif
  if (ET_G3C4_CONTEXT_STATE(context) == ET_G3C4_CONTEXT_DEAD) {
#ifdef ET_G3C4_ACTIVE_CALL_PRIVATE
    if (ET_G3C4_CONTEXT_BUSY(context) != ET_G3C4_CALL_IDLE ||
        context->call_kind != 0 ||
        context->budget != 0 || context->acquired_mask != 0u ||
        !et_g3c4_pins_idle(&context->pins))
      return et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
#endif
    return 0;
  }
  if (ET_G3C4_CONTEXT_STATE(context) != ET_G3C4_CONTEXT_LIVE ||
      ET_G3C4_CONTEXT_BUSY(context) != 0u ||
      context->owner == NULL || context->cache == NULL)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
#ifdef ET_G3C4_ACTIVE_CALL_PRIVATE
  if (context->call_kind != 0 || context->budget != 0 ||
      context->acquired_mask != 0u || !et_g3c4_pins_idle(&context->pins))
    return et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
#endif
  owner = et_g3c4_admit_owner(context->owner, 0);
  if (owner == NULL) return et_g3c4_error_state.category;
  if (owner != context->owner || owner->state != ET_G3C4_OWNER_SEALED ||
      owner->active != NULL)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_destroy_v1(&context->cache, &error), &error) != 0)
    return et_g3c4_error_state.category;
  context->owner = NULL;
  ET_G3C4_CONTEXT_STATE(context) = ET_G3C4_CONTEXT_DEAD;
  return 0;
}

#ifdef ET_G3C4_GENERATOR_PRIVATE
static int et_g3c4_positive_f32_bits(int64_t bits) {
  uint32_t value;
  if (bits < 0 || bits > (int64_t)UINT32_MAX) return 0;
  value = (uint32_t)bits;
  return (value & UINT32_C(0x80000000)) == 0u &&
         (value & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000) &&
         (value & UINT32_C(0x7fffffff)) != 0u;
}

static int et_g3c4_valid_generator_policy(
    int64_t mode, int64_t temperature_bits, int64_t k,
    int64_t p_bits, int64_t max_new, int64_t eos) {
  if ((mode != 0 && mode != 1) ||
      !et_g3c4_positive_f32_bits(temperature_bits) ||
      !et_g3c4_positive_f32_bits(p_bits) ||
      (uint32_t)p_bits > UINT32_C(0x3f800000) ||
      k < 1 || k > 256 || max_new < 0 || max_new > 1 ||
      eos < -1 || eos > 255)
    return 0;
  if (mode == 0 &&
      (temperature_bits != INT64_C(0x3f800000) || k != 256 ||
       p_bits != INT64_C(0x3f800000)))
    return 0;
  return 1;
}

void *et_g3c4_private_rng_seed_v1(int64_t seed) {
  et_g3c4_rng_internal *rng;
  et_g3c4_error_reset_internal();
  if (seed < 0) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
    return NULL;
  }
  rng = et_g3c4_rng_allocate();
  if (rng == NULL) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
    return NULL;
  }
  rng->transport.magic = ET_G3C4_RNG_MAGIC;
  rng->transport.kind = ET_G3C4_RNG_KIND;
  rng->transport.state = ET_G3C4_CONTEXT_LIVE;
  rng->words[0] = 1;
  rng->words[1] = seed;
  et_g3c4_enroll_rng(rng);
  return rng;
}

void *et_g3c4_private_rng_clone_v1(void *candidate) {
  et_g3c4_rng_internal *source;
  et_g3c4_rng_internal *clone;
  et_g3c4_error_reset_internal();
  source = et_g3c4_admit_rng(candidate, 0);
  if (source == NULL) return NULL;
  clone = et_g3c4_rng_allocate();
  if (clone == NULL) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
    return NULL;
  }
  clone->transport.magic = ET_G3C4_RNG_MAGIC;
  clone->transport.kind = ET_G3C4_RNG_KIND;
  clone->transport.state = ET_G3C4_CONTEXT_LIVE;
  memcpy(clone->words, source->words, sizeof(clone->words));
  et_g3c4_enroll_rng(clone);
  return clone;
}

int64_t et_g3c4_private_rng_word_v1(void *candidate, int64_t index) {
  et_g3c4_rng_internal *rng;
  et_g3c4_error_reset_internal();
  rng = et_g3c4_admit_rng(candidate, 0);
  if (rng == NULL) return 0;
  if (index < 0 || index >= 4) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
    return 0;
  }
  return rng->words[index];
}

int64_t et_g3c4_private_rng_release_v1(void *candidate) {
  et_g3c4_rng_internal *rng;
  et_g3c4_error_reset_internal();
  rng = et_g3c4_admit_rng(candidate, 1);
  if (rng == NULL) return et_g3c4_error_state.category;
  if (rng->transport.state == ET_G3C4_CONTEXT_DEAD) return 0;
  memset(rng->words, 0, sizeof(rng->words));
  rng->transport.state = ET_G3C4_CONTEXT_DEAD;
  return 0;
}

#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
static int64_t et_g3c4_capture_t1_shell(int64_t status) {
  switch (status) {
    case ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT:
      return et_g3c4_fail(
          ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    case ET_T1_I64_SHELL_STATUS_INVALID_STATE:
      return et_g3c4_fail(
          ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    case ET_T1_I64_SHELL_STATUS_RANGE:
      return et_g3c4_fail(
          ET_G3C4_SHAPE_MISMATCH, ET_G3C4_CODE_SHAPE);
    case ET_T1_I64_SHELL_STATUS_ALLOCATION_FAILED:
      return et_g3c4_fail(
          ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
    default:
      return et_g3c4_fail(
          ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
  }
}

static void et_g3c4_input_cleanup_unpublished(
    et_g3c4_input_internal *input) {
  if (input == NULL) return;
  if (input->tensor != NULL) {
    et_i64_tensor_error error;
    if (et_i64_tensor_destroy_v1(&input->tensor, &error) != 0) abort();
  }
  free(input);
}

void *et_g3c4_private_input_from_t1_v1(void *sealed_t1) {
  int64_t words[2] = {0, 0};
  int64_t length;
  int64_t status;
  uint64_t shape[2];
  et_g3c4_input_internal *input;
  et_i64_tensor_error error;
  et_g3c4_error_reset_internal();

  length = et_t1_i64_shell_length_v1(sealed_t1);
  status = et_t1_i64_shell_last_status_v1();
  if (status != ET_T1_I64_SHELL_STATUS_OK) {
    (void)et_g3c4_capture_t1_shell(status);
    return NULL;
  }
  if (length != 1 && length != 2) {
    (void)et_g3c4_fail(ET_G3C4_SHAPE_MISMATCH, ET_G3C4_CODE_SHAPE);
    return NULL;
  }
  for (int64_t index = 0; index < length; index++) {
    words[index] = et_t1_i64_shell_read_v1(sealed_t1, index);
    status = et_t1_i64_shell_last_status_v1();
    if (status != ET_T1_I64_SHELL_STATUS_OK) {
      (void)et_g3c4_capture_t1_shell(status);
      return NULL;
    }
    if (words[index] < 0 || words[index] > 255) {
      (void)et_g3c4_fail(ET_G3C4_SHAPE_MISMATCH, ET_G3C4_CODE_SHAPE);
      return NULL;
    }
  }

  input = et_g3c4_input_allocate();
  if (input == NULL) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
    return NULL;
  }
  shape[0] = 1u;
  shape[1] = (uint64_t)length;
  if (et_g3c4_capture_i64(
          et_i64_tensor_create_v1(2u, shape, &input->tensor, &error),
          &error) != 0) {
    et_g3c4_error_state_internal first =
        et_g3c4_error_snapshot_internal();
    et_g3c4_input_cleanup_unpublished(input);
    et_g3c4_error_restore_internal(first);
    return NULL;
  }
  if (et_g3c4_capture_i64(
          et_i64_tensor_copy_from_v1(
              input->tensor, words, (size_t)length, &error),
          &error) != 0) {
    et_g3c4_error_state_internal first =
        et_g3c4_error_snapshot_internal();
    et_g3c4_input_cleanup_unpublished(input);
    et_g3c4_error_restore_internal(first);
    return NULL;
  }
  input->transport.magic = ET_G3C4_INPUT_MAGIC;
  input->transport.kind = ET_G3C4_INPUT_KIND;
  input->transport.state = ET_G3C4_CONTEXT_LIVE;
  input->length = length;
  et_g3c4_enroll_input(input);
  return input;
}

int64_t et_g3c4_private_tensor_release_v1(void *candidate) {
  et_g3c4_input_internal *input;
  et_i64_tensor_error error;
  et_g3c4_error_reset_internal();
  input = et_g3c4_admit_input(candidate, 1);
  if (input == NULL) return et_g3c4_error_state.category;
  if (input->transport.state == ET_G3C4_CONTEXT_DEAD) return 0;
  if (et_g3c4_capture_i64(
          et_i64_tensor_destroy_v1(&input->tensor, &error), &error) != 0)
    return et_g3c4_error_state.category;
  input->length = 0;
  input->transport.state = ET_G3C4_CONTEXT_DEAD;
  return 0;
}
#endif

static void *et_g3c4_generator_create(
    et_g3c4_model_owner_internal *owner, const int64_t rng_words[4],
    int64_t mode, int64_t temperature_bits, int64_t k,
    int64_t p_bits, int64_t max_new, int64_t eos) {
  et_g3c4_context_internal *context;
  et_a2_kv_cache *cache = NULL;
  et_kernel_error error;
#ifdef ET_G3C4_PROVIDER_ROUTES_PRIVATE
  et_kernel_runtime *forward_runtime = NULL;
  et_kernel_runtime *sampler_runtime = NULL;
#endif
  if (!et_g3c4_valid_generator_policy(
          mode, temperature_bits, k, p_bits, max_new, eos)) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
    return NULL;
  }
#ifdef ET_G3C4_PROVIDER_ROUTES_PRIVATE
  if (et_g3c4_stage_provider_routes(
          et_g3c4_kernel_provider_v1(), et_g3s_kernel_provider_v1(),
          &forward_runtime, &sampler_runtime) != 0)
    return NULL;
#endif
  context = et_g3c4_context_allocate();
  if (context == NULL) {
#ifdef ET_G3C4_PROVIDER_ROUTES_PRIVATE
    et_g3c4_abort_staged_routes(forward_runtime, sampler_runtime);
#endif
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
    return NULL;
  }
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_create_v1(
              1u, 1u, 2u, 4u, 2u, &cache, &error),
          &error) != 0) {
    et_g3c4_error_state_internal first = et_g3c4_error_snapshot_internal();
    free(context);
#ifdef ET_G3C4_PROVIDER_ROUTES_PRIVATE
    et_g3c4_abort_staged_routes(forward_runtime, sampler_runtime);
#endif
    et_g3c4_error_restore_internal(first);
    return NULL;
  }
  context->transport.magic = ET_G3C4_CONTEXT_MAGIC;
  context->transport.kind = ET_G3C4_CONTEXT_KIND;
  context->transport.state = ET_G3C4_CONTEXT_LIVE;
  context->transport.busy = ET_G3C4_CALL_IDLE;
  context->owner = owner;
  context->cache = cache;
  context->generator_kind = 1u;
  context->generator_ready = 1u;
  context->generator_policy[0] = mode;
  context->generator_policy[1] = temperature_bits;
  context->generator_policy[2] = k;
  context->generator_policy[3] = p_bits;
  context->generator_policy[4] = max_new;
  context->generator_policy[5] = eos;
  memcpy(context->generator_rng_words, rng_words,
         sizeof(context->generator_rng_words));
#ifdef ET_G3C4_PROVIDER_ROUTES_PRIVATE
  et_g3c4_forward_runtime = forward_runtime;
  et_g3c4_sampler_runtime = sampler_runtime;
#endif
  et_g3c4_enroll_context(context);
  return context;
}

void *et_g3c4_private_generator_seed_v1(
    void *candidate, int64_t seed, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos) {
  et_g3c4_model_owner_internal *owner;
  int64_t rng_words[4] = {1, seed, 0, 0};
  et_g3c4_error_reset_internal();
  owner = et_g3c4_admit_owner(candidate, 0);
  if (owner == NULL) return NULL;
  if (owner->state != ET_G3C4_OWNER_SEALED || owner->active != NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
  if (seed < 0) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
    return NULL;
  }
  return et_g3c4_generator_create(
      owner, rng_words, mode, temperature_bits, k, p_bits, max_new, eos);
}

void *et_g3c4_private_generator_rng_v1(
    void *candidate, void *rng_candidate, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos) {
  et_g3c4_model_owner_internal *owner;
  et_g3c4_rng_internal *rng;
  int64_t rng_words[4];
  et_g3c4_error_reset_internal();
  owner = et_g3c4_admit_owner(candidate, 0);
  if (owner == NULL) return NULL;
  if (owner->state != ET_G3C4_OWNER_SEALED || owner->active != NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
  rng = et_g3c4_admit_rng(rng_candidate, 0);
  if (rng == NULL) return NULL;
  if (rng->words[0] != 1 || rng->words[1] < 0) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return NULL;
  }
  memcpy(rng_words, rng->words, sizeof(rng_words));
  return et_g3c4_generator_create(
      owner, rng_words, mode, temperature_bits, k, p_bits, max_new, eos);
}

static int et_g3c4_generator_dead_exact(
    const et_g3c4_context_internal *context) {
  return ET_G3C4_CONTEXT_BUSY(context) == ET_G3C4_CALL_IDLE &&
         context->owner == NULL && context->cache == NULL &&
         context->generator_ready == 0u &&
         context->call_kind == 0 && context->budget == 0 &&
         context->acquired_mask == 0u &&
         et_g3c4_pins_idle(&context->pins) &&
         et_g3c4_zero_i64_words(context->generator_policy, 6u) &&
         et_g3c4_zero_i64_words(context->generator_rng_words, 4u)
#ifdef ET_G3C4_PREFILL3_PRIVATE
         && et_g3c4_prefill_binding_valid(context) &&
         context->prefill_binding_ready == 0u
#endif
#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
         && et_g3c4_token_frame_idle(context)
#endif
         ;
}

int64_t et_g3c4_private_generator_close_v1(void *candidate) {
  et_g3c4_context_internal *context;
  et_g3c4_model_owner_internal *owner;
  et_kernel_error error;
  et_g3c4_error_reset_internal();
  context = et_g3c4_admit_context(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
  if (context->generator_kind != 1u)
    return et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  if (ET_G3C4_CONTEXT_STATE(context) == ET_G3C4_CONTEXT_DEAD) {
    if (!et_g3c4_generator_dead_exact(context))
      return et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return 0;
  }
  if (ET_G3C4_CONTEXT_STATE(context) != ET_G3C4_CONTEXT_LIVE ||
      ET_G3C4_CONTEXT_BUSY(context) != ET_G3C4_CALL_IDLE ||
      context->generator_ready != 1u || context->owner == NULL ||
      context->cache == NULL)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (context->call_kind != 0 || context->budget != 0 ||
      context->acquired_mask != 0u || !et_g3c4_pins_idle(&context->pins) ||
#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
      !et_g3c4_token_frame_idle(context) ||
#endif
      !et_g3c4_valid_generator_policy(
          context->generator_policy[0], context->generator_policy[1],
          context->generator_policy[2], context->generator_policy[3],
          context->generator_policy[4], context->generator_policy[5]) ||
      context->generator_rng_words[0] != 1 ||
      context->generator_rng_words[1] < 0)
    return et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
  owner = et_g3c4_admit_owner(context->owner, 0);
  if (owner == NULL) return et_g3c4_error_state.category;
  if (owner != context->owner || owner->state != ET_G3C4_OWNER_SEALED ||
      owner->active != NULL)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_destroy_v1(&context->cache, &error), &error) != 0)
    return et_g3c4_error_state.category;
  memset(context->generator_policy, 0, sizeof(context->generator_policy));
  memset(context->generator_rng_words, 0,
         sizeof(context->generator_rng_words));
#ifdef ET_G3C4_PREFILL3_PRIVATE
  et_g3c4_prefill_binding_clear(context);
#endif
  context->generator_ready = 0u;
  context->owner = NULL;
  ET_G3C4_CONTEXT_STATE(context) = ET_G3C4_CONTEXT_DEAD;
  return 0;
}
#endif

#ifdef ET_G3C4_ACTIVE_CALL_PRIVATE
static int et_g3c4_valid_call_tuple(int64_t call_kind, int64_t budget) {
  return (call_kind == 0 && budget == 0) ||
         (call_kind == 1 && budget == 0) ||
         (call_kind == 2 && (budget == 0 || budget == 1));
}

static int et_g3c4_pins_idle(
    const et_g3c4_model_pins_internal *pins) {
  size_t index;
  if (pins->self != NULL || pins->held_mask != 0u) return 0;
  for (index = 0u; index < 14u; index++) {
    const et_kernel_tensor_view_v1 *view = &pins->views[index];
    if (pins->parameters[index] != NULL || pins->identities[index] != NULL ||
        pins->values[index] != NULL || view->struct_size != 0u ||
        view->data != NULL || view->byte_length != 0u ||
        view->dtype != NULL || view->device != NULL || view->layout != 0u ||
        view->offset_bytes != 0u || view->rank != 0u || view->shape != NULL)
      return 0;
  }
  return 1;
}

static int64_t et_g3c4_cache_idle_preflight(
    et_g3c4_context_internal *context) {
  et_a2_kv_cache_read_borrow *borrow = NULL;
  et_kernel_error error;
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_read_borrow_begin_v1(
              context->cache, &borrow, &error), &error) != 0)
    return et_g3c4_error_state.category;
  if (et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) != 0) abort();
  return 0;
}

static et_g3c4_context_internal *et_g3c4_admit_idle_call(
    void *candidate) {
  et_g3c4_context_internal *context = et_g3c4_admit_context(candidate);
  et_g3c4_model_owner_internal *model;
  if (context == NULL) return NULL;
  if (ET_G3C4_CONTEXT_STATE(context) != ET_G3C4_CONTEXT_LIVE ||
      context->owner == NULL || context->cache == NULL ||
      ET_G3C4_CONTEXT_BUSY(context) != ET_G3C4_CALL_IDLE) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
#ifdef ET_G3C4_GENERATOR_PRIVATE
  if (context->generator_kind != 1u || context->generator_ready != 1u) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
#endif
  model = et_g3c4_admit_owner(context->owner, 0);
  if (model == NULL) return NULL;
  if (model->state != ET_G3C4_OWNER_SEALED || model->active != NULL) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
  if (context->acquired_mask != 0u || context->call_kind != 0 ||
      context->budget != 0 || !et_g3c4_pins_idle(&context->pins)
#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
      || !et_g3c4_token_frame_idle(context)
#endif
      ) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return NULL;
  }
  return context;
}

static et_g3c4_context_internal *et_g3c4_admit_active_call(
    void *candidate) {
  et_g3c4_context_internal *context = et_g3c4_admit_context(candidate);
  et_g3c4_model_owner_internal *model;
  et_f32_tensor_error error;
  if (context == NULL) return NULL;
  if (ET_G3C4_CONTEXT_STATE(context) != ET_G3C4_CONTEXT_LIVE ||
      context->owner == NULL || context->cache == NULL ||
      ET_G3C4_CONTEXT_BUSY(context) != ET_G3C4_CALL_ACTIVE) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
#ifdef ET_G3C4_GENERATOR_PRIVATE
  if (context->generator_kind != 1u || context->generator_ready != 1u) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return NULL;
  }
#endif
  model = et_g3c4_admit_owner(context->owner, 0);
  if (model == NULL) return NULL;
  if (model->state != ET_G3C4_OWNER_SEALED || model->active != context) {
    (void)et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return NULL;
  }
  if (context->acquired_mask != ET_G3C4_ACQUIRED_FULL ||
      !et_g3c4_valid_call_tuple(context->call_kind, context->budget) ||
      context->pins.self != &context->pins ||
      context->pins.held_mask != UINT16_C(0x3fff)) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return NULL;
  }
  if (et_g3c4_capture_f32(
          et_g3c4_model_pins_check_internal(&context->pins, &error),
          &error) != 0)
    return NULL;
  return context;
}

#ifdef ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE
typedef struct et_g3c4_full_prefix_scratch {
  float et[16], ep[16], x[16], n1[16];
  float qt[16], kt[16], vt[16];
  float qh[16], kh[16], vh[16], ah[16];
  float at[16], ao[16], r[16], n2[16];
  float fu[32], fg[32], fd[16], y[16], nf[16], z[1024];
} et_g3c4_full_prefix_scratch;

static int et_g3c4_dispatch_forward(
    const char *capability, const char *operation,
    size_t rank, const uint64_t *shape,
    et_kernel_tensor_view_v1 *inputs, size_t input_count,
    et_kernel_tensor_view_v1 output) {
  et_kernel_request_v1 request = {
    sizeof(request), operation, "f32", "cpu", rank, shape, 1u, {0}};
  et_kernel_call_v1 call = {
    sizeof(call), capability, &request,
    input_count, sizeof(inputs[0]), input_count * sizeof(inputs[0]), inputs,
    1u, sizeof(output), sizeof(output), &output};
  et_kernel_error error;
  return et_g3c4_capture_kernel(
      et_kernel_runtime_dispatch(et_g3c4_forward_runtime, &call, &error),
      &error);
}

int64_t et_g3c4_private_full_prefix_forward_v1(
    void *candidate, const int64_t token_ids[4], float logits[1024]) {
  static const uint64_t ids_shape[2] = {1u, 4u};
  static const uint64_t token_embedding_row[4] = {1u, 4u, 256u, 4u};
  static const uint64_t position_embedding_row[4] = {1u, 4u, 4u, 4u};
  static const uint64_t d4_shape[3] = {1u, 4u, 4u};
  static const uint64_t d8_shape[3] = {1u, 4u, 8u};
  static const uint64_t heads_shape[4] = {1u, 2u, 4u, 2u};
  static const uint64_t head_layout_row[4] = {1u, 4u, 2u, 2u};
  static const uint64_t linear_d4_d4[4] = {1u, 4u, 4u, 4u};
  static const uint64_t linear_d4_d8[4] = {1u, 4u, 4u, 8u};
  static const uint64_t linear_d8_d4[4] = {1u, 4u, 8u, 4u};
  static const uint64_t linear_d4_v256[4] = {1u, 4u, 4u, 256u};
  static const uint64_t attention_row[6] = {1u, 2u, 2u, 4u, 4u, 2u};
  static const uint64_t attention_positions_shape[2] = {1u, 4u};
  static const uint64_t attention_mask_shape[3] = {1u, 4u, 4u};
  static const uint64_t append_shape[1] = {1u};
  et_g3c4_context_internal *context;
  et_g3c4_full_prefix_scratch scratch;
  et_a2_kv_cache_transaction *transaction = NULL;
  et_a2_kv_cache_transaction_view *transaction_view = NULL;
  const et_kernel_tensor_view_v1 *cached_keys = NULL;
  const et_kernel_tensor_view_v1 *cached_values = NULL;
  const et_kernel_tensor_view_v1 *effective_lengths = NULL;
  const et_kernel_tensor_view_v1 *key_keep = NULL;
  et_kernel_tensor_view_v1 inputs[6];
  et_kernel_tensor_view_v1 output;
  et_kernel_tensor_view_v1 append_counts;
  et_kernel_tensor_view_v1 staged_keys;
  et_kernel_tensor_view_v1 staged_values;
  et_kernel_error error;
  et_g3c4_error_state_internal first;
  int64_t positions[4] = {0, 1, 2, 3};
  int64_t append_count = 4;
  uint8_t causal_keep[16];
  uint32_t epsilon_bits = UINT32_C(0x3727c5ac);
  float epsilon;
  size_t query;
  size_t key;

  et_g3c4_error_reset_internal();
  if (token_ids == NULL || logits == NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  for (query = 0u; query < 4u; query++)
    if (token_ids[query] < 0 || token_ids[query] > 255) {
      (void)et_g3c4_fail(
          ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
      return et_g3c4_error_state.category;
    }
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
  if ((context->call_kind != 0 && context->call_kind != 2) ||
      et_g3c4_forward_runtime == NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return et_g3c4_error_state.category;
  }
  memset(&scratch, 0, sizeof(scratch));
  memcpy(&epsilon, &epsilon_bits, sizeof(epsilon));

  inputs[0] = et_g3c4_view(
      (void *)token_ids, 4u * sizeof(token_ids[0]), "i64", 2u, ids_shape);
  inputs[1] = context->pins.views[10];
  output = et_g3c4_view(
      scratch.et, sizeof(scratch.et), "f32", 3u, d4_shape);
  if (et_g3c4_dispatch_forward(
          "g3c4.embedding-forward", "g3c4.embedding.forward",
          4u, token_embedding_row, inputs, 2u, output) != 0)
    goto fail;

  inputs[0] = et_g3c4_view(
      positions, sizeof(positions), "i64", 2u, ids_shape);
  inputs[1] = context->pins.views[13];
  output = et_g3c4_view(
      scratch.ep, sizeof(scratch.ep), "f32", 3u, d4_shape);
  if (et_g3c4_dispatch_forward(
          "g3c4.embedding-forward", "g3c4.embedding.forward",
          4u, position_embedding_row, inputs, 2u, output) != 0)
    goto fail;

  inputs[0] = et_g3c4_view(
      scratch.et, sizeof(scratch.et), "f32", 3u, d4_shape);
  inputs[1] = et_g3c4_view(
      scratch.ep, sizeof(scratch.ep), "f32", 3u, d4_shape);
  output = et_g3c4_view(scratch.x, sizeof(scratch.x), "f32", 3u, d4_shape);
  if (et_g3c4_dispatch_forward(
          "g3c4.residual", "g3c4.residual.forward",
          3u, d4_shape, inputs, 2u, output) != 0)
    goto fail;

  inputs[0] = output;
  inputs[1] = context->pins.views[7];
  inputs[2] = context->pins.views[6];
  inputs[3] = et_g3c4_view(&epsilon, sizeof(epsilon), "f32", 0u, NULL);
  output = et_g3c4_view(
      scratch.n1, sizeof(scratch.n1), "f32", 3u, d4_shape);
  if (et_g3c4_dispatch_forward(
          "g3c4.layer-norm", "g3c4.layer-norm.forward",
          3u, d4_shape, inputs, 4u, output) != 0)
    goto fail;

#define ET_G3C4_FORWARD_LINEAR(weight_index, source, target, row) do { \
  inputs[0] = et_g3c4_view( \
      (source), sizeof(source), "f32", 3u, \
      sizeof(source) == sizeof(scratch.fu) ? d8_shape : d4_shape); \
  inputs[1] = context->pins.views[(weight_index)]; \
  output = et_g3c4_view( \
      (target), sizeof(target), "f32", 3u, \
      sizeof(target) == sizeof(scratch.fu) ? d8_shape : \
      (sizeof(target) == sizeof(scratch.z) ? \
          (const uint64_t[3]){1u, 4u, 256u} : d4_shape)); \
  if (et_g3c4_dispatch_forward( \
          "g3c4.linear", "g3c4.linear.forward-no-bias", \
          4u, (row), inputs, 2u, output) != 0) \
    goto fail; \
} while (0)

  ET_G3C4_FORWARD_LINEAR(2u, scratch.n1, scratch.qt, linear_d4_d4);
  ET_G3C4_FORWARD_LINEAR(0u, scratch.n1, scratch.kt, linear_d4_d4);
  ET_G3C4_FORWARD_LINEAR(3u, scratch.n1, scratch.vt, linear_d4_d4);

#define ET_G3C4_FORWARD_LAYOUT( \
    operation, source, source_rank, source_shape, \
    target, target_rank, target_shape) do { \
  inputs[0] = et_g3c4_view( \
      (source), sizeof(source), "f32", (source_rank), (source_shape)); \
  output = et_g3c4_view( \
      (target), sizeof(target), "f32", (target_rank), (target_shape)); \
  if (et_g3c4_dispatch_forward( \
          "g3c4.head-layout", (operation), 4u, head_layout_row, \
          inputs, 1u, output) != 0) \
    goto fail; \
} while (0)

  ET_G3C4_FORWARD_LAYOUT(
      "g3c4.heads.split.forward", scratch.qt, 3u, d4_shape,
      scratch.qh, 4u, heads_shape);
  ET_G3C4_FORWARD_LAYOUT(
      "g3c4.heads.split.forward", scratch.kt, 3u, d4_shape,
      scratch.kh, 4u, heads_shape);
  ET_G3C4_FORWARD_LAYOUT(
      "g3c4.heads.split.forward", scratch.vt, 3u, d4_shape,
      scratch.vh, 4u, heads_shape);

  append_counts = et_g3c4_view(
      &append_count, sizeof(append_count), "i64", 1u, append_shape);
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_begin_v1(
              context->cache, 4u, &append_counts, &transaction, &error),
          &error) != 0)
    goto fail;
  staged_keys = et_g3c4_view(
      scratch.kh, sizeof(scratch.kh), "f32", 4u, heads_shape);
  staged_values = et_g3c4_view(
      scratch.vh, sizeof(scratch.vh), "f32", 4u, heads_shape);
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_stage_layer_v1(
              transaction, 0u, &staged_keys, &staged_values, &error),
          &error) != 0)
    goto fail;
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_view_begin_v1(
              transaction, 0u, &transaction_view, &error),
          &error) != 0)
    goto fail;
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_view_tensors_v1(
              transaction_view, &cached_keys, &cached_values,
              &effective_lengths, &key_keep, &error),
          &error) != 0)
    goto fail;
  if (effective_lengths == NULL || key_keep == NULL) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    goto fail;
  }
  for (query = 0u; query < 4u; query++)
    for (key = 0u; key < 4u; key++)
      causal_keep[query * 4u + key] =
          ((const uint8_t *)key_keep->data)[key] != 0u &&
          positions[key] <= positions[query] ? 1u : 0u;
  inputs[0] = et_g3c4_view(
      scratch.qh, sizeof(scratch.qh), "f32", 4u, heads_shape);
  inputs[1] = *cached_keys;
  inputs[2] = *cached_values;
  inputs[3] = et_g3c4_view(
      positions, sizeof(positions), "i64", 2u, attention_positions_shape);
  inputs[4] = inputs[3];
  inputs[5] = et_g3c4_view(
      causal_keep, sizeof(causal_keep), "bool", 3u,
      attention_mask_shape);
  output = et_g3c4_view(
      scratch.ah, sizeof(scratch.ah), "f32", 4u, heads_shape);
  if (et_g3c4_dispatch_forward(
          "g3c4.causal-attention", "g3c4.causal-attention.forward",
          6u, attention_row, inputs, 6u, output) != 0)
    goto fail;
  if (et_a2_kv_cache_transaction_view_end_v1(
          &transaction_view, &error) != 0)
    abort();

  ET_G3C4_FORWARD_LAYOUT(
      "g3c4.heads.merge.forward", scratch.ah, 4u, heads_shape,
      scratch.at, 3u, d4_shape);
  ET_G3C4_FORWARD_LINEAR(1u, scratch.at, scratch.ao, linear_d4_d4);

  inputs[0] = et_g3c4_view(
      scratch.x, sizeof(scratch.x), "f32", 3u, d4_shape);
  inputs[1] = et_g3c4_view(
      scratch.ao, sizeof(scratch.ao), "f32", 3u, d4_shape);
  output = et_g3c4_view(scratch.r, sizeof(scratch.r), "f32", 3u, d4_shape);
  if (et_g3c4_dispatch_forward(
          "g3c4.residual", "g3c4.residual.forward",
          3u, d4_shape, inputs, 2u, output) != 0)
    goto fail;

  inputs[0] = output;
  inputs[1] = context->pins.views[9];
  inputs[2] = context->pins.views[8];
  inputs[3] = et_g3c4_view(&epsilon, sizeof(epsilon), "f32", 0u, NULL);
  output = et_g3c4_view(
      scratch.n2, sizeof(scratch.n2), "f32", 3u, d4_shape);
  if (et_g3c4_dispatch_forward(
          "g3c4.layer-norm", "g3c4.layer-norm.forward",
          3u, d4_shape, inputs, 4u, output) != 0)
    goto fail;

  ET_G3C4_FORWARD_LINEAR(5u, scratch.n2, scratch.fu, linear_d4_d8);
  inputs[0] = et_g3c4_view(
      scratch.fu, sizeof(scratch.fu), "f32", 3u, d8_shape);
  output = et_g3c4_view(
      scratch.fg, sizeof(scratch.fg), "f32", 3u, d8_shape);
  if (et_g3c4_dispatch_forward(
          "g3c4.gelu", "g3c4.gelu.forward",
          3u, d8_shape, inputs, 1u, output) != 0)
    goto fail;
  ET_G3C4_FORWARD_LINEAR(4u, scratch.fg, scratch.fd, linear_d8_d4);

  inputs[0] = et_g3c4_view(
      scratch.r, sizeof(scratch.r), "f32", 3u, d4_shape);
  inputs[1] = et_g3c4_view(
      scratch.fd, sizeof(scratch.fd), "f32", 3u, d4_shape);
  output = et_g3c4_view(scratch.y, sizeof(scratch.y), "f32", 3u, d4_shape);
  if (et_g3c4_dispatch_forward(
          "g3c4.residual", "g3c4.residual.forward",
          3u, d4_shape, inputs, 2u, output) != 0)
    goto fail;

  inputs[0] = output;
  inputs[1] = context->pins.views[12];
  inputs[2] = context->pins.views[11];
  inputs[3] = et_g3c4_view(&epsilon, sizeof(epsilon), "f32", 0u, NULL);
  output = et_g3c4_view(
      scratch.nf, sizeof(scratch.nf), "f32", 3u, d4_shape);
  if (et_g3c4_dispatch_forward(
          "g3c4.layer-norm", "g3c4.layer-norm.forward",
          3u, d4_shape, inputs, 4u, output) != 0)
    goto fail;
  ET_G3C4_FORWARD_LINEAR(10u, scratch.nf, scratch.z, linear_d4_v256);

#undef ET_G3C4_FORWARD_LAYOUT
#undef ET_G3C4_FORWARD_LINEAR
  if (et_a2_kv_cache_transaction_abort_v1(&transaction, &error) != 0)
    abort();
  memcpy(logits, scratch.z, sizeof(scratch.z));
  return 0;

fail:
  first = et_g3c4_error_snapshot_internal();
  if (transaction_view != NULL &&
      et_a2_kv_cache_transaction_view_end_v1(
          &transaction_view, &error) != 0)
    abort();
  if (transaction != NULL &&
      et_a2_kv_cache_transaction_abort_v1(&transaction, &error) != 0)
    abort();
  et_g3c4_error_restore_internal(first);
  return et_g3c4_error_state.category;
}
#endif

#ifdef ET_G3C4_GENERATOR_PRIVATE
#ifdef ET_G3C4_SAMPLER_TRANSPORT_PRIVATE
static int et_g3c4_range_valid(const void *pointer, size_t bytes) {
  const uintptr_t start = (uintptr_t)pointer;
  return pointer != NULL && start <= UINTPTR_MAX - bytes;
}

static int et_g3c4_ranges_overlap(
    const void *left, size_t left_bytes,
    const void *right, size_t right_bytes) {
  const uintptr_t left_start = (uintptr_t)left;
  const uintptr_t right_start = (uintptr_t)right;
  return left_start < right_start + right_bytes &&
         right_start < left_start + left_bytes;
}

int64_t et_g3c4_private_sample_last_v1(
    void *candidate, const float full_logits[1024],
    int64_t *token_output, int64_t successor_output[4]) {
  static const uint64_t logits_shape[2] = {1u, 256u};
  static const uint64_t rng_shape[1] = {4u};
  static const uint64_t token_shape[1] = {1u};
  et_g3c4_context_internal *context;
  float last_logits[256];
  float temperature;
  float top_p;
  int64_t top_k;
  int64_t numeric_rng[4];
  int64_t token_candidate = -1;
  int64_t successor_candidate[4] = {0, 0, 0, 0};
  et_kernel_tensor_view_v1 inputs[5];
  et_kernel_tensor_view_v1 outputs[2];
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
  et_kernel_error error;
  size_t input_count;
  uint32_t temperature_bits;
  uint32_t top_p_bits;
  int categorical;

  et_g3c4_error_reset_internal();
  if (!et_g3c4_range_valid(full_logits, 1024u * sizeof(float)) ||
      !et_g3c4_range_valid(token_output, sizeof(*token_output)) ||
      !et_g3c4_range_valid(successor_output,
                           4u * sizeof(successor_output[0])) ||
      (uintptr_t)full_logits % _Alignof(float) != 0u ||
      (uintptr_t)token_output % _Alignof(int64_t) != 0u ||
      (uintptr_t)successor_output % _Alignof(int64_t) != 0u) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
  if (context->call_kind != 2 || context->budget != 1 ||
      et_g3c4_sampler_runtime == NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return et_g3c4_error_state.category;
  }
  if (et_g3c4_ranges_overlap(
          token_output, sizeof(*token_output),
          successor_output, 4u * sizeof(successor_output[0])) ||
      et_g3c4_ranges_overlap(
          token_output, sizeof(*token_output),
          full_logits, 1024u * sizeof(float)) ||
      et_g3c4_ranges_overlap(
          successor_output, 4u * sizeof(successor_output[0]),
          full_logits, 1024u * sizeof(float)) ||
      et_g3c4_ranges_overlap(
          token_output, sizeof(*token_output), context, sizeof(*context)) ||
      et_g3c4_ranges_overlap(
          successor_output, 4u * sizeof(successor_output[0]),
          context, sizeof(*context))) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }

  memcpy(last_logits, full_logits + 3u * 256u, sizeof(last_logits));
  memcpy(numeric_rng, context->generator_rng_words, sizeof(numeric_rng));
  categorical = context->generator_policy[0] == 1;
  temperature_bits = (uint32_t)context->generator_policy[1];
  top_k = context->generator_policy[2];
  top_p_bits = (uint32_t)context->generator_policy[3];
  memcpy(&temperature, &temperature_bits, sizeof(temperature));
  memcpy(&top_p, &top_p_bits, sizeof(top_p));

  inputs[0] = et_g3c4_view(
      last_logits, sizeof(last_logits), "f32", 2u, logits_shape);
  if (categorical) {
    inputs[1] = et_g3c4_view(
        &temperature, sizeof(temperature), "f32", 0u, NULL);
    inputs[2] = et_g3c4_view(&top_k, sizeof(top_k), "i64", 0u, NULL);
    inputs[3] = et_g3c4_view(&top_p, sizeof(top_p), "f32", 0u, NULL);
    inputs[4] = et_g3c4_view(
        numeric_rng, sizeof(numeric_rng), "i64", 1u, rng_shape);
    input_count = 5u;
  } else {
    inputs[1] = et_g3c4_view(
        numeric_rng, sizeof(numeric_rng), "i64", 1u, rng_shape);
    input_count = 2u;
  }
  outputs[0] = et_g3c4_view(
      &token_candidate, sizeof(token_candidate), "i64", 1u, token_shape);
  outputs[1] = et_g3c4_view(
      successor_candidate, sizeof(successor_candidate),
      "i64", 1u, rng_shape);
  request = (et_kernel_request_v1){
    sizeof(request),
    categorical ? "g3s.categorical.forward" : "g3s.greedy.forward",
    "f32", "cpu", 2u, logits_shape, 1u, {0}};
  call = (et_kernel_call_v1){
    sizeof(call), categorical ? "g3s.categorical" : "g3s.greedy",
    &request, input_count, sizeof(inputs[0]), input_count * sizeof(inputs[0]),
    inputs, 2u, sizeof(outputs[0]), sizeof(outputs), outputs};
  if (et_g3c4_capture_kernel(
          et_kernel_runtime_dispatch(
              et_g3c4_sampler_runtime, &call, &error),
          &error) != 0)
    return et_g3c4_error_state.category;
  if (token_candidate < 0 || token_candidate > 255 ||
      successor_candidate[0] != 1 ||
      successor_candidate[1] != numeric_rng[1]) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return et_g3c4_error_state.category;
  }
  memcpy(token_output, &token_candidate, sizeof(token_candidate));
  memcpy(successor_output, successor_candidate, sizeof(successor_candidate));
  return 0;
}
#endif

#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
#ifdef ET_G3C4_PREFILL3_PRIVATE
static int et_g3c4_prefill_binding_matches_pins(
    const et_g3c4_context_internal *context) {
  size_t index;
  size_t offset = 0u;
  if (context->prefill_binding_ready != 1u) return 0;
  for (index = 0u; index < 14u; index++) {
    const et_kernel_tensor_view_v1 *view = &context->pins.views[index];
    if (context->prefill_binding_identities[index] !=
            context->pins.identities[index] ||
        view->data == NULL ||
        view->byte_length > sizeof(context->prefill_binding_values) - offset ||
        memcmp(context->prefill_binding_values + offset,
               view->data, view->byte_length) != 0)
      return 0;
    offset += view->byte_length;
  }
  return offset == sizeof(context->prefill_binding_values);
}

static int et_g3c4_prefill_reject_owned_aliases(
    et_g3c4_context_internal *context, const int64_t token_ids[3],
    float last_logits_output[256]) {
  size_t index;
  int overlap = 0;

  if (et_g3c4_ranges_overlap(
          token_ids, 3u * sizeof(token_ids[0]),
          context->owner, sizeof(*context->owner)) ||
      et_g3c4_ranges_overlap(
          last_logits_output, 256u * sizeof(last_logits_output[0]),
          context->owner, sizeof(*context->owner)))
    overlap = 1;
  for (index = 0u; index < 14u; index++) {
    const et_kernel_tensor_view_v1 *view = &context->pins.views[index];
    if (et_g3c4_ranges_overlap(
            token_ids, 3u * sizeof(token_ids[0]),
            view->data, view->byte_length) ||
        et_g3c4_ranges_overlap(
            last_logits_output, 256u * sizeof(last_logits_output[0]),
            view->data, view->byte_length))
      overlap = 1;
  }
  if (et_a2_kv_cache_private_storage_overlap_v1(
          token_ids, 3u * sizeof(token_ids[0])) != 0 ||
      et_a2_kv_cache_private_storage_overlap_v1(
          last_logits_output,
          256u * sizeof(last_logits_output[0])) != 0)
    overlap = 1;
  if (overlap) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return -1;
  }
  return 0;
}
#endif

static void et_g3c4_token_frame_reset(
    et_g3c4_context_internal *context) {
  context->token_frame_transaction = NULL;
  context->token_frame_state = ET_G3C4_TOKEN_FRAME_IDLE;
  context->token_frame_candidate = 0;
  memset(context->token_frame_successor, 0,
         sizeof(context->token_frame_successor));
#ifdef ET_G3C4_TOKEN_FORWARD_PRIVATE
  context->token_frame_position = 0;
#endif
}

static void et_g3c4_token_frame_discard(
    et_g3c4_context_internal *context) {
  et_kernel_error error;
  if (context->token_frame_transaction != NULL &&
      et_a2_kv_cache_transaction_abort_v1(
          &context->token_frame_transaction, &error) != 0)
    abort();
  et_g3c4_token_frame_reset(context);
}

int64_t et_g3c4_private_token_frame_begin_v1(
    void *candidate, const float full_logits[1024],
    int64_t *speculative_token_output) {
  static const uint64_t append_shape[1] = {1u};
  et_g3c4_context_internal *context;
  et_a2_kv_cache_transaction *transaction = NULL;
  et_kernel_tensor_view_v1 append_counts;
  et_kernel_error error;
  int64_t token_candidate = -1;
  int64_t successor_candidate[4] = {0, 0, 0, 0};
  int64_t append_count = 1;
#ifdef ET_G3C4_TOKEN_FORWARD_PRIVATE
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *borrowed_keys = NULL;
  const et_kernel_tensor_view_v1 *borrowed_values = NULL;
  const et_kernel_tensor_view_v1 *committed_lengths = NULL;
  const et_kernel_tensor_view_v1 *committed_keep = NULL;
  int64_t token_position;
  et_g3c4_error_state_internal first;
#endif

  et_g3c4_error_reset_internal();
  if (!et_g3c4_range_valid(full_logits, 1024u * sizeof(float)) ||
      !et_g3c4_range_valid(
          speculative_token_output, sizeof(*speculative_token_output)) ||
      (uintptr_t)full_logits % _Alignof(float) != 0u ||
      (uintptr_t)speculative_token_output % _Alignof(int64_t) != 0u) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
  if (context->call_kind != 2 || context->budget != 1 ||
      !et_g3c4_token_frame_idle(context)) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return et_g3c4_error_state.category;
  }
#ifdef ET_G3C4_PREFILL3_PRIVATE
  if (!et_g3c4_prefill_binding_matches_pins(context)) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_STALE_BINDING);
    return et_g3c4_error_state.category;
  }
#endif
  if (et_g3c4_ranges_overlap(
          speculative_token_output, sizeof(*speculative_token_output),
          full_logits, 1024u * sizeof(float)) ||
      et_g3c4_ranges_overlap(
          speculative_token_output, sizeof(*speculative_token_output),
          context, sizeof(*context))) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  if (et_g3c4_private_sample_last_v1(
          context, full_logits, &token_candidate,
          successor_candidate) != 0)
    return et_g3c4_error_state.category;
#ifdef ET_G3C4_TOKEN_FORWARD_PRIVATE
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_read_borrow_begin_v1(
              context->cache, &borrow, &error), &error) != 0)
    return et_g3c4_error_state.category;
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_read_borrow_layer_v1(
              borrow, 0u, &borrowed_keys, &borrowed_values,
              &committed_lengths, &committed_keep, &error),
          &error) != 0) {
    first = et_g3c4_error_snapshot_internal();
    if (et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) != 0) abort();
    et_g3c4_error_restore_internal(first);
    return et_g3c4_error_state.category;
  }
  if (borrowed_keys == NULL || borrowed_values == NULL ||
      committed_lengths == NULL || committed_keep == NULL ||
      committed_lengths->data == NULL ||
      *(const int64_t *)committed_lengths->data < 0 ||
      *(const int64_t *)committed_lengths->data > 3) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    first = et_g3c4_error_snapshot_internal();
    if (et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) != 0) abort();
    et_g3c4_error_restore_internal(first);
    return et_g3c4_error_state.category;
  }
  token_position = *(const int64_t *)committed_lengths->data;
  if (et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) != 0) abort();
#endif
  append_counts = et_g3c4_view(
      &append_count, sizeof(append_count), "i64", 1u, append_shape);
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_begin_v1(
              context->cache, 1u, &append_counts, &transaction, &error),
          &error) != 0)
    return et_g3c4_error_state.category;
  context->token_frame_transaction = transaction;
  context->token_frame_state = ET_G3C4_TOKEN_FRAME_SAMPLED;
  context->token_frame_candidate = token_candidate;
  memcpy(context->token_frame_successor, successor_candidate,
         sizeof(successor_candidate));
#ifdef ET_G3C4_TOKEN_FORWARD_PRIVATE
  context->token_frame_position = token_position;
#endif
  memcpy(speculative_token_output, &token_candidate,
         sizeof(token_candidate));
  return 0;
}

int64_t et_g3c4_private_token_frame_stage_v1(
    void *candidate, int64_t speculative_token,
    const float keys[4], const float values[4]) {
  static const uint64_t kv_shape[4] = {1u, 2u, 1u, 2u};
  et_g3c4_context_internal *context;
  et_kernel_tensor_view_v1 key_view;
  et_kernel_tensor_view_v1 value_view;
  et_kernel_error error;
  et_g3c4_error_state_internal first;

  et_g3c4_error_reset_internal();
  if (!et_g3c4_range_valid(keys, 4u * sizeof(float)) ||
      !et_g3c4_range_valid(values, 4u * sizeof(float)) ||
      (uintptr_t)keys % _Alignof(float) != 0u ||
      (uintptr_t)values % _Alignof(float) != 0u) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
  if (context->call_kind != 2 || context->budget != 1 ||
      context->token_frame_state != ET_G3C4_TOKEN_FRAME_SAMPLED ||
      context->token_frame_transaction == NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return et_g3c4_error_state.category;
  }
  if (speculative_token != context->token_frame_candidate) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
    return et_g3c4_error_state.category;
  }
  key_view = et_g3c4_view(
      (void *)keys, 4u * sizeof(float), "f32", 4u, kv_shape);
  value_view = et_g3c4_view(
      (void *)values, 4u * sizeof(float), "f32", 4u, kv_shape);
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_stage_layer_v1(
              context->token_frame_transaction, 0u,
              &key_view, &value_view, &error),
          &error) != 0) {
    first = et_g3c4_error_snapshot_internal();
    et_g3c4_token_frame_discard(context);
    et_g3c4_error_restore_internal(first);
    return et_g3c4_error_state.category;
  }
  context->token_frame_state = ET_G3C4_TOKEN_FRAME_READY;
  return 0;
}

int64_t et_g3c4_private_token_frame_publish_v1(
    void *candidate, int64_t *token_output) {
  et_g3c4_context_internal *context;
  et_kernel_error error;
  int64_t token_candidate;
  int64_t successor_candidate[4];

  et_g3c4_error_reset_internal();
  if (!et_g3c4_range_valid(token_output, sizeof(*token_output)) ||
      (uintptr_t)token_output % _Alignof(int64_t) != 0u) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
  if (context->call_kind != 2 || context->budget != 1 ||
      context->token_frame_state != ET_G3C4_TOKEN_FRAME_READY ||
      context->token_frame_transaction == NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return et_g3c4_error_state.category;
  }
  if (et_g3c4_ranges_overlap(
          token_output, sizeof(*token_output), context, sizeof(*context))) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  token_candidate = context->token_frame_candidate;
  memcpy(successor_candidate, context->token_frame_successor,
         sizeof(successor_candidate));
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_commit_v1(
              &context->token_frame_transaction, &error),
          &error) != 0)
    return et_g3c4_error_state.category;

  memcpy(context->generator_rng_words, successor_candidate,
         sizeof(successor_candidate));
  memcpy(token_output, &token_candidate, sizeof(token_candidate));
  context->budget = 0;
  et_g3c4_token_frame_reset(context);
  return 0;
}

int64_t et_g3c4_private_token_frame_abort_v1(void *candidate) {
  et_g3c4_context_internal *context;
  et_g3c4_error_reset_internal();
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
  if (context->call_kind != 2)
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  if (et_g3c4_token_frame_idle(context)) return 0;
  et_g3c4_token_frame_discard(context);
  return 0;
}
#endif

#ifdef ET_G3C4_TOKEN_FORWARD_PRIVATE
typedef struct et_g3c4_token_forward_scratch {
  float et[4], ep[4], x[4], n1[4];
  float qt[4], kt[4], vt[4];
  float qh[4], kh[4], vh[4], ah[4];
  float at[4], ao[4], r[4], n2[4];
  float fu[8], fg[8], fd[4], y[4], nf[4], z[256];
} et_g3c4_token_forward_scratch;

static int et_g3c4_token_runtime_discover(
    const et_kernel_provider_v1 *provider,
    et_kernel_runtime **runtime_output) {
  et_kernel_error error;
  return et_g3c4_capture_kernel(
      et_kernel_runtime_discover(
          et_g3c4_resolve_provider, (void *)provider,
          runtime_output, &error), &error);
}

static int et_g3c4_token_dispatch(
    et_kernel_runtime *runtime, const char *capability,
    const char *operation, size_t rank, const uint64_t *shape,
    et_kernel_tensor_view_v1 *inputs, size_t input_count,
    et_kernel_tensor_view_v1 output) {
  et_kernel_request_v1 request = {
    sizeof(request), operation, "f32", "cpu", rank, shape, 1u, {0}};
  et_kernel_call_v1 call = {
    sizeof(call), capability, &request,
    input_count, sizeof(inputs[0]), input_count * sizeof(inputs[0]), inputs,
    1u, sizeof(output), sizeof(output), &output};
  et_kernel_error error;
  return et_g3c4_capture_kernel(
      et_kernel_runtime_dispatch(runtime, &call, &error), &error);
}

int64_t et_g3c4_private_token_forward_v1(
    void *candidate, int64_t speculative_token,
    float logits_output[256]) {
  static const uint64_t ids_shape[2] = {1u, 1u};
  static const uint64_t token_embedding_row[4] = {1u, 1u, 256u, 4u};
  static const uint64_t position_embedding_row[4] = {1u, 1u, 4u, 4u};
  static const uint64_t d4_shape[3] = {1u, 1u, 4u};
  static const uint64_t d8_shape[3] = {1u, 1u, 8u};
  static const uint64_t heads_shape[4] = {1u, 2u, 1u, 2u};
  static const uint64_t head_layout_row[4] = {1u, 1u, 2u, 2u};
  static const uint64_t linear_d4_d4[4] = {1u, 1u, 4u, 4u};
  static const uint64_t linear_d4_d8[4] = {1u, 1u, 4u, 8u};
  static const uint64_t linear_d8_d4[4] = {1u, 1u, 8u, 4u};
  static const uint64_t linear_d4_v256[4] = {1u, 1u, 4u, 256u};
  static const uint64_t attention_row[6] = {1u, 2u, 2u, 1u, 4u, 2u};
  static const uint64_t attention_query_shape[2] = {1u, 1u};
  static const uint64_t attention_key_shape[2] = {1u, 4u};
  static const uint64_t attention_mask_shape[3] = {1u, 1u, 4u};
  et_g3c4_context_internal *context;
  et_g3c4_token_forward_scratch scratch;
  et_kernel_runtime *g3n_runtime = NULL;
  et_kernel_runtime *n2_runtime = NULL;
  et_a2_kv_cache_transaction_view *transaction_view = NULL;
  const et_kernel_tensor_view_v1 *cached_keys = NULL;
  const et_kernel_tensor_view_v1 *cached_values = NULL;
  const et_kernel_tensor_view_v1 *effective_lengths = NULL;
  const et_kernel_tensor_view_v1 *key_keep = NULL;
  et_kernel_tensor_view_v1 inputs[6];
  et_kernel_tensor_view_v1 output;
  et_kernel_error error;
  et_g3c4_error_state_internal first;
  int64_t token_id = speculative_token;
  int64_t query_position;
  int64_t key_positions[4] = {0, 1, 2, 3};
  uint8_t causal_keep[4];
  uint32_t epsilon_bits = UINT32_C(0x3727c5ac);
  float epsilon;
  size_t key;

  et_g3c4_error_reset_internal();
  if (!et_g3c4_range_valid(logits_output, 256u * sizeof(float)) ||
      (uintptr_t)logits_output % _Alignof(float) != 0u) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
  if (context->call_kind != 2 || context->budget != 1 ||
      context->token_frame_state != ET_G3C4_TOKEN_FRAME_SAMPLED ||
      context->token_frame_transaction == NULL) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return et_g3c4_error_state.category;
  }
  if (speculative_token != context->token_frame_candidate) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
    return et_g3c4_error_state.category;
  }
  if (et_g3c4_ranges_overlap(
          logits_output, 256u * sizeof(float), context, sizeof(*context))) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  memset(&scratch, 0, sizeof(scratch));
  memcpy(&epsilon, &epsilon_bits, sizeof(epsilon));
  if (et_g3c4_token_runtime_discover(
          et_g3n_kernel_provider_v1(), &g3n_runtime) != 0)
    goto fail;
  if (et_g3c4_token_runtime_discover(
          et_n2_kernel_provider_v1(), &n2_runtime) != 0)
    goto fail;

  inputs[0] = et_g3c4_view(
      &token_id, sizeof(token_id), "i64", 2u, ids_shape);
  inputs[1] = context->pins.views[10];
  output = et_g3c4_view(
      scratch.et, sizeof(scratch.et), "f32", 3u, d4_shape);
  if (et_g3c4_token_dispatch(
          g3n_runtime, "g3n.embedding-forward", "g3n.embedding.forward",
          4u, token_embedding_row, inputs, 2u, output) != 0)
    goto fail;

  query_position = context->token_frame_position;
  if (et_g3c4_token_dispatch(
          et_g3c4_forward_runtime, "g3c4.embedding-forward",
          "g3c4.embedding.forward", 4u, position_embedding_row,
          (et_kernel_tensor_view_v1[2]){
            et_g3c4_view(&query_position, sizeof(query_position),
                          "i64", 2u, ids_shape),
            context->pins.views[13]}, 2u,
          et_g3c4_view(scratch.ep, sizeof(scratch.ep),
                       "f32", 3u, d4_shape)) != 0)
    goto fail;
  if (et_g3c4_token_dispatch(
          g3n_runtime, "g3n.residual-forward", "g3n.residual.forward",
          3u, d4_shape,
          (et_kernel_tensor_view_v1[2]){
            et_g3c4_view(scratch.et, sizeof(scratch.et),
                          "f32", 3u, d4_shape),
            et_g3c4_view(scratch.ep, sizeof(scratch.ep),
                          "f32", 3u, d4_shape)}, 2u,
          et_g3c4_view(scratch.x, sizeof(scratch.x),
                       "f32", 3u, d4_shape)) != 0)
    goto fail;
  inputs[0] = et_g3c4_view(
      scratch.x, sizeof(scratch.x), "f32", 3u, d4_shape);
  inputs[1] = context->pins.views[7];
  inputs[2] = context->pins.views[6];
  inputs[3] = et_g3c4_view(&epsilon, sizeof(epsilon), "f32", 0u, NULL);
  if (et_g3c4_token_dispatch(
          g3n_runtime, "g3n.layer-norm-forward",
          "g3n.layer-norm.forward", 3u, d4_shape, inputs, 4u,
          et_g3c4_view(scratch.n1, sizeof(scratch.n1),
                       "f32", 3u, d4_shape)) != 0)
    goto fail;

#define ET_G3C4_TOKEN_LINEAR(weight_index, source, target, row) do { \
  inputs[0] = et_g3c4_view( \
      (source), sizeof(source), "f32", 3u, \
      sizeof(source) == sizeof(scratch.fu) ? d8_shape : d4_shape); \
  inputs[1] = context->pins.views[(weight_index)]; \
  output = et_g3c4_view( \
      (target), sizeof(target), "f32", 3u, \
      sizeof(target) == sizeof(scratch.fu) ? d8_shape : \
      (sizeof(target) == sizeof(scratch.z) ? \
          (const uint64_t[3]){1u, 1u, 256u} : d4_shape)); \
  if (et_g3c4_token_dispatch( \
          g3n_runtime, "g3n.linear-forward", \
          "g3n.linear.forward-no-bias", 4u, (row), \
          inputs, 2u, output) != 0) \
    goto fail; \
} while (0)

  ET_G3C4_TOKEN_LINEAR(2u, scratch.n1, scratch.qt, linear_d4_d4);
  ET_G3C4_TOKEN_LINEAR(0u, scratch.n1, scratch.kt, linear_d4_d4);
  ET_G3C4_TOKEN_LINEAR(3u, scratch.n1, scratch.vt, linear_d4_d4);

#define ET_G3C4_TOKEN_LAYOUT(operation, source, source_rank, source_shape, \
                            target, target_rank, target_shape) do { \
  inputs[0] = et_g3c4_view( \
      (source), sizeof(source), "f32", (source_rank), (source_shape)); \
  output = et_g3c4_view( \
      (target), sizeof(target), "f32", (target_rank), (target_shape)); \
  if (et_g3c4_token_dispatch( \
          g3n_runtime, "g3n.head-layout-forward", (operation), \
          4u, head_layout_row, inputs, 1u, output) != 0) \
    goto fail; \
} while (0)

  ET_G3C4_TOKEN_LAYOUT(
      "g3n.heads.split.forward", scratch.qt, 3u, d4_shape,
      scratch.qh, 4u, heads_shape);
  ET_G3C4_TOKEN_LAYOUT(
      "g3n.heads.split.forward", scratch.kt, 3u, d4_shape,
      scratch.kh, 4u, heads_shape);
  ET_G3C4_TOKEN_LAYOUT(
      "g3n.heads.split.forward", scratch.vt, 3u, d4_shape,
      scratch.vh, 4u, heads_shape);

  if (et_g3c4_private_token_frame_stage_v1(
          context, speculative_token, scratch.kh, scratch.vh) != 0)
    goto fail;
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_view_begin_v1(
              context->token_frame_transaction, 0u,
              &transaction_view, &error), &error) != 0)
    goto fail;
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_view_tensors_v1(
              transaction_view, &cached_keys, &cached_values,
              &effective_lengths, &key_keep, &error), &error) != 0)
    goto fail;
  if (cached_keys == NULL || cached_values == NULL ||
      effective_lengths == NULL || key_keep == NULL ||
      effective_lengths->data == NULL || key_keep->data == NULL ||
      *(const int64_t *)effective_lengths->data < 1 ||
      *(const int64_t *)effective_lengths->data > 4 ||
      *(const int64_t *)effective_lengths->data != query_position + 1) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    goto fail;
  }
  for (key = 0u; key < 4u; key++)
    causal_keep[key] =
        ((const uint8_t *)key_keep->data)[key] != 0u &&
        key_positions[key] <= query_position ? 1u : 0u;
  inputs[0] = et_g3c4_view(
      scratch.qh, sizeof(scratch.qh), "f32", 4u, heads_shape);
  inputs[1] = *cached_keys;
  inputs[2] = *cached_values;
  inputs[3] = et_g3c4_view(
      &query_position, sizeof(query_position), "i64", 2u,
      attention_query_shape);
  inputs[4] = et_g3c4_view(
      key_positions, sizeof(key_positions), "i64", 2u,
      attention_key_shape);
  inputs[5] = et_g3c4_view(
      causal_keep, sizeof(causal_keep), "bool", 3u,
      attention_mask_shape);
  if (et_g3c4_token_dispatch(
          et_g3c4_forward_runtime, "g3c4.causal-attention",
          "g3c4.causal-attention.forward", 6u, attention_row,
          inputs, 6u,
          et_g3c4_view(scratch.ah, sizeof(scratch.ah),
                       "f32", 4u, heads_shape)) != 0)
    goto fail;
  if (et_a2_kv_cache_transaction_view_end_v1(
          &transaction_view, &error) != 0)
    abort();

  ET_G3C4_TOKEN_LAYOUT(
      "g3n.heads.merge.forward", scratch.ah, 4u, heads_shape,
      scratch.at, 3u, d4_shape);
  ET_G3C4_TOKEN_LINEAR(1u, scratch.at, scratch.ao, linear_d4_d4);
  inputs[0] = et_g3c4_view(
      scratch.x, sizeof(scratch.x), "f32", 3u, d4_shape);
  inputs[1] = et_g3c4_view(
      scratch.ao, sizeof(scratch.ao), "f32", 3u, d4_shape);
  if (et_g3c4_token_dispatch(
          g3n_runtime, "g3n.residual-forward", "g3n.residual.forward",
          3u, d4_shape, inputs, 2u,
          et_g3c4_view(scratch.r, sizeof(scratch.r),
                       "f32", 3u, d4_shape)) != 0)
    goto fail;
  inputs[0] = et_g3c4_view(
      scratch.r, sizeof(scratch.r), "f32", 3u, d4_shape);
  inputs[1] = context->pins.views[9];
  inputs[2] = context->pins.views[8];
  inputs[3] = et_g3c4_view(&epsilon, sizeof(epsilon), "f32", 0u, NULL);
  if (et_g3c4_token_dispatch(
          g3n_runtime, "g3n.layer-norm-forward",
          "g3n.layer-norm.forward", 3u, d4_shape, inputs, 4u,
          et_g3c4_view(scratch.n2, sizeof(scratch.n2),
                       "f32", 3u, d4_shape)) != 0)
    goto fail;
  ET_G3C4_TOKEN_LINEAR(5u, scratch.n2, scratch.fu, linear_d4_d8);
  inputs[0] = et_g3c4_view(
      scratch.fu, sizeof(scratch.fu), "f32", 3u, d8_shape);
  if (et_g3c4_token_dispatch(
          n2_runtime, "kernel.activation", "gelu.forward",
          3u, d8_shape, inputs, 1u,
          et_g3c4_view(scratch.fg, sizeof(scratch.fg),
                       "f32", 3u, d8_shape)) != 0)
    goto fail;
  ET_G3C4_TOKEN_LINEAR(4u, scratch.fg, scratch.fd, linear_d8_d4);
  inputs[0] = et_g3c4_view(
      scratch.r, sizeof(scratch.r), "f32", 3u, d4_shape);
  inputs[1] = et_g3c4_view(
      scratch.fd, sizeof(scratch.fd), "f32", 3u, d4_shape);
  if (et_g3c4_token_dispatch(
          g3n_runtime, "g3n.residual-forward", "g3n.residual.forward",
          3u, d4_shape, inputs, 2u,
          et_g3c4_view(scratch.y, sizeof(scratch.y),
                       "f32", 3u, d4_shape)) != 0)
    goto fail;
  inputs[0] = et_g3c4_view(
      scratch.y, sizeof(scratch.y), "f32", 3u, d4_shape);
  inputs[1] = context->pins.views[12];
  inputs[2] = context->pins.views[11];
  inputs[3] = et_g3c4_view(&epsilon, sizeof(epsilon), "f32", 0u, NULL);
  if (et_g3c4_token_dispatch(
          g3n_runtime, "g3n.layer-norm-forward",
          "g3n.layer-norm.forward", 3u, d4_shape, inputs, 4u,
          et_g3c4_view(scratch.nf, sizeof(scratch.nf),
                       "f32", 3u, d4_shape)) != 0)
    goto fail;
  ET_G3C4_TOKEN_LINEAR(10u, scratch.nf, scratch.z, linear_d4_v256);

#undef ET_G3C4_TOKEN_LAYOUT
#undef ET_G3C4_TOKEN_LINEAR
  et_kernel_runtime_destroy(n2_runtime);
  et_kernel_runtime_destroy(g3n_runtime);
  memcpy(logits_output, scratch.z, sizeof(scratch.z));
  return 0;

fail:
  first = et_g3c4_error_snapshot_internal();
  if (transaction_view != NULL &&
      et_a2_kv_cache_transaction_view_end_v1(
          &transaction_view, &error) != 0)
    abort();
  if (!et_g3c4_token_frame_idle(context))
    et_g3c4_token_frame_discard(context);
  if (n2_runtime != NULL) et_kernel_runtime_destroy(n2_runtime);
  if (g3n_runtime != NULL) et_kernel_runtime_destroy(g3n_runtime);
  et_g3c4_error_restore_internal(first);
  return et_g3c4_error_state.category;
}

#ifdef ET_G3C4_PREFILL3_PRIVATE
typedef struct et_g3c4_prefill3_scratch {
  float et[12], ep[12], x[12], n1[12];
  float qt[12], kt[12], vt[12];
  float qh[12], kh[12], vh[12], ah[12];
  float at[12], ao[12], r[12], n2[12];
  float fu[24], fg[24], fd[12], y[12], nf[12], z[768];
} et_g3c4_prefill3_scratch;

int64_t et_g3c4_private_prefill3_v1(
    void *candidate, const int64_t token_ids[3],
    float last_logits_output[256]) {
  static const uint64_t ids_shape[2] = {1u, 3u};
  static const uint64_t token_embedding_row[4] = {1u, 3u, 256u, 4u};
  static const uint64_t position_embedding_row[4] = {1u, 3u, 4u, 4u};
  static const uint64_t d4_shape[3] = {1u, 3u, 4u};
  static const uint64_t d8_shape[3] = {1u, 3u, 8u};
  static const uint64_t heads_shape[4] = {1u, 2u, 3u, 2u};
  static const uint64_t head_layout_row[4] = {1u, 3u, 2u, 2u};
  static const uint64_t linear_d4_d4[4] = {1u, 3u, 4u, 4u};
  static const uint64_t linear_d4_d8[4] = {1u, 3u, 4u, 8u};
  static const uint64_t linear_d8_d4[4] = {1u, 3u, 8u, 4u};
  static const uint64_t linear_d4_v256[4] = {1u, 3u, 4u, 256u};
  static const uint64_t attention_row[6] = {1u, 2u, 2u, 3u, 4u, 2u};
  static const uint64_t attention_query_shape[2] = {1u, 3u};
  static const uint64_t attention_key_shape[2] = {1u, 4u};
  static const uint64_t attention_mask_shape[3] = {1u, 3u, 4u};
  static const uint64_t append_shape[1] = {1u};
  et_g3c4_context_internal *context;
  et_g3c4_prefill3_scratch scratch;
  et_kernel_runtime *n2_runtime = NULL;
  et_a2_kv_cache *candidate_cache = NULL;
  et_a2_kv_cache *old_cache;
  et_a2_kv_cache_transaction *transaction = NULL;
  et_a2_kv_cache_transaction_view *transaction_view = NULL;
  const et_kernel_tensor_view_v1 *cached_keys = NULL;
  const et_kernel_tensor_view_v1 *cached_values = NULL;
  const et_kernel_tensor_view_v1 *effective_lengths = NULL;
  const et_kernel_tensor_view_v1 *key_keep = NULL;
  et_kernel_tensor_view_v1 inputs[6];
  et_kernel_tensor_view_v1 output;
  et_kernel_tensor_view_v1 append_counts;
  et_kernel_tensor_view_v1 staged_keys;
  et_kernel_tensor_view_v1 staged_values;
  const void *binding_identities[14];
  unsigned char binding_values[4768];
  et_kernel_error error;
  et_f32_tensor_error f32_error;
  et_g3c4_error_state_internal first;
  int64_t positions[3] = {0, 1, 2};
  int64_t key_positions[4] = {0, 1, 2, 3};
  int64_t append_count = 3;
  uint8_t causal_keep[12];
  uint32_t epsilon_bits = UINT32_C(0x3727c5ac);
  float epsilon;
  size_t index;
  size_t key;
  size_t binding_offset = 0u;

  et_g3c4_error_reset_internal();
  if (!et_g3c4_range_valid(token_ids, 3u * sizeof(token_ids[0])) ||
      !et_g3c4_range_valid(
          last_logits_output, 256u * sizeof(last_logits_output[0])) ||
      (uintptr_t)token_ids % _Alignof(int64_t) != 0u ||
      (uintptr_t)last_logits_output % _Alignof(float) != 0u ||
      et_g3c4_ranges_overlap(
          token_ids, 3u * sizeof(token_ids[0]),
          last_logits_output, 256u * sizeof(last_logits_output[0]))) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  for (index = 0u; index < 3u; index++)
    if (token_ids[index] < 0 || token_ids[index] > 255) {
      (void)et_g3c4_fail(
          ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
      return et_g3c4_error_state.category;
    }
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
  if (!((context->call_kind == 0 && context->budget == 0) ||
        (context->call_kind == 2 &&
         (context->budget == 0 || context->budget == 1))) ||
      !et_g3c4_token_frame_idle(context)) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return et_g3c4_error_state.category;
  }
  if (et_g3c4_ranges_overlap(
          token_ids, 3u * sizeof(token_ids[0]), context, sizeof(*context)) ||
      et_g3c4_ranges_overlap(
          last_logits_output, 256u * sizeof(last_logits_output[0]),
          context, sizeof(*context))) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  if (et_g3c4_prefill_reject_owned_aliases(
          context, token_ids, last_logits_output) != 0)
    return et_g3c4_error_state.category;
  memset(&scratch, 0, sizeof(scratch));
  memset(binding_identities, 0, sizeof(binding_identities));
  memset(binding_values, 0, sizeof(binding_values));
  memcpy(&epsilon, &epsilon_bits, sizeof(epsilon));
  if (et_g3c4_token_runtime_discover(
          et_n2_kernel_provider_v1(), &n2_runtime) != 0)
    goto fail;
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_create_v1(
              1u, 1u, 2u, 4u, 2u, &candidate_cache, &error),
          &error) != 0)
    goto fail;

  inputs[0] = et_g3c4_view(
      (void *)token_ids, 3u * sizeof(token_ids[0]), "i64", 2u, ids_shape);
  inputs[1] = context->pins.views[10];
  if (et_g3c4_token_dispatch(
          et_g3c4_forward_runtime, "g3c4.embedding-forward",
          "g3c4.embedding.forward", 4u, token_embedding_row, inputs, 2u,
          et_g3c4_view(scratch.et, sizeof(scratch.et),
                       "f32", 3u, d4_shape)) != 0)
    goto fail;
  inputs[0] = et_g3c4_view(
      positions, sizeof(positions), "i64", 2u, ids_shape);
  inputs[1] = context->pins.views[13];
  if (et_g3c4_token_dispatch(
          et_g3c4_forward_runtime, "g3c4.embedding-forward",
          "g3c4.embedding.forward", 4u, position_embedding_row,
          inputs, 2u,
          et_g3c4_view(scratch.ep, sizeof(scratch.ep),
                       "f32", 3u, d4_shape)) != 0)
    goto fail;

#define ET_G3C4_PREFILL3_RESIDUAL(source_a, source_b, target) do { \
  inputs[0] = et_g3c4_view( \
      (source_a), sizeof(source_a), "f32", 3u, d4_shape); \
  inputs[1] = et_g3c4_view( \
      (source_b), sizeof(source_b), "f32", 3u, d4_shape); \
  if (et_g3c4_token_dispatch( \
          n2_runtime, "kernel.residual", "residual.forward", \
          3u, d4_shape, inputs, 2u, \
          et_g3c4_view((target), sizeof(target), \
                       "f32", 3u, d4_shape)) != 0) \
    goto fail; \
} while (0)

#define ET_G3C4_PREFILL3_NORM(source, gamma_index, beta_index, target) do { \
  inputs[0] = et_g3c4_view( \
      (source), sizeof(source), "f32", 3u, d4_shape); \
  inputs[1] = context->pins.views[(gamma_index)]; \
  inputs[2] = context->pins.views[(beta_index)]; \
  inputs[3] = et_g3c4_view(&epsilon, sizeof(epsilon), "f32", 0u, NULL); \
  if (et_g3c4_token_dispatch( \
          et_g3c4_forward_runtime, "g3c4.layer-norm", \
          "g3c4.layer-norm.forward", 3u, d4_shape, inputs, 4u, \
          et_g3c4_view((target), sizeof(target), \
                       "f32", 3u, d4_shape)) != 0) \
    goto fail; \
} while (0)

#define ET_G3C4_PREFILL3_LINEAR(weight_index, source, target, row) do { \
  inputs[0] = et_g3c4_view( \
      (source), sizeof(source), "f32", 3u, \
      sizeof(source) == sizeof(scratch.fu) ? d8_shape : d4_shape); \
  inputs[1] = context->pins.views[(weight_index)]; \
  output = et_g3c4_view( \
      (target), sizeof(target), "f32", 3u, \
      sizeof(target) == sizeof(scratch.fu) ? d8_shape : \
      (sizeof(target) == sizeof(scratch.z) ? \
          (const uint64_t[3]){1u, 3u, 256u} : d4_shape)); \
  if (et_g3c4_token_dispatch( \
          et_g3c4_forward_runtime, "g3c4.linear", \
          "g3c4.linear.forward-no-bias", 4u, (row), \
          inputs, 2u, output) != 0) \
    goto fail; \
} while (0)

#define ET_G3C4_PREFILL3_LAYOUT(operation, source, source_rank, source_shape, \
                               target, target_rank, target_shape) do { \
  inputs[0] = et_g3c4_view( \
      (source), sizeof(source), "f32", (source_rank), (source_shape)); \
  if (et_g3c4_token_dispatch( \
          et_g3c4_forward_runtime, "g3c4.head-layout", (operation), \
          4u, head_layout_row, inputs, 1u, \
          et_g3c4_view((target), sizeof(target), \
                       "f32", (target_rank), (target_shape))) != 0) \
    goto fail; \
} while (0)

  ET_G3C4_PREFILL3_RESIDUAL(scratch.et, scratch.ep, scratch.x);
  ET_G3C4_PREFILL3_NORM(scratch.x, 7u, 6u, scratch.n1);
  ET_G3C4_PREFILL3_LINEAR(2u, scratch.n1, scratch.qt, linear_d4_d4);
  ET_G3C4_PREFILL3_LINEAR(0u, scratch.n1, scratch.kt, linear_d4_d4);
  ET_G3C4_PREFILL3_LINEAR(3u, scratch.n1, scratch.vt, linear_d4_d4);
  ET_G3C4_PREFILL3_LAYOUT(
      "g3c4.heads.split.forward", scratch.qt, 3u, d4_shape,
      scratch.qh, 4u, heads_shape);
  ET_G3C4_PREFILL3_LAYOUT(
      "g3c4.heads.split.forward", scratch.kt, 3u, d4_shape,
      scratch.kh, 4u, heads_shape);
  ET_G3C4_PREFILL3_LAYOUT(
      "g3c4.heads.split.forward", scratch.vt, 3u, d4_shape,
      scratch.vh, 4u, heads_shape);

  append_counts = et_g3c4_view(
      &append_count, sizeof(append_count), "i64", 1u, append_shape);
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_begin_v1(
              candidate_cache, 3u, &append_counts, &transaction, &error),
          &error) != 0)
    goto fail;
  staged_keys = et_g3c4_view(
      scratch.kh, sizeof(scratch.kh), "f32", 4u, heads_shape);
  staged_values = et_g3c4_view(
      scratch.vh, sizeof(scratch.vh), "f32", 4u, heads_shape);
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_stage_layer_v1(
              transaction, 0u, &staged_keys, &staged_values, &error),
          &error) != 0)
    goto fail;
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_view_begin_v1(
              transaction, 0u, &transaction_view, &error), &error) != 0)
    goto fail;
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_view_tensors_v1(
              transaction_view, &cached_keys, &cached_values,
              &effective_lengths, &key_keep, &error), &error) != 0)
    goto fail;
  if (cached_keys == NULL || cached_values == NULL ||
      effective_lengths == NULL || key_keep == NULL ||
      effective_lengths->data == NULL || key_keep->data == NULL ||
      *(const int64_t *)effective_lengths->data != 3) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    goto fail;
  }
  for (index = 0u; index < 3u; index++)
    for (key = 0u; key < 4u; key++)
      causal_keep[index * 4u + key] =
          ((const uint8_t *)key_keep->data)[key] != 0u &&
          key_positions[key] <= positions[index] ? 1u : 0u;
  inputs[0] = et_g3c4_view(
      scratch.qh, sizeof(scratch.qh), "f32", 4u, heads_shape);
  inputs[1] = *cached_keys;
  inputs[2] = *cached_values;
  inputs[3] = et_g3c4_view(
      positions, sizeof(positions), "i64", 2u, attention_query_shape);
  inputs[4] = et_g3c4_view(
      key_positions, sizeof(key_positions), "i64", 2u,
      attention_key_shape);
  inputs[5] = et_g3c4_view(
      causal_keep, sizeof(causal_keep), "bool", 3u,
      attention_mask_shape);
  if (et_g3c4_token_dispatch(
          et_g3c4_forward_runtime, "g3c4.causal-attention",
          "g3c4.causal-attention.forward", 6u, attention_row,
          inputs, 6u,
          et_g3c4_view(scratch.ah, sizeof(scratch.ah),
                       "f32", 4u, heads_shape)) != 0)
    goto fail;
  if (et_a2_kv_cache_transaction_view_end_v1(
          &transaction_view, &error) != 0)
    abort();

  ET_G3C4_PREFILL3_LAYOUT(
      "g3c4.heads.merge.forward", scratch.ah, 4u, heads_shape,
      scratch.at, 3u, d4_shape);
  ET_G3C4_PREFILL3_LINEAR(1u, scratch.at, scratch.ao, linear_d4_d4);
  ET_G3C4_PREFILL3_RESIDUAL(scratch.x, scratch.ao, scratch.r);
  ET_G3C4_PREFILL3_NORM(scratch.r, 9u, 8u, scratch.n2);
  ET_G3C4_PREFILL3_LINEAR(5u, scratch.n2, scratch.fu, linear_d4_d8);
  inputs[0] = et_g3c4_view(
      scratch.fu, sizeof(scratch.fu), "f32", 3u, d8_shape);
  if (et_g3c4_token_dispatch(
          et_g3c4_forward_runtime, "g3c4.gelu", "g3c4.gelu.forward",
          3u, d8_shape, inputs, 1u,
          et_g3c4_view(scratch.fg, sizeof(scratch.fg),
                       "f32", 3u, d8_shape)) != 0)
    goto fail;
  ET_G3C4_PREFILL3_LINEAR(4u, scratch.fg, scratch.fd, linear_d8_d4);
  ET_G3C4_PREFILL3_RESIDUAL(scratch.r, scratch.fd, scratch.y);
  ET_G3C4_PREFILL3_NORM(scratch.y, 12u, 11u, scratch.nf);
  ET_G3C4_PREFILL3_LINEAR(10u, scratch.nf, scratch.z, linear_d4_v256);

#undef ET_G3C4_PREFILL3_LAYOUT
#undef ET_G3C4_PREFILL3_LINEAR
#undef ET_G3C4_PREFILL3_NORM
#undef ET_G3C4_PREFILL3_RESIDUAL

  if (et_g3c4_capture_f32(
          et_g3c4_model_pins_check_internal(&context->pins, &f32_error),
          &f32_error) != 0)
    goto fail;
  for (index = 0u; index < 14u; index++) {
    const et_kernel_tensor_view_v1 *view = &context->pins.views[index];
    if (view->data == NULL ||
        view->byte_length > sizeof(binding_values) - binding_offset) {
      (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
      goto fail;
    }
    binding_identities[index] = context->pins.identities[index];
    memcpy(binding_values + binding_offset, view->data, view->byte_length);
    binding_offset += view->byte_length;
  }
  if (binding_offset != sizeof(binding_values)) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    goto fail;
  }
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_commit_v1(&transaction, &error),
          &error) != 0)
    goto fail;
  et_kernel_runtime_destroy(n2_runtime);
  n2_runtime = NULL;
  old_cache = context->cache;
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_destroy_v1(&old_cache, &error), &error) != 0)
    goto fail;

  context->cache = candidate_cache;
  candidate_cache = NULL;
  memcpy(context->prefill_binding_identities, binding_identities,
         sizeof(binding_identities));
  memcpy(context->prefill_binding_values, binding_values,
         sizeof(binding_values));
  memcpy(context->prefill_tokens, token_ids,
         sizeof(context->prefill_tokens));
  context->prefill_binding_ready = 1u;
  memcpy(last_logits_output, scratch.z + 2u * 256u,
         256u * sizeof(last_logits_output[0]));
  return 0;

fail:
  first = et_g3c4_error_snapshot_internal();
  if (transaction_view != NULL &&
      et_a2_kv_cache_transaction_view_end_v1(
          &transaction_view, &error) != 0)
    abort();
  if (transaction != NULL &&
      et_a2_kv_cache_transaction_abort_v1(&transaction, &error) != 0)
    abort();
  if (candidate_cache != NULL &&
      et_a2_kv_cache_destroy_v1(&candidate_cache, &error) != 0)
    abort();
  if (n2_runtime != NULL) et_kernel_runtime_destroy(n2_runtime);
  et_g3c4_error_restore_internal(first);
  return et_g3c4_error_state.category;
}
#endif

#ifdef ET_G3C4_LAST_LOGIT_FRAME_PRIVATE
int64_t et_g3c4_private_token_frame_begin_last_v1(
    void *candidate, const float last_logits[256],
    int64_t *speculative_token_output) {
  static const uint64_t logits_shape[2] = {1u, 256u};
  static const uint64_t rng_shape[1] = {4u};
  static const uint64_t token_shape[1] = {1u};
  static const uint64_t append_shape[1] = {1u};
  et_g3c4_context_internal *context;
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *borrowed_keys = NULL;
  const et_kernel_tensor_view_v1 *borrowed_values = NULL;
  const et_kernel_tensor_view_v1 *committed_lengths = NULL;
  const et_kernel_tensor_view_v1 *committed_keep = NULL;
  et_a2_kv_cache_transaction *transaction = NULL;
  et_kernel_tensor_view_v1 inputs[5];
  et_kernel_tensor_view_v1 outputs[2];
  et_kernel_tensor_view_v1 append_counts;
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
  et_kernel_error error;
  et_g3c4_error_state_internal first;
  float private_logits[256];
  float temperature;
  float top_p;
  int64_t top_k;
  int64_t numeric_rng[4];
  int64_t token_candidate = -1;
  int64_t successor_candidate[4] = {0, 0, 0, 0};
  int64_t token_position;
  int64_t append_count = 1;
  size_t input_count;
  size_t index;
  uint32_t temperature_bits;
  uint32_t top_p_bits;
  int categorical;
  int overlap = 0;

  et_g3c4_error_reset_internal();
  if (!et_g3c4_range_valid(last_logits, sizeof(private_logits)) ||
      !et_g3c4_range_valid(
          speculative_token_output, sizeof(*speculative_token_output)) ||
      (uintptr_t)last_logits % _Alignof(float) != 0u ||
      (uintptr_t)speculative_token_output % _Alignof(int64_t) != 0u ||
      et_g3c4_ranges_overlap(
          last_logits, sizeof(private_logits),
          speculative_token_output, sizeof(*speculative_token_output))) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
  if (context->call_kind != 2 || context->budget != 1 ||
      et_g3c4_sampler_runtime == NULL ||
      !et_g3c4_token_frame_idle(context)) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    return et_g3c4_error_state.category;
  }
  if (!et_g3c4_prefill_binding_matches_pins(context)) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_STALE_BINDING);
    return et_g3c4_error_state.category;
  }
  if (et_g3c4_ranges_overlap(
          last_logits, sizeof(private_logits), context, sizeof(*context)) ||
      et_g3c4_ranges_overlap(
          speculative_token_output, sizeof(*speculative_token_output),
          context, sizeof(*context)) ||
      et_g3c4_ranges_overlap(
          last_logits, sizeof(private_logits),
          context->owner, sizeof(*context->owner)) ||
      et_g3c4_ranges_overlap(
          speculative_token_output, sizeof(*speculative_token_output),
          context->owner, sizeof(*context->owner)))
    overlap = 1;
  for (index = 0u; index < 14u; index++) {
    const et_kernel_tensor_view_v1 *view = &context->pins.views[index];
    if (et_g3c4_ranges_overlap(
            last_logits, sizeof(private_logits),
            view->data, view->byte_length) ||
        et_g3c4_ranges_overlap(
            speculative_token_output, sizeof(*speculative_token_output),
            view->data, view->byte_length))
      overlap = 1;
  }
  if (et_a2_kv_cache_private_storage_overlap_v1(
          last_logits, sizeof(private_logits)) != 0 ||
      et_a2_kv_cache_private_storage_overlap_v1(
          speculative_token_output,
          sizeof(*speculative_token_output)) != 0)
    overlap = 1;
  if (overlap) {
    (void)et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    return et_g3c4_error_state.category;
  }

  memcpy(private_logits, last_logits, sizeof(private_logits));
  memcpy(numeric_rng, context->generator_rng_words, sizeof(numeric_rng));
  categorical = context->generator_policy[0] == 1;
  temperature_bits = (uint32_t)context->generator_policy[1];
  top_k = context->generator_policy[2];
  top_p_bits = (uint32_t)context->generator_policy[3];
  memcpy(&temperature, &temperature_bits, sizeof(temperature));
  memcpy(&top_p, &top_p_bits, sizeof(top_p));
  inputs[0] = et_g3c4_view(
      private_logits, sizeof(private_logits), "f32", 2u, logits_shape);
  if (categorical) {
    inputs[1] = et_g3c4_view(
        &temperature, sizeof(temperature), "f32", 0u, NULL);
    inputs[2] = et_g3c4_view(&top_k, sizeof(top_k), "i64", 0u, NULL);
    inputs[3] = et_g3c4_view(&top_p, sizeof(top_p), "f32", 0u, NULL);
    inputs[4] = et_g3c4_view(
        numeric_rng, sizeof(numeric_rng), "i64", 1u, rng_shape);
    input_count = 5u;
  } else {
    inputs[1] = et_g3c4_view(
        numeric_rng, sizeof(numeric_rng), "i64", 1u, rng_shape);
    input_count = 2u;
  }
  outputs[0] = et_g3c4_view(
      &token_candidate, sizeof(token_candidate), "i64", 1u, token_shape);
  outputs[1] = et_g3c4_view(
      successor_candidate, sizeof(successor_candidate),
      "i64", 1u, rng_shape);
  request = (et_kernel_request_v1){
    sizeof(request),
    categorical ? "g3s.categorical.forward" : "g3s.greedy.forward",
    "f32", "cpu", 2u, logits_shape, 1u, {0}};
  call = (et_kernel_call_v1){
    sizeof(call), categorical ? "g3s.categorical" : "g3s.greedy",
    &request, input_count, sizeof(inputs[0]), input_count * sizeof(inputs[0]),
    inputs, 2u, sizeof(outputs[0]), sizeof(outputs), outputs};
  if (et_g3c4_capture_kernel(
          et_kernel_runtime_dispatch(
              et_g3c4_sampler_runtime, &call, &error),
          &error) != 0)
    return et_g3c4_error_state.category;
  if (token_candidate < 0 || token_candidate > 255 ||
      successor_candidate[0] != 1 ||
      successor_candidate[1] != numeric_rng[1]) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    return et_g3c4_error_state.category;
  }

  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_read_borrow_begin_v1(
              context->cache, &borrow, &error), &error) != 0)
    return et_g3c4_error_state.category;
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_read_borrow_layer_v1(
              borrow, 0u, &borrowed_keys, &borrowed_values,
              &committed_lengths, &committed_keep, &error),
          &error) != 0) {
    first = et_g3c4_error_snapshot_internal();
    if (et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) != 0) abort();
    et_g3c4_error_restore_internal(first);
    return et_g3c4_error_state.category;
  }
  if (borrowed_keys == NULL || borrowed_values == NULL ||
      committed_lengths == NULL || committed_keep == NULL ||
      committed_lengths->data == NULL ||
      *(const int64_t *)committed_lengths->data < 0 ||
      *(const int64_t *)committed_lengths->data > 3) {
    (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
    first = et_g3c4_error_snapshot_internal();
    if (et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) != 0) abort();
    et_g3c4_error_restore_internal(first);
    return et_g3c4_error_state.category;
  }
  token_position = *(const int64_t *)committed_lengths->data;
  if (et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) != 0) abort();
  append_counts = et_g3c4_view(
      &append_count, sizeof(append_count), "i64", 1u, append_shape);
  if (et_g3c4_capture_kernel(
          et_a2_kv_cache_transaction_begin_v1(
              context->cache, 1u, &append_counts, &transaction, &error),
          &error) != 0)
    return et_g3c4_error_state.category;

  context->token_frame_transaction = transaction;
  context->token_frame_state = ET_G3C4_TOKEN_FRAME_SAMPLED;
  context->token_frame_candidate = token_candidate;
  memcpy(context->token_frame_successor, successor_candidate,
         sizeof(successor_candidate));
  context->token_frame_position = token_position;
  memcpy(speculative_token_output, &token_candidate,
         sizeof(token_candidate));
  return 0;
}
#endif
#endif
#endif

#ifdef ET_G3C4_ACTIVE_CALL_TESTING
static void et_g3c4_active_call_rollback(
    et_g3c4_context_internal *context,
    et_g3c4_error_state_internal first) {
  if ((context->acquired_mask & ET_G3C4_ACQUIRED_OWNER) != 0u) {
    if (context->owner->active != context) abort();
    context->owner->active = NULL;
    context->acquired_mask &= (uint32_t)~ET_G3C4_ACQUIRED_OWNER;
  }
  if ((context->acquired_mask & ET_G3C4_ACQUIRED_BUSY) != 0u) {
    ET_G3C4_CONTEXT_BUSY(context) = ET_G3C4_CALL_IDLE;
    context->acquired_mask &= (uint32_t)~ET_G3C4_ACQUIRED_BUSY;
  }
  if ((context->acquired_mask & ET_G3C4_ACQUIRED_METADATA) != 0u) {
    context->call_kind = 0;
    context->budget = 0;
    context->acquired_mask &= (uint32_t)~ET_G3C4_ACQUIRED_METADATA;
  }
  if ((context->acquired_mask & ET_G3C4_ACQUIRED_PINS) != 0u) {
    et_g3c4_model_pins_end_internal(&context->pins);
    context->acquired_mask &= (uint32_t)~ET_G3C4_ACQUIRED_PINS;
  }
  et_g3c4_error_restore_internal(first);
}

static int et_g3c4_active_call_inject(
    et_g3c4_context_internal *context, size_t completed) {
  et_g3c4_error_state_internal first;
  if (et_g3c4_active_call_fail_after != completed) return 0;
  (void)et_g3c4_fail(ET_G3C4_INTERNAL, ET_G3C4_CODE_INVARIANT);
  first = et_g3c4_error_snapshot_internal();
  et_g3c4_active_call_rollback(context, first);
  return 1;
}
#endif

int64_t et_g3c4_private_call_acquire_v1(
    void *candidate, int64_t call_kind, int64_t budget) {
  et_g3c4_context_internal *context;
  et_f32_tensor_error error;
  et_g3c4_error_reset_internal();
  context = et_g3c4_admit_idle_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
  if (!et_g3c4_valid_call_tuple(call_kind, budget))
    return et_g3c4_fail(
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_SELECTOR);
  if (et_g3c4_cache_idle_preflight(context) != 0)
    return et_g3c4_error_state.category;
  if (et_g3c4_capture_f32(
          et_g3c4_model_pins_begin_internal(
              context->owner->parameters, context->owner->handles,
              &context->pins, &error), &error) != 0)
    return et_g3c4_error_state.category;
  context->acquired_mask = ET_G3C4_ACQUIRED_PINS;
#ifdef ET_G3C4_ACTIVE_CALL_TESTING
  if (et_g3c4_active_call_inject(context, 1u))
    return et_g3c4_error_state.category;
#endif
  context->call_kind = call_kind;
  context->budget = budget;
  context->acquired_mask |= ET_G3C4_ACQUIRED_METADATA;
#ifdef ET_G3C4_ACTIVE_CALL_TESTING
  if (et_g3c4_active_call_inject(context, 2u))
    return et_g3c4_error_state.category;
#endif
  ET_G3C4_CONTEXT_BUSY(context) = ET_G3C4_CALL_ACTIVE;
  context->acquired_mask |= ET_G3C4_ACQUIRED_BUSY;
#ifdef ET_G3C4_ACTIVE_CALL_TESTING
  if (et_g3c4_active_call_inject(context, 3u))
    return et_g3c4_error_state.category;
#endif
  context->owner->active = context;
  context->acquired_mask |= ET_G3C4_ACQUIRED_OWNER;
#ifdef ET_G3C4_ACTIVE_CALL_TESTING
  if (et_g3c4_active_call_inject(context, 4u))
    return et_g3c4_error_state.category;
#endif
  return 0;
}

int64_t et_g3c4_private_call_prepare_end_v1(void *candidate) {
  et_g3c4_context_internal *context;
  et_g3c4_error_reset_internal();
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
  if (context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY)
    return 0;
  if (!et_g3c4_token_frame_idle(context))
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
#endif
  return et_g3c4_cache_idle_preflight(context);
}

static int64_t et_g3c4_active_call_drain(
    et_g3c4_context_internal *context) {
  et_g3c4_model_pins_end_internal(&context->pins);
  context->owner->active = NULL;
  context->call_kind = 0;
  context->budget = 0;
  ET_G3C4_CONTEXT_BUSY(context) = ET_G3C4_CALL_IDLE;
  context->acquired_mask = 0u;
  return 0;
}

int64_t et_g3c4_private_call_finish_v1(void *candidate) {
  et_g3c4_context_internal *context;
  et_g3c4_error_reset_internal();
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
  if (!et_g3c4_token_frame_idle(context))
    return et_g3c4_fail(
        ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
#endif
  return et_g3c4_active_call_drain(context);
}

int64_t et_g3c4_private_call_abort_v1(void *candidate) {
  et_g3c4_context_internal *context;
  et_g3c4_error_reset_internal();
  context = et_g3c4_admit_active_call(candidate);
  if (context == NULL) return et_g3c4_error_state.category;
#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
  if (!et_g3c4_token_frame_idle(context))
    et_g3c4_token_frame_discard(context);
#endif
  if (et_g3c4_cache_idle_preflight(context) != 0)
    return et_g3c4_error_state.category;
  return et_g3c4_active_call_drain(context);
}
#endif
#endif
