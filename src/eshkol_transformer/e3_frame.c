#include "e3_frame_internal.h"

#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/e3_evaluation_metrics_abi.h"
#include "eshkol_transformer/indexed_cross_entropy.h"
#include "eshkol_transformer/l3s_masked_objective_abi.h"
#include "eshkol_transformer/n2_primitives_abi.h"
#include "eshkol_transformer/n3k_primitives_abi.h"
#include "m3_call_pins.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* E3 is the native owner of the fixed M3 aggregate.  The package tuple compiles
 * m3_call_f32_integration.c separately so there is one I2 registry. */
#include "m3_model.c"

enum {
  E3_OK = 0,
  E3_INVALID_ARGUMENT = 1,
  E3_INVALID_STATE = 2,
  E3_SHAPE_MISMATCH = 3,
  E3_UNSUPPORTED = 4,
  E3_INTERNAL = 5
};
enum {
  E3_CODE_OK = 0,
  E3_CODE_IDENTITY = 1,
  E3_CODE_PHASE = 2,
  E3_CODE_SHAPE = 3,
  E3_CODE_PROFILE = 4,
  E3_CODE_ALLOCATION = 5,
  E3_CODE_CAPACITY = 6,
  E3_CODE_REENTRANCY = 7,
  E3_CODE_DESTINATION = 8,
  E3_CODE_INVARIANT = 9
};
enum {
  E3_IDLE = 0,
  E3_ACQUIRED = 1,
  E3_STAGING = 2,
  E3_FORWARD = 3,
  E3_CE_READY = 4,
  E3_REDUCED = 5,
  E3_CORRECTED = 6,
  E3_FINALIZED = 7,
  E3_DRAINED = 8
};
enum {
  E3_RUNTIME_N3K = 0,
  E3_RUNTIME_N2 = 1,
  E3_RUNTIME_A2 = 2,
  E3_RUNTIME_L2 = 3,
  E3_RUNTIME_L3S = 4,
  E3_RUNTIME_METRICS = 5
};

struct et_e3_frame_internal {
  struct et_e3_frame_internal *next;
  uint32_t lifecycle, phase, next_role, stage_plane;
  owner *model;
  et_g3t_model_pins_internal pins;
  et_f32_tensor *forward[21], *ce, *epsilon;
  et_f32_tensor *batch[4], *sum[2][3], *result[4], *destination[4];
  int64_t ids[2], targets[2], positions[2];
  unsigned char keep[4], mask[2];
  int64_t counts[2][2], active, staged_counts[2], published_counts[2];
  uint32_t sum_bank, committed, cleanup_ready, published_valid;
  et_kernel_runtime *runtime[6];
};

static et_e3_frame_internal *e3_registry;
static int64_t e3_last_domain, e3_last_category, e3_last_code;

#ifdef ET_E3_TESTING
static size_t e3_test_frame_alloc_remaining = SIZE_MAX;
void et_e3_test_frame_fail_alloc_after_v1(size_t count) {
  e3_test_frame_alloc_remaining = count;
}
#endif

static const uint64_t e3_shape_token[] = {1, 2, 4};
static const uint64_t e3_shape_head[] = {1, 2, 2, 2};
static const uint64_t e3_shape_wide[] = {1, 2, 8};
static const uint64_t e3_shape_logits[] = {1, 2, 256};
static const uint64_t e3_shape_pair[] = {1, 2};
static const uint64_t e3_shape_attention_mask[] = {1, 2, 2};
static const uint64_t e3_shape_embed_token[] = {1, 2, 256, 4};
static const uint64_t e3_shape_embed_position[] = {1, 2, 2, 4};
static const uint64_t e3_shape_linear_4[] = {1, 2, 4, 4};
static const uint64_t e3_shape_linear_up[] = {1, 2, 4, 8};
static const uint64_t e3_shape_linear_down[] = {1, 2, 8, 4};
static const uint64_t e3_shape_linear_vocab[] = {1, 2, 4, 256};
static const uint64_t e3_shape_attention[] = {1, 2, 2, 2, 2, 2};

static void e3_clear(void) {
  e3_last_domain = 0;
  e3_last_category = 0;
  e3_last_code = 0;
}

static int64_t e3_fail(int64_t status, int64_t code) {
  e3_last_domain = 0;
  e3_last_category = status;
  e3_last_code = code;
  return status;
}

static int64_t e3_dependency_status(int64_t domain, int64_t category) {
  if (domain == 1) {
    if (category == ET_KERNEL_ERROR_INVALID_ARGUMENT ||
        category == ET_KERNEL_ERROR_DTYPE_MISMATCH ||
        category == ET_KERNEL_ERROR_DEVICE_MISMATCH ||
        category == ET_KERNEL_ERROR_NONCONTIGUOUS)
      return E3_INVALID_ARGUMENT;
    if (category == ET_KERNEL_ERROR_SHAPE_MISMATCH) return E3_SHAPE_MISMATCH;
    if (category == ET_KERNEL_ERROR_UNSUPPORTED) return E3_UNSUPPORTED;
    return E3_INTERNAL;
  }
  if (domain == 2) {
    if (category == ET_I64_TENSOR_ERROR_INVALID_ARGUMENT ||
        category == ET_I64_TENSOR_ERROR_NONCONTIGUOUS)
      return E3_INVALID_ARGUMENT;
    if (category == ET_I64_TENSOR_ERROR_INVALID_STATE) return E3_INVALID_STATE;
    if (category == ET_I64_TENSOR_ERROR_SHAPE_MISMATCH)
      return E3_SHAPE_MISMATCH;
    return E3_INTERNAL;
  }
  if (domain == 3) {
    if (category == ET_F32_TENSOR_ERROR_INVALID_ARGUMENT ||
        category == ET_F32_TENSOR_ERROR_NONCONTIGUOUS)
      return E3_INVALID_ARGUMENT;
    if (category == ET_F32_TENSOR_ERROR_INVALID_STATE) return E3_INVALID_STATE;
    if (category == ET_F32_TENSOR_ERROR_SHAPE_MISMATCH)
      return E3_SHAPE_MISMATCH;
    return E3_INTERNAL;
  }
  return E3_INTERNAL;
}

static int e3_dependency_valid(int64_t domain, int64_t category,
                               int64_t code) {
  if (domain == 1)
    return category >= ET_KERNEL_ERROR_INVALID_ARGUMENT &&
        category <= ET_KERNEL_ERROR_INTERNAL &&
        code >= ET_KERNEL_CODE_NULL_ARGUMENT &&
        code <= ET_KERNEL_CODE_PROVIDER_REJECTED;
  if (domain == 2)
    return category >= ET_I64_TENSOR_ERROR_INVALID_ARGUMENT &&
        category <= ET_I64_TENSOR_ERROR_INTERNAL &&
        code >= ET_I64_TENSOR_CODE_NULL_ARGUMENT &&
        code <= ET_I64_TENSOR_CODE_PROVIDER_REJECTED;
  if (domain == 3)
    return category >= ET_F32_TENSOR_ERROR_INVALID_ARGUMENT &&
        category <= ET_F32_TENSOR_ERROR_INTERNAL &&
        code >= ET_F32_TENSOR_CODE_NULL_ARGUMENT &&
        code <= ET_F32_TENSOR_CODE_FLOAT_ENVIRONMENT;
  return 0;
}

static int64_t e3_dependency(int64_t domain, int64_t category, int64_t code) {
  if (!e3_dependency_valid(domain, category, code))
    return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  e3_last_domain = domain;
  e3_last_category = category;
  e3_last_code = code;
  return e3_dependency_status(domain, category);
}

static int64_t e3_f32_error(const et_f32_tensor_error *error) {
  return e3_dependency(3, error->category, error->code);
}

static int64_t e3_kernel_error(const et_kernel_error *error) {
  return e3_dependency(1, error->category, error->code);
}

static int64_t e3_m3_error(void) {
  if (error_domain == 0) {
    if (error_category == ET_M3T_INVALID_ARGUMENT &&
        error_code == ET_M3T_CODE_IDENTITY)
      return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_IDENTITY);
    if (error_category == ET_M3T_INVALID_STATE &&
        error_code == ET_M3T_CODE_LIFECYCLE)
      return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
    if (error_category == ET_M3T_INTERNAL &&
        error_code == ET_M3T_CODE_ALLOCATION)
      return e3_fail(E3_INTERNAL, E3_CODE_ALLOCATION);
    if (error_category == ET_M3T_SHAPE_MISMATCH &&
        error_code == ET_M3T_CODE_SHAPE)
      return e3_fail(E3_SHAPE_MISMATCH, E3_CODE_SHAPE);
    if (error_category == ET_M3T_INVALID_STATE &&
        error_code == ET_M3T_CODE_REENTRANCY)
      return e3_fail(E3_INVALID_STATE, E3_CODE_REENTRANCY);
    if (error_category == ET_M3T_INTERNAL &&
        error_code == ET_M3T_CODE_INTERNAL)
      return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
    return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  }
  if (error_domain >= 1 && error_domain <= 3)
    return e3_dependency(error_domain, error_category, error_code);
  return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
}

int64_t et_e3_private_last_error_domain_v1(void) { return e3_last_domain; }
int64_t et_e3_private_last_error_category_v1(void) { return e3_last_category; }
int64_t et_e3_private_last_error_code_v1(void) { return e3_last_code; }

static int e3_span(const void *pointer, size_t bytes) {
  return bytes == 0 ||
      (pointer != NULL && (uintptr_t)pointer <= UINTPTR_MAX - bytes);
}

static int e3_overlap(const void *left, size_t left_bytes,
                      const void *right, size_t right_bytes) {
  if (!left_bytes || !right_bytes) return 0;
  if (!e3_span(left, left_bytes) || !e3_span(right, right_bytes)) return 1;
  return (uintptr_t)left < (uintptr_t)right + right_bytes &&
         (uintptr_t)right < (uintptr_t)left + left_bytes;
}

static int e3_pins_idle(const et_g3t_model_pins_internal *pins) {
  if (pins->self || pins->held_mask) return 0;
  for (size_t i = 0; i < 14; ++i) {
    const et_kernel_tensor_view_v1 *view = &pins->views[i];
    if (pins->parameters[i] || pins->identities[i] || pins->values[i] ||
        view->struct_size || view->data || view->byte_length || view->dtype ||
        view->device || view->layout || view->offset_bytes || view->rank ||
        view->shape)
      return 0;
  }
  return 1;
}

static et_e3_frame_internal *e3_find(const void *pointer) {
  et_e3_frame_internal *frame = e3_registry;
  while (frame && frame != pointer) frame = frame->next;
  return frame;
}

static et_e3_frame_internal *e3_admit(void *pointer) {
  et_e3_frame_internal *frame = e3_find(pointer);
  if (!frame) {
    e3_fail(E3_INVALID_ARGUMENT, E3_CODE_IDENTITY);
    return NULL;
  }
  if (frame->lifecycle != 1) {
    e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
    return NULL;
  }
  clear();
  owner *model = admit(frame->model, OWNER, 0);
  if (!model || model != frame->model) {
    e3_m3_error();
    return NULL;
  }
  if (frame->phase == E3_IDLE) {
    if (model->active != NULL) {
      e3_fail(E3_INVALID_STATE, E3_CODE_REENTRANCY);
      return NULL;
    }
  } else if (frame->phase <= E3_DRAINED) {
    if (model->active != frame) {
      e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
      return NULL;
    }
  } else {
    e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
    return NULL;
  }
  return frame;
}

static int64_t e3_scope_begin(et_f32_tensor *tensor,
                              et_f32_scoped_guard_internal *guard,
                              int destination) {
  et_f32_tensor_error error;
  int32_t rc = et_f32_tensor_scoped_begin_internal(tensor, guard, &error);
  if (!rc) return 0;
  if (destination &&
      error.category == ET_F32_TENSOR_ERROR_INVALID_STATE &&
      error.code == ET_F32_TENSOR_CODE_ACTIVE_BORROW)
    return e3_fail(E3_INVALID_STATE, E3_CODE_DESTINATION);
  return e3_f32_error(&error);
}

static void e3_scope_end(et_f32_scoped_guard_internal *guard) {
  if (et_f32_tensor_scoped_end_internal(guard) != 0) abort();
}

static int64_t e3_tensor_shape(et_f32_tensor *tensor, size_t rank,
                               const uint64_t *shape, int destination) {
  et_f32_scoped_guard_internal guard = {0};
  int64_t rc = e3_scope_begin(tensor, &guard, destination);
  if (rc) return rc;
  size_t elements = 1;
  for (size_t i = 0; i < rank; ++i) elements *= (size_t)shape[i];
  if (guard.view.rank != rank || guard.view.byte_length != elements * sizeof(float) ||
      (rank == 0 ? guard.view.shape != NULL :
       guard.view.shape == NULL ||
       memcmp(guard.view.shape, shape, rank * sizeof(*shape)) != 0))
    rc = destination ? e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION) :
                       e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  e3_scope_end(&guard);
  return rc;
}

static size_t e3_all_tensors(et_e3_frame_internal *frame,
                             et_f32_tensor *tensors[37]) {
  size_t count = 0;
  for (size_t i = 0; i < 21; ++i) tensors[count++] = frame->forward[i];
  tensors[count++] = frame->ce;
  tensors[count++] = frame->epsilon;
  for (size_t i = 0; i < 4; ++i) tensors[count++] = frame->batch[i];
  for (size_t bank = 0; bank < 2; ++bank)
    for (size_t i = 0; i < 3; ++i) tensors[count++] = frame->sum[bank][i];
  for (size_t i = 0; i < 4; ++i) tensors[count++] = frame->result[i];
  return count;
}

static int64_t e3_storage_check(et_e3_frame_internal *frame, int require_runtime) {
  et_f32_tensor *owned[37];
  if (e3_all_tensors(frame, owned) != 37)
    return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  for (size_t i = 0; i < 37; ++i) {
    if (!owned[i] || et_f32_tensor_canonical_owner_v1(owned[i]) != owned[i])
      return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
    for (size_t j = 0; j < i; ++j)
      if (owned[i] == owned[j])
        return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
    for (size_t j = 0; j < 4; ++j)
      if (owned[i] == frame->destination[j])
        return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  }
  for (size_t i = 0; i < 4; ++i) {
    if (!frame->destination[i] ||
        et_f32_tensor_canonical_owner_v1(frame->destination[i]) !=
            frame->destination[i])
      return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_IDENTITY);
    for (size_t j = 0; j < i; ++j)
      if (frame->destination[i] == frame->destination[j])
        return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
    if (e3_tensor_shape(frame->destination[i], 0, NULL, 1))
      return e3_last_category == E3_INVALID_STATE ? E3_INVALID_STATE :
             e3_last_domain ? e3_dependency_status(e3_last_domain,
                                                     e3_last_category) :
             e3_last_category;
  }
  for (size_t i = 0; i < 21; ++i) {
    const uint64_t *shape = e3_shape_token;
    size_t rank = 3;
    if (i >= 7 && i <= 10) { shape = e3_shape_head; rank = 4; }
    if (i == 15 || i == 16) shape = e3_shape_wide;
    if (i == 20) shape = e3_shape_logits;
    if (e3_tensor_shape(frame->forward[i], rank, shape, 0))
      return e3_last_domain ? e3_dependency_status(e3_last_domain,
                                                    e3_last_category) :
             e3_last_category;
  }
  if (e3_tensor_shape(frame->ce, 2, e3_shape_pair, 0) ||
      e3_tensor_shape(frame->epsilon, 0, NULL, 0))
    return e3_last_domain ? e3_dependency_status(e3_last_domain,
                                                  e3_last_category) :
           e3_last_category;
  for (size_t i = 0; i < 4; ++i)
    if (e3_tensor_shape(frame->batch[i], 0, NULL, 0) ||
        e3_tensor_shape(frame->result[i], 0, NULL, 0))
      return e3_last_domain ? e3_dependency_status(e3_last_domain,
                                                    e3_last_category) :
             e3_last_category;
  for (size_t bank = 0; bank < 2; ++bank)
    for (size_t i = 0; i < 3; ++i)
      if (e3_tensor_shape(frame->sum[bank][i], 0, NULL, 0))
        return e3_last_domain ? e3_dependency_status(e3_last_domain,
                                                      e3_last_category) :
               e3_last_category;
  if (frame->positions[0] != 0 || frame->positions[1] != 1 ||
      memcmp(frame->keep, "\1\1\1\1", sizeof(frame->keep)) != 0)
    return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  {
    et_f32_scoped_guard_internal guard = {0};
    int64_t rc = e3_scope_begin(frame->epsilon, &guard, 0);
    uint32_t bits = 0;
    if (!rc) memcpy(&bits, guard.view.data, sizeof(bits));
    if (guard.owner) e3_scope_end(&guard);
    if (rc) return rc;
    if (bits != UINT32_C(0x3727c5ac))
      return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  }
  if (frame->phase == E3_IDLE) {
    if (!e3_pins_idle(&frame->pins))
      return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  } else if (frame->phase == E3_DRAINED) {
    if (!e3_pins_idle(&frame->pins))
      return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  } else {
    et_f32_tensor_error error;
    if (et_g3t_model_pins_check_internal(&frame->pins, &error) != 0)
      return e3_f32_error(&error);
  }
  if (require_runtime)
    for (size_t i = 0; i < 6; ++i)
      if (!frame->runtime[i]) return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  return 0;
}

static const et_kernel_provider_v1 *e3_resolver(void *context,
                                                 const char *symbol) {
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0 ? context : NULL;
}

static int64_t e3_discover(et_kernel_runtime **runtime,
                           const et_kernel_provider_v1 *provider) {
  et_kernel_error error;
  int32_t rc = et_kernel_runtime_discover(e3_resolver, (void *)provider,
                                          runtime, &error);
  return rc ? e3_kernel_error(&error) : 0;
}

static et_f32_tensor *e3_tensor_create(size_t rank, const uint64_t *shape) {
  et_f32_tensor *tensor = NULL;
  et_f32_tensor_error error;
  int32_t rc = et_f32_tensor_create_v1(rank, shape, &tensor, &error);
  if (rc) e3_f32_error(&error);
  return tensor;
}

static void e3_tensor_destroy(et_f32_tensor **tensor) {
  if (!*tensor) return;
  et_f32_tensor_error error;
  if (et_f32_tensor_destroy_v1(tensor, &error) != 0) abort();
}

static void e3_destroy_owned(et_e3_frame_internal *frame) {
  for (size_t i = 4; i > 0; --i) e3_tensor_destroy(&frame->result[i - 1]);
  for (size_t bank = 2; bank > 0; --bank)
    for (size_t i = 3; i > 0; --i)
      e3_tensor_destroy(&frame->sum[bank - 1][i - 1]);
  for (size_t i = 4; i > 0; --i) e3_tensor_destroy(&frame->batch[i - 1]);
  e3_tensor_destroy(&frame->epsilon);
  e3_tensor_destroy(&frame->ce);
  for (size_t i = 21; i > 0; --i) e3_tensor_destroy(&frame->forward[i - 1]);
}

static int64_t e3_create_owned(et_e3_frame_internal *frame) {
  for (size_t i = 0; i < 21; ++i) {
    const uint64_t *shape = e3_shape_token;
    size_t rank = 3;
    if (i >= 7 && i <= 10) { shape = e3_shape_head; rank = 4; }
    if (i == 15 || i == 16) shape = e3_shape_wide;
    if (i == 20) shape = e3_shape_logits;
    frame->forward[i] = e3_tensor_create(rank, shape);
    if (!frame->forward[i]) return e3_dependency_status(e3_last_domain,
                                                         e3_last_category);
  }
  frame->ce = e3_tensor_create(2, e3_shape_pair);
  frame->epsilon = e3_tensor_create(0, NULL);
  if (!frame->ce || !frame->epsilon)
    return e3_dependency_status(e3_last_domain, e3_last_category);
  for (size_t i = 0; i < 4; ++i) {
    frame->batch[i] = e3_tensor_create(0, NULL);
    if (!frame->batch[i]) return e3_dependency_status(e3_last_domain,
                                                       e3_last_category);
  }
  for (size_t bank = 0; bank < 2; ++bank)
    for (size_t i = 0; i < 3; ++i) {
      frame->sum[bank][i] = e3_tensor_create(0, NULL);
      if (!frame->sum[bank][i])
        return e3_dependency_status(e3_last_domain, e3_last_category);
    }
  for (size_t i = 0; i < 4; ++i) {
    frame->result[i] = e3_tensor_create(0, NULL);
    if (!frame->result[i]) return e3_dependency_status(e3_last_domain,
                                                        e3_last_category);
  }
  return 0;
}

static int64_t e3_destination_admit(owner *model,
                                    et_f32_tensor *destination[4]) {
  et_f32_scoped_guard_internal guards[4] = {{0}};
  size_t opened = 0;
  for (size_t i = 0; i < 4; ++i) {
    if (!destination[i] ||
        et_f32_tensor_canonical_owner_v1(destination[i]) != destination[i]) {
      e3_fail(E3_INVALID_ARGUMENT, E3_CODE_IDENTITY);
      goto failed;
    }
    for (size_t j = 0; j < i; ++j)
      if (destination[i] == destination[j]) {
        e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
        goto failed;
      }
    for (size_t j = 0; j < 14; ++j)
      if (et_f32_parameter_canonical_owner_v1(model->p[j]) == destination[i]) {
        e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
        goto failed;
      }
    if (e3_scope_begin(destination[i], &guards[opened], 1)) goto failed;
    ++opened;
    if (guards[i].view.rank != 0 || guards[i].view.shape != NULL ||
        guards[i].view.byte_length != sizeof(float)) {
      e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
      goto failed;
    }
    for (size_t j = 0; j < i; ++j)
      if (e3_overlap(guards[i].view.data, guards[i].view.byte_length,
                     guards[j].view.data, guards[j].view.byte_length)) {
        e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
        goto failed;
      }
  }
  while (opened) e3_scope_end(&guards[--opened]);
  return 0;
failed:
  while (opened) e3_scope_end(&guards[--opened]);
  return e3_last_domain ? e3_dependency_status(e3_last_domain,
                                                e3_last_category) :
         e3_last_category;
}

void *et_e3_private_frame_create_v1(
    void *model_pointer, et_f32_tensor *loss, et_f32_tensor *weight,
    et_f32_tensor *perplexity, et_f32_tensor *accuracy) {
  e3_clear();
  clear();
  owner *model = admit(model_pointer, OWNER, 0);
  if (!model) { e3_m3_error(); return NULL; }
  if (model->r.state != 1 || model->initialized != 14) {
    e3_fail(E3_UNSUPPORTED, E3_CODE_PROFILE);
    return NULL;
  }
  if (model->active) {
    e3_fail(E3_INVALID_STATE, E3_CODE_REENTRANCY);
    return NULL;
  }
  for (size_t i = 0; i < 14; ++i) {
    if (!model->handles[i]) {
      e3_fail(E3_UNSUPPORTED, E3_CODE_PROFILE);
      return NULL;
    }
    et_f32_tensor_error error;
    if (et_f32_parameter_validate_identity_v1(model->p[i], model->handles[i],
                                               &error) != 0) {
      e3_f32_error(&error);
      return NULL;
    }
  }
  et_f32_tensor *destinations[4] = {loss, weight, perplexity, accuracy};
  if (e3_destination_admit(model, destinations)) return NULL;
#ifdef ET_E3_TESTING
  if (e3_test_frame_alloc_remaining == 0) {
    e3_fail(E3_INTERNAL, E3_CODE_ALLOCATION);
    return NULL;
  }
  if (e3_test_frame_alloc_remaining != SIZE_MAX)
    --e3_test_frame_alloc_remaining;
#endif
  et_e3_frame_internal *frame = calloc(1, sizeof(*frame));
  if (!frame) {
    e3_fail(E3_INTERNAL, E3_CODE_ALLOCATION);
    return NULL;
  }
  frame->model = model;
  memcpy(frame->destination, destinations, sizeof(destinations));
  int64_t rc = e3_create_owned(frame);
  if (!rc) {
    const et_kernel_provider_v1 *providers[6] = {
      et_n3k_kernel_provider_v1(), et_n2_kernel_provider_v1(),
      et_a2_kernel_provider_v1(), et_l2_indexed_cross_entropy_provider_v1(),
      et_l3s_kernel_provider_v1(), et_e3_metrics_kernel_provider_v1()
    };
    for (size_t i = 0; i < 6 && !rc; ++i)
      rc = e3_discover(&frame->runtime[i], providers[i]);
  }
  if (!rc) {
    const uint32_t epsilon = UINT32_C(0x3727c5ac);
    et_f32_tensor_error error;
    if (et_f32_tensor_copy_bits_from_v1(frame->epsilon, &epsilon, 1,
                                        &error) != 0)
      rc = e3_f32_error(&error);
  }
  if (!rc) {
    frame->positions[0] = 0;
    frame->positions[1] = 1;
    memset(frame->keep, 1, sizeof(frame->keep));
    rc = e3_storage_check(frame, 1);
  }
  if (rc) {
    const int64_t saved[3] = {e3_last_domain, e3_last_category, e3_last_code};
    for (size_t i = 6; i > 0; --i) {
      et_kernel_runtime_destroy(frame->runtime[i - 1]);
      frame->runtime[i - 1] = NULL;
    }
    e3_destroy_owned(frame);
    free(frame);
    e3_last_domain = saved[0];
    e3_last_category = saved[1];
    e3_last_code = saved[2];
    return NULL;
  }
  frame->lifecycle = 1;
  frame->phase = E3_IDLE;
  frame->next = e3_registry;
  e3_registry = frame;
  e3_clear();
  return frame;
}

int64_t et_e3_private_frame_check_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  return e3_storage_check(frame, 1);
}

static int64_t e3_zero_tensor(et_f32_tensor *tensor) {
  et_f32_scoped_guard_internal guard = {0};
  int64_t rc = e3_scope_begin(tensor, &guard, 0);
  if (!rc) memset(guard.view.data, 0, guard.view.byte_length);
  if (guard.owner) e3_scope_end(&guard);
  return rc;
}

int64_t et_e3_private_acquire_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase != E3_IDLE || frame->cleanup_ready)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  int64_t rc = e3_storage_check(frame, 1);
  if (rc) return rc;
  for (size_t bank = 0; bank < 2; ++bank)
    for (size_t i = 0; i < 3; ++i)
      if ((rc = e3_zero_tensor(frame->sum[bank][i])) != 0) return rc;
  memset(frame->counts, 0, sizeof(frame->counts));
  frame->active = 0;
  memset(frame->staged_counts, 0, sizeof(frame->staged_counts));
  frame->sum_bank = 0;
  frame->committed = 0;
  frame->next_role = 0;
  frame->stage_plane = 0;
  et_f32_tensor_error error;
  if (et_g3t_model_pins_begin_internal(frame->model->p,
                                       (const void *const *)frame->model->handles,
                                       &frame->pins, &error) != 0)
    return e3_f32_error(&error);
  frame->model->active = frame;
  frame->phase = E3_ACQUIRED;
  return 0;
}

static et_kernel_tensor_view_v1 e3_native_view(
    void *data, size_t bytes, const char *dtype, size_t rank,
    const uint64_t *shape) {
  return (et_kernel_tensor_view_v1){
    sizeof(et_kernel_tensor_view_v1), data, bytes, dtype, "cpu",
    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, rank, shape
  };
}

static int64_t e3_stage_span_alias(et_e3_frame_internal *frame,
                                   const void *pointer, size_t bytes) {
  if (!e3_span(pointer, bytes) ||
      e3_overlap(pointer, bytes, frame, sizeof(*frame)))
    return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
  for (size_t i = 0; i < 14; ++i)
    if (e3_overlap(pointer, bytes, frame->pins.views[i].data,
                   frame->pins.views[i].byte_length) ||
        e3_overlap(pointer, bytes, frame->pins.views[i].shape,
                   frame->pins.views[i].rank *
                       sizeof(*frame->pins.views[i].shape)))
      return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
  et_f32_tensor *owned[37];
  (void)e3_all_tensors(frame, owned);
  for (size_t i = 0; i < 41; ++i) {
    et_f32_tensor *tensor = i < 37 ? owned[i] : frame->destination[i - 37];
    et_f32_scoped_guard_internal guard = {0};
    int64_t rc = e3_scope_begin(tensor, &guard, i >= 37);
    if (rc) return rc;
    int overlap = e3_overlap(pointer, bytes, tensor, 1) ||
        e3_overlap(pointer, bytes, guard.view.data, guard.view.byte_length) ||
        e3_overlap(pointer, bytes, guard.view.shape,
                   guard.view.rank * sizeof(*guard.view.shape));
    e3_scope_end(&guard);
    if (overlap) return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
  }
  return 0;
}

static int64_t e3_stage_alias_check(et_e3_frame_internal *frame,
                                    const void *prefix, size_t prefix_bytes,
                                    const void *shape, size_t shape_bytes,
                                    const void *data, size_t data_bytes) {
  const void *source[3] = {prefix, shape, data};
  const size_t source_bytes[3] = {prefix_bytes, shape_bytes, data_bytes};
  for (size_t i = 0; i < 3; ++i) {
    int64_t rc = e3_stage_span_alias(frame, source[i], source_bytes[i]);
    if (rc) return rc;
    for (size_t j = 0; j < i; ++j)
      if (e3_overlap(source[i], source_bytes[i], source[j], source_bytes[j]))
        return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
  }
  return 0;
}

static int64_t e3_stage_view(et_e3_frame_internal *frame,
                             const et_kernel_tensor_view_v1 *view,
                             const char *dtype, size_t bytes,
                             size_t alignment) {
  if (!view || !e3_span(view, sizeof(*view)))
    return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
  int64_t rc = e3_stage_span_alias(frame, view, sizeof(*view));
  if (rc) return rc;
  const et_kernel_tensor_view_v1 copy = *view;
  if (copy.struct_size != sizeof(copy) || !copy.data ||
      !e3_span(copy.data, copy.byte_length))
    return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
  if (!copy.dtype || !copy.device || strcmp(copy.dtype, dtype) != 0 ||
      strcmp(copy.device, "cpu") != 0 ||
      copy.layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR || copy.offset_bytes != 0)
    return e3_fail(E3_UNSUPPORTED, E3_CODE_PROFILE);
  if (copy.rank != 2 || !copy.shape || copy.byte_length != bytes ||
      !e3_span(copy.shape, 2 * sizeof(*copy.shape)))
    return e3_fail(E3_SHAPE_MISMATCH, E3_CODE_SHAPE);
  rc = e3_stage_alias_check(frame, view, sizeof(*view), copy.shape,
                            2 * sizeof(*copy.shape), copy.data,
                            copy.byte_length);
  if (rc) return rc;
  if (copy.shape[0] != 1 || copy.shape[1] != 2)
    return e3_fail(E3_SHAPE_MISMATCH, E3_CODE_SHAPE);
  if ((uintptr_t)copy.data % alignment != 0)
    return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
  return 0;
}

int64_t et_e3_private_stage_inputs_v1(
    void *pointer, const et_kernel_tensor_view_v1 *view) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase != E3_ACQUIRED || frame->stage_plane != 0 ||
      frame->cleanup_ready)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  int64_t rc = e3_stage_view(frame, view, "i64", 2 * sizeof(int64_t),
                             _Alignof(int64_t));
  if (rc) return rc;
  const int64_t *ids = view->data;
  for (size_t i = 0; i < 2; ++i)
    if (ids[i] < 0 || ids[i] >= 256)
      return e3_fail(E3_SHAPE_MISMATCH, E3_CODE_SHAPE);
  memcpy(frame->ids, ids, sizeof(frame->ids));
  frame->phase = E3_STAGING;
  frame->stage_plane = 1;
  return 0;
}

int64_t et_e3_private_stage_targets_v1(
    void *pointer, const et_kernel_tensor_view_v1 *view) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase != E3_STAGING || frame->stage_plane != 1 ||
      frame->cleanup_ready)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  int64_t rc = e3_stage_view(frame, view, "i64", 2 * sizeof(int64_t),
                             _Alignof(int64_t));
  if (rc) return rc;
  const int64_t *targets = view->data;
  for (size_t i = 0; i < 2; ++i)
    if (targets[i] < 0 || targets[i] >= 256)
      return e3_fail(E3_SHAPE_MISMATCH, E3_CODE_SHAPE);
  memcpy(frame->targets, targets, sizeof(frame->targets));
  frame->stage_plane = 2;
  return 0;
}

int64_t et_e3_private_stage_mask_v1(
    void *pointer, const et_kernel_tensor_view_v1 *view) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase != E3_STAGING || frame->stage_plane != 2 ||
      frame->cleanup_ready)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  int64_t rc = e3_stage_view(frame, view, "bool", 2, 1);
  if (rc) return rc;
  const unsigned char *mask = view->data;
  if (mask[0] > 1 || mask[1] > 1 || (mask[0] == 0 && mask[1] == 0))
    return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_PROFILE);
  memcpy(frame->mask, mask, sizeof(frame->mask));
  frame->stage_plane = 3;
  frame->next_role = 0;
  frame->phase = E3_FORWARD;
  return 0;
}

typedef struct e3_operand {
  et_f32_tensor *tensor;
  const et_kernel_tensor_view_v1 *view;
} e3_operand;

static e3_operand e3_tensor_operand(et_f32_tensor *tensor) {
  e3_operand operand = {tensor, NULL};
  return operand;
}

static e3_operand e3_view_operand(const et_kernel_tensor_view_v1 *view) {
  e3_operand operand = {NULL, view};
  return operand;
}

static int64_t e3_dispatch(et_e3_frame_internal *frame, size_t runtime,
                           const char *capability, const char *operation,
                           size_t rank, const uint64_t *shape,
                           const e3_operand *inputs, size_t input_count,
                           et_f32_tensor *const *outputs, size_t output_count) {
  et_f32_scoped_guard_internal guards[14] = {{0}};
  et_kernel_tensor_view_v1 input_views[9];
  et_kernel_tensor_view_v1 output_views[5];
  size_t opened = 0;
  int64_t rc = 0;
  for (size_t i = 0; i < input_count && !rc; ++i) {
    if (inputs[i].tensor) {
      rc = e3_scope_begin(inputs[i].tensor, &guards[opened], 0);
      if (!rc) input_views[i] = guards[opened++].view;
    } else if (inputs[i].view) {
      input_views[i] = *inputs[i].view;
    } else {
      rc = e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
    }
  }
  for (size_t i = 0; i < output_count && !rc; ++i) {
    rc = e3_scope_begin(outputs[i], &guards[opened], 0);
    if (!rc) output_views[i] = guards[opened++].view;
  }
  if (!rc) {
    et_kernel_request_v1 request = {
      sizeof(request), operation, "f32", "cpu", rank, shape, 1, {0}
    };
    et_kernel_call_v1 call = {
      sizeof(call), capability, &request,
      input_count, sizeof(input_views[0]), input_count * sizeof(input_views[0]),
      input_views,
      output_count, sizeof(output_views[0]), output_count * sizeof(output_views[0]),
      output_views
    };
    et_kernel_error error;
    if (et_kernel_runtime_dispatch(frame->runtime[runtime], &call, &error) != 0)
      rc = e3_kernel_error(&error);
  }
  while (opened) e3_scope_end(&guards[--opened]);
  return rc;
}

int64_t et_e3_private_forward_role_v1(void *pointer, int64_t role) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (role < 0 || role > 20)
    return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_PHASE);
  if (frame->phase != E3_FORWARD || frame->cleanup_ready ||
      frame->next_role != (uint32_t)role)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  et_kernel_tensor_view_v1 ids = e3_native_view(
      frame->ids, sizeof(frame->ids), "i64", 2, e3_shape_pair);
  et_kernel_tensor_view_v1 positions = e3_native_view(
      frame->positions, sizeof(frame->positions), "i64", 2, e3_shape_pair);
  et_kernel_tensor_view_v1 keep = e3_native_view(
      frame->keep, sizeof(frame->keep), "bool", 3, e3_shape_attention_mask);
  e3_operand in[6];
  et_f32_tensor *out[] = {frame->forward[role]};
  const char *capability = NULL, *operation = NULL;
  const uint64_t *shape = NULL;
  size_t rank = 0, count = 0, runtime = E3_RUNTIME_N3K;
#define TENSOR_INPUT(value) in[count++] = e3_tensor_operand(value)
#define VIEW_INPUT(value) in[count++] = e3_view_operand(value)
#define PIN_INPUT(index) in[count++] = e3_view_operand(&frame->pins.views[index])
  switch (role) {
    case 0:
      capability = "n3k.embedding-forward"; operation = "n3k.embedding.forward";
      shape = e3_shape_embed_token; rank = 4; VIEW_INPUT(&ids); PIN_INPUT(10); break;
    case 1:
      capability = "n3k.embedding-forward"; operation = "n3k.embedding.forward";
      shape = e3_shape_embed_position; rank = 4; VIEW_INPUT(&positions); PIN_INPUT(13); break;
    case 2:
      capability = "n3k.residual"; operation = "n3k.residual.forward";
      shape = e3_shape_token; rank = 3; TENSOR_INPUT(frame->forward[0]);
      TENSOR_INPUT(frame->forward[1]); break;
    case 3:
      runtime = E3_RUNTIME_N2; capability = "kernel.norm";
      operation = "layer-norm.forward"; shape = e3_shape_token; rank = 3;
      TENSOR_INPUT(frame->forward[2]); PIN_INPUT(7); PIN_INPUT(6);
      TENSOR_INPUT(frame->epsilon); break;
    case 4: case 5: case 6: {
      static const size_t pin[] = {2, 0, 3};
      capability = "n3k.linear"; operation = "n3k.linear.forward-no-bias";
      shape = e3_shape_linear_4; rank = 4; TENSOR_INPUT(frame->forward[3]);
      PIN_INPUT(pin[role - 4]); break;
    }
    case 7: case 8: case 9:
      capability = "n3k.head-layout"; operation = "n3k.heads.split.forward";
      shape = e3_shape_head; rank = 4; TENSOR_INPUT(frame->forward[role - 3]); break;
    case 10:
      runtime = E3_RUNTIME_A2; capability = "kernel.causal-attention";
      operation = "causal-attention.forward"; shape = e3_shape_attention; rank = 6;
      TENSOR_INPUT(frame->forward[7]); TENSOR_INPUT(frame->forward[8]);
      TENSOR_INPUT(frame->forward[9]); VIEW_INPUT(&positions);
      VIEW_INPUT(&positions); VIEW_INPUT(&keep); break;
    case 11:
      capability = "n3k.head-layout"; operation = "n3k.heads.merge.forward";
      shape = e3_shape_head; rank = 4; TENSOR_INPUT(frame->forward[10]); break;
    case 12:
      capability = "n3k.linear"; operation = "n3k.linear.forward-no-bias";
      shape = e3_shape_linear_4; rank = 4; TENSOR_INPUT(frame->forward[11]);
      PIN_INPUT(1); break;
    case 13:
      capability = "n3k.residual"; operation = "n3k.residual.forward";
      shape = e3_shape_token; rank = 3; TENSOR_INPUT(frame->forward[2]);
      TENSOR_INPUT(frame->forward[12]); break;
    case 14:
      runtime = E3_RUNTIME_N2; capability = "kernel.norm";
      operation = "layer-norm.forward"; shape = e3_shape_token; rank = 3;
      TENSOR_INPUT(frame->forward[13]); PIN_INPUT(9); PIN_INPUT(8);
      TENSOR_INPUT(frame->epsilon); break;
    case 15:
      capability = "n3k.linear"; operation = "n3k.linear.forward-no-bias";
      shape = e3_shape_linear_up; rank = 4; TENSOR_INPUT(frame->forward[14]);
      PIN_INPUT(5); break;
    case 16:
      capability = "n3k.gelu"; operation = "n3k.gelu.forward";
      shape = e3_shape_wide; rank = 3; TENSOR_INPUT(frame->forward[15]); break;
    case 17:
      capability = "n3k.linear"; operation = "n3k.linear.forward-no-bias";
      shape = e3_shape_linear_down; rank = 4; TENSOR_INPUT(frame->forward[16]);
      PIN_INPUT(4); break;
    case 18:
      capability = "n3k.residual"; operation = "n3k.residual.forward";
      shape = e3_shape_token; rank = 3; TENSOR_INPUT(frame->forward[13]);
      TENSOR_INPUT(frame->forward[17]); break;
    case 19:
      runtime = E3_RUNTIME_N2; capability = "kernel.norm";
      operation = "layer-norm.forward"; shape = e3_shape_token; rank = 3;
      TENSOR_INPUT(frame->forward[18]); PIN_INPUT(12); PIN_INPUT(11);
      TENSOR_INPUT(frame->epsilon); break;
    case 20:
      capability = "n3k.linear"; operation = "n3k.linear.forward-no-bias";
      shape = e3_shape_linear_vocab; rank = 4; TENSOR_INPUT(frame->forward[19]);
      PIN_INPUT(10); break;
  }
#undef TENSOR_INPUT
#undef VIEW_INPUT
#undef PIN_INPUT
  int64_t rc = e3_dispatch(frame, runtime, capability, operation, rank, shape,
                           in, count, out, 1);
  if (rc) return rc;
  ++frame->next_role;
  if (role == 20) frame->phase = E3_CE_READY;
  return 0;
}

int64_t et_e3_private_ce_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase != E3_CE_READY || frame->next_role != 21 ||
      frame->cleanup_ready)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  et_kernel_tensor_view_v1 targets = e3_native_view(
      frame->targets, sizeof(frame->targets), "i64", 2, e3_shape_pair);
  e3_operand inputs[] = {e3_tensor_operand(frame->forward[20]),
                         e3_view_operand(&targets)};
  et_f32_tensor *outputs[] = {frame->ce};
  int64_t rc = e3_dispatch(frame, E3_RUNTIME_L2,
      "kernel.indexed-cross-entropy", "indexed-cross-entropy.forward",
      3, e3_shape_logits, inputs, 2, outputs, 1);
  if (!rc) frame->next_role = 22;
  return rc;
}

int64_t et_e3_private_reduce_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase != E3_CE_READY || frame->next_role != 22 ||
      frame->cleanup_ready)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  et_kernel_tensor_view_v1 mask = e3_native_view(
      frame->mask, sizeof(frame->mask), "bool", 2, e3_shape_pair);
  e3_operand inputs[] = {e3_tensor_operand(frame->ce), e3_view_operand(&mask)};
  et_f32_tensor *outputs[] = {frame->batch[0], frame->batch[1], frame->batch[2]};
  int64_t rc = e3_dispatch(frame, E3_RUNTIME_L3S,
      "l3s.masked-objective", "l3s.masked-objective.reduce.bool",
      2, e3_shape_pair, inputs, 2, outputs, 3);
  if (!rc) frame->phase = E3_REDUCED;
  return rc;
}

int64_t et_e3_private_correct_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase != E3_REDUCED || frame->cleanup_ready)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  et_kernel_tensor_view_v1 targets = e3_native_view(
      frame->targets, sizeof(frame->targets), "i64", 2, e3_shape_pair);
  et_kernel_tensor_view_v1 mask = e3_native_view(
      frame->mask, sizeof(frame->mask), "bool", 2, e3_shape_pair);
  et_kernel_tensor_view_v1 active = e3_native_view(
      &frame->active, sizeof(frame->active), "i64", 0, NULL);
  et_f32_tensor *float_output[] = {frame->batch[3]};
  et_f32_scoped_guard_internal output_guard = {0};
  int64_t rc = e3_scope_begin(float_output[0], &output_guard, 0);
  if (!rc) {
    et_kernel_tensor_view_v1 outputs[] = {output_guard.view, active};
    et_kernel_tensor_view_v1 input_views[3];
    et_f32_scoped_guard_internal input_guard = {0};
    rc = e3_scope_begin(frame->forward[20], &input_guard, 0);
    if (!rc) {
      input_views[0] = input_guard.view;
      input_views[1] = targets;
      input_views[2] = mask;
      et_kernel_request_v1 request = {sizeof(request),
        "e3.evaluation-metrics.correct.bool", "f32", "cpu", 3,
        e3_shape_logits, 1, {0}};
      et_kernel_call_v1 call = {sizeof(call), "e3.evaluation-metrics", &request,
        3, sizeof(input_views[0]), sizeof(input_views), input_views,
        2, sizeof(outputs[0]), sizeof(outputs), outputs};
      et_kernel_error error;
      if (et_kernel_runtime_dispatch(frame->runtime[E3_RUNTIME_METRICS],
                                     &call, &error) != 0)
        rc = e3_kernel_error(&error);
      e3_scope_end(&input_guard);
    }
    e3_scope_end(&output_guard);
  }
  if (!rc) frame->phase = E3_CORRECTED;
  return rc;
}

static int64_t e3_accumulate_dispatch(et_e3_frame_internal *frame) {
  const size_t current = frame->sum_bank, next = current ^ 1u;
  et_f32_scoped_guard_internal guards[9] = {{0}};
  et_kernel_tensor_view_v1 inputs[9], outputs[5];
  size_t opened = 0;
  int64_t rc = 0;
  for (size_t i = 0; i < 3 && !rc; ++i) {
    rc = e3_scope_begin(frame->sum[current][i], &guards[opened], 0);
    if (!rc) inputs[i] = guards[opened++].view;
  }
  inputs[3] = e3_native_view(&frame->counts[current][0], sizeof(int64_t),
                             "i64", 0, NULL);
  inputs[4] = e3_native_view(&frame->counts[current][1], sizeof(int64_t),
                             "i64", 0, NULL);
  for (size_t i = 0; i < 3 && !rc; ++i) {
    const size_t batch_index = i < 2 ? i : 3;
    rc = e3_scope_begin(frame->batch[batch_index], &guards[opened], 0);
    if (!rc) inputs[5 + i] = guards[opened++].view;
  }
  inputs[8] = e3_native_view(&frame->active, sizeof(frame->active),
                             "i64", 0, NULL);
  for (size_t i = 0; i < 3 && !rc; ++i) {
    rc = e3_scope_begin(frame->sum[next][i], &guards[opened], 0);
    if (!rc) outputs[i] = guards[opened++].view;
  }
  outputs[3] = e3_native_view(&frame->counts[next][0], sizeof(int64_t),
                              "i64", 0, NULL);
  outputs[4] = e3_native_view(&frame->counts[next][1], sizeof(int64_t),
                              "i64", 0, NULL);
  if (!rc) {
    et_kernel_request_v1 request = {sizeof(request),
      "e3.evaluation-metrics.accumulate", "f32", "cpu", 3,
      e3_shape_logits, 1, {0}};
    et_kernel_call_v1 call = {sizeof(call), "e3.evaluation-metrics", &request,
      9, sizeof(inputs[0]), sizeof(inputs), inputs,
      5, sizeof(outputs[0]), sizeof(outputs), outputs};
    et_kernel_error error;
    if (et_kernel_runtime_dispatch(frame->runtime[E3_RUNTIME_METRICS],
                                   &call, &error) != 0)
      rc = e3_kernel_error(&error);
  }
  while (opened) e3_scope_end(&guards[--opened]);
  return rc;
}

int64_t et_e3_private_accumulate_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase != E3_CORRECTED || frame->cleanup_ready)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  int64_t rc = e3_accumulate_dispatch(frame);
  if (!rc) {
    frame->sum_bank ^= 1u;
    frame->phase = E3_ACQUIRED;
    frame->stage_plane = 0;
    frame->next_role = 0;
  }
  return rc;
}

int64_t et_e3_private_finalize_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase != E3_ACQUIRED || frame->stage_plane != 0 ||
      frame->cleanup_ready)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  const size_t bank = frame->sum_bank;
  if (frame->counts[bank][1] <= 0 || frame->counts[bank][0] <= 0)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  e3_operand inputs[] = {e3_tensor_operand(frame->sum[bank][0]),
                         e3_tensor_operand(frame->sum[bank][1]),
                         e3_tensor_operand(frame->sum[bank][2])};
  et_f32_tensor *outputs[] = {frame->result[0], frame->result[1],
                              frame->result[2], frame->result[3]};
  int64_t rc = e3_dispatch(frame, E3_RUNTIME_METRICS,
      "e3.evaluation-metrics", "e3.evaluation-metrics.finalize",
      3, e3_shape_logits, inputs, 3, outputs, 4);
  if (!rc) {
    frame->staged_counts[0] = frame->counts[bank][0];
    frame->staged_counts[1] = frame->counts[bank][1];
    frame->committed = 1;
    frame->phase = E3_FINALIZED;
  }
  return rc;
}

int64_t et_e3_private_cleanup_preflight_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase < E3_ACQUIRED || frame->phase > E3_FINALIZED ||
      frame->cleanup_ready)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  if ((frame->phase == E3_FINALIZED) != (frame->committed == 1))
    return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  et_f32_tensor_error error;
  if (et_g3t_model_pins_check_internal(&frame->pins, &error) != 0)
    return e3_f32_error(&error);
  int64_t rc = e3_storage_check(frame, 1);
  if (rc) return rc;
  frame->cleanup_ready = 1;
  return 0;
}

void et_e3_private_drain_v1(void *pointer) {
  et_e3_frame_internal *frame = e3_find(pointer);
  if (!frame || frame->lifecycle != 1 || !frame->cleanup_ready ||
      frame->phase < E3_ACQUIRED || frame->phase > E3_FINALIZED ||
      frame->model->active != frame)
    abort();
  et_g3t_model_pins_end_internal(&frame->pins);
  frame->phase = E3_DRAINED;
}

static int64_t e3_publication_counts(et_e3_frame_internal *frame) {
  const int64_t tokens = frame->staged_counts[0];
  const int64_t batches = frame->staged_counts[1];
  if (tokens < 1 || tokens > 16384 || batches < 1 || batches > 8192 ||
      batches > tokens || tokens - batches > batches)
    return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  return 0;
}

int64_t et_e3_private_publish_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase != E3_DRAINED || !frame->cleanup_ready ||
      frame->committed != 1)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  if (!e3_pins_idle(&frame->pins))
    return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  et_f32_scoped_guard_internal destinations[4] = {{0}}, results[4] = {{0}};
  size_t destination_count = 0, result_count = 0;
  int64_t rc = 0;
  float staged[4];
  for (size_t i = 0; i < 4 && !rc; ++i) {
    rc = e3_scope_begin(frame->destination[i], &destinations[i], 1);
    if (!rc) ++destination_count;
  }
  for (size_t i = 0; i < 4 && !rc; ++i) {
    rc = e3_scope_begin(frame->result[i], &results[i], 0);
    if (!rc) ++result_count;
  }
  if (!rc) rc = e3_publication_counts(frame);
  if (!rc) {
    for (size_t i = 0; i < 4; ++i) {
      if (destinations[i].self != &destinations[i] ||
          destinations[i].owner != frame->destination[i] ||
          destinations[i].view.rank != 0 || destinations[i].view.shape ||
          destinations[i].view.byte_length != sizeof(float) ||
          results[i].self != &results[i] ||
          results[i].owner != frame->result[i] || results[i].view.rank != 0 ||
          results[i].view.shape || results[i].view.byte_length != sizeof(float)) {
        rc = e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
        break;
      }
      memcpy(&staged[i], results[i].view.data, sizeof(staged[i]));
    }
  }
  while (result_count) e3_scope_end(&results[--result_count]);
  if (rc) {
    while (destination_count) e3_scope_end(&destinations[--destination_count]);
    return rc;
  }
  for (size_t i = 0; i < 4; ++i)
    if (destinations[i].self != &destinations[i] ||
        destinations[i].owner != frame->destination[i] ||
        destinations[i].view.rank != 0 || destinations[i].view.shape ||
        destinations[i].view.byte_length != sizeof(float))
      abort();
  for (size_t i = 0; i < 4; ++i)
    memcpy(destinations[i].view.data, &staged[i], sizeof(staged[i]));
  frame->published_counts[0] = frame->staged_counts[0];
  frame->published_counts[1] = frame->staged_counts[1];
  frame->published_valid = 1;
  frame->committed = 2;
  while (destination_count) e3_scope_end(&destinations[--destination_count]);
  return 0;
}

void et_e3_private_finish_v1(void *pointer) {
  et_e3_frame_internal *frame = e3_find(pointer);
  if (!frame || frame->lifecycle != 1 || frame->phase != E3_DRAINED ||
      !frame->cleanup_ready || frame->model->active != frame ||
      !e3_pins_idle(&frame->pins))
    abort();
  frame->model->active = NULL;
  frame->phase = E3_IDLE;
  frame->next_role = 0;
  frame->stage_plane = 0;
  frame->sum_bank = 0;
  frame->cleanup_ready = 0;
  frame->active = 0;
  memset(frame->ids, 0, sizeof(frame->ids));
  memset(frame->targets, 0, sizeof(frame->targets));
  memset(frame->mask, 0, sizeof(frame->mask));
  memset(frame->staged_counts, 0, sizeof(frame->staged_counts));
  if (frame->committed != 2) frame->committed = 0;
}

int64_t et_e3_private_counter_ref_v1(void *pointer, int64_t selector) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return -1;
  if (selector < 0 || selector > 1) {
    e3_fail(E3_INVALID_ARGUMENT, E3_CODE_PHASE);
    return -1;
  }
  if (frame->phase != E3_IDLE || !frame->published_valid) {
    e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
    return -1;
  }
  return frame->published_counts[selector];
}

int64_t et_e3_private_frame_destroy_preflight_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (frame->phase != E3_IDLE || frame->model->active ||
      !e3_pins_idle(&frame->pins))
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  return e3_storage_check(frame, 1);
}

void et_e3_private_frame_destroy_v1(void *pointer) {
  et_e3_frame_internal *frame = e3_find(pointer);
  if (!frame || frame->lifecycle != 1 || frame->phase != E3_IDLE ||
      !frame->model || frame->model->active || !e3_pins_idle(&frame->pins))
    abort();
  for (size_t i = 0; i < 6; ++i)
    if (!frame->runtime[i]) abort();
  for (size_t i = 6; i > 0; --i) {
    et_kernel_runtime_destroy(frame->runtime[i - 1]);
    frame->runtime[i - 1] = NULL;
  }
  e3_destroy_owned(frame);
  frame->model = NULL;
  memset(frame->destination, 0, sizeof(frame->destination));
  memset(&frame->pins, 0, sizeof(frame->pins));
  memset(frame->ids, 0, sizeof(frame->ids));
  memset(frame->targets, 0, sizeof(frame->targets));
  memset(frame->positions, 0, sizeof(frame->positions));
  memset(frame->keep, 0, sizeof(frame->keep));
  memset(frame->mask, 0, sizeof(frame->mask));
  memset(frame->counts, 0, sizeof(frame->counts));
  memset(frame->staged_counts, 0, sizeof(frame->staged_counts));
  memset(frame->published_counts, 0, sizeof(frame->published_counts));
  frame->active = 0;
  frame->phase = E3_IDLE;
  frame->lifecycle = 2;
  frame->next_role = frame->stage_plane = frame->sum_bank = 0;
  frame->committed = frame->cleanup_ready = frame->published_valid = 0;
}
