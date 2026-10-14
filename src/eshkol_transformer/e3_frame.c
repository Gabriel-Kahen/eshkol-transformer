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

#if defined(ET_E3_NUMERICAL_SUM_ORDER) && !defined(ET_E3_NUMERICAL_TESTING)
#error "sum-order mutation requires the private numerical test tuple"
#endif
#if defined(ET_E3_NUMERICAL_PADDED_POSITION) && !defined(ET_E3_NUMERICAL_TESTING)
#error "padded-position mutation requires the private numerical test tuple"
#endif

#if defined(ET_E3_NUMERICAL_TESTING) || defined(ET_E3_ZERO_CONTROL_TESTING)
#include <inttypes.h>
#include <stdio.h>
#endif
#ifdef ET_E3_ZERO_CONTROL_TESTING
#include "f32_parameter_internal.h"
extern et_f32_tensor *et_m3t_test_parameter_gradient(et_f32_parameter *);
#endif

#ifdef ET_E3_RETENTION_TESTING
#include "d2_native.h"
extern et_f32_tensor *et_m3t_test_parameter_gradient(et_f32_parameter *);
#endif

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
#ifdef ET_E3_NUMERICAL_SUM_ORDER
  float sum_order_first, sum_order_second;
  uint32_t sum_order_seen;
#endif
#ifdef ET_E3_FAULT_TESTING
  /* State: 0 clear, 1 armed, 2 fired, 3 unreached after one call. */
  uint32_t fault_state, fault_site, fault_ordinal, fault_occurrence;
  uint32_t fault_reached, fault_started;
#endif
};

static et_e3_frame_internal *e3_registry;
static int64_t e3_last_domain, e3_last_category, e3_last_code;

#ifdef ET_E3_TESTING
static size_t e3_test_frame_alloc_remaining = SIZE_MAX;
/* Select one genuine K1 provider discovery during a source-composed test. */
static int64_t e3_test_discovery_fail_index = -1;
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

#ifdef ET_E3_FAULT_TESTING
static int e3_fault_idle(const et_e3_frame_internal *frame) {
  return frame->phase == E3_IDLE && !frame->cleanup_ready &&
      !frame->model->active && e3_pins_idle(&frame->pins);
}

int64_t et_e3_test_fault_arm_v1(void *pointer, int64_t site,
                                int64_t ordinal, int64_t occurrence) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (!e3_fault_idle(frame)) return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  if (frame->fault_state) return e3_fail(E3_INVALID_STATE, E3_CODE_REENTRANCY);
  if (!((site == 1 && ordinal >= 0 && ordinal <= 20 &&
         occurrence >= 1 && occurrence <= 8192) ||
        (site == 2 && ordinal == 0 && occurrence == 1)))
    return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_PHASE);
  frame->fault_state = 1;
  frame->fault_site = (uint32_t)site;
  frame->fault_ordinal = (uint32_t)ordinal;
  frame->fault_occurrence = (uint32_t)occurrence;
  frame->fault_reached = 0;
  frame->fault_started = 0;
  return 0;
}

int64_t et_e3_test_fault_ref_v1(void *pointer, int64_t field) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return -1;
  if (!e3_fault_idle(frame)) {
    e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
    return -1;
  }
  switch (field) {
    case 0: return frame->fault_state;
    case 1: return frame->fault_site;
    case 2: return frame->fault_ordinal;
    case 3: return frame->fault_occurrence;
    case 4: return frame->fault_reached;
    default: e3_fail(E3_INVALID_ARGUMENT, E3_CODE_PHASE); return -1;
  }
}

int64_t et_e3_test_fault_reset_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (!e3_fault_idle(frame)) return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  frame->fault_state = frame->fault_site = frame->fault_ordinal = 0;
  frame->fault_occurrence = frame->fault_reached = frame->fault_started = 0;
  return 0;
}

/* Distinct, finite pre-call words make each of the six publication stores
 * observable. Only an idle frame with an earlier successful publication may
 * be seeded; a copy failure is a fixture invariant, not an E3 fault cut. */
int64_t et_e3_test_fault_seed_publication_v1(void *pointer) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return e3_last_domain ?
      e3_dependency_status(e3_last_domain, e3_last_category) : e3_last_category;
  if (!e3_fault_idle(frame) || frame->fault_state ||
      !frame->published_valid || frame->committed != 2)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  const uint32_t sentinels[4] = {UINT32_C(0x3f800000), UINT32_C(0x40000000),
                                  UINT32_C(0x40400000), UINT32_C(0x40800000)};
  et_f32_tensor_error error;
  for (size_t i = 0; i < 4; ++i) {
    et_f32_tensor_error_clear_v1(&error);
    if (et_f32_tensor_copy_bits_from_v1(frame->destination[i],
                                        &sentinels[i], 1, &error) != 0)
      abort();
  }
  frame->published_counts[0] = 7;
  frame->published_counts[1] = 4;
  return 0;
}
#endif

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

#ifdef ET_E3_NUMERICAL_TESTING
/* Exact noninstalled tuple only. Read each completed role before the next D2
 * batch reuses the frame; no value is fabricated from the reference bundle. */
static const char *const e3_numerical_role_names[21] = {
  "ET", "EP", "X", "N1", "QT", "KT", "VT", "QH", "KH", "VH", "AH",
  "AT", "AO", "R", "N2", "FU", "FG", "FD", "Y", "NF", "Z"
};

static void e3_numerical_words(et_f32_tensor *tensor, size_t expected) {
  et_f32_scoped_guard_internal guard = {0};
  if (e3_scope_begin(tensor, &guard, 0) != 0 ||
      guard.view.byte_length != expected * sizeof(uint32_t)) abort();
  for (size_t i = 0; i < expected; ++i) {
    uint32_t word;
    memcpy(&word, (const unsigned char *)guard.view.data + i * sizeof(word),
           sizeof(word));
    if (i) printf(",");
    printf("%08" PRIx32, word);
  }
  e3_scope_end(&guard);
}

static void e3_numerical_batch(et_e3_frame_internal *frame) {
  const size_t bank = frame->sum_bank;
  const int64_t ordinal = frame->counts[bank][1] - 1;
  if (ordinal < 0 || ordinal >= 3) abort();
  printf("E3ROW\t%" PRId64 "\t%" PRId64 ",%" PRId64
         "\t%" PRId64 ",%" PRId64 "\t%u,%u\n", ordinal,
         frame->ids[0], frame->ids[1], frame->targets[0], frame->targets[1],
         (unsigned)frame->mask[0], (unsigned)frame->mask[1]);
  for (size_t role = 0; role < 21; ++role) {
    et_f32_scoped_guard_internal guard = {0};
    if (e3_scope_begin(frame->forward[role], &guard, 0) != 0 ||
        guard.view.rank < 1 || guard.view.rank > 4 ||
        guard.view.byte_length % sizeof(uint32_t) != 0) abort();
    printf("E3ROLE\t%" PRId64 "\t%s\t", ordinal,
           e3_numerical_role_names[role]);
    for (size_t axis = 0; axis < guard.view.rank; ++axis) {
      if (axis) printf(",");
      printf("%" PRIu64, guard.view.shape[axis]);
    }
    printf("\t");
    const size_t count = guard.view.byte_length / sizeof(uint32_t);
    for (size_t i = 0; i < count; ++i) {
      uint32_t word;
      memcpy(&word, (const unsigned char *)guard.view.data + i * sizeof(word),
             sizeof(word));
      if (i) printf(",");
      printf("%08" PRIx32, word);
    }
    printf("\n");
    e3_scope_end(&guard);
  }
  printf("E3CE\t%" PRId64 "\t", ordinal);
  e3_numerical_words(frame->ce, 2);
  printf("\n");
  printf("E3BATCH\t%" PRId64 "\t", ordinal);
  for (size_t i = 0; i < 4; ++i) {
    if (i) printf(",");
    e3_numerical_words(frame->batch[i], 1);
  }
  printf("\t%" PRId64 "\n", frame->active);
  printf("E3STATE\t%" PRId64 "\t", ordinal);
  for (size_t i = 0; i < 3; ++i) {
    if (i) printf(",");
    e3_numerical_words(frame->sum[bank][i], 1);
  }
  printf("\t%" PRId64 ",%" PRId64 "\n",
         frame->counts[bank][0], frame->counts[bank][1]);
}

static void e3_numerical_final(et_e3_frame_internal *frame) {
  printf("E3FINAL\t");
  for (size_t i = 0; i < 4; ++i) {
    if (i) printf(",");
    e3_numerical_words(frame->destination[i], 1);
  }
  printf("\t%" PRId64 ",%" PRId64 "\n",
         frame->published_counts[0], frame->published_counts[1]);
  if (fflush(NULL) != 0) abort();
}
#endif

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
    for (size_t i = 0; i < 6 && !rc; ++i) {
      const et_kernel_provider_v1 *provider = providers[i];
#ifdef ET_E3_TESTING
      if (e3_test_discovery_fail_index == (int64_t)i) {
        e3_test_discovery_fail_index = -1;
        provider = NULL;
      }
#endif
      rc = e3_discover(&frame->runtime[i], provider);
    }
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
#ifdef ET_E3_FAULT_TESTING
  if (frame->fault_state > 1 ||
      (frame->fault_state == 1 && frame->fault_started))
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  if (frame->fault_state == 1) frame->fault_started = 1;
#endif
  int64_t rc = e3_storage_check(frame, 1);
  if (rc) {
#ifdef ET_E3_FAULT_TESTING
    if (frame->fault_state == 1) frame->fault_state = 3;
#endif
    return rc;
  }
  for (size_t bank = 0; bank < 2; ++bank)
    for (size_t i = 0; i < 3; ++i)
      if ((rc = e3_zero_tensor(frame->sum[bank][i])) != 0) {
#ifdef ET_E3_FAULT_TESTING
        if (frame->fault_state == 1) frame->fault_state = 3;
#endif
        return rc;
      }
  memset(frame->counts, 0, sizeof(frame->counts));
  frame->active = 0;
  memset(frame->staged_counts, 0, sizeof(frame->staged_counts));
  frame->sum_bank = 0;
#ifdef ET_E3_NUMERICAL_SUM_ORDER
  frame->sum_order_first = frame->sum_order_second = 0.0f;
  frame->sum_order_seen = 0;
#endif
  frame->committed = 0;
  frame->next_role = 0;
  frame->stage_plane = 0;
  et_f32_tensor_error error;
  if (et_g3t_model_pins_begin_internal(frame->model->p,
                                       (const void *const *)frame->model->handles,
                                       &frame->pins, &error) != 0) {
#ifdef ET_E3_FAULT_TESTING
    if (frame->fault_state == 1) frame->fault_state = 3;
#endif
    return e3_f32_error(&error);
  }
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
#ifdef ET_E3_FAULT_TESTING
  if (frame->fault_state == 1 && frame->fault_site == 1 &&
      frame->fault_ordinal == (uint32_t)role) {
    ++frame->fault_reached;
    if (frame->fault_reached == frame->fault_occurrence) {
      frame->fault_state = 2; /* Consume before any provider dispatch. */
      return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
    }
  }
#endif
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
#ifdef ET_E3_NUMERICAL_WRONG_TARGET
      frame->ids, sizeof(frame->ids), "i64", 2, e3_shape_pair);
#else
      frame->targets, sizeof(frame->targets), "i64", 2, e3_shape_pair);
#endif
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
#ifdef ET_E3_NUMERICAL_PADDED_POSITION
  unsigned char padded_mask[2] = {1, 1};
  if (frame->mask[0] == 1 && frame->mask[1] == 0)
    mask.data = padded_mask;
#endif
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
#ifdef ET_E3_NUMERICAL_PADDED_POSITION
  unsigned char padded_mask[2] = {1, 1};
  if (frame->mask[0] == 1 && frame->mask[1] == 0)
    mask.data = padded_mask;
#endif
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
#ifdef ET_E3_NUMERICAL_SUM_ORDER
  const int64_t ordinal = frame->counts[current][1];
  float batch_n = 0.0f, grouped_tail = 0.0f;
#endif
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
#ifdef ET_E3_NUMERICAL_SUM_ORDER
    /* Only the third real CE batch changes operand grouping. The accepted K1
     * accumulate dispatch still computes and writes the mutated state. */
    memcpy(&batch_n, inputs[5].data, sizeof(batch_n));
    if (ordinal == 2) {
      if (frame->sum_order_seen != 2) abort();
      volatile float rounded_tail = frame->sum_order_second + batch_n;
      grouped_tail = rounded_tail;
      inputs[0].data = &frame->sum_order_first;
      inputs[5].data = &grouped_tail;
    }
#endif
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
#ifdef ET_E3_NUMERICAL_SUM_ORDER
    if (!rc && ordinal < 2) {
      if (ordinal < 0 || frame->sum_order_seen != (uint32_t)ordinal) abort();
      if (ordinal == 0) frame->sum_order_first = batch_n;
      else frame->sum_order_second = batch_n;
      ++frame->sum_order_seen;
    }
#endif
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
#ifdef ET_E3_NUMERICAL_TESTING
    e3_numerical_batch(frame);
#endif
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
#ifdef ET_E3_FAULT_TESTING
  if (frame->fault_state == 1 && frame->fault_site == 2) {
    ++frame->fault_reached;
    if (frame->fault_reached == frame->fault_occurrence) {
      frame->fault_state = 2;
      while (destination_count) e3_scope_end(&destinations[--destination_count]);
      return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
    }
  }
#endif
  for (size_t i = 0; i < 4; ++i)
    memcpy(destinations[i].view.data, &staged[i], sizeof(staged[i]));
  frame->published_counts[0] = frame->staged_counts[0];
  frame->published_counts[1] = frame->staged_counts[1];
  frame->published_valid = 1;
  frame->committed = 2;
  while (destination_count) e3_scope_end(&destinations[--destination_count]);
#ifdef ET_E3_NUMERICAL_TESTING
  e3_numerical_final(frame);
#endif
  return 0;
}

void et_e3_private_finish_v1(void *pointer) {
  et_e3_frame_internal *frame = e3_find(pointer);
  if (!frame || frame->lifecycle != 1 || frame->phase != E3_DRAINED ||
      !frame->cleanup_ready || frame->model->active != frame ||
      !e3_pins_idle(&frame->pins))
    abort();
  frame->model->active = NULL;
#ifdef ET_E3_FAULT_TESTING
  if (frame->fault_state == 1 && frame->fault_started)
    frame->fault_state = 3;
#endif
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

int64_t et_e3_private_selected_metric_bits_ref_v1(
    void *pointer, int64_t selector) {
  e3_clear();
  et_e3_frame_internal *frame = e3_admit(pointer);
  if (!frame) return -1;
  if (selector < 0 || selector > 1) {
    e3_fail(E3_INVALID_ARGUMENT, E3_CODE_PHASE);
    return -1;
  }
  if (frame->phase != E3_IDLE || !frame->published_valid ||
      frame->committed != 2) {
    e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
    return -1;
  }
  if (frame->model->active || frame->cleanup_ready || frame->next_role ||
      frame->stage_plane || frame->sum_bank || frame->active ||
      frame->staged_counts[0] || frame->staged_counts[1] ||
      !e3_pins_idle(&frame->pins)) {
    e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
    return -1;
  }
  et_f32_scoped_guard_internal guard = {0};
  uint32_t bits = 0;
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (et_f32_tensor_scoped_begin_internal(frame->destination[selector], &guard,
                                          &error) != 0) {
    e3_f32_error(&error);
    return -1;
  }
  if (et_f32_tensor_scoped_end_internal(&guard) != 0) abort();
  if (et_f32_tensor_copy_bits_to_v1(frame->destination[selector], &bits, 1,
                                    &error) != 0) {
    e3_f32_error(&error);
    return -1;
  }
  return (int64_t)bits;
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
#ifdef ET_E3_FAULT_TESTING
  if (frame->fault_state)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
#endif
  return e3_storage_check(frame, 1);
}

void et_e3_private_frame_destroy_v1(void *pointer) {
  et_e3_frame_internal *frame = e3_find(pointer);
  if (!frame || frame->lifecycle != 1 || frame->phase != E3_IDLE ||
      !frame->model || frame->model->active || !e3_pins_idle(&frame->pins))
    abort();
#ifdef ET_E3_FAULT_TESTING
  if (frame->fault_state) abort();
#endif
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

#ifdef ET_E3_RETENTION_TESTING
/* Mode/graph-only noninstalled probes. Snapshots do not create owners. */
int64_t et_e3_test_retention_probe_v1(int64_t selector) {
  et_f32_test_live_counts_v1 live = {.struct_size = sizeof(live)};
  et_f32_test_retired_counts_v1 retired = {.struct_size = sizeof(retired)};
  et_f32_test_borrow_event_counts_v1 events = {.struct_size = sizeof(events)};
  et_f32_test_live_counts_snapshot_v1(&live);
  et_f32_test_retired_counts_snapshot_v1(&retired);
  et_f32_test_borrow_event_counts_snapshot_v1(&events);
  size_t frames = 0, dead_frames = 0, busy_frames = 0, graphs = 0;
  for (const et_e3_frame_internal *f = e3_registry; f; f = f->next) {
    if (f->lifecycle == 1) {
      ++frames;
      if (f->phase != E3_IDLE || !e3_pins_idle(&f->pins)) ++busy_frames;
    } else if (f->lifecycle == 2) {
      ++dead_frames;
    }
  }
  for (const record *r = registry; r; r = r->next)
    if (r->kind == M3_GRAPH && r->state >= 0) ++graphs;
  switch (selector) {
    case 0: return (int64_t)live.tensors;
    case 1: return (int64_t)live.parameters;
    case 2: return (int64_t)live.owned_clones;
    case 3: return (int64_t)live.borrows;
    case 4: return (int64_t)live.copy_plans;
    case 5: return (int64_t)live.gradient_plans;
    case 6: return (int64_t)live.reset_plans;
    case 7: return (int64_t)retired.tensors;
    case 8: return (int64_t)retired.parameters;
    case 9: return (int64_t)retired.borrows;
    case 10: return (int64_t)retired.copy_plans;
    case 11: return (int64_t)retired.gradient_plans;
    case 12: return (int64_t)retired.reset_plans;
    case 13: return (int64_t)retired.retained_control_bytes;
    case 14: return (int64_t)events.begin_calls;
    case 15: return (int64_t)events.view_calls;
    case 16: return (int64_t)events.end_calls;
    case 17: return (int64_t)frames;
    case 18: return (int64_t)dead_frames;
    case 19: return (int64_t)(dead_frames * sizeof(et_e3_frame_internal));
    case 20: return (int64_t)busy_frames;
    case 21: return (int64_t)graphs;
    case 22: return et_d2_dataset_test_live_count_v1();
    case 23: return et_d2_batch_test_live_count_v1();
    case 24: return et_d2_batch_test_borrow_count_v1();
    case 25: return et_d2_test_owned_allocation_count_v1();
    case 26: return et_d2_exact_read_test_fd_live_count_v1();
    case 27: return et_d2_exact_read_test_fd_peak_count_v1();
    case 28: return (int64_t)et_f32_tensor_test_borrow_bytes_v1();
    default: return -1;
  }
}

#ifdef ET_E3_OVERFLOW_TESTING
/* Overflow-only private witness: observe storage even before publication.
 * e3_find checks registry identity before any caller-pointer dereference.
 * This deliberately does not call e3_admit or change native error state. */
int64_t et_e3_test_published_counter_raw_v1(void *pointer, int64_t selector) {
  et_e3_frame_internal *frame = e3_find(pointer);
  if (!frame || frame->lifecycle != 1 || frame->phase != E3_IDLE ||
      !frame->model || frame->model->active || !e3_pins_idle(&frame->pins) ||
      selector < 0 || selector > 1) return -1;
  return frame->published_counts[selector];
}
#endif

static et_f32_tensor *e3_test_model_tensor(owner *model, int64_t parameter,
                                            int64_t gradient) {
  if (!model || model->r.state != 1 || parameter < 0 || parameter >= 14 ||
      (gradient != 0 && gradient != 1)) return NULL;
  return gradient ? et_m3t_test_parameter_gradient(model->p[parameter])
                  : value(model, (size_t)parameter);
}

int64_t et_e3_test_model_word_count_v1(void *pointer, int64_t parameter,
                                        int64_t gradient) {
  owner *model = admit(pointer, OWNER, 0);
  et_f32_tensor *tensor = e3_test_model_tensor(model, parameter, gradient);
  size_t bytes = 0;
  et_f32_tensor_error error;
  if (!tensor || et_f32_tensor_byte_length_v1(tensor, &bytes, &error) != 0 ||
      bytes % sizeof(uint32_t) || bytes > 4096) return -1;
  return (int64_t)(bytes / sizeof(uint32_t));
}

int64_t et_e3_test_model_word_v1(void *pointer, int64_t parameter,
                                  int64_t gradient, int64_t index) {
  const int64_t count =
      et_e3_test_model_word_count_v1(pointer, parameter, gradient);
  if (count < 0 || index < 0 || index >= count) return -1;
  owner *model = admit(pointer, OWNER, 0);
  et_f32_tensor *tensor = e3_test_model_tensor(model, parameter, gradient);
  uint32_t words[1024] = {0};
  et_f32_tensor_error error;
  if (et_f32_tensor_copy_bits_to_v1(tensor, words, (size_t)count, &error) != 0)
    return -1;
  return (int64_t)words[index];
}

int64_t et_e3_test_gradient_metadata_v1(void *pointer, int64_t parameter,
                                         int64_t field) {
  owner *model = admit(pointer, OWNER, 0);
  if (!model || model->r.state != 1 || parameter < 0 || parameter >= 14 ||
      field < 0 || field > 2) return -1;
  et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
  et_f32_tensor_error error;
  if (et_f32_parameter_gradient_metadata_v1(model->p[parameter], &metadata,
                                             &error) != 0) return -1;
  return field == 0 ? (int64_t)metadata.state
       : field == 1 ? (int64_t)metadata.contribution_count
                    : (int64_t)metadata.normalization_weight_bits;
}

int64_t et_e3_test_seed_gradient_v1(void *pointer) {
  owner *model = admit(pointer, OWNER, 0);
  if (!model || model->r.state != 1 || model->active) return -1;
  et_f32_tensor *gradient = e3_test_model_tensor(model, 0, 1);
  size_t bytes = 0;
  et_f32_tensor_error error;
  if (!gradient || et_f32_tensor_byte_length_v1(gradient, &bytes, &error) != 0 ||
      bytes % sizeof(uint32_t) || bytes == 0 || bytes > 4096) return -1;
  uint32_t words[1024] = {UINT32_C(0x3f000000)};
  et_f32_parameter_test_set_gradient_bits_v1(model->p[0], words,
                                               bytes / sizeof(uint32_t));
  et_f32_parameter_test_set_metadata_v1(model->p[0], ET_F32_GRADIENT_PRESENT,
                                        1u, UINT32_C(0x3f800000));
  return 0;
}
#endif

#ifdef ET_E3_ZERO_CONTROL_TESTING
/* Exact noninstalled read-only observation of the registered M3 gradient.
 * The fixture seeds via I2's existing accumulation API, never by this seam. */
static et_f32_tensor *e3_zero_gradient(void *pointer, int64_t parameter) {
  owner *model = admit(pointer, OWNER, 0);
  if (!model || model->r.state != 1 || parameter < 0 || parameter >= 14)
    return NULL;
  return et_m3t_test_parameter_gradient(model->p[parameter]);
}

int64_t et_e3_zero_gradient_word_count_v1(void *pointer, int64_t parameter) {
  et_f32_tensor *tensor = e3_zero_gradient(pointer, parameter);
  et_f32_tensor_error error;
  size_t bytes = 0;
  et_f32_tensor_error_clear_v1(&error);
  if (!tensor || et_f32_tensor_byte_length_v1(tensor, &bytes, &error) != 0 ||
      bytes % sizeof(uint32_t) != 0 || bytes > 4096) return -1;
  return (int64_t)(bytes / sizeof(uint32_t));
}

int64_t et_e3_zero_gradient_word_v1(void *pointer, int64_t parameter,
                                     int64_t index) {
  const int64_t count = et_e3_zero_gradient_word_count_v1(pointer, parameter);
  et_f32_tensor *tensor = e3_zero_gradient(pointer, parameter);
  et_f32_tensor_error error;
  uint32_t words[1024] = {0};
  et_f32_tensor_error_clear_v1(&error);
  if (count < 0 || index < 0 || index >= count || !tensor ||
      et_f32_tensor_copy_bits_to_v1(tensor, words, (size_t)count, &error) != 0)
    return -1;
  return (int64_t)words[index];
}

int64_t et_e3_zero_gradient_metadata_v1(void *pointer, int64_t parameter,
                                         int64_t field) {
  owner *model = admit(pointer, OWNER, 0);
  et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (!model || model->r.state != 1 || parameter < 0 || parameter >= 14 ||
      field < 0 || field > 2 ||
      et_f32_parameter_gradient_metadata_v1(model->p[parameter], &metadata,
                                             &error) != 0) return -1;
  return field == 0 ? (int64_t)metadata.state
       : field == 1 ? (int64_t)metadata.contribution_count
                    : (int64_t)metadata.normalization_weight_bits;
}

/* Test-only controls are bound to one registered M3 owner and the exact decoded
 * clone rooted by the Eshkol driver. They never write a parameter or gradient. */
extern void et_i2_test_fail_alloc_after_v1(size_t);
extern void et_i2_test_reset_allocator_v1(void);
extern void et_i2_test_live_builder_counts_v1(size_t *, size_t *, size_t *);
extern int64_t et_i2_private_owned_clone_live_count_v1(void);

static struct e3_zero_control_slot {
  owner *model;
  int64_t index;
  et_f32_tensor *source;
  const et_f32_tensor *destination;
  et_f32_tensor_borrow *borrow;
  et_f32_tensor_copy_plan *pin;
  owner *fail_model;
  int fail_domain;
} e3_zero_control;

static int e3_zero_control_owner(void *pointer, int64_t index,
                                 owner **actual,
                                 const et_f32_tensor **value) {
  owner *model = admit(pointer, OWNER, 0);
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (!model || model->r.state != 1 || index < 0 || index >= 14 ||
      !model->p[index] || !model->handles[index] ||
      et_f32_parameter_validate_identity_v1(model->p[index],
                                             model->handles[index], &error) != 0 ||
      et_f32_parameter_value_tensor_v1(model->p[index], value, &error) != 0)
    return -1;
  *actual = model;
  return 0;
}

static int e3_zero_control_identical(void *pointer, int64_t index,
                                     void *source) {
  owner *model = NULL;
  const et_f32_tensor *value = NULL;
  return e3_zero_control_owner(pointer, index, &model, &value) == 0 &&
      model == e3_zero_control.model && index == e3_zero_control.index &&
      source == e3_zero_control.source && value == e3_zero_control.destination;
}

int64_t et_e3_zero_control_anchor_v1(void *pointer, int64_t index,
                                     void *source) {
  owner *model = NULL;
  const et_f32_tensor *value = NULL;
  if (e3_zero_control.model || e3_zero_control.borrow || e3_zero_control.pin ||
      e3_zero_control.fail_domain || !source ||
      e3_zero_control_owner(pointer, index, &model, &value) != 0 ||
      source == value ||
      et_i2_private_owned_clone_match_v1(
        (const et_f32_tensor *)source, value,
        ET_I2_PRIVATE_OWNED_VALUE_FINITE) != ET_I2_PRIVATE_OWNED_RESULT_OK)
    return -1;
  e3_zero_control.model = model;
  e3_zero_control.index = index;
  e3_zero_control.source = source;
  e3_zero_control.destination = value;
  return 0;
}

int64_t et_e3_zero_control_unanchor_v1(void *pointer, int64_t index,
                                       void *source) {
  if (!e3_zero_control_identical(pointer, index, source) ||
      e3_zero_control.borrow || e3_zero_control.pin ||
      e3_zero_control.fail_domain ||
      et_i2_private_owned_clone_match_v1(
        e3_zero_control.source, e3_zero_control.destination,
        ET_I2_PRIVATE_OWNED_VALUE_FINITE) != ET_I2_PRIVATE_OWNED_RESULT_OK)
    return -1;
  memset(&e3_zero_control, 0, sizeof(e3_zero_control));
  return 0;
}

int64_t et_e3_zero_control_borrow_begin_v1(void *pointer, int64_t index,
                                            void *source) {
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (!e3_zero_control_identical(pointer, index, source) ||
      e3_zero_control.borrow || e3_zero_control.pin ||
      et_i2_private_owned_clone_match_v1(
        e3_zero_control.source, e3_zero_control.destination,
        ET_I2_PRIVATE_OWNED_VALUE_FINITE) != ET_I2_PRIVATE_OWNED_RESULT_OK)
    return -1;
  if (et_f32_tensor_borrow_begin_v1(
        (et_f32_tensor *)e3_zero_control.destination,
        &e3_zero_control.borrow, &error) == 0) return 0;
  if (e3_zero_control.borrow) {
    et_f32_tensor_error_clear_v1(&error);
    (void)et_f32_tensor_borrow_end_v1(&e3_zero_control.borrow, &error);
  }
  return -1;
}

int64_t et_e3_zero_control_borrow_end_v1(void *pointer, int64_t index,
                                          void *source) {
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (!e3_zero_control_identical(pointer, index, source) ||
      !e3_zero_control.borrow || e3_zero_control.pin ||
      et_f32_tensor_borrow_end_v1(&e3_zero_control.borrow, &error) != 0)
    return -1;
  return et_i2_private_owned_clone_match_v1(
           e3_zero_control.source, e3_zero_control.destination,
           ET_I2_PRIVATE_OWNED_VALUE_FINITE) == ET_I2_PRIVATE_OWNED_RESULT_OK
             ? 0 : -1;
}

int64_t et_e3_zero_control_pin_begin_v1(void *pointer, int64_t index,
                                         void *source) {
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (!e3_zero_control_identical(pointer, index, source) ||
      e3_zero_control.borrow || e3_zero_control.pin ||
      et_i2_private_owned_clone_match_v1(
        e3_zero_control.source, e3_zero_control.destination,
        ET_I2_PRIVATE_OWNED_VALUE_FINITE) != ET_I2_PRIVATE_OWNED_RESULT_OK)
    return -1;
  const et_f32_tensor_copy_assignment_v1 assignment = {
    .struct_size = sizeof(assignment),
    .destination = (et_f32_tensor *)e3_zero_control.destination,
    .source = e3_zero_control.source,
  };
  if (et_f32_tensor_copy_plan_prepare_v1(
        1u, &assignment, &e3_zero_control.pin, &error) == 0) return 0;
  if (e3_zero_control.pin) {
    et_f32_tensor_error_clear_v1(&error);
    (void)et_f32_tensor_copy_plan_release_v1(&e3_zero_control.pin, &error);
  }
  return -1;
}

int64_t et_e3_zero_control_pin_end_v1(void *pointer, int64_t index,
                                       void *source) {
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (!e3_zero_control_identical(pointer, index, source) ||
      e3_zero_control.borrow || !e3_zero_control.pin ||
      et_f32_tensor_copy_plan_release_v1(&e3_zero_control.pin, &error) != 0)
    return -1;
  return et_i2_private_owned_clone_match_v1(
           e3_zero_control.source, e3_zero_control.destination,
           ET_I2_PRIVATE_OWNED_VALUE_FINITE) == ET_I2_PRIVATE_OWNED_RESULT_OK
             ? 0 : -1;
}

int64_t et_e3_zero_control_fail_after_v1(void *pointer, int64_t domain,
                                          int64_t allowed) {
  owner *model = admit(pointer, OWNER, 0);
  if (!model || model->r.state != 1 || e3_zero_control.fail_domain ||
      e3_zero_control.borrow || e3_zero_control.pin || allowed < 0 ||
      (domain != 1 && domain != 2)) return -1;
  if (e3_zero_control.model && e3_zero_control.model != model) return -1;
  e3_zero_control.fail_model = model;
  e3_zero_control.fail_domain = (int)domain;
  if (domain == 1) et_i2_test_fail_alloc_after_v1((size_t)allowed);
  else et_f32_tensor_test_fail_alloc_after_v1((size_t)allowed);
  return 0;
}

int64_t et_e3_zero_control_fail_reset_v1(void *pointer, int64_t domain) {
  owner *model = admit(pointer, OWNER, 0);
  if (!model || model->r.state != 1 ||
      model != e3_zero_control.fail_model ||
      domain != e3_zero_control.fail_domain) return -1;
  if (domain == 1) et_i2_test_reset_allocator_v1();
  else et_f32_tensor_test_reset_allocator_v1();
  e3_zero_control.fail_domain = 0;
  e3_zero_control.fail_model = NULL;
  return 0;
}

int64_t et_e3_zero_control_census_v1(void *pointer, int64_t field) {
  owner *model = admit(pointer, OWNER, 0);
  et_f32_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
  size_t copies = 0, resets = 0, decodes = 0, pins = 0, source_pins = 0;
  if (!model || model->r.state != 1 || field < 0 || field > 10 ||
      (e3_zero_control.model && e3_zero_control.model != model) ||
      (e3_zero_control.fail_model && e3_zero_control.fail_model != model))
    return -1;
  et_i2_test_live_builder_counts_v1(&copies, &resets, &decodes);
  et_f32_test_live_counts_snapshot_v1(&counts);
  if (e3_zero_control.destination &&
      et_f32_tensor_test_plan_pins_v1(e3_zero_control.destination,
                                       &pins) != 0) return -1;
  if (e3_zero_control.source &&
      et_f32_tensor_test_plan_pins_v1(e3_zero_control.source,
                                       &source_pins) != 0) return -1;
  const size_t values[] = {
    copies, resets, decodes, counts.borrows, counts.copy_plans,
    counts.owned_clones, pins, (size_t)(e3_zero_control.borrow != NULL),
    (size_t)(e3_zero_control.pin != NULL),
    et_f32_tensor_test_successful_allocations_v1(),
    source_pins,
  };
  return values[field] <= (size_t)INT64_MAX ? (int64_t)values[field] : -1;
}
#endif
