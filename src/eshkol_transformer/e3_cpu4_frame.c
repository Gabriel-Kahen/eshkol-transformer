/* Separate, noninstalled CPU4 implementation of the opaque E3 native spine.
 * Compile instead of e3_frame.c; never enroll a C4 lease as an M3T OWNER. */
#include "e3_cpu4_frame.h"
#include "g3c4_model_owner_internal.h"
#include "m3_call_pins.h"
#include "m3t_f32_scoped.h"
#include "eshkol_transformer/i64_tensor.h"
#include "eshkol_transformer/n3k_primitives_abi.h"
#include "eshkol_transformer/n2_primitives_abi.h"
#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/indexed_cross_entropy.h"
#include "eshkol_transformer/l3s_masked_objective_abi.h"
#include "eshkol_transformer/e3_evaluation_metrics_abi.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#if !defined(ET_CPU4_TRAINING_PRIVATE) || !defined(ET_G3C4_NATIVE_PINS_PRIVATE) || !defined(ET_G3R_CANDIDATE_RETIRE_PRIVATE)
#error "CPU4 E3 requires the exact lease, C4 pins and I2 release-preflight features"
#endif

enum { E3_OK, E3_INVALID_ARGUMENT, E3_INVALID_STATE, E3_SHAPE_MISMATCH,
       E3_UNSUPPORTED, E3_INTERNAL };
enum { E3_CODE_OK, E3_CODE_IDENTITY, E3_CODE_PHASE, E3_CODE_SHAPE,
       E3_CODE_PROFILE, E3_CODE_ALLOCATION, E3_CODE_CAPACITY,
       E3_CODE_REENTRANCY, E3_CODE_DESTINATION, E3_CODE_INVARIANT };
enum { E3_IDLE, E3_ACQUIRED, E3_STAGING, E3_FORWARD, E3_CE_READY,
       E3_REDUCED, E3_CORRECTED, E3_FINALIZED, E3_DRAINED };
enum { E3_RUNTIME_N3K, E3_RUNTIME_N2, E3_RUNTIME_A2, E3_RUNTIME_L2,
       E3_RUNTIME_L3S, E3_RUNTIME_METRICS };
typedef struct e3_cpu4_payload {
  et_f32_tensor *forward[21], *ce, *epsilon, *batch[4], *sum[2][3], *result[4];
  int64_t ids[4], targets[4], positions[4], counts[2][2], active, staged_counts[2];
  unsigned char keep[16], mask[4];
  et_kernel_runtime *runtime[6];
} e3_cpu4_payload;
typedef struct e3_cpu4_workspace {
  struct e3_cpu4_workspace *next;
  et_e3_frame_internal *frame;
  void *lease;
  e3_cpu4_payload *data;
  et_g3c4_model_pins_internal pins;
  int live, workspace_child, pin_child;
} e3_cpu4_workspace;
struct et_e3_frame_internal {
  struct et_e3_frame_internal *next;
  void *lease;
  et_f32_tensor *destination[4];
  e3_cpu4_workspace *work;
  uint32_t lifecycle, phase, next_role, stage_plane, sum_bank;
  uint32_t committed, cleanup_ready, published_valid;
  int64_t published_counts[2];
};
static et_e3_frame_internal *e3_registry;
static e3_cpu4_workspace *e3_workspaces;
static int64_t e3_last_domain, e3_last_category, e3_last_code;
static int64_t e3_raw_domain, e3_raw_category, e3_raw_code;
#ifdef ET_E3_CPU4_TESTING
static size_t e3_alloc_remaining = SIZE_MAX;
void et_e3_cpu4_test_fail_after(size_t count) { e3_alloc_remaining = count; }
static int e3_allocation_allowed(void) {
  if (!e3_alloc_remaining) return 0;
  if (e3_alloc_remaining != SIZE_MAX) --e3_alloc_remaining;
  return 1;
}
#else
static int e3_allocation_allowed(void) { return 1; }
#endif
static void *e3_allocate(size_t bytes) {
  return e3_allocation_allowed() ? calloc(1, bytes) : NULL;
}
static const uint64_t e3_shape_token[] = {1,4,4};
static const uint64_t e3_shape_head[] = {1,2,4,2};
static const uint64_t e3_shape_head_request[] = {1,4,2,2};
static const uint64_t e3_shape_wide[] = {1,4,8};
static const uint64_t e3_shape_logits[] = {1,4,256};
static const uint64_t e3_shape_pair[] = {1,4};
static const uint64_t e3_shape_attention_mask[] = {1,4,4};
static const uint64_t e3_shape_embed_token[] = {1,4,256,4};
static const uint64_t e3_shape_embed_position[] = {1,4,4,4};
static const uint64_t e3_shape_linear_4[] = {1,4,4,4};
static const uint64_t e3_shape_linear_up[] = {1,4,4,8};
static const uint64_t e3_shape_linear_down[] = {1,4,8,4};
static const uint64_t e3_shape_linear_vocab[] = {1,4,4,256};
static const uint64_t e3_shape_attention[] = {1,2,2,4,4,2};

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

/* Closed category/code pairs emitted by these existing native dependencies.
 * A known category with an unrelated known code is still an invariant failure. */
static int e3_dependency_valid(int64_t domain,int64_t category,int64_t code) {
  if (domain == 1) {
    static const int64_t pairs[][2]={
      {ET_KERNEL_ERROR_DEVICE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER},
      {ET_KERNEL_ERROR_DEVICE_MISMATCH,ET_KERNEL_CODE_INVALID_TEXT},
      {ET_KERNEL_ERROR_DTYPE_MISMATCH,ET_KERNEL_CODE_INVALID_TEXT},
      {ET_KERNEL_ERROR_DTYPE_MISMATCH,ET_KERNEL_CODE_PROVIDER_REJECTED},
      {ET_KERNEL_ERROR_INTERNAL,ET_KERNEL_CODE_ALLOCATION_FAILED},
      {ET_KERNEL_ERROR_INTERNAL,ET_KERNEL_CODE_INTEGER_OVERFLOW},
      {ET_KERNEL_ERROR_INTERNAL,ET_KERNEL_CODE_PROVIDER_REJECTED},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_BUFFER_TOO_SMALL},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_DUPLICATE_ENTRY},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INTEGER_OVERFLOW},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_BUFFER},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_ENUM},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_SHAPE},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_TEXT},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_NULL_ARGUMENT},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_PROVIDER_REJECTED},
      {ET_KERNEL_ERROR_NONCONTIGUOUS,ET_KERNEL_CODE_INVALID_BUFFER},
      {ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INTEGER_OVERFLOW},
      {ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER},
      {ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_SHAPE},
      {ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_PROVIDER_REJECTED},
      {ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED},
      {ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_PROVIDER_REJECTED},
      {ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_SYMBOL_MISSING},
      {ET_KERNEL_ERROR_VERSION_MISMATCH,ET_KERNEL_CODE_ABI_MAJOR_MISMATCH},
      {ET_KERNEL_ERROR_VERSION_MISMATCH,ET_KERNEL_CODE_INVALID_STRUCT_SIZE},
      {ET_KERNEL_ERROR_VERSION_MISMATCH,ET_KERNEL_CODE_PROVIDER_REJECTED},
      {ET_KERNEL_ERROR_VERSION_MISMATCH,ET_KERNEL_CODE_UNKNOWN_REQUIRED_FEATURE},
    };
    for (size_t i=0;i<sizeof(pairs)/sizeof(*pairs);++i)
      if (category==pairs[i][0] && code==pairs[i][1]) return 1;
    return 0;
  }
  if (domain == 2) {
    static const int64_t pairs[][2]={
      {ET_I64_TENSOR_ERROR_INTERNAL,ET_I64_TENSOR_CODE_ALLOCATION_FAILED},
      {ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,ET_I64_TENSOR_CODE_INVALID_BUFFER},
      {ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,ET_I64_TENSOR_CODE_NULL_ARGUMENT},
      {ET_I64_TENSOR_ERROR_INVALID_STATE,ET_I64_TENSOR_CODE_ACTIVE_BORROW},
      {ET_I64_TENSOR_ERROR_INVALID_STATE,ET_I64_TENSOR_CODE_INVALID_HANDLE},
      {ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,ET_I64_TENSOR_CODE_BUFFER_SIZE_MISMATCH},
      {ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,ET_I64_TENSOR_CODE_INTEGER_OVERFLOW},
      {ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,ET_I64_TENSOR_CODE_INVALID_SHAPE},
      {ET_I64_TENSOR_ERROR_VERSION_MISMATCH,ET_I64_TENSOR_CODE_ABI_VERSION_MISMATCH},
    };
    for (size_t i=0;i<sizeof(pairs)/sizeof(*pairs);++i)
      if (category==pairs[i][0] && code==pairs[i][1]) return 1;
    return 0;
  }
  if (domain == 3) {
    static const int64_t pairs[][2]={
      {ET_F32_TENSOR_ERROR_INTERNAL,ET_F32_TENSOR_CODE_ALLOCATION_FAILED},
      {ET_F32_TENSOR_ERROR_INTERNAL,ET_F32_TENSOR_CODE_PROVIDER_REJECTED},
      {ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,ET_F32_TENSOR_CODE_INTEGER_OVERFLOW},
      {ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,ET_F32_TENSOR_CODE_INVALID_BUFFER},
      {ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,ET_F32_TENSOR_CODE_INVALID_HANDLE},
      {ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,ET_F32_TENSOR_CODE_NULL_ARGUMENT},
      {ET_F32_TENSOR_ERROR_INVALID_STATE,ET_F32_TENSOR_CODE_ACTIVE_BORROW},
      {ET_F32_TENSOR_ERROR_INVALID_STATE,ET_F32_TENSOR_CODE_FLOAT_ENVIRONMENT},
      {ET_F32_TENSOR_ERROR_INVALID_STATE,ET_F32_TENSOR_CODE_INTEGER_OVERFLOW},
      {ET_F32_TENSOR_ERROR_INVALID_STATE,ET_F32_TENSOR_CODE_INVALID_HANDLE},
      {ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,ET_F32_TENSOR_CODE_BUFFER_SIZE_MISMATCH},
      {ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,ET_F32_TENSOR_CODE_INTEGER_OVERFLOW},
      {ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,ET_F32_TENSOR_CODE_INVALID_SHAPE},
      {ET_F32_TENSOR_ERROR_VERSION_MISMATCH,ET_F32_TENSOR_CODE_ABI_VERSION_MISMATCH},
      {ET_F32_TENSOR_ERROR_VERSION_MISMATCH,ET_F32_TENSOR_CODE_INVALID_BUFFER},
    };
    for (size_t i=0;i<sizeof(pairs)/sizeof(*pairs);++i)
      if (category==pairs[i][0] && code==pairs[i][1]) return 1;
    return 0;
  }
  return 0;
}

static int64_t e3_dependency(int64_t domain, int64_t category, int64_t code) {
  if (!e3_raw_category) { e3_raw_domain=domain; e3_raw_category=category; e3_raw_code=code; }
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

/* C4 error state is shared downstream: snapshot before any cleanup call. */
static int64_t e3_c4_error(void) {
  et_g3c4_error_state_internal raw = et_g3c4_error_snapshot_internal();
  if (!e3_raw_category) { e3_raw_domain=raw.domain ? raw.domain : 4;
    e3_raw_category=raw.category; e3_raw_code=raw.code; }
  if (raw.domain >= 1 && raw.domain <= 3)
    return e3_dependency(raw.domain, raw.category, raw.code);
  if (raw.domain != 0) return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
  const int64_t category = raw.category, code = raw.code;
  if (category == ET_G3C4_INVALID_ARGUMENT &&
      (code == ET_G3C4_CODE_IDENTITY || code == ET_G3C4_CODE_SELECTOR ||
       code == ET_G3C4_CODE_ALIAS || code == ET_G3C4_CODE_TOPOLOGY))
    return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_IDENTITY);
  if (category == ET_G3C4_INVALID_STATE &&
      (code == ET_G3C4_CODE_IDENTITY || code == ET_G3C4_CODE_LIFECYCLE ||
       code == ET_G3C4_CODE_STALE_BINDING || code == ET_G3C4_CODE_READINESS))
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  if (category == ET_G3C4_SHAPE_MISMATCH && code == ET_G3C4_CODE_SHAPE)
    return e3_fail(E3_SHAPE_MISMATCH, E3_CODE_SHAPE);
  if ((category == ET_G3C4_UNSUPPORTED && code == ET_G3C4_CODE_CAPABILITY) ||
      (category == ET_G3C4_DETERMINISM_UNAVAILABLE && code == ET_G3C4_CODE_FP_CONTROL))
    return e3_fail(E3_UNSUPPORTED, E3_CODE_PROFILE);
  if (category == ET_G3C4_INTERNAL && code == ET_G3C4_CODE_ALLOCATION)
    return e3_fail(E3_INTERNAL, E3_CODE_ALLOCATION);
  return e3_fail(E3_INTERNAL, E3_CODE_INVARIANT);
}
int64_t et_e3_private_last_error_domain_v1(void) { return e3_last_domain; }
int64_t et_e3_private_last_error_category_v1(void) { return e3_last_category; }
int64_t et_e3_private_last_error_code_v1(void) { return e3_last_code; }
#ifdef ET_E3_CPU4_TESTING
int64_t et_e3_cpu4_private_raw_error_ref_v1(int64_t selector) {
  return selector == 0 ? e3_raw_domain : selector == 1 ? e3_raw_category :
         selector == 2 ? e3_raw_code : -1;
}
#endif

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

static int e3_pins_idle(const et_g3c4_model_pins_internal *pins) {
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

static int64_t e3_scope_begin(et_f32_tensor *tensor,
                              et_f32_scoped_guard_internal *guard,
                              int destination) {
  et_f32_tensor_error error;
  int32_t rc = et_f32_tensor_scoped_begin_internal(tensor, guard, &error);
  if (!rc) return 0;
  if (!e3_raw_category) {
    e3_raw_domain=3; e3_raw_category=error.category; e3_raw_code=error.code;
  }
  if (destination &&
      error.category == ET_F32_TENSOR_ERROR_INVALID_STATE &&
      error.code == ET_F32_TENSOR_CODE_ACTIVE_BORROW)
    return e3_fail(E3_INVALID_STATE, E3_CODE_DESTINATION);
  return e3_f32_error(&error);
}

static void e3_scope_end(et_f32_scoped_guard_internal *guard) {
  if (et_f32_tensor_scoped_end_internal(guard) != 0) abort();
}

static et_e3_frame_internal *e3_admit(void *pointer) {
  et_e3_frame_internal *frame = e3_find(pointer);
  if (!frame) { e3_fail(E3_INVALID_ARGUMENT, E3_CODE_IDENTITY); return NULL; }
  if (frame->lifecycle != 1) { e3_fail(E3_INVALID_STATE, E3_CODE_PHASE); return NULL; }
  if (et_g3c4_private_training_lease_check_v1(frame->lease)) {
    (void)e3_c4_error(); return NULL;
  }
  if (frame->work) {
    e3_cpu4_workspace *work = e3_workspaces;
    while (work && work != frame->work) work = work->next;
    if (!work || !work->live || !work->workspace_child ||
        work->frame != frame || work->lease != frame->lease || !work->data) {
      e3_fail(E3_INTERNAL, E3_CODE_INVARIANT); return NULL;
    }
    /* The registered workspace prevents lease leave/reenter, hence its
     * acquisition ordinal cannot change during any admitted E3 stage. */
  } else if (frame->phase != E3_IDLE) {
    e3_fail(E3_INTERNAL, E3_CODE_INVARIANT); return NULL;
  }
  return frame;
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
  for (size_t i = 0; i < 21; ++i) tensors[count++] = frame->work->data->forward[i];
  tensors[count++] = frame->work->data->ce;
  tensors[count++] = frame->work->data->epsilon;
  for (size_t i = 0; i < 4; ++i) tensors[count++] = frame->work->data->batch[i];
  for (size_t bank = 0; bank < 2; ++bank)
    for (size_t i = 0; i < 3; ++i) tensors[count++] = frame->work->data->sum[bank][i];
  for (size_t i = 0; i < 4; ++i) tensors[count++] = frame->work->data->result[i];
  return count;
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
  if (!e3_allocation_allowed()) { e3_fail(E3_INTERNAL,E3_CODE_ALLOCATION); return NULL; }
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
  for (size_t i = 4; i > 0; --i) e3_tensor_destroy(&frame->work->data->result[i - 1]);
  for (size_t bank = 2; bank > 0; --bank)
    for (size_t i = 3; i > 0; --i)
      e3_tensor_destroy(&frame->work->data->sum[bank - 1][i - 1]);
  for (size_t i = 4; i > 0; --i) e3_tensor_destroy(&frame->work->data->batch[i - 1]);
  e3_tensor_destroy(&frame->work->data->epsilon);
  e3_tensor_destroy(&frame->work->data->ce);
  for (size_t i = 21; i > 0; --i) e3_tensor_destroy(&frame->work->data->forward[i - 1]);
}

static int64_t e3_create_owned(et_e3_frame_internal *frame) {
  for (size_t i = 0; i < 21; ++i) {
    const uint64_t *shape = e3_shape_token;
    size_t rank = 3;
    if (i >= 7 && i <= 10) { shape = e3_shape_head; rank = 4; }
    if (i == 15 || i == 16) shape = e3_shape_wide;
    if (i == 20) shape = e3_shape_logits;
    frame->work->data->forward[i] = e3_tensor_create(rank, shape);
    if (!frame->work->data->forward[i]) return e3_dependency_status(e3_last_domain,
                                                         e3_last_category);
  }
  frame->work->data->ce = e3_tensor_create(2, e3_shape_pair);
  frame->work->data->epsilon = e3_tensor_create(0, NULL);
  if (!frame->work->data->ce || !frame->work->data->epsilon)
    return e3_dependency_status(e3_last_domain, e3_last_category);
  for (size_t i = 0; i < 4; ++i) {
    frame->work->data->batch[i] = e3_tensor_create(0, NULL);
    if (!frame->work->data->batch[i]) return e3_dependency_status(e3_last_domain,
                                                       e3_last_category);
  }
  for (size_t bank = 0; bank < 2; ++bank)
    for (size_t i = 0; i < 3; ++i) {
      frame->work->data->sum[bank][i] = e3_tensor_create(0, NULL);
      if (!frame->work->data->sum[bank][i])
        return e3_dependency_status(e3_last_domain, e3_last_category);
    }
  for (size_t i = 0; i < 4; ++i) {
    frame->work->data->result[i] = e3_tensor_create(0, NULL);
    if (!frame->work->data->result[i]) return e3_dependency_status(e3_last_domain,
                                                        e3_last_category);
  }
  return 0;
}

static int64_t e3_roster(void *lease, et_f32_parameter *parameters[14],
                           const void *identities[14]) {
  for (int64_t i = 0; i < 14; ++i) {
    parameters[i] = et_g3c4_private_training_lease_parameter_v1(lease, i);
    if (!parameters[i]) return e3_c4_error();
    identities[i] = et_g3c4_private_training_lease_identity_v1(lease, i);
    if (!identities[i]) return e3_c4_error();
  }
  return 0;
}
static int64_t e3_destinations(et_e3_frame_internal *frame) {
  et_f32_parameter *parameters[14]; const void *identities[14];
  int64_t rc = e3_roster(frame->lease,parameters,identities);
  if (rc) return rc;
  for (size_t i = 0; i < 4; ++i) {
    if (!frame->destination[i] ||
        et_f32_tensor_canonical_owner_v1(frame->destination[i]) != frame->destination[i])
      return e3_fail(E3_INVALID_ARGUMENT,E3_CODE_DESTINATION);
    for (size_t j = 0; j < i; ++j)
      if (frame->destination[i] == frame->destination[j])
        return e3_fail(E3_INVALID_ARGUMENT,E3_CODE_DESTINATION);
    for (size_t j = 0; j < 14; ++j)
      if (frame->destination[i] == et_f32_parameter_canonical_owner_v1(parameters[j]))
        return e3_fail(E3_INVALID_ARGUMENT,E3_CODE_DESTINATION);
    if ((rc = e3_tensor_shape(frame->destination[i],0,NULL,1))) return rc;
  }
  return 0;
}
static int64_t e3_release_ready(et_e3_frame_internal *frame) {
  et_f32_tensor *owned[37]; et_f32_tensor_error error;
  (void)e3_all_tensors(frame,owned);
  for (size_t i = 0; i < 37; ++i)
    if (owned[i] && et_f32_tensor_private_destroy_ready_v1(owned[i],&error))
      return e3_f32_error(&error);
  return 0;
}
static int64_t e3_storage_check(et_e3_frame_internal *frame, int runtime) {
  int64_t rc=0;
  for (size_t i=0;i<4;++i)
    if ((rc=e3_tensor_shape(frame->destination[i],0,NULL,1))) return rc;
  if (!frame->work) return 0;
  for (size_t i = 0; i < 21; ++i) {
    const uint64_t *shape = e3_shape_token; size_t rank = 3;
    if (i >= 7 && i <= 10) { shape=e3_shape_head; rank=4; }
    if (i==15 || i==16) shape=e3_shape_wide;
    if (i==20) shape=e3_shape_logits;
    if ((rc=e3_tensor_shape(frame->work->data->forward[i],rank,shape,0))) return rc;
  }
  if ((rc=e3_tensor_shape(frame->work->data->ce,2,e3_shape_pair,0)) ||
      (rc=e3_tensor_shape(frame->work->data->epsilon,0,NULL,0))) return rc;
  for (size_t i=0;i<4;++i) {
    if ((rc=e3_tensor_shape(frame->work->data->batch[i],0,NULL,0)) ||
        (rc=e3_tensor_shape(frame->work->data->result[i],0,NULL,0))) return rc;
    if (frame->work->data->positions[i] != (int64_t)i)
      return e3_fail(E3_INTERNAL,E3_CODE_INVARIANT);
  }
  for (size_t i=0;i<16;++i) if (frame->work->data->keep[i]!=1)
    return e3_fail(E3_INTERNAL,E3_CODE_INVARIANT);
  for (size_t b=0;b<2;++b) for (size_t i=0;i<3;++i)
    if ((rc=e3_tensor_shape(frame->work->data->sum[b][i],0,NULL,0))) return rc;
  if (runtime) for (size_t i=0;i<6;++i)
    if (!frame->work->data->runtime[i]) return e3_fail(E3_INTERNAL,E3_CODE_INVARIANT);
  return 0;
}
/* Preflight every releasable object before changing child enrollment. */
static void e3_release_workspace(et_e3_frame_internal *frame) {
  e3_cpu4_workspace *work=frame->work;
  if (!work) return;
  et_f32_tensor_error error;
  if (!e3_pins_idle(&work->pins)) {
    if (et_g3c4_model_pins_check_internal(&work->pins,&error)) abort();
    et_g3c4_model_pins_end_internal(&work->pins);
  }
  if (work->pin_child) {
    if (et_g3c4_training_child_release_internal(frame->lease,&work->pins,5)) abort();
    work->pin_child=0;
  }
  if (work->data) {
    if (e3_release_ready(frame)) abort();
    e3_destroy_owned(frame);
    for (size_t i=6;i>0;--i)
      et_kernel_runtime_destroy(work->data->runtime[i-1]);
    memset(work->data,0,sizeof(*work->data)); free(work->data); work->data=NULL;
  }
  if (work->workspace_child) {
    if (et_g3c4_training_child_release_internal(frame->lease,work,2)) abort();
    work->workspace_child=0;
  }
  work->live=0; work->frame=NULL; work->lease=NULL;
  frame->work=NULL;
}
void *et_e3_private_cpu4_frame_create_v1(void *lease, et_f32_tensor *loss,
    et_f32_tensor *weight, et_f32_tensor *perplexity, et_f32_tensor *accuracy) {
  e3_clear(); e3_raw_domain=e3_raw_category=e3_raw_code=0;
  if (et_g3c4_private_training_lease_check_v1(lease)) { (void)e3_c4_error(); return NULL; }
  et_e3_frame_internal staged={0}; staged.lease=lease;
  staged.destination[0]=loss; staged.destination[1]=weight;
  staged.destination[2]=perplexity; staged.destination[3]=accuracy;
  if (e3_destinations(&staged)) return NULL;
  et_e3_frame_internal *frame=e3_allocate(sizeof(*frame));
  if (!frame) { e3_fail(E3_INTERNAL,E3_CODE_ALLOCATION); return NULL; }
  *frame=staged; frame->lifecycle=1; frame->next=e3_registry; e3_registry=frame;
  return frame;
}
int64_t et_e3_private_frame_check_v1(void *pointer) {
  e3_clear(); et_e3_frame_internal *frame=e3_admit(pointer);
  return !frame ? (e3_last_domain ? e3_dependency_status(e3_last_domain,e3_last_category) : e3_last_category) :
      e3_storage_check(frame,1);
}
int64_t et_e3_private_acquire_v1(void *pointer) {
  e3_clear(); e3_raw_domain=e3_raw_category=e3_raw_code=0; et_e3_frame_internal *frame=e3_admit(pointer);
  if (!frame) return e3_last_domain ? e3_dependency_status(e3_last_domain,e3_last_category) : e3_last_category;
  if (frame->phase!=E3_IDLE || frame->work) return e3_fail(E3_INVALID_STATE,E3_CODE_PHASE);
  if (e3_destinations(frame)) return e3_last_domain ? e3_dependency_status(e3_last_domain,e3_last_category) : e3_last_category;
  if (et_g3c4_private_training_lease_enter_v1(frame->lease,1)) return e3_c4_error();
  int64_t rc=0;
  e3_cpu4_workspace *work=e3_allocate(sizeof(*work));
  if (!work) { rc=e3_fail(E3_INTERNAL,E3_CODE_ALLOCATION); goto failed; }
  work->next=e3_workspaces; e3_workspaces=work; frame->work=work;
  work->frame=frame; work->lease=frame->lease; work->live=1;
  if (et_g3c4_training_child_acquire_internal(frame->lease,work,2)) { rc=e3_c4_error(); goto failed; }
  work->workspace_child=1;
  work->data=e3_allocate(sizeof(*work->data));
  if (!work->data) { rc=e3_fail(E3_INTERNAL,E3_CODE_ALLOCATION); goto failed; }
  if ((rc=e3_create_owned(frame))) goto failed;
  const et_kernel_provider_v1 *providers[6]={et_n3k_kernel_provider_v1(),et_n2_kernel_provider_v1(),
    et_a2_kernel_provider_v1(),et_l2_indexed_cross_entropy_provider_v1(),et_l3s_kernel_provider_v1(),et_e3_metrics_kernel_provider_v1()};
  for (size_t i=0;i<6;++i) if ((rc=e3_discover(&work->data->runtime[i],providers[i]))) goto failed;
  uint32_t epsilon=UINT32_C(0x3727c5ac); et_f32_tensor_error error;
  if (et_f32_tensor_copy_bits_from_v1(work->data->epsilon,&epsilon,1,&error)) { rc=e3_f32_error(&error); goto failed; }
  for (size_t i=0;i<4;++i) work->data->positions[i]=(int64_t)i;
  memset(work->data->keep,1,sizeof(work->data->keep));
  for (size_t b=0;b<2;++b) for (size_t i=0;i<3;++i) {
    uint32_t zero=0;
    if (et_f32_tensor_copy_bits_from_v1(work->data->sum[b][i],&zero,1,&error)) { rc=e3_f32_error(&error); goto failed; }
  }
  et_f32_parameter *parameters[14]; const void *identities[14];
  if ((rc=e3_roster(frame->lease,parameters,identities))) goto failed;
  if (et_g3c4_model_pins_begin_internal(parameters,identities,&work->pins,&error)) { rc=e3_f32_error(&error); goto failed; }
  if (et_g3c4_training_child_acquire_internal(frame->lease,&work->pins,5)) { rc=e3_c4_error(); goto failed; }
  work->pin_child=1;
  if ((rc=e3_storage_check(frame,1))) goto failed;
  frame->phase=E3_ACQUIRED; frame->next_role=frame->stage_plane=frame->sum_bank=frame->cleanup_ready=0;
  frame->committed=0;
  return 0;
failed: {
    const int64_t saved[6]={e3_last_domain,e3_last_category,e3_last_code,e3_raw_domain,e3_raw_category,e3_raw_code};
    e3_release_workspace(frame);
    if (et_g3c4_private_training_lease_leave_v1(frame->lease)) abort();
    e3_last_domain=saved[0];e3_last_category=saved[1];e3_last_code=saved[2];
    e3_raw_domain=saved[3];e3_raw_category=saved[4];e3_raw_code=saved[5];
    return rc;
  }
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
    if (e3_overlap(pointer, bytes, frame->work->pins.views[i].data,
                   frame->work->pins.views[i].byte_length) ||
        e3_overlap(pointer, bytes, frame->work->pins.views[i].shape,
                   frame->work->pins.views[i].rank *
                       sizeof(*frame->work->pins.views[i].shape)))
      return e3_fail(E3_INVALID_ARGUMENT, E3_CODE_DESTINATION);
  if (e3_overlap(pointer,bytes,frame->work,sizeof(*frame->work)) ||
      e3_overlap(pointer,bytes,frame->work->data,sizeof(*frame->work->data)))
    return e3_fail(E3_INVALID_ARGUMENT,E3_CODE_DESTINATION);
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
  if (copy.shape[0] != 1 || copy.shape[1] != 4)
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
  int64_t rc = e3_stage_view(frame, view, "i64", 4 * sizeof(int64_t),
                             _Alignof(int64_t));
  if (rc) return rc;
  const int64_t *ids = view->data;
  for (size_t i = 0; i < 4; ++i)
    if (ids[i] < 0 || ids[i] >= 256)
      return e3_fail(E3_SHAPE_MISMATCH, E3_CODE_SHAPE);
  memcpy(frame->work->data->ids, ids, sizeof(frame->work->data->ids));
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
  int64_t rc = e3_stage_view(frame, view, "i64", 4 * sizeof(int64_t),
                             _Alignof(int64_t));
  if (rc) return rc;
  const int64_t *targets = view->data;
  for (size_t i = 0; i < 4; ++i)
    if (targets[i] < 0 || targets[i] >= 256)
      return e3_fail(E3_SHAPE_MISMATCH, E3_CODE_SHAPE);
  memcpy(frame->work->data->targets, targets, sizeof(frame->work->data->targets));
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
  int64_t rc = e3_stage_view(frame, view, "bool", 4, 1);
  if (rc) return rc;
  const unsigned char *mask = view->data;
  unsigned active=0;
  for (size_t i=0;i<4;++i) { if (mask[i]>1) return e3_fail(E3_INVALID_ARGUMENT,E3_CODE_PROFILE); active+=mask[i]; }
  if (!active) return e3_fail(E3_INVALID_ARGUMENT,E3_CODE_PROFILE);
  memcpy(frame->work->data->mask, mask, sizeof(frame->work->data->mask));
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
    if (et_kernel_runtime_dispatch(frame->work->data->runtime[runtime], &call, &error) != 0)
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
      frame->work->data->ids, sizeof(frame->work->data->ids), "i64", 2, e3_shape_pair);
  et_kernel_tensor_view_v1 positions = e3_native_view(
      frame->work->data->positions, sizeof(frame->work->data->positions), "i64", 2, e3_shape_pair);
  et_kernel_tensor_view_v1 keep = e3_native_view(
      frame->work->data->keep, sizeof(frame->work->data->keep), "bool", 3, e3_shape_attention_mask);
  e3_operand in[6];
  et_f32_tensor *out[] = {frame->work->data->forward[role]};
  const char *capability = NULL, *operation = NULL;
  const uint64_t *shape = NULL;
  size_t rank = 0, count = 0, runtime = E3_RUNTIME_N3K;
#define TENSOR_INPUT(value) in[count++] = e3_tensor_operand(value)
#define VIEW_INPUT(value) in[count++] = e3_view_operand(value)
#define PIN_INPUT(index) in[count++] = e3_view_operand(&frame->work->pins.views[index])
  switch (role) {
    case 0:
      capability = "n3k.embedding-forward"; operation = "n3k.embedding.forward";
      shape = e3_shape_embed_token; rank = 4; VIEW_INPUT(&ids); PIN_INPUT(10); break;
    case 1:
      capability = "n3k.embedding-forward"; operation = "n3k.embedding.forward";
      shape = e3_shape_embed_position; rank = 4; VIEW_INPUT(&positions); PIN_INPUT(13); break;
    case 2:
      capability = "n3k.residual"; operation = "n3k.residual.forward";
      shape = e3_shape_token; rank = 3; TENSOR_INPUT(frame->work->data->forward[0]);
      TENSOR_INPUT(frame->work->data->forward[1]); break;
    case 3:
      runtime = E3_RUNTIME_N2; capability = "kernel.norm";
      operation = "layer-norm.forward"; shape = e3_shape_token; rank = 3;
      TENSOR_INPUT(frame->work->data->forward[2]); PIN_INPUT(7); PIN_INPUT(6);
      TENSOR_INPUT(frame->work->data->epsilon); break;
    case 4: case 5: case 6: {
      static const size_t pin[] = {2, 0, 3};
      capability = "n3k.linear"; operation = "n3k.linear.forward-no-bias";
      shape = e3_shape_linear_4; rank = 4; TENSOR_INPUT(frame->work->data->forward[3]);
      PIN_INPUT(pin[role - 4]); break;
    }
    case 7: case 8: case 9:
      capability = "n3k.head-layout"; operation = "n3k.heads.split.forward";
      shape = e3_shape_head_request; rank = 4; TENSOR_INPUT(frame->work->data->forward[role - 3]); break;
    case 10:
      runtime = E3_RUNTIME_A2; capability = "kernel.causal-attention";
      operation = "causal-attention.forward"; shape = e3_shape_attention; rank = 6;
      TENSOR_INPUT(frame->work->data->forward[7]); TENSOR_INPUT(frame->work->data->forward[8]);
      TENSOR_INPUT(frame->work->data->forward[9]); VIEW_INPUT(&positions);
      VIEW_INPUT(&positions); VIEW_INPUT(&keep); break;
    case 11:
      capability = "n3k.head-layout"; operation = "n3k.heads.merge.forward";
      shape = e3_shape_head_request; rank = 4; TENSOR_INPUT(frame->work->data->forward[10]); break;
    case 12:
      capability = "n3k.linear"; operation = "n3k.linear.forward-no-bias";
      shape = e3_shape_linear_4; rank = 4; TENSOR_INPUT(frame->work->data->forward[11]);
      PIN_INPUT(1); break;
    case 13:
      capability = "n3k.residual"; operation = "n3k.residual.forward";
      shape = e3_shape_token; rank = 3; TENSOR_INPUT(frame->work->data->forward[2]);
      TENSOR_INPUT(frame->work->data->forward[12]); break;
    case 14:
      runtime = E3_RUNTIME_N2; capability = "kernel.norm";
      operation = "layer-norm.forward"; shape = e3_shape_token; rank = 3;
      TENSOR_INPUT(frame->work->data->forward[13]); PIN_INPUT(9); PIN_INPUT(8);
      TENSOR_INPUT(frame->work->data->epsilon); break;
    case 15:
      capability = "n3k.linear"; operation = "n3k.linear.forward-no-bias";
      shape = e3_shape_linear_up; rank = 4; TENSOR_INPUT(frame->work->data->forward[14]);
      PIN_INPUT(5); break;
    case 16:
      capability = "n3k.gelu"; operation = "n3k.gelu.forward";
      shape = e3_shape_wide; rank = 3; TENSOR_INPUT(frame->work->data->forward[15]); break;
    case 17:
      capability = "n3k.linear"; operation = "n3k.linear.forward-no-bias";
      shape = e3_shape_linear_down; rank = 4; TENSOR_INPUT(frame->work->data->forward[16]);
      PIN_INPUT(4); break;
    case 18:
      capability = "n3k.residual"; operation = "n3k.residual.forward";
      shape = e3_shape_token; rank = 3; TENSOR_INPUT(frame->work->data->forward[13]);
      TENSOR_INPUT(frame->work->data->forward[17]); break;
    case 19:
      runtime = E3_RUNTIME_N2; capability = "kernel.norm";
      operation = "layer-norm.forward"; shape = e3_shape_token; rank = 3;
      TENSOR_INPUT(frame->work->data->forward[18]); PIN_INPUT(12); PIN_INPUT(11);
      TENSOR_INPUT(frame->work->data->epsilon); break;
    case 20:
      capability = "n3k.linear"; operation = "n3k.linear.forward-no-bias";
      shape = e3_shape_linear_vocab; rank = 4; TENSOR_INPUT(frame->work->data->forward[19]);
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
      frame->work->data->targets, sizeof(frame->work->data->targets), "i64", 2, e3_shape_pair);
  e3_operand inputs[] = {e3_tensor_operand(frame->work->data->forward[20]),
                         e3_view_operand(&targets)};
  et_f32_tensor *outputs[] = {frame->work->data->ce};
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
      frame->work->data->mask, sizeof(frame->work->data->mask), "bool", 2, e3_shape_pair);
  e3_operand inputs[] = {e3_tensor_operand(frame->work->data->ce), e3_view_operand(&mask)};
  et_f32_tensor *outputs[] = {frame->work->data->batch[0], frame->work->data->batch[1], frame->work->data->batch[2]};
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
      frame->work->data->targets, sizeof(frame->work->data->targets), "i64", 2, e3_shape_pair);
  et_kernel_tensor_view_v1 mask = e3_native_view(
      frame->work->data->mask, sizeof(frame->work->data->mask), "bool", 2, e3_shape_pair);
  et_kernel_tensor_view_v1 active = e3_native_view(
      &frame->work->data->active, sizeof(frame->work->data->active), "i64", 0, NULL);
  et_f32_tensor *float_output[] = {frame->work->data->batch[3]};
  et_f32_scoped_guard_internal output_guard = {0};
  int64_t rc = e3_scope_begin(float_output[0], &output_guard, 0);
  if (!rc) {
    et_kernel_tensor_view_v1 outputs[] = {output_guard.view, active};
    et_kernel_tensor_view_v1 input_views[3];
    et_f32_scoped_guard_internal input_guard = {0};
    rc = e3_scope_begin(frame->work->data->forward[20], &input_guard, 0);
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
      if (et_kernel_runtime_dispatch(frame->work->data->runtime[E3_RUNTIME_METRICS],
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
    rc = e3_scope_begin(frame->work->data->sum[current][i], &guards[opened], 0);
    if (!rc) inputs[i] = guards[opened++].view;
  }
  inputs[3] = e3_native_view(&frame->work->data->counts[current][0], sizeof(int64_t),
                             "i64", 0, NULL);
  inputs[4] = e3_native_view(&frame->work->data->counts[current][1], sizeof(int64_t),
                             "i64", 0, NULL);
  for (size_t i = 0; i < 3 && !rc; ++i) {
    const size_t batch_index = i < 2 ? i : 3;
    rc = e3_scope_begin(frame->work->data->batch[batch_index], &guards[opened], 0);
    if (!rc) inputs[5 + i] = guards[opened++].view;
  }
  inputs[8] = e3_native_view(&frame->work->data->active, sizeof(frame->work->data->active),
                             "i64", 0, NULL);
  for (size_t i = 0; i < 3 && !rc; ++i) {
    rc = e3_scope_begin(frame->work->data->sum[next][i], &guards[opened], 0);
    if (!rc) outputs[i] = guards[opened++].view;
  }
  outputs[3] = e3_native_view(&frame->work->data->counts[next][0], sizeof(int64_t),
                              "i64", 0, NULL);
  outputs[4] = e3_native_view(&frame->work->data->counts[next][1], sizeof(int64_t),
                              "i64", 0, NULL);
  if (!rc) {
    et_kernel_request_v1 request = {sizeof(request),
      "e3.evaluation-metrics.accumulate", "f32", "cpu", 3,
      e3_shape_logits, 1, {0}};
    et_kernel_call_v1 call = {sizeof(call), "e3.evaluation-metrics", &request,
      9, sizeof(inputs[0]), sizeof(inputs), inputs,
      5, sizeof(outputs[0]), sizeof(outputs), outputs};
    et_kernel_error error;
    if (et_kernel_runtime_dispatch(frame->work->data->runtime[E3_RUNTIME_METRICS],
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
  if (frame->work->data->counts[bank][1] <= 0 || frame->work->data->counts[bank][0] <= 0)
    return e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
  e3_operand inputs[] = {e3_tensor_operand(frame->work->data->sum[bank][0]),
                         e3_tensor_operand(frame->work->data->sum[bank][1]),
                         e3_tensor_operand(frame->work->data->sum[bank][2])};
  et_f32_tensor *outputs[] = {frame->work->data->result[0], frame->work->data->result[1],
                              frame->work->data->result[2], frame->work->data->result[3]};
  int64_t rc = e3_dispatch(frame, E3_RUNTIME_METRICS,
      "e3.evaluation-metrics", "e3.evaluation-metrics.finalize",
      3, e3_shape_logits, inputs, 3, outputs, 4);
  if (!rc) {
    frame->work->data->staged_counts[0] = frame->work->data->counts[bank][0];
    frame->work->data->staged_counts[1] = frame->work->data->counts[bank][1];
    frame->committed = 1;
    frame->phase = E3_FINALIZED;
  }
  return rc;
}

int64_t et_e3_private_cleanup_preflight_v1(void *pointer) {
  e3_clear(); et_e3_frame_internal *frame=e3_admit(pointer);
  if (!frame) return e3_last_domain ? e3_dependency_status(e3_last_domain,e3_last_category) : e3_last_category;
  if (frame->phase<E3_ACQUIRED || frame->phase>E3_FINALIZED || frame->cleanup_ready)
    return e3_fail(E3_INVALID_STATE,E3_CODE_PHASE);
  if ((frame->phase==E3_FINALIZED)!=(frame->committed==1))
    return e3_fail(E3_INTERNAL,E3_CODE_INVARIANT);
  et_f32_tensor_error error;
  if (et_g3c4_model_pins_check_internal(&frame->work->pins,&error)) return e3_f32_error(&error);
  int64_t rc=e3_storage_check(frame,1);
  if (!rc) rc=e3_release_ready(frame);
  if (!rc) frame->cleanup_ready=1;
  return rc;
}
void et_e3_private_drain_v1(void *pointer) {
  et_e3_frame_internal *frame=e3_admit(pointer);
  if (!frame || !frame->cleanup_ready || !frame->work->pin_child ||
      frame->phase<E3_ACQUIRED || frame->phase>E3_FINALIZED) abort();
  et_g3c4_model_pins_end_internal(&frame->work->pins);
  if (et_g3c4_training_child_release_internal(frame->lease,&frame->work->pins,5)) abort();
  frame->work->pin_child=0; frame->phase=E3_DRAINED;
}

static int64_t e3_publication_counts(et_e3_frame_internal *frame) {
  const int64_t tokens = frame->work->data->staged_counts[0];
  const int64_t batches = frame->work->data->staged_counts[1];
  if (tokens < 1 || tokens > 32768 || batches < 1 || batches > 8192 ||
      batches > tokens || tokens - batches > 3 * batches)
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
  if (!e3_pins_idle(&frame->work->pins))
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
    rc = e3_scope_begin(frame->work->data->result[i], &results[i], 0);
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
          results[i].owner != frame->work->data->result[i] || results[i].view.rank != 0 ||
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
  frame->published_counts[0] = frame->work->data->staged_counts[0];
  frame->published_counts[1] = frame->work->data->staged_counts[1];
  frame->published_valid = 1;
  frame->committed = 2;
  while (destination_count) e3_scope_end(&destinations[--destination_count]);
  return 0;
}

void et_e3_private_finish_v1(void *pointer) {
  et_e3_frame_internal *frame=e3_admit(pointer);
  if (!frame || frame->phase!=E3_DRAINED || !frame->cleanup_ready ||
      !e3_pins_idle(&frame->work->pins)) abort();
  e3_release_workspace(frame);
  if (et_g3c4_private_training_lease_leave_v1(frame->lease)) abort();
  frame->phase=E3_IDLE;frame->next_role=frame->stage_plane=frame->sum_bank=frame->cleanup_ready=0;
  if (frame->committed!=2) frame->committed=0;
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
  /* Transaction-local committed may be zero after rollback; the last
   * publication remains valid until another atomic publication succeeds. */
  if (frame->phase != E3_IDLE || !frame->published_valid) {
    e3_fail(E3_INVALID_STATE, E3_CODE_PHASE);
    return -1;
  }
  if (frame->work || frame->cleanup_ready || frame->next_role || frame->stage_plane || frame->sum_bank)
    return e3_fail(E3_INTERNAL,E3_CODE_INVARIANT);
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

/* Idle stale destruction is safe: no lease dereference is needed when no
 * workspace survives. The frame never owns or releases the lease/destinations. */
int64_t et_e3_private_frame_destroy_preflight_v1(void *pointer) {
  e3_clear(); et_e3_frame_internal *frame=e3_find(pointer);
  if (!frame) return e3_fail(E3_INVALID_ARGUMENT,E3_CODE_IDENTITY);
  if (frame->lifecycle==2) return 0;
  if (frame->phase!=E3_IDLE || frame->work) return e3_fail(E3_INVALID_STATE,E3_CODE_PHASE);
  return 0;
}
void et_e3_private_frame_destroy_v1(void *pointer) {
  if (et_e3_private_frame_destroy_preflight_v1(pointer)) abort();
  et_e3_frame_internal *frame=e3_find(pointer);
  frame->lifecycle=2;frame->lease=NULL;memset(frame->destination,0,sizeof(frame->destination));
  memset(frame->published_counts,0,sizeof(frame->published_counts));frame->published_valid=frame->committed=0;
}
