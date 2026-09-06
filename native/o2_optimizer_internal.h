#ifndef ESHKOL_TRANSFORMER_O2_OPTIMIZER_INTERNAL_H
#define ESHKOL_TRANSFORMER_O2_OPTIMIZER_INTERNAL_H

#include "f32_parameter_internal.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_O2_PROVIDER_ABI_MAJOR 2u
#define ET_O2_PROVIDER_ABI_MINOR 0u
#define ET_O2_PROVIDER_ID "i2-dense-cpu-f32-v1"
#define ET_O2_MAX_PARAMETERS ET_F32_PARAMETER_MAX_BATCH

enum {
  ET_O2_STATUS_OK = 0,
  ET_O2_STATUS_INVALID_ARGUMENT = 1,
  ET_O2_STATUS_SHAPE_MISMATCH = 2,
  ET_O2_STATUS_DTYPE_MISMATCH = 3,
  ET_O2_STATUS_DEVICE_MISMATCH = 4,
  ET_O2_STATUS_NONCONTIGUOUS = 5,
  ET_O2_STATUS_UNSUPPORTED = 6,
  ET_O2_STATUS_INVALID_STATE = 7,
  ET_O2_STATUS_VERSION_MISMATCH = 8,
  ET_O2_STATUS_INTERNAL = 9
};

enum {
  ET_O2_CODE_NONE = 0,
  ET_O2_CODE_NULL_ARGUMENT = 1,
  ET_O2_CODE_INVALID_HANDLE = 2,
  ET_O2_CODE_INVALID_OPTION = 3,
  ET_O2_CODE_DUPLICATE_PARAMETER = 4,
  ET_O2_CODE_MISSING_PARAMETER = 5,
  ET_O2_CODE_ALLOCATION_FAILED = 6,
  ET_O2_CODE_GRADIENT_ABSENT = 7,
  ET_O2_CODE_GRADIENT_METADATA = 8,
  ET_O2_CODE_NONFINITE = 9,
  ET_O2_CODE_COUNTER_OVERFLOW = 10,
  ET_O2_CODE_ACTIVE_BORROW = 11,
  ET_O2_CODE_CONSUMED = 12,
  ET_O2_CODE_PROVIDER_MISMATCH = 13,
  ET_O2_CODE_FLOAT_ENVIRONMENT = 14,
  ET_O2_CODE_PROVIDER_DEFECT = 15
};

enum { ET_O2_CLIP_NONE = 0, ET_O2_CLIP_GLOBAL_L2 = 1 };

enum { ET_O2_SCHEDULE_CONSTANT = 0, ET_O2_SCHEDULE_LINEAR = 1 };

enum {
  ET_O2_OPTION_LEARNING_RATE = 0,
  ET_O2_OPTION_BETA1 = 1,
  ET_O2_OPTION_BETA2 = 2,
  ET_O2_OPTION_EPSILON = 3,
  ET_O2_OPTION_WEIGHT_DECAY = 4
};

enum { ET_O2_MOMENT_EXP_AVG = 0, ET_O2_MOMENT_EXP_AVG_SQ = 1 };

enum {
  ET_O2_OPTIMIZER_STATE_LIVE = 1,
  ET_O2_OPTIMIZER_STATE_RELEASING = 2,
  ET_O2_OPTIMIZER_STATE_DEAD = 3
};

typedef struct et_o2_optimizer_builder et_o2_optimizer_builder;
typedef struct et_o2_optimizer et_o2_optimizer;
typedef struct et_o2_optimizer_state et_o2_optimizer_state;
typedef struct et_o2_optimizer_state_handle et_o2_optimizer_state_handle;
typedef struct et_o2_optimizer_state_borrow et_o2_optimizer_state_borrow;

typedef struct et_o2_optimizer_error {
  uint32_t category;
  uint32_t code;
  char operation[64];
  char message[192];
} et_o2_optimizer_error;

typedef et_o2_optimizer_error et_o2_error_v1;

#if defined(__cplusplus)
static_assert(sizeof(et_o2_optimizer_error) == 264u,
              "O2 error record layout changed");
#else
_Static_assert(sizeof(et_o2_optimizer_error) == 264u,
               "O2 error record layout changed");
#endif

int32_t et_o2_optimizer_builder_create_v1(
    size_t parameter_count, uint32_t clip_kind, uint32_t clip_max_bits,
    uint32_t schedule_kind, uint64_t warmup_updates, uint64_t total_updates,
    uint32_t minimum_ratio_bits, et_o2_optimizer_builder **builder,
    et_o2_error_v1 *error);
int32_t et_o2_optimizer_builder_set_v1(
    et_o2_optimizer_builder *builder, size_t index, et_f32_parameter *parameter,
    const void *p1_handle, uint32_t learning_rate_bits, uint32_t beta1_bits,
    uint32_t beta2_bits, uint32_t epsilon_bits, uint32_t weight_decay_bits,
    et_o2_error_v1 *error);
int32_t et_o2_optimizer_builder_finish_v1(et_o2_optimizer_builder **builder,
                                          et_o2_optimizer **optimizer,
                                          et_o2_error_v1 *error);
int32_t et_o2_optimizer_builder_abort_v1(et_o2_optimizer_builder **builder,
                                         et_o2_error_v1 *error);

int32_t et_o2_optimizer_step_v1(et_o2_optimizer *optimizer,
                                et_o2_error_v1 *error);
int32_t et_o2_optimizer_zero_grad_v1(et_o2_optimizer *optimizer,
                                     et_o2_error_v1 *error);
int32_t et_o2_optimizer_completed_updates_v1(const et_o2_optimizer *optimizer,
                                             uint64_t *completed_updates,
                                             et_o2_error_v1 *error);
int32_t
et_o2_optimizer_schedule_factor_bits_v1(const et_o2_optimizer *optimizer,
                                        uint32_t *factor_bits,
                                        et_o2_error_v1 *error);

int32_t et_o2_optimizer_state_snapshot_v1(et_o2_optimizer *optimizer,
                                          et_o2_optimizer_state **state,
                                          et_o2_error_v1 *error);
int32_t et_o2_optimizer_load_state_v1(et_o2_optimizer *optimizer,
                                      et_o2_optimizer_state *state,
                                      et_o2_error_v1 *error);
int32_t et_o2_optimizer_state_entry_count_v1(const et_o2_optimizer_state *state,
                                             size_t *entry_count,
                                             et_o2_error_v1 *error);
int32_t
et_o2_optimizer_state_completed_updates_v1(const et_o2_optimizer_state *state,
                                           uint64_t *completed_updates,
                                           et_o2_error_v1 *error);
int32_t et_o2_optimizer_state_option_bits_v1(const et_o2_optimizer_state *state,
                                             size_t index, uint32_t option,
                                             uint32_t *bits,
                                             et_o2_error_v1 *error);
int32_t et_o2_optimizer_state_moment_handle_v1(
    et_o2_optimizer_state *state, size_t index, uint32_t moment_kind,
    et_o2_optimizer_state_handle **handle, et_o2_error_v1 *error);
int32_t et_o2_optimizer_state_borrow_begin_v1(
    et_o2_optimizer_state *state, et_o2_optimizer_state_handle *handle,
    et_o2_optimizer_state_borrow **borrow,
    const et_kernel_tensor_view_v1 **view, et_o2_error_v1 *error);
int32_t
et_o2_optimizer_state_borrow_end_v1(et_o2_optimizer_state_borrow **borrow,
                                    et_o2_error_v1 *error);
/* Unlike all other non-release state accessors, this fixed lifecycle probe
 * succeeds for the exact registered dead owner so a trusted wrapper can
 * publish DEAD after a post-admission release defect. */
int32_t et_o2_optimizer_state_lifecycle_v1(const et_o2_optimizer_state *state,
                                           uint32_t *lifecycle,
                                           et_o2_error_v1 *error);
int32_t et_o2_optimizer_state_release_v1(et_o2_optimizer_state *state,
                                         et_o2_error_v1 *error);

#ifdef ET_O2_TESTING
typedef struct et_o2_test_live_counts_v1 {
  size_t struct_size;
  size_t builders;
  size_t optimizers;
  size_t live_states;
  size_t dead_states;
  size_t live_state_handles;
  size_t dead_state_handles;
  size_t state_borrows;
  size_t owned_state_clones;
} et_o2_test_live_counts_v1;

void et_o2_test_fail_alloc_after_v1(size_t successful_allocations);
void et_o2_test_fail_release_after_v1(size_t successful_releases);
void et_o2_test_reset_failpoints_v1(void);
void et_o2_test_live_counts_snapshot_v1(et_o2_test_live_counts_v1 *counts);
int32_t
et_o2_test_optimizer_set_completed_updates_v1(et_o2_optimizer *optimizer,
                                              uint64_t completed_updates);
int32_t et_o2_test_state_set_provider_version_v1(et_o2_optimizer_state *state,
                                                 uint32_t major,
                                                 uint32_t minor);
int32_t et_o2_test_state_set_owned_clone_count_v1(et_o2_optimizer_state *state,
                                                  uint64_t owned_clone_count);
int32_t et_o2_test_optimizer_moment_bits_v1(const et_o2_optimizer *optimizer,
                                            size_t index, uint32_t moment_kind,
                                            uint32_t *bits, size_t count,
                                            et_o2_error_v1 *error);
#endif

#ifdef __cplusplus
}
#endif

#endif
