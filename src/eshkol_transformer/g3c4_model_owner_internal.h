#ifndef ET_G3C4_MODEL_OWNER_INTERNAL_H
#define ET_G3C4_MODEL_OWNER_INTERNAL_H

#include "../../native/f32_parameter_internal.h"

#include <stdint.h>

/* Source-private ABI. All owner calls require external serialization. */
enum {
  ET_G3C4_DOMAIN_C4 = 0,
  ET_G3C4_DOMAIN_K1 = 1,
  ET_G3C4_DOMAIN_I1 = 2,
  ET_G3C4_DOMAIN_I2 = 3
};

enum {
  ET_G3C4_OK = 0,
  ET_G3C4_INVALID_ARGUMENT = 1,
  ET_G3C4_INVALID_STATE = 2,
  ET_G3C4_SHAPE_MISMATCH = 3,
  ET_G3C4_UNSUPPORTED = 4,
  ET_G3C4_INTERNAL = 5,
  ET_G3C4_DETERMINISM_UNAVAILABLE = 6
};

enum {
  ET_G3C4_CODE_OK = 0,
  ET_G3C4_CODE_IDENTITY = 1,
  ET_G3C4_CODE_LIFECYCLE = 2,
  ET_G3C4_CODE_SELECTOR = 3,
  ET_G3C4_CODE_SHAPE = 4,
  ET_G3C4_CODE_ALIAS = 5,
  ET_G3C4_CODE_TOPOLOGY = 6,
  ET_G3C4_CODE_ALLOCATION = 7,
  ET_G3C4_CODE_CAPABILITY = 8,
  ET_G3C4_CODE_STALE_BINDING = 9,
  ET_G3C4_CODE_INVARIANT = 10,
  ET_G3C4_CODE_FP_CONTROL = 11,
  ET_G3C4_CODE_REQUIRED_DRAW_EXHAUSTION = 12,
  ET_G3C4_CODE_READINESS = 13
};

typedef struct et_g3c4_error_state_internal {
  int64_t domain;
  int64_t category;
  int64_t code;
} et_g3c4_error_state_internal;

void et_g3c4_error_reset_internal(void);
void et_g3c4_error_set_internal(
    int64_t domain, int64_t category, int64_t code);
et_g3c4_error_state_internal et_g3c4_error_snapshot_internal(void);
void et_g3c4_error_restore_internal(et_g3c4_error_state_internal snapshot);

void *et_g3c4_private_model_owner_create_seeded_v1(int64_t seed);
void *et_g3c4_private_model_owner_parameter_v1(
    void *owner, int64_t index);
int64_t et_g3c4_private_model_owner_bind_v1(
    void *owner, int64_t index, void *exact_handle);
int64_t et_g3c4_private_model_owner_initialize_v1(
    void *owner, int64_t index);
int64_t et_g3c4_private_model_owner_word_v1(
    void *owner, int64_t index);
int64_t et_g3c4_private_model_owner_prepare_seal_v1(void *owner);
int64_t et_g3c4_private_model_owner_commit_seal_v1(void *owner);
int64_t et_g3c4_private_model_owner_abort_v1(void *owner);

int32_t et_g3c4_construction_parameter_preflight_internal(
    const void *exact_owner, void *parameter,
    const void *exact_handle, et_f32_tensor_error *error);

int64_t et_g3c4_private_last_error_domain_v1(void);
int64_t et_g3c4_private_last_error_category_v1(void);
int64_t et_g3c4_private_last_error_code_v1(void);

#endif
