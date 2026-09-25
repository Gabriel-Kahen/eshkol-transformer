#ifndef ESHKOL_TRANSFORMER_TR3_C_I2_RESTORE_INTERNAL_H
#define ESHKOL_TRANSFORMER_TR3_C_I2_RESTORE_INTERNAL_H

#include "f32_parameter_internal.h"

#include <stddef.h>
#include <stdint.h>

#ifndef ET_I2_PRIVATE_OWNED_CLONE_MATCH
#error "TR3-C restore requires the shared private owned-clone authorizer"
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
  ET_TR3_C_MODEL_DESTINATIONS = 14,
  ET_TR3_C_O2_MOMENT_ASSIGNMENTS = 28,
  ET_TR3_C_RESTORE_ASSIGNMENTS = 42,
  ET_TR3_C_O2_FIRST_INDEX = 14
};

enum {
  ET_TR3_C_I2_TERMINAL_COMMITTED = 1,
  ET_TR3_C_I2_TERMINAL_ABORTED = 2
};

typedef struct et_tr3_c_i2_restore42_append_v1 {
  size_t struct_size;
  const void *restore_authority;
  et_f32_parameter *parameters[ET_TR3_C_MODEL_DESTINATIONS];
  const void *p1_handles[ET_TR3_C_MODEL_DESTINATIONS];
  et_f32_tensor *moment_destinations[ET_TR3_C_O2_MOMENT_ASSIGNMENTS];
  const et_f32_tensor *moment_sources[ET_TR3_C_O2_MOMENT_ASSIGNMENTS];
} et_tr3_c_i2_restore42_append_v1;

#define ET_TR3_C_I2_RESTORE42_APPEND_V1_0_SIZE                               \
  ((size_t)sizeof(et_tr3_c_i2_restore42_append_v1))

#if defined(__cplusplus)
#define ET_TR3_C_I2_STATIC_ASSERT(condition, message)                         \
  static_assert(condition, message)
#else
#define ET_TR3_C_I2_STATIC_ASSERT(condition, message)                         \
  _Static_assert(condition, message)
#endif

ET_TR3_C_I2_STATIC_ASSERT(sizeof(size_t) == 8u && sizeof(void *) == 8u,
                          "TR3-C restore requires LP64 pointer carriers");
ET_TR3_C_I2_STATIC_ASSERT(ET_TR3_C_I2_TERMINAL_COMMITTED == 1 &&
                              ET_TR3_C_I2_TERMINAL_ABORTED == 2,
                          "TR3-C restore terminal selectors changed");
ET_TR3_C_I2_STATIC_ASSERT(
    sizeof(et_tr3_c_i2_restore42_append_v1) == 688u,
    "TR3-C restore42 append request size changed");
ET_TR3_C_I2_STATIC_ASSERT(
    offsetof(et_tr3_c_i2_restore42_append_v1, parameters) == 16u,
    "TR3-C restore42 parameter offset changed");
ET_TR3_C_I2_STATIC_ASSERT(
    offsetof(et_tr3_c_i2_restore42_append_v1, p1_handles) == 128u,
    "TR3-C restore42 P1 offset changed");
ET_TR3_C_I2_STATIC_ASSERT(
    offsetof(et_tr3_c_i2_restore42_append_v1, moment_destinations) == 240u,
    "TR3-C restore42 destination offset changed");
ET_TR3_C_I2_STATIC_ASSERT(
    offsetof(et_tr3_c_i2_restore42_append_v1, moment_sources) == 464u,
    "TR3-C restore42 source offset changed");

#undef ET_TR3_C_I2_STATIC_ASSERT

int32_t et_tr3_c_i2_copy_builder_create_restore42_v1(
    const void *restore_authority, void **builder);
int32_t et_tr3_c_i2_copy_builder_append_parameter_v1(
    void *builder, const void *restore_authority, et_f32_parameter *parameter,
    const void *p1_handle, const et_f32_tensor *owned_source);
int32_t et_tr3_c_i2_copy_builder_append_restore42_v1(
    void *builder, const et_tr3_c_i2_restore42_append_v1 *request);
int32_t et_tr3_c_i2_copy_builder_verify_restore42_v1(
    const void *builder, const et_tr3_c_i2_restore42_append_v1 *request);
int32_t et_tr3_c_i2_copy_builder_prepare_restore42_v1(
    void *builder, const void *restore_authority);
int32_t et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(
    void *builder, const void *restore_authority);
int32_t et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
    void *builder, const void *restore_authority);
int32_t et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
    const void *builder, uint32_t expected_terminal);

#ifdef __cplusplus
}
#endif

#endif
