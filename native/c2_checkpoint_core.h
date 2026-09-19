#ifndef ESHKOL_TRANSFORMER_C2_CHECKPOINT_CORE_H
#define ESHKOL_TRANSFORMER_C2_CHECKPOINT_CORE_H

#include "c2_checkpoint_format.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_c2_private_checkpoint_image et_c2_private_checkpoint_image;

enum et_c2_checkpoint_core_status {
  ET_C2_CORE_OK = ET_C2_FORMAT_OK,
  ET_C2_CORE_INVALID_ARGUMENT = ET_C2_FORMAT_INVALID_ARGUMENT,
  ET_C2_CORE_CORRUPT_DATA = ET_C2_FORMAT_CORRUPT_DATA,
  ET_C2_CORE_VERSION_MISMATCH = ET_C2_FORMAT_VERSION_MISMATCH,
  ET_C2_CORE_UNSUPPORTED = ET_C2_FORMAT_UNSUPPORTED,
  ET_C2_CORE_DETERMINISM_UNAVAILABLE = ET_C2_FORMAT_DETERMINISM_UNAVAILABLE,
  ET_C2_CORE_SHAPE_MISMATCH = ET_C2_FORMAT_SHAPE_MISMATCH,
  ET_C2_CORE_DTYPE_MISMATCH = ET_C2_FORMAT_DTYPE_MISMATCH,
  ET_C2_CORE_DEVICE_MISMATCH = ET_C2_FORMAT_DEVICE_MISMATCH,
  ET_C2_CORE_NONCONTIGUOUS = ET_C2_FORMAT_NONCONTIGUOUS,
  ET_C2_CORE_IO = 10,
  ET_C2_CORE_INTERNAL = 11
};

enum et_c2_checkpoint_core_code {
  ET_C2_CORE_CODE_NONE = 0,
  ET_C2_CORE_CODE_ARGUMENT = 1,
  ET_C2_CORE_CODE_PHYSICAL_SIZE = 2,
  ET_C2_CORE_CODE_OUTER_PROBE = 3,
  ET_C2_CORE_CODE_C1_PROBE = 4,
  ET_C2_CORE_CODE_PROBED_HEADER_CHANGED = 5,
  ET_C2_CORE_CODE_STAGING_ALLOCATION = 6,
  ET_C2_CORE_CODE_NATIVE_IO = 7,
  ET_C2_CORE_CODE_READER_CLOSE = 8,
  ET_C2_CORE_CODE_FORMAT = 9,
  ET_C2_CORE_CODE_INVALID_IMAGE = 10
};

typedef struct et_c2_checkpoint_core_error_v1 {
  size_t struct_size;
  uint32_t category;
  uint32_t code;
  uint64_t offset;
  int64_t native_status;
  et_c2_checkpoint_format_error_v1 format_error;
} et_c2_checkpoint_core_error_v1;

#define ET_C2_CHECKPOINT_CORE_ERROR_V1_SIZE                                \
  ((size_t)sizeof(et_c2_checkpoint_core_error_v1))

/*
 * Private C2 retained-image loader.  It opens one O_NOFOLLOW descriptor,
 * probes both fixed headers, reads the complete bounded image once through
 * that descriptor, compares both probes, validates all retained bytes, and
 * rechecks final size/EOF before publishing one guarded image owner.
 */
int32_t et_c2_private_checkpoint_load_image_v1(
    const char *path, const et_c2_checkpoint_limits_v1 *limits, uint32_t mode,
    et_c2_private_checkpoint_image **image,
    et_c2_checkpoint_core_error_v1 *error);

int32_t et_c2_private_checkpoint_image_view_v1(
    const et_c2_private_checkpoint_image *image,
    const et_c2_checkpoint_view_v1 **view,
    et_c2_checkpoint_core_error_v1 *error);

/* A nonnull slot containing NULL is an idempotent release. */
int32_t et_c2_private_checkpoint_image_release_v1(
    et_c2_private_checkpoint_image **image,
    et_c2_checkpoint_core_error_v1 *error);

enum et_c2_checkpoint_core_test_stage {
  ET_C2_CORE_TEST_AFTER_OUTER_PROBE = 1,
  ET_C2_CORE_TEST_AFTER_C1_PROBE = 2,
  ET_C2_CORE_TEST_AFTER_FULL_READ = 3
};
#ifdef ET_C2_CHECKPOINT_CORE_TESTING
typedef void (*et_c2_checkpoint_core_test_hook_v1)(uint32_t stage,
                                                    const char *path);
void et_c2_checkpoint_core_test_reset_v1(void);
void et_c2_checkpoint_core_test_set_hook_v1(
    et_c2_checkpoint_core_test_hook_v1 hook);
void et_c2_checkpoint_core_test_fail_alloc_after_v1(
    size_t successful_allocations);
size_t et_c2_checkpoint_core_test_live_images_v1(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
