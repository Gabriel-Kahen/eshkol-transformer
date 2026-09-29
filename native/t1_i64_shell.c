#include "t1_i64_shell.h"

#include "eshkol_transformer/i64_tensor.h"
#ifdef ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
#include "i64_t1_pair_private.h"
#endif

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ET_T1_I64_SHELL_MAGIC UINT64_C(0x455454315348454c)

typedef struct et_t1_i64_shell {
  uint64_t magic;
  struct et_t1_i64_shell *registry_next;
  et_i64_tensor *tensor;
  et_i64_tensor_borrow *borrow;
  const et_kernel_tensor_view_v1 *view;
  int64_t length;
  int sealed;
#ifdef ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
  et_i64_tensor *birth_tensor;
  et_i64_tensor_borrow *birth_borrow;
  const et_kernel_tensor_view_v1 *birth_view;
  const void *birth_data;
  const uint64_t *birth_shape;
  int64_t birth_length;
#ifdef ET_T1_I64_SHELL_TESTING
  int test_corrupted;
  et_i64_tensor *test_tensor;
  et_i64_tensor_borrow *test_borrow;
  const et_kernel_tensor_view_v1 *test_view;
  et_kernel_tensor_view_v1 test_view_copy;
  int64_t test_length;
  uint64_t test_shape_extent;
#endif
#endif
} et_t1_i64_shell;

static et_t1_i64_shell *et_t1_i64_shell_registry;
static int64_t et_t1_i64_shell_last_status;

enum {
  ET_T1_I64_SHELL_TEST_FAIL_NONE_INTERNAL = 0,
  ET_T1_I64_SHELL_TEST_FAIL_SHELL_ALLOC_INTERNAL = 1,
  ET_T1_I64_SHELL_TEST_FAIL_AFTER_TENSOR_INTERNAL = 2,
  ET_T1_I64_SHELL_TEST_FAIL_AFTER_BORROW_INTERNAL = 3
};

#ifdef ET_T1_I64_SHELL_TESTING
static int64_t et_t1_i64_shell_fail_stage;

void et_t1_i64_shell_test_fail_stage_v1(int64_t stage) {
  et_t1_i64_shell_fail_stage = stage;
}

int64_t et_t1_i64_shell_test_live_count_v1(void) {
  int64_t count = 0;
  const et_t1_i64_shell *entry = et_t1_i64_shell_registry;
  while (entry != NULL) {
    ++count;
    entry = entry->registry_next;
  }
  return count;
}

static int et_t1_i64_shell_should_fail(int64_t stage) {
  return et_t1_i64_shell_fail_stage == stage;
}
#else
static int et_t1_i64_shell_should_fail(int64_t stage) {
  (void)stage;
  return 0;
}
#endif

/* Pointer-value comparison is the admission boundary.  candidate is not read. */
static et_t1_i64_shell *et_t1_i64_shell_admit(const void *candidate) {
  et_t1_i64_shell *entry = et_t1_i64_shell_registry;
  while (entry != NULL) {
    if ((const void *)entry == candidate) {
      return entry;
    }
    entry = entry->registry_next;
  }
  return NULL;
}

#ifdef ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
int32_t et_t1_i64_shell_private_c4_read_v1(const void *candidate,
                                            int64_t output[2]) {
  const et_t1_i64_shell *shell = et_t1_i64_shell_admit(candidate);
  int32_t pair_status;
  const int64_t *data;
  int64_t words[2] = {0, 0};
  if (shell == NULL || output == NULL) return ET_T1_C4_READ_INVALID_ARGUMENT;
  if (!shell->sealed || shell->tensor != shell->birth_tensor ||
      shell->borrow != shell->birth_borrow || shell->view != shell->birth_view ||
      shell->length != shell->birth_length)
    return ET_T1_C4_READ_INVALID_STATE;
  if (shell->birth_length != 1 && shell->birth_length != 2)
    return ET_T1_C4_READ_SHAPE;
  pair_status = et_i64_tensor_private_t1_pair_validate_v1(
      shell->birth_tensor, shell->birth_borrow, shell->birth_view,
      shell->birth_length);
  if (pair_status == ET_I64_T1_PAIR_INVALID_STATE)
    return ET_T1_C4_READ_INVALID_STATE;
  if (pair_status == ET_I64_T1_PAIR_SHAPE) return ET_T1_C4_READ_SHAPE;
  if (pair_status == ET_I64_T1_PAIR_DTYPE) return ET_T1_C4_READ_DTYPE;
  if (pair_status == ET_I64_T1_PAIR_DEVICE) return ET_T1_C4_READ_DEVICE;
  if (pair_status == ET_I64_T1_PAIR_LAYOUT) return ET_T1_C4_READ_LAYOUT;
  if (pair_status == ET_I64_T1_PAIR_STORAGE) return ET_T1_C4_READ_STORAGE;
  if (pair_status != ET_I64_T1_PAIR_OK) return ET_T1_C4_READ_INVALID_STATE;
  if (shell->view->data != shell->birth_data ||
      shell->view->shape != shell->birth_shape)
    return ET_T1_C4_READ_STORAGE;
  data = (const int64_t *)shell->birth_data;
  for (int64_t i = 0; i < shell->birth_length; ++i) {
    words[i] = data[i];
    if (words[i] < 0 || words[i] > 255) return ET_T1_C4_READ_RANGE;
  }
  for (int64_t i = 0; i < shell->birth_length; ++i) output[i] = words[i];
  return ET_T1_C4_READ_OK;
}
#ifdef ET_T1_I64_SHELL_TESTING
int32_t et_t1_i64_shell_test_corrupt_v1(void *candidate, int32_t field,
                                        uintptr_t value) {
  et_t1_i64_shell *shell = et_t1_i64_shell_admit(candidate);
  et_t1_i64_shell *other = NULL;
  et_kernel_tensor_view_v1 *view;
  if (shell == NULL || !shell->sealed || shell->test_corrupted ||
      shell->birth_view == NULL || shell->birth_shape == NULL)
    return -1;
  if (field < ET_T1_TEST_CORRUPT_RANK ||
      field > ET_T1_TEST_CORRUPT_WHOLE_PAIR)
    return -1;
  if (field == ET_T1_TEST_CORRUPT_WHOLE_PAIR) {
    other = et_t1_i64_shell_admit((const void *)value);
    if (other == NULL || other == shell || !other->sealed ||
        other->test_corrupted || other->tensor != other->birth_tensor ||
        other->borrow != other->birth_borrow ||
        other->view != other->birth_view ||
        other->length != shell->length)
      return -1;
  }
  shell->test_tensor = shell->tensor;
  shell->test_borrow = shell->borrow;
  shell->test_view = shell->view;
  shell->test_length = shell->length;
  shell->test_view_copy = *shell->birth_view;
  shell->test_shape_extent = shell->birth_shape[0];
  shell->test_corrupted = 1;
  view = (et_kernel_tensor_view_v1 *)shell->birth_view;
  switch (field) {
    case ET_T1_TEST_CORRUPT_RANK: view->rank = (size_t)value; break;
    case ET_T1_TEST_CORRUPT_SHAPE_POINTER:
      view->shape = (const uint64_t *)value; break;
    case ET_T1_TEST_CORRUPT_SHAPE_EXTENT:
      *(uint64_t *)shell->birth_shape = (uint64_t)value; break;
    case ET_T1_TEST_CORRUPT_STRUCT_SIZE: view->struct_size = (size_t)value; break;
    case ET_T1_TEST_CORRUPT_BYTE_LENGTH: view->byte_length = (size_t)value; break;
    case ET_T1_TEST_CORRUPT_DTYPE: view->dtype = (const char *)value; break;
    case ET_T1_TEST_CORRUPT_DEVICE: view->device = (const char *)value; break;
    case ET_T1_TEST_CORRUPT_LAYOUT:
      view->layout = (et_kernel_layout_v1)value; break;
    case ET_T1_TEST_CORRUPT_OFFSET: view->offset_bytes = (uint64_t)value; break;
    case ET_T1_TEST_CORRUPT_DATA: view->data = (void *)value; break;
    case ET_T1_TEST_CORRUPT_TENSOR:
      shell->tensor = (et_i64_tensor *)value; break;
    case ET_T1_TEST_CORRUPT_BORROW:
      shell->borrow = (et_i64_tensor_borrow *)value; break;
    case ET_T1_TEST_CORRUPT_VIEW:
      shell->view = (const et_kernel_tensor_view_v1 *)value; break;
    case ET_T1_TEST_CORRUPT_LENGTH: shell->length = (int64_t)value; break;
    case ET_T1_TEST_CORRUPT_WHOLE_PAIR:
      shell->tensor = other->tensor;
      shell->borrow = other->borrow;
      shell->view = other->view;
      shell->length = other->length;
      break;
  }
  return 0;
}

int32_t et_t1_i64_shell_test_restore_v1(void *candidate) {
  et_t1_i64_shell *shell = et_t1_i64_shell_admit(candidate);
  if (shell == NULL || !shell->test_corrupted) return -1;
  *(uint64_t *)shell->birth_shape = shell->test_shape_extent;
  *(et_kernel_tensor_view_v1 *)shell->birth_view = shell->test_view_copy;
  shell->tensor = shell->test_tensor;
  shell->borrow = shell->test_borrow;
  shell->view = shell->test_view;
  shell->length = shell->test_length;
  shell->test_corrupted = 0;
  return 0;
}
#endif
#endif

static void et_t1_i64_shell_cleanup_unpublished(et_t1_i64_shell *shell) {
  et_i64_tensor_error error = {0};
  if (shell == NULL) {
    return;
  }
  if (shell->borrow != NULL) {
    (void)et_i64_tensor_borrow_end_v1(&shell->borrow, &error);
  }
  if (shell->tensor != NULL) {
    et_i64_tensor_error_clear_v1(&error);
    (void)et_i64_tensor_destroy_v1(&shell->tensor, &error);
  }
  shell->magic = 0u;
  free(shell);
}

void *et_t1_i64_shell_create_v1(int64_t length) {
  et_t1_i64_shell *shell;
  et_i64_tensor_error error = {0};
  uint64_t shape;

  if (length < 0) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT;
    return NULL;
  }
  if ((uint64_t)length > (uint64_t)(SIZE_MAX / sizeof(int64_t))) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_RANGE;
    return NULL;
  }
  if (et_t1_i64_shell_should_fail(
          ET_T1_I64_SHELL_TEST_FAIL_SHELL_ALLOC_INTERNAL)) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_ALLOCATION_FAILED;
    return NULL;
  }
  shape = (uint64_t)length;
  shell = (et_t1_i64_shell *)calloc(1u, sizeof(*shell));
  if (shell == NULL) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_ALLOCATION_FAILED;
    return NULL;
  }
  shell->magic = ET_T1_I64_SHELL_MAGIC;
  shell->length = length;
  if (et_i64_tensor_create_v1(1u, &shape, &shell->tensor, &error) != 0) {
    et_t1_i64_shell_last_status =
        error.code == ET_I64_TENSOR_CODE_ALLOCATION_FAILED
            ? ET_T1_I64_SHELL_STATUS_ALLOCATION_FAILED
            : ET_T1_I64_SHELL_STATUS_INTERNAL;
    et_t1_i64_shell_cleanup_unpublished(shell);
    return NULL;
  }
  if (et_t1_i64_shell_should_fail(
          ET_T1_I64_SHELL_TEST_FAIL_AFTER_TENSOR_INTERNAL)) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INTERNAL;
    et_t1_i64_shell_cleanup_unpublished(shell);
    return NULL;
  }
  et_i64_tensor_error_clear_v1(&error);
  if (et_i64_tensor_borrow_begin_v1(shell->tensor, &shell->borrow, &error) != 0) {
    et_t1_i64_shell_last_status =
        error.code == ET_I64_TENSOR_CODE_ALLOCATION_FAILED
            ? ET_T1_I64_SHELL_STATUS_ALLOCATION_FAILED
            : ET_T1_I64_SHELL_STATUS_INTERNAL;
    et_t1_i64_shell_cleanup_unpublished(shell);
    return NULL;
  }
  if (et_t1_i64_shell_should_fail(
          ET_T1_I64_SHELL_TEST_FAIL_AFTER_BORROW_INTERNAL)) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INTERNAL;
    et_t1_i64_shell_cleanup_unpublished(shell);
    return NULL;
  }
  et_i64_tensor_error_clear_v1(&error);
  if (et_i64_tensor_borrow_view_v1(shell->borrow, &shell->view, &error) != 0 ||
      shell->view == NULL ||
      shell->view->struct_size != sizeof(et_kernel_tensor_view_v1) ||
      shell->view->dtype == NULL || strcmp(shell->view->dtype, "i64") != 0 ||
      shell->view->device == NULL || strcmp(shell->view->device, "cpu") != 0 ||
      shell->view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      shell->view->offset_bytes != 0u || shell->view->rank != 1u ||
      shell->view->shape == NULL || shell->view->shape[0] != (uint64_t)length ||
      shell->view->byte_length != (size_t)length * sizeof(int64_t) ||
      (length != 0 && shell->view->data == NULL)) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INTERNAL;
    et_t1_i64_shell_cleanup_unpublished(shell);
    return NULL;
  }
#ifdef ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
  shell->birth_tensor = shell->tensor;
  shell->birth_borrow = shell->borrow;
  shell->birth_view = shell->view;
  shell->birth_data = shell->view->data;
  shell->birth_shape = shell->view->shape;
  shell->birth_length = length;
#endif
  shell->registry_next = et_t1_i64_shell_registry;
  et_t1_i64_shell_registry = shell;
  et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_OK;
  return shell;
}

int64_t et_t1_i64_shell_length_v1(const void *candidate) {
  const et_t1_i64_shell *shell = et_t1_i64_shell_admit(candidate);
  if (shell == NULL) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT;
    return -1;
  }
  et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_OK;
  return shell->length;
}

int64_t et_t1_i64_shell_write_v1(void *candidate, int64_t index,
                                 int64_t value) {
  et_t1_i64_shell *shell = et_t1_i64_shell_admit(candidate);
  int64_t *data;
  if (shell == NULL) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT;
    return et_t1_i64_shell_last_status;
  }
  if (shell->sealed) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INVALID_STATE;
    return et_t1_i64_shell_last_status;
  }
  if (index < 0 || index >= shell->length) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_RANGE;
    return et_t1_i64_shell_last_status;
  }
  data = (int64_t *)shell->view->data;
  data[(size_t)index] = value;
  et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_OK;
  return et_t1_i64_shell_last_status;
}

int64_t et_t1_i64_shell_read_v1(const void *candidate, int64_t index) {
  const et_t1_i64_shell *shell = et_t1_i64_shell_admit(candidate);
  const int64_t *data;
  if (shell == NULL) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT;
    return INT64_MIN;
  }
  if (!shell->sealed) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INVALID_STATE;
    return INT64_MIN;
  }
  if (index < 0 || index >= shell->length) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_RANGE;
    return INT64_MIN;
  }
  data = (const int64_t *)shell->view->data;
  et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_OK;
  return data[(size_t)index];
}

int64_t et_t1_i64_shell_seal_v1(void *candidate) {
  et_t1_i64_shell *shell = et_t1_i64_shell_admit(candidate);
  if (shell == NULL) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT;
    return et_t1_i64_shell_last_status;
  }
  if (shell->sealed) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INVALID_STATE;
    return et_t1_i64_shell_last_status;
  }
  shell->sealed = 1;
  et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_OK;
  return et_t1_i64_shell_last_status;
}

int64_t et_t1_i64_shell_abort_v1(void *candidate) {
  et_t1_i64_shell *shell = et_t1_i64_shell_admit(candidate);
  et_t1_i64_shell **link;
  if (shell == NULL) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT;
    return et_t1_i64_shell_last_status;
  }
  if (shell->sealed) {
    et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_INVALID_STATE;
    return et_t1_i64_shell_last_status;
  }
  link = &et_t1_i64_shell_registry;
  while (*link != shell) {
    link = &(*link)->registry_next;
  }
  *link = shell->registry_next;
  et_t1_i64_shell_cleanup_unpublished(shell);
  et_t1_i64_shell_last_status = ET_T1_I64_SHELL_STATUS_OK;
  return et_t1_i64_shell_last_status;
}

int64_t et_t1_i64_shell_last_status_v1(void) {
  return et_t1_i64_shell_last_status;
}
