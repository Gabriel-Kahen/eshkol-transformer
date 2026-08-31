#include "t1_i64_shell.h"

#include "eshkol_transformer/i64_tensor.h"

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
