#define _POSIX_C_SOURCE 200809L

#include "d2_native.h"

#include "eshkol_transformer/i64_tensor.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct et_d2_batch_control et_d2_batch_control;
typedef struct et_d2_dataset_control et_d2_dataset_control;

struct et_d2_batch_control {
  int64_t generation;
  int64_t active_lease_generation;
  size_t rows;
  size_t columns;
  size_t elements;
  et_i64_tensor *input_tensor;
  et_i64_tensor_borrow *input_storage_borrow;
  const et_kernel_tensor_view_v1 *input_view;
  et_i64_tensor *target_tensor;
  et_i64_tensor_borrow *target_storage_borrow;
  const et_kernel_tensor_view_v1 *target_view;
  uint8_t *mask_data;
  uint64_t mask_shape[2];
  et_kernel_tensor_view_v1 mask_view;
  int wrote_any;
  int sealed;
};

struct et_d2_dataset_control {
  et_d2_dataset_control *registry_next;
  const void *owner;
  uint64_t next_batch_generation;
  uint64_t next_lease_generation;
  et_d2_batch_control *current_batch;
};

static et_d2_dataset_control *live_datasets;
static int64_t last_status;

#ifdef ET_D2_NATIVE_TESTING
static int64_t fail_stage;
static int64_t read_fail_stage;
static int64_t owned_allocations;
#endif

/* Exclusive span ends must be representable, matching accepted I1 policy. */
static int span_fits(const void *pointer, size_t bytes) {
  return bytes == 0u ||
         (pointer != NULL && (uintptr_t)pointer <= UINTPTR_MAX - bytes);
}

static int overlaps(const void *left, size_t left_bytes, const void *right,
                    size_t right_bytes) {
  uintptr_t left_start = (uintptr_t)left;
  uintptr_t right_start = (uintptr_t)right;
  if (left_bytes == 0u || right_bytes == 0u) {
    return 0;
  }
  if (!span_fits(left, left_bytes) || !span_fits(right, right_bytes)) {
    return 1;
  }
  return left_start < right_start + right_bytes &&
         right_start < left_start + left_bytes;
}

static int multiply_size(size_t left, size_t right, size_t *result) {
  if (left != 0u && right > SIZE_MAX / left) {
    return 0;
  }
  *result = left * right;
  return 1;
}

static int64_t status(int64_t value) {
  last_status = value;
  return value;
}

int64_t et_d2_batch_last_status_v1(void) { return last_status; }
static int should_fail(int64_t stage);

/* Owner is compared but never dereferenced. */
static et_d2_dataset_control *find_dataset(const void *owner) {
  et_d2_dataset_control *entry;
  for (entry = live_datasets; entry != NULL; entry = entry->registry_next) {
    if (entry->owner == owner) {
      return entry;
    }
  }
  return NULL;
}

static et_d2_batch_control *find_batch(const void *owner, int64_t generation) {
  et_d2_dataset_control *dataset = find_dataset(owner);
  et_d2_batch_control *batch =
      dataset == NULL ? NULL : dataset->current_batch;
  if (batch == NULL || generation <= 0 || batch->generation != generation) {
    return NULL;
  }
  return batch;
}

static int live_storage_overlap(const void *pointer, size_t bytes) {
  const et_d2_dataset_control *dataset;
  for (dataset = live_datasets; dataset != NULL;
       dataset = dataset->registry_next) {
    const et_d2_batch_control *batch = dataset->current_batch;
    if (overlaps(pointer, bytes, dataset, sizeof(*dataset))) {
      return 1;
    }
    if (batch != NULL &&
        (overlaps(pointer, bytes, batch, sizeof(*batch)) ||
        overlaps(pointer, bytes, batch->mask_data, batch->elements) ||
        overlaps(pointer, bytes, batch->input_view,
                 sizeof(*batch->input_view)) ||
        overlaps(pointer, bytes, batch->target_view,
                 sizeof(*batch->target_view)) ||
        overlaps(pointer, bytes, batch->input_view->shape,
                 2u * sizeof(*batch->input_view->shape)) ||
        overlaps(pointer, bytes, batch->target_view->shape,
                 2u * sizeof(*batch->target_view->shape)) ||
        overlaps(pointer, bytes, batch->input_view->data,
                 batch->input_view->byte_length) ||
        overlaps(pointer, bytes, batch->target_view->data,
                 batch->target_view->byte_length))) {
      return 1;
    }
  }
  return 0;
}

static int peek_generation(uint64_t next_generation, int64_t *generation) {
  if (next_generation == 0u || next_generation > (uint64_t)INT64_MAX) {
    return 0;
  }
  *generation = (int64_t)next_generation;
  return 1;
}

static void unlink_dataset(et_d2_dataset_control *target) {
  et_d2_dataset_control **link = &live_datasets;
  while (*link != NULL) {
    if (*link == target) {
      *link = target->registry_next;
      target->registry_next = NULL;
      return;
    }
    link = &(*link)->registry_next;
  }
  abort();
}

static int valid_i64_view(const et_kernel_tensor_view_v1 *view, size_t rows,
                          size_t columns, size_t elements) {
  size_t bytes;
  return multiply_size(elements, sizeof(int64_t), &bytes) && view != NULL &&
         view->struct_size == sizeof(*view) && view->data != NULL &&
         view->byte_length == bytes && view->dtype != NULL &&
         strcmp(view->dtype, "i64") == 0 && view->device != NULL &&
         strcmp(view->device, "cpu") == 0 &&
         view->layout == ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR &&
         view->offset_bytes == 0u && view->rank == 2u && view->shape != NULL &&
         view->shape[0] == rows && view->shape[1] == columns;
}

/* Admitted D2 ownership makes these direct I1 cleanup calls infallible. */
static void require_i1_cleanup(int32_t result) {
  if (result != 0) {
    abort();
  }
}

static void cleanup_control(et_d2_batch_control *batch) {
  et_i64_tensor_error error = {0};
  if (batch == NULL) {
    return;
  }
  if (batch->target_storage_borrow != NULL) {
    require_i1_cleanup(
        et_i64_tensor_borrow_end_v1(&batch->target_storage_borrow, &error));
  }
  if (batch->target_tensor != NULL) {
    et_i64_tensor_error_clear_v1(&error);
    require_i1_cleanup(et_i64_tensor_destroy_v1(&batch->target_tensor, &error));
  }
  if (batch->input_storage_borrow != NULL) {
    et_i64_tensor_error_clear_v1(&error);
    require_i1_cleanup(
        et_i64_tensor_borrow_end_v1(&batch->input_storage_borrow, &error));
  }
  if (batch->input_tensor != NULL) {
    et_i64_tensor_error_clear_v1(&error);
    require_i1_cleanup(et_i64_tensor_destroy_v1(&batch->input_tensor, &error));
  }
  if (batch->mask_data != NULL) {
    free(batch->mask_data);
#ifdef ET_D2_NATIVE_TESTING
    owned_allocations--;
#endif
  }
  free(batch);
#ifdef ET_D2_NATIVE_TESTING
  owned_allocations--;
#endif
}

int64_t et_d2_dataset_open_v1(const void *owner) {
  et_d2_dataset_control *dataset;
  if (owner == NULL || !span_fits(owner, 1u) ||
      live_storage_overlap(owner, 1u)) {
    return status(ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  }
  if (find_dataset(owner) != NULL) {
    return status(ET_D2_NATIVE_STATUS_INVALID_STATE);
  }
  if (should_fail(ET_D2_TEST_FAIL_DATASET)) {
    return status(ET_D2_NATIVE_STATUS_ALLOCATION_FAILED);
  }
  dataset = (et_d2_dataset_control *)calloc(1u, sizeof(*dataset));
  if (dataset == NULL) {
    return status(ET_D2_NATIVE_STATUS_ALLOCATION_FAILED);
  }
#ifdef ET_D2_NATIVE_TESTING
  owned_allocations++;
#endif
  dataset->owner = owner;
  dataset->next_batch_generation = 1u;
  dataset->next_lease_generation = 1u;
  dataset->registry_next = live_datasets;
  live_datasets = dataset;
  return status(ET_D2_NATIVE_STATUS_OK);
}

int64_t et_d2_dataset_close_v1(const void *owner) {
  et_d2_dataset_control *dataset = find_dataset(owner);
  et_d2_batch_control *batch;
  if (dataset == NULL) {
    return status(ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  }
  if (dataset->current_batch != NULL &&
      dataset->current_batch->active_lease_generation != 0) {
    return status(ET_D2_NATIVE_STATUS_INVALID_STATE);
  }
  /* Native authority is revoked before the fixed destruction tail. */
  unlink_dataset(dataset);
  batch = dataset->current_batch;
  dataset->current_batch = NULL;
  if (batch != NULL) {
    cleanup_control(batch);
  }
  free(dataset);
#ifdef ET_D2_NATIVE_TESTING
  owned_allocations--;
#endif
  return status(ET_D2_NATIVE_STATUS_OK);
}

#ifdef ET_D2_NATIVE_TESTING
void et_d2_batch_test_fail_stage_v1(int64_t stage) { fail_stage = stage; }

void et_d2_exact_read_test_fail_stage_v1(int64_t stage) {
  read_fail_stage = stage;
}

static int should_fail(int64_t stage) { return fail_stage == stage; }

static int should_fail_read(int64_t stage) {
  return read_fail_stage == stage;
}

int64_t et_d2_batch_test_live_count_v1(void) {
  int64_t count = 0;
  const et_d2_dataset_control *entry;
  for (entry = live_datasets; entry != NULL; entry = entry->registry_next) {
    if (entry->current_batch != NULL) {
      count++;
    }
  }
  return count;
}

int64_t et_d2_dataset_test_live_count_v1(void) {
  int64_t count = 0;
  const et_d2_dataset_control *entry;
  for (entry = live_datasets; entry != NULL; entry = entry->registry_next) {
    count++;
  }
  return count;
}

int64_t et_d2_batch_test_borrow_count_v1(void) {
  int64_t count = 0;
  const et_d2_dataset_control *entry;
  for (entry = live_datasets; entry != NULL; entry = entry->registry_next) {
    if (entry->current_batch != NULL &&
        entry->current_batch->active_lease_generation != 0) {
      count++;
    }
  }
  return count;
}

int64_t et_d2_test_owned_allocation_count_v1(void) {
  return owned_allocations;
}

size_t et_d2_dataset_test_control_bytes_v1(void) {
  return sizeof(et_d2_dataset_control);
}

size_t et_d2_batch_test_control_bytes_v1(void) {
  return sizeof(et_d2_batch_control);
}

size_t et_d2_batch_test_borrow_bytes_v1(void) { return 0u; }

size_t et_d2_batch_test_view_metadata_bytes_v1(void) {
  return 3u * sizeof(et_kernel_tensor_view_v1) + 6u * sizeof(uint64_t);
}

size_t et_d2_batch_test_payload_bytes_v1(const void *owner,
                                         int64_t generation) {
  const et_d2_batch_control *batch = find_batch(owner, generation);
  return batch == NULL ? 0u : 17u * batch->elements;
}

void et_d2_batch_test_exhaust_generations_v1(void) {
  et_d2_dataset_control *entry;
  for (entry = live_datasets; entry != NULL; entry = entry->registry_next) {
    entry->next_batch_generation = (uint64_t)INT64_MAX + 1u;
    entry->next_lease_generation = (uint64_t)INT64_MAX + 1u;
  }
}
#else
static int should_fail(int64_t stage) {
  (void)stage;
  return 0;
}

static int should_fail_read(int64_t stage) {
  (void)stage;
  return 0;
}
#endif

static int64_t read_status(int stage, int error_number) {
  uint32_t stable_error = (uint32_t)(error_number == 0 ? EIO : error_number);
  return ((int64_t)stage << 32) | (int64_t)stable_error;
}

static int bytevector_payload(const void *header, int64_t expected_length,
                              const uint8_t **payload) {
  int64_t declared_length;
  size_t total_bytes;
  if (header == NULL || expected_length < 0 ||
      (uint64_t)expected_length > SIZE_MAX - sizeof(int64_t)) {
    return 0;
  }
  total_bytes = sizeof(int64_t) + (size_t)expected_length;
  if ((uintptr_t)header % _Alignof(int64_t) != 0u ||
      !span_fits(header, total_bytes) ||
      live_storage_overlap(header, total_bytes)) {
    return 0;
  }
  memcpy(&declared_length, header, sizeof(declared_length));
  if (declared_length != expected_length) {
    return 0;
  }
  *payload = (const uint8_t *)header + sizeof(declared_length);
  return 1;
}

int64_t et_d2_exact_read_v1(const char *path, int64_t path_bytes_value,
                            int64_t offset_value, void *header,
                            int64_t expected_length) {
  const uint8_t *const_payload;
  uint8_t *next;
  size_t path_bytes;
  size_t remaining;
  int descriptor;
  int saved_error;
  uint64_t current;

  if (path_bytes_value < 2 || offset_value < 0 ||
      (uint64_t)path_bytes_value > SIZE_MAX ||
      !bytevector_payload(header, expected_length, &const_payload)) {
    return read_status(ET_D2_EXACT_READ_ARGUMENT, EINVAL);
  }
  path_bytes = (size_t)path_bytes_value;
  remaining = (size_t)expected_length;
  current = (uint64_t)offset_value;
  next = (uint8_t *)(uintptr_t)const_payload;
  if (path == NULL || !span_fits(path, path_bytes) ||
      live_storage_overlap(path, path_bytes) || path[path_bytes - 1u] != '\0' ||
      memchr(path, '\0', path_bytes - 1u) != NULL ||
      overlaps(path, path_bytes, header,
               sizeof(int64_t) + (size_t)expected_length) ||
      (uint64_t)expected_length > (uint64_t)INT64_MAX - current) {
    return read_status(ET_D2_EXACT_READ_ARGUMENT, EINVAL);
  }
  descriptor = open(path, O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return read_status(ET_D2_EXACT_READ_OPEN, errno);
  }
  while (remaining != 0u) {
    size_t request =
        remaining > (size_t)SSIZE_MAX ? (size_t)SSIZE_MAX : remaining;
    ssize_t received;
    if (should_fail_read(ET_D2_TEST_READ_FAIL_READ)) {
      (void)close(descriptor);
      return read_status(ET_D2_EXACT_READ_READ, EIO);
    }
    received = pread(descriptor, next, request, (off_t)current);
    if (received < 0) {
      if (errno == EINTR) {
        continue;
      }
      saved_error = errno;
      (void)close(descriptor);
      return read_status(ET_D2_EXACT_READ_READ, saved_error);
    }
    if (received == 0) {
      (void)close(descriptor);
      return read_status(ET_D2_EXACT_READ_READ, EIO);
    }
    next += (size_t)received;
    remaining -= (size_t)received;
    current += (uint64_t)received;
  }
  if (close(descriptor) != 0) {
    return read_status(ET_D2_EXACT_READ_CLOSE, errno);
  }
  if (should_fail_read(ET_D2_TEST_READ_FAIL_CLOSE)) {
    return read_status(ET_D2_EXACT_READ_CLOSE, EIO);
  }
  return 0;
}

static int64_t create_i64_storage(const uint64_t shape[2],
                                  et_i64_tensor **tensor,
                                  et_i64_tensor_borrow **storage_borrow,
                                  const et_kernel_tensor_view_v1 **view,
                                  int64_t after_tensor_stage,
                                  int64_t after_borrow_stage) {
  et_i64_tensor_error error = {0};
  int32_t result = et_i64_tensor_create_v1(2u, shape, tensor, &error);
  if (result != 0) {
    return error.code == ET_I64_TENSOR_CODE_ALLOCATION_FAILED
               ? ET_D2_NATIVE_STATUS_ALLOCATION_FAILED
               : ET_D2_NATIVE_STATUS_INTERNAL;
  }
  if (should_fail(after_tensor_stage)) {
    return ET_D2_NATIVE_STATUS_ALLOCATION_FAILED;
  }
  result = et_i64_tensor_borrow_begin_v1(*tensor, storage_borrow, &error);
  if (result != 0) {
    return error.code == ET_I64_TENSOR_CODE_ALLOCATION_FAILED
               ? ET_D2_NATIVE_STATUS_ALLOCATION_FAILED
               : ET_D2_NATIVE_STATUS_INTERNAL;
  }
  if (should_fail(after_borrow_stage)) {
    return ET_D2_NATIVE_STATUS_ALLOCATION_FAILED;
  }
  if (et_i64_tensor_borrow_view_v1(*storage_borrow, view, &error) != 0) {
    return ET_D2_NATIVE_STATUS_INTERNAL;
  }
  return ET_D2_NATIVE_STATUS_OK;
}

int64_t et_d2_batch_create_v1(const void *owner, int64_t rows_value,
                              int64_t columns_value, int64_t max_payload) {
  et_d2_dataset_control *dataset = find_dataset(owner);
  et_d2_batch_control *batch = NULL;
  uint64_t shape[2];
  int64_t generation;
  size_t elements;
  size_t payload_bytes;
  int64_t result;

  if (owner == NULL || dataset == NULL) {
    (void)status(ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  if (rows_value <= 0 || columns_value <= 0 || max_payload <= 0 ||
      (uint64_t)rows_value > SIZE_MAX || (uint64_t)columns_value > SIZE_MAX ||
      !multiply_size((size_t)rows_value, (size_t)columns_value, &elements) ||
      !multiply_size(elements, 17u, &payload_bytes) ||
      payload_bytes > (uint64_t)max_payload) {
    (void)status(ET_D2_NATIVE_STATUS_RANGE);
    return 0;
  }
  if (dataset->current_batch != NULL) {
    (void)status(ET_D2_NATIVE_STATUS_INVALID_STATE);
    return 0;
  }
  if (!peek_generation(dataset->next_batch_generation, &generation)) {
    (void)status(ET_D2_NATIVE_STATUS_UNSUPPORTED);
    return 0;
  }
  if (should_fail(ET_D2_TEST_FAIL_CONTROL)) {
    (void)status(ET_D2_NATIVE_STATUS_ALLOCATION_FAILED);
    return 0;
  }
  batch = (et_d2_batch_control *)calloc(1u, sizeof(*batch));
  if (batch == NULL) {
    (void)status(ET_D2_NATIVE_STATUS_ALLOCATION_FAILED);
    return 0;
  }
#ifdef ET_D2_NATIVE_TESTING
  owned_allocations++;
#endif
  batch->generation = generation;
  batch->rows = (size_t)rows_value;
  batch->columns = (size_t)columns_value;
  batch->elements = elements;
  shape[0] = (uint64_t)rows_value;
  shape[1] = (uint64_t)columns_value;

  result = create_i64_storage(shape, &batch->input_tensor,
                              &batch->input_storage_borrow, &batch->input_view,
                              ET_D2_TEST_FAIL_AFTER_INPUT_TENSOR,
                              ET_D2_TEST_FAIL_AFTER_INPUT_BORROW);
  if (result != 0) {
    cleanup_control(batch);
    (void)status(result);
    return 0;
  }
  result = create_i64_storage(
      shape, &batch->target_tensor, &batch->target_storage_borrow,
      &batch->target_view, ET_D2_TEST_FAIL_AFTER_TARGET_TENSOR,
      ET_D2_TEST_FAIL_AFTER_TARGET_BORROW);
  if (result != 0) {
    cleanup_control(batch);
    (void)status(result);
    return 0;
  }
  if (!valid_i64_view(batch->input_view, batch->rows, batch->columns,
                      batch->elements) ||
      !valid_i64_view(batch->target_view, batch->rows, batch->columns,
                      batch->elements)) {
    cleanup_control(batch);
    (void)status(ET_D2_NATIVE_STATUS_INTERNAL);
    return 0;
  }
  if (should_fail(ET_D2_TEST_FAIL_MASK)) {
    cleanup_control(batch);
    (void)status(ET_D2_NATIVE_STATUS_ALLOCATION_FAILED);
    return 0;
  }
  batch->mask_data = (uint8_t *)calloc(batch->elements, sizeof(uint8_t));
  if (batch->mask_data == NULL) {
    cleanup_control(batch);
    (void)status(ET_D2_NATIVE_STATUS_ALLOCATION_FAILED);
    return 0;
  }
#ifdef ET_D2_NATIVE_TESTING
  owned_allocations++;
#endif
  batch->mask_shape[0] = shape[0];
  batch->mask_shape[1] = shape[1];
  batch->mask_view =
      (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1),
                                 batch->mask_data,
                                 batch->elements,
                                 "bool",
                                 "cpu",
                                 ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
                                 0u,
                                 2u,
                                 batch->mask_shape};
  if (should_fail(ET_D2_TEST_FAIL_BEFORE_PUBLISH)) {
    cleanup_control(batch);
    (void)status(ET_D2_NATIVE_STATUS_ALLOCATION_FAILED);
    return 0;
  }
  dataset->current_batch = batch;
  dataset->next_batch_generation++;
  (void)status(ET_D2_NATIVE_STATUS_OK);
  return generation;
}

static int64_t decode_i64_le(const uint8_t *source) {
  uint64_t bits = 0u;
  int64_t value;
  size_t index;
  for (index = 0u; index < 8u; index++) {
    bits |= (uint64_t)source[index] << (8u * index);
  }
  memcpy(&value, &bits, sizeof(value));
  return value;
}

int64_t et_d2_batch_write_pair_i64le_source_span_v1(
    const void *owner, int64_t generation, int64_t destination_value,
    const void *inputs_header, int64_t input_bytes, int64_t input_offset_value,
    const void *targets_header, int64_t target_bytes, int64_t target_offset_value,
    int64_t count_value) {
  et_d2_batch_control *batch = find_batch(owner, generation);
  const uint8_t *inputs;
  const uint8_t *targets;
  size_t destination;
  size_t count;
  size_t source_bytes;
  size_t input_offset;
  size_t target_offset;
  size_t index;
  if (batch == NULL) {
    return status(ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  }
  if (batch->sealed || batch->active_lease_generation != 0) {
    return status(ET_D2_NATIVE_STATUS_INVALID_STATE);
  }
  if (destination_value < 0 || count_value <= 0 || input_bytes < 0 ||
      target_bytes < 0 || input_offset_value < 0 || target_offset_value < 0 ||
      (uint64_t)destination_value > SIZE_MAX ||
      (uint64_t)count_value > SIZE_MAX ||
      (uint64_t)input_bytes > SIZE_MAX || (uint64_t)target_bytes > SIZE_MAX ||
      (uint64_t)input_offset_value > SIZE_MAX ||
      (uint64_t)target_offset_value > SIZE_MAX) {
    return status(ET_D2_NATIVE_STATUS_RANGE);
  }
  destination = (size_t)destination_value;
  count = (size_t)count_value;
  input_offset = (size_t)input_offset_value;
  target_offset = (size_t)target_offset_value;
  if (destination > batch->elements || count > batch->elements - destination ||
      !multiply_size(count, sizeof(int64_t), &source_bytes) ||
      input_offset > (size_t)input_bytes ||
      source_bytes > (size_t)input_bytes - input_offset ||
      target_offset > (size_t)target_bytes ||
      source_bytes > (size_t)target_bytes - target_offset) {
    return status(ET_D2_NATIVE_STATUS_RANGE);
  }
  if (!bytevector_payload(inputs_header, input_bytes, &inputs) ||
      !bytevector_payload(targets_header, target_bytes, &targets)) {
    return status(ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  }
  inputs += input_offset;
  targets += target_offset;
  for (index = 0u; index < count; index++) {
    ((int64_t *)batch->input_view->data)[destination + index] =
        decode_i64_le(inputs + 8u * index);
    ((int64_t *)batch->target_view->data)[destination + index] =
        decode_i64_le(targets + 8u * index);
    batch->mask_data[destination + index] = 1u;
  }
  batch->wrote_any = 1;
  return status(ET_D2_NATIVE_STATUS_OK);
}

int64_t et_d2_batch_write_pair_i64le_span_v1(
    const void *owner, int64_t generation, int64_t destination_value,
    const void *inputs_header, int64_t input_bytes, const void *targets_header,
    int64_t target_bytes, int64_t count_value) {
  et_d2_batch_control *batch = find_batch(owner, generation);
  size_t expected_bytes;
  if (batch == NULL) {
    return status(ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  }
  if (batch->sealed || batch->active_lease_generation != 0) {
    return status(ET_D2_NATIVE_STATUS_INVALID_STATE);
  }
  if (count_value <= 0 || (uint64_t)count_value > SIZE_MAX ||
      !multiply_size((size_t)count_value, sizeof(int64_t), &expected_bytes) ||
      input_bytes < 0 || target_bytes < 0 ||
      (uint64_t)input_bytes != expected_bytes ||
      (uint64_t)target_bytes != expected_bytes) {
    return status(ET_D2_NATIVE_STATUS_RANGE);
  }
  return et_d2_batch_write_pair_i64le_source_span_v1(
      owner, generation, destination_value, inputs_header, input_bytes, 0,
      targets_header, target_bytes, 0, count_value);
}

int64_t et_d2_batch_seal_v1(const void *owner, int64_t generation) {
  et_d2_batch_control *batch = find_batch(owner, generation);
  if (batch == NULL) {
    return status(ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  }
  if (batch->sealed || !batch->wrote_any ||
      batch->active_lease_generation != 0) {
    return status(ET_D2_NATIVE_STATUS_INVALID_STATE);
  }
  batch->sealed = 1;
  return status(ET_D2_NATIVE_STATUS_OK);
}

int64_t et_d2_batch_borrow_begin_v1(const void *owner, int64_t generation) {
  et_d2_dataset_control *dataset = find_dataset(owner);
  et_d2_batch_control *batch = find_batch(owner, generation);
  int64_t lease;
  if (batch == NULL) {
    (void)status(ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
    return 0;
  }
  if (!batch->sealed || batch->active_lease_generation != 0) {
    (void)status(ET_D2_NATIVE_STATUS_INVALID_STATE);
    return 0;
  }
  if (dataset == NULL ||
      !peek_generation(dataset->next_lease_generation, &lease)) {
    (void)status(ET_D2_NATIVE_STATUS_UNSUPPORTED);
    return 0;
  }
  batch->active_lease_generation = lease;
  dataset->next_lease_generation++;
  (void)status(ET_D2_NATIVE_STATUS_OK);
  return lease;
}

static et_d2_batch_control *admit_borrow(const void *owner, int64_t generation,
                                         int64_t lease) {
  et_d2_batch_control *batch = find_batch(owner, generation);
  return batch != NULL && lease > 0 && batch->active_lease_generation == lease
             ? batch
             : NULL;
}

static const et_kernel_tensor_view_v1 *
borrow_view(const void *owner, int64_t generation, int64_t lease, int which) {
  et_d2_batch_control *batch = admit_borrow(owner, generation, lease);
  if (batch == NULL) {
    (void)status(ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
    return NULL;
  }
  (void)status(ET_D2_NATIVE_STATUS_OK);
  if (which == 0) {
    return batch->input_view;
  }
  if (which == 1) {
    return batch->target_view;
  }
  return &batch->mask_view;
}

const et_kernel_tensor_view_v1 *et_d2_batch_borrow_inputs_v1(const void *owner,
                                                             int64_t generation,
                                                             int64_t lease) {
  return borrow_view(owner, generation, lease, 0);
}

const et_kernel_tensor_view_v1 *
et_d2_batch_borrow_targets_v1(const void *owner, int64_t generation,
                              int64_t lease) {
  return borrow_view(owner, generation, lease, 1);
}

const et_kernel_tensor_view_v1 *
et_d2_batch_borrow_loss_mask_v1(const void *owner, int64_t generation,
                                int64_t lease) {
  return borrow_view(owner, generation, lease, 2);
}

int64_t et_d2_batch_borrow_end_v1(const void *owner, int64_t generation,
                                  int64_t lease) {
  et_d2_batch_control *batch = admit_borrow(owner, generation, lease);
  if (batch == NULL) {
    return status(ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  }
  batch->active_lease_generation = 0;
  return status(ET_D2_NATIVE_STATUS_OK);
}

int64_t et_d2_batch_release_preflight_v1(const void *owner,
                                         int64_t generation) {
  et_d2_batch_control *batch = find_batch(owner, generation);
  if (batch == NULL) {
    return status(ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  }
  if (batch->active_lease_generation != 0) {
    return status(ET_D2_NATIVE_STATUS_INVALID_STATE);
  }
  return status(ET_D2_NATIVE_STATUS_OK);
}

int64_t et_d2_batch_release_v1(const void *owner, int64_t generation) {
  et_d2_dataset_control *dataset = find_dataset(owner);
  et_d2_batch_control *batch = find_batch(owner, generation);
  if (dataset == NULL || batch == NULL ||
      batch->active_lease_generation != 0) {
    abort();
  }
  dataset->current_batch = NULL;
  cleanup_control(batch);
  return status(ET_D2_NATIVE_STATUS_OK);
}
