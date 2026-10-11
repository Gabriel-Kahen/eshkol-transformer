/* Private test-only inspection of the actual I1 owners returned to Eshkol. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eshkol_transformer/i64_tensor.h"
#include "eshkol_transformer/g3c4_context_internal.h"

static int64_t fault_code;
static int64_t scalar_calls;
static int64_t ids_allocation_cut = -1;

int64_t et_g3c4_p2g2_typed_aggregate_fault_v1(int64_t code) {
  if (code < 0 || code > 7 || fault_code != 0 || ids_allocation_cut >= 0)
    return -1;
  fault_code = code;
  scalar_calls = 0;
  return 0;
}

int64_t et_g3c4_p2g2_typed_aggregate_fault_pending_v1(void) {
  return fault_code != 0 || ids_allocation_cut >= 0;
}

int64_t et_g3c4_p2g2_typed_aggregate_i1_cut_v1(int64_t cut) {
  if (cut < 0 || cut > 6 || fault_code != 0 || ids_allocation_cut >= 0)
    return -1;
  ids_allocation_cut = cut;
  return 0;
}

int64_t __real_et_g3c4_private_p2g2_terminal_snapshot_v1(
    void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_terminal_snapshot_v1(
    void *source, void *destination, int64_t bytes) {
  int64_t cut = fault_code;
  if (cut == 1 || cut == 2) {
    fault_code = 0;
    if (cut == 1)
      return __real_et_g3c4_private_p2g2_terminal_snapshot_v1(
          source, destination, 0);
    return __real_et_g3c4_private_p2g2_terminal_snapshot_v1(
        source, source, bytes);
  }
  return __real_et_g3c4_private_p2g2_terminal_snapshot_v1(
      source, destination, bytes);
}

void *__real_et_g3c4_private_p2g2_terminal_ids_clone_v1(void *);
void *__wrap_et_g3c4_private_p2g2_terminal_ids_clone_v1(void *source) {
  if (fault_code == 3) {
    fault_code = 0;
    return __real_et_g3c4_private_p2g2_terminal_ids_clone_v1(NULL);
  }
  return __real_et_g3c4_private_p2g2_terminal_ids_clone_v1(source);
}

int64_t __real_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
    void *, int64_t, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
    void *clone, int64_t length, void *destination, int64_t bytes) {
  if (fault_code == 4) {
    fault_code = 0;
    return __real_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
        clone, length, destination, 0);
  }
  return __real_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      clone, length, destination, bytes);
}

void *__real_et_g3c4_private_p2g2_typed_ids_create_v1(void *);
void *__wrap_et_g3c4_private_p2g2_typed_ids_create_v1(void *clone) {
  void *result;
  if (fault_code == 5) {
    fault_code = 0;
    return __real_et_g3c4_private_p2g2_typed_ids_create_v1(NULL);
  }
  if (ids_allocation_cut >= 0) {
    const int64_t cut = ids_allocation_cut;
    ids_allocation_cut = -1;
    et_i64_tensor_test_fail_alloc_after_v1((size_t)cut);
    result = __real_et_g3c4_private_p2g2_typed_ids_create_v1(clone);
    et_i64_tensor_test_reset_allocator_v1();
    return result;
  }
  return __real_et_g3c4_private_p2g2_typed_ids_create_v1(clone);
}

void *__real_et_g3c4_private_p2g2_typed_scalar_create_v1(
    int64_t, int64_t);
void *__wrap_et_g3c4_private_p2g2_typed_scalar_create_v1(
    int64_t kind, int64_t value) {
  scalar_calls++;
  if ((fault_code == 6 && scalar_calls == 1) ||
      (fault_code == 7 && scalar_calls == 2)) {
    fault_code = 0;
    return __real_et_g3c4_private_p2g2_typed_scalar_create_v1(99, value);
  }
  return __real_et_g3c4_private_p2g2_typed_scalar_create_v1(kind, value);
}

typedef struct typed_header {
  struct typed_header *registry_next;
  uint64_t magic;
  uint32_t kind;
  uint32_t state;
  uint32_t busy;
} typed_header;

/* Keep this layout equal to et_g3c4_typed_result_internal in the pinned C4
 * owner. The native kind call authenticates the candidate before copying. */
typedef struct typed_owner {
  typed_header transport;
  et_i64_tensor *tensor;
  int64_t length;
  int64_t values[2];
} typed_owner;

static et_i64_tensor_borrow *held_typed_borrow;

int64_t et_g3c4_p2g2_typed_aggregate_lease_v1(
    void *candidate, int64_t active) {
  et_i64_tensor_error error;
  typed_owner owner;
  if (active == 1) {
    if (held_typed_borrow != NULL ||
        et_g3c4_private_p2g2_typed_kind_v1(candidate) != 10)
      return -1;
    memcpy(&owner, candidate, sizeof(owner));
    return et_i64_tensor_borrow_begin_v1(
        owner.tensor, &held_typed_borrow, &error);
  }
  if (active == 0 && held_typed_borrow != NULL)
    return et_i64_tensor_borrow_end_v1(&held_typed_borrow, &error);
  return -1;
}

int64_t et_g3c4_p2g2_typed_aggregate_shape_v1(
    void *candidate, int64_t kind, int64_t length,
    int64_t first, int64_t second) {
  typed_owner owner;
  et_i64_tensor_error error;
  size_t rank = 0, elements = 0, bytes = 0, stride = 0;
  uint64_t extent = 0;
  int64_t actual[2] = {-1, -1};
  if (candidate == NULL || (length != 1 && length != 2) ||
      (kind != 10 && kind != 11 && kind != 12) ||
      et_g3c4_private_p2g2_typed_kind_v1(candidate) != kind)
    return -1;
  memcpy(&owner, candidate, sizeof(owner));
  if (owner.transport.kind != (uint32_t)kind ||
      owner.length != length || owner.tensor == NULL ||
      et_i64_tensor_rank_v1(owner.tensor, &rank, &error) != 0 ||
      et_i64_tensor_shape_at_v1(owner.tensor, 0u, &extent, &error) != 0 ||
      et_i64_tensor_stride_bytes_at_v1(owner.tensor, 0u, &stride, &error) != 0 ||
      et_i64_tensor_element_count_v1(owner.tensor, &elements, &error) != 0 ||
      et_i64_tensor_byte_length_v1(owner.tensor, &bytes, &error) != 0 ||
      rank != 1u || extent != (uint64_t)length ||
      stride != sizeof(int64_t) || elements != (size_t)length ||
      bytes != (size_t)length * sizeof(int64_t) ||
      et_i64_tensor_copy_to_v1(owner.tensor, actual,
                               (size_t)length, &error) != 0 ||
      actual[0] != first || (length == 2 && actual[1] != second))
    return -1;
  return 0;
}
