/* Source-only candidate for the later reviewed native owner gate. */
#define ET_G3C4_CONTEXT_TESTING 1
#define ET_I64_TENSOR_TESTING 1
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#include <stdio.h>

static size_t checks;
static int fail_copy_once;
int32_t __real_et_i64_tensor_copy_from_v1(
    et_i64_tensor *, const int64_t *, size_t, et_i64_tensor_error *);
int32_t __wrap_et_i64_tensor_copy_from_v1(
    et_i64_tensor *tensor, const int64_t *source, size_t count,
    et_i64_tensor_error *error) {
  if (fail_copy_once) {
    fail_copy_once = 0;
    memset(error, 0, sizeof(*error));
    error->category = ET_I64_TENSOR_ERROR_INTERNAL;
    error->code = ET_I64_TENSOR_CODE_PROVIDER_REJECTED;
    return ET_I64_TENSOR_ERROR_INTERNAL;
  }
  return __real_et_i64_tensor_copy_from_v1(tensor, source, count, error);
}
#define CHECK(predicate) do { \
  checks++; \
  if (!(predicate)) { \
    fprintf(stderr, "P2/G2 result owner FAIL %s:%d: %s\n", \
            __FILE__, __LINE__, #predicate); \
    return 1; \
  } \
} while (0)

static int inspect(void *candidate, int64_t count,
                   int64_t value0, int64_t value1) {
  et_g3c4_result_tensor_internal *owner = candidate;
  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;
  int64_t words[2] = {0, 0};
  CHECK(owner != NULL && owner->terminal_result == 1u);
  CHECK(owner->count == count && owner->words[0] == value0 &&
        owner->words[1] == value1);
  CHECK(et_i64_tensor_borrow_begin_v1(owner->tensor, &borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  CHECK(view != NULL && view->rank == 1u && view->shape != NULL &&
        view->shape[0] == (uint64_t)count &&
        view->byte_length == (size_t)count * sizeof(int64_t) &&
        strcmp(view->dtype, "i64") == 0 &&
        strcmp(view->device, "cpu") == 0 &&
        view->layout == ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR);
  memcpy(words, view->data, (size_t)count * sizeof(int64_t));
  CHECK(words[0] == value0 && (count == 1 || words[1] == value1));
  CHECK(et_g3c4_private_result_tensor_release_v1(candidate) ==
        ET_I64_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I1 &&
        et_g3c4_private_last_error_code_v1() ==
            ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(owner->transport.state == ET_G3C4_CONTEXT_LIVE &&
        owner->tensor != NULL);
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
  return 0;
}

int main(void) {
  et_g3c4_result_tensor_internal *owner;
  et_g3c4_result_tensor_internal *other;
  void *rng;
  et_i64_tensor_error error;
  const int64_t altered[2] = {7, 12};
  const int64_t restored[2] = {7, 11};
  uint64_t *shape;
  const int64_t bad[][4] = {
    {5, 0, 7, 0}, {5, 3, 7, 11}, {5, 1, 7, 11},
    {5, 2, -1, 11}, {5, 2, 7, 256}, {6, 2, 2, 0},
    {6, 1, 0, 0}, {6, 1, 2, 1}, {7, 1, 5, 0},
    {7, 1, 4, 1}, {8, 1, 7, 0}
  };
  for (size_t i = 0u; i < sizeof(bad) / sizeof(bad[0]); i++) {
    CHECK(et_g3c4_private_terminal_result_tensor_create_v1(
              bad[i][0], bad[i][1], bad[i][2], bad[i][3]) == NULL);
    CHECK(et_g3c4_private_last_error_category_v1() ==
          ET_G3C4_INVALID_ARGUMENT);
  }
  et_g3c4_context_allocation_limit = 0u;
  CHECK(et_g3c4_private_terminal_result_tensor_create_v1(5, 2, 7, 11)
        == NULL);
  et_g3c4_context_allocation_limit = SIZE_MAX;
  for (size_t cut = 0u; cut <= 4u; cut++) {
    et_i64_tensor_test_fail_alloc_after_v1(cut);
    CHECK(et_g3c4_private_terminal_result_tensor_create_v1(5, 2, 7, 11)
          == NULL);
    CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I1);
  }
  et_i64_tensor_test_reset_allocator_v1();
  fail_copy_once = 1;
  CHECK(et_g3c4_private_terminal_result_tensor_create_v1(5, 2, 7, 11)
        == NULL);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I1);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_I64_TENSOR_CODE_PROVIDER_REJECTED);
  CHECK(fail_copy_once == 0);

  owner = et_g3c4_private_terminal_result_tensor_create_v1(5, 2, 7, 11);
  other = et_g3c4_private_terminal_result_tensor_create_v1(5, 2, 7, 11);
  CHECK(other != NULL && other != owner && other->tensor != owner->tensor);
  CHECK(et_g3c4_private_result_tensor_release_v1(other) == 0);
  rng = et_g3c4_private_result_rng_create_v1(1, 0, 0, 0);
  CHECK(rng != NULL);
  CHECK(et_g3c4_private_result_tensor_release_v1(rng) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_rng_release_v1(rng) == 0);
  CHECK(inspect(owner, 2, 7, 11) == 0);
  CHECK(et_g3c4_private_result_tensor_release_v1((void *)(uintptr_t)1u)
        == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_i64_tensor_copy_from_v1(owner->tensor, altered, 2u, &error) == 0);
  CHECK(et_g3c4_private_result_tensor_release_v1(owner) == ET_G3C4_INTERNAL);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_C4);
  CHECK(owner->transport.state == ET_G3C4_CONTEXT_LIVE);
  CHECK(et_i64_tensor_copy_from_v1(owner->tensor, restored, 2u, &error) == 0);
  owner->words[1] = 12;
  CHECK(et_g3c4_private_result_tensor_release_v1(owner) == ET_G3C4_INTERNAL);
  owner->words[1] = 11;
  shape = (uint64_t *)et_i64_tensor_test_shape_storage_v1(owner->tensor);
  CHECK(shape != NULL);
  shape[0] = 1;
  CHECK(et_g3c4_private_result_tensor_release_v1(owner) != 0);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I1);
  shape[0] = 2;
  CHECK(et_g3c4_private_result_tensor_release_v1(owner) == 0);
  CHECK(et_g3c4_private_result_tensor_release_v1(owner) == 0);

  owner = et_g3c4_private_terminal_result_tensor_create_v1(5, 1, 255, 0);
  CHECK(inspect(owner, 1, 255, 0) == 0);
  owner->words[1] = 1;
  CHECK(et_g3c4_private_result_tensor_release_v1(owner) == ET_G3C4_INTERNAL);
  owner->words[1] = 0;
  CHECK(et_g3c4_private_result_tensor_release_v1(owner) == 0);
  owner = et_g3c4_private_terminal_result_tensor_create_v1(6, 1, 2, 0);
  CHECK(inspect(owner, 1, 2, 0) == 0);
  CHECK(et_g3c4_private_result_tensor_release_v1(owner) == 0);
  owner = et_g3c4_private_terminal_result_tensor_create_v1(7, 1, 4, 0);
  CHECK(inspect(owner, 1, 4, 0) == 0);
  CHECK(et_g3c4_private_result_tensor_release_v1(owner) == 0);
  owner = et_g3c4_private_result_tensor_create_v1(5, 7);
  CHECK(owner != NULL && owner->terminal_result == 0u && owner->value == 7);
  CHECK(et_g3c4_private_result_tensor_release_v1(owner) == 0);
  printf("P2/G2 result owner PASS: checks=%zu\n", checks);
  return 0;
}
