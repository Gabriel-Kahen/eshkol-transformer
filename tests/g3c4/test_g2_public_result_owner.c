#define ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE 1
#define ET_G3C4_P2_G1_COPYOUT_PRIVATE 1
#define ET_G3C4_P2_G1_COORDINATOR_PRIVATE 1
#define ET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE 1
#define ET_G3C4_P2_G1_PUBLIC_RESULT_TESTING 1
#define ET_G3C4_P2_G2_PUBLIC_RESULT_PRIVATE 1
#define ET_I64_TENSOR_TESTING 1
#include "p2_g1_shell_native.c"

static size_t g2_checks;
#define G2_CHECK(condition) do { \
  g2_checks++; \
  if (!(condition)) { \
    fprintf(stderr, "G2 result owner failure %s:%d: %s\n", \
            __FILE__, __LINE__, #condition); \
    return 1; \
  } \
} while (0)

static size_t result_registry_count(void) {
  size_t count = 0;
  for (et_g3c4_transport_header_internal *head = et_g3c4_transport_registry;
       head != NULL; head = head->registry_next)
    if (head->kind >= ET_G3C4_IDS_KIND &&
        head->kind <= ET_G3C4_CACHE_LENGTHS_KIND) count++;
  return count;
}

static int exact_owner(void *candidate, int64_t kind, int64_t count,
                       int64_t first, int64_t second) {
  et_g3c4_result_tensor_internal *owner =
      et_g3c4_admit_result(candidate, 0);
  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_error error;
  const et_kernel_tensor_view_v1 *view = NULL;
  int64_t copied[2] = {-1, -1};
  if (owner == NULL || owner->transport.kind != (uint32_t)kind ||
      owner->logical_count != count || owner->value != first ||
      owner->second_value != second ||
      et_i64_tensor_borrow_begin_v1(owner->tensor, &borrow, &error) != 0 ||
      et_i64_tensor_borrow_view_v1(borrow, &view, &error) != 0)
    return 0;
  int valid = view != NULL && view->rank == 1u && view->shape != NULL &&
      view->shape[0] == (uint64_t)count &&
      view->byte_length == (size_t)count * sizeof(int64_t) &&
      strcmp(view->dtype, "i64") == 0 && strcmp(view->device, "cpu") == 0;
  if (valid) {
    memcpy(copied, view->data, (size_t)count * sizeof(int64_t));
    valid = copied[0] == first && (count == 1 || copied[1] == second);
  }
  if (et_i64_tensor_borrow_end_v1(&borrow, &error) != 0) abort();
  return valid;
}

int main(void) {
  void *g1 = et_g3c4_private_result_tensor_create_rank1_v1(5, 1, 17, 0);
  void *g2 = et_g3c4_private_result_tensor_create_rank1_v1(5, 2, 19, 23);
  void *length = et_g3c4_private_result_tensor_create_rank1_v1(6, 1, 2, 0);
  void *cache = et_g3c4_private_result_tensor_create_rank1_v1(7, 1, 4, 0);
  G2_CHECK(g1 && g2 && length && cache);
  G2_CHECK(exact_owner(g1, 5, 1, 17, 0));
  G2_CHECK(exact_owner(g2, 5, 2, 19, 23));
  G2_CHECK(exact_owner(length, 6, 1, 2, 0));
  G2_CHECK(exact_owner(cache, 7, 1, 4, 0));
  G2_CHECK(et_g3c4_private_result_tensor_inspect_rank1_v1(
               g2, 5, 2, 19, 23) == 0);
  G2_CHECK(et_g3c4_private_result_tensor_inspect_rank1_v1(
               g2, 5, 2, 19, 24) == ET_G3C4_INVALID_STATE);
  G2_CHECK(et_g3c4_private_result_tensor_inspect_rank1_v1(
               g2, 6, 2, 19, 23) == ET_G3C4_INVALID_STATE);
  G2_CHECK(et_g3c4_private_result_tensor_inspect_rank1_v1(
               &g2, 5, 2, 19, 23) == ET_G3C4_INVALID_ARGUMENT);
  uint64_t *shape = (uint64_t *)et_i64_tensor_test_shape_storage_v1(
      ((et_g3c4_result_tensor_internal *)g2)->tensor);
  G2_CHECK(shape != NULL);
  shape[0] = 1u;
  G2_CHECK(et_g3c4_private_result_tensor_inspect_rank1_v1(
               g2, 5, 2, 19, 23) == ET_G3C4_INVALID_STATE);
  shape[0] = 2u;
  G2_CHECK(et_g3c4_private_result_tensor_inspect_rank1_v1(
               g2, 5, 2, 19, 23) == 0);
  int64_t *data = (int64_t *)et_i64_tensor_test_data_storage_v1(
      ((et_g3c4_result_tensor_internal *)g2)->tensor);
  G2_CHECK(data != NULL);
  data[1] = 24;
  G2_CHECK(et_g3c4_private_result_tensor_inspect_rank1_v1(
               g2, 5, 2, 19, 23) == ET_G3C4_INVALID_STATE);
  data[1] = 23;
  G2_CHECK(et_g3c4_private_result_tensor_inspect_rank1_v1(
               g2, 5, 2, 19, 23) == 0);
  G2_CHECK(et_g3c4_private_result_tensor_create_rank1_v1(5, 1, 17, 23) == NULL);
  G2_CHECK(et_g3c4_private_result_tensor_create_rank1_v1(5, 3, 17, 23) == NULL);
  G2_CHECK(et_g3c4_private_result_tensor_create_rank1_v1(5, 2, 17, 256) == NULL);
  G2_CHECK(et_g3c4_private_result_tensor_create_rank1_v1(6, 2, 2, 0) == NULL);
  G2_CHECK(et_g3c4_private_result_tensor_create_rank1_v1(7, 1, 2, 0) == NULL);
  G2_CHECK(et_g3c4_private_result_tensor_create_v1(6, 2) == NULL);
  G2_CHECK(et_g3c4_private_result_tensor_create_v1(7, 4) == NULL);
  for (size_t cut = 0; cut < 8u; cut++) {
    size_t before = result_registry_count();
    et_i64_tensor_test_fail_alloc_after_v1(cut);
    void *attempt = et_g3c4_private_result_tensor_create_rank1_v1(
        5, 2, 19, 23);
    et_i64_tensor_test_reset_allocator_v1();
    if (attempt != NULL) {
      G2_CHECK(exact_owner(attempt, 5, 2, 19, 23));
      G2_CHECK(et_g3c4_private_result_tensor_release_v1(attempt) == 0);
    } else G2_CHECK(result_registry_count() == before);
  }
  size_t before_copy_cut = result_registry_count();
  G2_CHECK(et_g3c4_result_test_copy_cut_v1(1) == 0);
  G2_CHECK(et_g3c4_private_result_tensor_create_rank1_v1(
               5, 2, 19, 23) == NULL);
  G2_CHECK(et_g3c4_result_test_copy_consumed_v1() == 1);
  G2_CHECK(result_registry_count() == before_copy_cut);
  G2_CHECK(g1 != g2 && g2 != length && length != cache);
  et_i64_tensor_error error;
  et_i64_tensor_borrow *held = NULL;
  G2_CHECK(et_i64_tensor_borrow_begin_v1(
               ((et_g3c4_result_tensor_internal *)g2)->tensor,
               &held, &error) == 0);
  G2_CHECK(et_g3c4_private_result_tensor_release_v1(g2) != 0);
  G2_CHECK(((et_g3c4_result_tensor_internal *)g2)->transport.state ==
           ET_G3C4_CONTEXT_LIVE);
  G2_CHECK(et_i64_tensor_borrow_end_v1(&held, &error) == 0);
  G2_CHECK(exact_owner(g2, 5, 2, 19, 23));
  G2_CHECK(et_g3c4_private_result_tensor_release_v1(g1) == 0);
  G2_CHECK(et_g3c4_private_result_tensor_release_v1(g1) == 0);
  G2_CHECK(et_g3c4_private_result_tensor_inspect_rank1_v1(
               g1, 5, 1, 17, 0) == ET_G3C4_INVALID_STATE);
  G2_CHECK(exact_owner(g2, 5, 2, 19, 23));
  G2_CHECK(et_g3c4_private_result_tensor_release_v1(g2) == 0);
  G2_CHECK(et_g3c4_private_result_tensor_release_v1(length) == 0);
  G2_CHECK(et_g3c4_private_result_tensor_release_v1(cache) == 0);
  printf("G3-C4 G2 result owner PASS: checks=%zu\n", g2_checks);
  return 0;
}
