#define ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE 1
#define ET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE 1
#define ET_G3C4_P2_G2_TYPED_RESULT_PRIVATE 1
#define ET_G3C4_P2_G2_CLONE_TEST_MAIN terminal_clone_test_main
#include "test_p2_g2_terminal_ids_clone.c"

static size_t typed_live_count(void) {
  size_t count = 0;
  for (const et_g3c4_transport_header_internal *record =
           et_g3c4_transport_registry; record != NULL;
       record = record->registry_next)
    if (record->kind >= 10u && record->kind <= 12u &&
        record->state == ET_G3C4_CONTEXT_LIVE) count++;
  return count;
}

static void typed_case(et_g3c4_model_owner_internal *model, int first_eos) {
  int64_t expected[2] = {0, 0}, value = -99;
  et_g3c4_output_internal *output =
      published_terminal(model, first_eos, expected);
  void *clone = et_g3c4_private_p2g2_terminal_ids_clone_v1(output);
  CHECK(clone != NULL);
  const size_t before = typed_live_count();
  et_g3c4_context_allocation_limit =
      et_g3c4_context_successful_allocations;
  CHECK(et_g3c4_private_p2g2_typed_ids_create_v1(clone) == NULL);
  CHECK(typed_live_count() == before);
  et_g3c4_context_allocation_limit = SIZE_MAX;
  for (size_t cut = 0; cut < 2u; cut++) {
    et_i64_tensor_test_fail_alloc_after_v1(cut);
    CHECK(et_g3c4_private_p2g2_typed_ids_create_v1(clone) == NULL);
    et_i64_tensor_test_reset_allocator_v1();
    CHECK(typed_live_count() == before);
  }
  et_i64_tensor_error error;
  et_i64_tensor_borrow *lease = NULL;
  OK(et_i64_tensor_borrow_begin_v1(
      ((clone_control *)clone)->ids, &lease, &error));
  CHECK(et_g3c4_private_p2g2_typed_ids_create_v1(clone) == NULL);
  CHECK(typed_live_count() == before);
  OK(et_i64_tensor_borrow_end_v1(&lease, &error));
  void *ids = et_g3c4_private_p2g2_typed_ids_create_v1(clone);
  void *ids_again = et_g3c4_private_p2g2_typed_ids_create_v1(clone);
  const int64_t length = first_eos ? 1 : 2;
  void *lengths = et_g3c4_private_p2g2_typed_scalar_create_v1(11, length);
  void *cache = et_g3c4_private_p2g2_typed_scalar_create_v1(12, 2 + length);
  CHECK(ids && ids_again && lengths && cache && ids != ids_again);
  uint64_t extent = 0;
  OK(et_i64_tensor_shape_at_v1(
      ((et_g3c4_typed_result_internal *)ids)->tensor, 0u,
      &extent, &error));
  CHECK(extent == (uint64_t)length);
  OK(et_i64_tensor_shape_at_v1(
      ((et_g3c4_typed_result_internal *)lengths)->tensor, 0u,
      &extent, &error));
  CHECK(extent == 1u);
  CHECK(et_g3c4_private_p2g2_typed_length_v1(ids) == length);
  CHECK(et_g3c4_private_p2g2_typed_length_v1(lengths) == 1);
  CHECK(et_g3c4_private_p2g2_typed_length_v1(cache) == 1);
  for (int64_t index = 0; index < length; index++) {
    OK(et_g3c4_private_p2g2_typed_element_v1(ids, index, &value));
    CHECK(value == expected[index]);
  }
  OK(et_g3c4_private_p2g2_typed_element_v1(lengths, 0, &value));
  CHECK(value == length);
  OK(et_g3c4_private_p2g2_typed_element_v1(cache, 0, &value));
  CHECK(value == 2 + length);
  value = -99;
  CHECK(et_g3c4_private_p2g2_typed_element_v1(ids, length, &value) != 0);
  CHECK(value == -99);
  CHECK(et_g3c4_private_p2g2_typed_element_v1(ids, 0,
        &((et_g3c4_typed_result_internal *)ids)->length) != 0);
  CHECK(et_g3c4_private_p2g2_typed_length_v1(clone) == -1);
  CHECK(et_g3c4_private_p2g2_typed_release_v1(clone) != 0);
  CHECK(et_g3c4_private_result_tensor_release_v1(ids) != 0);
  CHECK(et_g3c4_private_p2g2_typed_scalar_create_v1(11, 3) == NULL);
  CHECK(et_g3c4_private_p2g2_typed_scalar_create_v1(12, 2) == NULL);
  CHECK(et_g3c4_private_p2g2_typed_scalar_create_v1(10, expected[0]) == NULL);
  uint64_t *shape = (uint64_t *)et_i64_tensor_test_shape_storage_v1(
      ((et_g3c4_typed_result_internal *)ids)->tensor);
  int64_t *data = (int64_t *)et_i64_tensor_test_data_storage_v1(
      ((et_g3c4_typed_result_internal *)ids)->tensor);
  CHECK(shape != NULL && data != NULL);
  shape[0] = 3u;
  CHECK(et_g3c4_private_p2g2_typed_length_v1(ids) == -1);
  CHECK(et_g3c4_private_p2g2_typed_release_v1(ids) != 0);
  shape[0] = (uint64_t)length;
  data[0] = 256;
  CHECK(et_g3c4_private_p2g2_typed_element_v1(ids, 0, &value) != 0);
  CHECK(et_g3c4_private_p2g2_typed_release_v1(ids) != 0);
  data[0] = expected[0];
  OK(et_i64_tensor_borrow_begin_v1(
      ((et_g3c4_typed_result_internal *)ids)->tensor, &lease, &error));
  CHECK(et_g3c4_private_p2g2_typed_release_v1(ids) != 0);
  CHECK(et_g3c4_private_p2g2_typed_length_v1(ids) == -1);
  OK(et_i64_tensor_borrow_end_v1(&lease, &error));
  OK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(clone));
  OK(et_g3c4_private_output_release_v1(output));
  OK(et_g3c4_private_p2g2_typed_element_v1(ids, 0, &value));
  CHECK(value == expected[0]);
  OK(et_g3c4_private_p2g2_typed_release_v1(ids));
  OK(et_g3c4_private_p2g2_typed_release_v1(ids));
  CHECK(et_g3c4_private_p2g2_typed_element_v1(ids_again, 0,
        &((et_g3c4_typed_result_internal *)ids)->length) != 0);
  CHECK(et_g3c4_private_p2g2_typed_length_v1(ids) == -1);
  OK(et_g3c4_private_p2g2_typed_element_v1(ids_again, 0, &value));
  CHECK(value == expected[0]);
  OK(et_g3c4_private_p2g2_typed_release_v1(ids_again));
  OK(et_g3c4_private_p2g2_typed_release_v1(lengths));
  OK(et_g3c4_private_p2g2_typed_release_v1(cache));
}

int main(void) {
  et_g3c4_model_owner_internal *model = create_owner();
  typed_case(model, 1);
  typed_case(model, 0);
  printf("G3-C4 P2/G2 typed I1 PASS: checks=%zu\n", checks);
  return 0;
}
