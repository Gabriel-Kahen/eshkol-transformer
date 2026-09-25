#define ET_G3C4_LOGITS_RESERVATION_PRIVATE 1
#define ET_G3C4_OUTPUT_TEXT_TEST_MAIN et_g3c4_output_text_predecessor_main
#include "test_output_text.c"
#undef ET_G3C4_OUTPUT_TEXT_TEST_MAIN

static size_t logits_count(void) {
  size_t count = 0u;
  et_g3c4_transport_header_internal *header;
  for (header = et_g3c4_transport_registry;
       header != NULL; header = header->registry_next)
    if (header->kind == ET_G3C4_LOGITS_KIND) count++;
  return count;
}

static void check_logits(
    et_g3c4_logits_internal *logits,
    et_g3c4_context_internal *context) {
  size_t rank = 0u;
  uint64_t rows = 0u, columns = 0u;
  uint32_t bits[256];
  et_f32_tensor_error error;
  CHECK(logits->transport.magic == ET_G3C4_LOGITS_MAGIC);
  CHECK(logits->transport.kind == ET_G3C4_LOGITS_KIND);
  CHECK(logits->transport.state == 0u);
  CHECK(logits->parent_ctx == context);
  CHECK(logits->tensor != NULL);
  OK(et_f32_tensor_rank_v1(logits->tensor, &rank, &error));
  OK(et_f32_tensor_shape_at_v1(logits->tensor, 0u, &rows, &error));
  OK(et_f32_tensor_shape_at_v1(logits->tensor, 1u, &columns, &error));
  CHECK(rank == 2u && rows == 1u && columns == 256u);
  OK(et_f32_tensor_copy_bits_to_v1(logits->tensor, bits, 256u, &error));
  for (size_t i = 0u; i < 256u; i++) CHECK(bits[i] == 0u);
}

static void check_dead_logits(et_g3c4_logits_internal *logits) {
  CHECK(logits->transport.state == ET_G3C4_CONTEXT_DEAD);
  CHECK(logits->parent_ctx == NULL);
  CHECK(logits->tensor == NULL);
}

static void manual_abort(
    et_g3c4_model_owner_internal *owner, int64_t kind) {
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_logits_internal *logits;
  et_f32_tensor_borrow *borrow = NULL;
  et_f32_tensor_error error;
  const et_kernel_tensor_view_v1 *view = NULL;
  const size_t before = logits_count();
  OK(et_g3c4_private_call_acquire_v1(context, kind, 0));
  logits = et_g3c4_private_logits_reserve_v1(context);
  CHECK(logits != NULL && logits_count() == before + 1u);
  check_logits(logits, context);
  CHECK(et_g3c4_private_logits_reserve_v1(context) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_call_finish_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_output_release_v1(logits) == ET_G3C4_INVALID_ARGUMENT);
  OK(et_f32_tensor_borrow_begin_v1(logits->tensor, &borrow, &error));
  OK(et_f32_tensor_borrow_view_v1(borrow, &view, &error));
  CHECK(view->rank == 2u && view->shape[0] == 1u && view->shape[1] == 256u);
  CHECK(et_g3c4_private_tensor_release_v1(logits) != 0);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I2);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I2);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  check_logits(logits, context);
  CHECK(ET_G3C4_CONTEXT_BUSY(context) == ET_G3C4_CALL_ACTIVE);
  OK(et_f32_tensor_borrow_end_v1(&borrow, &error));
  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_logits(logits);
  OK(et_g3c4_private_tensor_release_v1(logits));
  CHECK(et_g3c4_private_logits_reserve_v1(context) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_generator_close_v1(context));
}

static void explicit_release(et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context = create_generator(owner);
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  et_g3c4_logits_internal *logits =
      et_g3c4_private_logits_reserve_v1(context);
  CHECK(logits != NULL);
  CHECK(et_g3c4_private_tensor_release_v1(context) == ET_G3C4_INVALID_ARGUMENT);
  OK(et_g3c4_private_tensor_release_v1(logits));
  check_dead_logits(logits);
  OK(et_g3c4_private_tensor_release_v1(logits));
  OK(et_g3c4_private_call_prepare_end_v1(context));
  OK(et_g3c4_private_call_finish_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void logits_allocation_cuts(et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context = create_generator(owner);
  size_t before;
  OK(et_g3c4_private_call_acquire_v1(context, 1, 0));
  before = logits_count();
  et_g3c4_context_allocation_limit = et_g3c4_context_successful_allocations;
  CHECK(et_g3c4_private_logits_reserve_v1(context) == NULL);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALLOCATION);
  CHECK(logits_count() == before);
  et_g3c4_context_allocation_limit = SIZE_MAX;
  for (size_t cut = 0u; cut < 4u; cut++) {
    et_f32_tensor_test_fail_alloc_after_v1(cut);
    CHECK(et_g3c4_private_logits_reserve_v1(context) == NULL);
    CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I2);
    CHECK(et_g3c4_private_last_error_code_v1() ==
          ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    CHECK(logits_count() == before);
    et_f32_tensor_test_reset_allocator_v1();
  }
  et_g3c4_logits_internal *logits =
      et_g3c4_private_logits_reserve_v1(context);
  CHECK(logits != NULL && logits_count() == before + 1u);
  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_logits(logits);
  OK(et_g3c4_private_generator_close_v1(context));
}

static void generate_rejects(et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context = create_generator(owner);
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  CHECK(et_g3c4_private_logits_reserve_v1(context) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  CHECK(et_g3c4_private_logits_reserve_v1(&checks) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_ARGUMENT);
  manual_abort(owner, 0);
  manual_abort(owner, 1);
  explicit_release(owner);
  logits_allocation_cuts(owner);
  generate_rejects(owner);
  printf("G3-C4 manual logits reservation PASS: checks=%zu routes=2 f32-cuts=4 owner-cuts=1 borrow-cuts=2\n", checks);
  return 0;
}
