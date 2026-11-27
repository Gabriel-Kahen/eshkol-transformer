#define ET_G3C4_P2_G2_FIRST_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_CALL_I1_LEASE_PRIVATE 1
#define ET_G3C4_P2_G1_PENDING_TEST_MAIN et_g3c4_p2g1_lease_predecessor_main
#include "test_p2_g1_pending.c"
#undef ET_G3C4_P2_G1_PENDING_TEST_MAIN

static et_g3c4_model_owner_internal *make_foreign_owner(void) {
  static unsigned char foreign_identities[14];
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(owner != NULL);
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_bind_v1(owner, index, &foreign_identities[index]));
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, index));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  return owner;
}

static et_g3c4_context_internal *make_generator(
    et_g3c4_model_owner_internal *owner, int64_t budget) {
  et_g3c4_context_internal *context = et_g3c4_private_generator_seed_v1(
      owner, 7, 0, INT64_C(0x3f800000), 256,
      INT64_C(0x3f800000), budget, -1);
  CHECK(context != NULL);
  return context;
}

static et_g3c4_output_internal *reserve(et_g3c4_context_internal *context,
                                      int64_t budget) {
  OK(et_g3c4_private_call_acquire_v1(context, 2, budget));
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(output != NULL);
  return output;
}

static void check_real_borrow(const et_i64_tensor_borrow *borrow) {
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;
  CHECK(borrow != NULL);
  OK(et_i64_tensor_borrow_view_v1(borrow, &view, &error));
  CHECK(view != NULL && strcmp(view->dtype, "i64") == 0);
  CHECK(view->rank == 1u && view->shape[0] == 2u && view->byte_length == 16u);
  CHECK(memcmp(view->data, (const int64_t[2]){0, 0}, 16u) == 0);
}

static void check_abort_blocked(et_g3c4_context_internal *context,
                                et_g3c4_output_internal *output) {
  unsigned char saved_context[sizeof(*context)], saved_output[sizeof(*output)];
  memcpy(saved_context, context, sizeof(saved_context));
  memcpy(saved_output, output, sizeof(saved_output));
  CHECK(et_g3c4_private_call_abort_v1(context) ==
        ET_I64_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(memcmp(context, saved_context, sizeof(saved_context)) == 0);
  CHECK(memcmp(output, saved_output, sizeof(saved_output)) == 0);
  check_real_borrow(context->p2g2_i1_lease.borrow);
}

static void abort_and_check_drain(et_g3c4_context_internal *context,
                                  et_g3c4_output_internal *output) {
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(et_g3c4_admit_idle_call(context) == context);
  CHECK(context->p2g2_i1_lease.output == NULL &&
        context->p2g2_i1_lease.ids == NULL &&
        context->p2g2_i1_lease.borrow == NULL &&
        context->p2g2_i1_lease.used == 0u);
  CHECK(output->transport.state == ET_G3C4_CONTEXT_DEAD &&
        output->parent_ctx == NULL && output->ids == NULL);
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  et_g3c4_context_internal *context = make_generator(owner, 2);
  et_g3c4_context_internal *foreign = make_generator(make_foreign_owner(), 2);
  et_g3c4_context_internal *budget_one = make_generator(owner, 1);
  cache_snapshot cache;
  binding_snapshot binding;
  int64_t rng[4];
  snapshot_cache(context->cache, &cache);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));

  et_g3c4_output_internal *short_output = reserve(budget_one, 1);
  CHECK(et_g3c4_p2g2_call_i1_lease_begin(budget_one, short_output) != 0);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_LIFECYCLE);
  abort_and_check_drain(budget_one, short_output);
  OK(et_g3c4_private_generator_close_v1(budget_one));
  CHECK(et_g3c4_p2g2_call_i1_lease_begin(context, short_output) != 0);

  et_g3c4_output_internal *output = reserve(context, 2);
  et_g3c4_output_internal *foreign_output = reserve(foreign, 2);
  et_g3c4_output_internal equal_copy = *output;
  et_i64_tensor *ids = output->ids;
  CHECK(context->p2g2_i1_lease.output == output &&
        context->p2g2_i1_lease.ids == ids);
  CHECK(ids != foreign_output->ids);
  CHECK(et_g3c4_p2g2_call_i1_lease_begin(context, &equal_copy) != 0);
  CHECK(et_g3c4_p2g2_call_i1_lease_begin(context, foreign_output) != 0);
  output->ids = foreign_output->ids;
  CHECK(et_g3c4_p2g2_call_i1_lease_begin(context, output) != 0);
  output->ids = ids;

  /* A genuine unrelated borrower is a negative control, never our ticket. */
  et_i64_tensor_borrow *unmanaged = NULL;
  et_i64_tensor_error error;
  OK(et_i64_tensor_borrow_begin_v1(ids, &unmanaged, &error));
  CHECK(et_g3c4_p2g2_call_i1_lease_end(context, output) != 0);
  CHECK(et_g3c4_p2g2_call_i1_lease_begin(context, output) != 0);
  check_real_borrow(unmanaged);
  CHECK(context->p2g2_i1_lease.used == 0u &&
        context->p2g2_i1_lease.borrow == NULL);
  OK(et_i64_tensor_borrow_end_v1(&unmanaged, &error));

  et_i64_tensor_test_fail_alloc_after_v1(0u);
  CHECK(et_g3c4_p2g2_call_i1_lease_begin(context, output) != 0);
  et_i64_tensor_test_reset_allocator_v1();
  CHECK(context->p2g2_i1_lease.used == 0u &&
        context->p2g2_i1_lease.borrow == NULL);
  OK(et_g3c4_p2g2_call_i1_lease_begin(context, output));
  OK(et_g3c4_p2g2_call_i1_lease_begin(foreign, foreign_output));
  et_i64_tensor_borrow *held = context->p2g2_i1_lease.borrow;
  et_i64_tensor_borrow *foreign_held = foreign->p2g2_i1_lease.borrow;
  CHECK(held != foreign_held);
  check_real_borrow(held);
  check_real_borrow(foreign_held);
  CHECK(et_g3c4_p2g2_call_i1_lease_begin(context, output) != 0);

  CHECK(et_g3c4_p2g2_call_i1_lease_end(context, &equal_copy) != 0);
  CHECK(et_g3c4_p2g2_call_i1_lease_end(foreign, output) != 0);
  CHECK(et_g3c4_p2g2_call_i1_lease_end(context, foreign_output) != 0);
  output->ids = foreign_output->ids;
  CHECK(et_g3c4_p2g2_call_i1_lease_end(context, output) != 0);
  CHECK(et_g3c4_private_call_abort_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_output_release_v1(output) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  output->ids = ids;
  CHECK(context->p2g2_i1_lease.borrow == held &&
        foreign->p2g2_i1_lease.borrow == foreign_held);
  check_abort_blocked(context, output);
  check_preserved(context, &cache, &binding, rng);

  OK(et_g3c4_p2g2_call_i1_lease_end(context, output));
  CHECK(context->p2g2_i1_lease.borrow == NULL &&
        foreign->p2g2_i1_lease.borrow == foreign_held);
  CHECK(et_g3c4_p2g2_call_i1_lease_end(context, output) != 0);
  CHECK(et_g3c4_p2g2_call_i1_lease_begin(context, output) != 0);
  abort_and_check_drain(context, output);
  check_preserved(context, &cache, &binding, rng);

  et_g3c4_output_internal *next = reserve(context, 2);
  CHECK(next != output);
  OK(et_g3c4_p2g2_call_i1_lease_begin(context, next));
  CHECK(et_g3c4_p2g2_call_i1_lease_end(context, output) != 0);
  check_abort_blocked(context, next);
  OK(et_g3c4_p2g2_call_i1_lease_end(context, next));
  abort_and_check_drain(context, next);

  et_g3c4_output_internal *retired = reserve(context, 2);
  OK(et_g3c4_private_output_release_v1(retired));
  et_g3c4_output_internal *replacement =
      et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(replacement != NULL && replacement != retired);
  CHECK(et_g3c4_p2g2_call_i1_lease_begin(context, retired) != 0);
  CHECK(et_g3c4_p2g2_call_i1_lease_begin(context, replacement) != 0);
  CHECK(context->p2g2_i1_lease.output == retired &&
        context->p2g2_i1_lease.borrow == NULL);
  abort_and_check_drain(context, replacement);

  check_abort_blocked(foreign, foreign_output);
  OK(et_g3c4_p2g2_call_i1_lease_end(foreign, foreign_output));
  abort_and_check_drain(foreign, foreign_output);
  OK(et_g3c4_private_generator_close_v1(foreign));
  OK(et_g3c4_private_generator_close_v1(context));
  printf("G3-C4 P2/G2 call I1 lease PASS: checks=%zu\n", checks);
  return 0;
}
