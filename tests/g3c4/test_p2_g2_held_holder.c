#define ET_G3C4_P2_G2_HELD_HOLDER_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN holder_terminal_predecessor_main
#include "test_p2_g2_native_terminal.c"

static void holder_flip_staged(et_g3c4_p2g2_holder *holder) {
  et_a2_kv_cache_transaction_view *view = holder->a2;
  et_a2_kv_cache_transaction_view *temporary = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *mask = NULL;
  et_kernel_error error;
  if (view == NULL) {
    OK(et_a2_kv_cache_transaction_view_begin_v1(
        holder->context->token_frame_transaction, 0u, &temporary, &error));
    view = temporary;
  }
  OK(et_a2_kv_cache_transaction_view_tensors_v1(
      view, &keys, &values, &lengths, &mask, &error));
  CHECK(keys != NULL && keys->byte_length == 64u && keys->data != NULL);
  ((unsigned char *)keys->data)[0] ^= 1u;
  if (temporary != NULL)
    OK(et_a2_kv_cache_transaction_view_end_v1(&temporary, &error));
}

static void holder_case(et_g3c4_model_owner_internal *owner,
                        int phase, int kind, int retry) {
  const int64_t prompt[2] = {7, 11};
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  terminal_raw first, second;
  terminal_ready(owner, prompt, phase == 2, &context, &input, &output,
                 &first, &second);
  void *holder = et_g3c4_private_p2g2_holder_new_v1(kind, phase);
  CHECK(holder != NULL);
  int64_t generation =
      et_g3c4_private_p2g2_holder_generation_v1(holder);
  CHECK(generation > 0);
  CHECK(et_g3c4_private_p2g2_holder_bind_v1(
      holder, generation + 1, context, output) != 0);
  CHECK(et_g3c4_private_p2g2_holder_bind_v1(
      holder, generation, context, input) != 0);
  OK(et_g3c4_private_p2g2_holder_bind_v1(
      holder, generation, context, output));
  CHECK(et_g3c4_private_p2g2_holder_bind_v1(
      holder, generation, context, output) != 0);
  OK(et_g3c4_private_p2g2_holder_begin_capture_v1(
      holder, generation, context, output));
  if (kind == 2)
    CHECK(!et_a2_kv_cache_private_terminal_transaction_witness_v1(
        context->token_frame_transaction, context->cache, phase));
  CHECK(et_g3c4_private_p2g2_holder_begin_capture_v1(
      holder, generation, context, output) != 0);
  CHECK(et_g3c4_private_p2g2_holder_dispose_v1(holder, generation) != 0);
  CHECK(et_g3c4_private_p2g2_holder_witness_v1(
      holder, generation + 1, context, output) != 0);
  CHECK(et_g3c4_private_p2g2_holder_witness_v1(
      holder, generation, input, output) != 0);
  CHECK(et_g3c4_private_p2g2_holder_witness_v1(
      holder, generation, context, input) != 0);
  OK(et_g3c4_private_p2g2_holder_witness_v1(
      holder, generation, context, output));
  et_g3c4_p2g2_holder *private_holder =
      et_g3c4_holder_find(holder, generation);
  CHECK(private_holder != NULL);
  if (kind == 1) {
    int64_t *ids = (int64_t *)et_i64_tensor_test_data_storage_v1(output->ids);
    CHECK(ids != NULL);
    ids[0] ^= 1;
    CHECK(et_g3c4_private_p2g2_holder_witness_v1(
        holder, generation, context, output) != 0);
    ids[0] ^= 1;
  } else {
    holder_flip_staged(private_holder);
    CHECK(et_g3c4_private_p2g2_holder_witness_v1(
        holder, generation, context, output) != 0);
    holder_flip_staged(private_holder);
  }
  OK(et_g3c4_private_p2g2_holder_witness_v1(
      holder, generation, context, output));
  if (retry) {
    const int64_t category = kind == 1 ? ET_I64_TENSOR_ERROR_INVALID_STATE :
                                         ET_KERNEL_ERROR_INVALID_ARGUMENT;
    const int64_t domain = kind == 1 ? ET_G3C4_DOMAIN_I1 : ET_G3C4_DOMAIN_K1;
    const int64_t code = kind == 1 ? ET_I64_TENSOR_CODE_ACTIVE_BORROW :
                                     ET_KERNEL_CODE_PROVIDER_REJECTED;
    CHECK(et_g3c4_private_p2g2_holder_end_after_busy_v1(
        holder, generation, context, output) != 0);
    CHECK(et_g3c4_private_p2g2_holder_abort_attempt_v1(
        holder, generation, context, output, 2) != 0);
    CHECK(et_g3c4_private_p2g2_holder_abort_attempt_v1(
        holder, generation, context, output, 1) == category);
    CHECK(et_g3c4_private_last_error_domain_v1() == domain);
    CHECK(et_g3c4_private_last_error_category_v1() == category);
    CHECK(et_g3c4_private_last_error_code_v1() == code);
    pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
      struct rlimit limit = {0u, 0u};
      (void)setrlimit(RLIMIT_CORE, &limit);
      private_holder->first_error.code++;
      (void)et_g3c4_private_p2g2_holder_end_after_busy_v1(
          holder, generation, context, output);
      _exit(99);
    }
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    CHECK(et_g3c4_private_p2g2_holder_dispose_v1(holder, generation) != 0);
    CHECK(et_g3c4_private_p2g2_holder_end_after_busy_v1(
        holder, generation, context, input) != 0);
    OK(et_g3c4_private_p2g2_holder_witness_v1(
        holder, generation, context, output));
    OK(et_g3c4_private_p2g2_holder_end_after_busy_v1(
        holder, generation, context, output));
    CHECK(et_g3c4_private_p2g2_holder_end_after_busy_v1(
        holder, generation, context, output) != 0);
    OK(et_g3c4_private_p2g2_holder_witness_v1(
        holder, generation, context, output));
    if (kind == 1) {
      et_i64_tensor_error error;
      int64_t altered[2] = {1, 0}, clean[2] = {0, 0};
      OK(et_i64_tensor_copy_from_v1(output->ids, altered, 2u, &error));
      CHECK(et_g3c4_private_p2g2_holder_witness_v1(
          holder, generation, context, output) != 0);
      OK(et_i64_tensor_copy_from_v1(output->ids, clean, 2u, &error));
    } else {
      holder_flip_staged(private_holder);
      CHECK(et_g3c4_private_p2g2_holder_witness_v1(
          holder, generation, context, output) != 0);
      holder_flip_staged(private_holder);
    }
    OK(et_g3c4_private_p2g2_holder_witness_v1(
        holder, generation, context, output));
    OK(et_g3c4_private_p2g2_holder_abort_attempt_v1(
        holder, generation, context, output, 2));
    CHECK(et_g3c4_private_p2g2_holder_abort_attempt_v1(
        holder, generation, context, output, 2) != 0);
    OK(et_g3c4_private_p2g2_holder_dispose_v1(holder, generation));
    CHECK(et_g3c4_private_p2g2_holder_generation_v1(holder) == 0);
    CHECK(et_g3c4_private_p2g2_holder_dispose_v1(holder, generation) != 0);
    check_dead_output(output);
    OK(et_g3c4_private_tensor_release_v1(input));
    OK(et_g3c4_private_generator_close_v1(context));
  } else {
    OK(et_g3c4_private_p2g2_holder_normal_close_v1(
        holder, generation, context, output));
    CHECK(et_g3c4_private_p2g2_holder_normal_close_v1(
        holder, generation, context, output) != 0);
    OK(et_g3c4_private_p2g2_holder_witness_v1(
        holder, generation, context, output));
    OK(et_g3c4_private_p2g2_holder_dispose_v1(holder, generation));
    CHECK(et_g3c4_private_p2g2_holder_witness_v1(
        holder, generation, context, output) != 0);
    OK(et_g3c4_private_p2g2_terminal_commit_v1(
        context, output, &first, 9, phase == 2 ? NULL : &second,
        phase == 2 ? 0 : 9));
    OK(et_g3c4_private_tensor_release_v1(input));
    OK(et_g3c4_private_output_release_v1(output));
    OK(et_g3c4_private_generator_close_v1(context));
  }
}

static void holder_unbound_and_wrong_phase(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  terminal_raw first, second;
  void *unbound = et_g3c4_private_p2g2_holder_new_v1(1, 2);
  CHECK(unbound != NULL);
  int64_t generation = et_g3c4_private_p2g2_holder_generation_v1(unbound);
  OK(et_g3c4_private_p2g2_holder_dispose_v1(unbound, generation));
  CHECK(et_g3c4_private_p2g2_holder_new_v1(0, 2) == NULL);
  CHECK(et_g3c4_private_p2g2_holder_new_v1(3, 2) == NULL);
  CHECK(et_g3c4_private_p2g2_holder_new_v1(1, 4) == NULL);
  terminal_ready(owner, prompt, 1, &context, &input, &output,
                 &first, &second);
  void *wrong = et_g3c4_private_p2g2_holder_new_v1(2, 3);
  CHECK(wrong != NULL);
  CHECK(et_g3c4_private_p2g2_holder_bind_v1(
      unbound, generation, context, output) != 0);
  generation = et_g3c4_private_p2g2_holder_generation_v1(wrong);
  OK(et_g3c4_private_p2g2_holder_bind_v1(wrong, generation, context, output));
  CHECK(et_g3c4_private_p2g2_holder_begin_capture_v1(
      wrong, generation, context, output) != 0);
  CHECK(et_g3c4_private_p2g2_holder_dispose_v1(wrong, generation) != 0);
  close_pair(context, input, output);
  OK(et_g3c4_private_p2g2_holder_dispose_v1(wrong, generation));
}

static void holder_token_exhaustion(void) {
  CHECK(et_g3c4_holder_token_count < ET_G3C4_HOLDER_TOKEN_CAPACITY);
  void *stale = NULL;
  for (size_t used = et_g3c4_holder_token_count;
       used < ET_G3C4_HOLDER_TOKEN_CAPACITY; used++) {
    void *holder = et_g3c4_private_p2g2_holder_new_v1(1, 2);
    CHECK(holder != NULL);
    int64_t generation =
        et_g3c4_private_p2g2_holder_generation_v1(holder);
    CHECK(generation == (int64_t)(used + 1u));
    if (stale != NULL)
      CHECK(et_g3c4_private_p2g2_holder_generation_v1(stale) == 0);
    OK(et_g3c4_private_p2g2_holder_dispose_v1(holder, generation));
    stale = holder;
  }
  CHECK(et_g3c4_private_p2g2_holder_new_v1(1, 2) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INTERNAL);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALLOCATION);
  CHECK(et_g3c4_private_p2g2_holder_generation_v1(stale) == 0);
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  holder_unbound_and_wrong_phase(owner);
  for (int phase = 2; phase <= 3; phase++)
    for (int kind = 1; kind <= 2; kind++) {
      holder_case(owner, phase, kind, 0);
      holder_case(owner, phase, kind, 1);
    }
  holder_token_exhaustion();
  printf("G3-C4 P2/G2 private holder PASS: checks=%zu\n", checks);
  return 0;
}
