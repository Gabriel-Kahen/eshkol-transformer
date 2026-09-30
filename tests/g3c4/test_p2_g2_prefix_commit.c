#define ET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE 1
#define ET_G3C4_P2_G2_FIRST_FRAME_TEST_MAIN et_g3c4_p2g2_first_frame_main
#include "test_p2_g2_first_frame.c"
#undef ET_G3C4_P2_G2_FIRST_FRAME_TEST_MAIN
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static int fail_prefix_commit;
int32_t __real_et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **, et_kernel_error *);
int32_t __wrap_et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **slot, et_kernel_error *error) {
  if (fail_prefix_commit) return -1;
  return __real_et_a2_kv_cache_transaction_commit_v1(slot, error);
}

static void pending_clean(et_g3c4_output_internal *output) {
  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;
  CHECK(output->transport.state == 0u && output->generated_length == 2);
  CHECK(output->length == 0 && output->cache_length == 0);
  CHECK(output->numeric_ready == 0u && output->ids_copied == 0u &&
        output->text_ready == 0u);
  OK(et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &error));
  OK(et_i64_tensor_borrow_view_v1(borrow, &view, &error));
  CHECK(view->rank == 1u && view->shape[0] == 2u);
  CHECK(memcmp(view->data, (const int64_t[2]){0, 0},
               2u * sizeof(int64_t)) == 0);
  OK(et_i64_tensor_borrow_end_v1(&borrow, &error));
}

static void prefix_case(et_g3c4_model_owner_internal *owner,
                        const int64_t prompt[2], int categorical) {
  et_g3c4_context_internal *context = p2g2_generator(owner, categorical);
  et_g3c4_input_internal *input = create_prompt(prompt, 2);
  et_g3c4_output_internal *output;
  float last[256], next[256], reference_logits[256];
  int64_t token = -1, before_rng[4], after_rng[4];
  int64_t reference_ids[3] = {prompt[0], prompt[1], 0};
  raw_carrier raw = {1, {0}};
  cache_snapshot committed, after, reference_cache;
  et_kernel_error cache_error;
  et_i64_tensor_error ids_error;

  memcpy(before_rng, context->generator_rng_words, sizeof(before_rng));
  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, 2));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 2));
  output = et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, last));
  OK(et_g3c4_private_token_frame_begin_last_v1(context, last, &token));
  memcpy(after_rng, context->token_frame_successor, sizeof(after_rng));
  print_oracle_case(categorical ? "categorical" : "greedy",
                    last, token, before_rng, after_rng);
  OK(et_g3c4_private_token_forward_v1(context, token, next));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  pending_clean(output);
  raw.bytes[0] = (unsigned char)token;

  context->generator_policy[5] = token;
  CHECK(et_g3c4_private_p2g2_prefix_commit_v1(context, output, &raw) ==
        ET_G3C4_INVALID_STATE);
  context->generator_policy[5] = -1;

  raw.length = 2;
  CHECK(et_g3c4_private_p2g2_prefix_commit_v1(context, output, &raw) != 0);
  raw.length = 1;
  raw.bytes[0] ^= 1u;
  CHECK(et_g3c4_private_p2g2_prefix_commit_v1(context, output, &raw) != 0);
  raw.bytes[0] = (unsigned char)token;
  CHECK(et_g3c4_private_p2g2_prefix_commit_v1(context, output,
          context) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_p2g2_prefix_commit_v1(context, input, &raw) ==
        ET_G3C4_INVALID_ARGUMENT);
  int64_t *output_ids =
      (int64_t *)et_i64_tensor_test_data_storage_v1(output->ids);
  CHECK(output_ids != NULL);
  output_ids[1] = 256;
  CHECK(et_g3c4_private_p2g2_prefix_commit_v1(context, output, &raw) != 0);
  output_ids[1] = 0;
  et_i64_tensor_borrow *held_ids = NULL;
  OK(et_i64_tensor_borrow_begin_v1(output->ids, &held_ids, &ids_error));
  CHECK(et_g3c4_private_p2g2_prefix_commit_v1(context, output, &raw) != 0);
  OK(et_i64_tensor_borrow_end_v1(&held_ids, &ids_error));
  et_a2_kv_cache_transaction_view *held_view = NULL;
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      context->token_frame_transaction, 0u, &held_view, &cache_error));
  CHECK(et_g3c4_private_p2g2_prefix_commit_v1(context, output, &raw) != 0);
  OK(et_a2_kv_cache_transaction_view_end_v1(&held_view, &cache_error));
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *mask = NULL;
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      context->token_frame_transaction, 0u, &held_view, &cache_error));
  OK(et_a2_kv_cache_transaction_view_tensors_v1(
      held_view, &keys, &values, &lengths, &mask, &cache_error));
  unsigned char saved_key = *(unsigned char *)keys->data;
  *(unsigned char *)keys->data ^= 1u;
  OK(et_a2_kv_cache_transaction_view_end_v1(&held_view, &cache_error));
  CHECK(et_g3c4_private_p2g2_prefix_commit_v1(context, output, &raw) != 0);
  keys = values = lengths = mask = NULL;
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      context->token_frame_transaction, 0u, &held_view, &cache_error));
  OK(et_a2_kv_cache_transaction_view_tensors_v1(
      held_view, &keys, &values, &lengths, &mask, &cache_error));
  *(unsigned char *)keys->data = saved_key;
  OK(et_a2_kv_cache_transaction_view_end_v1(&held_view, &cache_error));
  uint32_t changed_logit;
  memcpy(&changed_logit, &next[0], sizeof(changed_logit));
  changed_logit ^= 1u;
  memcpy(&next[0], &changed_logit, sizeof(changed_logit));
  CHECK(memcmp(context->p2g2_frame_next_logits, next,
               sizeof(next)) != 0);
  CHECK(memcmp(context->generator_rng_words, before_rng,
               sizeof(before_rng)) == 0);
  pending_clean(output);

  pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    fail_prefix_commit = 1;
    (void)et_g3c4_private_p2g2_prefix_commit_v1(context, output, &raw);
    _exit(97);
  }
  int status = 0;
  CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);

  OK(et_g3c4_private_p2g2_prefix_commit_v1(context, output, &raw));
  CHECK(context->p2g2_prefix_committed == 1u && context->budget == 2);
  CHECK(context->p2g2_prefix_token == token &&
        context->p2g2_prefix_raw == raw.bytes[0]);
  memcpy(next, context->p2g2_prefix_next_logits, sizeof(next));
  CHECK(memcmp(context->generator_rng_words, after_rng,
               sizeof(after_rng)) == 0);
  CHECK(et_g3c4_token_frame_idle(context));
  pending_clean(output);
  snapshot_cache(context->cache, &committed);
  CHECK(committed.length == 3);
  CHECK(memcmp(committed.keep, (const uint8_t[4]){1, 1, 1, 0}, 4u) == 0);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_call_finish_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_token_frame_begin_last_v1(context, last, &token) != 0);
  CHECK(et_g3c4_private_output_prepare_v1(context, output) ==
        ET_G3C4_INVALID_STATE);

  et_a2_kv_cache_read_borrow *held_cache = NULL;
  OK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &held_cache, &cache_error));
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  OK(et_a2_kv_cache_read_borrow_end_v1(&held_cache, &cache_error));
  pending_clean(output);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&committed, &after);
  CHECK(memcmp(context->generator_rng_words, after_rng,
               sizeof(after_rng)) == 0);
  OK(et_i64_tensor_borrow_begin_v1(output->ids, &held_ids, &ids_error));
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  OK(et_i64_tensor_borrow_end_v1(&held_ids, &ids_error));
  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  CHECK(context->p2g2_prefix_committed == 0u &&
        context->p2g2_prefix_token == 0 &&
        context->p2g2_prefix_raw == 0u);
  CHECK(memcmp(context->p2g2_prefix_next_logits,
               (const float[256]){0}, sizeof(next)) == 0);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&committed, &after);
  CHECK(memcmp(context->generator_rng_words, after_rng,
               sizeof(after_rng)) == 0);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));

  /* Use the saved local ID; abort deliberately scrubbed the private copy. */
  reference_ids[2] = (int64_t)raw.bytes[0];
  et_g3c4_context_internal *reference = create_generator(owner);
  OK(et_g3c4_private_call_acquire_v1(reference, 2, 1));
  OK(et_g3c4_private_prefill3_v1(
      reference, reference_ids, reference_logits));
  CHECK(memcmp(next, reference_logits, sizeof(next)) == 0);
  snapshot_cache(reference->cache, &reference_cache);
  check_cache_snapshot_equal(&committed, &reference_cache);
  OK(et_g3c4_private_call_abort_v1(reference));
  OK(et_g3c4_private_generator_close_v1(reference));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  prefix_case(owner, (const int64_t[2]){0, 255}, 0);
  prefix_case(owner, (const int64_t[2]){7, 11}, 1);
  printf("G3-C4 P2/G2 prefix commit PASS: checks=%zu\n", checks);
  return 0;
}
