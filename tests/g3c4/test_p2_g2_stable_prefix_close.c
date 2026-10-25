#define ET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE 1
#define ET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE 1
#define ET_G3C4_P2_G2_STABLE_PREFIX_PRIVATE 1
#define ET_G3C4_P2_G2_STABLE_PREFIX_TESTING 1
#define ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_FIRST_FRAME_TEST_MAIN stable_predecessor_main
#include "test_p2_g2_first_frame.c"
#undef ET_G3C4_P2_G2_FIRST_FRAME_TEST_MAIN
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static void stable_case(et_g3c4_model_owner_internal *owner,
                        const int64_t prompt[2], int categorical) {
  et_g3c4_context_internal *context = p2g2_generator(owner, categorical);
  et_g3c4_context_internal *foreign = p2g2_generator(owner, 0);
  et_g3c4_input_internal *input = create_prompt(prompt, 2);
  et_g3c4_output_internal *output;
  float last[256], next[256], keys[16], values[16];
  int64_t token = -1, rng[4];
  raw_carrier raw = {1, {0}};
  cache_snapshot before, after;
  et_kernel_error cache_error;
  et_i64_tensor_error ids_error;
  et_a2_kv_cache_read_borrow *held_cache = NULL;
  et_i64_tensor_borrow *held_ids = NULL;

  CHECK(et_g3c4_private_p2g2_stable_prefix_check_v1(context) != 0);
  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, 2));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 2));
  output = et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, last));
  OK(et_g3c4_private_token_frame_begin_last_v1(context, last, &token));
  OK(et_g3c4_private_token_forward_v1(context, token, next));
  raw.bytes[0] = (unsigned char)token;
  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, output) != 0);
  OK(et_g3c4_private_p2g2_prefix_commit_v1(context, output, &raw));
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  snapshot_cache(context->cache, &before);
  CHECK(before.length == 3 &&
        memcmp(before.keep, (const uint8_t[4]){1, 1, 1, 0}, 4u) == 0);
  memcpy(keys, before.keys, sizeof(keys));
  memcpy(values, before.values, sizeof(values));

  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, input) != 0);
  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(input, output) != 0);
  output->parent_ctx = foreign;
  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, output) != 0);
  output->parent_ctx = context;
  output->parent_ctx = NULL;
  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, output) != 0);
  output->parent_ctx = context;
  context->generator_policy[4] = 1;
  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, output) != 0);
  context->generator_policy[4] = 2;
  context->generator_policy[5] = token;
  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, output) != 0);
  context->generator_policy[5] = -1;
  context->prefill_binding_values[0] ^= 1u;
  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, output) != 0);
  context->prefill_binding_values[0] ^= 1u;
  int64_t *ids_data = (int64_t *)et_i64_tensor_test_data_storage_v1(output->ids);
  CHECK(ids_data != NULL);
  ids_data[0] = 1;
  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, output) != 0);
  ids_data[0] = 0;
  OK(et_i64_tensor_borrow_begin_v1(output->ids, &held_ids, &ids_error));
  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, output) != 0);
  OK(et_i64_tensor_borrow_end_v1(&held_ids, &ids_error));
  OK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &held_cache, &cache_error));
  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, output) != 0);
  OK(et_a2_kv_cache_read_borrow_end_v1(&held_cache, &cache_error));
  CHECK(context->p2g2_prefix_committed == 1u &&
        context->stable_prefix_ready == 0u &&
        owner->active == context);
  CHECK(output->transport.state == 0u && output->parent_ctx == context);

  pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    et_g3c4_stable_prefix_fail_tail = 1;
    (void)et_g3c4_private_p2g2_stable_prefix_close_v1(context, output);
    _exit(97);
  }
  int status = 0;
  CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
  CHECK(context->p2g2_prefix_committed == 1u && owner->active == context);

  OK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, output));
  CHECK(context->stable_prefix_ready == 1u &&
        context->p2g2_prefix_committed == 0u &&
        context->stable_prompt[0] == prompt[0] &&
        context->stable_prompt[1] == prompt[1] &&
        context->stable_token == token &&
        context->stable_raw == raw.bytes[0]);
  CHECK(memcmp(context->stable_rng, rng, sizeof(rng)) == 0 &&
        memcmp(context->stable_logits, next, sizeof(next)) == 0 &&
        memcmp(context->stable_keys, keys, sizeof(keys)) == 0 &&
        memcmp(context->stable_values, values, sizeof(values)) == 0);
  CHECK(owner->active == NULL && context->transport.busy == ET_G3C4_CALL_IDLE &&
        context->call_kind == 0 && context->budget == 0 &&
        context->acquired_mask == 0u && et_g3c4_pins_idle(&context->pins));
  check_dead_output(output);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&before, &after);
  OK(et_g3c4_private_p2g2_stable_prefix_check_v1(context));
  OK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &held_cache, &cache_error));
  const et_kernel_tensor_view_v1 *borrowed_keys = NULL, *borrowed_values = NULL;
  const et_kernel_tensor_view_v1 *borrowed_lengths = NULL, *borrowed_keep = NULL;
  OK(et_a2_kv_cache_read_borrow_layer_v1(
      held_cache, 0u, &borrowed_keys, &borrowed_values,
      &borrowed_lengths, &borrowed_keep, &cache_error));
  CHECK(borrowed_keys != NULL && borrowed_keys->data != NULL);
  ((unsigned char *)borrowed_keys->data)[0] ^= 1u;
  OK(et_a2_kv_cache_read_borrow_end_v1(&held_cache, &cache_error));
  CHECK(et_g3c4_private_p2g2_stable_prefix_check_v1(context) != 0);
  OK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &held_cache, &cache_error));
  borrowed_keys = borrowed_values = borrowed_lengths = borrowed_keep = NULL;
  OK(et_a2_kv_cache_read_borrow_layer_v1(
      held_cache, 0u, &borrowed_keys, &borrowed_values,
      &borrowed_lengths, &borrowed_keep, &cache_error));
  ((unsigned char *)borrowed_keys->data)[0] ^= 1u;
  OK(et_a2_kv_cache_read_borrow_end_v1(&held_cache, &cache_error));
  OK(et_g3c4_private_p2g2_stable_prefix_check_v1(context));
  CHECK(et_g3c4_private_call_acquire_v1(context, 2, 2) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_p2g2_stable_prefix_close_v1(context, output) != 0);
  OK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &held_cache, &cache_error));
  CHECK(et_g3c4_private_p2g2_stable_prefix_check_v1(context) != 0);
  OK(et_a2_kv_cache_read_borrow_end_v1(&held_cache, &cache_error));
  CHECK(context->stable_prefix_ready == 1u);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
  OK(et_g3c4_private_generator_close_v1(foreign));
  CHECK(et_g3c4_private_p2g2_stable_prefix_check_v1(context) != 0);
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  stable_case(owner, (const int64_t[2]){0, 255}, 0);
  stable_case(owner, (const int64_t[2]){7, 11}, 1);
  printf("G3-C4 native stable prefix close PASS: checks=%zu\n", checks);
  return 0;
}
