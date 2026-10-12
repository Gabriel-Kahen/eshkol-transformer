/* Test-only C4 aggregate: retain the terminal fault hooks and add public
 * rank-one result inspection/lease plus a fresh P2 first-draw oracle row. */
#define ET_G3C4_P2_G1_PUBLICATION_PRIVATE 1
#define ET_G3C4_P2_G1_COPYOUT_PRIVATE 1
#define ET_G3C4_P2_G1_COORDINATOR_PRIVATE 1
#define ET_G3C4_P2_G1_PUBLIC_RESULT_PRIVATE 1
#define ET_G3C4_P2_G2_PUBLIC_RESULT_PRIVATE 1
#include "p2_g2_terminal_close_native.c"

static et_i64_tensor_borrow *public_result_borrow;

int64_t et_g3c4_p2g2_public_test_result_borrow_v1(
    void *candidate, int64_t active) {
  et_g3c4_result_tensor_internal *result =
      et_g3c4_admit_result(candidate, 0);
  et_i64_tensor_error error;
  if (result == NULL) return -1;
  if (active == 1 && public_result_borrow == NULL)
    return et_i64_tensor_borrow_begin_v1(
        result->tensor, &public_result_borrow, &error);
  if (active == 0 && public_result_borrow != NULL)
    return et_i64_tensor_borrow_end_v1(&public_result_borrow, &error);
  return -1;
}

/* The prefix oracle only emits after a non-EOS first draw. This fresh
 * authentic P2 prefill records the first logits for first-EOS as well. */
int64_t et_g3c4_p2g2_public_test_first_oracle_v1(
    void *context_candidate, int64_t selected) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_idle_call(context_candidate);
  et_g3c4_context_internal *oracle;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  float logits[256];
  if (context == NULL || coordinator_owner == NULL ||
      context->owner != coordinator_owner || selected < 0 || selected > 255)
    return -1;
  oracle = et_g3c4_private_generator_seed_v1(
      coordinator_owner, coordinator_entry_rng[1],
      coordinator_policy[0], coordinator_policy[1],
      coordinator_policy[2], coordinator_policy[3],
      coordinator_policy[4], coordinator_policy[5]);
  input = create_prompt(coordinator_prompt, 2);
  if (oracle == NULL || input == NULL ||
      et_g3c4_private_prompt_prefill_preflight_v1(oracle, input, 2) != 0 ||
      et_g3c4_private_call_acquire_v1(oracle, 2, 2) != 0)
    return -1;
  output = et_g3c4_private_output_reserve_v1(oracle, 2);
  if (output == NULL ||
      et_g3c4_private_prompt_prefill_v1(oracle, input, logits) != 0 ||
      et_g3c4_private_call_abort_v1(oracle) != 0 ||
      et_g3c4_private_tensor_release_v1(input) != 0 ||
      et_g3c4_private_generator_close_v1(oracle) != 0)
    return -1;
  printf("PUBLIC_FIRST %s %lld %lld %lld %lld %lld %lld %lld",
         coordinator_policy[0] == 0 ? "greedy" : "categorical",
         (long long)coordinator_prompt[0],
         (long long)coordinator_prompt[1],
         (long long)coordinator_policy[1],
         (long long)coordinator_policy[2],
         (long long)coordinator_policy[3],
         (long long)coordinator_policy[5],
         (long long)selected);
  for (size_t index = 0u; index < 4u; index++)
    printf(" %lld", (long long)coordinator_entry_rng[index]);
  for (size_t index = 0u; index < 4u; index++)
    printf(" %lld", (long long)context->generator_rng_words[index]);
  for (size_t index = 0u; index < 256u; index++) {
    uint32_t bits;
    memcpy(&bits, &logits[index], sizeof(bits));
    printf(" %08x", bits);
  }
  putchar('\n');
  return 0;
}
