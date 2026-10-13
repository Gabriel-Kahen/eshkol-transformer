#define ET_G3R_COMMITTED_HISTORY_PRIVATE 1
#define ET_G3T_MANUAL_DECODE_PRIVATE 1
#define ET_G3T_MANUAL_P2_PREFILL_PRIVATE 1
#define ET_G3T_MANUAL_P1_PREFILL_PRIVATE 1
#define ET_G3T_MANUAL_LOGITS_PRIVATE 1
#define ET_G3T_OWNED_TOKEN_INPUT_PRIVATE 1
#define ET_G3T_P2_ZERO_BUDGET_PRIVATE 1
#define ET_G3T_ZERO_BUDGET_PRIVATE 1
#define ET_G3T_OUTPUT_TEXT_PRIVATE 1
#define ET_G3T_FINAL_PUBLICATION_PRIVATE 1
#define ET_G3T_INPUT_FROM_T1_PRIVATE 1
#define ET_G3T_PREFILL_SAMPLE_PRIVATE 1
#define ET_I64_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE 1
#include "../../src/eshkol_transformer/g3t_transport.c"
#include <assert.h>
#include <stdio.h>

int main(void) {
  g3t_context c = {0};
  c.policy[5] = 65;
  assert(g3t_history_valid(&c));
  c.frame.kind = 1;
  c.frame.prompt_length = 1;
  c.frame.token = 65;
  assert(!g3t_history_prompt_preflight(&c));
  g3t_history_commit_prompt(&c);
  c.prefill_committed = 1;
  assert(g3t_history_valid(&c));
  assert(c.history_length == 1 && c.history_prompt_length == 1);
  assert(!c.history_generated_eos_terminal);
  c.call_kind = 1;
  c.frame.kind = 2;
  c.frame.token = 65;
  assert(!g3t_history_append_preflight(&c));
  g3t_history_commit_append(&c, 0);
  assert(g3t_history_valid(&c));
  assert(c.history_manual_count == 1 && !c.history_generated_count);
  assert(!c.history_generated_eos_terminal);
  assert(c.history_ids[0] == 65 && c.history_ids[1] == 65);

  /* A new manual P2 commit replaces P1 plus manual append. */
  c.call_kind = 0;
  c.frame.kind = 1;
  c.frame.prompt_length = 2;
  c.frame.token = 66;
  c.frame.prompt_ids[0] = 66;
  c.frame.prompt_ids[1] = 256;
  assert(g3t_history_prompt_preflight(&c));
  assert(c.history_ids[0] == 65 && c.history_ids[1] == 65);
  assert(c.history_manual_count == 1 && c.history_prompt_length == 1);
  c.frame.prompt_ids[1] = 65;
  assert(!g3t_history_prompt_preflight(&c));
  g3t_history_commit_prompt(&c);
  assert(g3t_history_valid(&c));
  assert(c.history_ids[0] == 66 && c.history_ids[1] == 65);
  assert(c.history_length == 2 && c.history_prompt_length == 2);
  assert(!c.history_generated_count && !c.history_manual_count);
  assert(!c.history_generated_eos_terminal);
  c.frame.prompt_ids[1] = 256;
  assert(g3t_history_prompt_preflight(&c));
  assert(c.history_ids[0] == 66 && c.history_ids[1] == 65);
  assert(c.history_prompt_length == 2 && !c.history_manual_count);
  c.frame.prompt_ids[1] = 65;
  c.frame.kind = 2;
  assert(g3t_history_append_preflight(&c));

  c.call_kind = 2;
  c.policy[4] = 1;
  c.frame.kind = 1;
  c.frame.prompt_length = 1;
  c.frame.token = 66;
  g3t_history_commit_prompt(&c);
  assert(c.history_pending_g1 && g3t_history_valid(&c));
  c.frame.token = 67;
  assert(g3t_history_prompt_preflight(&c));
  c.frame.kind = 2;
  c.frame.token = 65;
  c.call_kind = 1;
  assert(g3t_history_append_preflight(&c));
  c.call_kind = 2;
  assert(!g3t_history_append_preflight(&c));
  g3t_history_commit_append(&c, 1);
  assert(!c.history_pending_g1);
  assert(g3t_history_valid(&c));
  assert(c.history_generated_count == 1 && !c.history_manual_count);
  assert(c.history_generated_eos_terminal == 1);
  c.policy[4] = 0;
  assert(!g3t_history_valid(&c));
  c.policy[4] = 1;
  c.history_generated_eos_terminal = 0;
  assert(!g3t_history_valid(&c));
  c.history_generated_eos_terminal = 1;
  c.history_ids[1] = 256;
  assert(!g3t_history_valid(&c));
  c.history_ids[1] = 65;
  assert(g3t_history_valid(&c));
  c.history_generated_count = 0;
  assert(!g3t_history_valid(&c));
  c.history_generated_count = 1;
  c.policy[5] = -1;
  c.history_generated_eos_terminal = 0;
  assert(g3t_history_valid(&c));
  c.frame.token = 64;
  c.history_generated_count = 0;
  c.history_length = c.history_prompt_length = 1;
  c.history_ids[1] = 0;
  assert(g3t_history_append_preflight(&c));
  c.history_pending_g1 = 1;
  assert(!g3t_history_append_preflight(&c));
  assert(c.history_ids[1] == 0 && c.history_length == 1);
  puts("G3-R committed history owner core PASS");
  return 0;
}
