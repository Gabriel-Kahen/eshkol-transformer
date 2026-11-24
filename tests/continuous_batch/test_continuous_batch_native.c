#include "continuous_batch_eshkol_bridge.h"
#include "eshkol_transformer/continuous_batch_abi.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* Allocation trap: enforces zero heap allocations on hot-path scheduler
 * operations (et_cb_init, et_cb_enqueue, et_cb_cancel, et_cb_step_plan,
 * et_cb_step_commit, et_cb_get_request_info, et_cb_get_stats,
 * et_cb_verify_invariants, and all per-step AOT bridge calls). */
static int hot_path_active = 0;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!hot_path_active);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!hot_path_active);
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!hot_path_active);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!hot_path_active);
  __real_free(ptr);
}

static et_cb_scheduler g_sched;

static void check_invariants_and_budget(const et_cb_scheduler *s,
                                        const et_cb_step_batch *plan) {
  CHECK(et_cb_verify_invariants(s) == ET_CB_OK);
  et_cb_stats st;
  CHECK(et_cb_get_stats(s, &st) == ET_CB_OK);
  CHECK(st.used_kv_blocks + st.free_kv_blocks == st.total_kv_blocks);
  CHECK(st.running_requests <= s->cfg.max_running_seqs);
  if (plan != NULL) {
    CHECK(plan->total_tokens <= s->cfg.max_tokens_per_step);
    CHECK(plan->entry_count <= s->cfg.max_running_seqs);
    CHECK(plan->used_kv_blocks + plan->free_kv_blocks == s->cfg.total_kv_blocks);
    CHECK(plan->prefill_tokens + plan->decode_tokens == plan->total_tokens);
    uint32_t sum = 0u;
    for (uint32_t i = 0u; i < plan->entry_count; ++i)
      sum += plan->entries[i].chunk_tokens;
    CHECK(sum == plan->total_tokens);
  }
}

static void test_config_and_negative_validation(void) {
  hot_path_active = 1;

  CHECK(et_cb_validate_config(NULL) == ET_CB_ERR_ARGUMENT);
  CHECK(et_cb_init(NULL, NULL) == ET_CB_ERR_ARGUMENT);

  const et_cb_config base = {4u, 16u, 32u, 16u, 4u,
                             ET_CB_POLICY_DECODE_FIRST_CHUNKED_PREFILL};
  CHECK(et_cb_validate_config(&base) == ET_CB_OK);

  et_cb_config bad = base;
  bad.max_running_seqs = 0u;
  CHECK(et_cb_validate_config(&bad) == ET_CB_ERR_CONFIG);
  bad = base;
  bad.max_running_seqs = ET_CB_MAX_RUNNING_SEQS_LIMIT + 1u;
  CHECK(et_cb_validate_config(&bad) == ET_CB_ERR_CONFIG);

  bad = base;
  bad.max_waiting_seqs = 0u;
  CHECK(et_cb_validate_config(&bad) == ET_CB_ERR_CONFIG);
  bad = base;
  bad.max_waiting_seqs = ET_CB_MAX_WAITING_SEQS_LIMIT + 1u;
  CHECK(et_cb_validate_config(&bad) == ET_CB_ERR_CONFIG);

  bad = base;
  bad.max_tokens_per_step = 0u;
  CHECK(et_cb_validate_config(&bad) == ET_CB_ERR_CONFIG);
  bad = base;
  bad.max_tokens_per_step = ET_CB_MAX_TOKENS_PER_STEP_LIMIT + 1u;
  CHECK(et_cb_validate_config(&bad) == ET_CB_ERR_CONFIG);

  bad = base;
  bad.total_kv_blocks = 0u;
  CHECK(et_cb_validate_config(&bad) == ET_CB_ERR_CONFIG);
  bad = base;
  bad.total_kv_blocks = ET_CB_TOTAL_KV_BLOCKS_LIMIT + 1u;
  CHECK(et_cb_validate_config(&bad) == ET_CB_ERR_CONFIG);

  bad = base;
  bad.block_size = 0u;
  CHECK(et_cb_validate_config(&bad) == ET_CB_ERR_CONFIG);
  bad = base;
  bad.block_size = ET_CB_BLOCK_SIZE_LIMIT + 1u;
  CHECK(et_cb_validate_config(&bad) == ET_CB_ERR_CONFIG);

  bad = base;
  bad.policy = 2u;
  CHECK(et_cb_validate_config(&bad) == ET_CB_ERR_CONFIG);
  CHECK(et_cb_init(&g_sched, &bad) == ET_CB_ERR_CONFIG);

  /* Initialize valid small scheduler: total capacity = 4 * 4 = 16 tokens. */
  const et_cb_config small = {2u, 2u, 6u, 4u, 4u,
                              ET_CB_POLICY_DECODE_FIRST_CHUNKED_PREFILL};
  CHECK(et_cb_init(&g_sched, &small) == ET_CB_OK);
  CHECK(et_cb_verify_invariants(&g_sched) == ET_CB_OK);

  /* Invalid request specs. */
  const et_cb_request_spec zero_prompt = {1u, 0u, 2u, 0, 0u};
  CHECK(et_cb_enqueue(&g_sched, &zero_prompt) == ET_CB_ERR_ARGUMENT);
  const et_cb_request_spec zero_gen = {1u, 4u, 0u, 0, 0u};
  CHECK(et_cb_enqueue(&g_sched, &zero_gen) == ET_CB_ERR_ARGUMENT);
  const et_cb_request_spec bad_rsv = {1u, 4u, 2u, 0, 1u};
  CHECK(et_cb_enqueue(&g_sched, &bad_rsv) == ET_CB_ERR_ARGUMENT);

  /* Oversized prompt (> 16) and oversized prompt+max_new-1 (> 16). */
  const et_cb_request_spec huge_prompt = {1u, 17u, 1u, 0, 0u};
  CHECK(et_cb_enqueue(&g_sched, &huge_prompt) == ET_CB_ERR_REQUEST_TOO_LARGE);
  const et_cb_request_spec huge_total = {1u, 12u, 6u, 0, 0u}; /* 12+6-1=17 */
  CHECK(et_cb_enqueue(&g_sched, &huge_total) == ET_CB_ERR_REQUEST_TOO_LARGE);
  const et_cb_request_spec exact_fit = {1u, 12u, 5u, 0, 0u}; /* 12+5-1=16 */
  CHECK(et_cb_enqueue(&g_sched, &exact_fit) == ET_CB_OK);

  /* Duplicate ID rejection. */
  const et_cb_request_spec dup = {1u, 4u, 2u, 0, 0u};
  CHECK(et_cb_enqueue(&g_sched, &dup) == ET_CB_ERR_DUPLICATE_ID);

  /* Waiting queue full (max_waiting_seqs = 2). */
  const et_cb_request_spec r2 = {2u, 4u, 2u, 0, 0u};
  const et_cb_request_spec r3 = {3u, 4u, 2u, 0, 0u};
  CHECK(et_cb_enqueue(&g_sched, &r2) == ET_CB_OK);
  CHECK(et_cb_enqueue(&g_sched, &r3) == ET_CB_ERR_QUEUE_FULL);

  /* Commit before plan -> NO_PENDING_PLAN. */
  const et_cb_commit_item dummy = {1u, 0u, 0u};
  CHECK(et_cb_step_commit(&g_sched, 0u, &dummy, 1u) ==
        ET_CB_ERR_NO_PENDING_PLAN);

  /* Plan step 0: req 1 takes 6 of 12 prompt tokens (non-final prefill). */
  et_cb_step_batch plan;
  CHECK(et_cb_step_plan(&g_sched, &plan) == ET_CB_OK);
  check_invariants_and_budget(&g_sched, &plan);
  CHECK(plan.step_index == 0u);
  CHECK(plan.entry_count == 1u);
  CHECK(plan.entries[0].request_id == 1u);
  CHECK(plan.entries[0].chunk_tokens == 6u);
  CHECK(plan.entries[0].produces_output_token == 0u);

  /* Double plan -> PLAN_PENDING. */
  CHECK(et_cb_step_plan(&g_sched, &plan) == ET_CB_ERR_PLAN_PENDING);

  /* Cancel while in active pending plan -> PLAN_PENDING. */
  CHECK(et_cb_cancel(&g_sched, 1u) == ET_CB_ERR_PLAN_PENDING);
  /* Cancel of unknown ID -> NOT_FOUND. */
  CHECK(et_cb_cancel(&g_sched, 999u) == ET_CB_ERR_NOT_FOUND);

  /* Commit mismatch checks (all must fail closed without mutating state). */
  const et_cb_commit_item wrong_id = {2u, 0u, 0u};
  const et_cb_commit_item wrong_gen = {1u, 1u, 0u};
  const et_cb_commit_item eos_on_prefill = {1u, 0u, 1u};
  const et_cb_commit_item bad_eos = {1u, 0u, 2u};
  const et_cb_commit_item ok_item = {1u, 0u, 0u};

  CHECK(et_cb_step_commit(&g_sched, 1u, &ok_item, 1u) ==
        ET_CB_ERR_COMMIT_MISMATCH);
  CHECK(et_cb_step_commit(&g_sched, 0u, &ok_item, 0u) ==
        ET_CB_ERR_COMMIT_MISMATCH);
  CHECK(et_cb_step_commit(&g_sched, 0u, NULL, 1u) == ET_CB_ERR_ARGUMENT);
  CHECK(et_cb_step_commit(&g_sched, 0u, &wrong_id, 1u) ==
        ET_CB_ERR_COMMIT_MISMATCH);
  CHECK(et_cb_step_commit(&g_sched, 0u, &wrong_gen, 1u) ==
        ET_CB_ERR_COMMIT_MISMATCH);
  CHECK(et_cb_step_commit(&g_sched, 0u, &eos_on_prefill, 1u) ==
        ET_CB_ERR_COMMIT_MISMATCH);
  CHECK(et_cb_step_commit(&g_sched, 0u, &bad_eos, 1u) == ET_CB_ERR_ARGUMENT);

  /* Valid commit succeeds after failed commit attempts. */
  CHECK(et_cb_step_commit(&g_sched, 0u, &ok_item, 1u) == ET_CB_OK);
  check_invariants_and_budget(&g_sched, NULL);

  /* Cancel running request 1 and waiting request 2; reject double-cancel. */
  CHECK(et_cb_cancel(&g_sched, 1u) == ET_CB_OK);
  CHECK(et_cb_cancel(&g_sched, 1u) == ET_CB_ERR_ALREADY_TERMINAL);
  CHECK(et_cb_cancel(&g_sched, 2u) == ET_CB_OK);
  CHECK(et_cb_cancel(&g_sched, 2u) == ET_CB_ERR_ALREADY_TERMINAL);

  /* All KV blocks must be freed. */
  et_cb_stats st;
  CHECK(et_cb_get_stats(&g_sched, &st) == ET_CB_OK);
  CHECK(st.used_kv_blocks == 0u && st.free_kv_blocks == small.total_kv_blocks);
  CHECK(st.cancelled_requests == 2u && st.running_requests == 0u &&
        st.waiting_requests == 0u);
  CHECK(et_cb_verify_invariants(&g_sched) == ET_CB_OK);

  hot_path_active = 0;
}

static void test_preemption_and_chunked_prefill_progress(void) {
  hot_path_active = 1;

  /* Tight KV pool: 2 blocks x 4 tokens = 8 tokens total capacity.
   * R1: prompt=4, max_new=3 (needs 6 tokens = 2 blocks)
   * R2: prompt=4, max_new=2 (needs 5 tokens = 2 blocks)
   * Both admit in step 0 (1 block each -> free=0).
   * In step 1, R1 needs a 2nd block -> preempts R2 -> finishes at step 2 ->
   * R2 re-admits at step 3 with effective_prefill = 4+1 = 5 -> finishes! */
  const et_cb_config cfg = {2u, 8u, 8u, 2u, 4u,
                            ET_CB_POLICY_DECODE_FIRST_CHUNKED_PREFILL};
  CHECK(et_cb_init(&g_sched, &cfg) == ET_CB_OK);

  const et_cb_request_spec r1 = {11u, 4u, 3u, 0, 0u};
  const et_cb_request_spec r2 = {22u, 4u, 2u, 0, 0u};
  CHECK(et_cb_enqueue(&g_sched, &r1) == ET_CB_OK);
  CHECK(et_cb_enqueue(&g_sched, &r2) == ET_CB_OK);

  et_cb_step_batch plan;
  et_cb_commit_item items[2];

  /* Step 0: both prefill 4 tokens, produce first token. */
  CHECK(et_cb_step_plan(&g_sched, &plan) == ET_CB_OK);
  check_invariants_and_budget(&g_sched, &plan);
  CHECK(plan.entry_count == 2u && plan.preempted_count == 0u);
  CHECK(plan.used_kv_blocks == 2u && plan.free_kv_blocks == 0u);
  items[0] = (et_cb_commit_item){11u, 1u, 0u};
  items[1] = (et_cb_commit_item){22u, 1u, 0u};
  CHECK(et_cb_step_commit(&g_sched, 0u, items, 2u) == ET_CB_OK);

  /* Step 1: R1 needs block #2 -> preempts R2! */
  CHECK(et_cb_step_plan(&g_sched, &plan) == ET_CB_OK);
  check_invariants_and_budget(&g_sched, &plan);
  CHECK(plan.preempted_count == 1u && plan.preempted_ids[0] == 22u);
  CHECK(plan.entry_count == 1u && plan.entries[0].request_id == 11u);
  CHECK(plan.entries[0].is_prefill == 0u && plan.entries[0].new_kv_blocks == 1u);
  et_cb_request_info r2_info;
  CHECK(et_cb_get_request_info(&g_sched, 22u, &r2_info) == ET_CB_OK);
  CHECK(r2_info.state == ET_CB_STATE_PREEMPTED);
  CHECK(r2_info.prompt_len == 4u && r2_info.generated_tokens == 1u);
  CHECK(r2_info.computed_tokens == 0u && r2_info.allocated_blocks == 0u);
  CHECK(r2_info.preempt_count == 1u);

  items[0] = (et_cb_commit_item){11u, 1u, 0u};
  CHECK(et_cb_step_commit(&g_sched, 1u, items, 1u) == ET_CB_OK);

  /* Step 2: R1 decodes token #3 (0 new blocks) and finishes. */
  CHECK(et_cb_step_plan(&g_sched, &plan) == ET_CB_OK);
  check_invariants_and_budget(&g_sched, &plan);
  CHECK(plan.entry_count == 1u && plan.entries[0].request_id == 11u &&
        plan.entries[0].new_kv_blocks == 0u);
  items[0] = (et_cb_commit_item){11u, 1u, 0u};
  CHECK(et_cb_step_commit(&g_sched, 2u, items, 1u) == ET_CB_OK);

  et_cb_request_info r1_info;
  CHECK(et_cb_get_request_info(&g_sched, 11u, &r1_info) == ET_CB_OK);
  CHECK(r1_info.state == ET_CB_STATE_FINISHED && r1_info.generated_tokens == 3u);
  CHECK(r1_info.first_token_step == 0u && r1_info.ttft_steps == 1u);
  CHECK(r1_info.completion_step == 2u && r1_info.total_latency_steps == 3u);
  CHECK(r1_info.allocated_blocks == 0u);
  /* Cancel of finished request -> ALREADY_TERMINAL. */
  CHECK(et_cb_cancel(&g_sched, 11u) == ET_CB_ERR_ALREADY_TERMINAL);

  /* Step 3: R2 re-admits, recomputes 4+1=5 tokens in 1 chunk, generates #2, finishes! */
  CHECK(et_cb_step_plan(&g_sched, &plan) == ET_CB_OK);
  check_invariants_and_budget(&g_sched, &plan);
  CHECK(plan.entry_count == 1u && plan.entries[0].request_id == 22u);
  CHECK(plan.entries[0].is_prefill == 1u && plan.entries[0].chunk_tokens == 5u);
  CHECK(plan.entries[0].is_final_prefill_chunk == 1u &&
        plan.entries[0].produces_output_token == 1u);
  CHECK(plan.entries[0].new_kv_blocks == 2u &&
        plan.entries[0].total_kv_blocks == 2u);
  items[0] = (et_cb_commit_item){22u, 1u, 0u};
  CHECK(et_cb_step_commit(&g_sched, 3u, items, 1u) == ET_CB_OK);

  CHECK(et_cb_get_request_info(&g_sched, 22u, &r2_info) == ET_CB_OK);
  CHECK(r2_info.state == ET_CB_STATE_FINISHED && r2_info.generated_tokens == 2u);
  CHECK(r2_info.first_token_step == 0u && r2_info.ttft_steps == 1u);
  CHECK(r2_info.completion_step == 3u && r2_info.preempt_count == 1u);
  CHECK(r2_info.allocated_blocks == 0u);

  et_cb_stats st;
  CHECK(et_cb_get_stats(&g_sched, &st) == ET_CB_OK);
  CHECK(st.finished_requests == 2u && st.used_kv_blocks == 0u &&
        st.free_kv_blocks == 2u && st.total_preemptions == 1u &&
        st.total_generated_tokens == 5u);
  CHECK(et_cb_verify_invariants(&g_sched) == ET_CB_OK);

  hot_path_active = 0;
}

static void test_both_policies_stress_and_bridge_zero_alloc(void) {
  /* Stress test both policies under tight KV blocks, priorities, arrivals,
   * mid-flight cancellation, and EOS early-stop, verifying zero deadlock and
   * zero KV leaks. */
  for (uint32_t pol = 0u; pol < 2u; ++pol) {
    hot_path_active = 1;
    const et_cb_config cfg = {4u, 32u, 10u, 8u, 4u, pol};
    CHECK(et_cb_init(&g_sched, &cfg) == ET_CB_OK);

    for (uint32_t i = 0u; i < 12u; ++i) {
      const et_cb_request_spec spec = {
          (uint64_t)(1000u + i),
          (uint32_t)(3u + (i * 5u) % 19u), /* 3..21 */
          (uint32_t)(1u + (i * 3u) % 6u),  /* 1..6 */
          (int32_t)(i % 3u),
          0u};
      CHECK(et_cb_enqueue(&g_sched, &spec) == ET_CB_OK);
    }

    uint32_t steps = 0u;
    int cancelled_one = 0;
    while (1) {
      et_cb_stats st;
      CHECK(et_cb_get_stats(&g_sched, &st) == ET_CB_OK);
      if (st.running_requests == 0u && st.waiting_requests == 0u)
        break;
      CHECK(steps < 500u); /* Guarantee no deadlock or livelock. */

      if (steps == 5u && !cancelled_one) {
        et_cb_request_info info;
        CHECK(et_cb_get_request_info(&g_sched, 1009u, &info) == ET_CB_OK);
        if (info.state != ET_CB_STATE_FINISHED &&
            info.state != ET_CB_STATE_CANCELLED) {
          CHECK(et_cb_cancel(&g_sched, 1009u) == ET_CB_OK);
          cancelled_one = 1;
        }
      }

      et_cb_step_batch plan;
      CHECK(et_cb_step_plan(&g_sched, &plan) == ET_CB_OK);
      check_invariants_and_budget(&g_sched, &plan);
      CHECK(plan.entry_count > 0u); /* Strict forward progress every step! */

      et_cb_commit_item items[ET_CB_MAX_RUNNING_SEQS_LIMIT];
      for (uint32_t i = 0u; i < plan.entry_count; ++i) {
        items[i].request_id = plan.entries[i].request_id;
        items[i].generated_token_count = plan.entries[i].produces_output_token;
        /* Early EOS on request 1002 once it produces a token. */
        items[i].hit_eos =
            (plan.entries[i].request_id == 1002u &&
             plan.entries[i].produces_output_token == 1u)
                ? 1u
                : 0u;
      }
      CHECK(et_cb_step_commit(&g_sched, plan.step_index, items,
                              plan.entry_count) == ET_CB_OK);
      check_invariants_and_budget(&g_sched, NULL);
      steps += 1u;
    }

    et_cb_stats st;
    CHECK(et_cb_get_stats(&g_sched, &st) == ET_CB_OK);
    CHECK(st.finished_requests + st.cancelled_requests == 12u);
    CHECK(st.used_kv_blocks == 0u && st.free_kv_blocks == cfg.total_kv_blocks);
    CHECK(et_cb_verify_invariants(&g_sched) == ET_CB_OK);
    hot_path_active = 0;
  }

  /* Verify bridge per-step hot path also performs zero heap allocations. */
  et_cb_aot_ctx *ctx = NULL;
  CHECK(et_cb_aot_create(3, 16, 8, 6, 4, 0, &ctx) == ET_CB_OK && ctx != NULL);
  hot_path_active = 1;
  CHECK(et_cb_aot_enqueue(ctx, 50, 12, 3, 1) == ET_CB_OK);
  CHECK(et_cb_aot_enqueue(ctx, 51, 6, 3, 0) == ET_CB_OK);
  for (int guard = 0; guard < 50; ++guard) {
    if (et_cb_aot_running_count(ctx) == 0 && et_cb_aot_waiting_count(ctx) == 0)
      break;
    CHECK(et_cb_aot_step_plan(ctx) == ET_CB_OK);
    CHECK(et_cb_aot_verify_invariants(ctx) == ET_CB_OK);
    const int64_t step = et_cb_aot_plan_step_index(ctx);
    const int64_t n = et_cb_aot_plan_entry_count(ctx);
    CHECK(et_cb_aot_commit_begin(ctx) == ET_CB_OK);
    CHECK(et_cb_aot_step_commit(ctx, step, n) == ET_CB_OK);
    CHECK(et_cb_aot_verify_invariants(ctx) == ET_CB_OK);
  }
  CHECK(et_cb_aot_finished_count(ctx) == 2);
  CHECK(et_cb_aot_used_blocks(ctx) == 0 && et_cb_aot_free_blocks(ctx) == 6);
  hot_path_active = 0;
  et_cb_aot_destroy(ctx);
}

/* Oracle trace emitter: runs 4 deterministic scenarios (both policies,
 * chunked prefill, tight KV preemption, mid-flight arrivals, cancel, EOS)
 * and prints the full step-by-step schedule & final per-request metrics for
 * independent verification by tests/continuous_batch/test_continuous_batch_oracle.py. */
typedef struct oracle_arrival {
  uint32_t step;
  uint64_t id;
  uint32_t prompt_len;
  uint32_t max_new_tokens;
  int32_t priority;
} oracle_arrival;

typedef struct oracle_event {
  uint32_t step;
  uint64_t id;
} oracle_event;

static void emit_oracle_case(const char *name, const et_cb_config *cfg,
                             const oracle_arrival *arrivals,
                             size_t arrival_count,
                             const oracle_event *cancels, size_t cancel_count,
                             const oracle_event *eos_on_gen,
                             size_t eos_count) {
  printf("CASE %s %u %u %u %u %u %u\n", name, cfg->max_running_seqs,
         cfg->max_waiting_seqs, cfg->max_tokens_per_step, cfg->total_kv_blocks,
         cfg->block_size, cfg->policy);

  CHECK(et_cb_init(&g_sched, cfg) == ET_CB_OK);

  uint32_t max_event_step = 0u;
  for (size_t i = 0u; i < arrival_count; ++i) {
    if (arrivals[i].step > max_event_step)
      max_event_step = arrivals[i].step;
  }

  for (uint32_t step = 0u; step < 300u; ++step) {
    for (size_t i = 0u; i < arrival_count; ++i) {
      if (arrivals[i].step == step) {
        const et_cb_request_spec spec = {arrivals[i].id, arrivals[i].prompt_len,
                                         arrivals[i].max_new_tokens,
                                         arrivals[i].priority, 0u};
        CHECK(et_cb_enqueue(&g_sched, &spec) == ET_CB_OK);
        printf("ENQUEUE %u %" PRIu64 " %u %u %d\n", step, spec.request_id,
               spec.prompt_len, spec.max_new_tokens, spec.priority);
      }
    }
    for (size_t i = 0u; i < cancel_count; ++i) {
      if (cancels[i].step == step) {
        CHECK(et_cb_cancel(&g_sched, cancels[i].id) == ET_CB_OK);
        printf("CANCEL %u %" PRIu64 "\n", step, cancels[i].id);
      }
    }

    et_cb_stats st;
    CHECK(et_cb_get_stats(&g_sched, &st) == ET_CB_OK);
    if (step > max_event_step && st.running_requests == 0u &&
        st.waiting_requests == 0u)
      break;

    et_cb_step_batch plan;
    CHECK(et_cb_step_plan(&g_sched, &plan) == ET_CB_OK);
    check_invariants_and_budget(&g_sched, &plan);

    printf("STEP %u %u %u %u %u %u %u %u", plan.step_index, plan.entry_count,
           plan.total_tokens, plan.prefill_tokens, plan.decode_tokens,
           plan.preempted_count, plan.used_kv_blocks, plan.free_kv_blocks);
    for (uint32_t p = 0u; p < plan.preempted_count; ++p)
      printf(" P:%" PRIu64, plan.preempted_ids[p]);
    printf("\n");

    et_cb_commit_item items[ET_CB_MAX_RUNNING_SEQS_LIMIT];
    for (uint32_t i = 0u; i < plan.entry_count; ++i) {
      const et_cb_plan_entry *e = &plan.entries[i];
      uint32_t eos = 0u;
      if (e->produces_output_token) {
        et_cb_request_info rinfo;
        CHECK(et_cb_get_request_info(&g_sched, e->request_id, &rinfo) ==
              ET_CB_OK);
        const uint32_t next_gen = rinfo.generated_tokens + 1u;
        for (size_t k = 0u; k < eos_count; ++k) {
          if (eos_on_gen[k].id == e->request_id &&
              eos_on_gen[k].step == next_gen)
            eos = 1u;
        }
      }
      items[i].request_id = e->request_id;
      items[i].generated_token_count = e->produces_output_token;
      items[i].hit_eos = eos;
      printf("ENTRY %" PRIu64 " %u %u %u %u %u %u %u %u\n", e->request_id,
             e->token_offset, e->chunk_tokens, e->is_prefill,
             e->is_final_prefill_chunk, e->produces_output_token,
             e->new_kv_blocks, e->total_kv_blocks, eos);
    }

    CHECK(et_cb_step_commit(&g_sched, plan.step_index, items,
                            plan.entry_count) == ET_CB_OK);
    check_invariants_and_budget(&g_sched, NULL);
    CHECK(et_cb_get_stats(&g_sched, &st) == ET_CB_OK);
    printf("POST %u %u %u %u %u %u\n", plan.step_index, st.running_requests,
           st.waiting_requests, st.finished_requests, st.used_kv_blocks,
           st.free_kv_blocks);
  }

  for (size_t i = 0u; i < arrival_count; ++i) {
    et_cb_request_info info;
    CHECK(et_cb_get_request_info(&g_sched, arrivals[i].id, &info) == ET_CB_OK);
    printf("REQ %" PRIu64 " %u %u %u %u %u %u %u %u %u\n", info.request_id,
           info.state, info.computed_tokens, info.generated_tokens,
           info.allocated_blocks, info.first_token_step, info.completion_step,
           info.ttft_steps, info.preempt_count, info.eos_stopped);
  }
  et_cb_stats final_st;
  CHECK(et_cb_get_stats(&g_sched, &final_st) == ET_CB_OK);
  printf("END %u %u %u %u %u %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
         final_st.current_step, final_st.finished_requests,
         final_st.cancelled_requests, final_st.used_kv_blocks,
         final_st.free_kv_blocks, final_st.total_preemptions,
         final_st.total_generated_tokens, final_st.total_scheduled_tokens);
}

static void run_oracle_dump(void) {
  /* Case 1: DECODE_FIRST_CHUNKED_PREFILL with chunking, preemption, cancel, EOS. */
  const et_cb_config c1 = {3u, 16u, 8u, 6u, 4u,
                           ET_CB_POLICY_DECODE_FIRST_CHUNKED_PREFILL};
  const oracle_arrival a1[] = {
      {0u, 101u, 15u, 3u, 2}, {0u, 102u, 3u, 5u, 1}, {0u, 103u, 7u, 4u, 0},
      {2u, 104u, 8u, 4u, 0},  {3u, 105u, 11u, 2u, 1}};
  const oracle_event can1[] = {{4u, 104u}};
  const oracle_event eos1[] = {{2u, 102u}}; /* EOS on 2nd generated token */
  emit_oracle_case("decode_first_mixed", &c1, a1, 5u, can1, 1u, eos1, 1u);

  /* Case 2: PREFILL_FIRST_CHUNKED with identical arrivals & events. */
  const et_cb_config c2 = {3u, 16u, 8u, 6u, 4u,
                           ET_CB_POLICY_PREFILL_FIRST_CHUNKED};
  emit_oracle_case("prefill_first_mixed", &c2, a1, 5u, can1, 1u, eos1, 1u);

  /* Case 3: Extreme KV pressure (3 blocks x 3 tokens = 9 cap) forcing multiple
   * preemptions and exact recomputation preservation. */
  const et_cb_config c3 = {3u, 16u, 5u, 3u, 3u,
                           ET_CB_POLICY_DECODE_FIRST_CHUNKED_PREFILL};
  const oracle_arrival a3[] = {{0u, 201u, 3u, 5u, 0},
                               {0u, 202u, 3u, 4u, 0},
                               {0u, 203u, 3u, 3u, 0},
                               {2u, 204u, 5u, 3u, 1}};
  emit_oracle_case("extreme_kv_pressure", &c3, a3, 4u, NULL, 0u, NULL, 0u);

  /* Case 4: PREFILL_FIRST_CHUNKED under extreme KV pressure (livelock check). */
  const et_cb_config c4 = {3u, 16u, 5u, 3u, 3u,
                           ET_CB_POLICY_PREFILL_FIRST_CHUNKED};
  emit_oracle_case("prefill_first_pressure", &c4, a3, 4u, NULL, 0u, NULL, 0u);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    run_oracle_dump();
    return 0;
  }

  test_config_and_negative_validation();
  test_preemption_and_chunked_prefill_progress();
  test_both_policies_stress_and_bridge_zero_alloc();
  printf("CONTINUOUS_BATCH native C11 tests (allocation-trapped hot path): "
         "PASS\n");
  return 0;
}
