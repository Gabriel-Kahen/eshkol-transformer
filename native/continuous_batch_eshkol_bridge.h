#ifndef ESHKOL_TRANSFORMER_CONTINUOUS_BATCH_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_CONTINUOUS_BATCH_ESHKOL_BRIDGE_H

#include "eshkol_transformer/continuous_batch_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_cb_aot_ctx et_cb_aot_ctx;

/* Context lifecycle for compiled Eshkol AOT continuous-batching driver. */
int32_t et_cb_aot_create(int64_t max_running_seqs, int64_t max_waiting_seqs,
                         int64_t max_tokens_per_step, int64_t total_kv_blocks,
                         int64_t block_size, int64_t policy,
                         et_cb_aot_ctx **out_ctx);
void et_cb_aot_destroy(et_cb_aot_ctx *ctx);

/* Zero-heap-allocation per-step scheduling operations. */
int32_t et_cb_aot_enqueue(et_cb_aot_ctx *ctx, int64_t request_id,
                          int64_t prompt_len, int64_t max_new_tokens,
                          int64_t priority);
int32_t et_cb_aot_cancel(et_cb_aot_ctx *ctx, int64_t request_id);
int32_t et_cb_aot_step_plan(et_cb_aot_ctx *ctx);
int32_t et_cb_aot_commit_begin(et_cb_aot_ctx *ctx);
int32_t et_cb_aot_commit_set_eos(et_cb_aot_ctx *ctx, int64_t entry_index,
                                 int64_t hit_eos);
int32_t et_cb_aot_commit_set_raw(et_cb_aot_ctx *ctx, int64_t entry_index,
                                 int64_t request_id,
                                 int64_t generated_token_count,
                                 int64_t hit_eos);
int32_t et_cb_aot_step_commit(et_cb_aot_ctx *ctx, int64_t step_index,
                              int64_t item_count);
int32_t et_cb_aot_verify_invariants(const et_cb_aot_ctx *ctx);

/* Last planned batch accessors (zero heap allocations). */
int64_t et_cb_aot_plan_step_index(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_plan_entry_count(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_plan_total_tokens(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_plan_prefill_tokens(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_plan_decode_tokens(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_plan_preempted_count(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_plan_preempted_id(const et_cb_aot_ctx *ctx, int64_t index);
int64_t et_cb_aot_plan_used_blocks(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_plan_free_blocks(const et_cb_aot_ctx *ctx);

int64_t et_cb_aot_plan_entry_req_id(const et_cb_aot_ctx *ctx, int64_t index);
int64_t et_cb_aot_plan_entry_token_offset(const et_cb_aot_ctx *ctx,
                                          int64_t index);
int64_t et_cb_aot_plan_entry_chunk_tokens(const et_cb_aot_ctx *ctx,
                                          int64_t index);
int64_t et_cb_aot_plan_entry_is_prefill(const et_cb_aot_ctx *ctx,
                                        int64_t index);
int64_t et_cb_aot_plan_entry_produces_token(const et_cb_aot_ctx *ctx,
                                            int64_t index);
int64_t et_cb_aot_plan_entry_new_blocks(const et_cb_aot_ctx *ctx,
                                        int64_t index);
int64_t et_cb_aot_plan_entry_total_blocks(const et_cb_aot_ctx *ctx,
                                          int64_t index);

/* Aggregate scheduler stats accessors (zero heap allocations). */
int64_t et_cb_aot_current_step(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_running_count(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_waiting_count(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_finished_count(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_cancelled_count(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_used_blocks(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_free_blocks(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_total_preemptions(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_total_generated_tokens(const et_cb_aot_ctx *ctx);
int64_t et_cb_aot_total_scheduled_tokens(const et_cb_aot_ctx *ctx);

/* Per-request metric accessors (zero heap allocations). */
int64_t et_cb_aot_req_state(const et_cb_aot_ctx *ctx, int64_t request_id);
int64_t et_cb_aot_req_computed_tokens(const et_cb_aot_ctx *ctx,
                                      int64_t request_id);
int64_t et_cb_aot_req_generated_tokens(const et_cb_aot_ctx *ctx,
                                       int64_t request_id);
int64_t et_cb_aot_req_ttft_steps(const et_cb_aot_ctx *ctx, int64_t request_id);
int64_t et_cb_aot_req_completion_step(const et_cb_aot_ctx *ctx,
                                      int64_t request_id);
int64_t et_cb_aot_req_preempt_count(const et_cb_aot_ctx *ctx,
                                    int64_t request_id);
int64_t et_cb_aot_req_eos_stopped(const et_cb_aot_ctx *ctx, int64_t request_id);
int64_t et_cb_aot_req_allocated_blocks(const et_cb_aot_ctx *ctx,
                                       int64_t request_id);

#ifdef __cplusplus
}
#endif

#endif
