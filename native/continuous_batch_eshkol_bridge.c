#include "continuous_batch_eshkol_bridge.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct et_cb_aot_ctx {
  et_cb_scheduler sched;
  et_cb_step_batch last_plan;
  et_cb_commit_item commit_items[ET_CB_MAX_RUNNING_SEQS_LIMIT];
};

static int u32_from_i64(int64_t v, uint32_t *out) {
  if (v < 0 || v > (int64_t)UINT32_MAX)
    return 0;
  *out = (uint32_t)v;
  return 1;
}

int32_t et_cb_aot_create(int64_t max_running_seqs, int64_t max_waiting_seqs,
                         int64_t max_tokens_per_step, int64_t total_kv_blocks,
                         int64_t block_size, int64_t policy,
                         et_cb_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_CB_ERR_ARGUMENT;
  *out_ctx = NULL;

  et_cb_config cfg;
  if (!u32_from_i64(max_running_seqs, &cfg.max_running_seqs) ||
      !u32_from_i64(max_waiting_seqs, &cfg.max_waiting_seqs) ||
      !u32_from_i64(max_tokens_per_step, &cfg.max_tokens_per_step) ||
      !u32_from_i64(total_kv_blocks, &cfg.total_kv_blocks) ||
      !u32_from_i64(block_size, &cfg.block_size) ||
      !u32_from_i64(policy, &cfg.policy))
    return ET_CB_ERR_CONFIG;

  const et_cb_status vrc = et_cb_validate_config(&cfg);
  if (vrc != ET_CB_OK)
    return vrc;

  et_cb_aot_ctx *ctx = (et_cb_aot_ctx *)calloc(1u, sizeof(*ctx));
  if (ctx == NULL)
    return ET_CB_ERR_NOMEM;

  const et_cb_status irc = et_cb_init(&ctx->sched, &cfg);
  if (irc != ET_CB_OK) {
    free(ctx);
    return irc;
  }

  *out_ctx = ctx;
  return ET_CB_OK;
}

void et_cb_aot_destroy(et_cb_aot_ctx *ctx) { free(ctx); }

int32_t et_cb_aot_enqueue(et_cb_aot_ctx *ctx, int64_t request_id,
                          int64_t prompt_len, int64_t max_new_tokens,
                          int64_t priority) {
  if (ctx == NULL || request_id < 0 || priority < (int64_t)INT32_MIN ||
      priority > (int64_t)INT32_MAX)
    return ET_CB_ERR_ARGUMENT;

  et_cb_request_spec spec;
  spec.request_id = (uint64_t)request_id;
  if (!u32_from_i64(prompt_len, &spec.prompt_len) ||
      !u32_from_i64(max_new_tokens, &spec.max_new_tokens))
    return ET_CB_ERR_ARGUMENT;
  spec.priority = (int32_t)priority;
  spec.reserved = 0u;

  return et_cb_enqueue(&ctx->sched, &spec);
}

int32_t et_cb_aot_cancel(et_cb_aot_ctx *ctx, int64_t request_id) {
  if (ctx == NULL || request_id < 0)
    return ET_CB_ERR_ARGUMENT;
  return et_cb_cancel(&ctx->sched, (uint64_t)request_id);
}

int32_t et_cb_aot_step_plan(et_cb_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_CB_ERR_ARGUMENT;
  return et_cb_step_plan(&ctx->sched, &ctx->last_plan);
}

int32_t et_cb_aot_commit_begin(et_cb_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_CB_ERR_ARGUMENT;
  const uint32_t n = ctx->last_plan.entry_count;
  if (n > ET_CB_MAX_RUNNING_SEQS_LIMIT)
    return ET_CB_ERR_INVARIANT;
  memset(ctx->commit_items, 0, sizeof(ctx->commit_items));
  for (uint32_t i = 0u; i < n; ++i) {
    ctx->commit_items[i].request_id = ctx->last_plan.entries[i].request_id;
    ctx->commit_items[i].generated_token_count =
        ctx->last_plan.entries[i].produces_output_token;
    ctx->commit_items[i].hit_eos = 0u;
  }
  return ET_CB_OK;
}

int32_t et_cb_aot_commit_set_eos(et_cb_aot_ctx *ctx, int64_t entry_index,
                                 int64_t hit_eos) {
  if (ctx == NULL || entry_index < 0 ||
      entry_index >= (int64_t)ET_CB_MAX_RUNNING_SEQS_LIMIT || hit_eos < 0 ||
      hit_eos > (int64_t)UINT32_MAX)
    return ET_CB_ERR_ARGUMENT;
  ctx->commit_items[entry_index].hit_eos = (uint32_t)hit_eos;
  return ET_CB_OK;
}

int32_t et_cb_aot_commit_set_raw(et_cb_aot_ctx *ctx, int64_t entry_index,
                                 int64_t request_id,
                                 int64_t generated_token_count,
                                 int64_t hit_eos) {
  if (ctx == NULL || entry_index < 0 ||
      entry_index >= (int64_t)ET_CB_MAX_RUNNING_SEQS_LIMIT || request_id < 0 ||
      generated_token_count < 0 || generated_token_count > (int64_t)UINT32_MAX ||
      hit_eos < 0 || hit_eos > (int64_t)UINT32_MAX)
    return ET_CB_ERR_ARGUMENT;
  ctx->commit_items[entry_index].request_id = (uint64_t)request_id;
  ctx->commit_items[entry_index].generated_token_count =
      (uint32_t)generated_token_count;
  ctx->commit_items[entry_index].hit_eos = (uint32_t)hit_eos;
  return ET_CB_OK;
}

int32_t et_cb_aot_step_commit(et_cb_aot_ctx *ctx, int64_t step_index,
                              int64_t item_count) {
  if (ctx == NULL || step_index < 0 || step_index > (int64_t)UINT32_MAX ||
      item_count < 0 || item_count > (int64_t)ET_CB_MAX_RUNNING_SEQS_LIMIT)
    return ET_CB_ERR_ARGUMENT;
  return et_cb_step_commit(&ctx->sched, (uint32_t)step_index,
                           ctx->commit_items, (uint32_t)item_count);
}

int32_t et_cb_aot_verify_invariants(const et_cb_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_CB_ERR_ARGUMENT;
  return et_cb_verify_invariants(&ctx->sched);
}

int64_t et_cb_aot_plan_step_index(const et_cb_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->last_plan.step_index;
}

int64_t et_cb_aot_plan_entry_count(const et_cb_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->last_plan.entry_count;
}

int64_t et_cb_aot_plan_total_tokens(const et_cb_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->last_plan.total_tokens;
}

int64_t et_cb_aot_plan_prefill_tokens(const et_cb_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->last_plan.prefill_tokens;
}

int64_t et_cb_aot_plan_decode_tokens(const et_cb_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->last_plan.decode_tokens;
}

int64_t et_cb_aot_plan_preempted_count(const et_cb_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->last_plan.preempted_count;
}

int64_t et_cb_aot_plan_preempted_id(const et_cb_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 ||
      index >= (int64_t)ctx->last_plan.preempted_count)
    return -1;
  return (int64_t)ctx->last_plan.preempted_ids[index];
}

int64_t et_cb_aot_plan_used_blocks(const et_cb_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->last_plan.used_kv_blocks;
}

int64_t et_cb_aot_plan_free_blocks(const et_cb_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->last_plan.free_kv_blocks;
}

static const et_cb_plan_entry *get_plan_entry(const et_cb_aot_ctx *ctx,
                                              int64_t index) {
  if (ctx == NULL || index < 0 || index >= (int64_t)ctx->last_plan.entry_count)
    return NULL;
  return &ctx->last_plan.entries[index];
}

int64_t et_cb_aot_plan_entry_req_id(const et_cb_aot_ctx *ctx, int64_t index) {
  const et_cb_plan_entry *e = get_plan_entry(ctx, index);
  return e == NULL ? -1 : (int64_t)e->request_id;
}

int64_t et_cb_aot_plan_entry_token_offset(const et_cb_aot_ctx *ctx,
                                          int64_t index) {
  const et_cb_plan_entry *e = get_plan_entry(ctx, index);
  return e == NULL ? -1 : (int64_t)e->token_offset;
}

int64_t et_cb_aot_plan_entry_chunk_tokens(const et_cb_aot_ctx *ctx,
                                          int64_t index) {
  const et_cb_plan_entry *e = get_plan_entry(ctx, index);
  return e == NULL ? -1 : (int64_t)e->chunk_tokens;
}

int64_t et_cb_aot_plan_entry_is_prefill(const et_cb_aot_ctx *ctx,
                                        int64_t index) {
  const et_cb_plan_entry *e = get_plan_entry(ctx, index);
  return e == NULL ? -1 : (int64_t)e->is_prefill;
}

int64_t et_cb_aot_plan_entry_produces_token(const et_cb_aot_ctx *ctx,
                                            int64_t index) {
  const et_cb_plan_entry *e = get_plan_entry(ctx, index);
  return e == NULL ? -1 : (int64_t)e->produces_output_token;
}

int64_t et_cb_aot_plan_entry_new_blocks(const et_cb_aot_ctx *ctx,
                                        int64_t index) {
  const et_cb_plan_entry *e = get_plan_entry(ctx, index);
  return e == NULL ? -1 : (int64_t)e->new_kv_blocks;
}

int64_t et_cb_aot_plan_entry_total_blocks(const et_cb_aot_ctx *ctx,
                                          int64_t index) {
  const et_cb_plan_entry *e = get_plan_entry(ctx, index);
  return e == NULL ? -1 : (int64_t)e->total_kv_blocks;
}

int64_t et_cb_aot_current_step(const et_cb_aot_ctx *ctx) {
  et_cb_stats s;
  if (ctx == NULL || et_cb_get_stats(&ctx->sched, &s) != ET_CB_OK)
    return -1;
  return (int64_t)s.current_step;
}

int64_t et_cb_aot_running_count(const et_cb_aot_ctx *ctx) {
  et_cb_stats s;
  if (ctx == NULL || et_cb_get_stats(&ctx->sched, &s) != ET_CB_OK)
    return -1;
  return (int64_t)s.running_requests;
}

int64_t et_cb_aot_waiting_count(const et_cb_aot_ctx *ctx) {
  et_cb_stats s;
  if (ctx == NULL || et_cb_get_stats(&ctx->sched, &s) != ET_CB_OK)
    return -1;
  return (int64_t)s.waiting_requests;
}

int64_t et_cb_aot_finished_count(const et_cb_aot_ctx *ctx) {
  et_cb_stats s;
  if (ctx == NULL || et_cb_get_stats(&ctx->sched, &s) != ET_CB_OK)
    return -1;
  return (int64_t)s.finished_requests;
}

int64_t et_cb_aot_cancelled_count(const et_cb_aot_ctx *ctx) {
  et_cb_stats s;
  if (ctx == NULL || et_cb_get_stats(&ctx->sched, &s) != ET_CB_OK)
    return -1;
  return (int64_t)s.cancelled_requests;
}

int64_t et_cb_aot_used_blocks(const et_cb_aot_ctx *ctx) {
  et_cb_stats s;
  if (ctx == NULL || et_cb_get_stats(&ctx->sched, &s) != ET_CB_OK)
    return -1;
  return (int64_t)s.used_kv_blocks;
}

int64_t et_cb_aot_free_blocks(const et_cb_aot_ctx *ctx) {
  et_cb_stats s;
  if (ctx == NULL || et_cb_get_stats(&ctx->sched, &s) != ET_CB_OK)
    return -1;
  return (int64_t)s.free_kv_blocks;
}

int64_t et_cb_aot_total_preemptions(const et_cb_aot_ctx *ctx) {
  et_cb_stats s;
  if (ctx == NULL || et_cb_get_stats(&ctx->sched, &s) != ET_CB_OK)
    return -1;
  return (int64_t)s.total_preemptions;
}

int64_t et_cb_aot_total_generated_tokens(const et_cb_aot_ctx *ctx) {
  et_cb_stats s;
  if (ctx == NULL || et_cb_get_stats(&ctx->sched, &s) != ET_CB_OK)
    return -1;
  return (int64_t)s.total_generated_tokens;
}

int64_t et_cb_aot_total_scheduled_tokens(const et_cb_aot_ctx *ctx) {
  et_cb_stats s;
  if (ctx == NULL || et_cb_get_stats(&ctx->sched, &s) != ET_CB_OK)
    return -1;
  return (int64_t)s.total_scheduled_tokens;
}

static int fetch_req_info(const et_cb_aot_ctx *ctx, int64_t request_id,
                          et_cb_request_info *out) {
  if (ctx == NULL || request_id < 0)
    return 0;
  return et_cb_get_request_info(&ctx->sched, (uint64_t)request_id, out) ==
         ET_CB_OK;
}

int64_t et_cb_aot_req_state(const et_cb_aot_ctx *ctx, int64_t request_id) {
  et_cb_request_info info;
  return fetch_req_info(ctx, request_id, &info) ? (int64_t)info.state : -1;
}

int64_t et_cb_aot_req_computed_tokens(const et_cb_aot_ctx *ctx,
                                      int64_t request_id) {
  et_cb_request_info info;
  return fetch_req_info(ctx, request_id, &info) ? (int64_t)info.computed_tokens
                                                : -1;
}

int64_t et_cb_aot_req_generated_tokens(const et_cb_aot_ctx *ctx,
                                       int64_t request_id) {
  et_cb_request_info info;
  return fetch_req_info(ctx, request_id, &info) ? (int64_t)info.generated_tokens
                                                : -1;
}

int64_t et_cb_aot_req_ttft_steps(const et_cb_aot_ctx *ctx, int64_t request_id) {
  et_cb_request_info info;
  return fetch_req_info(ctx, request_id, &info) ? (int64_t)info.ttft_steps : -1;
}

int64_t et_cb_aot_req_completion_step(const et_cb_aot_ctx *ctx,
                                      int64_t request_id) {
  et_cb_request_info info;
  if (!fetch_req_info(ctx, request_id, &info) ||
      info.completion_step == ET_CB_STEP_NONE)
    return -1;
  return (int64_t)info.completion_step;
}

int64_t et_cb_aot_req_preempt_count(const et_cb_aot_ctx *ctx,
                                    int64_t request_id) {
  et_cb_request_info info;
  return fetch_req_info(ctx, request_id, &info) ? (int64_t)info.preempt_count
                                                : -1;
}

int64_t et_cb_aot_req_eos_stopped(const et_cb_aot_ctx *ctx,
                                  int64_t request_id) {
  et_cb_request_info info;
  return fetch_req_info(ctx, request_id, &info) ? (int64_t)info.eos_stopped
                                                : -1;
}

int64_t et_cb_aot_req_allocated_blocks(const et_cb_aot_ctx *ctx,
                                       int64_t request_id) {
  et_cb_request_info info;
  return fetch_req_info(ctx, request_id, &info) ? (int64_t)info.allocated_blocks
                                                : -1;
}
