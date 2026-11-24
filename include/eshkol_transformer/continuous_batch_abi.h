#ifndef ESHKOL_TRANSFORMER_CONTINUOUS_BATCH_ABI_H
#define ESHKOL_TRANSFORMER_CONTINUOUS_BATCH_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_CB_ABI_MAJOR 1u
#define ET_CB_ABI_MINOR 0u

#define ET_CB_MAX_RUNNING_SEQS_LIMIT 64u
#define ET_CB_MAX_WAITING_SEQS_LIMIT 256u
#define ET_CB_MAX_TOKENS_PER_STEP_LIMIT 4096u
#define ET_CB_TOTAL_KV_BLOCKS_LIMIT 4096u
#define ET_CB_BLOCK_SIZE_LIMIT 64u

/* Fixed-capacity request table and waiting queue buffer (waiting + preempted
 * running sequences) enabling strictly zero heap allocations on all core
 * scheduler operations. */
#define ET_CB_MAX_REQUESTS_CAPACITY 512u
#define ET_CB_WAITING_QUEUE_CAPACITY \
  (ET_CB_MAX_WAITING_SEQS_LIMIT + ET_CB_MAX_RUNNING_SEQS_LIMIT)

#define ET_CB_STEP_NONE UINT32_MAX
#define ET_CB_BLOCK_NONE UINT16_MAX

typedef int32_t et_cb_status;

enum {
  ET_CB_OK = 0,
  ET_CB_ERR_ARGUMENT = -1,
  ET_CB_ERR_CONFIG = -2,
  ET_CB_ERR_REQUEST_TOO_LARGE = -3,
  ET_CB_ERR_DUPLICATE_ID = -4,
  ET_CB_ERR_QUEUE_FULL = -5,
  ET_CB_ERR_CAPACITY = -6,
  ET_CB_ERR_NOT_FOUND = -7,
  ET_CB_ERR_ALREADY_TERMINAL = -8,
  ET_CB_ERR_PLAN_PENDING = -9,
  ET_CB_ERR_NO_PENDING_PLAN = -10,
  ET_CB_ERR_COMMIT_MISMATCH = -11,
  ET_CB_ERR_INVARIANT = -12,
  ET_CB_ERR_NOMEM = -13
};

typedef enum et_cb_policy {
  ET_CB_POLICY_DECODE_FIRST_CHUNKED_PREFILL = 0,
  ET_CB_POLICY_PREFILL_FIRST_CHUNKED = 1
} et_cb_policy;

typedef enum et_cb_req_state {
  ET_CB_STATE_WAITING = 0,
  ET_CB_STATE_PREFILLING = 1,
  ET_CB_STATE_DECODING = 2,
  ET_CB_STATE_PREEMPTED = 3,
  ET_CB_STATE_FINISHED = 4,
  ET_CB_STATE_CANCELLED = 5
} et_cb_req_state;

/* Scheduler configuration.
 * Bounds:
 *   1 <= max_running_seqs    <= ET_CB_MAX_RUNNING_SEQS_LIMIT (64)
 *   1 <= max_waiting_seqs    <= ET_CB_MAX_WAITING_SEQS_LIMIT (256)
 *   1 <= max_tokens_per_step <= ET_CB_MAX_TOKENS_PER_STEP_LIMIT (4096)
 *   1 <= total_kv_blocks     <= ET_CB_TOTAL_KV_BLOCKS_LIMIT (4096)
 *   1 <= block_size          <= ET_CB_BLOCK_SIZE_LIMIT (64)
 *   policy in {DECODE_FIRST_CHUNKED_PREFILL, PREFILL_FIRST_CHUNKED}
 */
typedef struct et_cb_config {
  uint32_t max_running_seqs;
  uint32_t max_waiting_seqs;
  uint32_t max_tokens_per_step;
  uint32_t total_kv_blocks;
  uint32_t block_size;
  uint32_t policy;
} et_cb_config;

/* Request enqueue specification.
 * Admission requirement:
 *   prompt_len >= 1, max_new_tokens >= 1, reserved == 0,
 *   prompt_len + max_new_tokens - 1 <= total_kv_blocks * block_size
 *   (else ET_CB_ERR_REQUEST_TOO_LARGE, guaranteeing zero-deadlock progress).
 * Higher priority value = higher priority; ties break by older arrival order.
 */
typedef struct et_cb_request_spec {
  uint64_t request_id;
  uint32_t prompt_len;
  uint32_t max_new_tokens;
  int32_t priority;
  uint32_t reserved;
} et_cb_request_spec;

typedef struct et_cb_plan_entry {
  uint64_t request_id;
  uint32_t token_offset;
  uint32_t chunk_tokens;
  uint32_t is_prefill;
  uint32_t is_final_prefill_chunk;
  uint32_t produces_output_token;
  uint32_t new_kv_blocks;
  uint32_t total_kv_blocks;
  uint32_t reserved;
} et_cb_plan_entry;

typedef struct et_cb_step_batch {
  uint32_t step_index;
  uint32_t policy;
  uint32_t entry_count;
  uint32_t total_tokens;
  uint32_t prefill_tokens;
  uint32_t decode_tokens;
  uint32_t preempted_count;
  uint32_t used_kv_blocks;
  uint32_t free_kv_blocks;
  uint32_t active_running_seqs;
  uint32_t waiting_seqs;
  uint32_t reserved;
  uint64_t preempted_ids[ET_CB_MAX_RUNNING_SEQS_LIMIT];
  et_cb_plan_entry entries[ET_CB_MAX_RUNNING_SEQS_LIMIT];
} et_cb_step_batch;

typedef struct et_cb_commit_item {
  uint64_t request_id;
  uint32_t generated_token_count;
  uint32_t hit_eos;
} et_cb_commit_item;

typedef struct et_cb_request_info {
  uint64_t request_id;
  uint32_t state;
  uint32_t prompt_len;
  uint32_t max_new_tokens;
  int32_t priority;
  uint32_t arrival_seq;
  uint32_t enqueue_step;
  uint32_t computed_tokens;
  uint32_t generated_tokens;
  uint32_t allocated_blocks;
  uint32_t first_token_step;
  uint32_t completion_step;
  uint32_t ttft_steps;
  uint32_t total_latency_steps;
  uint32_t preempt_count;
  uint32_t eos_stopped;
  uint32_t reserved;
} et_cb_request_info;

typedef struct et_cb_stats {
  uint32_t current_step;
  uint32_t total_requests;
  uint32_t running_requests;
  uint32_t waiting_requests;
  uint32_t finished_requests;
  uint32_t cancelled_requests;
  uint32_t used_kv_blocks;
  uint32_t free_kv_blocks;
  uint32_t total_kv_blocks;
  uint32_t peak_used_kv_blocks;
  uint32_t peak_running_seqs;
  uint32_t has_pending_plan;
  uint64_t total_preemptions;
  uint64_t total_scheduled_tokens;
  uint64_t total_prefill_tokens;
  uint64_t total_decode_tokens;
  uint64_t total_generated_tokens;
} et_cb_stats;

/* Internal fixed-capacity request slot (exposed so callers can instantiate
 * et_cb_scheduler on stack or static storage with zero heap allocations). */
typedef struct et_cb_slot {
  uint64_t request_id;
  uint32_t prompt_len;
  uint32_t max_new_tokens;
  int32_t priority;
  uint32_t arrival_seq;
  uint32_t enqueue_step;
  uint32_t computed_tokens;
  uint32_t generated_tokens;
  uint32_t allocated_blocks;
  uint32_t first_token_step;
  uint32_t completion_step;
  uint32_t preempt_count;
  uint32_t state;
  uint32_t eos_stopped;
  uint32_t active_in_plan;
  uint32_t planned_chunk_tokens;
  uint32_t planned_is_prefill;
  uint32_t planned_produces_token;
  uint32_t planned_new_blocks;
} et_cb_slot;

typedef struct et_cb_scheduler {
  et_cb_config cfg;
  uint32_t initialized;
  uint32_t current_step;
  uint32_t next_arrival_seq;
  uint32_t slot_count;
  uint32_t running_count;
  uint32_t waiting_count;
  uint32_t finished_requests;
  uint32_t cancelled_requests;
  uint32_t used_kv_blocks;
  uint32_t free_kv_blocks;
  uint32_t peak_used_kv_blocks;
  uint32_t peak_running_seqs;
  uint32_t has_pending_plan;
  uint16_t free_block_head;
  uint16_t reserved_pad;
  uint64_t total_preemptions;
  uint64_t total_scheduled_tokens;
  uint64_t total_prefill_tokens;
  uint64_t total_decode_tokens;
  uint64_t total_generated_tokens;
  et_cb_step_batch pending_plan;
  et_cb_slot slots[ET_CB_MAX_REQUESTS_CAPACITY];
  uint16_t waiting_queue[ET_CB_WAITING_QUEUE_CAPACITY];
  uint16_t running_list[ET_CB_MAX_RUNNING_SEQS_LIMIT];
  uint16_t slot_block_head[ET_CB_MAX_REQUESTS_CAPACITY];
  uint16_t block_next[ET_CB_TOTAL_KV_BLOCKS_LIMIT];
  uint16_t block_owner[ET_CB_TOTAL_KV_BLOCKS_LIMIT];
} et_cb_scheduler;

#if defined(__cplusplus)
#define ET_CB_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_CB_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_CB_STATIC_ASSERT(sizeof(et_cb_config) == 24u &&
                        offsetof(et_cb_config, policy) == 20u,
                    "et_cb_config ABI layout changed");
ET_CB_STATIC_ASSERT(sizeof(et_cb_request_spec) == 24u &&
                        offsetof(et_cb_request_spec, priority) == 16u,
                    "et_cb_request_spec ABI layout changed");
ET_CB_STATIC_ASSERT(sizeof(et_cb_plan_entry) == 40u &&
                        offsetof(et_cb_plan_entry, total_kv_blocks) == 32u,
                    "et_cb_plan_entry ABI layout changed");
ET_CB_STATIC_ASSERT(sizeof(et_cb_step_batch) == 3120u &&
                        offsetof(et_cb_step_batch, preempted_ids) == 48u &&
                        offsetof(et_cb_step_batch, entries) == 560u,
                    "et_cb_step_batch ABI layout changed");
ET_CB_STATIC_ASSERT(sizeof(et_cb_commit_item) == 16u &&
                        offsetof(et_cb_commit_item, hit_eos) == 12u,
                    "et_cb_commit_item ABI layout changed");
ET_CB_STATIC_ASSERT(sizeof(et_cb_request_info) == 72u &&
                        offsetof(et_cb_request_info, preempt_count) == 60u,
                    "et_cb_request_info ABI layout changed");
ET_CB_STATIC_ASSERT(sizeof(et_cb_stats) == 88u &&
                        offsetof(et_cb_stats, total_preemptions) == 48u &&
                        offsetof(et_cb_stats, total_generated_tokens) == 80u,
                    "et_cb_stats ABI layout changed");
#undef ET_CB_STATIC_ASSERT

/* Validate scheduler configuration bounds and policy. */
et_cb_status et_cb_validate_config(const et_cb_config *cfg);

/* Initialize a caller-provided scheduler instance (zero heap allocations). */
et_cb_status et_cb_init(et_cb_scheduler *sched, const et_cb_config *cfg);

/* Enqueue a new request into the waiting queue (zero heap allocations).
 * Rejects invalid specs, oversized requests (ET_CB_ERR_REQUEST_TOO_LARGE),
 * duplicate request_ids (ET_CB_ERR_DUPLICATE_ID), full waiting queues
 * (ET_CB_ERR_QUEUE_FULL), and exhausted table capacity (ET_CB_ERR_CAPACITY). */
et_cb_status et_cb_enqueue(et_cb_scheduler *sched,
                           const et_cb_request_spec *spec);

/* Cancel an active (waiting, preempted, prefilling, or decoding) request and
 * immediately release any held KV blocks (zero heap allocations).
 * Rejects unknown IDs (ET_CB_ERR_NOT_FOUND), terminal/double-cancel requests
 * (ET_CB_ERR_ALREADY_TERMINAL), and requests active in an uncommitted plan
 * (ET_CB_ERR_PLAN_PENDING). */
et_cb_status et_cb_cancel(et_cb_scheduler *sched, uint64_t request_id);

/* Plan the next continuous-batching + chunked-prefill step (zero heap
 * allocations). Fails with ET_CB_ERR_PLAN_PENDING if the previous plan has not
 * yet been committed. */
et_cb_status et_cb_step_plan(et_cb_scheduler *sched,
                             et_cb_step_batch *out_plan);

/* Atomically validate and commit the pending step plan (zero heap allocations).
 * On any mismatch (step_index, item_count, request_id order,
 * generated_token_count, or EOS on non-output step), leaves scheduler state
 * completely untouched and returns ET_CB_ERR_COMMIT_MISMATCH. */
et_cb_status et_cb_step_commit(et_cb_scheduler *sched, uint32_t step_index,
                               const et_cb_commit_item *items,
                               uint32_t item_count);

/* Query per-request lifecycle and latency metrics (zero heap allocations). */
et_cb_status et_cb_get_request_info(const et_cb_scheduler *sched,
                                    uint64_t request_id,
                                    et_cb_request_info *out_info);

/* Query aggregate scheduler metrics and KV-block accounting (zero heap
 * allocations). */
et_cb_status et_cb_get_stats(const et_cb_scheduler *sched,
                             et_cb_stats *out_stats);

/* Verify all KV-block conservation, ownership, queue disjointness, and state
 * machine invariants (zero heap allocations). */
et_cb_status et_cb_verify_invariants(const et_cb_scheduler *sched);

#ifdef __cplusplus
}
#endif

#endif
