#include "eshkol_transformer/continuous_batch_abi.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define ET_CB_INIT_MAGIC 0x45544342u /* "ETCB" */

static uint32_t ceil_div_u32(uint32_t num, uint32_t den) {
  if (num == 0u)
    return 0u;
  return (num - 1u) / den + 1u;
}

static int sched_valid(const et_cb_scheduler *sched) {
  return sched != NULL && sched->initialized == ET_CB_INIT_MAGIC &&
         et_cb_validate_config(&sched->cfg) == ET_CB_OK;
}

/* Returns 1 if slot `a` has strictly higher scheduling priority than slot `b`
 * (higher priority field wins; ties break by smaller arrival_seq). */
static int slot_higher_rank(const et_cb_slot *a, const et_cb_slot *b) {
  if (a->priority != b->priority)
    return a->priority > b->priority;
  return a->arrival_seq < b->arrival_seq;
}

/* Returns index in sched->slots, or -1 if not found. */
static int32_t find_slot_by_id(const et_cb_scheduler *sched,
                               uint64_t request_id) {
  for (uint32_t i = 0u; i < sched->slot_count; ++i) {
    if (sched->slots[i].request_id == request_id)
      return (int32_t)i;
  }
  return -1;
}

static et_cb_status alloc_kv_blocks(et_cb_scheduler *sched, uint16_t slot_idx,
                                    uint32_t count) {
  if (count == 0u)
    return ET_CB_OK;
  if (count > sched->free_kv_blocks)
    return ET_CB_ERR_INVARIANT;

  for (uint32_t i = 0u; i < count; ++i) {
    const uint16_t blk = sched->free_block_head;
    if (blk == ET_CB_BLOCK_NONE || blk >= sched->cfg.total_kv_blocks)
      return ET_CB_ERR_INVARIANT;
    if (sched->block_owner[blk] != ET_CB_BLOCK_NONE)
      return ET_CB_ERR_INVARIANT;
    sched->free_block_head = sched->block_next[blk];
    sched->block_owner[blk] = slot_idx;
    sched->block_next[blk] = sched->slot_block_head[slot_idx];
    sched->slot_block_head[slot_idx] = blk;
    sched->free_kv_blocks -= 1u;
    sched->used_kv_blocks += 1u;
    sched->slots[slot_idx].allocated_blocks += 1u;
  }
  if (sched->used_kv_blocks > sched->peak_used_kv_blocks)
    sched->peak_used_kv_blocks = sched->used_kv_blocks;
  return ET_CB_OK;
}

static et_cb_status release_all_kv_blocks(et_cb_scheduler *sched,
                                          uint16_t slot_idx) {
  uint32_t released = 0u;
  const uint32_t expected = sched->slots[slot_idx].allocated_blocks;
  uint16_t blk = sched->slot_block_head[slot_idx];

  while (blk != ET_CB_BLOCK_NONE) {
    if (blk >= sched->cfg.total_kv_blocks ||
        sched->block_owner[blk] != slot_idx ||
        released >= sched->cfg.total_kv_blocks)
      return ET_CB_ERR_INVARIANT;
    const uint16_t next = sched->block_next[blk];
    sched->block_owner[blk] = ET_CB_BLOCK_NONE;
    sched->block_next[blk] = sched->free_block_head;
    sched->free_block_head = blk;
    sched->used_kv_blocks -= 1u;
    sched->free_kv_blocks += 1u;
    released += 1u;
    blk = next;
  }

  if (released != expected)
    return ET_CB_ERR_INVARIANT;

  sched->slot_block_head[slot_idx] = ET_CB_BLOCK_NONE;
  sched->slots[slot_idx].allocated_blocks = 0u;
  return ET_CB_OK;
}

static int remove_from_list(uint16_t *list, uint32_t *count,
                            uint16_t slot_idx) {
  for (uint32_t i = 0u; i < *count; ++i) {
    if (list[i] == slot_idx) {
      for (uint32_t j = i + 1u; j < *count; ++j)
        list[j - 1u] = list[j];
      *count -= 1u;
      return 1;
    }
  }
  return 0;
}

static et_cb_status push_waiting_front(et_cb_scheduler *sched,
                                       uint16_t slot_idx) {
  if (sched->waiting_count >= ET_CB_WAITING_QUEUE_CAPACITY)
    return ET_CB_ERR_INVARIANT;
  for (uint32_t i = sched->waiting_count; i > 0u; --i)
    sched->waiting_queue[i] = sched->waiting_queue[i - 1u];
  sched->waiting_queue[0] = slot_idx;
  sched->waiting_count += 1u;
  return ET_CB_OK;
}

static uint16_t pop_waiting_front(et_cb_scheduler *sched) {
  const uint16_t slot_idx = sched->waiting_queue[0];
  for (uint32_t i = 1u; i < sched->waiting_count; ++i)
    sched->waiting_queue[i - 1u] = sched->waiting_queue[i];
  sched->waiting_count -= 1u;
  return slot_idx;
}

/* Sort an array of running slot indices by highest rank first
 * (priority DESC, arrival_seq ASC) via deterministic insertion sort. */
static void sort_slots_by_rank_desc(const et_cb_scheduler *sched,
                                    uint16_t *arr, uint32_t n) {
  for (uint32_t i = 1u; i < n; ++i) {
    const uint16_t key = arr[i];
    uint32_t j = i;
    while (j > 0u &&
           slot_higher_rank(&sched->slots[key], &sched->slots[arr[j - 1u]])) {
      arr[j] = arr[j - 1u];
      j -= 1u;
    }
    arr[j] = key;
  }
}

/* Find the lowest-rank (lowest priority, newest arrival) running sequence that
 * is NOT already scheduled in the current step's pending plan. Returns -1 if
 * no eligible victim exists. */
static int32_t find_preemption_victim(const et_cb_scheduler *sched) {
  int32_t victim = -1;
  for (uint32_t i = 0u; i < sched->running_count; ++i) {
    const uint16_t idx = sched->running_list[i];
    const et_cb_slot *cand = &sched->slots[idx];
    if (cand->active_in_plan != 0u)
      continue;
    if (victim < 0) {
      victim = (int32_t)idx;
    } else {
      const et_cb_slot *cur = &sched->slots[victim];
      /* Lower rank means cur has strictly higher rank than cand. */
      if (slot_higher_rank(cur, cand))
        victim = (int32_t)idx;
    }
  }
  return victim;
}

static et_cb_status preempt_slot(et_cb_scheduler *sched, uint16_t slot_idx,
                                 et_cb_step_batch *plan) {
  et_cb_slot *slot = &sched->slots[slot_idx];
  if (slot->active_in_plan != 0u)
    return ET_CB_ERR_INVARIANT;

  const et_cb_status rel = release_all_kv_blocks(sched, slot_idx);
  if (rel != ET_CB_OK)
    return rel;

  if (!remove_from_list(sched->running_list, &sched->running_count, slot_idx))
    return ET_CB_ERR_INVARIANT;

  /* Preserve prompt_len, max_new_tokens, generated_tokens, first_token_step,
   * arrival_seq, priority; reset computed_tokens for recomputation prefill. */
  slot->computed_tokens = 0u;
  slot->state = ET_CB_STATE_PREEMPTED;
  slot->preempt_count += 1u;
  sched->total_preemptions += 1u;

  const et_cb_status push_rc = push_waiting_front(sched, slot_idx);
  if (push_rc != ET_CB_OK)
    return push_rc;

  if (plan->preempted_count < ET_CB_MAX_RUNNING_SEQS_LIMIT) {
    plan->preempted_ids[plan->preempted_count] = slot->request_id;
    plan->preempted_count += 1u;
  }
  return ET_CB_OK;
}

static et_cb_status append_plan_entry(et_cb_scheduler *sched,
                                      et_cb_step_batch *plan, uint16_t slot_idx,
                                      uint32_t chunk_tokens,
                                      uint32_t is_prefill,
                                      uint32_t is_final_prefill,
                                      uint32_t new_blocks) {
  if (plan->entry_count >= ET_CB_MAX_RUNNING_SEQS_LIMIT)
    return ET_CB_ERR_INVARIANT;
  et_cb_slot *slot = &sched->slots[slot_idx];
  const uint32_t produces = (!is_prefill || is_final_prefill) ? 1u : 0u;

  et_cb_plan_entry *e = &plan->entries[plan->entry_count];
  e->request_id = slot->request_id;
  e->token_offset = slot->computed_tokens;
  e->chunk_tokens = chunk_tokens;
  e->is_prefill = is_prefill;
  e->is_final_prefill_chunk = is_final_prefill;
  e->produces_output_token = produces;
  e->new_kv_blocks = new_blocks;
  e->total_kv_blocks = slot->allocated_blocks;
  e->reserved = 0u;

  plan->entry_count += 1u;
  plan->total_tokens += chunk_tokens;
  if (is_prefill)
    plan->prefill_tokens += chunk_tokens;
  else
    plan->decode_tokens += chunk_tokens;

  slot->active_in_plan = 1u;
  slot->planned_chunk_tokens = chunk_tokens;
  slot->planned_is_prefill = is_prefill;
  slot->planned_produces_token = produces;
  slot->planned_new_blocks = new_blocks;
  return ET_CB_OK;
}

static et_cb_status plan_running_decodes(et_cb_scheduler *sched,
                                         et_cb_step_batch *plan,
                                         uint32_t *remaining_budget) {
  uint16_t decodes[ET_CB_MAX_RUNNING_SEQS_LIMIT];
  uint32_t decode_count = 0u;

  for (uint32_t i = 0u; i < sched->running_count; ++i) {
    const uint16_t idx = sched->running_list[i];
    if (sched->slots[idx].state == ET_CB_STATE_DECODING) {
      decodes[decode_count] = idx;
      decode_count += 1u;
    }
  }
  sort_slots_by_rank_desc(sched, decodes, decode_count);

  for (uint32_t i = 0u; i < decode_count; ++i) {
    if (*remaining_budget < 1u)
      break;
    const uint16_t idx = decodes[i];
    et_cb_slot *slot = &sched->slots[idx];
    /* Sequence might have been preempted earlier in this loop. */
    if (slot->state != ET_CB_STATE_DECODING)
      continue;

    const uint32_t new_blocks =
        (slot->computed_tokens % sched->cfg.block_size == 0u) ? 1u : 0u;

    if (new_blocks > sched->free_kv_blocks) {
      /* KV blocks exhausted during decode planning: deterministically preempt
       * the lowest-priority / newest unscheduled running sequence. */
      const int32_t victim = find_preemption_victim(sched);
      if (victim < 0)
        continue;
      /* In PREFILL_FIRST policy, do not self-preempt if another sequence
       * already made progress this step and has lower/equal rank? Actually,
       * if victim == idx and plan->entry_count > 0, deferring idx avoids
       * unnecessary eviction while the already-planned prefill finishes. */
      if ((uint16_t)victim == idx && plan->entry_count > 0u)
        continue;

      const et_cb_status prc = preempt_slot(sched, (uint16_t)victim, plan);
      if (prc != ET_CB_OK)
        return prc;
      if ((uint16_t)victim == idx)
        continue;
    }

    if (new_blocks > sched->free_kv_blocks)
      return ET_CB_ERR_INVARIANT;

    const et_cb_status arc = alloc_kv_blocks(sched, idx, new_blocks);
    if (arc != ET_CB_OK)
      return arc;

    const et_cb_status erc =
        append_plan_entry(sched, plan, idx, 1u, 0u, 0u, new_blocks);
    if (erc != ET_CB_OK)
      return erc;

    *remaining_budget -= 1u;
  }
  return ET_CB_OK;
}

static et_cb_status plan_continuing_prefills(et_cb_scheduler *sched,
                                             et_cb_step_batch *plan,
                                             uint32_t *remaining_budget) {
  uint16_t prefills[ET_CB_MAX_RUNNING_SEQS_LIMIT];
  uint32_t prefill_count = 0u;

  for (uint32_t i = 0u; i < sched->running_count; ++i) {
    const uint16_t idx = sched->running_list[i];
    if (sched->slots[idx].state == ET_CB_STATE_PREFILLING) {
      prefills[prefill_count] = idx;
      prefill_count += 1u;
    }
  }
  sort_slots_by_rank_desc(sched, prefills, prefill_count);

  for (uint32_t i = 0u; i < prefill_count; ++i) {
    if (*remaining_budget == 0u)
      break;
    const uint16_t idx = prefills[i];
    et_cb_slot *slot = &sched->slots[idx];
    if (slot->state != ET_CB_STATE_PREFILLING)
      continue;

    const uint32_t target_prefill = slot->prompt_len + slot->generated_tokens;
    if (slot->computed_tokens >= target_prefill)
      return ET_CB_ERR_INVARIANT;
    const uint32_t remaining_prefill = target_prefill - slot->computed_tokens;

    const uint32_t allocated_cap =
        slot->allocated_blocks * sched->cfg.block_size;
    if (allocated_cap < slot->computed_tokens)
      return ET_CB_ERR_INVARIANT;
    const uint32_t slack = allocated_cap - slot->computed_tokens;
    uint32_t max_fit = slack + sched->free_kv_blocks * sched->cfg.block_size;

    if (max_fit == 0u) {
      /* Continuing prefill needs >= 1 new KV block and free pool is empty. */
      const int32_t victim = find_preemption_victim(sched);
      if (victim < 0)
        continue;
      /* If idx itself is the lowest-rank unscheduled running sequence and
       * other work is already scheduled this step, defer rather than evict. */
      if ((uint16_t)victim == idx && plan->entry_count > 0u)
        continue;
      const et_cb_status prc = preempt_slot(sched, (uint16_t)victim, plan);
      if (prc != ET_CB_OK)
        return prc;
      if ((uint16_t)victim == idx)
        continue;
      max_fit = slack + sched->free_kv_blocks * sched->cfg.block_size;
    }

    if (max_fit == 0u)
      continue;

    uint32_t chunk = remaining_prefill;
    if (*remaining_budget < chunk)
      chunk = *remaining_budget;
    if (max_fit < chunk)
      chunk = max_fit;

    const uint32_t total_needed_blocks =
        ceil_div_u32(slot->computed_tokens + chunk, sched->cfg.block_size);
    if (total_needed_blocks < slot->allocated_blocks)
      return ET_CB_ERR_INVARIANT;
    const uint32_t new_blocks = total_needed_blocks - slot->allocated_blocks;
    if (new_blocks > sched->free_kv_blocks)
      return ET_CB_ERR_INVARIANT;

    const et_cb_status arc = alloc_kv_blocks(sched, idx, new_blocks);
    if (arc != ET_CB_OK)
      return arc;

    const uint32_t is_final =
        (slot->computed_tokens + chunk == target_prefill) ? 1u : 0u;
    const et_cb_status erc =
        append_plan_entry(sched, plan, idx, chunk, 1u, is_final, new_blocks);
    if (erc != ET_CB_OK)
      return erc;

    *remaining_budget -= chunk;
  }
  return ET_CB_OK;
}

static et_cb_status admit_waiting_prefills(et_cb_scheduler *sched,
                                           et_cb_step_batch *plan,
                                           uint32_t *remaining_budget) {
  /* Never admit from waiting queue in a step where preemption occurred,
   * preventing eviction/re-admission thrashing and guaranteeing forward
   * progress of higher-rank running sequences. */
  if (plan->preempted_count > 0u)
    return ET_CB_OK;

  while (sched->waiting_count > 0u &&
         sched->running_count < sched->cfg.max_running_seqs &&
         *remaining_budget > 0u && sched->free_kv_blocks > 0u) {
    const uint16_t idx = pop_waiting_front(sched);
    et_cb_slot *slot = &sched->slots[idx];

    const uint32_t target_prefill = slot->prompt_len + slot->generated_tokens;
    const uint32_t max_fit = sched->free_kv_blocks * sched->cfg.block_size;
    uint32_t chunk = target_prefill;
    if (*remaining_budget < chunk)
      chunk = *remaining_budget;
    if (max_fit < chunk)
      chunk = max_fit;

    const uint32_t new_blocks = ceil_div_u32(chunk, sched->cfg.block_size);
    if (chunk == 0u || new_blocks == 0u || new_blocks > sched->free_kv_blocks)
      return ET_CB_ERR_INVARIANT;

    slot->state = ET_CB_STATE_PREFILLING;
    sched->running_list[sched->running_count] = idx;
    sched->running_count += 1u;
    if (sched->running_count > sched->peak_running_seqs)
      sched->peak_running_seqs = sched->running_count;

    const et_cb_status arc = alloc_kv_blocks(sched, idx, new_blocks);
    if (arc != ET_CB_OK)
      return arc;

    const uint32_t is_final = (chunk == target_prefill) ? 1u : 0u;
    const et_cb_status erc =
        append_plan_entry(sched, plan, idx, chunk, 1u, is_final, new_blocks);
    if (erc != ET_CB_OK)
      return erc;

    *remaining_budget -= chunk;
  }
  return ET_CB_OK;
}

et_cb_status et_cb_validate_config(const et_cb_config *cfg) {
  if (cfg == NULL)
    return ET_CB_ERR_ARGUMENT;
  if (cfg->max_running_seqs < 1u ||
      cfg->max_running_seqs > ET_CB_MAX_RUNNING_SEQS_LIMIT)
    return ET_CB_ERR_CONFIG;
  if (cfg->max_waiting_seqs < 1u ||
      cfg->max_waiting_seqs > ET_CB_MAX_WAITING_SEQS_LIMIT)
    return ET_CB_ERR_CONFIG;
  if (cfg->max_tokens_per_step < 1u ||
      cfg->max_tokens_per_step > ET_CB_MAX_TOKENS_PER_STEP_LIMIT)
    return ET_CB_ERR_CONFIG;
  if (cfg->total_kv_blocks < 1u ||
      cfg->total_kv_blocks > ET_CB_TOTAL_KV_BLOCKS_LIMIT)
    return ET_CB_ERR_CONFIG;
  if (cfg->block_size < 1u || cfg->block_size > ET_CB_BLOCK_SIZE_LIMIT)
    return ET_CB_ERR_CONFIG;
  if (cfg->policy != (uint32_t)ET_CB_POLICY_DECODE_FIRST_CHUNKED_PREFILL &&
      cfg->policy != (uint32_t)ET_CB_POLICY_PREFILL_FIRST_CHUNKED)
    return ET_CB_ERR_CONFIG;
  return ET_CB_OK;
}

et_cb_status et_cb_init(et_cb_scheduler *sched, const et_cb_config *cfg) {
  if (sched == NULL)
    return ET_CB_ERR_ARGUMENT;
  const et_cb_status vrc = et_cb_validate_config(cfg);
  if (vrc != ET_CB_OK)
    return vrc;

  memset(sched, 0, sizeof(*sched));
  sched->cfg = *cfg;
  sched->initialized = ET_CB_INIT_MAGIC;
  sched->free_kv_blocks = cfg->total_kv_blocks;

  for (uint32_t i = 0u; i < ET_CB_MAX_REQUESTS_CAPACITY; ++i)
    sched->slot_block_head[i] = ET_CB_BLOCK_NONE;

  /* Build deterministic ascending free list: 0 -> 1 -> ... -> N-1 -> NONE. */
  sched->free_block_head = 0u;
  for (uint32_t b = 0u; b < cfg->total_kv_blocks; ++b) {
    sched->block_owner[b] = ET_CB_BLOCK_NONE;
    sched->block_next[b] =
        (b + 1u < cfg->total_kv_blocks) ? (uint16_t)(b + 1u) : ET_CB_BLOCK_NONE;
  }
  for (uint32_t b = cfg->total_kv_blocks; b < ET_CB_TOTAL_KV_BLOCKS_LIMIT;
       ++b) {
    sched->block_owner[b] = ET_CB_BLOCK_NONE;
    sched->block_next[b] = ET_CB_BLOCK_NONE;
  }
  return ET_CB_OK;
}

et_cb_status et_cb_enqueue(et_cb_scheduler *sched,
                           const et_cb_request_spec *spec) {
  if (!sched_valid(sched) || spec == NULL)
    return ET_CB_ERR_ARGUMENT;
  if (spec->prompt_len == 0u || spec->max_new_tokens == 0u ||
      spec->reserved != 0u)
    return ET_CB_ERR_ARGUMENT;

  const uint64_t max_kv_tokens =
      (uint64_t)spec->prompt_len + (uint64_t)spec->max_new_tokens - 1u;
  const uint64_t max_capacity_tokens =
      (uint64_t)sched->cfg.total_kv_blocks * (uint64_t)sched->cfg.block_size;
  if (max_kv_tokens > max_capacity_tokens)
    return ET_CB_ERR_REQUEST_TOO_LARGE;

  if (find_slot_by_id(sched, spec->request_id) >= 0)
    return ET_CB_ERR_DUPLICATE_ID;

  if (sched->waiting_count >= sched->cfg.max_waiting_seqs)
    return ET_CB_ERR_QUEUE_FULL;

  if (sched->slot_count >= ET_CB_MAX_REQUESTS_CAPACITY)
    return ET_CB_ERR_CAPACITY;

  const uint16_t slot_idx = (uint16_t)sched->slot_count;
  et_cb_slot *slot = &sched->slots[slot_idx];
  memset(slot, 0, sizeof(*slot));
  slot->request_id = spec->request_id;
  slot->prompt_len = spec->prompt_len;
  slot->max_new_tokens = spec->max_new_tokens;
  slot->priority = spec->priority;
  slot->arrival_seq = sched->next_arrival_seq;
  slot->enqueue_step = sched->current_step;
  slot->first_token_step = ET_CB_STEP_NONE;
  slot->completion_step = ET_CB_STEP_NONE;
  slot->state = ET_CB_STATE_WAITING;

  sched->slot_block_head[slot_idx] = ET_CB_BLOCK_NONE;
  sched->waiting_queue[sched->waiting_count] = slot_idx;
  sched->waiting_count += 1u;
  sched->slot_count += 1u;
  sched->next_arrival_seq += 1u;
  return ET_CB_OK;
}

et_cb_status et_cb_cancel(et_cb_scheduler *sched, uint64_t request_id) {
  if (!sched_valid(sched))
    return ET_CB_ERR_ARGUMENT;

  const int32_t found = find_slot_by_id(sched, request_id);
  if (found < 0)
    return ET_CB_ERR_NOT_FOUND;

  const uint16_t slot_idx = (uint16_t)found;
  et_cb_slot *slot = &sched->slots[slot_idx];

  if (slot->state == ET_CB_STATE_FINISHED ||
      slot->state == ET_CB_STATE_CANCELLED)
    return ET_CB_ERR_ALREADY_TERMINAL;

  if (slot->active_in_plan != 0u)
    return ET_CB_ERR_PLAN_PENDING;

  if (slot->state == ET_CB_STATE_WAITING ||
      slot->state == ET_CB_STATE_PREEMPTED) {
    if (!remove_from_list(sched->waiting_queue, &sched->waiting_count,
                          slot_idx))
      return ET_CB_ERR_INVARIANT;
  } else {
    const et_cb_status rel = release_all_kv_blocks(sched, slot_idx);
    if (rel != ET_CB_OK)
      return rel;
    if (!remove_from_list(sched->running_list, &sched->running_count, slot_idx))
      return ET_CB_ERR_INVARIANT;
  }

  slot->state = ET_CB_STATE_CANCELLED;
  slot->completion_step = sched->current_step;
  sched->cancelled_requests += 1u;
  return ET_CB_OK;
}

et_cb_status et_cb_step_plan(et_cb_scheduler *sched,
                             et_cb_step_batch *out_plan) {
  if (!sched_valid(sched) || out_plan == NULL)
    return ET_CB_ERR_ARGUMENT;
  if (sched->has_pending_plan != 0u)
    return ET_CB_ERR_PLAN_PENDING;

  et_cb_step_batch plan;
  memset(&plan, 0, sizeof(plan));
  plan.step_index = sched->current_step;
  plan.policy = sched->cfg.policy;

  uint32_t remaining_budget = sched->cfg.max_tokens_per_step;
  et_cb_status rc = ET_CB_OK;

  if (sched->cfg.policy == (uint32_t)ET_CB_POLICY_DECODE_FIRST_CHUNKED_PREFILL) {
    rc = plan_running_decodes(sched, &plan, &remaining_budget);
    if (rc == ET_CB_OK)
      rc = plan_continuing_prefills(sched, &plan, &remaining_budget);
    if (rc == ET_CB_OK)
      rc = admit_waiting_prefills(sched, &plan, &remaining_budget);
  } else {
    rc = plan_continuing_prefills(sched, &plan, &remaining_budget);
    if (rc == ET_CB_OK)
      rc = admit_waiting_prefills(sched, &plan, &remaining_budget);
    if (rc == ET_CB_OK)
      rc = plan_running_decodes(sched, &plan, &remaining_budget);
  }

  if (rc != ET_CB_OK)
    return rc;

  plan.used_kv_blocks = sched->used_kv_blocks;
  plan.free_kv_blocks = sched->free_kv_blocks;
  plan.active_running_seqs = sched->running_count;
  plan.waiting_seqs = sched->waiting_count;

  sched->pending_plan = plan;
  sched->has_pending_plan = 1u;

  const et_cb_status irc = et_cb_verify_invariants(sched);
  if (irc != ET_CB_OK)
    return irc;

  *out_plan = plan;
  return ET_CB_OK;
}

et_cb_status et_cb_step_commit(et_cb_scheduler *sched, uint32_t step_index,
                               const et_cb_commit_item *items,
                               uint32_t item_count) {
  if (!sched_valid(sched))
    return ET_CB_ERR_ARGUMENT;
  if (sched->has_pending_plan == 0u)
    return ET_CB_ERR_NO_PENDING_PLAN;
  if (step_index != sched->pending_plan.step_index)
    return ET_CB_ERR_COMMIT_MISMATCH;
  if (item_count != sched->pending_plan.entry_count)
    return ET_CB_ERR_COMMIT_MISMATCH;
  if (item_count > 0u && items == NULL)
    return ET_CB_ERR_ARGUMENT;

  /* Phase 1: Validate every commit item before mutating any state. */
  uint16_t resolved_slots[ET_CB_MAX_RUNNING_SEQS_LIMIT];
  for (uint32_t i = 0u; i < item_count; ++i) {
    const et_cb_plan_entry *pe = &sched->pending_plan.entries[i];
    const et_cb_commit_item *ci = &items[i];
    if (ci->hit_eos > 1u)
      return ET_CB_ERR_ARGUMENT;
    if (ci->request_id != pe->request_id)
      return ET_CB_ERR_COMMIT_MISMATCH;
    if (ci->generated_token_count != pe->produces_output_token)
      return ET_CB_ERR_COMMIT_MISMATCH;
    if (ci->hit_eos == 1u && pe->produces_output_token == 0u)
      return ET_CB_ERR_COMMIT_MISMATCH;

    const int32_t found = find_slot_by_id(sched, pe->request_id);
    if (found < 0)
      return ET_CB_ERR_INVARIANT;
    const uint16_t slot_idx = (uint16_t)found;
    if (sched->slots[slot_idx].active_in_plan == 0u)
      return ET_CB_ERR_INVARIANT;
    resolved_slots[i] = slot_idx;
  }

  /* Phase 2: Commit state transitions atomically. */
  for (uint32_t i = 0u; i < item_count; ++i) {
    const et_cb_plan_entry *pe = &sched->pending_plan.entries[i];
    const et_cb_commit_item *ci = &items[i];
    const uint16_t slot_idx = resolved_slots[i];
    et_cb_slot *slot = &sched->slots[slot_idx];

    slot->computed_tokens += pe->chunk_tokens;
    slot->active_in_plan = 0u;
    slot->planned_chunk_tokens = 0u;
    slot->planned_is_prefill = 0u;
    slot->planned_produces_token = 0u;
    slot->planned_new_blocks = 0u;

    if (pe->produces_output_token != 0u) {
      if (slot->generated_tokens == 0u)
        slot->first_token_step = step_index;
      slot->generated_tokens += 1u;
      sched->total_generated_tokens += 1u;

      if (ci->hit_eos == 1u || slot->generated_tokens >= slot->max_new_tokens) {
        const et_cb_status rel = release_all_kv_blocks(sched, slot_idx);
        if (rel != ET_CB_OK)
          return rel;
        if (!remove_from_list(sched->running_list, &sched->running_count,
                              slot_idx))
          return ET_CB_ERR_INVARIANT;
        slot->state = ET_CB_STATE_FINISHED;
        slot->eos_stopped = (ci->hit_eos == 1u) ? 1u : 0u;
        slot->completion_step = step_index;
        sched->finished_requests += 1u;
      } else {
        slot->state = ET_CB_STATE_DECODING;
      }
    } else {
      slot->state = ET_CB_STATE_PREFILLING;
    }
  }

  sched->total_scheduled_tokens += sched->pending_plan.total_tokens;
  sched->total_prefill_tokens += sched->pending_plan.prefill_tokens;
  sched->total_decode_tokens += sched->pending_plan.decode_tokens;
  sched->has_pending_plan = 0u;
  memset(&sched->pending_plan, 0, sizeof(sched->pending_plan));
  sched->current_step += 1u;

  return et_cb_verify_invariants(sched);
}

et_cb_status et_cb_get_request_info(const et_cb_scheduler *sched,
                                    uint64_t request_id,
                                    et_cb_request_info *out_info) {
  if (!sched_valid(sched) || out_info == NULL)
    return ET_CB_ERR_ARGUMENT;

  const int32_t found = find_slot_by_id(sched, request_id);
  if (found < 0)
    return ET_CB_ERR_NOT_FOUND;

  const et_cb_slot *slot = &sched->slots[found];
  memset(out_info, 0, sizeof(*out_info));
  out_info->request_id = slot->request_id;
  out_info->state = slot->state;
  out_info->prompt_len = slot->prompt_len;
  out_info->max_new_tokens = slot->max_new_tokens;
  out_info->priority = slot->priority;
  out_info->arrival_seq = slot->arrival_seq;
  out_info->enqueue_step = slot->enqueue_step;
  out_info->computed_tokens = slot->computed_tokens;
  out_info->generated_tokens = slot->generated_tokens;
  out_info->allocated_blocks = slot->allocated_blocks;
  out_info->first_token_step = slot->first_token_step;
  out_info->completion_step = slot->completion_step;
  out_info->ttft_steps =
      (slot->first_token_step != ET_CB_STEP_NONE)
          ? (slot->first_token_step - slot->enqueue_step + 1u)
          : 0u;
  out_info->total_latency_steps =
      (slot->completion_step != ET_CB_STEP_NONE)
          ? (slot->completion_step - slot->enqueue_step + 1u)
          : 0u;
  out_info->preempt_count = slot->preempt_count;
  out_info->eos_stopped = slot->eos_stopped;
  return ET_CB_OK;
}

et_cb_status et_cb_get_stats(const et_cb_scheduler *sched,
                             et_cb_stats *out_stats) {
  if (!sched_valid(sched) || out_stats == NULL)
    return ET_CB_ERR_ARGUMENT;

  memset(out_stats, 0, sizeof(*out_stats));
  out_stats->current_step = sched->current_step;
  out_stats->total_requests = sched->slot_count;
  out_stats->running_requests = sched->running_count;
  out_stats->waiting_requests = sched->waiting_count;
  out_stats->finished_requests = sched->finished_requests;
  out_stats->cancelled_requests = sched->cancelled_requests;
  out_stats->used_kv_blocks = sched->used_kv_blocks;
  out_stats->free_kv_blocks = sched->free_kv_blocks;
  out_stats->total_kv_blocks = sched->cfg.total_kv_blocks;
  out_stats->peak_used_kv_blocks = sched->peak_used_kv_blocks;
  out_stats->peak_running_seqs = sched->peak_running_seqs;
  out_stats->has_pending_plan = sched->has_pending_plan;
  out_stats->total_preemptions = sched->total_preemptions;
  out_stats->total_scheduled_tokens = sched->total_scheduled_tokens;
  out_stats->total_prefill_tokens = sched->total_prefill_tokens;
  out_stats->total_decode_tokens = sched->total_decode_tokens;
  out_stats->total_generated_tokens = sched->total_generated_tokens;
  return ET_CB_OK;
}

et_cb_status et_cb_verify_invariants(const et_cb_scheduler *sched) {
  if (!sched_valid(sched))
    return ET_CB_ERR_ARGUMENT;

  if (sched->used_kv_blocks + sched->free_kv_blocks !=
      sched->cfg.total_kv_blocks)
    return ET_CB_ERR_INVARIANT;

  if (sched->running_count > sched->cfg.max_running_seqs)
    return ET_CB_ERR_INVARIANT;

  if (sched->waiting_count > ET_CB_WAITING_QUEUE_CAPACITY)
    return ET_CB_ERR_INVARIANT;

  if (sched->slot_count > ET_CB_MAX_REQUESTS_CAPACITY)
    return ET_CB_ERR_INVARIANT;

  if (sched->running_count + sched->waiting_count + sched->finished_requests +
          sched->cancelled_requests !=
      sched->slot_count)
    return ET_CB_ERR_INVARIANT;

  /* Verify free block chain has exact length free_kv_blocks and no cycles. */
  uint8_t block_seen[ET_CB_TOTAL_KV_BLOCKS_LIMIT];
  memset(block_seen, 0, sched->cfg.total_kv_blocks);

  uint32_t free_walk = 0u;
  uint16_t cur = sched->free_block_head;
  while (cur != ET_CB_BLOCK_NONE) {
    if (cur >= sched->cfg.total_kv_blocks || block_seen[cur] != 0u ||
        sched->block_owner[cur] != ET_CB_BLOCK_NONE)
      return ET_CB_ERR_INVARIANT;
    block_seen[cur] = 1u;
    free_walk += 1u;
    cur = sched->block_next[cur];
  }
  if (free_walk != sched->free_kv_blocks)
    return ET_CB_ERR_INVARIANT;

  /* Verify per-slot state and block ownership. */
  uint32_t sum_allocated = 0u;
  uint32_t count_running = 0u;
  uint32_t count_waiting = 0u;
  uint32_t count_finished = 0u;
  uint32_t count_cancelled = 0u;

  for (uint32_t s = 0u; s < sched->slot_count; ++s) {
    const et_cb_slot *slot = &sched->slots[s];
    uint32_t slot_walk = 0u;
    uint16_t b = sched->slot_block_head[s];
    while (b != ET_CB_BLOCK_NONE) {
      if (b >= sched->cfg.total_kv_blocks || block_seen[b] != 0u ||
          sched->block_owner[b] != (uint16_t)s)
        return ET_CB_ERR_INVARIANT;
      block_seen[b] = 1u;
      slot_walk += 1u;
      b = sched->block_next[b];
    }
    if (slot_walk != slot->allocated_blocks)
      return ET_CB_ERR_INVARIANT;
    sum_allocated += slot_walk;

    switch (slot->state) {
    case ET_CB_STATE_WAITING:
    case ET_CB_STATE_PREEMPTED:
      if (slot->allocated_blocks != 0u || slot->active_in_plan != 0u)
        return ET_CB_ERR_INVARIANT;
      count_waiting += 1u;
      break;
    case ET_CB_STATE_PREFILLING:
    case ET_CB_STATE_DECODING: {
      const uint32_t effective_tokens =
          slot->computed_tokens +
          (slot->active_in_plan ? slot->planned_chunk_tokens : 0u);
      const uint32_t expected_blocks =
          ceil_div_u32(effective_tokens, sched->cfg.block_size);
      if (slot->allocated_blocks == 0u ||
          slot->allocated_blocks != expected_blocks)
        return ET_CB_ERR_INVARIANT;
      count_running += 1u;
      break;
    }
    case ET_CB_STATE_FINISHED:
      if (slot->allocated_blocks != 0u || slot->active_in_plan != 0u)
        return ET_CB_ERR_INVARIANT;
      count_finished += 1u;
      break;
    case ET_CB_STATE_CANCELLED:
      if (slot->allocated_blocks != 0u || slot->active_in_plan != 0u)
        return ET_CB_ERR_INVARIANT;
      count_cancelled += 1u;
      break;
    default:
      return ET_CB_ERR_INVARIANT;
    }
  }

  if (sum_allocated != sched->used_kv_blocks)
    return ET_CB_ERR_INVARIANT;
  if (count_running != sched->running_count ||
      count_waiting != sched->waiting_count ||
      count_finished != sched->finished_requests ||
      count_cancelled != sched->cancelled_requests)
    return ET_CB_ERR_INVARIANT;

  for (uint32_t b = 0u; b < sched->cfg.total_kv_blocks; ++b) {
    if (block_seen[b] != 1u)
      return ET_CB_ERR_INVARIANT;
  }

  /* Verify queue memberships have no duplicates and match states. */
  uint8_t slot_seen[ET_CB_MAX_REQUESTS_CAPACITY];
  memset(slot_seen, 0, sched->slot_count);
  for (uint32_t i = 0u; i < sched->running_count; ++i) {
    const uint16_t idx = sched->running_list[i];
    if (idx >= sched->slot_count || slot_seen[idx] != 0u)
      return ET_CB_ERR_INVARIANT;
    const uint32_t st = sched->slots[idx].state;
    if (st != ET_CB_STATE_PREFILLING && st != ET_CB_STATE_DECODING)
      return ET_CB_ERR_INVARIANT;
    slot_seen[idx] = 1u;
  }
  for (uint32_t i = 0u; i < sched->waiting_count; ++i) {
    const uint16_t idx = sched->waiting_queue[i];
    if (idx >= sched->slot_count || slot_seen[idx] != 0u)
      return ET_CB_ERR_INVARIANT;
    const uint32_t st = sched->slots[idx].state;
    if (st != ET_CB_STATE_WAITING && st != ET_CB_STATE_PREEMPTED)
      return ET_CB_ERR_INVARIANT;
    slot_seen[idx] = 1u;
  }

  if (sched->has_pending_plan != 0u) {
    if (sched->pending_plan.total_tokens > sched->cfg.max_tokens_per_step ||
        sched->pending_plan.entry_count > sched->cfg.max_running_seqs)
      return ET_CB_ERR_INVARIANT;
    uint32_t sum_chunk = 0u;
    for (uint32_t i = 0u; i < sched->pending_plan.entry_count; ++i)
      sum_chunk += sched->pending_plan.entries[i].chunk_tokens;
    if (sum_chunk != sched->pending_plan.total_tokens)
      return ET_CB_ERR_INVARIANT;
  }

  return ET_CB_OK;
}
