#ifndef ESHKOL_TRANSFORMER_SPEC_PAGED_ROUND_ABI_H
#define ESHKOL_TRANSFORMER_SPEC_PAGED_ROUND_ABI_H

#include "eshkol_transformer/paged_kv_abi.h"
#include "eshkol_transformer/spec_decode_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* On a CACHE or VERIFY error, code is the corresponding paged KV or SPEC5
 * status code. On RECOMPUTE it is the callback's return value. Otherwise zero. */
enum {
  ET_SPEC_PAGED_ROUND_OK = 0,
  ET_SPEC_PAGED_ROUND_ERR_ARGUMENT = 1,
  ET_SPEC_PAGED_ROUND_ERR_PREFIX_LENGTH = 2,
  ET_SPEC_PAGED_ROUND_ERR_DRAFT_CACHE = 3,
  ET_SPEC_PAGED_ROUND_ERR_TARGET_CACHE = 4,
  ET_SPEC_PAGED_ROUND_ERR_VERIFY = 5,
  ET_SPEC_PAGED_ROUND_ERR_RECOMPUTE = 6
};

typedef struct et_spec_paged_round_status_v1 {
  int32_t kind;
  int32_t code;
} et_spec_paged_round_status_v1;

/* Contiguous f32 K and V, each with element_count floats and layout
 * [num_tokens, num_layers, num_kv_heads, head_dim] for the respective pool. */
typedef struct et_spec_paged_kv_span_v1 {
  const float *keys;
  const float *values;
  size_t element_count;
} et_spec_paged_kv_span_v1;

typedef struct et_spec_paged_round_model_v1 {
  et_paged_kv_pool *pool;
  uint32_t parent_seq_id;
  et_spec_paged_kv_span_v1 candidate_kv;
} et_spec_paged_round_model_v1;

/* A read-only view of a temporary child, valid during the callback only. */
typedef struct et_spec_paged_round_prefix_v1 {
  const et_paged_kv_pool *pool;
  uint32_t seq_id;
} et_spec_paged_round_prefix_v1;

/* Produce K/V for one correction or bonus token at position, using the
 * accepted prefix in each child. May read the pools and stage private model
 * state. Return normally without modifying either pool or publishing state.
 * Set both spans and return zero on success. The span descriptors and prefix
 * views are borrowed for the callback only. Span data must remain valid,
 * disjoint from pool storage and the result, and unchanged until the round
 * returns. A nonzero return discards both spans. */
typedef int32_t (*et_spec_paged_recompute_v1)(void *context, uint32_t token,
                                              uint32_t position,
                                              et_spec_paged_round_prefix_v1 draft,
                                              et_spec_paged_round_prefix_v1 target,
                                              et_spec_paged_kv_span_v1 *out_draft,
                                              et_spec_paged_kv_span_v1 *out_target);

typedef struct et_spec_paged_round_request_v1 {
  et_spec_paged_round_model_v1 draft;
  et_spec_paged_round_model_v1 target;
  const uint32_t *draft_tokens; /* horizon elements */
  const float *target_logits;   /* [horizon + 1, vocab_size] */
  size_t target_logit_count;
  uint32_t horizon;
  uint32_t vocab_size;
  uint32_t eos_token_id;
  et_spec_paged_recompute_v1 recompute;
  void *context;
} et_spec_paged_round_request_v1;

typedef struct et_spec_paged_round_result_v1 {
  uint32_t draft_seq_id;
  uint32_t target_seq_id;
  et_spec_verify_result_v1 verification;
  uint32_t tokens[ET_SPEC_MAX_HORIZON + 1u]; /* emitted_count valid entries */
} et_spec_paged_round_result_v1;

/* Run a greedy round over distinct, model-scoped pools. The live parents must
 * represent the same token prefix. Each candidate K/V span represents all
 * horizon draft tokens run against that producer's parent. Target logit row
 * i predicts a token after the parent and the first i candidate tokens.
 * The coordinator checks equal parent lengths. The caller binds token and
 * model identity to the supplied logits, K/V, and parents.
 *
 * The coordinator forks, stages, verifies, and truncates both children before
 * calling recompute. When the draft EOS is accepted, recompute is skipped
 * and may be NULL. A correction or bonus requires recompute, including when
 * that token is EOS. Callers may publish private model state on success and
 * own both returned sequence IDs. They retain ownership of both parents.
 *
 * On failure, the result is untouched and temporary sequence IDs and live
 * block references are released. Parent block tables and live K/V remain
 * unchanged. Cumulative CoW telemetry and freed storage may reflect staging.
 * All buffers must be valid and aligned. Keep request inputs fixed for the
 * duration of the round. The result must be disjoint from inputs and callback
 * storage. Caller buffers and the two pools must have disjoint backing
 * storage (including any external K/V backing). Use exclusive pool access
 * throughout the round. Callback scratch is caller-owned.
 */
et_spec_paged_round_status_v1
et_spec_paged_round_greedy_v1(const et_spec_paged_round_request_v1 *request,
                              et_spec_paged_round_result_v1 *out);

#ifdef __cplusplus
}
#endif

#endif
