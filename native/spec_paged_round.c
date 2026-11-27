#include "eshkol_transformer/spec_paged_round_abi.h"

static et_paged_kv_status append(et_paged_kv_pool *pool, uint32_t seq_id,
                                 uint32_t num_tokens, size_t floats_per_token,
                                 const et_spec_paged_kv_span_v1 *kv) {
  if (kv->element_count != num_tokens * floats_per_token)
    return ET_PAGED_KV_ERR_SHAPE;
  return et_paged_kv_append_tokens(pool, seq_id, num_tokens, kv->keys, kv->values);
}

et_spec_paged_round_status_v1
et_spec_paged_round_greedy_v1(const et_spec_paged_round_request_v1 *request,
                              et_spec_paged_round_result_v1 *out) {
  if (request == NULL || out == NULL ||
      (uintptr_t)request % _Alignof(et_spec_paged_round_request_v1) != 0u ||
      (uintptr_t)out % _Alignof(et_spec_paged_round_result_v1) != 0u ||
      request->draft.pool == NULL || request->target.pool == NULL ||
      request->draft.pool == request->target.pool)
    return (et_spec_paged_round_status_v1){ET_SPEC_PAGED_ROUND_ERR_ARGUMENT, 0};

  const uint32_t horizon = request->horizon;
  const uint32_t vocab_size = request->vocab_size;
  if (horizon < ET_SPEC_MIN_HORIZON || horizon > ET_SPEC_MAX_HORIZON ||
      vocab_size < ET_SPEC_MIN_VOCAB || vocab_size > ET_SPEC_MAX_VOCAB ||
      request->target_logit_count != ((size_t)horizon + 1u) * vocab_size)
    return (et_spec_paged_round_status_v1){ET_SPEC_PAGED_ROUND_ERR_VERIFY,
                                           ET_SPEC_ERR_BOUNDS};

  enum { DRAFT, TARGET, MODELS };
  const et_spec_paged_round_model_v1 models[MODELS] = {request->draft, request->target};
  const int32_t cache_error[MODELS] = {ET_SPEC_PAGED_ROUND_ERR_DRAFT_CACHE,
                                       ET_SPEC_PAGED_ROUND_ERR_TARGET_CACHE};
  uint32_t parent_len[MODELS];
  size_t floats_per_token[MODELS];
  uint32_t children[MODELS] = {UINT32_MAX, UINT32_MAX};
  et_spec_paged_round_status_v1 status;
  et_spec_paged_round_result_v1 result = {0};

  for (size_t i = 0; i < MODELS; ++i) {
    const et_spec_paged_round_model_v1 *model = &models[i];
    et_paged_kv_config cfg;
    et_paged_kv_status code = et_paged_kv_config_get(model->pool, &cfg);
    if (code != ET_PAGED_KV_OK)
      return (et_spec_paged_round_status_v1){cache_error[i], code};
    code = et_paged_kv_seq_len(model->pool, model->parent_seq_id, &parent_len[i]);
    if (code != ET_PAGED_KV_OK)
      return (et_spec_paged_round_status_v1){cache_error[i], code};
    floats_per_token[i] = (size_t)cfg.num_layers * cfg.num_kv_heads * cfg.head_dim;
  }
  if (parent_len[DRAFT] != parent_len[TARGET])
    return (et_spec_paged_round_status_v1){ET_SPEC_PAGED_ROUND_ERR_PREFIX_LENGTH, 0};

  for (size_t i = 0; i < MODELS; ++i) {
    const et_spec_paged_round_model_v1 *model = &models[i];
    et_paged_kv_status code =
        et_paged_kv_seq_fork(model->pool, model->parent_seq_id, &children[i]);
    if (code != ET_PAGED_KV_OK) {
      status = (et_spec_paged_round_status_v1){cache_error[i], code};
      goto discard;
    }
  }
  for (size_t i = 0; i < MODELS; ++i) {
    et_paged_kv_status code = append(models[i].pool, children[i], horizon,
                                     floats_per_token[i], &models[i].candidate_kv);
    if (code != ET_PAGED_KV_OK) {
      status = (et_spec_paged_round_status_v1){cache_error[i], code};
      goto discard;
    }
  }

  et_spec_status verify =
      et_spec_verify_greedy_v1(request->draft_tokens, request->target_logits, horizon,
                               vocab_size, request->eos_token_id, result.tokens,
                               ET_SPEC_MAX_HORIZON + 1u, &result.verification);
  if (verify != ET_SPEC_OK) {
    status = (et_spec_paged_round_status_v1){ET_SPEC_PAGED_ROUND_ERR_VERIFY, verify};
    goto discard;
  }

  const uint32_t accepted = result.verification.accepted_count;
  const uint32_t tail_pos = parent_len[DRAFT] + accepted;
  for (size_t i = 0; i < MODELS; ++i) {
    et_paged_kv_status code =
        et_paged_kv_seq_truncate(models[i].pool, children[i], tail_pos);
    if (code != ET_PAGED_KV_OK) {
      status = (et_spec_paged_round_status_v1){cache_error[i], code};
      goto discard;
    }
  }

  if (result.verification.emitted_count > accepted) {
    if (request->recompute == NULL) {
      status = (et_spec_paged_round_status_v1){ET_SPEC_PAGED_ROUND_ERR_ARGUMENT, 0};
      goto discard;
    }
    const et_spec_paged_round_prefix_v1 draft = {models[DRAFT].pool, children[DRAFT]};
    const et_spec_paged_round_prefix_v1 target = {models[TARGET].pool,
                                                  children[TARGET]};
    et_spec_paged_kv_span_v1 tail[MODELS] = {0};
    int32_t code =
        request->recompute(request->context, result.tokens[accepted], tail_pos, draft,
                           target, &tail[DRAFT], &tail[TARGET]);
    if (code != 0) {
      status = (et_spec_paged_round_status_v1){ET_SPEC_PAGED_ROUND_ERR_RECOMPUTE, code};
      goto discard;
    }
    for (size_t i = 0; i < MODELS; ++i) {
      code = append(models[i].pool, children[i], 1u, floats_per_token[i], &tail[i]);
      if (code != ET_PAGED_KV_OK) {
        status = (et_spec_paged_round_status_v1){cache_error[i], code};
        goto discard;
      }
    }
  }

  result.draft_seq_id = children[DRAFT];
  result.target_seq_id = children[TARGET];
  *out = result;
  return (et_spec_paged_round_status_v1){ET_SPEC_PAGED_ROUND_OK, 0};

discard:
  /* Exclusive access and the read-only callback keep each forked ID live. */
  for (size_t i = 0; i < MODELS; ++i)
    if (children[i] != UINT32_MAX)
      (void)et_paged_kv_seq_free(models[i].pool, children[i]);
  return status;
}
