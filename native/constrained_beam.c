#include "eshkol_transformer/constrained_beam_abi.h"

#include <string.h>

static et_beam_status mask_error(et_cd_status status) {
  switch (status) {
  case ET_CD_ERR_NONFINITE:
    return ET_BEAM_ERR_NONFINITE;
  case ET_CD_ERR_NO_VALID_TOKENS:
    return ET_BEAM_ERR_NO_VALID_TOKENS;
  case ET_CD_ERR_FP_ENV:
    return ET_BEAM_ERR_FP_ENV;
  case ET_CD_ERR_ARGUMENT:
    return ET_BEAM_ERR_INVALID_ARG;
  case ET_CD_ERR_ALIASING:
    return ET_BEAM_ERR_ALIASING;
  default:
    return ET_BEAM_ERR_STATE;
  }
}

static et_beam_status replay_suffix(const et_cd_vocab_index *index,
                                    const uint32_t *tokens, size_t len,
                                    uint32_t completed, uint16_t *state) {
  for (size_t i = 0; i < len; ++i) {
    uint32_t finished;
    if (et_cd_advance_token_v1(index, *state, tokens[i], state, &finished) !=
            ET_CD_OK ||
        finished != (completed && i + 1u == len))
      return ET_BEAM_ERR_STATE;
  }
  return ET_BEAM_OK;
}

et_beam_status et_constrained_beam_step_v1(et_beam_state_v1 *beam,
                                           const et_cd_vocab_index *index,
                                           const float *logits,
                                           size_t logits_rows,
                                           size_t logits_vocab) {
  if (beam == NULL || index == NULL || logits == NULL)
    return ET_BEAM_ERR_INVALID_ARG;

  /* The BEAM5 accessor checks the FP mode and the state invariants,
   * including the shared length of all active hypotheses. */
  uint32_t first_tokens[ET_BEAM_MAX_SEQ_LEN];
  size_t first_len;
  double first_log_prob;
  et_beam_status status = et_beam_get_active_beam_v1(
      beam, 0u, first_tokens, ET_BEAM_MAX_SEQ_LEN, &first_len, &first_log_prob);
  if (status == ET_BEAM_ERR_INVALID_ARG)
    return ET_BEAM_ERR_STATE; /* A completed search has no active beam. */
  if (status != ET_BEAM_OK)
    return status;
  if (beam->done)
    return ET_BEAM_ERR_STATE;

  et_cd_vocab_index_info info;
  if (et_cd_vocab_index_info_get(index, &info) != ET_CD_OK ||
      logits_rows != beam->active_count ||
      logits_vocab != beam->config.vocab_size ||
      logits_vocab != info.vocab_size ||
      beam->config.eos_token_id != info.eos_token_id)
    return ET_BEAM_ERR_INVALID_ARG;

  const size_t logits_bytes = logits_rows * logits_vocab * sizeof(float);
  const uintptr_t logits_addr = (uintptr_t)logits;
  const uintptr_t beam_addr = (uintptr_t)beam;
  if (logits_addr % _Alignof(float) != 0u ||
      logits_addr > UINTPTR_MAX - logits_bytes)
    return ET_BEAM_ERR_INVALID_ARG;
  if (logits_addr < beam_addr + sizeof(*beam) &&
      beam_addr < logits_addr + logits_bytes)
    return ET_BEAM_ERR_ALIASING;

  const size_t prompt_len = beam->prompt_len;
  for (uint32_t i = 0; i < beam->finished_count; ++i) {
    const et_beam_hypothesis_v1 *hyp = &beam->finished_pool[i];
    uint16_t state = info.start_state;
    status = replay_suffix(index, hyp->tokens + prompt_len, hyp->generated_len,
                           1u, &state);
    if (status != ET_BEAM_OK)
      return status;
  }

  double staged_logits[ET_BEAM_MAX_BEAMS * ET_BEAM_MAX_VOCAB];
  float masked[ET_BEAM_MAX_VOCAB];
  for (size_t b = 0; b < logits_rows; ++b) {
    uint16_t state = info.start_state;
    const uint32_t *tokens = b == 0u ? first_tokens : beam->active_tokens[b];
    status = replay_suffix(index, tokens + prompt_len, first_len - prompt_len,
                           0u, &state);
    if (status != ET_BEAM_OK)
      return status;
    const size_t offset = b * logits_vocab;
    const et_cd_status mask_status =
        et_cd_apply_logit_mask_v1(index, state, logits + offset, masked);
    if (mask_status != ET_CD_OK)
      return mask_error(mask_status);
    for (size_t t = 0; t < logits_vocab; ++t)
      staged_logits[offset + t] = (double)masked[t];
  }

  /* BEAM5 may overwrite its scratch arrays before reporting an error. */
  et_beam_state_v1 staged_beam;
  memcpy(&staged_beam, beam, sizeof(staged_beam));
  status = et_beam_step_v1(&staged_beam, staged_logits, logits_rows);
  if (status == ET_BEAM_OK)
    memcpy(beam, &staged_beam, sizeof(staged_beam));
  return status;
}
