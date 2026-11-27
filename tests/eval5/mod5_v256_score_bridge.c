#include "mod5_v256_score_bridge.h"
#include "eshkol_transformer/eval5_benchmark_abi.h"

#include <math.h>
#include <string.h>

enum { TOKENS = 8, VOCAB = 256, DOC_IDS = 18, CONTEXT_FLOATS = TOKENS * VOCAB };

int32_t et_eval5_v256_audit(const uint8_t train_bytes[18],
                            const uint32_t train_tokens[18],
                            const uint8_t eval_bytes[18],
                            const uint32_t eval_tokens[18],
                            uint64_t summary_out[8]) {
  if (train_bytes == NULL || train_tokens == NULL || eval_bytes == NULL ||
      eval_tokens == NULL || summary_out == NULL)
    return ET_EVAL5_ERR_ARGUMENT;
  for (size_t i = 0; i < DOC_IDS; ++i)
    if (train_tokens[i] != train_bytes[i] || eval_tokens[i] != eval_bytes[i])
      return ET_EVAL5_ERR_IDENTITY;
  const et_eval5_document train[2] = {
      {train_bytes, 9u, train_tokens, 9u},
      {train_bytes + 9u, 9u, train_tokens + 9u, 9u},
  };
  const et_eval5_document eval[2] = {
      {eval_bytes, 9u, eval_tokens, 9u},
      {eval_bytes + 9u, 9u, eval_tokens + 9u, 9u},
  };
  const et_eval5_contamination_config policy = {
      ET_EVAL5_NGRAM_TOKENS, 3u, 1u, 0.25, 0.0,
  };
  uint64_t indices[2] = {0, 0};
  et_eval5_contamination_summary summary;
  et_eval5_status status = et_eval5_audit_contamination(
      train, 2u, eval, 2u, &policy, indices, 2u, &summary);
  if (status != ET_EVAL5_OK && status != ET_EVAL5_ERR_CONTAMINATED)
    return status;
  const uint64_t result[8] = {
      summary.eval_documents, summary.contaminated_documents,
      summary.exact_doc_matches, summary.ngram_contaminated_docs,
      summary.total_eval_ngrams, summary.matched_eval_ngrams,
      indices[0], indices[1],
  };
  memcpy(summary_out, result, sizeof(result));
  return status;
}

static et_eval5_sequence sequence(const float *logits,
                                  const uint32_t *targets, size_t count) {
  return (et_eval5_sequence){logits, targets, count, VOCAB, count * VOCAB};
}

double et_eval5_v256_metric(const double metrics[5], int64_t index) {
  if (metrics == NULL || index < 0 || index >= 5) return NAN;
  return metrics[index];
}

int32_t et_eval5_v256_score(const float *doc0, const float *doc1,
                            const float *context_a, const float *context_b,
                            const uint32_t *eval_tokens, double metrics_out[5],
                            int64_t counts_out[2], float probes_out[8]) {
  if (doc0 == NULL || doc1 == NULL || context_a == NULL || context_b == NULL ||
      eval_tokens == NULL || metrics_out == NULL || counts_out == NULL ||
      probes_out == NULL)
    return ET_EVAL5_ERR_ARGUMENT;

  const float *contexts[4] = {doc0, doc1, context_a, context_b};
  for (size_t i = 0; i < DOC_IDS; ++i)
    if (eval_tokens[i] >= VOCAB) return ET_EVAL5_ERR_TARGET;
  for (size_t c = 0; c < 4u; ++c)
    for (size_t i = 0; i < CONTEXT_FLOATS; ++i)
      if (!isfinite(contexts[c][i])) return ET_EVAL5_ERR_NONFINITE;

  float lm_logits[2u * CONTEXT_FLOATS];
  uint32_t lm_targets[2u * TOKENS];
  memcpy(lm_logits, doc0, sizeof(*doc0) * CONTEXT_FLOATS);
  memcpy(lm_logits + CONTEXT_FLOATS, doc1, sizeof(*doc1) * CONTEXT_FLOATS);
  for (size_t i = 0; i < TOKENS; ++i) {
    lm_targets[i] = eval_tokens[i + 1u];
    lm_targets[TOKENS + i] = eval_tokens[TOKENS + 2u + i];
  }
  const et_eval5_sequence lm = sequence(lm_logits, lm_targets, 2u * TOKENS);
  et_eval5_lm_metrics lm_result;
  et_eval5_status status = et_eval5_score_lm(&lm, 2u * TOKENS, &lm_result);
  if (status != ET_EVAL5_OK) return status;

  const uint32_t item0_true[2] = {eval_tokens[7], eval_tokens[8]};
  const uint32_t item0_other[2] = {1u, 4u};
  const uint32_t item1_other[2] = {4u, 1u};
  const uint32_t item1_true[2] = {eval_tokens[16], eval_tokens[17]};
  const uint32_t item2_a[2] = {10u, 11u};
  const uint32_t item2_b[2] = {12u, 13u};
  const uint32_t item3_b[2] = {14u, 15u};
  const uint32_t item3_a[2] = {3u, 0u};
  et_eval5_sequence choices[4][2] = {
      {sequence(doc0 + 6u * VOCAB, item0_true, 2u),
       sequence(doc0 + 6u * VOCAB, item0_other, 2u)},
      {sequence(doc1 + 6u * VOCAB, item1_other, 2u),
       sequence(doc1 + 6u * VOCAB, item1_true, 2u)},
      {sequence(context_a + 6u * VOCAB, item2_a, 2u),
       sequence(context_b + 6u * VOCAB, item2_b, 2u)},
      {sequence(context_b + 6u * VOCAB, item3_b, 2u),
       sequence(context_a + 6u * VOCAB, item3_a, 2u)},
  };
  const et_eval5_choice_item items[4] = {
      {choices[0], 2u, 0u}, {choices[1], 2u, 1u},
      {choices[2], 2u, 0u}, {choices[3], 2u, 1u},
  };
  et_eval5_mc_metrics mc_result;
  status = et_eval5_score_mc(items, 4u, 1u, &mc_result);
  if (status != ET_EVAL5_OK) return status;

  const double metrics[5] = {lm_result.total_nll, lm_result.mean_nll,
                             lm_result.perplexity, lm_result.bits_per_byte,
                             mc_result.accuracy};
  const int64_t counts[2] = {(int64_t)mc_result.correct_count,
                             (int64_t)mc_result.total_count};
  float probes[8];
  for (size_t c = 0; c < 4u; ++c) {
    probes[2u * c] = contexts[c][0];
    probes[2u * c + 1u] = contexts[c][CONTEXT_FLOATS - 1u];
  }
  memcpy(metrics_out, metrics, sizeof(metrics));
  memcpy(counts_out, counts, sizeof(counts));
  memcpy(probes_out, probes, sizeof(probes));
  return ET_EVAL5_OK;
}
