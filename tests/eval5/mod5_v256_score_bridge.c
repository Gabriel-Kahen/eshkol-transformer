#include "mod5_v256_score_bridge.h"
#include "eshkol_transformer/eval5_benchmark_abi.h"

#include <math.h>
#include <string.h>

enum { TOKENS = 8, VOCAB = 256, DOC_BYTES = 9, SPLIT_IDS = 2 * DOC_BYTES,
       CONTEXT_FLOATS = TOKENS * VOCAB, FINGERPRINT_BYTES = 96 };

static const et_eval5_contamination_config policy = {
    ET_EVAL5_NGRAM_TOKENS, 3u, 1u, 0.25, 0.0,
};

static et_eval5_status split_documents(const uint8_t *bytes, const uint32_t *ids,
                                       et_eval5_document documents[2]) {
  if (bytes == NULL || ids == NULL) return ET_EVAL5_ERR_ARGUMENT;
  for (size_t i = 0; i < SPLIT_IDS; ++i)
    if (ids[i] != bytes[i]) return ET_EVAL5_ERR_IDENTITY;
  documents[0] = (et_eval5_document){bytes, DOC_BYTES, ids, DOC_BYTES};
  documents[1] = (et_eval5_document){
      bytes + DOC_BYTES, DOC_BYTES, ids + DOC_BYTES, DOC_BYTES,
  };
  return ET_EVAL5_OK;
}

int32_t et_eval5_v256_audit(const uint8_t train_bytes[18],
                            const uint32_t train_tokens[18],
                            const uint8_t eval_bytes[18],
                            const uint32_t eval_tokens[18],
                            uint64_t summary_out[8]) {
  if (train_bytes == NULL || train_tokens == NULL || eval_bytes == NULL ||
      eval_tokens == NULL || summary_out == NULL) return ET_EVAL5_ERR_ARGUMENT;
  et_eval5_document train[2], eval[2];
  et_eval5_status status = split_documents(train_bytes, train_tokens, train);
  if (status != ET_EVAL5_OK) return status;
  status = split_documents(eval_bytes, eval_tokens, eval);
  if (status != ET_EVAL5_OK) return status;
  uint64_t indices[2] = {0, 0};
  et_eval5_contamination_summary summary;
  status = et_eval5_audit_contamination(
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

typedef struct {
  float lm_logits[2u * CONTEXT_FLOATS];
  uint32_t lm_targets[2u * TOKENS];
  uint32_t choice_targets[4][2][2];
  et_eval5_sequence lm;
  et_eval5_sequence choices[4][2];
  et_eval5_choice_item items[4];
} v256_score_inputs;

static et_eval5_status prepare_scores(const float *contexts[4],
                                      const uint32_t *eval_tokens,
                                      v256_score_inputs *inputs) {
  if (eval_tokens == NULL) return ET_EVAL5_ERR_ARGUMENT;
  for (size_t c = 0; c < 4u; ++c)
    if (contexts[c] == NULL) return ET_EVAL5_ERR_ARGUMENT;
  for (size_t i = 0; i < SPLIT_IDS; ++i)
    if (eval_tokens[i] >= VOCAB) return ET_EVAL5_ERR_TARGET;
  for (size_t c = 0; c < 4u; ++c)
    for (size_t i = 0; i < CONTEXT_FLOATS; ++i)
      if (!isfinite(contexts[c][i])) return ET_EVAL5_ERR_NONFINITE;

  memcpy(inputs->lm_logits, contexts[0], sizeof(inputs->lm_logits) / 2u);
  memcpy(inputs->lm_logits + CONTEXT_FLOATS, contexts[1],
         sizeof(inputs->lm_logits) / 2u);
  for (size_t i = 0; i < TOKENS; ++i) {
    inputs->lm_targets[i] = eval_tokens[i + 1u];
    inputs->lm_targets[TOKENS + i] = eval_tokens[DOC_BYTES + i + 1u];
  }
  inputs->lm = sequence(inputs->lm_logits, inputs->lm_targets, 2u * TOKENS);

  const uint32_t targets[4][2][2] = {
      {{eval_tokens[7], eval_tokens[8]}, {1u, 4u}},
      {{4u, 1u}, {eval_tokens[16], eval_tokens[17]}},
      {{10u, 11u}, {12u, 13u}},
      {{14u, 15u}, {3u, 0u}},
  };
  memcpy(inputs->choice_targets, targets, sizeof(targets));
  const size_t context_indices[4][2] = {{0, 0}, {1, 1}, {2, 3}, {3, 2}};
  const size_t correct_indices[4] = {0, 1, 0, 1};
  for (size_t item = 0; item < 4u; ++item) {
    for (size_t choice = 0; choice < 2u; ++choice)
      inputs->choices[item][choice] = sequence(
          contexts[context_indices[item][choice]] + 6u * VOCAB,
          inputs->choice_targets[item][choice], 2u);
    inputs->items[item] =
        (et_eval5_choice_item){inputs->choices[item], 2u, correct_indices[item]};
  }
  return ET_EVAL5_OK;
}

int32_t et_eval5_v256_score(const float *doc0, const float *doc1,
                            const float *context_a, const float *context_b,
                            const uint32_t *eval_tokens, double metrics_out[5],
                            int64_t counts_out[2], float probes_out[8]) {
  if (metrics_out == NULL || counts_out == NULL || probes_out == NULL)
    return ET_EVAL5_ERR_ARGUMENT;
  const float *contexts[4] = {doc0, doc1, context_a, context_b};
  v256_score_inputs inputs;
  et_eval5_status status = prepare_scores(contexts, eval_tokens, &inputs);
  if (status != ET_EVAL5_OK) return status;
  et_eval5_lm_metrics lm_result;
  status = et_eval5_score_lm(&inputs.lm, 2u * TOKENS, &lm_result);
  if (status != ET_EVAL5_OK) return status;

  et_eval5_mc_metrics mc_result;
  status = et_eval5_score_mc(inputs.items, 4u, 1u, &mc_result);
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

int32_t et_eval5_v256_compare_runner(
    const uint8_t fingerprints[288], const uint8_t digests[96],
    const uint8_t documents[36], const uint32_t train_tokens[18],
    const uint32_t eval_tokens[18], const float *doc0, const float *doc1,
    const float *context_a, const float *context_b,
    const char *expected, uint64_t expected_size) {
  if (fingerprints == NULL || digests == NULL || documents == NULL ||
      expected == NULL) return ET_EVAL5_ERR_ARGUMENT;
  if (expected_size == 0 || expected_size > ET_EVAL5_MANIFEST_MAX_BYTES)
    return ET_EVAL5_ERR_BOUNDS;
  et_eval5_document train[2], eval[2];
  et_eval5_status status = split_documents(documents, train_tokens, train);
  if (status != ET_EVAL5_OK) return status;
  status = split_documents(documents + SPLIT_IDS, eval_tokens, eval);
  if (status != ET_EVAL5_OK) return status;

  const float *contexts[4] = {doc0, doc1, context_a, context_b};
  v256_score_inputs inputs;
  status = prepare_scores(contexts, eval_tokens, &inputs);
  if (status != ET_EVAL5_OK) return status;
  et_eval5_benchmark_request request = {
      .suite = "mod5-v256-development",
      .task = "lm-mc",
      .split = "eval-clean",
      .context_length = TOKENS,
      .contamination = policy,
      .tokenizer = {fingerprints, FINGERPRINT_BYTES, VOCAB},
      .train = {train, 2u, {fingerprints + FINGERPRINT_BYTES,
                            FINGERPRINT_BYTES, VOCAB}, {0}},
      .eval = {eval, 2u, {fingerprints + 2u * FINGERPRINT_BYTES,
                          FINGERPRINT_BYTES, VOCAB}, {0}},
      .metric_mask = ET_EVAL5_METRIC_LM | ET_EVAL5_METRIC_MC,
      .mc_normalize_by_length = 1u,
      .lm_sequence = &inputs.lm,
      .lm_byte_count = 2u * TOKENS,
      .lm_sample_count = 2u,
      .mc_items = inputs.items,
      .mc_item_count = 4u,
  };
  memcpy(request.train.dataset_sha256, digests, ET_EVAL5_SHA256_BYTES);
  memcpy(request.eval.dataset_sha256, digests + ET_EVAL5_SHA256_BYTES,
         ET_EVAL5_SHA256_BYTES);
  memcpy(request.checkpoint_sha256, digests + 2u * ET_EVAL5_SHA256_BYTES,
         ET_EVAL5_SHA256_BYTES);

  et_eval5_benchmark_result result;
  status = et_eval5_run_benchmark(&request, &result);
  if (status != ET_EVAL5_OK) return status;
  if (result.canonical_size != expected_size ||
      memcmp(result.canonical_bytes, expected, result.canonical_size + 1u) != 0)
    return ET_EVAL5_ERR_FORMAT;
  return ET_EVAL5_OK;
}
