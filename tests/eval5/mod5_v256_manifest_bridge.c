#include "mod5_v256_manifest_bridge.h"

#include <string.h>

et_eval5_status et_eval5_v256_manifest_roundtrip(
    const et_eval5_v256_manifest_input *input,
    char out[ET_EVAL5_MANIFEST_MAX_BYTES + 1u], size_t capacity,
    size_t *out_size) {
  if (input == NULL || out == NULL || out_size == NULL ||
      input->tokenizer_fingerprint == NULL || input->corpus_fingerprint == NULL)
    return ET_EVAL5_ERR_ARGUMENT;
  if (input->audit_status != ET_EVAL5_OK) return input->audit_status;
  const uint64_t clean[8] = {2u, 0u, 0u, 0u, 14u, 0u, 0u, 0u};
  if (memcmp(input->audit_counts, clean, sizeof(clean)) != 0)
    return ET_EVAL5_ERR_CONTAMINATED;

  et_eval5_manifest manifest = {0};
  manifest.version_major = ET_EVAL5_VERSION_MAJOR;
  manifest.version_minor = ET_EVAL5_VERSION_MINOR;
  memcpy(manifest.suite, "mod5-v256-development", sizeof("mod5-v256-development"));
  memcpy(manifest.task, "lm-mc", sizeof("lm-mc"));
  memcpy(manifest.split, "eval-clean", sizeof("eval-clean"));
  et_eval5_status status = et_eval5_bind_tokenizer_identity(
      input->tokenizer_fingerprint, input->tokenizer_length, 256u,
      input->corpus_fingerprint, input->corpus_length, 256u,
      manifest.tokenizer_sha256);
  if (status != ET_EVAL5_OK) return status;
  memcpy(manifest.train_dataset_sha256, input->train_manifest_sha256,
         ET_EVAL5_SHA256_BYTES);
  memcpy(manifest.eval_dataset_sha256, input->eval_manifest_sha256,
         ET_EVAL5_SHA256_BYTES);
  memcpy(manifest.checkpoint_sha256, input->checkpoint_sha256,
         ET_EVAL5_SHA256_BYTES);
  manifest.context_length = 8u;
  manifest.max_samples = 0u;
  manifest.contamination = (et_eval5_contamination_config){
      ET_EVAL5_NGRAM_TOKENS, 3u, 1u, 0.25, 0.0};
  manifest.metric_mask = ET_EVAL5_METRIC_LM | ET_EVAL5_METRIC_MC;
  manifest.mc_normalize_by_length = 1u;
  manifest.lm = (et_eval5_lm_metrics){
      16u, 16u, input->metrics[0], input->metrics[1],
      input->metrics[2], input->metrics[3]};
  if (input->mc_counts[0] < 0 || input->mc_counts[1] < 0)
    return ET_EVAL5_ERR_FORMAT;
  manifest.mc = (et_eval5_mc_metrics){
      (uint64_t)input->mc_counts[0], (uint64_t)input->mc_counts[1],
      input->metrics[4]};
  manifest.audit = (et_eval5_contamination_summary){
      input->audit_counts[0], input->audit_counts[1], input->audit_counts[2],
      input->audit_counts[3], input->audit_counts[4], input->audit_counts[5], 0.0};

  char canonical[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  char reformatted[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  size_t size = 0, second_size = 0;
  status = et_eval5_manifest_format(&manifest, canonical, sizeof(canonical), &size);
  if (status != ET_EVAL5_OK) return status;
  if (capacity <= size) return ET_EVAL5_ERR_BOUNDS;
  et_eval5_manifest parsed;
  status = et_eval5_manifest_parse(canonical, size, &parsed);
  if (status != ET_EVAL5_OK) return status;
  status = et_eval5_manifest_format(&parsed, reformatted,
                                    sizeof(reformatted), &second_size);
  if (status != ET_EVAL5_OK) return status;
  if (second_size != size || memcmp(canonical, reformatted, size + 1u) != 0)
    return ET_EVAL5_ERR_FORMAT;
  memcpy(out, canonical, size + 1u);
  *out_size = size;
  return ET_EVAL5_OK;
}
