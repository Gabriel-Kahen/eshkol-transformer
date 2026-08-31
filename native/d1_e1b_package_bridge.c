#include "e1b_error_consumer_bridge.h"

extern eshkol_tagged_value_t et_e1b_private_d1_token_corpus_write_cabi_v1(
    eshkol_tagged_value_t directory, eshkol_tagged_value_t fingerprint,
    eshkol_tagged_value_t vocab_size,
    eshkol_tagged_value_t shard_token_limit, eshkol_tagged_value_t tokens);
extern eshkol_tagged_value_t et_e1b_private_d1_token_corpus_validate_cabi_v1(
    eshkol_tagged_value_t directory,
    eshkol_tagged_value_t maximum_manifest_bytes,
    eshkol_tagged_value_t maximum_shard_bytes,
    eshkol_tagged_value_t maximum_total_tokens);
extern eshkol_tagged_value_t
et_e1b_private_d1_token_corpus_summary_shard_count_cabi_v1(
    eshkol_tagged_value_t summary);
extern eshkol_tagged_value_t
et_e1b_private_d1_token_corpus_summary_total_tokens_cabi_v1(
    eshkol_tagged_value_t summary);
extern eshkol_tagged_value_t
et_e1b_private_d1_token_corpus_summary_vocab_size_cabi_v1(
    eshkol_tagged_value_t summary);
extern eshkol_tagged_value_t
et_e1b_private_d1_token_corpus_summary_shard_token_limit_cabi_v1(
    eshkol_tagged_value_t summary);
extern eshkol_tagged_value_t
et_e1b_private_d1_token_corpus_summary_tokenizer_fingerprint_cabi_v1(
    eshkol_tagged_value_t summary);
extern eshkol_tagged_value_t
et_e1b_private_d1_token_corpus_summary_total_shard_bytes_cabi_v1(
    eshkol_tagged_value_t summary);

void et_e1b_public_d1_token_corpus_write_v1(
    void *directory, void *fingerprint, void *vocab_size,
    void *shard_token_limit, void *tokens, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_d1_token_corpus_write_cabi_v1(
          *et_e1b_box_value_v1(directory),
          *et_e1b_box_value_v1(fingerprint),
          *et_e1b_box_value_v1(vocab_size),
          *et_e1b_box_value_v1(shard_token_limit),
          *et_e1b_box_value_v1(tokens));
}

void et_e1b_public_d1_token_corpus_validate_v1(
    void *directory, void *maximum_manifest_bytes,
    void *maximum_shard_bytes, void *maximum_total_tokens, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_d1_token_corpus_validate_cabi_v1(
          *et_e1b_box_value_v1(directory),
          *et_e1b_box_value_v1(maximum_manifest_bytes),
          *et_e1b_box_value_v1(maximum_shard_bytes),
          *et_e1b_box_value_v1(maximum_total_tokens));
}

#define ET_D1_E1B_UNARY_BRIDGE(name, target)                               \
  void name(void *summary, void *output) {                                 \
    et_e1b_ensure_private_initialized_v1();                                \
    *et_e1b_box_value_v1(output) =                                         \
        target(*et_e1b_box_value_v1(summary));                             \
  }

ET_D1_E1B_UNARY_BRIDGE(
    et_e1b_public_d1_token_corpus_summary_shard_count_v1,
    et_e1b_private_d1_token_corpus_summary_shard_count_cabi_v1)
ET_D1_E1B_UNARY_BRIDGE(
    et_e1b_public_d1_token_corpus_summary_total_tokens_v1,
    et_e1b_private_d1_token_corpus_summary_total_tokens_cabi_v1)
ET_D1_E1B_UNARY_BRIDGE(
    et_e1b_public_d1_token_corpus_summary_vocab_size_v1,
    et_e1b_private_d1_token_corpus_summary_vocab_size_cabi_v1)
ET_D1_E1B_UNARY_BRIDGE(
    et_e1b_public_d1_token_corpus_summary_shard_token_limit_v1,
    et_e1b_private_d1_token_corpus_summary_shard_token_limit_cabi_v1)
ET_D1_E1B_UNARY_BRIDGE(
    et_e1b_public_d1_token_corpus_summary_tokenizer_fingerprint_v1,
    et_e1b_private_d1_token_corpus_summary_tokenizer_fingerprint_cabi_v1)
ET_D1_E1B_UNARY_BRIDGE(
    et_e1b_public_d1_token_corpus_summary_total_shard_bytes_v1,
    et_e1b_private_d1_token_corpus_summary_total_shard_bytes_cabi_v1)
