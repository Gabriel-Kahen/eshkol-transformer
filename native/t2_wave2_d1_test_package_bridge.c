#include "t2_wave2_package_bridge.c"

extern eshkol_tagged_value_t et_e1b_private_t2_test_token_corpus_read_cabi_v1(
    eshkol_tagged_value_t tokenizer, eshkol_tagged_value_t directory,
    eshkol_tagged_value_t maximum_manifest,
    eshkol_tagged_value_t maximum_shard,
    eshkol_tagged_value_t maximum_tokens);

void et_e1b_public_t2_test_token_corpus_read_v1(
    void *tokenizer, void *directory, void *maximum_manifest,
    void *maximum_shard, void *maximum_tokens, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_t2_test_token_corpus_read_cabi_v1(
          *et_e1b_box_value_v1(tokenizer),
          *et_e1b_box_value_v1(directory),
          *et_e1b_box_value_v1(maximum_manifest),
          *et_e1b_box_value_v1(maximum_shard),
          *et_e1b_box_value_v1(maximum_tokens));
}
