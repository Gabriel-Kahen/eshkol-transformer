#ifndef ESHKOL_TRANSFORMER_CONSTRAINED_DECODE_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_CONSTRAINED_DECODE_ESHKOL_BRIDGE_H

#include "eshkol_transformer/constrained_decode_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_cd_aot_ctx et_cd_aot_ctx;

/* Context lifecycle:
 *   create_json_schema : compiles {"answer":<enum:yes|no|maybe>,"score":<int>,"ok":<bool>}
 *                        and binds a byte + multi-byte + adversarial-token vocabulary.
 *   create_enum_grammar: compiles exact literal-choices grammar ("yes"|"no"|"maybe")
 *                        and binds the same vocabulary. */
int32_t et_cd_aot_create_json_schema(et_cd_aot_ctx **out_ctx);
int32_t et_cd_aot_create_enum_grammar(et_cd_aot_ctx **out_ctx);
void et_cd_aot_destroy(et_cd_aot_ctx *ctx);

/* Vocabulary and state inspection. */
int64_t et_cd_aot_vocab_size(const et_cd_aot_ctx *ctx);
int64_t et_cd_aot_eos_token_id(const et_cd_aot_ctx *ctx);
int64_t et_cd_aot_start_state(const et_cd_aot_ctx *ctx);
int64_t et_cd_aot_num_states(const et_cd_aot_ctx *ctx);
int64_t et_cd_aot_live_states(const et_cd_aot_ctx *ctx);
int64_t et_cd_aot_current_state(const et_cd_aot_ctx *ctx);
int64_t et_cd_aot_is_finished(const et_cd_aot_ctx *ctx);

/* Token ID lookup by exact C string tag (for constructing adversarial logits
 * in Eshkol without hardcoding table indices). Returns -1 if not found. */
int64_t et_cd_aot_find_token_by_tag(const et_cd_aot_ctx *ctx, int64_t tag_id);

/* Reset generation state and accumulated output buffer. */
int32_t et_cd_aot_reset(et_cd_aot_ctx *ctx);

/* Per-step logit setup, masking, inspection, and token advance. */
int32_t et_cd_aot_set_logit(et_cd_aot_ctx *ctx, int64_t token_id, double value);
int32_t et_cd_aot_apply_mask(et_cd_aot_ctx *ctx);
int64_t et_cd_aot_masked_is_allowed(const et_cd_aot_ctx *ctx, int64_t token_id);
double et_cd_aot_get_masked_logit(const et_cd_aot_ctx *ctx, int64_t token_id);
int32_t et_cd_aot_advance_token(et_cd_aot_ctx *ctx, int64_t token_id);

/* Accumulated output byte inspection and DFA acceptance verification. */
int64_t et_cd_aot_output_len(const et_cd_aot_ctx *ctx);
int64_t et_cd_aot_output_byte(const et_cd_aot_ctx *ctx, int64_t index);
int64_t et_cd_aot_output_accepted_by_dfa(const et_cd_aot_ctx *ctx);

/* Artifact roundtrip serialization check + SHA-256 word accessors (0..3). */
int32_t et_cd_aot_verify_artifact_roundtrip(et_cd_aot_ctx *ctx);
int64_t et_cd_aot_artifact_sha256_word(const et_cd_aot_ctx *ctx,
                                       int64_t word_index);

/* Negative-path verification helpers returning exact et_cd_status codes. */
int32_t et_cd_aot_check_dead_end_dfa(void);
int32_t et_cd_aot_check_illegal_advance(const et_cd_aot_ctx *ctx,
                                        int64_t token_id);
int32_t et_cd_aot_check_nan_logit_preserves_output(et_cd_aot_ctx *ctx);
int32_t et_cd_aot_check_inf_logit_preserves_output(et_cd_aot_ctx *ctx);
int32_t et_cd_aot_check_dead_state_mask(et_cd_aot_ctx *ctx);
int32_t et_cd_aot_check_tampered_artifact(const et_cd_aot_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif
