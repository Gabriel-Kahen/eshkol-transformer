#ifndef ESHKOL_TRANSFORMER_DPO_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_DPO_ESHKOL_BRIDGE_H

#include "eshkol_transformer/dpo_alignment_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_DPO_AOT_MAX_HIDDEN 64u

typedef struct et_dpo_aot_ctx et_dpo_aot_ctx;

/* Context lifecycle for compiled Eshkol AOT DPO alignment & verification. */
int32_t et_dpo_aot_create(int64_t batch_size, int64_t seq_len,
                          int64_t vocab_size, int64_t hidden_dim, double beta,
                          double label_smoothing, int64_t reference_free,
                          et_dpo_aot_ctx **out_ctx);
void et_dpo_aot_destroy(et_dpo_aot_ctx *ctx);

/* Update runtime DPO hyperparameters (validated on dispatch). */
int32_t et_dpo_aot_set_hyperparams(et_dpo_aot_ctx *ctx, double beta,
                                   double label_smoothing,
                                   int64_t reference_free);

/* Buffer element counts. */
int64_t et_dpo_aot_logit_count(const et_dpo_aot_ctx *ctx);
int64_t et_dpo_aot_token_count(const et_dpo_aot_ctx *ctx);
int64_t et_dpo_aot_hidden_count(const et_dpo_aot_ctx *ctx);
int64_t et_dpo_aot_weight_count(const et_dpo_aot_ctx *ctx);
int64_t et_dpo_aot_bias_count(const et_dpo_aot_ctx *ctx);

/* Logit, target, and mask buffer marshaling. */
int32_t et_dpo_aot_set_policy_chosen_logit(et_dpo_aot_ctx *ctx, int64_t index,
                                           double value);
int32_t et_dpo_aot_set_policy_rejected_logit(et_dpo_aot_ctx *ctx, int64_t index,
                                             double value);
int32_t et_dpo_aot_set_ref_chosen_logit(et_dpo_aot_ctx *ctx, int64_t index,
                                        double value);
int32_t et_dpo_aot_set_ref_rejected_logit(et_dpo_aot_ctx *ctx, int64_t index,
                                          double value);
double et_dpo_aot_get_policy_chosen_logit(const et_dpo_aot_ctx *ctx,
                                          int64_t index);
double et_dpo_aot_get_policy_rejected_logit(const et_dpo_aot_ctx *ctx,
                                            int64_t index);
double et_dpo_aot_get_d_logit_chosen(const et_dpo_aot_ctx *ctx, int64_t index);
double et_dpo_aot_get_d_logit_rejected(const et_dpo_aot_ctx *ctx,
                                       int64_t index);
int64_t et_dpo_aot_d_logit_chosen_bits(const et_dpo_aot_ctx *ctx,
                                       int64_t index);
int64_t et_dpo_aot_d_logit_rejected_bits(const et_dpo_aot_ctx *ctx,
                                         int64_t index);
int32_t et_dpo_aot_poison_grads(et_dpo_aot_ctx *ctx, double sentinel);

int32_t et_dpo_aot_set_chosen_target(et_dpo_aot_ctx *ctx, int64_t index,
                                     int64_t token_id);
int32_t et_dpo_aot_set_rejected_target(et_dpo_aot_ctx *ctx, int64_t index,
                                       int64_t token_id);
int32_t et_dpo_aot_set_chosen_mask(et_dpo_aot_ctx *ctx, int64_t index,
                                   int64_t mask_val);
int32_t et_dpo_aot_set_rejected_mask(et_dpo_aot_ctx *ctx, int64_t index,
                                     int64_t mask_val);

/* Linear LM head buffers: hidden [B, T, D], frozen ref (W_ref [V, D], b_ref [V])
 * and trainable policy (W_pi [V, D], b_pi [V]). */
int32_t et_dpo_aot_set_hidden_chosen(et_dpo_aot_ctx *ctx, int64_t index,
                                     double value);
int32_t et_dpo_aot_set_hidden_rejected(et_dpo_aot_ctx *ctx, int64_t index,
                                       double value);
int32_t et_dpo_aot_set_ref_weight(et_dpo_aot_ctx *ctx, int64_t index,
                                  double value);
int32_t et_dpo_aot_set_ref_bias(et_dpo_aot_ctx *ctx, int64_t index,
                                double value);
int32_t et_dpo_aot_set_policy_weight(et_dpo_aot_ctx *ctx, int64_t index,
                                     double value);
int32_t et_dpo_aot_set_policy_bias(et_dpo_aot_ctx *ctx, int64_t index,
                                   double value);
double et_dpo_aot_get_policy_weight(const et_dpo_aot_ctx *ctx, int64_t index);
double et_dpo_aot_get_policy_bias(const et_dpo_aot_ctx *ctx, int64_t index);
double et_dpo_aot_get_d_policy_weight(const et_dpo_aot_ctx *ctx, int64_t index);
double et_dpo_aot_get_d_policy_bias(const et_dpo_aot_ctx *ctx, int64_t index);
int64_t et_dpo_aot_ref_weight_bits(const et_dpo_aot_ctx *ctx, int64_t index);
int64_t et_dpo_aot_ref_bias_bits(const et_dpo_aot_ctx *ctx, int64_t index);

/* Initialize policy head as an exact bitwise copy of the frozen reference head
 * and compute frozen reference logits from hidden states. */
int32_t et_dpo_aot_init_policy_from_ref(et_dpo_aot_ctx *ctx);

/* Project policy logits = hidden @ W_pi^T + b_pi for chosen & rejected. */
int32_t et_dpo_aot_project_policy_logits(et_dpo_aot_ctx *ctx);

/* Backpropagate d_logits_{chosen,rejected} into dW_pi and db_pi. */
int32_t et_dpo_aot_head_backward(et_dpo_aot_ctx *ctx);

/* Core DPO forward and fused forward+backward dispatch. */
int32_t et_dpo_aot_forward(et_dpo_aot_ctx *ctx);
int32_t et_dpo_aot_forward_backward(et_dpo_aot_ctx *ctx);

/* Telemetry accessors (binary64). */
double et_dpo_aot_mean_loss(const et_dpo_aot_ctx *ctx);
double et_dpo_aot_chosen_reward_mean(const et_dpo_aot_ctx *ctx);
double et_dpo_aot_rejected_reward_mean(const et_dpo_aot_ctx *ctx);
double et_dpo_aot_reward_margin_mean(const et_dpo_aot_ctx *ctx);
double et_dpo_aot_preference_accuracy(const et_dpo_aot_ctx *ctx);

/* Canonical SHA-256 digest of frozen reference head parameters (W_ref || b_ref).
 * word_index in [0, 3] returns big-endian 64-bit chunks of the 32-byte digest. */
int32_t et_dpo_aot_compute_ref_sha256(et_dpo_aot_ctx *ctx);
int64_t et_dpo_aot_ref_sha256_word(const et_dpo_aot_ctx *ctx,
                                   int64_t word_index);

#ifdef __cplusplus
}
#endif

#endif
