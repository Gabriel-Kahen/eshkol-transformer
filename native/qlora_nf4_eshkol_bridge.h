#ifndef ESHKOL_TRANSFORMER_QLORA_NF4_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_QLORA_NF4_ESHKOL_BRIDGE_H

#include "eshkol_transformer/qlora_nf4_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_qlora_aot_ctx et_qlora_aot_ctx;

/* Context lifecycle for compiled Eshkol AOT QLoRA NF4 fine-tuning & verification.
 * quant_mode: 1 = NF4_SQ, 2 = NF4_DQ. */
int32_t et_qlora_aot_create(int64_t m, int64_t d_in, int64_t d_hidden,
                            int64_t d_out, int64_t r, int64_t block_size,
                            int64_t dq_block_size, int64_t quant_mode,
                            double alpha, et_qlora_aot_ctx **out_ctx);
void et_qlora_aot_destroy(et_qlora_aot_ctx *ctx);

/* Populate deterministic Gaussian base weights (Box-Muller), inputs, biases,
 * structured target, and quantize both layers in both SQ and DQ modes. */
int32_t et_qlora_aot_prepare_model(et_qlora_aot_ctx *ctx, int64_t base_seed,
                                   int64_t lora_seed);

/* Footprint and quantization accuracy metrics (SQ vs DQ vs Uniform INT4). */
int64_t et_qlora_aot_f32_bytes(const et_qlora_aot_ctx *ctx);
int64_t et_qlora_aot_sq_packed_bytes(const et_qlora_aot_ctx *ctx);
int64_t et_qlora_aot_dq_packed_bytes(const et_qlora_aot_ctx *ctx);
double et_qlora_aot_sq_bits_per_param(const et_qlora_aot_ctx *ctx);
double et_qlora_aot_dq_bits_per_param(const et_qlora_aot_ctx *ctx);
double et_qlora_aot_sq_compression_ratio(const et_qlora_aot_ctx *ctx);
double et_qlora_aot_dq_compression_ratio(const et_qlora_aot_ctx *ctx);
double et_qlora_aot_nf4_sq_rmse(const et_qlora_aot_ctx *ctx);
double et_qlora_aot_nf4_dq_rmse(const et_qlora_aot_ctx *ctx);
double et_qlora_aot_uniform_int4_rmse(const et_qlora_aot_ctx *ctx);

/* Central finite-difference gradient check on dX, dA, dB for layer 1
 * (returns ET_QLORA_OK and records max_err / check count). */
int32_t et_qlora_aot_run_grad_checks(et_qlora_aot_ctx *ctx);
double et_qlora_aot_grad_max_err(const et_qlora_aot_ctx *ctx);
int64_t et_qlora_aot_grad_check_count(const et_qlora_aot_ctx *ctx);

/* AdamW hyperparameters and single QLoRA training step. */
int32_t et_qlora_aot_configure_adamw(et_qlora_aot_ctx *ctx, double lr,
                                     double beta1, double beta2, double eps,
                                     double weight_decay);
int32_t et_qlora_aot_eval_loss(et_qlora_aot_ctx *ctx);
int32_t et_qlora_aot_train_step(et_qlora_aot_ctx *ctx);
double et_qlora_aot_last_loss(const et_qlora_aot_ctx *ctx);
int64_t et_qlora_aot_adamw_step(const et_qlora_aot_ctx *ctx);

/* Fused NF4+LoRA vs merge_to_f32 dense inference parity across both layers. */
int32_t et_qlora_aot_verify_merge_parity(et_qlora_aot_ctx *ctx);
double et_qlora_aot_max_merge_diff(const et_qlora_aot_ctx *ctx);

/* SHA-256 digests: frozen NF4 base, current MLP output Y, and LoRA+AdamW state.
 * word_index in [0, 3] returns big-endian 64-bit chunks of the 32-byte digest. */
int32_t et_qlora_aot_compute_digests(et_qlora_aot_ctx *ctx);
int64_t et_qlora_aot_base_sha256_word(const et_qlora_aot_ctx *ctx,
                                      int64_t word_index);
int64_t et_qlora_aot_y_sha256_word(const et_qlora_aot_ctx *ctx,
                                   int64_t word_index);
int64_t et_qlora_aot_state_sha256_word(const et_qlora_aot_ctx *ctx,
                                       int64_t word_index);

/* Poison LoRA weights and AdamW state before checkpoint reload. */
int32_t et_qlora_aot_poison_trainable_state(et_qlora_aot_ctx *ctx);

/* Multi-process phase, checkpoint save/load, and negative/tamper checks driven
 * by QLORA_AOT_PHASE, QLORA_AOT_CKPT, and QLORA_AOT_TAMPER.
 * Phases: 0=all-in-one, 1=full12, 2=part1 (6 steps + save), 3=part2 (reload + 6 steps + negatives). */
int64_t et_qlora_aot_env_phase(void);
int32_t et_qlora_aot_save_env_checkpoint(const et_qlora_aot_ctx *ctx);
int32_t et_qlora_aot_load_env_checkpoint(et_qlora_aot_ctx *ctx);
int32_t et_qlora_aot_check_negative_suite(const et_qlora_aot_ctx *ctx);
int64_t et_qlora_aot_negative_checks_passed(const et_qlora_aot_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif
