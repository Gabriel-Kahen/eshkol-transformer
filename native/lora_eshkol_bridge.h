#ifndef ESHKOL_TRANSFORMER_LORA_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_LORA_ESHKOL_BRIDGE_H

#include "eshkol_transformer/lora_adapter_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_lora_aot_ctx et_lora_aot_ctx;

/* Context lifecycle for compiled Eshkol AOT LoRA fine-tuning & verification. */
int32_t et_lora_aot_create(int64_t m, int64_t k, int64_t n, int64_t r,
                           double alpha, et_lora_aot_ctx **out_ctx);
void et_lora_aot_destroy(et_lora_aot_ctx *ctx);

/* Element counts. */
int64_t et_lora_aot_x_count(const et_lora_aot_ctx *ctx);
int64_t et_lora_aot_w_count(const et_lora_aot_ctx *ctx);
int64_t et_lora_aot_a_count(const et_lora_aot_ctx *ctx);
int64_t et_lora_aot_b_count(const et_lora_aot_ctx *ctx);
int64_t et_lora_aot_y_count(const et_lora_aot_ctx *ctx);

/* Deterministic initialization and tensor element accessors. */
int32_t et_lora_aot_init_adapter(et_lora_aot_ctx *ctx, int64_t seed);
int32_t et_lora_aot_set_x(et_lora_aot_ctx *ctx, int64_t index, double value);
int32_t et_lora_aot_set_w_base(et_lora_aot_ctx *ctx, int64_t index,
                               double value);
int32_t et_lora_aot_set_target(et_lora_aot_ctx *ctx, int64_t index,
                               double value);
int32_t et_lora_aot_set_a(et_lora_aot_ctx *ctx, int64_t index, double value);
int32_t et_lora_aot_set_b(et_lora_aot_ctx *ctx, int64_t index, double value);
int32_t et_lora_aot_set_dy(et_lora_aot_ctx *ctx, int64_t index, double value);

double et_lora_aot_get_a(const et_lora_aot_ctx *ctx, int64_t index);
double et_lora_aot_get_b(const et_lora_aot_ctx *ctx, int64_t index);
double et_lora_aot_get_y(const et_lora_aot_ctx *ctx, int64_t index);
double et_lora_aot_get_y_base(const et_lora_aot_ctx *ctx, int64_t index);
double et_lora_aot_get_y_merged(const et_lora_aot_ctx *ctx, int64_t index);
double et_lora_aot_get_target(const et_lora_aot_ctx *ctx, int64_t index);
double et_lora_aot_get_dx(const et_lora_aot_ctx *ctx, int64_t index);
double et_lora_aot_get_da(const et_lora_aot_ctx *ctx, int64_t index);
double et_lora_aot_get_db(const et_lora_aot_ctx *ctx, int64_t index);
double et_lora_aot_get_w_base(const et_lora_aot_ctx *ctx, int64_t index);
double et_lora_aot_get_w_restored(const et_lora_aot_ctx *ctx, int64_t index);

/* Exact 32-bit IEEE-754 bit patterns returned as non-negative i64. */
int64_t et_lora_aot_w_base_bits(const et_lora_aot_ctx *ctx, int64_t index);
int64_t et_lora_aot_y_bits(const et_lora_aot_ctx *ctx, int64_t index);
int64_t et_lora_aot_y_base_bits(const et_lora_aot_ctx *ctx, int64_t index);

/* Core LoRA operations. */
int32_t et_lora_aot_base_forward(et_lora_aot_ctx *ctx);
int32_t et_lora_aot_forward(et_lora_aot_ctx *ctx);
int32_t et_lora_aot_backward(et_lora_aot_ctx *ctx);
int32_t et_lora_aot_merge_and_forward(et_lora_aot_ctx *ctx);
int32_t et_lora_aot_unmerge(et_lora_aot_ctx *ctx);

/* SHA-256 digests of frozen base weight W_base and current output Y.
 * word_index in [0, 3] returns big-endian 64-bit chunks of the 32-byte digest. */
int32_t et_lora_aot_compute_base_sha256(et_lora_aot_ctx *ctx);
int64_t et_lora_aot_base_sha256_word(const et_lora_aot_ctx *ctx,
                                     int64_t word_index);
int32_t et_lora_aot_compute_y_sha256(et_lora_aot_ctx *ctx);
int64_t et_lora_aot_y_sha256_word(const et_lora_aot_ctx *ctx,
                                  int64_t word_index);

/* Multi-process checkpoint persistence and negative tamper helpers driven by
 * environment variables LORA_AOT_PHASE, LORA_AOT_CKPT, LORA_AOT_TAMPER,
 * and LORA_AOT_SYMLINK. */
int64_t et_lora_aot_env_phase(void);
int32_t et_lora_aot_save_env_checkpoint(const et_lora_aot_ctx *ctx);
int32_t et_lora_aot_load_env_checkpoint(et_lora_aot_ctx *ctx);
int32_t et_lora_aot_check_wrong_base_hash(const et_lora_aot_ctx *ctx);
int32_t et_lora_aot_check_wrong_rank(const et_lora_aot_ctx *ctx);
int32_t et_lora_aot_check_tampered_env(const et_lora_aot_ctx *ctx,
                                       int64_t tamper_kind);

#ifdef __cplusplus
}
#endif

#endif
