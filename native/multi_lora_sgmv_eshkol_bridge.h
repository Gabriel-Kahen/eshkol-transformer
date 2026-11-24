#ifndef ESHKOL_TRANSFORMER_MULTI_LORA_SGMV_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_MULTI_LORA_SGMV_ESHKOL_BRIDGE_H

#include "eshkol_transformer/multi_lora_sgmv_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_mlora_aot_ctx et_mlora_aot_ctx;

/* Context lifecycle for compiled Eshkol AOT Multi-LoRA SGMV driver. */
int32_t et_mlora_aot_create(int64_t m, int64_t in_dim, int64_t out_dim,
                            int64_t max_adapters, int64_t max_rank,
                            int64_t has_bias, et_mlora_aot_ctx **out_ctx);
void et_mlora_aot_destroy(et_mlora_aot_ctx *ctx);

/* Populate frozen base weight W_base[out_dim, in_dim] and optional
 * bias[out_dim], then freeze into the immutable et_mlora_registry. */
int32_t et_mlora_aot_set_w_base(et_mlora_aot_ctx *ctx, int64_t index,
                                double value);
int32_t et_mlora_aot_set_bias(et_mlora_aot_ctx *ctx, int64_t index,
                              double value);
int32_t et_mlora_aot_finalize_base(et_mlora_aot_ctx *ctx);

/* Batch input/output/target/routing accessors. */
int64_t et_mlora_aot_x_count(const et_mlora_aot_ctx *ctx);
int64_t et_mlora_aot_w_count(const et_mlora_aot_ctx *ctx);
int64_t et_mlora_aot_y_count(const et_mlora_aot_ctx *ctx);

int32_t et_mlora_aot_set_x(et_mlora_aot_ctx *ctx, int64_t index, double value);
double et_mlora_aot_get_x(const et_mlora_aot_ctx *ctx, int64_t index);
int32_t et_mlora_aot_set_dy(et_mlora_aot_ctx *ctx, int64_t index, double value);
int32_t et_mlora_aot_set_target(et_mlora_aot_ctx *ctx, int64_t index,
                                double value);
double et_mlora_aot_get_target(const et_mlora_aot_ctx *ctx, int64_t index);
int32_t et_mlora_aot_set_row_adapter(et_mlora_aot_ctx *ctx, int64_t row,
                                     int64_t adapter_id);
int64_t et_mlora_aot_get_row_adapter(const et_mlora_aot_ctx *ctx, int64_t row);

double et_mlora_aot_get_y(const et_mlora_aot_ctx *ctx, int64_t index);
double et_mlora_aot_get_y_ref(const et_mlora_aot_ctx *ctx, int64_t index);
double et_mlora_aot_get_dx(const et_mlora_aot_ctx *ctx, int64_t index);
int64_t et_mlora_aot_y_bits(const et_mlora_aot_ctx *ctx, int64_t index);
int64_t et_mlora_aot_y_ref_bits(const et_mlora_aot_ctx *ctx, int64_t index);

/* Adapter pool operations & weight/grad inspection. */
int32_t et_mlora_aot_register_seeded(et_mlora_aot_ctx *ctx, int64_t adapter_id,
                                     int64_t rank, double alpha, int64_t seed);
int32_t et_mlora_aot_unregister(et_mlora_aot_ctx *ctx, int64_t adapter_id);
int32_t et_mlora_aot_pin(et_mlora_aot_ctx *ctx, int64_t adapter_id);
int32_t et_mlora_aot_unpin(et_mlora_aot_ctx *ctx, int64_t adapter_id);

int64_t et_mlora_aot_adapter_rank(const et_mlora_aot_ctx *ctx,
                                  int64_t adapter_id);
int64_t et_mlora_aot_adapter_generation(const et_mlora_aot_ctx *ctx,
                                        int64_t adapter_id);
int64_t et_mlora_aot_adapter_pin_count(const et_mlora_aot_ctx *ctx,
                                       int64_t adapter_id);
int64_t et_mlora_aot_adapter_step(const et_mlora_aot_ctx *ctx,
                                  int64_t adapter_id);

int32_t et_mlora_aot_set_a(et_mlora_aot_ctx *ctx, int64_t adapter_id,
                           int64_t index, double value);
int32_t et_mlora_aot_set_b(et_mlora_aot_ctx *ctx, int64_t adapter_id,
                           int64_t index, double value);
double et_mlora_aot_get_a(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                          int64_t index);
double et_mlora_aot_get_b(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                          int64_t index);
double et_mlora_aot_get_da(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                           int64_t index);
double et_mlora_aot_get_db(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                           int64_t index);
int64_t et_mlora_aot_da_bits(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                             int64_t index);
int64_t et_mlora_aot_db_bits(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                             int64_t index);

/* Core SGMV forward, per-row single-adapter reference, backward, and AdamW. */
int32_t et_mlora_aot_sgmv_forward(et_mlora_aot_ctx *ctx);
int32_t et_mlora_aot_single_row_ref_forward(et_mlora_aot_ctx *ctx);
int32_t et_mlora_aot_sgmv_backward(et_mlora_aot_ctx *ctx);
int32_t et_mlora_aot_adamw_step(et_mlora_aot_ctx *ctx, int64_t adapter_id,
                                double lr, double beta1, double beta2,
                                double eps, double weight_decay);

int64_t et_mlora_aot_last_base_rows(const et_mlora_aot_ctx *ctx);
int64_t et_mlora_aot_last_adapted_rows(const et_mlora_aot_ctx *ctx);
int64_t et_mlora_aot_last_active_segments(const et_mlora_aot_ctx *ctx);

/* SHA-256 digests of frozen base (W_base + bias) and current batched Y.
 * word_index in [0, 3] returns big-endian 64-bit chunks of the 32-byte digest. */
int32_t et_mlora_aot_compute_base_sha256(et_mlora_aot_ctx *ctx);
int64_t et_mlora_aot_base_sha256_word(const et_mlora_aot_ctx *ctx,
                                      int64_t word_index);
int32_t et_mlora_aot_compute_y_sha256(et_mlora_aot_ctx *ctx);
int64_t et_mlora_aot_y_sha256_word(const et_mlora_aot_ctx *ctx,
                                   int64_t word_index);

/* Multi-process ETML1000 bundle persistence & negative checks driven by
 * MLORA_AOT_PHASE, MLORA_AOT_BUNDLE, MLORA_AOT_TAMPER. */
int64_t et_mlora_aot_env_phase(void);
int32_t et_mlora_aot_save_env_bundle(const et_mlora_aot_ctx *ctx,
                                     int64_t include_adamw);
int32_t et_mlora_aot_load_env_bundle(et_mlora_aot_ctx *ctx);
int32_t et_mlora_aot_check_tampered_env(const et_mlora_aot_ctx *ctx,
                                        int64_t tamper_kind);
int32_t et_mlora_aot_check_nonfinite_x(const et_mlora_aot_ctx *ctx);
int32_t et_mlora_aot_check_fp_env_trap(const et_mlora_aot_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif
