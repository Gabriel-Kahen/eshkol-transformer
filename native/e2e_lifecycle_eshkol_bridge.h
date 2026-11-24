#ifndef ESHKOL_TRANSFORMER_E2E_LIFECYCLE_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_E2E_LIFECYCLE_ESHKOL_BRIDGE_H

#include "eshkol_transformer/e2e_lifecycle_abi.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_e2e_aot_ctx et_e2e_aot_ctx;

/* Integer metric IDs for et_e2e_aot_get_i64. */
enum {
  ET_E2E_M_S1_INPUT_DOCS = 100,
  ET_E2E_M_S1_KEPT_DOCS = 101,
  ET_E2E_M_S1_DROPPED_EXACT = 102,
  ET_E2E_M_S1_DROPPED_NEAR = 103,
  ET_E2E_M_S1_DROPPED_UTF8 = 104,
  ET_E2E_M_S1_DROPPED_LENGTH = 105,
  ET_E2E_M_S1_DROPPED_QUALITY = 106,
  ET_E2E_M_S1_KEPT_BYTES = 107,

  ET_E2E_M_S2_VOCAB_SIZE = 200,
  ET_E2E_M_S2_MERGE_COUNT = 201,
  ET_E2E_M_S2_BOS_ID = 202,
  ET_E2E_M_S2_EOS_ID = 203,
  ET_E2E_M_S2_RAW_BYTES = 204,
  ET_E2E_M_S2_ENCODED_TOKENS = 205,
  ET_E2E_M_S2_LOSSLESS_VERIFIED = 206,
  ET_E2E_M_S2_RELOAD_SHA_MATCH = 207,

  ET_E2E_M_S3_PARAM_COUNT = 300,
  ET_E2E_M_S3_SHARD_TOKENS = 301,
  ET_E2E_M_S3_STEPS_FULL = 302,
  ET_E2E_M_S3_STEPS_SPLIT = 303,
  ET_E2E_M_S3_WEIGHTS_BITWISE_EQUAL = 304,
  ET_E2E_M_S3_SNAPSHOT_SHA_EQUAL = 305,
  ET_E2E_M_S3_CORRUPT_REJECTED = 306,

  ET_E2E_M_S4_RANK = 400,
  ET_E2E_M_S4_STEPS = 401,
  ET_E2E_M_S4_BASE_FROZEN_EQUAL = 402,
  ET_E2E_M_S4_RELOAD_EQUAL = 403,

  ET_E2E_M_S5_STEPS = 500,
  ET_E2E_M_S5_REF_FROZEN_EQUAL = 501,

  ET_E2E_M_S6_CONTAM_STATUS = 600,
  ET_E2E_M_S6_CONTAM_MANIFEST_ABSENT = 601,
  ET_E2E_M_S6_CONTAM_DOCS_FOUND = 602,
  ET_E2E_M_S6_CLEAN_CONTAM_DOCS = 603,
  ET_E2E_M_S6_FRESH_RELOAD_VERIFIED = 604,

  ET_E2E_M_S7_HORIZON = 700,
  ET_E2E_M_S7_GREEDY_EMITTED = 701,
  ET_E2E_M_S7_GREEDY_ACCEPTED = 702,
  ET_E2E_M_S7_GREEDY_EXACT_MATCH = 703,
  ET_E2E_M_S7_STOCH_EMITTED = 704,
  ET_E2E_M_S7_STOCH_REPLAY_MATCH = 705,
  ET_E2E_M_S7_EOS_HIT = 706,
  ET_E2E_M_S7_EOS_EMITTED = 707,

  ET_E2E_M_S8_WEIGHT_COUNT = 800,
  ET_E2E_M_S8_Q8_0_BYTES = 801
};

/* Floating-point metric IDs for et_e2e_aot_get_f64. */
enum {
  ET_E2E_F_S2_COMPRESSION_RATIO = 250,

  ET_E2E_F_S3_LOSS_STEP1 = 350,
  ET_E2E_F_S3_LOSS_STEP4 = 351,
  ET_E2E_F_S3_LOSS_STEP8 = 352,

  ET_E2E_F_S4_INITIAL_LOSS = 450,
  ET_E2E_F_S4_FINAL_LOSS = 451,
  ET_E2E_F_S4_MERGE_MAX_ABS_ERR = 452,
  ET_E2E_F_S4_UNMERGE_MAX_ABS_ERR = 453,

  ET_E2E_F_S5_INITIAL_LOSS = 550,
  ET_E2E_F_S5_FINAL_LOSS = 551,
  ET_E2E_F_S5_INITIAL_MARGIN = 552,
  ET_E2E_F_S5_FINAL_MARGIN = 553,
  ET_E2E_F_S5_FINAL_ACCURACY = 554,

  ET_E2E_F_S6_LM_MEAN_NLL = 650,
  ET_E2E_F_S6_LM_PERPLEXITY = 651,
  ET_E2E_F_S6_MC_ACCURACY = 652,

  ET_E2E_F_S8_F16_MAX_ABS_ERR = 850,
  ET_E2E_F_S8_BF16_MAX_ABS_ERR = 851,
  ET_E2E_F_S8_Q8_0_MAX_ABS_ERR = 852,
  ET_E2E_F_S8_Q8_0_REL_L2_ERR = 853
};

/* Digest IDs for et_e2e_aot_get_sha256_hex. */
enum {
  ET_E2E_DIGEST_S1_REPORT = 1,
  ET_E2E_DIGEST_S2_TOKENIZER = 2,
  ET_E2E_DIGEST_S3_FULL_SNAP = 3,
  ET_E2E_DIGEST_S3_RESUMED_SNAP = 4,
  ET_E2E_DIGEST_S4_BASE_WEIGHT = 5,
  ET_E2E_DIGEST_S4_LORA_CKPT = 6,
  ET_E2E_DIGEST_S5_REF_WEIGHT = 7,
  ET_E2E_DIGEST_S6_MANIFEST = 8
};

/* Context lifecycle for compiled Eshkol AOT callers. */
ET_E2E_API int32_t et_e2e_aot_create(et_e2e_aot_ctx **out_ctx);
ET_E2E_API void et_e2e_aot_destroy(et_e2e_aot_ctx *ctx);

/* Multi-process phase entrypoints composing Stages 1-8. */
ET_E2E_API int32_t et_e2e_aot_run_stage1_data4(et_e2e_aot_ctx *ctx,
                                               const char *work_dir);
ET_E2E_API int32_t et_e2e_aot_run_stage2_train_save(et_e2e_aot_ctx *ctx,
                                                    const char *work_dir);
ET_E2E_API int32_t et_e2e_aot_run_stage2_reload_encode(et_e2e_aot_ctx *ctx,
                                                       const char *work_dir);
ET_E2E_API int32_t et_e2e_aot_run_stage3_pretrain_part1(et_e2e_aot_ctx *ctx,
                                                        const char *work_dir);
ET_E2E_API int32_t et_e2e_aot_run_stage3_pretrain_resume(et_e2e_aot_ctx *ctx,
                                                         const char *work_dir);
ET_E2E_API int32_t et_e2e_aot_run_stage4_lora_train_save(et_e2e_aot_ctx *ctx,
                                                         const char *work_dir);
ET_E2E_API int32_t et_e2e_aot_run_stage4_lora_reload(et_e2e_aot_ctx *ctx,
                                                     const char *work_dir);
ET_E2E_API int32_t et_e2e_aot_run_stage5_dpo(et_e2e_aot_ctx *ctx,
                                             const char *work_dir);
ET_E2E_API int32_t et_e2e_aot_run_stage6_eval5(et_e2e_aot_ctx *ctx,
                                               const char *work_dir);
ET_E2E_API int32_t et_e2e_aot_run_stage6_verify_fresh(et_e2e_aot_ctx *ctx,
                                                      const char *work_dir);
ET_E2E_API int32_t et_e2e_aot_run_stage7_spec_decode(et_e2e_aot_ctx *ctx,
                                                     const char *work_dir);
ET_E2E_API int32_t et_e2e_aot_run_stage8_io5_quant(et_e2e_aot_ctx *ctx,
                                                   const char *work_dir);

/* Scalar, digest, and C-string accessors for Eshkol AOT callers. */
ET_E2E_API int64_t et_e2e_aot_get_i64(const et_e2e_aot_ctx *ctx,
                                      int32_t metric_id);
ET_E2E_API double et_e2e_aot_get_f64(const et_e2e_aot_ctx *ctx,
                                     int32_t metric_id);
ET_E2E_API int32_t et_e2e_aot_get_sha256_hex(const et_e2e_aot_ctx *ctx,
                                             int32_t digest_id, char *out_buf,
                                             int64_t capacity);
ET_E2E_API int64_t et_e2e_aot_read_byte(const char *buf, int64_t index);
ET_E2E_API et_e2e_status
et_e2e_aot_get_metrics(const et_e2e_aot_ctx *ctx,
                       et_e2e_stage_metrics_v1 *out_metrics);

#ifdef __cplusplus
}
#endif

#endif
