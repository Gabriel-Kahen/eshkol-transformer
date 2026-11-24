#ifndef ESHKOL_TRANSFORMER_E2E_LIFECYCLE_ABI_H
#define ESHKOL_TRANSFORMER_E2E_LIFECYCLE_ABI_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_E2E_API __attribute__((visibility("default")))
#else
#define ET_E2E_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ET_E2E_ABI_MAJOR 1u
#define ET_E2E_ABI_MINOR 0u
#define ET_E2E_SHA256_BYTES 32u
#define ET_E2E_SHA256_HEX_BYTES 65u

/* Model & pipeline shape constants shared across Stages 1-8.
 * Stage 3 transformer block:
 *   D = 16 (H = 2 heads, Dh = 8 head_dim; valid for MOD5 RoPE v1)
 *   D_ff = 32 (W_ff1 [32, 16] adapted by Stage 4 LoRA with rank r = 4)
 *   T = 8 sequence length per pretraining window
 *   Total params = 32 * V_bpe + 2048 (always a multiple of 32 for IO5 Q8_0).
 */
#define ET_E2E_MODEL_DIM 16u
#define ET_E2E_MODEL_HEADS 2u
#define ET_E2E_MODEL_HEAD_DIM 8u
#define ET_E2E_MODEL_FF_DIM 32u
#define ET_E2E_SEQ_LEN 8u
#define ET_E2E_PRETRAIN_STEPS 8u
#define ET_E2E_PRETRAIN_SPLIT_STEP 4u
#define ET_E2E_LORA_RANK 4u
#define ET_E2E_LORA_STEPS 10u
#define ET_E2E_DPO_BATCH 2u
#define ET_E2E_DPO_SEQ 4u
#define ET_E2E_DPO_VOCAB 64u
#define ET_E2E_DPO_STEPS 12u
#define ET_E2E_SPEC_HORIZON 4u

typedef int32_t et_e2e_status;
enum {
  ET_E2E_OK = 0,
  ET_E2E_ERR_ARGUMENT = -1,
  ET_E2E_ERR_ALLOC = -2,
  ET_E2E_ERR_IO = -3,
  ET_E2E_ERR_STAGE1_DATA4 = -10,
  ET_E2E_ERR_STAGE2_BPE = -20,
  ET_E2E_ERR_STAGE3_MOD5 = -30,
  ET_E2E_ERR_STAGE4_LORA = -40,
  ET_E2E_ERR_STAGE5_DPO = -50,
  ET_E2E_ERR_STAGE6_EVAL5 = -60,
  ET_E2E_ERR_STAGE7_SPEC = -70,
  ET_E2E_ERR_STAGE8_IO5 = -80,
  ET_E2E_ERR_INVARIANT = -90,
  ET_E2E_ERR_CONTAMINATED = -91,
  ET_E2E_ERR_NONFINITE = -92
};

/* Standard-layout aggregate telemetry across all 8 lifecycle stages. */
typedef struct et_e2e_stage_metrics_v1 {
  /* Stage 1: DATA4 corpus hygiene & near-dedup */
  uint64_t s1_input_docs;
  uint64_t s1_kept_docs;
  uint64_t s1_dropped_exact;
  uint64_t s1_dropped_near;
  uint64_t s1_dropped_utf8;
  uint64_t s1_dropped_length;
  uint64_t s1_dropped_quality;
  uint64_t s1_kept_bytes;
  uint8_t s1_report_sha256[ET_E2E_SHA256_BYTES];

  /* Stage 2: T2 deterministic BPE training & save/reload */
  uint32_t s2_vocab_size;
  uint32_t s2_merge_count;
  uint32_t s2_bos_id;
  uint32_t s2_eos_id;
  uint64_t s2_raw_bytes;
  uint64_t s2_encoded_tokens;
  double s2_compression_ratio;
  uint32_t s2_lossless_verified;
  uint32_t s2_reload_sha_match;
  uint8_t s2_tokenizer_sha256[ET_E2E_SHA256_BYTES];

  /* Stage 3: MOD5 + RoPE + AdamW pretraining & 4+4 snapshot resume */
  uint64_t s3_param_count;
  uint64_t s3_shard_tokens;
  uint64_t s3_steps_full;
  uint64_t s3_steps_split;
  double s3_loss_step1;
  double s3_loss_step4;
  double s3_loss_step8;
  uint32_t s3_weights_bitwise_equal;
  uint32_t s3_snapshot_sha_equal;
  uint32_t s3_corrupt_rejected;
  uint32_t s3_reserved0;
  uint8_t s3_full_snap_sha256[ET_E2E_SHA256_BYTES];
  uint8_t s3_resumed_snap_sha256[ET_E2E_SHA256_BYTES];

  /* Stage 4: LoRA adapter fine-tuning & frozen base invariance */
  uint32_t s4_rank;
  uint32_t s4_steps;
  double s4_initial_loss;
  double s4_final_loss;
  double s4_merge_max_abs_err;
  double s4_unmerge_max_abs_err;
  uint32_t s4_base_frozen_equal;
  uint32_t s4_reload_equal;
  uint8_t s4_base_sha256[ET_E2E_SHA256_BYTES];
  uint8_t s4_ckpt_sha256[ET_E2E_SHA256_BYTES];

  /* Stage 5: DPO preference alignment */
  uint32_t s5_steps;
  uint32_t s5_ref_frozen_equal;
  double s5_initial_loss;
  double s5_final_loss;
  double s5_initial_margin;
  double s5_final_margin;
  double s5_final_accuracy;
  uint8_t s5_ref_sha256[ET_E2E_SHA256_BYTES];

  /* Stage 6: EVAL5 contamination-gated benchmark */
  int32_t s6_contam_status;
  uint32_t s6_contam_manifest_absent;
  uint64_t s6_contam_docs_found;
  uint64_t s6_clean_contam_docs;
  double s6_lm_mean_nll;
  double s6_lm_perplexity;
  double s6_mc_accuracy;
  uint32_t s6_fresh_reload_verified;
  uint32_t s6_reserved0;
  uint8_t s6_manifest_sha256[ET_E2E_SHA256_BYTES];

  /* Stage 7: Exact lossless speculative decoding */
  uint32_t s7_horizon;
  uint32_t s7_greedy_emitted;
  uint32_t s7_greedy_accepted;
  uint32_t s7_greedy_exact_match;
  uint32_t s7_stoch_emitted;
  uint32_t s7_stoch_replay_match;
  uint32_t s7_eos_hit;
  uint32_t s7_eos_emitted;

  /* Stage 8: Native IO5 F16/BF16/Q8_0 codec round-trip */
  uint64_t s8_weight_count;
  uint64_t s8_q8_0_bytes;
  double s8_f16_max_abs_err;
  double s8_bf16_max_abs_err;
  double s8_q8_0_max_abs_err;
  double s8_q8_0_rel_l2_err;
} et_e2e_stage_metrics_v1;

#if defined(__cplusplus)
#define ET_E2E_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_E2E_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_E2E_STATIC_ASSERT(offsetof(et_e2e_stage_metrics_v1, s1_input_docs) == 0u,
                     "et_e2e_stage_metrics_v1 layout drift");
ET_E2E_STATIC_ASSERT(sizeof(et_e2e_stage_metrics_v1) % 8u == 0u,
                     "et_e2e_stage_metrics_v1 must be 8-byte aligned");
#undef ET_E2E_STATIC_ASSERT

/* Execute all 8 stages end-to-end in a single process using work_dir for
 * persistent artifacts and save/reload boundaries. */
ET_E2E_API et_e2e_status et_e2e_run_full_in_process(
    const char *work_dir, et_e2e_stage_metrics_v1 *out_metrics);

/* Numerical central finite-difference gradient check for the Stage 3 MOD5 +
 * RoPE tiny transformer block across all 8 parameter tensors (W_emb, W_q,
 * W_k, W_v, W_o, W_ff1, W_ff2, W_lm). */
ET_E2E_API et_e2e_status
et_e2e_stage3_check_fd_gradients(double *out_max_abs_err);

#ifdef __cplusplus
}
#endif

#endif
