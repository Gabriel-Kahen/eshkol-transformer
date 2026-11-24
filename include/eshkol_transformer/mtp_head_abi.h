#ifndef ESHKOL_TRANSFORMER_MTP_HEAD_ABI_H
#define ESHKOL_TRANSFORMER_MTP_HEAD_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MTP_ABI_MAJOR 1u
#define ET_MTP_ABI_MINOR 0u

/* DeepSeek-V3 sequential Multi-Token Prediction (MTP) bounded dimensions. */
#define ET_MTP_MIN_DEPTH 1u
#define ET_MTP_MAX_DEPTH 4u
#define ET_MTP_MIN_VOCAB 2u
#define ET_MTP_MAX_VOCAB 1024u
#define ET_MTP_MIN_HIDDEN 2u
#define ET_MTP_MAX_HIDDEN 128u
#define ET_MTP_MIN_FFN 2u
#define ET_MTP_MAX_FFN 256u
#define ET_MTP_MIN_BATCH 1u
#define ET_MTP_MAX_BATCH 16u
#define ET_MTP_MAX_SEQ_LEN 256u

#define ET_MTP_NO_EOS UINT32_MAX
#define ET_MTP_ANCHOR_FROM_MAIN UINT32_MAX

/* Versioned SHA-256 checksummed checkpoint format ("ETMT1000" v1.0). */
#define ET_MTP_CKPT_MAGIC "ETMT1000"
#define ET_MTP_CKPT_MAGIC_BYTES 8u
#define ET_MTP_CKPT_VERSION_MAJOR 1u
#define ET_MTP_CKPT_VERSION_MINOR 0u
#define ET_MTP_CKPT_HEADER_BYTES 128u
#define ET_MTP_CKPT_CHECKSUM_BYTES 32u
#define ET_MTP_MAX_CKPT_BYTES (64u * 1024u * 1024u)

typedef int32_t et_mtp_status_t;

enum {
  ET_MTP_OK = 0,
  ET_MTP_ERR_INVALID_ARG = -1,
  ET_MTP_ERR_UNSUPPORTED = -2,
  ET_MTP_ERR_SHAPE_MISMATCH = -3,
  ET_MTP_ERR_TOKEN_RANGE = -4,
  ET_MTP_ERR_NONFINITE = -5,
  ET_MTP_ERR_FP_ENV = -6,
  ET_MTP_ERR_ALLOC = -7,
  ET_MTP_ERR_IO = -8,
  ET_MTP_ERR_BAD_MAGIC = -9,
  ET_MTP_ERR_VERSION_MISMATCH = -10,
  ET_MTP_ERR_CHECKSUM_MISMATCH = -11,
  ET_MTP_ERR_TRUNCATED = -12,
  ET_MTP_ERR_CONFIG_MISMATCH = -13,
  ET_MTP_ERR_SYMLINK = -14
};

/*
 * Standard-layout DeepSeek-V3 MTP model + AdamW optimizer configuration.
 *
 * Architecture (depth D in 1..4):
 *   - Shared token embedding table E[V, H] and shared output projection
 *     W_lm[V, H] (when tie_weights == 1, W_lm shares storage with E and all
 *     main + MTP output-head + MTP embedding gradients accumulate into d_E).
 *   - Main trunk next-token head (k = 0) over positions t = 0 .. T - 2:
 *       logits_0[b, t] = W_lm @ RMSNorm(h^0[b, t], gamma_main)
 *       predicts tokens[b, t + 1], contributing L_main.
 *   - For each sequential MTP depth k = 1 .. D over positions
 *     t = 0 .. T - 2 - k (count = T - 1 - k >= 1, requiring T >= D + 2):
 *       u_h    = RMSNorm(h^{k-1}[b, t], gamma_h_k)
 *       u_e    = RMSNorm(E[tokens[b, t + k]], gamma_e_k)
 *       z_k    = W_proj_k @ concat(u_h, u_e)             ([H, 2H] @ [2H])
 *       h^k_t  = z_k + SwiGLU_k(RMSNorm(z_k, gamma_f_k)) (residual FFN)
 *       logits_k[b, t] = W_lm @ RMSNorm(h^k_t, gamma_o_k)
 *       predicts tokens[b, t + 1 + k], contributing L_k.
 *   - Total objective:
 *       L_total = L_main + (lambda_mtp / D) * sum_{k=1..D} L_k.
 *
 * All inner numerical operations accumulate in IEEE-754 binary64 under
 * FE_TONEAREST with FP contraction disabled and verified x87/MXCSR state.
 * Every API call is failure-atomic: outputs and model state remain byte-for-
 * byte untouched on any non-OK status.
 */
typedef struct et_mtp_config_v1 {
  size_t struct_size;
  uint32_t vocab_size;   /* V in [2, 1024] */
  uint32_t hidden_dim;   /* H in [2, 128] */
  uint32_t ffn_dim;      /* F in [2, 256] */
  uint32_t mtp_depth;    /* D in [1, 4] */
  uint32_t tie_weights;  /* 0 = untied (E, W_lm), 1 = tied (W_lm aliases E) */
  uint32_t reserved;     /* must be 0 */
  double rms_eps;        /* finite in (0.0, 1.0] */
  double lambda_mtp;     /* finite >= 0.0 */
  double lr;             /* AdamW learning rate, finite >= 0.0 */
  double beta1;          /* AdamW beta1, finite in [0.0, 1.0) */
  double beta2;          /* AdamW beta2, finite in [0.0, 1.0) */
  double eps;            /* AdamW epsilon, finite > 0.0 */
  double weight_decay;   /* AdamW decoupled weight decay, finite >= 0.0 */
  double max_grad_norm;  /* Global L2 clip norm, finite >= 0.0 (0 = off) */
} et_mtp_config_v1;

/* Exact deterministic flat-buffer parameter offsets for all parameter groups. */
typedef struct et_mtp_layout_v1 {
  size_t struct_size;
  size_t total_params;
  size_t embed_offset;
  size_t embed_len;
  size_t lm_head_offset; /* Equals embed_offset when tie_weights == 1. */
  size_t lm_head_len;    /* 0 when tie_weights == 1, else V * H. */
  size_t gamma_main_offset;
  size_t gamma_main_len;
  size_t head_offset[ET_MTP_MAX_DEPTH];
  size_t head_stride;
  size_t rel_gamma_h;
  size_t rel_gamma_e;
  size_t rel_w_proj;
  size_t rel_gamma_f;
  size_t rel_w_gate;
  size_t rel_w_up;
  size_t rel_w_down;
  size_t rel_gamma_o;
} et_mtp_layout_v1;

/* Exact binary64 loss breakdown returned by forward and forward_backward. */
typedef struct et_mtp_loss_stats_v1 {
  double loss_total;
  double loss_main;
  double loss_mtp_avg;
  double loss_depth[ET_MTP_MAX_DEPTH];
  uint32_t main_token_count;
  uint32_t depth_token_count[ET_MTP_MAX_DEPTH];
} et_mtp_loss_stats_v1;

/* Diagnostics returned by an AdamW optimization step. */
typedef struct et_mtp_step_stats_v1 {
  uint64_t step;
  double grad_norm;
  double clip_scale;
  uint32_t clipped;
  uint32_t reserved;
} et_mtp_step_stats_v1;

/* Explicit deterministic counter-based SplitMix64 RNG state. */
typedef struct et_mtp_rng_state_v1 {
  uint64_t seed;
  uint64_t counter;
} et_mtp_rng_state_v1;

/* Verification result for self-speculative MTP draft + exact verification. */
typedef struct et_mtp_verify_result_v1 {
  uint32_t anchor_token_used;
  uint32_t accepted_count; /* 0 .. D accepted draft tokens */
  uint32_t emitted_count;  /* 1 .. D + 1 emitted tokens */
  uint32_t eos_hit;        /* 1 if EOS token was emitted */
  uint32_t rejected;       /* 1 if any draft token was rejected */
  uint32_t reserved;
  uint64_t rng_draws;      /* Exact SplitMix64 draws consumed (0 for greedy) */
  et_mtp_rng_state_v1 next_rng;
} et_mtp_verify_result_v1;

/* Opaque handle owning MTP weights, AdamW moments (m, v, step), and config. */
typedef struct et_mtp_model et_mtp_model_t;

et_mtp_status_t et_mtp_validate_config_v1(const et_mtp_config_v1 *cfg);
et_mtp_status_t et_mtp_compute_layout_v1(const et_mtp_config_v1 *cfg,
                                         et_mtp_layout_v1 *out_layout);

et_mtp_status_t et_mtp_model_create_v1(const et_mtp_config_v1 *cfg,
                                       et_mtp_model_t **out_model);
void et_mtp_model_destroy_v1(et_mtp_model_t *model);

/* Deterministic SplitMix64 initialization of weights; zeroes m, v, step. */
et_mtp_status_t et_mtp_model_init_weights_v1(et_mtp_model_t *model,
                                             uint64_t seed);

et_mtp_status_t et_mtp_model_get_config_v1(const et_mtp_model_t *model,
                                           et_mtp_config_v1 *out_cfg);
et_mtp_status_t et_mtp_model_get_layout_v1(const et_mtp_model_t *model,
                                           et_mtp_layout_v1 *out_layout);
et_mtp_status_t et_mtp_model_get_step_v1(const et_mtp_model_t *model,
                                         uint64_t *out_step);

et_mtp_status_t et_mtp_model_get_params_v1(const et_mtp_model_t *model,
                                           float *out_params, size_t count);
et_mtp_status_t et_mtp_model_set_params_v1(et_mtp_model_t *model,
                                           const float *params, size_t count);
et_mtp_status_t et_mtp_model_get_moments_v1(const et_mtp_model_t *model,
                                            float *out_m, float *out_v,
                                            size_t count);

/* Canonical SHA-256 digest over step + params + m + v (little-endian). */
et_mtp_status_t et_mtp_model_state_sha256_v1(const et_mtp_model_t *model,
                                             uint8_t out_digest[32]);

/* Main head logits: logits_out[V] = W_lm @ RMSNorm(h0_vec[H], gamma_main). */
et_mtp_status_t et_mtp_main_logits_v1(const et_mtp_model_t *model,
                                      const float *h0_vec, float *logits_out);

/*
 * Loss-only forward pass for main + D sequential MTP heads.
 *   h0     : float[batch, seq_len, hidden_dim]
 *   tokens : uint32_t[batch, seq_len], each < vocab_size
 * Requires seq_len >= mtp_depth + 2.
 */
et_mtp_status_t et_mtp_forward_v1(const et_mtp_model_t *model, size_t batch,
                                  size_t seq_len, const float *h0,
                                  const uint32_t *tokens,
                                  et_mtp_loss_stats_v1 *out_stats);

/*
 * Exact forward + analytic reverse-depth backward pass through k = D .. 1 and
 * main head k = 0.
 *   d_h0_out     : float[batch, seq_len, hidden_dim] (exact gradient w.r.t. h0)
 *   d_params_out : float[total_params] (exact summed gradients across main + D)
 */
et_mtp_status_t et_mtp_forward_backward_v1(
    const et_mtp_model_t *model, size_t batch, size_t seq_len, const float *h0,
    const uint32_t *tokens, et_mtp_loss_stats_v1 *out_stats, float *d_h0_out,
    float *d_params_out, size_t d_params_count);

/* Failure-atomic binary64 decoupled AdamW update with optional L2 clip. */
et_mtp_status_t et_mtp_adamw_step_v1(et_mtp_model_t *model,
                                     const float *d_params,
                                     size_t d_params_count,
                                     et_mtp_step_stats_v1 *out_stats);

/*
 * Self-speculative MTP drafting (sequential heads k = 1 .. D from h0_prefix
 * and anchor_token) + exact verification against target_logits[D + 1, V].
 *
 * - temperature == 0.0 (positive zero): exact greedy draft & verification
 *   with lowest-index tie breaking (0 RNG draws; rng_in may be NULL).
 * - temperature > 0.0: exact stochastic draft + Leviathan et al. rejection
 *   sampling in binary64 using deterministic SplitMix64 counter RNG (rng_in
 *   must be non-NULL).
 * - target_logits[i, :] (i = 0 .. D - 1) verifies draft_tokens[i];
 *   target_logits[D, :] provides the bonus token when all D drafts accept.
 */
et_mtp_status_t et_mtp_draft_verify_v1(
    const et_mtp_model_t *model, const float *h0_prefix, uint32_t anchor_token,
    const float *target_logits, double temperature, uint32_t eos_token_id,
    const et_mtp_rng_state_v1 *rng_in, uint32_t *draft_tokens_out,
    uint32_t *out_tokens, uint32_t out_capacity,
    et_mtp_verify_result_v1 *out_result);

/* Atomic 0600 temp+fsync+rename save and O_NOFOLLOW verified load (ETMT1000). */
et_mtp_status_t et_mtp_checkpoint_save_v1(const et_mtp_model_t *model,
                                          const char *path);
et_mtp_status_t et_mtp_checkpoint_load_v1(et_mtp_model_t *model,
                                          const char *path);

/* Standalone FIPS 180-4 SHA-256 helper. */
et_mtp_status_t et_mtp_sha256_bytes_v1(const void *data, size_t len,
                                       uint8_t out_digest[32]);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TRANSFORMER_MTP_HEAD_ABI_H */
