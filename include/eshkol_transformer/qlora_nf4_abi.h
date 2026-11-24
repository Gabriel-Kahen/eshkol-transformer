#ifndef ESHKOL_TRANSFORMER_QLORA_NF4_ABI_H
#define ESHKOL_TRANSFORMER_QLORA_NF4_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_QLORA_ABI_MAJOR 1u
#define ET_QLORA_ABI_MINOR 0u

/* Bounded dimensions preventing size_t overflow while covering multi-layer
 * QLoRA projections and canonical 256-block double quantization. */
#define ET_QLORA_MAX_M 128u
#define ET_QLORA_MAX_IN 512u
#define ET_QLORA_MAX_OUT 512u
#define ET_QLORA_MAX_R 32u

/* NF4 first-level block sizes (power of 2 in {32, 64, 128}, default 64). */
#define ET_QLORA_BLOCK_SIZE_32 32u
#define ET_QLORA_BLOCK_SIZE_64 64u
#define ET_QLORA_BLOCK_SIZE_128 128u
#define ET_QLORA_DEFAULT_BLOCK_SIZE ET_QLORA_BLOCK_SIZE_64

/* NF4 second-level Double Quantization (DQ) block sizes in {16, 32, 64, 256}. */
#define ET_QLORA_DQ_BLOCK_SIZE_16 16u
#define ET_QLORA_DQ_BLOCK_SIZE_32 32u
#define ET_QLORA_DQ_BLOCK_SIZE_64 64u
#define ET_QLORA_DQ_BLOCK_SIZE_256 256u
#define ET_QLORA_DEFAULT_DQ_BLOCK_SIZE ET_QLORA_DQ_BLOCK_SIZE_64

/* Versioned checksummed QLoRA checkpoint format (ETQL1000 v1.0). */
#define ET_QLORA_SHA256_BYTES 32u
#define ET_QLORA_CHECKPOINT_MAGIC "ETQL1000"
#define ET_QLORA_CHECKPOINT_MAGIC_BYTES 8u
#define ET_QLORA_CHECKPOINT_VERSION_MAJOR 1u
#define ET_QLORA_CHECKPOINT_VERSION_MINOR 0u
#define ET_QLORA_CHECKPOINT_HEADER_BYTES 128u
#define ET_QLORA_CHECKPOINT_CHECKSUM_BYTES 32u
#define ET_QLORA_CHECKPOINT_MIN_BYTES \
  (ET_QLORA_CHECKPOINT_HEADER_BYTES + ET_QLORA_CHECKPOINT_CHECKSUM_BYTES)

typedef int32_t et_qlora_status;
typedef uint32_t et_qlora_quant_mode;

enum {
  ET_QLORA_QUANT_NF4_SQ = 1u, /* Single-quantized: 4-bit NF4 + FP32 absmax */
  ET_QLORA_QUANT_NF4_DQ = 2u  /* Double-quantized: 4-bit NF4 + uint8 absmax + per-DQ (offset_f32, scale_f32) */
};

enum {
  ET_QLORA_OK = 0,
  ET_QLORA_ERR_ARGUMENT = -1,
  ET_QLORA_ERR_SHAPE = -2,
  ET_QLORA_ERR_ALIGNMENT = -3,
  ET_QLORA_ERR_ALIAS = -4,
  ET_QLORA_ERR_NONFINITE = -5,
  ET_QLORA_ERR_FP_ENV = -6,
  ET_QLORA_ERR_FORMAT = -7,
  ET_QLORA_ERR_VERSION = -8,
  ET_QLORA_ERR_TRUNCATED = -9,
  ET_QLORA_ERR_CHECKSUM = -10,
  ET_QLORA_ERR_BASE_HASH = -11,
  ET_QLORA_ERR_SYMLINK = -12,
  ET_QLORA_ERR_IO = -13,
  ET_QLORA_ERR_NOMEM = -14,
  ET_QLORA_ERR_OVERFLOW = -15,
  ET_QLORA_ERR_CAPACITY = -16
};

/* Exact Dettmers et al. (2023) 16-entry NormalFloat4 (NF4) quantile codebook. */
extern const float ET_NF4_CODEBOOK[16];

/* NF4 quantization specification for a row-major weight matrix [rows_out, cols_in].
 * Constraints:
 *   1 <= rows_out <= ET_QLORA_MAX_OUT (512)
 *   block_size in {32, 64, 128}; 1 <= cols_in <= ET_QLORA_MAX_IN (512);
 *   cols_in % block_size == 0 (blocks never cross rows).
 *   mode == ET_QLORA_QUANT_NF4_SQ => dq_block_size must be 0.
 *   mode == ET_QLORA_QUANT_NF4_DQ => dq_block_size in {16, 32, 64, 256}.
 *   reserved must be 0. */
typedef struct et_qlora_quant_spec {
  uint32_t rows_out;
  uint32_t cols_in;
  uint32_t block_size;
  uint32_t dq_block_size;
  et_qlora_quant_mode mode;
  uint32_t reserved;
} et_qlora_quant_spec;

/* Exact memory footprint report for a quantized NF4 weight matrix. */
typedef struct et_qlora_quant_footprint {
  uint64_t f32_bytes;
  uint64_t packed_bytes;
  uint64_t nibble_bytes;
  uint64_t scale_bytes;
  double bits_per_param;
} et_qlora_quant_footprint;

/* Fused frozen-NF4 + LoRA linear layer configuration.
 * Tensor shapes (row-major):
 *   X:      [m, cols_in]            (read-only f32)
 *   W_nf4:  packed NF4 [rows_out, cols_in] (strictly frozen const uint8_t *)
 *   bias:   optional [rows_out]     (read-only f32 or NULL)
 *   A:      [r, cols_in]            (trainable f32)
 *   B:      [rows_out, r]           (trainable f32)
 *   Y, dY:  [m, rows_out]           (f32)
 *   dX:     [m, cols_in]            (f32)
 *   dA:     [r, cols_in]            (f32)
 *   dB:     [rows_out, r]           (f32)
 *   dbias:  optional [rows_out]     (f32 or NULL)
 */
typedef struct et_qlora_linear_config {
  et_qlora_quant_spec quant;
  uint32_t m;
  uint32_t r;
  float alpha;
  uint32_t reserved;
} et_qlora_linear_config;

/* Decoupled AdamW optimizer configuration and step counter. */
typedef struct et_qlora_adamw_config {
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  uint32_t reserved;
  uint64_t step;
} et_qlora_adamw_config;

/* 2-layer QLoRA MLP/projection model configuration:
 *   Layer 1: [d_in] -> [d_hidden] (frozen NF4 W1 + optional b1 + trainable A1, B1)
 *   Activation: SiLU(z) = z / (1 + exp(-z))
 *   Layer 2: [d_hidden] -> [d_out] (frozen NF4 W2 + optional b2 + trainable A2, B2)
 */
typedef struct et_qlora_mlp_config {
  uint32_t m;
  uint32_t d_in;
  uint32_t d_hidden;
  uint32_t d_out;
  uint32_t r;
  uint32_t block_size;
  uint32_t dq_block_size;
  et_qlora_quant_mode quant_mode;
  float alpha;
  uint32_t reserved;
} et_qlora_mlp_config;

/* Metadata decoded from an ETQL1000 v1.0 QLoRA checkpoint. */
typedef struct et_qlora_checkpoint_meta {
  uint32_t version_major;
  uint32_t version_minor;
  uint32_t d_in;
  uint32_t d_hidden;
  uint32_t d_out;
  uint32_t r;
  uint32_t block_size;
  uint32_t dq_block_size;
  et_qlora_quant_mode quant_mode;
  float alpha;
  uint64_t adamw_step;
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  uint32_t flags;
  uint8_t base_nf4_sha256[ET_QLORA_SHA256_BYTES];
  uint8_t checkpoint_sha256[ET_QLORA_SHA256_BYTES];
} et_qlora_checkpoint_meta;

#if defined(__cplusplus)
#define ET_QLORA_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_QLORA_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_QLORA_STATIC_ASSERT(sizeof(et_qlora_quant_spec) == 24u &&
                           offsetof(et_qlora_quant_spec, mode) == 16u,
                       "et_qlora_quant_spec ABI layout changed");
ET_QLORA_STATIC_ASSERT(sizeof(et_qlora_quant_footprint) == 40u &&
                           offsetof(et_qlora_quant_footprint, bits_per_param) ==
                               32u,
                       "et_qlora_quant_footprint ABI layout changed");
ET_QLORA_STATIC_ASSERT(sizeof(et_qlora_linear_config) == 40u &&
                           offsetof(et_qlora_linear_config, alpha) == 32u,
                       "et_qlora_linear_config ABI layout changed");
ET_QLORA_STATIC_ASSERT(sizeof(et_qlora_adamw_config) == 32u &&
                           offsetof(et_qlora_adamw_config, step) == 24u,
                       "et_qlora_adamw_config ABI layout changed");
ET_QLORA_STATIC_ASSERT(sizeof(et_qlora_mlp_config) == 40u &&
                           offsetof(et_qlora_mlp_config, alpha) == 32u,
                       "et_qlora_mlp_config ABI layout changed");
ET_QLORA_STATIC_ASSERT(
    sizeof(et_qlora_checkpoint_meta) == 136u &&
        offsetof(et_qlora_checkpoint_meta, adamw_step) == 40u &&
        offsetof(et_qlora_checkpoint_meta, base_nf4_sha256) == 72u &&
        offsetof(et_qlora_checkpoint_meta, checkpoint_sha256) == 104u,
    "et_qlora_checkpoint_meta ABI layout changed");
#undef ET_QLORA_STATIC_ASSERT

/* ---- Validation, footprint & SHA-256 ---- */

et_qlora_status et_qlora_validate_quant_spec(const et_qlora_quant_spec *spec);
et_qlora_status et_qlora_validate_linear_config(
    const et_qlora_linear_config *cfg);
et_qlora_status et_qlora_validate_mlp_config(const et_qlora_mlp_config *cfg);
et_qlora_status et_qlora_validate_adamw_config(
    const et_qlora_adamw_config *opt);

et_qlora_status et_qlora_packed_footprint(
    const et_qlora_quant_spec *spec, et_qlora_quant_footprint *out_footprint);

et_qlora_status et_qlora_sha256_bytes(const uint8_t *data, size_t size,
                                      uint8_t out_digest[ET_QLORA_SHA256_BYTES]);

/* ---- NF4 Quantization & Dequantization (SQ and DQ, zero heap alloc) ----
 * Packed layout:
 *   Single Quantization (ET_QLORA_QUANT_NF4_SQ):
 *     [0 .. N*K/2 - 1]         : packed 4-bit NF4 nibbles, 2/byte, low-nibble-first
 *                                (element 2*p in low nibble, 2*p+1 in high nibble)
 *     [N*K/2 .. +num_blocks*4] : little-endian IEEE-754 f32 absmax per block
 *
 *   Double Quantization (ET_QLORA_QUANT_NF4_DQ):
 *     [0 .. N*K/2 - 1]         : packed 4-bit NF4 nibbles, 2/byte, low-nibble-first
 *     [N*K/2 .. +num_blocks]   : uint8 quantized absmax q_s[b] in [0, 255]
 *     [+num_dq_blocks * 8]     : per-DQ-block (offset_f32, scale_f32) LE pairs
 *                                absmax_hat[b] = (float)((double)offset + (double)scale * q_s[b])
 *
 * Nearest-codebook rounding rule:
 *   Normalize in binary64: v = (double)w / (double)absmax.
 *   Midpoints mid[i] = 0.5 * ((double)ET_NF4_CODEBOOK[i] + (double)ET_NF4_CODEBOOK[i+1]).
 *   Select largest index q in [0, 15] such that v >= mid[q-1] (strict midpoint
 *   tie-break to larger index). All-zero blocks encode absmax=0.0f and nibble=7. */
et_qlora_status et_qlora_nf4_quantize(const et_qlora_quant_spec *spec,
                                      const float *weights_f32,
                                      uint8_t *out_packed,
                                      size_t out_packed_bytes);

et_qlora_status et_qlora_nf4_dequantize(const et_qlora_quant_spec *spec,
                                        const uint8_t *packed,
                                        size_t packed_bytes,
                                        float *out_weights_f32);

/* Deterministic LoRA initialization:
 * A [r, cols_in] receives deterministic finite non-zero seeded values;
 * B [rows_out, r] is zero-filled (+0.0f) so initial QLoRA forward matches
 * frozen NF4 base forward bitwise. */
et_qlora_status et_qlora_init_lora(uint32_t cols_in, uint32_t rows_out,
                                   uint32_t r, uint64_t seed, float *a,
                                   float *b);

/* ---- Fused Frozen-NF4 + LoRA Linear Kernels (zero heap alloc, binary64) ----
 * Forward:
 *   Y[m, out] = sum_k dequant(W_nf4[out, k]) * X[m, k]
 *               + (bias ? bias[out] : 0)
 *               + (alpha / r) * ((X @ A^T) @ B^T)[m, out]
 * If both a and b are NULL, computes frozen NF4 base-only projection. */
et_qlora_status et_qlora_linear_forward(const et_qlora_linear_config *cfg,
                                        const float *x, const uint8_t *w_nf4,
                                        size_t w_nf4_bytes, const float *bias,
                                        const float *a, const float *b,
                                        float *y);

/* Analytic backward (W_nf4 is strictly const / read-only):
 *   H     = X @ A^T                         [m, r]
 *   dB    = (alpha / r) * dY^T @ H          [rows_out, r]
 *   dH    = (alpha / r) * dY @ B            [m, r]
 *   dA    = dH^T @ X                        [r, cols_in]
 *   dX    = dY @ dequant(W_nf4) + dH @ A    [m, cols_in]
 *   dbias = sum_m dY[m, :]                  [rows_out] (optional; NULL skips) */
et_qlora_status et_qlora_linear_backward(const et_qlora_linear_config *cfg,
                                         const float *x, const uint8_t *w_nf4,
                                         size_t w_nf4_bytes, const float *a,
                                         const float *b, const float *dy,
                                         float *dx, float *da, float *db,
                                         float *dbias);

/* Merge dequantized NF4 base + LoRA delta into dense f32 [rows_out, cols_in]:
 *   W_merged[out, k] = dequant(W_nf4[out, k]) + (alpha / r) * (B @ A)[out, k] */
et_qlora_status et_qlora_merge_to_f32(const et_qlora_linear_config *cfg,
                                      const uint8_t *w_nf4, size_t w_nf4_bytes,
                                      const float *a, const float *b,
                                      float *out_w_f32);

/* Dense binary32 reference linear forward (binary64 inner accumulation):
 *   Y[m, out] = X[m, in] @ W_f32[out, in]^T + (bias ? bias[out] : 0) */
et_qlora_status et_qlora_f32_linear_forward(uint32_t m, uint32_t rows_out,
                                            uint32_t cols_in, const float *x,
                                            const float *w_f32,
                                            const float *bias, float *y);

/* ---- Decoupled AdamW Optimizer (zero heap alloc, binary64 state math) ---- */
et_qlora_status et_qlora_adamw_step(size_t count,
                                    const et_qlora_adamw_config *opt,
                                    float *params, const float *grads,
                                    float *m_state, float *v_state);

/* ---- 2-Layer QLoRA MLP/Projection Trainer (zero heap alloc) ---- */

/* Canonical SHA-256 binding both frozen NF4 layers (length-prefixed W1 || W2). */
et_qlora_status et_qlora_mlp_base_sha256(
    const et_qlora_mlp_config *cfg, const uint8_t *w1_nf4, size_t w1_bytes,
    const uint8_t *w2_nf4, size_t w2_bytes,
    uint8_t out_digest[ET_QLORA_SHA256_BYTES]);

/* Exact float count needed in scratch_f32 for et_qlora_mlp_train_step. */
et_qlora_status et_qlora_mlp_scratch_floats(const et_qlora_mlp_config *cfg,
                                            size_t *out_floats);

/* 2-layer QLoRA MLP forward:
 *   Z1 = Linear1(X; W1_nf4, b1, A1, B1)
 *   H1 = SiLU(Z1)
 *   Y  = Linear2(H1; W2_nf4, b2, A2, B2) */
et_qlora_status et_qlora_mlp_forward(
    const et_qlora_mlp_config *cfg, const float *x, const uint8_t *w1_nf4,
    size_t w1_bytes, const float *b1, const float *a1, const float *b1_lora,
    const uint8_t *w2_nf4, size_t w2_bytes, const float *b2, const float *a2,
    const float *b2_lora, float *out_z1, float *out_h1, float *out_y);

/* 2-layer QLoRA MLP single AdamW training step on LoRA (A1, B1, A2, B2) ONLY.
 * Frozen NF4 base weights W1_nf4 and W2_nf4 are strictly const and never mutated.
 * Reports MSE loss before parameter update in *out_loss and increments opt->step. */
et_qlora_status et_qlora_mlp_train_step(
    const et_qlora_mlp_config *cfg, et_qlora_adamw_config *opt, const float *x,
    const float *target, const uint8_t *w1_nf4, size_t w1_bytes,
    const float *b1, const uint8_t *w2_nf4, size_t w2_bytes, const float *b2,
    float *a1, float *b1_lora, float *a2, float *b2_lora, float *m_a1,
    float *v_a1, float *m_b1, float *v_b1, float *m_a2, float *v_a2,
    float *m_b2, float *v_b2, float *scratch_f32, size_t scratch_floats,
    double *out_loss);

/* ---- Versioned SHA-256 Checksummed Checkpoint (ETQL1000 v1.0) ---- */

et_qlora_status et_qlora_checkpoint_size(const et_qlora_mlp_config *cfg,
                                         size_t *out_bytes);

et_qlora_status et_qlora_checkpoint_serialize(
    const et_qlora_mlp_config *cfg, const et_qlora_adamw_config *opt,
    const uint8_t base_nf4_sha256[ET_QLORA_SHA256_BYTES], const float *a1,
    const float *b1_lora, const float *a2, const float *b2_lora,
    const float *m_a1, const float *v_a1, const float *m_b1, const float *v_b1,
    const float *m_a2, const float *v_a2, const float *m_b2, const float *v_b2,
    uint8_t *out_bytes, size_t capacity, size_t *out_size);

et_qlora_status et_qlora_checkpoint_parse(
    const uint8_t *bytes, size_t size, const et_qlora_mlp_config *expected_cfg,
    const uint8_t expected_base_sha256[ET_QLORA_SHA256_BYTES],
    et_qlora_checkpoint_meta *out_meta, et_qlora_adamw_config *out_opt,
    float *out_a1, float *out_b1_lora, float *out_a2, float *out_b2_lora,
    float *out_m_a1, float *out_v_a1, float *out_m_b1, float *out_v_b1,
    float *out_m_a2, float *out_v_a2, float *out_m_b2, float *out_v_b2);

/* Atomic file save (mode 0600 sibling mkstemp + write + fsync + close + rename)
 * and O_NOFOLLOW + S_ISREG verified load. */
et_qlora_status et_qlora_checkpoint_save(
    const char *path, const et_qlora_mlp_config *cfg,
    const et_qlora_adamw_config *opt,
    const uint8_t base_nf4_sha256[ET_QLORA_SHA256_BYTES], const float *a1,
    const float *b1_lora, const float *a2, const float *b2_lora,
    const float *m_a1, const float *v_a1, const float *m_b1, const float *v_b1,
    const float *m_a2, const float *v_a2, const float *m_b2, const float *v_b2);

et_qlora_status et_qlora_checkpoint_load(
    const char *path, const et_qlora_mlp_config *expected_cfg,
    const uint8_t expected_base_sha256[ET_QLORA_SHA256_BYTES],
    et_qlora_checkpoint_meta *out_meta, et_qlora_adamw_config *out_opt,
    float *out_a1, float *out_b1_lora, float *out_a2, float *out_b2_lora,
    float *out_m_a1, float *out_v_a1, float *out_m_b1, float *out_v_b1,
    float *out_m_a2, float *out_v_a2, float *out_m_b2, float *out_v_b2);

#ifdef __cplusplus
}
#endif

#endif
