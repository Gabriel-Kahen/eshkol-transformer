#ifndef ESHKOL_TRANSFORMER_LORA_ADAPTER_ABI_H
#define ESHKOL_TRANSFORMER_LORA_ADAPTER_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_LORA_ABI_MAJOR 1u
#define ET_LORA_ABI_MINOR 0u

#define ET_LORA_MAX_M 64u
#define ET_LORA_MAX_K 64u
#define ET_LORA_MAX_N 64u
#define ET_LORA_MAX_R 16u

#define ET_LORA_SHA256_BYTES 32u
#define ET_LORA_CHECKPOINT_MAGIC "ETLR1000"
#define ET_LORA_CHECKPOINT_MAGIC_BYTES 8u
#define ET_LORA_CHECKPOINT_VERSION_MAJOR 1u
#define ET_LORA_CHECKPOINT_VERSION_MINOR 0u
#define ET_LORA_CHECKPOINT_HEADER_BYTES 64u
#define ET_LORA_CHECKPOINT_CHECKSUM_BYTES 32u
#define ET_LORA_CHECKPOINT_MIN_BYTES \
  (ET_LORA_CHECKPOINT_HEADER_BYTES + ET_LORA_CHECKPOINT_CHECKSUM_BYTES)
#define ET_LORA_CHECKPOINT_MAX_PAYLOAD_FLOATS \
  ((ET_LORA_MAX_R * ET_LORA_MAX_K) + (ET_LORA_MAX_N * ET_LORA_MAX_R))
#define ET_LORA_CHECKPOINT_MAX_BYTES \
  (ET_LORA_CHECKPOINT_HEADER_BYTES + \
   (ET_LORA_CHECKPOINT_MAX_PAYLOAD_FLOATS * 4u) + \
   ET_LORA_CHECKPOINT_CHECKSUM_BYTES)

typedef int32_t et_lora_status;

enum {
  ET_LORA_OK = 0,
  ET_LORA_ERR_ARGUMENT = -1,
  ET_LORA_ERR_SHAPE = -2,
  ET_LORA_ERR_ALIGNMENT = -3,
  ET_LORA_ERR_ALIAS = -4,
  ET_LORA_ERR_NONFINITE = -5,
  ET_LORA_ERR_FP_ENV = -6,
  ET_LORA_ERR_FORMAT = -7,
  ET_LORA_ERR_VERSION = -8,
  ET_LORA_ERR_TRUNCATED = -9,
  ET_LORA_ERR_CHECKSUM = -10,
  ET_LORA_ERR_BASE_HASH = -11,
  ET_LORA_ERR_SYMLINK = -12,
  ET_LORA_ERR_IO = -13,
  ET_LORA_ERR_NOMEM = -14,
  ET_LORA_ERR_OVERFLOW = -15
};

/* Dense row-major binary32 LoRA linear projection configuration.
 * Constraints:
 *   1 <= m <= ET_LORA_MAX_M (64)
 *   1 <= k <= ET_LORA_MAX_K (64)
 *   1 <= n <= ET_LORA_MAX_N (64)
 *   1 <= r <= ET_LORA_MAX_R (16), and r <= min(k, n)
 *   alpha finite binary32, alpha > 0.0f, and (alpha / (float)r) finite.
 *
 * Tensor shapes (row-major binary32):
 *   X:      [m, k]
 *   W:      [n, k] (frozen base weight; read-only const float *)
 *   A:      [r, k]
 *   B:      [n, r]
 *   Y, dY:  [m, n]
 *   dX:     [m, k]
 *   dA:     [r, k]
 *   dB:     [n, r]
 */
typedef struct et_lora_config {
  uint32_t m;
  uint32_t k;
  uint32_t n;
  uint32_t r;
  float alpha;
} et_lora_config;

typedef struct et_lora_checkpoint_meta {
  uint32_t version_major;
  uint32_t version_minor;
  uint32_t k;
  uint32_t n;
  uint32_t r;
  float alpha;
  uint8_t base_weight_sha256[ET_LORA_SHA256_BYTES];
  uint8_t checkpoint_sha256[ET_LORA_SHA256_BYTES];
} et_lora_checkpoint_meta;

#if defined(__cplusplus)
#define ET_LORA_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_LORA_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_LORA_STATIC_ASSERT(sizeof(et_lora_config) == 20u &&
                          offsetof(et_lora_config, alpha) == 16u,
                      "et_lora_config ABI layout changed");
ET_LORA_STATIC_ASSERT(sizeof(et_lora_checkpoint_meta) == 88u &&
                          offsetof(et_lora_checkpoint_meta,
                                   base_weight_sha256) == 24u &&
                          offsetof(et_lora_checkpoint_meta,
                                   checkpoint_sha256) == 56u,
                      "et_lora_checkpoint_meta ABI layout changed");
#undef ET_LORA_STATIC_ASSERT

/* Validate configuration bounds and finite alpha > 0 (plus FP environment). */
et_lora_status et_lora_validate_config(const et_lora_config *cfg);

/* Deterministic adapter initialization:
 * A [r, k] receives finite seeded non-zero values; B [n, r] is zero-filled
 * (+0.0f) so initial adapted forward matches base forward bitwise. */
et_lora_status et_lora_init_adapter(const et_lora_config *cfg, uint64_t seed,
                                    float *a, float *b);
et_lora_status et_lora_init_adapter_raw(const et_lora_config *cfg,
                                        uint64_t seed, void *a, void *b);

/* Frozen base linear projection: Y[m, n] = X[m, k] @ W[n, k]^T. */
et_lora_status et_lora_base_forward(const et_lora_config *cfg, const float *x,
                                    const float *w, float *y);
et_lora_status et_lora_base_forward_raw(const et_lora_config *cfg,
                                        const void *x, const void *w, void *y);

/* LoRA linear projection forward:
 * Y = X @ W^T + (alpha / r) * ((X @ A^T) @ B^T).
 * Base weight W is strictly frozen (const float *). All buffers are pairwise
 * disjoint and 4-byte aligned. Failure leaves Y untouched. */
et_lora_status et_lora_forward(const et_lora_config *cfg, const float *x,
                               const float *w, const float *a, const float *b,
                               float *y);
et_lora_status et_lora_forward_raw(const et_lora_config *cfg, const void *x,
                                   const void *w, const void *a, const void *b,
                                   void *y);

/* LoRA linear projection analytic backward:
 *   H  = X @ A^T                         [m, r]
 *   dB = (alpha / r) * dY^T @ H          [n, r]
 *   dH = (alpha / r) * dY @ B            [m, r]
 *   dA = dH^T @ X                        [r, k]
 *   dX = dY @ W + dH @ A                 [m, k]
 * Base weight W is strictly frozen (const float *). All buffers are pairwise
 * disjoint and 4-byte aligned. Failure leaves dX, dA, dB untouched. */
et_lora_status et_lora_backward(const et_lora_config *cfg, const float *x,
                                const float *w, const float *a, const float *b,
                                const float *dy, float *dx, float *da,
                                float *db);
et_lora_status et_lora_backward_raw(const et_lora_config *cfg, const void *x,
                                    const void *w, const void *a, const void *b,
                                    const void *dy, void *dx, void *da,
                                    void *db);

/* Weight merge: W_merged[n, k] = W_base[n, k] + (alpha / r) * (B @ A). */
et_lora_status et_lora_merge(const et_lora_config *cfg, const float *w_base,
                             const float *a, const float *b, float *w_merged);
et_lora_status et_lora_merge_raw(const et_lora_config *cfg, const void *w_base,
                                 const void *a, const void *b, void *w_merged);

/* Weight unmerge: W_restored[n, k] = W_merged[n, k] - (alpha / r) * (B @ A). */
et_lora_status et_lora_unmerge(const et_lora_config *cfg, const float *w_merged,
                               const float *a, const float *b,
                               float *w_restored);
et_lora_status et_lora_unmerge_raw(const et_lora_config *cfg,
                                   const void *w_merged, const void *a,
                                   const void *b, void *w_restored);

/* Canonical SHA-256 digest of frozen base weight W[n, k] (little-endian f32). */
et_lora_status et_lora_base_weight_sha256(
    uint32_t k, uint32_t n, const float *w_base,
    uint8_t out_digest[ET_LORA_SHA256_BYTES]);
et_lora_status et_lora_base_weight_sha256_raw(uint32_t k, uint32_t n,
                                              const void *w_base,
                                              void *out_digest);

/* Exact serialized byte count for a valid (k, n, r) adapter checkpoint. */
et_lora_status et_lora_checkpoint_size(uint32_t k, uint32_t n, uint32_t r,
                                       size_t *out_bytes);

/* Buffer-level ETLR1000 v1.0 serialize and parse. */
et_lora_status et_lora_checkpoint_serialize(const et_lora_config *cfg,
                                            const float *w_base, const float *a,
                                            const float *b, uint8_t *out_bytes,
                                            size_t capacity, size_t *out_size);
et_lora_status et_lora_checkpoint_parse(
    const uint8_t *bytes, size_t size, uint32_t expected_k, uint32_t expected_n,
    uint32_t expected_r, const float *w_base, et_lora_checkpoint_meta *out_meta,
    float *out_a, float *out_b);

/* Atomic fsync + rename binary adapter checkpoint save (mode 0600,
 * non-executable) and O_NOFOLLOW checked load. */
et_lora_status et_lora_checkpoint_save(const char *path,
                                       const et_lora_config *cfg,
                                       const float *w_base, const float *a,
                                       const float *b);
et_lora_status et_lora_checkpoint_save_raw(const char *path,
                                           const et_lora_config *cfg,
                                           const void *w_base, const void *a,
                                           const void *b);

et_lora_status et_lora_checkpoint_load(const char *path, uint32_t expected_k,
                                       uint32_t expected_n, uint32_t expected_r,
                                       const float *w_base,
                                       et_lora_checkpoint_meta *out_meta,
                                       float *out_a, float *out_b);
et_lora_status et_lora_checkpoint_load_raw(const char *path,
                                           uint32_t expected_k,
                                           uint32_t expected_n,
                                           uint32_t expected_r,
                                           const void *w_base,
                                           et_lora_checkpoint_meta *out_meta,
                                           void *out_a, void *out_b);

#ifdef __cplusplus
}
#endif

#endif
