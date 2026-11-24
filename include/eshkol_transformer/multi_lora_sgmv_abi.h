#ifndef ESHKOL_TRANSFORMER_MULTI_LORA_SGMV_ABI_H
#define ESHKOL_TRANSFORMER_MULTI_LORA_SGMV_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MLORA_ABI_MAJOR 1u
#define ET_MLORA_ABI_MINOR 0u

#define ET_MLORA_MAX_ADAPTERS 16u
#define ET_MLORA_MAX_M 64u
#define ET_MLORA_MAX_IN_DIM 64u
#define ET_MLORA_MAX_OUT_DIM 64u
#define ET_MLORA_MAX_RANK 16u

#define ET_MLORA_BASE_ONLY_ID (-1)

#define ET_MLORA_SHA256_BYTES 32u

/* Versioned non-executable Multi-Adapter Bundle format (ETML1000 v1.0). */
#define ET_MLORA_BUNDLE_MAGIC "ETML1000"
#define ET_MLORA_BUNDLE_MAGIC_BYTES 8u
#define ET_MLORA_BUNDLE_VERSION_MAJOR 1u
#define ET_MLORA_BUNDLE_VERSION_MINOR 0u
#define ET_MLORA_BUNDLE_HEADER_BYTES 72u
#define ET_MLORA_BUNDLE_ADAPTER_HEADER_BYTES 24u
#define ET_MLORA_BUNDLE_CHECKSUM_BYTES 32u
#define ET_MLORA_BUNDLE_MIN_BYTES \
  (ET_MLORA_BUNDLE_HEADER_BYTES + ET_MLORA_BUNDLE_CHECKSUM_BYTES)

#define ET_MLORA_BUNDLE_FLAG_HAS_BIAS 1u
#define ET_MLORA_BUNDLE_FLAG_HAS_ADAMW 2u

#define ET_MLORA_BUNDLE_MAX_ADAPTER_FLOATS \
  (3u * ((ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM) + \
         (ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK)))
#define ET_MLORA_BUNDLE_MAX_ADAPTER_BYTES \
  (ET_MLORA_BUNDLE_ADAPTER_HEADER_BYTES + \
   (ET_MLORA_BUNDLE_MAX_ADAPTER_FLOATS * 4u))
#define ET_MLORA_BUNDLE_MAX_BYTES \
  (ET_MLORA_BUNDLE_HEADER_BYTES + \
   (ET_MLORA_MAX_ADAPTERS * ET_MLORA_BUNDLE_MAX_ADAPTER_BYTES) + \
   ET_MLORA_BUNDLE_CHECKSUM_BYTES)

typedef int32_t et_mlora_status;

enum {
  ET_MLORA_OK = 0,
  ET_MLORA_ERR_ARGUMENT = -1,
  ET_MLORA_ERR_SHAPE = -2,
  ET_MLORA_ERR_ALIGNMENT = -3,
  ET_MLORA_ERR_ALIAS = -4,
  ET_MLORA_ERR_NONFINITE = -5,
  ET_MLORA_ERR_FP_ENV = -6,
  ET_MLORA_ERR_FORMAT = -7,
  ET_MLORA_ERR_VERSION = -8,
  ET_MLORA_ERR_TRUNCATED = -9,
  ET_MLORA_ERR_CHECKSUM = -10,
  ET_MLORA_ERR_BASE_HASH = -11,
  ET_MLORA_ERR_SYMLINK = -12,
  ET_MLORA_ERR_IO = -13,
  ET_MLORA_ERR_NOMEM = -14,
  ET_MLORA_ERR_OVERFLOW = -15,
  ET_MLORA_ERR_NOT_FOUND = -16,
  ET_MLORA_ERR_FULL = -17,
  ET_MLORA_ERR_BUSY = -18,
  ET_MLORA_ERR_ALREADY_EXISTS = -19,
  ET_MLORA_ERR_UNDERFLOW = -20
};

/* Standard-layout registry configuration.
 * Constraints:
 *   1 <= in_dim       <= ET_MLORA_MAX_IN_DIM (64)
 *   1 <= out_dim      <= ET_MLORA_MAX_OUT_DIM (64)
 *   1 <= max_adapters <= ET_MLORA_MAX_ADAPTERS (16)
 *   1 <= max_rank     <= min(ET_MLORA_MAX_RANK (16), in_dim, out_dim)
 *   has_bias          in {0, 1}
 *   reserved          == 0
 */
typedef struct et_mlora_registry_config {
  uint32_t in_dim;
  uint32_t out_dim;
  uint32_t max_adapters;
  uint32_t max_rank;
  uint32_t has_bias;
  uint32_t reserved;
} et_mlora_registry_config;

/* Metadata for one registered heterogeneous-rank adapter slot. */
typedef struct et_mlora_adapter_info {
  int32_t adapter_id;
  uint32_t slot_index;
  uint32_t generation;
  uint32_t rank;
  float alpha;
  uint32_t pin_count;
  uint32_t active;
  uint32_t adamw_step;
} et_mlora_adapter_info;

/* Per-adapter AdamW hyperparameters (all finite; lr > 0, eps > 0,
 * 0 <= beta1 < 1, 0 <= beta2 < 1, weight_decay >= 0). */
typedef struct et_mlora_adamw_config {
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
} et_mlora_adamw_config;

/* Observability stats for one heterogeneous SGMV forward/backward dispatch. */
typedef struct et_mlora_sgmv_stats {
  uint32_t batch_rows;
  uint32_t base_only_rows;
  uint32_t adapted_rows;
  uint32_t active_segments;
} et_mlora_sgmv_stats;

/* Metadata decoded from an ETML1000 v1.0 multi-adapter bundle. */
typedef struct et_mlora_bundle_meta {
  uint32_t version_major;
  uint32_t version_minor;
  uint32_t in_dim;
  uint32_t out_dim;
  uint32_t max_rank;
  uint32_t adapter_count;
  uint32_t flags;
  uint32_t total_bytes;
  uint8_t base_sha256[ET_MLORA_SHA256_BYTES];
  uint8_t bundle_sha256[ET_MLORA_SHA256_BYTES];
} et_mlora_bundle_meta;

typedef struct et_mlora_registry et_mlora_registry;

#if defined(__cplusplus)
#define ET_MLORA_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_MLORA_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_MLORA_STATIC_ASSERT(sizeof(et_mlora_registry_config) == 24u &&
                           offsetof(et_mlora_registry_config, reserved) == 20u,
                       "et_mlora_registry_config ABI layout changed");
ET_MLORA_STATIC_ASSERT(sizeof(et_mlora_adapter_info) == 32u &&
                           offsetof(et_mlora_adapter_info, alpha) == 16u &&
                           offsetof(et_mlora_adapter_info, adamw_step) == 28u,
                       "et_mlora_adapter_info ABI layout changed");
ET_MLORA_STATIC_ASSERT(sizeof(et_mlora_adamw_config) == 20u &&
                           offsetof(et_mlora_adamw_config, weight_decay) == 16u,
                       "et_mlora_adamw_config ABI layout changed");
ET_MLORA_STATIC_ASSERT(sizeof(et_mlora_sgmv_stats) == 16u &&
                           offsetof(et_mlora_sgmv_stats, active_segments) == 12u,
                       "et_mlora_sgmv_stats ABI layout changed");
ET_MLORA_STATIC_ASSERT(sizeof(et_mlora_bundle_meta) == 96u &&
                           offsetof(et_mlora_bundle_meta, base_sha256) == 32u &&
                           offsetof(et_mlora_bundle_meta, bundle_sha256) == 64u,
                       "et_mlora_bundle_meta ABI layout changed");
#undef ET_MLORA_STATIC_ASSERT

/* ---- Registry Lifecycle & Configuration ---- */

et_mlora_status et_mlora_validate_registry_config_v1(
    const et_mlora_registry_config *cfg);

/* Create a fixed-slot Multi-LoRA registry bound to frozen base weight
 * W_base[out_dim, in_dim] and optional bias[out_dim] (required iff
 * cfg->has_bias == 1; must be NULL iff cfg->has_bias == 0).
 * Allocates the fixed adapter pool once; hot SGMV kernels perform zero heap
 * allocations. */
et_mlora_status et_mlora_registry_create_v1(
    const et_mlora_registry_config *cfg, const float *w_base,
    const float *bias, et_mlora_registry **out_reg);

void et_mlora_registry_destroy_v1(et_mlora_registry *reg);

et_mlora_status et_mlora_registry_config_get_v1(
    const et_mlora_registry *reg, et_mlora_registry_config *out_cfg);

/* Recompute canonical SHA-256 over (in_dim, out_dim, has_bias, W_base, bias)
 * directly from the registry's frozen base buffers. */
et_mlora_status et_mlora_get_base_sha256_v1(
    const et_mlora_registry *reg, uint8_t out_digest[ET_MLORA_SHA256_BYTES]);

/* ---- Heterogeneous-Rank Adapter Pool Management (Zero-Alloc) ---- */

/* Register an adapter with explicit finite A[rank, in_dim] and
 * B[out_dim, rank] weights. adapter_id must be >= 0 and not currently active. */
et_mlora_status et_mlora_register_adapter_v1(et_mlora_registry *reg,
                                             int32_t adapter_id, uint32_t rank,
                                             float alpha, const float *a,
                                             const float *b,
                                             et_mlora_adapter_info *out_info);

/* Register an adapter with deterministic seeded A[rank, in_dim] and zero-filled
 * B[out_dim, rank] (+0.0f) so initial adapted output matches base-only bitwise. */
et_mlora_status et_mlora_register_seeded_adapter_v1(
    et_mlora_registry *reg, int32_t adapter_id, uint32_t rank, float alpha,
    uint64_t seed, et_mlora_adapter_info *out_info);

/* Unregister an active adapter. Rejects with ET_MLORA_ERR_BUSY if pinned
 * (pin_count > 0). Preserves slot generation counter so re-use advances it. */
et_mlora_status et_mlora_unregister_adapter_v1(et_mlora_registry *reg,
                                               int32_t adapter_id);

et_mlora_status et_mlora_pin_adapter_v1(et_mlora_registry *reg,
                                        int32_t adapter_id,
                                        uint32_t *out_pin_count);

/* Decrement pin_count. Rejects with ET_MLORA_ERR_UNDERFLOW if pin_count == 0. */
et_mlora_status et_mlora_unpin_adapter_v1(et_mlora_registry *reg,
                                          int32_t adapter_id,
                                          uint32_t *out_pin_count);

et_mlora_status et_mlora_get_adapter_info_v1(const et_mlora_registry *reg,
                                             int32_t adapter_id,
                                             et_mlora_adapter_info *out_info);

/* List all active adapters sorted in ascending adapter_id order. */
et_mlora_status et_mlora_list_adapters_v1(const et_mlora_registry *reg,
                                          et_mlora_adapter_info *out_infos,
                                          uint32_t capacity,
                                          uint32_t *out_count);

et_mlora_status et_mlora_get_adapter_weights_v1(const et_mlora_registry *reg,
                                                int32_t adapter_id,
                                                float *out_a, float *out_b);

et_mlora_status et_mlora_set_adapter_weights_v1(et_mlora_registry *reg,
                                                int32_t adapter_id,
                                                const float *a, const float *b);

et_mlora_status et_mlora_get_adapter_grads_v1(const et_mlora_registry *reg,
                                              int32_t adapter_id,
                                              float *out_da, float *out_db);

/* ---- Batched Heterogeneous SGMV Forward & Backward (Zero-Alloc) ---- */

/* Segmented Gather-Matrix-Multiply Forward (S-LoRA / Punica).
 * Inputs:
 *   x[m, in_dim]         finite row-major binary32
 *   row_adapter_ids[m]   each element is ET_MLORA_BASE_ONLY_ID (-1) or an
 *                        active registered adapter_id (arbitrary unsorted order)
 * Outputs:
 *   y[m, out_dim]        row-major binary32, failure-atomic staged commit
 *   out_stats            optional dispatch summary (may be NULL)
 *
 * Computes in IEEE-754 binary64 accumulation under FE_TONEAREST:
 *   Y[i, o] = sum_k W_base[o, k] * X[i, k] + bias[o]
 *           + (alpha_a / r_a) * sum_r B_a[o, r] * (sum_k A_a[r, k] * X[i, k])
 * Guaranteed bitwise identical (0.000e+00) to single-adapter per-row execution. */
et_mlora_status et_mlora_sgmv_forward_v1(const et_mlora_registry *reg,
                                         uint32_t m, const float *x,
                                         const int32_t *row_adapter_ids,
                                         float *y,
                                         et_mlora_sgmv_stats *out_stats);

/* Single-adapter (or base-only when adapter_id == -1) reference forward. */
et_mlora_status et_mlora_single_adapter_forward_v1(
    const et_mlora_registry *reg, int32_t adapter_id, uint32_t m,
    const float *x, float *y);

/* Multi-Adapter Analytic Backward.
 * Inputs:
 *   x[m, in_dim]         finite row-major binary32
 *   row_adapter_ids[m]   -1 or active adapter_id (arbitrary unsorted order)
 *   dy[m, out_dim]       finite upstream gradient
 * Outputs:
 *   dx[m, in_dim]        input gradient (failure-atomic staged commit)
 *   per-adapter dA_a[r_a, in_dim] and dB_a[out_dim, r_a] stored inside registry:
 *     - active adapters present in row_adapter_ids receive the exact sum over
 *       ONLY their own rows (preserving ascending row order),
 *     - registered adapters NOT present in row_adapter_ids receive exact +0.0f
 *       (zero cross-adapter gradient bleed),
 *     - frozen W_base and bias remain strictly unchanged. */
et_mlora_status et_mlora_sgmv_backward_v1(et_mlora_registry *reg, uint32_t m,
                                          const float *x,
                                          const int32_t *row_adapter_ids,
                                          const float *dy, float *dx,
                                          et_mlora_sgmv_stats *out_stats);

/* ---- Per-Adapter AdamW Update (Zero-Alloc) ---- */

et_mlora_status et_mlora_validate_adamw_config_v1(
    const et_mlora_adamw_config *cfg);

/* Apply one decoupled AdamW step to a single registered adapter using its
 * stored (dA_a, dB_a) gradients and persistent (m_A, v_A, m_B, v_B, step).
 * Failure-atomic: non-finite updates abort without mutating adapter state. */
et_mlora_status et_mlora_adamw_step_adapter_v1(
    et_mlora_registry *reg, int32_t adapter_id,
    const et_mlora_adamw_config *cfg);

/* ---- Versioned SHA-256 Multi-Adapter Bundle (ETML1000 v1.0) ---- */

et_mlora_status et_mlora_bundle_size_v1(const et_mlora_registry *reg,
                                        uint32_t include_adamw,
                                        size_t *out_bytes);

et_mlora_status et_mlora_bundle_serialize_v1(const et_mlora_registry *reg,
                                             uint32_t include_adamw,
                                             uint8_t *out_bytes,
                                             size_t capacity,
                                             size_t *out_size);

/* Failure-atomic parse into reg. Rejects with ET_MLORA_ERR_BUSY if any adapter
 * in reg is currently pinned. Verifies magic, version, size/truncation,
 * SHA-256 integrity, base SHA-256 binding, shape/bias match, and finiteness
 * before committing all adapters atomically. */
et_mlora_status et_mlora_bundle_parse_v1(const uint8_t *bytes, size_t size,
                                         et_mlora_registry *reg,
                                         et_mlora_bundle_meta *out_meta);

/* Atomic 0600 temp + fsync + rename save. */
et_mlora_status et_mlora_bundle_save_v1(const char *path,
                                        const et_mlora_registry *reg,
                                        uint32_t include_adamw);

/* O_NOFOLLOW checked load (rejects symlinks with ET_MLORA_ERR_SYMLINK). */
et_mlora_status et_mlora_bundle_load_v1(const char *path,
                                        et_mlora_registry *reg,
                                        et_mlora_bundle_meta *out_meta);

#ifdef __cplusplus
}
#endif

#endif
