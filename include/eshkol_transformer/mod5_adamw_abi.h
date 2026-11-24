#ifndef ESHKOL_TRANSFORMER_MOD5_ADAMW_ABI_H
#define ESHKOL_TRANSFORMER_MOD5_ADAMW_ABI_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_MOD5_ADAMW_API __attribute__((visibility("default")))
#else
#define ET_MOD5_ADAMW_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MOD5_ADAMW_ABI_MAJOR 1u
#define ET_MOD5_ADAMW_ABI_MINOR 0u

/* Versioned, SHA-256 checksummed, non-executable binary training-state format. */
#define ET_MOD5_ADAMW_SNAPSHOT_MAGIC "ETM5TRN1"
#define ET_MOD5_ADAMW_SNAPSHOT_MAGIC_BYTES 8u
#define ET_MOD5_ADAMW_SNAPSHOT_VERSION_MAJOR 1u
#define ET_MOD5_ADAMW_SNAPSHOT_VERSION_MINOR 0u
#define ET_MOD5_ADAMW_SNAPSHOT_HEADER_BYTES 224u
#define ET_MOD5_ADAMW_SNAPSHOT_CHECKSUM_BYTES 32u
#define ET_MOD5_ADAMW_MAX_TOKENIZER_ID_BYTES 128u
#define ET_MOD5_ADAMW_MAX_PARAMS UINT64_C(16777216)

/* Explicit status codes. Zero is success; negative values report exact errors. */
typedef int32_t et_mod5_adamw_status;
enum {
  ET_MOD5_ADAMW_OK = 0,
  ET_MOD5_ADAMW_ERR_INVALID_ARG = -1,
  ET_MOD5_ADAMW_ERR_NONFINITE = -2,
  ET_MOD5_ADAMW_ERR_BAD_MAGIC = -3,
  ET_MOD5_ADAMW_ERR_VERSION_MISMATCH = -4,
  ET_MOD5_ADAMW_ERR_CHECKSUM_MISMATCH = -5,
  ET_MOD5_ADAMW_ERR_TRUNCATED = -6,
  ET_MOD5_ADAMW_ERR_CONFIG_MISMATCH = -7,
  ET_MOD5_ADAMW_ERR_IO = -8,
  ET_MOD5_ADAMW_ERR_ALLOC = -9,
  ET_MOD5_ADAMW_ERR_DETERMINISM = -10,
  ET_MOD5_ADAMW_ERR_OVERFLOW = -11
};

/* Deterministic learning-rate schedules. */
typedef uint32_t et_mod5_lr_schedule_kind;
enum {
  ET_MOD5_LR_CONSTANT = 0u,
  ET_MOD5_LR_LINEAR_WARMUP_LINEAR_DECAY = 1u,
  ET_MOD5_LR_LINEAR_WARMUP_COSINE_DECAY = 2u
};

/* Standard-layout scheduler configuration.
 * - base_lr: finite, >= 0.0
 * - min_lr: finite, >= 0.0, <= base_lr
 * - For warmup/decay schedules: total_steps > 0 and warmup_steps <= total_steps.
 */
typedef struct et_mod5_lr_schedule_config_v1 {
  uint32_t kind;
  uint32_t reserved;
  double base_lr;
  double min_lr;
  uint64_t warmup_steps;
  uint64_t total_steps;
} et_mod5_lr_schedule_config_v1;

/* Deterministic dataset cursor persisted across fresh-process resume. */
typedef struct et_mod5_dataset_cursor_v1 {
  uint64_t shard_index;
  uint64_t token_offset;
  uint64_t epoch;
} et_mod5_dataset_cursor_v1;

/* Complete AdamW + scheduler + reproducibility identity configuration.
 * - param_count: 1 .. ET_MOD5_ADAMW_MAX_PARAMS
 * - beta1, beta2: finite in [0.0f, 1.0f)
 * - eps: finite, > 0.0f
 * - weight_decay: finite, >= 0.0f
 * - max_grad_norm: finite, >= 0.0f (0.0f disables clipping)
 * - tokenizer_id: non-empty NUL-terminated UTF-8 string (1..128 bytes)
 * - model_config_hash: caller-supplied 32-byte SHA-256 of canonical model config
 */
typedef struct et_mod5_adamw_config_v1 {
  uint64_t param_count;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float max_grad_norm;
  uint32_t reserved;
  et_mod5_lr_schedule_config_v1 schedule;
  uint8_t model_config_hash[32];
  const char *tokenizer_id;
} et_mod5_adamw_config_v1;

/* Per-step telemetry returned by et_mod5_adamw_step. */
typedef struct et_mod5_adamw_step_info_v1 {
  uint64_t step;
  double lr;
  double grad_norm;
  double clip_scale;
  uint32_t clipped;
  uint32_t reserved;
} et_mod5_adamw_step_info_v1;

/* Opaque live training-state owner (params, m, v, step, schedule, RNG, cursor,
 * tokenizer identity, config hashes, cumulative token count). */
typedef struct et_mod5_adamw_state et_mod5_adamw_state;

/* --- Core C11 / C++17 API --- */

ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_compute_config_hash(const et_mod5_adamw_config_v1 *cfg,
                                  uint8_t out_hash[32]);

ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_compute_lr(const et_mod5_lr_schedule_config_v1 *sched,
                         uint64_t step, double *out_lr);

ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_create(const et_mod5_adamw_config_v1 *cfg,
                     et_mod5_adamw_state **out_state);

ET_MOD5_ADAMW_API void et_mod5_adamw_destroy(et_mod5_adamw_state *state);

/* Copies finite binary32 parameters into live state; atomic on error. */
ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_set_params(et_mod5_adamw_state *state, const float *params,
                         size_t count);

ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_get_params(const et_mod5_adamw_state *state, float *out_params,
                         size_t count);

ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_get_moments(const et_mod5_adamw_state *state, float *out_m,
                          float *out_v, size_t count);

ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_set_cursor(et_mod5_adamw_state *state,
                         const et_mod5_dataset_cursor_v1 *cursor);

ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_get_cursor(const et_mod5_adamw_state *state,
                         et_mod5_dataset_cursor_v1 *out_cursor);

ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_set_rng(et_mod5_adamw_state *state, const uint64_t rng_state[4]);

ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_get_rng(const et_mod5_adamw_state *state,
                      uint64_t out_rng_state[4]);

/* Deterministic xoshiro256** step updating live RNG state. */
ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_rng_next_u64(et_mod5_adamw_state *state, uint64_t *out_value);

ET_MOD5_ADAMW_API uint64_t
et_mod5_adamw_get_step(const et_mod5_adamw_state *state);

ET_MOD5_ADAMW_API uint64_t
et_mod5_adamw_get_cumulative_tokens(const et_mod5_adamw_state *state);

/* Atomic CPU-f32 AdamW update with binary64 global L2 gradient-norm clipping,
 * exact bias correction 1 - beta1^t and 1 - beta2^t, and decoupled weight decay.
 * Any non-finite input, intermediate, or output leaves live state untouched.
 */
ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_step(et_mod5_adamw_state *state, const float *grads,
                   size_t grad_count, uint64_t step_tokens,
                   et_mod5_adamw_step_info_v1 *out_info);

/* Atomic fsync + rename non-executable (0600) snapshot save. Refuses symlink or
 * non-regular existing destinations. */
ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_save_snapshot(const et_mod5_adamw_state *state, const char *path);

/* O_NOFOLLOW checked read rejecting symlinks, non-regular files, bad magic,
 * version mismatch, truncation/trailing bytes, corrupt SHA-256, config mismatch,
 * and non-finite or negative second moments. Atomic: live state is untouched on
 * any error. */
ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_load_snapshot(et_mod5_adamw_state *state, const char *path);

ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_sha256_bytes(const uint8_t *data, size_t len,
                           uint8_t out_digest[32]);

ET_MOD5_ADAMW_API et_mod5_adamw_status
et_mod5_adamw_sha256_file(const char *path, uint8_t out_digest[32]);

/* --- Compiled Eshkol AOT Bridge API --- */

ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_create(
    int64_t param_count, double beta1, double beta2, double eps,
    double weight_decay, double max_grad_norm, int32_t sched_kind,
    double base_lr, double min_lr, int64_t warmup_steps, int64_t total_steps,
    const char *tokenizer_id, const char *model_config_tag,
    et_mod5_adamw_state **out_state);

ET_MOD5_ADAMW_API void et_mod5_adamw_aot_destroy(et_mod5_adamw_state *state);

ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_set_param_f64(
    et_mod5_adamw_state *state, int64_t index, double value);

ET_MOD5_ADAMW_API double et_mod5_adamw_aot_get_param_f64(
    const et_mod5_adamw_state *state, int64_t index);

ET_MOD5_ADAMW_API int64_t et_mod5_adamw_aot_get_param_bits(
    const et_mod5_adamw_state *state, int64_t index);

ET_MOD5_ADAMW_API int64_t et_mod5_adamw_aot_get_m_bits(
    const et_mod5_adamw_state *state, int64_t index);

ET_MOD5_ADAMW_API int64_t et_mod5_adamw_aot_get_v_bits(
    const et_mod5_adamw_state *state, int64_t index);

ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_step(
    et_mod5_adamw_state *state, const float *grads, int64_t grad_count,
    int64_t step_tokens, double *out_lr, double *out_grad_norm,
    int32_t *out_clipped);

ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_save(
    const et_mod5_adamw_state *state, const char *path);

ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_load(et_mod5_adamw_state *state,
                                                 const char *path);

ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_set_cursor(
    et_mod5_adamw_state *state, int64_t shard_index, int64_t token_offset,
    int64_t epoch);

ET_MOD5_ADAMW_API int64_t et_mod5_adamw_aot_get_cursor_shard(
    const et_mod5_adamw_state *state);

ET_MOD5_ADAMW_API int64_t et_mod5_adamw_aot_get_cursor_offset(
    const et_mod5_adamw_state *state);

ET_MOD5_ADAMW_API int64_t et_mod5_adamw_aot_get_cursor_epoch(
    const et_mod5_adamw_state *state);

ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_set_rng(et_mod5_adamw_state *state,
                                                    int64_t s0, int64_t s1,
                                                    int64_t s2, int64_t s3);

ET_MOD5_ADAMW_API int64_t et_mod5_adamw_aot_get_rng_word(
    const et_mod5_adamw_state *state, int32_t lane);

ET_MOD5_ADAMW_API int64_t et_mod5_adamw_aot_rng_next(
    et_mod5_adamw_state *state);

ET_MOD5_ADAMW_API int64_t et_mod5_adamw_aot_get_step(
    const et_mod5_adamw_state *state);

ET_MOD5_ADAMW_API int64_t et_mod5_adamw_aot_get_tokens(
    const et_mod5_adamw_state *state);

ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_compute_lr(
    int32_t sched_kind, double base_lr, double min_lr, int64_t warmup_steps,
    int64_t total_steps, int64_t step, double *out_lr);

ET_MOD5_ADAMW_API double et_mod5_adamw_aot_load_f64(const double *cell);

/* Writes 64 lowercase hex characters + NUL (requires capacity >= 65). */
ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_state_sha256_hex(
    const et_mod5_adamw_state *state, char *out_hex, int64_t capacity);

ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_file_sha256_hex(
    const char *path, char *out_hex, int64_t capacity);

/* Validated D1 v1.0 .ets token shard atomic write and O_NOFOLLOW checked read. */
ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_write_d1_shard(
    const char *path, int64_t shard_index, int64_t vocab_size,
    const char *tokenizer_id, const int64_t *tokens, int64_t token_count);

ET_MOD5_ADAMW_API int32_t et_mod5_adamw_aot_read_d1_shard(
    const char *path, int64_t expected_shard_index, int64_t expected_vocab_size,
    const char *expected_tokenizer_id, int64_t *out_tokens, int64_t max_tokens,
    int64_t *out_token_count);

#ifdef __cplusplus
}
#endif

#endif
