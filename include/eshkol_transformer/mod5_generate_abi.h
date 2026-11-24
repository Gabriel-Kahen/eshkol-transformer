#ifndef ESHKOL_TRANSFORMER_MOD5_GENERATE_ABI_H
#define ESHKOL_TRANSFORMER_MOD5_GENERATE_ABI_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_MOD5_GEN_API __attribute__((visibility("default")))
#else
#define ET_MOD5_GEN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MOD5_GEN_ABI_MAJOR 1u
#define ET_MOD5_GEN_ABI_MINOR 0u

#define ET_MOD5_GEN_MAX_SEQ 64u
#define ET_MOD5_GEN_MAX_DIM 32u
#define ET_MOD5_GEN_MAX_FFN 64u
#define ET_MOD5_GEN_MAX_LAYERS 8u
#define ET_MOD5_GEN_MAX_HEADS 16u
#define ET_MOD5_GEN_MAX_VOCAB 1024u

typedef int32_t et_mod5_gen_status;

enum {
  ET_MOD5_GEN_OK = 0,
  ET_MOD5_GEN_ERR_ARGUMENT = -1,
  ET_MOD5_GEN_ERR_CONFIG = -2,
  ET_MOD5_GEN_ERR_BOUNDS = -3,
  ET_MOD5_GEN_ERR_OVERFLOW = -4,
  ET_MOD5_GEN_ERR_NONFINITE = -5,
  ET_MOD5_GEN_ERR_CACHE_FULL = -6,
  ET_MOD5_GEN_ERR_NOMEM = -7,
  ET_MOD5_GEN_ERR_PROVIDER = -8,
  ET_MOD5_GEN_ERR_FP_ENV = -9,
  ET_MOD5_GEN_ERR_STATE = -10,
  ET_MOD5_GEN_ERR_UNALIGNED = -11
};

enum {
  ET_MOD5_GEN_SAMPLE_GREEDY = 0,
  ET_MOD5_GEN_SAMPLE_CATEGORICAL = 1
};

/* Summary indices written by et_mod5_gen_aot_generate. */
enum {
  ET_MOD5_GEN_AOT_SUMMARY_GENERATED_LEN = 0,
  ET_MOD5_GEN_AOT_SUMMARY_STOPPED_EOS = 1,
  ET_MOD5_GEN_AOT_SUMMARY_RNG_KEY = 2,
  ET_MOD5_GEN_AOT_SUMMARY_RNG_COUNTER = 3,
  ET_MOD5_GEN_AOT_SUMMARY_CACHE_LEN = 4,
  ET_MOD5_GEN_AOT_SUMMARY_LEN = 5
};

typedef struct et_mod5_gen_engine et_mod5_gen_engine;
typedef struct et_mod5_gen_kv_cache et_mod5_gen_kv_cache;

/*
 * MOD5 autoregressive decoder configuration.
 * - vocab_size:   1 .. ET_MOD5_GEN_MAX_VOCAB (1024)
 * - max_seq:      1 .. ET_MOD5_GEN_MAX_SEQ (64)
 * - num_layers:   1 .. ET_MOD5_GEN_MAX_LAYERS (8)
 * - hidden_dim:   1 .. ET_MOD5_GEN_MAX_DIM (32), divisible by num_heads
 * - ffn_dim:      1 .. ET_MOD5_GEN_MAX_FFN (64)
 * - num_heads:    1 .. ET_MOD5_GEN_MAX_HEADS (16)
 * - num_kv_heads: 1 .. num_heads, dividing num_heads (MQA=1, GQA=2.., MHA=Hq)
 * - window_size:  0 = full causal; >0 = sliding window of exact width
 * - use_alibi:    0 = disabled; 1 = enabled (alibi_slopes[0..num_heads-1] finite)
 * - rms_epsilon:  strictly positive finite binary32
 */
typedef struct {
  uint32_t vocab_size;
  uint32_t max_seq;
  uint32_t num_layers;
  uint32_t hidden_dim;
  uint32_t ffn_dim;
  uint32_t num_heads;
  uint32_t num_kv_heads;
  uint32_t window_size;
  uint32_t use_alibi;
  float rms_epsilon;
  float alibi_slopes[ET_MOD5_GEN_MAX_HEADS];
} et_mod5_gen_config;

/*
 * Per-layer borrowed weights in canonical MOD5 row-major [out, in] order:
 *   norm1: [D]
 *   wq:    [D, D]
 *   wk:    [Hkv * Dh, D]
 *   wv:    [Hkv * Dh, D]
 *   wo:    [D, D]
 *   norm2: [D]
 *   wgate: [F, D]
 *   wup:   [F, D]
 *   wdown: [D, F]
 */
typedef struct {
  const float *norm1;
  const float *wq;
  const float *wk;
  const float *wv;
  const float *wo;
  const float *norm2;
  const float *wgate;
  const float *wup;
  const float *wdown;
} et_mod5_gen_layer_weights;

/*
 * Full model borrowed weights:
 *   token_embedding: [V, D]
 *   norm_final:      [D]
 *   head:            [V, D] (may alias token_embedding for tied weights)
 *   layers:          [L]
 */
typedef struct {
  const float *token_embedding;
  const float *norm_final;
  const float *head;
  const et_mod5_gen_layer_weights *layers;
} et_mod5_gen_model_weights;

/*
 * Explicit observable deterministic counter-based SplitMix64 RNG state.
 * Counter UINT64_MAX is the exhausted sentinel and refuses further draws.
 */
typedef struct {
  uint64_t key;
  uint64_t counter;
} et_mod5_gen_rng;

/*
 * Sampling configuration.
 * - mode:           ET_MOD5_GEN_SAMPLE_GREEDY or ET_MOD5_GEN_SAMPLE_CATEGORICAL
 * - temperature:    finite > 0.0f for categorical
 * - top_k:          0 = disabled (all V); 1..V = keep top k
 * - top_p:          finite in (0.0f, 1.0f] for categorical (1.0f = disabled)
 * - eos_token_id:   -1 = disabled; 0..V-1 = stop token ID
 * - max_new_tokens: number of tokens to generate (0..max_seq)
 */
typedef struct {
  uint32_t mode;
  float temperature;
  uint32_t top_k;
  float top_p;
  int64_t eos_token_id;
  uint32_t max_new_tokens;
} et_mod5_gen_sampling_config;

typedef struct {
  uint32_t prompt_len;
  uint32_t generated_len;
  uint32_t total_len;
  uint32_t stopped_by_eos;
  et_mod5_gen_rng final_rng;
} et_mod5_gen_result;

/* Counter-based SplitMix64 RNG primitives. */
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_rng_init(et_mod5_gen_rng *rng,
                                                        uint64_t seed);
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_rng_next_u64(
    et_mod5_gen_rng *rng, uint64_t *out_value);
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_rng_next_f64(
    et_mod5_gen_rng *rng, double *out_unit);

/* Validate configuration without allocation. */
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_validate_config(
    const et_mod5_gen_config *config);

/* Verify internal serial f32 kernels match et_mod5_kernel_provider_v1(). */
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_verify_provider(
    double *out_max_abs_diff);

/* Engine lifecycle and weight loading (copies and finite-checks all weights). */
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_engine_create(
    const et_mod5_gen_config *config, et_mod5_gen_engine **out_engine);
ET_MOD5_GEN_API void et_mod5_gen_engine_destroy(et_mod5_gen_engine *engine);
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_engine_load_weights(
    et_mod5_gen_engine *engine, const et_mod5_gen_model_weights *weights);
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_engine_set_global_weights(
    et_mod5_gen_engine *engine, const float *token_embedding,
    const float *norm_final, const float *head);
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_engine_set_layer_weights(
    et_mod5_gen_engine *engine, uint32_t layer_index, const float *norm1,
    const float *wq, const float *wk, const float *wv, const float *wo,
    const float *norm2, const float *wgate, const float *wup,
    const float *wdown);

/* KV-cache lifecycle and inspection. Physical layout: [L, Hkv, max_seq, Dh]. */
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_kv_cache_create(
    const et_mod5_gen_config *config, et_mod5_gen_kv_cache **out_cache);
ET_MOD5_GEN_API void et_mod5_gen_kv_cache_destroy(et_mod5_gen_kv_cache *cache);
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_kv_cache_reset(
    et_mod5_gen_kv_cache *cache);
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_kv_cache_len(
    const et_mod5_gen_kv_cache *cache, uint32_t *out_len);
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_kv_cache_capacity(
    const et_mod5_gen_kv_cache *cache, uint32_t *out_max_seq);
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_kv_cache_read_slot(
    const et_mod5_gen_kv_cache *cache, uint32_t layer, uint32_t kv_head,
    uint32_t pos, float *k_out, float *v_out);

/* Prefill, incremental single-token decode, and full-context recompute. */
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_prefill(
    const et_mod5_gen_engine *engine, et_mod5_gen_kv_cache *cache,
    const int64_t *prompt_tokens, size_t prompt_len, float *out_last_logits,
    float *out_all_logits);
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_decode_step(
    const et_mod5_gen_engine *engine, et_mod5_gen_kv_cache *cache,
    int64_t token_id, float *out_logits);
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_recompute_forward(
    const et_mod5_gen_engine *engine, const int64_t *tokens, size_t seq_len,
    float *out_all_logits);

/* Single-step deterministic sampling (greedy or temperature/top-k/top-p). */
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_sample(
    const float *logits, size_t vocab_size,
    const et_mod5_gen_sampling_config *sampling, et_mod5_gen_rng *rng,
    int64_t *out_token);

/* Autoregressive prefill + decode + sampling loop with EOS early stop. */
ET_MOD5_GEN_API et_mod5_gen_status et_mod5_gen_generate(
    const et_mod5_gen_engine *engine, et_mod5_gen_kv_cache *cache,
    const int64_t *prompt_tokens, size_t prompt_len,
    const et_mod5_gen_sampling_config *sampling, et_mod5_gen_rng *rng,
    int64_t *out_generated_tokens, size_t out_token_capacity,
    float *out_step_logits, size_t out_logits_capacity,
    et_mod5_gen_result *out_result);

/*
 * Flat-scalar AOT bridge layer callable directly from compiled Eshkol AOT
 * without C struct marshaling.
 */
ET_MOD5_GEN_API int32_t et_mod5_gen_aot_engine_create(
    int64_t vocab_size, int64_t max_seq, int64_t num_layers,
    int64_t hidden_dim, int64_t ffn_dim, int64_t num_heads,
    int64_t num_kv_heads, int64_t window_size, int64_t use_alibi,
    double rms_epsilon, const float *alibi_slopes,
    et_mod5_gen_engine **out_engine);
ET_MOD5_GEN_API void et_mod5_gen_aot_engine_destroy(et_mod5_gen_engine *engine);
ET_MOD5_GEN_API int32_t et_mod5_gen_aot_set_global_weights(
    et_mod5_gen_engine *engine, const float *token_embedding,
    const float *norm_final, const float *head);
ET_MOD5_GEN_API int32_t et_mod5_gen_aot_set_layer_weights(
    et_mod5_gen_engine *engine, int64_t layer_index, const float *norm1,
    const float *wq, const float *wk, const float *wv, const float *wo,
    const float *norm2, const float *wgate, const float *wup,
    const float *wdown);
ET_MOD5_GEN_API int32_t et_mod5_gen_aot_kv_cache_create(
    const et_mod5_gen_engine *engine, et_mod5_gen_kv_cache **out_cache);
ET_MOD5_GEN_API void et_mod5_gen_aot_kv_cache_destroy(
    et_mod5_gen_kv_cache *cache);
ET_MOD5_GEN_API int32_t et_mod5_gen_aot_kv_cache_reset(
    et_mod5_gen_kv_cache *cache);
ET_MOD5_GEN_API int64_t et_mod5_gen_aot_kv_cache_len(
    const et_mod5_gen_kv_cache *cache);
ET_MOD5_GEN_API int32_t et_mod5_gen_aot_prefill(
    const et_mod5_gen_engine *engine, et_mod5_gen_kv_cache *cache,
    const int64_t *prompt_tokens, int64_t prompt_len, float *out_last_logits,
    float *out_all_logits);
ET_MOD5_GEN_API int32_t et_mod5_gen_aot_decode_step(
    const et_mod5_gen_engine *engine, et_mod5_gen_kv_cache *cache,
    int64_t token_id, float *out_logits);
ET_MOD5_GEN_API int32_t et_mod5_gen_aot_recompute_forward(
    const et_mod5_gen_engine *engine, const int64_t *tokens, int64_t seq_len,
    float *out_all_logits);
ET_MOD5_GEN_API int32_t et_mod5_gen_aot_generate(
    const et_mod5_gen_engine *engine, et_mod5_gen_kv_cache *cache,
    const int64_t *prompt_tokens, int64_t prompt_len, int64_t mode,
    double temperature, int64_t top_k, double top_p, int64_t eos_token_id,
    int64_t max_new_tokens, int64_t rng_seed, int64_t *out_tokens,
    float *out_step_logits, int64_t *out_summary);
ET_MOD5_GEN_API int32_t et_mod5_gen_aot_verify_provider(
    double *out_max_abs_diff);
ET_MOD5_GEN_API int32_t et_mod5_gen_aot_max_abs_diff_f32(
    const float *a, const float *b, int64_t count, double *out_max_diff);
ET_MOD5_GEN_API double et_mod5_gen_aot_load_f64(const double *buffer,
                                                int64_t index);

#ifdef __cplusplus
}
#endif

#endif
