#include "eshkol_transformer/mod5_generate_abi.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL (%s:%d): %s\n", __FILE__, __LINE__, (msg));        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

typedef struct {
  et_mod5_gen_config cfg;
  float *emb;
  float *nf;
  float *head;
  float *norm1[ET_MOD5_GEN_MAX_LAYERS];
  float *wq[ET_MOD5_GEN_MAX_LAYERS];
  float *wk[ET_MOD5_GEN_MAX_LAYERS];
  float *wv[ET_MOD5_GEN_MAX_LAYERS];
  float *wo[ET_MOD5_GEN_MAX_LAYERS];
  float *norm2[ET_MOD5_GEN_MAX_LAYERS];
  float *wgate[ET_MOD5_GEN_MAX_LAYERS];
  float *wup[ET_MOD5_GEN_MAX_LAYERS];
  float *wdown[ET_MOD5_GEN_MAX_LAYERS];
  et_mod5_gen_layer_weights layers[ET_MOD5_GEN_MAX_LAYERS];
  et_mod5_gen_model_weights weights;
} synth_model;

static void fill_det_f32(float *buf, size_t n, uint64_t tag, float base,
                         float scale) {
  for (size_t i = 0u; i < n; ++i) {
    const uint64_t h = (tag * UINT64_C(2654435761) + (uint64_t)i * 37u) % 41u;
    buf[i] = base + scale * (float)((int)h - 20);
  }
}

static void init_synth_model(synth_model *m, const et_mod5_gen_config *cfg,
                             uint64_t seed_tag) {
  memset(m, 0, sizeof(*m));
  m->cfg = *cfg;
  const size_t V = cfg->vocab_size;
  const size_t D = cfg->hidden_dim;
  const size_t F = cfg->ffn_dim;
  const size_t L = cfg->num_layers;
  const size_t Dh = D / cfg->num_heads;
  const size_t KVD = cfg->num_kv_heads * Dh;

  m->emb = (float *)calloc(V * D, sizeof(float));
  m->nf = (float *)calloc(D, sizeof(float));
  m->head = (float *)calloc(V * D, sizeof(float));
  CHECK(m->emb && m->nf && m->head, "alloc global weights");
  fill_det_f32(m->emb, V * D, seed_tag + 1u, 0.0f, 0.03125f);
  fill_det_f32(m->nf, D, seed_tag + 2u, 1.0f, 0.015625f);
  fill_det_f32(m->head, V * D, seed_tag + 3u, 0.0f, 0.03125f);

  for (size_t l = 0u; l < L; ++l) {
    const uint64_t lt = seed_tag + 100u * (l + 1u);
    m->norm1[l] = (float *)calloc(D, sizeof(float));
    m->wq[l] = (float *)calloc(D * D, sizeof(float));
    m->wk[l] = (float *)calloc(KVD * D, sizeof(float));
    m->wv[l] = (float *)calloc(KVD * D, sizeof(float));
    m->wo[l] = (float *)calloc(D * D, sizeof(float));
    m->norm2[l] = (float *)calloc(D, sizeof(float));
    m->wgate[l] = (float *)calloc(F * D, sizeof(float));
    m->wup[l] = (float *)calloc(F * D, sizeof(float));
    m->wdown[l] = (float *)calloc(D * F, sizeof(float));
    CHECK(m->norm1[l] && m->wq[l] && m->wk[l] && m->wv[l] && m->wo[l] &&
              m->norm2[l] && m->wgate[l] && m->wup[l] && m->wdown[l],
          "alloc layer weights");
    fill_det_f32(m->norm1[l], D, lt + 1u, 1.0f, 0.015625f);
    fill_det_f32(m->wq[l], D * D, lt + 2u, 0.0f, 0.0234375f);
    fill_det_f32(m->wk[l], KVD * D, lt + 3u, 0.0f, 0.0234375f);
    fill_det_f32(m->wv[l], KVD * D, lt + 4u, 0.0f, 0.0234375f);
    fill_det_f32(m->wo[l], D * D, lt + 5u, 0.0f, 0.0234375f);
    fill_det_f32(m->norm2[l], D, lt + 6u, 1.0f, 0.015625f);
    fill_det_f32(m->wgate[l], F * D, lt + 7u, 0.0f, 0.0234375f);
    fill_det_f32(m->wup[l], F * D, lt + 8u, 0.0f, 0.0234375f);
    fill_det_f32(m->wdown[l], D * F, lt + 9u, 0.0f, 0.0234375f);

    m->layers[l].norm1 = m->norm1[l];
    m->layers[l].wq = m->wq[l];
    m->layers[l].wk = m->wk[l];
    m->layers[l].wv = m->wv[l];
    m->layers[l].wo = m->wo[l];
    m->layers[l].norm2 = m->norm2[l];
    m->layers[l].wgate = m->wgate[l];
    m->layers[l].wup = m->wup[l];
    m->layers[l].wdown = m->wdown[l];
  }
  m->weights.token_embedding = m->emb;
  m->weights.norm_final = m->nf;
  m->weights.head = m->head;
  m->weights.layers = m->layers;
}

static void free_synth_model(synth_model *m) {
  free(m->emb);
  free(m->nf);
  free(m->head);
  for (size_t l = 0u; l < m->cfg.num_layers; ++l) {
    free(m->norm1[l]);
    free(m->wq[l]);
    free(m->wk[l]);
    free(m->wv[l]);
    free(m->wo[l]);
    free(m->norm2[l]);
    free(m->wgate[l]);
    free(m->wup[l]);
    free(m->wdown[l]);
  }
}

static void test_provider_parity(void) {
  double max_diff = -1.0;
  CHECK(et_mod5_gen_verify_provider(&max_diff) == ET_MOD5_GEN_OK,
        "et_mod5_gen_verify_provider failed");
  CHECK(max_diff == 0.0, "provider parity expected exact 0.0 diff");
  printf("PARITY provider_k1_mod5: max_abs_diff=%.3e PASS\n", max_diff);
}

static void run_cached_vs_recompute_case(const char *label,
                                         const et_mod5_gen_config *cfg,
                                         size_t prefill_len, size_t total_len,
                                         uint64_t seed_tag) {
  synth_model m;
  init_synth_model(&m, cfg, seed_tag);

  et_mod5_gen_engine *eng = NULL;
  et_mod5_gen_kv_cache *cache = NULL;
  CHECK(et_mod5_gen_engine_create(cfg, &eng) == ET_MOD5_GEN_OK,
        "engine_create");
  CHECK(et_mod5_gen_engine_load_weights(eng, &m.weights) == ET_MOD5_GEN_OK,
        "load_weights");
  CHECK(et_mod5_gen_kv_cache_create(cfg, &cache) == ET_MOD5_GEN_OK,
        "kv_cache_create");

  int64_t seq[ET_MOD5_GEN_MAX_SEQ];
  for (size_t i = 0u; i < total_len; ++i)
    seq[i] = (int64_t)((i * 7u + seed_tag) % cfg->vocab_size);

  const size_t V = cfg->vocab_size;
  float *recomp_logits = (float *)calloc(total_len * V, sizeof(float));
  float *cached_logits = (float *)calloc(total_len * V, sizeof(float));
  float *last_logits = (float *)calloc(V, sizeof(float));
  CHECK(recomp_logits && cached_logits && last_logits, "alloc logit buffers");

  CHECK(et_mod5_gen_recompute_forward(eng, seq, total_len, recomp_logits) ==
            ET_MOD5_GEN_OK,
        "recompute_forward");

  CHECK(et_mod5_gen_prefill(eng, cache, seq, prefill_len, last_logits,
                            cached_logits) == ET_MOD5_GEN_OK,
        "prefill");

  uint32_t clen = 0u;
  uint32_t ccap = 0u;
  CHECK(et_mod5_gen_kv_cache_len(cache, &clen) == ET_MOD5_GEN_OK &&
            clen == prefill_len,
        "kv_cache_len after prefill");
  CHECK(et_mod5_gen_kv_cache_capacity(cache, &ccap) == ET_MOD5_GEN_OK &&
            ccap == cfg->max_seq,
        "kv_cache_capacity");

  for (size_t step = prefill_len; step < total_len; ++step) {
    CHECK(et_mod5_gen_decode_step(eng, cache, seq[step],
                                  cached_logits + step * V) == ET_MOD5_GEN_OK,
          "decode_step");
  }
  CHECK(et_mod5_gen_kv_cache_len(cache, &clen) == ET_MOD5_GEN_OK &&
            clen == total_len,
        "kv_cache_len after decode");

  double max_diff = 0.0;
  CHECK(et_mod5_gen_aot_max_abs_diff_f32(cached_logits, recomp_logits,
                                         (int64_t)(total_len * V),
                                         &max_diff) == ET_MOD5_GEN_OK,
        "max_abs_diff");
  CHECK(max_diff <= 1e-6, "cached vs recompute exceeded 1e-6");

  /* Also verify token-by-token (prefill=1, decode=total_len-1) parity. */
  CHECK(et_mod5_gen_kv_cache_reset(cache) == ET_MOD5_GEN_OK, "reset");
  float *step1_logits = (float *)calloc(total_len * V, sizeof(float));
  CHECK(step1_logits != NULL, "alloc step1_logits");
  CHECK(et_mod5_gen_prefill(eng, cache, seq, 1u, step1_logits, NULL) ==
            ET_MOD5_GEN_OK,
        "prefill 1");
  for (size_t step = 1u; step < total_len; ++step) {
    CHECK(et_mod5_gen_decode_step(eng, cache, seq[step],
                                  step1_logits + step * V) == ET_MOD5_GEN_OK,
          "decode step-by-step");
  }
  double max_diff_1 = 0.0;
  CHECK(et_mod5_gen_aot_max_abs_diff_f32(step1_logits, recomp_logits,
                                         (int64_t)(total_len * V),
                                         &max_diff_1) == ET_MOD5_GEN_OK,
        "max_abs_diff step1");
  CHECK(max_diff_1 <= 1e-6, "1-by-1 decode vs recompute exceeded 1e-6");

  /* Inspect a cached KV slot and confirm out-of-range slot rejects. */
  float k_slot[ET_MOD5_GEN_MAX_DIM];
  float v_slot[ET_MOD5_GEN_MAX_DIM];
  CHECK(et_mod5_gen_kv_cache_read_slot(cache, 0u, 0u, (uint32_t)(total_len - 1u),
                                       k_slot, v_slot) == ET_MOD5_GEN_OK,
        "read_slot valid");
  CHECK(et_mod5_gen_kv_cache_read_slot(cache, 0u, 0u, (uint32_t)total_len,
                                       k_slot, v_slot) == ET_MOD5_GEN_ERR_BOUNDS,
        "read_slot out of bounds");

  printf("PARITY %-28s L=%u D=%u F=%u Hq=%u Hkv=%u win=%u alibi=%u "
         "prefill=%zu/%zu max_abs_diff=%.3e step1_diff=%.3e PASS\n",
         label, cfg->num_layers, cfg->hidden_dim, cfg->ffn_dim, cfg->num_heads,
         cfg->num_kv_heads, cfg->window_size, cfg->use_alibi, prefill_len,
         total_len, max_diff, max_diff_1);

  free(step1_logits);
  free(recomp_logits);
  free(cached_logits);
  free(last_logits);
  et_mod5_gen_kv_cache_destroy(cache);
  et_mod5_gen_engine_destroy(eng);
  free_synth_model(&m);
}

static void test_kv_parity_matrix(void) {
  et_mod5_gen_config cfg;

  /* 1. MHA (Hkv == Hq == 4) */
  memset(&cfg, 0, sizeof(cfg));
  cfg.vocab_size = 24u;
  cfg.max_seq = 16u;
  cfg.num_layers = 2u;
  cfg.hidden_dim = 16u;
  cfg.ffn_dim = 32u;
  cfg.num_heads = 4u;
  cfg.num_kv_heads = 4u;
  cfg.window_size = 0u;
  cfg.use_alibi = 0u;
  cfg.rms_epsilon = 0.125f;
  run_cached_vs_recompute_case("MHA_full_causal", &cfg, 5u, 14u, 11u);

  /* 2. GQA (Hkv == 2, Hq == 4) + exact MOD5 K1 T=8 shape */
  cfg.num_kv_heads = 2u;
  run_cached_vs_recompute_case("GQA_Hkv2_K1_T8", &cfg, 3u, 8u, 22u);
  run_cached_vs_recompute_case("GQA_Hkv2_full", &cfg, 6u, 16u, 23u);

  /* 3. MQA (Hkv == 1, Hq == 4) + exact MOD5 K1 T=8 shape */
  cfg.num_kv_heads = 1u;
  run_cached_vs_recompute_case("MQA_Hkv1_K1_T8", &cfg, 4u, 8u, 33u);
  run_cached_vs_recompute_case("MQA_Hkv1_full", &cfg, 5u, 15u, 34u);

  /* 4. ALiBi slopes (GQA and MHA) */
  cfg.max_seq = 20u;
  cfg.num_kv_heads = 2u;
  cfg.use_alibi = 1u;
  cfg.alibi_slopes[0] = 0.5f;
  cfg.alibi_slopes[1] = 0.25f;
  cfg.alibi_slopes[2] = 0.125f;
  cfg.alibi_slopes[3] = 0.0625f;
  run_cached_vs_recompute_case("GQA_ALiBi", &cfg, 7u, 18u, 44u);
  cfg.num_kv_heads = 4u;
  run_cached_vs_recompute_case("MHA_ALiBi", &cfg, 4u, 16u, 45u);

  /* 5. Sliding-window attention (window=1, window=3, window=5) + ALiBi */
  cfg.num_kv_heads = 2u;
  cfg.window_size = 1u;
  run_cached_vs_recompute_case("GQA_ALiBi_window1", &cfg, 5u, 12u, 55u);
  cfg.window_size = 3u;
  run_cached_vs_recompute_case("GQA_ALiBi_window3", &cfg, 4u, 16u, 56u);
  cfg.window_size = 5u;
  cfg.use_alibi = 0u;
  run_cached_vs_recompute_case("GQA_window5", &cfg, 6u, 19u, 57u);

  /* 6. Max-scale boundary: max_seq=64, D=32, F=64, L=8, GQA + ALiBi + window=16 */
  memset(&cfg, 0, sizeof(cfg));
  cfg.vocab_size = 32u;
  cfg.max_seq = 64u;
  cfg.num_layers = 8u;
  cfg.hidden_dim = 32u;
  cfg.ffn_dim = 64u;
  cfg.num_heads = 4u;
  cfg.num_kv_heads = 2u;
  cfg.window_size = 16u;
  cfg.use_alibi = 1u;
  cfg.alibi_slopes[0] = 0.5f;
  cfg.alibi_slopes[1] = 0.25f;
  cfg.alibi_slopes[2] = 0.125f;
  cfg.alibi_slopes[3] = 0.0625f;
  cfg.rms_epsilon = 0.125f;
  run_cached_vs_recompute_case("MAX_L8_D32_F64_S64_win16", &cfg, 16u, 64u,
                               66u);
}

static void test_sampling_semantics(void) {
  /* 1. Greedy lowest-index tie-break including -0.0f vs +0.0f. */
  {
    const float tied_pos[5] = {1.0f, 3.5f, 3.5f, 2.0f, 3.5f};
    const float tied_zero[4] = {-1.0f, -0.0f, +0.0f, -0.5f};
    et_mod5_gen_sampling_config sc;
    memset(&sc, 0, sizeof(sc));
    sc.mode = ET_MOD5_GEN_SAMPLE_GREEDY;
    sc.eos_token_id = -1;
    et_mod5_gen_rng rng;
    CHECK(et_mod5_gen_rng_init(&rng, 99u) == ET_MOD5_GEN_OK, "rng_init");
    int64_t tok = -1;
    CHECK(et_mod5_gen_sample(tied_pos, 5u, &sc, &rng, &tok) == ET_MOD5_GEN_OK,
          "greedy pos");
    CHECK(tok == 1 && rng.counter == 0u, "greedy tie-break lowest index 1");
    CHECK(et_mod5_gen_sample(tied_zero, 4u, &sc, NULL, &tok) == ET_MOD5_GEN_OK,
          "greedy signed zero");
    CHECK(tok == 1, "greedy -0.0f vs +0.0f lowest index 1");
  }

  /* 2. Top-k filtering strictly restricts support. */
  {
    const float logits[8] = {0.1f, 4.0f, 0.2f, 3.9f, -2.0f, 0.3f, 0.4f, 0.5f};
    et_mod5_gen_sampling_config sc;
    memset(&sc, 0, sizeof(sc));
    sc.mode = ET_MOD5_GEN_SAMPLE_CATEGORICAL;
    sc.temperature = 1.0f;
    sc.top_k = 2u;
    sc.top_p = 1.0f;
    sc.eos_token_id = -1;

    et_mod5_gen_rng rng;
    CHECK(et_mod5_gen_rng_init(&rng, 12345u) == ET_MOD5_GEN_OK, "rng_init");
    int seen_1 = 0;
    int seen_3 = 0;
    for (int i = 0; i < 200; ++i) {
      int64_t tok = -1;
      CHECK(et_mod5_gen_sample(logits, 8u, &sc, &rng, &tok) == ET_MOD5_GEN_OK,
            "top_k=2 sample");
      CHECK(tok == 1 || tok == 3, "top_k=2 emitted token outside top-2");
      if (tok == 1)
        ++seen_1;
      if (tok == 3)
        ++seen_3;
    }
    CHECK(seen_1 > 0 && seen_3 > 0, "top_k=2 should sample both top-2 tokens");
    CHECK(rng.counter == 200u, "rng counter exact advance");

    /* top_k=1 must always return argmax (lowest index on tie). */
    const float tie_logits[4] = {2.0f, 5.0f, 5.0f, 1.0f};
    sc.top_k = 1u;
    for (int i = 0; i < 32; ++i) {
      int64_t tok = -1;
      CHECK(et_mod5_gen_sample(tie_logits, 4u, &sc, &rng, &tok) ==
                ET_MOD5_GEN_OK,
            "top_k=1 sample");
      CHECK(tok == 1, "top_k=1 must select lowest-index argmax");
    }
  }

  /* 3. Top-p (nucleus) filtering strictly restricts support. */
  {
    /* Probabilities at T=1: token 2 (~0.64), token 5 (~0.24), others small.
     * With top_p = 0.80, prefix is {2 (0.64), 5 (0.24)} cum=0.88 >= 0.80. */
    const float logits[6] = {-4.0f, -4.0f, 2.0f, -4.0f, -4.0f, 1.0f};
    et_mod5_gen_sampling_config sc;
    memset(&sc, 0, sizeof(sc));
    sc.mode = ET_MOD5_GEN_SAMPLE_CATEGORICAL;
    sc.temperature = 1.0f;
    sc.top_k = 0u;
    sc.top_p = 0.80f;
    sc.eos_token_id = -1;
    et_mod5_gen_rng rng;
    CHECK(et_mod5_gen_rng_init(&rng, 777u) == ET_MOD5_GEN_OK, "rng_init");
    int seen_2 = 0;
    int seen_5 = 0;
    for (int i = 0; i < 200; ++i) {
      int64_t tok = -1;
      CHECK(et_mod5_gen_sample(logits, 6u, &sc, &rng, &tok) == ET_MOD5_GEN_OK,
            "top_p=0.80 sample");
      CHECK(tok == 2 || tok == 5, "top_p=0.80 emitted outside nucleus");
      if (tok == 2)
        ++seen_2;
      if (tok == 5)
        ++seen_5;
    }
    CHECK(seen_2 > 0 && seen_5 > 0, "top_p=0.80 samples both nucleus items");

    /* Very small top_p still keeps top-1 candidate. */
    sc.top_p = 0.01f;
    for (int i = 0; i < 32; ++i) {
      int64_t tok = -1;
      CHECK(et_mod5_gen_sample(logits, 6u, &sc, &rng, &tok) == ET_MOD5_GEN_OK,
            "top_p=0.01 sample");
      CHECK(tok == 2, "top_p=0.01 must keep top-1");
    }
  }

  /* 4. Temperature scaling effect (low T sharper than high T). */
  {
    const float logits[4] = {1.0f, 0.2f, 0.0f, -0.2f};
    et_mod5_gen_sampling_config sc;
    memset(&sc, 0, sizeof(sc));
    sc.mode = ET_MOD5_GEN_SAMPLE_CATEGORICAL;
    sc.top_k = 0u;
    sc.top_p = 1.0f;
    sc.eos_token_id = -1;

    et_mod5_gen_rng rng_low, rng_high;
    CHECK(et_mod5_gen_rng_init(&rng_low, 42u) == ET_MOD5_GEN_OK, "rng_low");
    CHECK(et_mod5_gen_rng_init(&rng_high, 42u) == ET_MOD5_GEN_OK, "rng_high");

    int hits_low = 0;
    int hits_high = 0;
    for (int i = 0; i < 300; ++i) {
      int64_t t_low = -1;
      int64_t t_high = -1;
      sc.temperature = 0.05f;
      CHECK(et_mod5_gen_sample(logits, 4u, &sc, &rng_low, &t_low) ==
                ET_MOD5_GEN_OK,
            "low T");
      sc.temperature = 5.0f;
      CHECK(et_mod5_gen_sample(logits, 4u, &sc, &rng_high, &t_high) ==
                ET_MOD5_GEN_OK,
            "high T");
      if (t_low == 0)
        ++hits_low;
      if (t_high == 0)
        ++hits_high;
    }
    CHECK(hits_low == 300 && hits_high < 150,
          "temperature failed to sharpen/flatten distribution");
  }

  /* 5. Observable RNG state replay & exhausted sentinel rejection. */
  {
    const float logits[6] = {0.1f, 0.4f, 0.2f, 0.5f, 0.3f, 0.0f};
    et_mod5_gen_sampling_config sc;
    memset(&sc, 0, sizeof(sc));
    sc.mode = ET_MOD5_GEN_SAMPLE_CATEGORICAL;
    sc.temperature = 1.0f;
    sc.top_k = 0u;
    sc.top_p = 1.0f;
    sc.eos_token_id = -1;

    et_mod5_gen_rng r1;
    CHECK(et_mod5_gen_rng_init(&r1, 2026u) == ET_MOD5_GEN_OK, "r1 init");
    int64_t dummy = -1;
    for (int i = 0; i < 5; ++i)
      CHECK(et_mod5_gen_sample(logits, 6u, &sc, &r1, &dummy) == ET_MOD5_GEN_OK,
            "advance r1");
    et_mod5_gen_rng saved = r1;
    CHECK(saved.key == 2026u && saved.counter == 5u, "observable RNG state");

    int64_t seq_a[16];
    int64_t seq_b[16];
    for (int i = 0; i < 16; ++i)
      CHECK(et_mod5_gen_sample(logits, 6u, &sc, &r1, &seq_a[i]) ==
                ET_MOD5_GEN_OK,
            "seq_a");
    for (int i = 0; i < 16; ++i)
      CHECK(et_mod5_gen_sample(logits, 6u, &sc, &saved, &seq_b[i]) ==
                ET_MOD5_GEN_OK,
            "seq_b");
    CHECK(memcmp(seq_a, seq_b, sizeof(seq_a)) == 0 &&
              r1.counter == saved.counter && r1.counter == 21u,
          "mid-stream RNG replay mismatch");

    /* Exhausted counter sentinel UINT64_MAX must reject without mutation. */
    et_mod5_gen_rng exhausted = {2026u, UINT64_MAX};
    int64_t sentinel = 999;
    CHECK(et_mod5_gen_sample(logits, 6u, &sc, &exhausted, &sentinel) ==
              ET_MOD5_GEN_ERR_OVERFLOW,
          "exhausted RNG must reject");
    CHECK(exhausted.counter == UINT64_MAX && sentinel == 999,
          "exhausted RNG mutated output");
  }

  printf("SAMPLING greedy/temp/top-k/top-p/RNG-replay: PASS\n");
}

static void test_eos_and_end_to_end_generate(void) {
  et_mod5_gen_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.vocab_size = 16u;
  cfg.max_seq = 20u;
  cfg.num_layers = 2u;
  cfg.hidden_dim = 16u;
  cfg.ffn_dim = 32u;
  cfg.num_heads = 4u;
  cfg.num_kv_heads = 2u;
  cfg.window_size = 8u;
  cfg.use_alibi = 1u;
  cfg.alibi_slopes[0] = 0.5f;
  cfg.alibi_slopes[1] = 0.25f;
  cfg.alibi_slopes[2] = 0.125f;
  cfg.alibi_slopes[3] = 0.0625f;
  cfg.rms_epsilon = 0.125f;

  synth_model m;
  init_synth_model(&m, &cfg, 88u);

  et_mod5_gen_engine *eng = NULL;
  et_mod5_gen_kv_cache *cache = NULL;
  CHECK(et_mod5_gen_engine_create(&cfg, &eng) == ET_MOD5_GEN_OK, "eng create");
  CHECK(et_mod5_gen_engine_load_weights(eng, &m.weights) == ET_MOD5_GEN_OK,
        "load weights");
  CHECK(et_mod5_gen_kv_cache_create(&cfg, &cache) == ET_MOD5_GEN_OK,
        "cache create");

  const int64_t prompt[4] = {1, 4, 1, 5};
  et_mod5_gen_sampling_config sc;
  memset(&sc, 0, sizeof(sc));
  sc.mode = ET_MOD5_GEN_SAMPLE_GREEDY;
  sc.eos_token_id = -1;
  sc.max_new_tokens = 8u;

  int64_t full_out[8];
  float full_logits[8 * 16];
  et_mod5_gen_rng rng;
  CHECK(et_mod5_gen_rng_init(&rng, 10u) == ET_MOD5_GEN_OK, "rng init");
  et_mod5_gen_result res;
  CHECK(et_mod5_gen_generate(eng, cache, prompt, 4u, &sc, &rng, full_out, 8u,
                             full_logits, 8u * 16u, &res) == ET_MOD5_GEN_OK,
        "full greedy generate");
  CHECK(res.generated_len == 8u && res.stopped_by_eos == 0u &&
            res.final_rng.counter == 0u,
        "full greedy result");

  /* Set EOS to the 3rd generated token (index 2) and verify early stop. */
  sc.eos_token_id = full_out[2];
  int64_t eos_out[8];
  for (size_t i = 0u; i < 8u; ++i)
    eos_out[i] = -99;
  et_mod5_gen_result eos_res;
  CHECK(et_mod5_gen_generate(eng, cache, prompt, 4u, &sc, &rng, eos_out, 8u,
                             NULL, 0u, &eos_res) == ET_MOD5_GEN_OK,
        "eos generate");
  /* Note: full_out[2] might also have appeared at index 0 or 1; find first. */
  uint32_t expected_stop_len = 0u;
  for (uint32_t i = 0u; i < 8u; ++i) {
    if (full_out[i] == sc.eos_token_id) {
      expected_stop_len = i + 1u;
      break;
    }
  }
  CHECK(eos_res.stopped_by_eos == 1u &&
            eos_res.generated_len == expected_stop_len &&
            expected_stop_len <= 3u,
        "EOS early stop length");
  for (uint32_t i = 0u; i < expected_stop_len; ++i)
    CHECK(eos_out[i] == full_out[i], "EOS prefix tokens match");
  CHECK(eos_out[expected_stop_len] == -99, "EOS wrote past stop position");

  /* Seeded categorical generation replay in two runs. */
  sc.mode = ET_MOD5_GEN_SAMPLE_CATEGORICAL;
  sc.temperature = 0.8f;
  sc.top_k = 6u;
  sc.top_p = 0.9f;
  sc.eos_token_id = -1;
  sc.max_new_tokens = 8u;

  int64_t cat_a[8], cat_b[8];
  float cat_logits_a[8 * 16], cat_logits_b[8 * 16];
  et_mod5_gen_rng r_a, r_b;
  et_mod5_gen_result res_a, res_b;
  CHECK(et_mod5_gen_rng_init(&r_a, 314159u) == ET_MOD5_GEN_OK, "r_a");
  CHECK(et_mod5_gen_rng_init(&r_b, 314159u) == ET_MOD5_GEN_OK, "r_b");
  CHECK(et_mod5_gen_generate(eng, cache, prompt, 4u, &sc, &r_a, cat_a, 8u,
                             cat_logits_a, 8u * 16u, &res_a) == ET_MOD5_GEN_OK,
        "cat_a");
  CHECK(et_mod5_gen_generate(eng, cache, prompt, 4u, &sc, &r_b, cat_b, 8u,
                             cat_logits_b, 8u * 16u, &res_b) == ET_MOD5_GEN_OK,
        "cat_b");
  CHECK(memcmp(cat_a, cat_b, sizeof(cat_a)) == 0 &&
            memcmp(cat_logits_a, cat_logits_b, sizeof(cat_logits_a)) == 0 &&
            res_a.final_rng.counter == 8u && res_b.final_rng.counter == 8u,
        "categorical replay exact match");

  printf("GENERATE EOS early stop (len=%u) + categorical replay: PASS\n",
         eos_res.generated_len);

  et_mod5_gen_kv_cache_destroy(cache);
  et_mod5_gen_engine_destroy(eng);
  free_synth_model(&m);
}

static void test_negative_and_bounds(void) {
  et_mod5_gen_config base;
  memset(&base, 0, sizeof(base));
  base.vocab_size = 16u;
  base.max_seq = 8u;
  base.num_layers = 1u;
  base.hidden_dim = 16u;
  base.ffn_dim = 32u;
  base.num_heads = 4u;
  base.num_kv_heads = 2u;
  base.window_size = 0u;
  base.use_alibi = 0u;
  base.rms_epsilon = 0.125f;

  /* Invalid configs */
  et_mod5_gen_config bad;
  bad = base; bad.vocab_size = 0u;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "V=0");
  bad = base; bad.vocab_size = ET_MOD5_GEN_MAX_VOCAB + 1u;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "V>max");
  bad = base; bad.max_seq = 0u;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "S=0");
  bad = base; bad.max_seq = ET_MOD5_GEN_MAX_SEQ + 1u;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "S>64");
  bad = base; bad.num_layers = 0u;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "L=0");
  bad = base; bad.num_layers = ET_MOD5_GEN_MAX_LAYERS + 1u;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "L>8");
  bad = base; bad.hidden_dim = 0u;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "D=0");
  bad = base; bad.hidden_dim = ET_MOD5_GEN_MAX_DIM + 1u;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "D>32");
  bad = base; bad.ffn_dim = 0u;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "F=0");
  bad = base; bad.ffn_dim = ET_MOD5_GEN_MAX_FFN + 1u;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "F>64");
  bad = base; bad.num_heads = 3u; /* 16 % 3 != 0 */
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "D%Hq");
  bad = base; bad.num_kv_heads = 3u; /* 4 % 3 != 0 */
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "Hq%Hkv");
  bad = base; bad.num_kv_heads = 8u; /* Hkv > Hq */
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "Hkv>Hq");
  bad = base; bad.rms_epsilon = 0.0f;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "eps=0");
  bad = base; bad.rms_epsilon = -0.0f;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "eps=-0");
  bad = base; bad.rms_epsilon = -0.1f;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "eps<0");
  bad = base; bad.rms_epsilon = NAN;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "eps=NaN");
  bad = base; bad.use_alibi = 2u;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_CONFIG, "alibi=2");
  bad = base; bad.use_alibi = 1u; bad.alibi_slopes[2] = INFINITY;
  CHECK(et_mod5_gen_validate_config(&bad) == ET_MOD5_GEN_ERR_NONFINITE,
        "alibi inf");

  /* Unloaded weights state rejection */
  et_mod5_gen_engine *eng = NULL;
  et_mod5_gen_kv_cache *cache = NULL;
  CHECK(et_mod5_gen_engine_create(&base, &eng) == ET_MOD5_GEN_OK, "create eng");
  CHECK(et_mod5_gen_kv_cache_create(&base, &cache) == ET_MOD5_GEN_OK,
        "create cache");
  const int64_t tok0[2] = {1, 2};
  float logits_out[16];
  CHECK(et_mod5_gen_prefill(eng, cache, tok0, 2u, logits_out, NULL) ==
            ET_MOD5_GEN_ERR_STATE,
        "prefill before weights must fail");

  /* Non-finite weights rejection (must leave weights unloaded) */
  synth_model m;
  init_synth_model(&m, &base, 99u);
  m.head[3] = NAN;
  CHECK(et_mod5_gen_engine_load_weights(eng, &m.weights) ==
            ET_MOD5_GEN_ERR_NONFINITE,
        "NaN head weight rejected");
  CHECK(et_mod5_gen_prefill(eng, cache, tok0, 2u, logits_out, NULL) ==
            ET_MOD5_GEN_ERR_STATE,
        "engine still unloaded after atomic NaN weight rejection");
  m.head[3] = 0.0f;
  CHECK(et_mod5_gen_engine_load_weights(eng, &m.weights) == ET_MOD5_GEN_OK,
        "valid weights load");

  /* Cache config mismatch rejection */
  et_mod5_gen_config diff_cfg = base;
  diff_cfg.num_kv_heads = 1u;
  et_mod5_gen_kv_cache *wrong_cache = NULL;
  CHECK(et_mod5_gen_kv_cache_create(&diff_cfg, &wrong_cache) == ET_MOD5_GEN_OK,
        "wrong cache");
  CHECK(et_mod5_gen_prefill(eng, wrong_cache, tok0, 2u, logits_out, NULL) ==
            ET_MOD5_GEN_ERR_CONFIG,
        "mismatched cache rejected");
  et_mod5_gen_kv_cache_destroy(wrong_cache);

  /* Token bounds and cache overflow rejection (preserving cache->seq_len) */
  const int64_t bad_tok_neg[2] = {1, -1};
  const int64_t bad_tok_hi[2] = {1, 16};
  CHECK(et_mod5_gen_prefill(eng, cache, bad_tok_neg, 2u, logits_out, NULL) ==
            ET_MOD5_GEN_ERR_BOUNDS,
        "neg token rejected");
  CHECK(et_mod5_gen_prefill(eng, cache, bad_tok_hi, 2u, logits_out, NULL) ==
            ET_MOD5_GEN_ERR_BOUNDS,
        "oob token rejected");
  CHECK(et_mod5_gen_prefill(eng, cache, tok0, 0u, logits_out, NULL) ==
            ET_MOD5_GEN_ERR_BOUNDS,
        "zero prompt rejected");

  const int64_t full8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  CHECK(et_mod5_gen_prefill(eng, cache, full8, 8u, logits_out, NULL) ==
            ET_MOD5_GEN_OK,
        "prefill to max_seq=8");
  CHECK(et_mod5_gen_decode_step(eng, cache, 1, logits_out) ==
            ET_MOD5_GEN_ERR_CACHE_FULL,
        "decode overflow rejected");
  uint32_t clen = 0u;
  CHECK(et_mod5_gen_kv_cache_len(cache, &clen) == ET_MOD5_GEN_OK && clen == 8u,
        "cache len unchanged on overflow");

  /* Generate overflow rejection */
  et_mod5_gen_sampling_config sc;
  memset(&sc, 0, sizeof(sc));
  sc.mode = ET_MOD5_GEN_SAMPLE_GREEDY;
  sc.eos_token_id = -1;
  sc.max_new_tokens = 6u; /* 4 + 6 = 10 > max_seq(8) */
  int64_t gen_out[6];
  et_mod5_gen_result res;
  CHECK(et_mod5_gen_generate(eng, cache, full8, 4u, &sc, NULL, gen_out, 6u,
                             NULL, 0u, &res) == ET_MOD5_GEN_ERR_CACHE_FULL,
        "generate overflow rejected");

  /* Non-finite logits and invalid sampling configs */
  float nan_logits[4] = {0.1f, NAN, 0.2f, 0.3f};
  float inf_logits[4] = {0.1f, INFINITY, 0.2f, 0.3f};
  float ninf_logits[4] = {0.1f, -INFINITY, 0.2f, 0.3f};
  float ok_logits[4] = {0.1f, 0.2f, 0.3f, 0.4f};
  et_mod5_gen_rng rng;
  CHECK(et_mod5_gen_rng_init(&rng, 7u) == ET_MOD5_GEN_OK, "rng");
  int64_t tok = 777;
  sc.mode = ET_MOD5_GEN_SAMPLE_GREEDY;
  CHECK(et_mod5_gen_sample(nan_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_NONFINITE,
        "greedy NaN");
  CHECK(et_mod5_gen_sample(inf_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_NONFINITE,
        "greedy +Inf");
  CHECK(et_mod5_gen_sample(ninf_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_NONFINITE,
        "greedy -Inf");
  CHECK(tok == 777 && rng.counter == 0u, "greedy error preserved output");

  sc.mode = ET_MOD5_GEN_SAMPLE_CATEGORICAL;
  sc.temperature = 1.0f;
  sc.top_k = 0u;
  sc.top_p = 1.0f;
  sc.eos_token_id = -1;
  CHECK(et_mod5_gen_sample(nan_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_NONFINITE,
        "cat NaN");
  CHECK(et_mod5_gen_sample(inf_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_NONFINITE,
        "cat +Inf");
  sc.temperature = 0.0f;
  CHECK(et_mod5_gen_sample(ok_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_CONFIG,
        "T=0");
  sc.temperature = -1.0f;
  CHECK(et_mod5_gen_sample(ok_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_CONFIG,
        "T<0");
  sc.temperature = 1.0f;
  sc.top_k = 5u; /* > vocab_size 4 */
  CHECK(et_mod5_gen_sample(ok_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_CONFIG,
        "top_k>V");
  sc.top_k = 0u;
  sc.top_p = 0.0f;
  CHECK(et_mod5_gen_sample(ok_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_CONFIG,
        "top_p=0");
  sc.top_p = 1.01f;
  CHECK(et_mod5_gen_sample(ok_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_CONFIG,
        "top_p>1");
  sc.top_p = 1.0f;
  sc.eos_token_id = 4; /* >= vocab_size */
  CHECK(et_mod5_gen_sample(ok_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_CONFIG,
        "eos>=V");
  sc.eos_token_id = -2;
  CHECK(et_mod5_gen_sample(ok_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_CONFIG,
        "eos<-1");
  sc.eos_token_id = -1;
  sc.mode = 9u;
  CHECK(et_mod5_gen_sample(ok_logits, 4u, &sc, &rng, &tok) ==
            ET_MOD5_GEN_ERR_CONFIG,
        "bad mode");
  CHECK(tok == 777 && rng.counter == 0u, "cat errors preserved rng & tok");

  /* Misaligned pointer rejection without dereference (UBSan clean) */
  unsigned char unaligned_buf[64];
  memset(unaligned_buf, 0, sizeof(unaligned_buf));
  const float *bad_f32_ptr =
      (const float *)(uintptr_t)((uintptr_t)unaligned_buf + 1u);
  sc.mode = ET_MOD5_GEN_SAMPLE_GREEDY;
  CHECK(et_mod5_gen_sample(bad_f32_ptr, 4u, &sc, NULL, &tok) ==
            ET_MOD5_GEN_ERR_UNALIGNED,
        "misaligned float ptr rejected");

  /* Null argument rejection */
  CHECK(et_mod5_gen_sample(NULL, 4u, &sc, NULL, &tok) ==
            ET_MOD5_GEN_ERR_ARGUMENT,
        "null logits rejected");

  printf("NEGATIVE config/bounds/overflow/nonfinite/alignment: PASS\n");

  et_mod5_gen_kv_cache_destroy(cache);
  et_mod5_gen_engine_destroy(eng);
  free_synth_model(&m);
}

int main(void) {
  test_provider_parity();
  test_kv_parity_matrix();
  test_sampling_semantics();
  test_eos_and_end_to_end_generate();
  test_negative_and_bounds();
  printf("ALL MOD5 GENERATION NATIVE TESTS: PASS\n");
  return 0;
}
