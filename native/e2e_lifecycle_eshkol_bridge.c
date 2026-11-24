#define _POSIX_C_SOURCE 200809L

#include "e2e_lifecycle_eshkol_bridge.h"

#include "eshkol_transformer/data4_pipeline_abi.h"
#include "eshkol_transformer/dpo_alignment_abi.h"
#include "eshkol_transformer/eval5_benchmark_abi.h"
#include "eshkol_transformer/io5_quant_abi.h"
#include "eshkol_transformer/lora_adapter_abi.h"
#include "eshkol_transformer/mod5_adamw_abi.h"
#include "eshkol_transformer/mod5_rope_abi.h"
#include "eshkol_transformer/spec_decode_abi.h"
#include "eshkol_transformer/t2_bpe_trainer_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#pragma STDC FP_CONTRACT OFF

#define ET_E2E_CORPUS_MAGIC "ETE2ECP1"
#define ET_E2E_MAX_DOCS 32u
#define ET_E2E_MAX_SHARD_TOKENS 4096u
#define ET_E2E_PATH_MAX 1024u
#define ET_E2E_BPE_TARGET_VOCAB 360u
#define ET_E2E_MODEL_TAG "e2e-mod5-rope-d16-h2-dh8-ff32-v1"
#define ET_E2E_NUM_TRAIN_WINDOWS 4u

/* Stage 1 raw synthetic corpus:
 * 11 documents -> 6 kept, 5 dropped across 5 distinct DATA4 dispositions:
 *   - 6 valid distinct technical documents with shared recurring vocabulary
 *     so Stage 2 BPE achieves > 1.5x compression
 *   - 1 exact duplicate of doc_00
 *   - 1 near duplicate of doc_01 (2-byte suffix edit, high 5-gram Jaccard)
 *   - 1 invalid UTF-8 document (lone 0xFF/0xFE/0x80 bytes)
 *   - 1 too-short document (< 40 bytes)
 *   - 1 low-quality non-alnum punctuation document (alnum ratio = 0.0)
 */
static const char k_doc0[] =
    "Deterministic transformer training combines rotary position embeddings "
    "with AdamW optimization and checksummed snapshot persistence. "
    "Every deterministic transformer training run tracks tokenizer identity, "
    "optimizer state, and checkpoint verification across process boundaries.";

static const char k_doc1[] =
    "Byte pair encoding tokenization compresses recurring subword sequences "
    "in transformer corpora while preserving exact lossless byte decoding. "
    "Deterministic tokenization verifies checksummed artifacts and stable "
    "vocabulary compression across fresh process reload boundaries.";

static const char k_doc2[] =
    "Low rank adapter fine tuning updates compact projection matrices while "
    "keeping frozen base transformer weights bit identical under SHA256. "
    "Adapter merge and unmerge parity checks confirm deterministic linear "
    "projection behavior before and after checkpoint serialization.";

static const char k_doc3[] =
    "Direct preference optimization aligns policy logits against frozen "
    "reference logits using masked sequence log probabilities and reward "
    "margins. Preference alignment reduces training loss while raising "
    "reward margin and preserving exact frozen reference parameters.";

static const char k_doc4[] =
    "Contamination gated benchmark evaluation audits held out datasets "
    "against training corpora using exact document matching and byte ngram "
    "overlap before emitting checksummed evaluation manifests and perplexity "
    "metrics across deterministic verification pipelines.";

static const char k_doc5[] =
    "Lossless speculative decoding verifies draft token proposals against "
    "target transformer logits using greedy equivalence and seeded rejection "
    "sampling, followed by native F16 BF16 and Q8 quantization codec round "
    "trip validation with bounded reconstruction error metrics.";

static const char k_doc_near_dup[] =
    "Byte pair encoding tokenization compresses recurring subword sequences "
    "in transformer corpora while preserving exact lossless byte decoding. "
    "Deterministic tokenization verifies checksummed artifacts and stable "
    "vocabulary compression across fresh process reload boundaries!!";

static const uint8_t k_doc_bad_utf8[] = {
    'B', 'a', 'd', ' ', 'U', 'T', 'F', '8', ' ', 'p', 'a', 'y', 'l', 'o', 'a',
    'd', ' ', 0xFFu, 0xFEu, 0x80u, ' ', 't', 'r', 'a', 'n', 's', 'f', 'o', 'r',
    'm', 'e', 'r', ' ', 'c', 'o', 'r', 'p', 'u', 's', ' ', 'd', 'o', 'c', '.'};

static const char k_doc_too_short[] = "tiny doc";

static const char k_doc_low_quality[] =
    "!@#$%^&*()-_=+[]{};:'\",.<>/?!@#$%^&*()-_=+[]{};:'\",.<>/?!!!!";

/* Disjoint clean held-out eval documents (zero 8-byte n-gram overlap with
 * Stage 1 ML training corpus). */
static const char k_clean_eval_doc0[] =
    "Amber orchid petals unfold briskly beneath quiet autumn moonlight near "
    "mossy riverbanks.";
static const char k_clean_eval_doc1[] =
    "Nordic violins echo softly across icy basalt fjords while cedar canoes "
    "drift toward harbor lanterns.";
static const char k_clean_eval_doc2[] =
    "Fragrant saffron cardamom broth simmers slowly inside hammered copper "
    "kettles beside woven reed baskets.";

typedef struct owned_corpus {
  size_t doc_count;
  uint8_t *docs[ET_E2E_MAX_DOCS];
  size_t lens[ET_E2E_MAX_DOCS];
} owned_corpus;

struct et_e2e_aot_ctx {
  et_e2e_stage_metrics_v1 m;
  uint32_t vocab_size;
  uint32_t bos_id;
  uint32_t eos_id;
  char tokenizer_id[64];
  size_t param_count;
  float *pretrain_weights; /* Stage 3 resumed weights (base/draft reference) */
  float *live_weights;     /* Post-LoRA + post-DPO weights */
  int64_t shard_tokens[ET_E2E_MAX_SHARD_TOKENS];
  size_t shard_token_count;
  uint8_t clean_corpus_sha256[ET_E2E_SHA256_BYTES];
};

/* ---- Utility helpers ---- */

static void put_u32_le(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xFFu);
  dst[1] = (uint8_t)((v >> 8u) & 0xFFu);
  dst[2] = (uint8_t)((v >> 16u) & 0xFFu);
  dst[3] = (uint8_t)((v >> 24u) & 0xFFu);
}

static uint32_t get_u32_le(const uint8_t *src) {
  return (uint32_t)src[0] | ((uint32_t)src[1] << 8u) |
         ((uint32_t)src[2] << 16u) | ((uint32_t)src[3] << 24u);
}

static void sha256_to_hex(const uint8_t digest[32], char out_hex[65]) {
  static const char hex[] = "0123456789abcdef";
  for (size_t i = 0u; i < 32u; ++i) {
    out_hex[i * 2u] = hex[(digest[i] >> 4u) & 0x0Fu];
    out_hex[i * 2u + 1u] = hex[digest[i] & 0x0Fu];
  }
  out_hex[64] = '\0';
}

static int build_path(char *out, size_t cap, const char *dir,
                      const char *name) {
  int n;
  if (dir == NULL || name == NULL || dir[0] == '\0' || name[0] == '\0') {
    return 0;
  }
  n = snprintf(out, cap, "%s/%s", dir, name);
  return n > 0 && (size_t)n < cap;
}

static et_e2e_status write_file_atomic(const char *path, const uint8_t *data,
                                       size_t len) {
  char tmp_path[ET_E2E_PATH_MAX];
  int fd;
  size_t written = 0u;
  int n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%ld", path,
                   (long)getpid());
  if (n <= 0 || (size_t)n >= sizeof(tmp_path)) {
    return ET_E2E_ERR_IO;
  }
  (void)unlink(tmp_path);
  fd = open(tmp_path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  if (fd < 0) {
    return ET_E2E_ERR_IO;
  }
  while (written < len) {
    ssize_t w = write(fd, data + written, len - written);
    if (w < 0) {
      if (errno == EINTR) {
        continue;
      }
      (void)close(fd);
      (void)unlink(tmp_path);
      return ET_E2E_ERR_IO;
    }
    written += (size_t)w;
  }
  if (fsync(fd) != 0) {
    (void)close(fd);
    (void)unlink(tmp_path);
    return ET_E2E_ERR_IO;
  }
  if (close(fd) != 0) {
    (void)unlink(tmp_path);
    return ET_E2E_ERR_IO;
  }
  if (rename(tmp_path, path) != 0) {
    (void)unlink(tmp_path);
    return ET_E2E_ERR_IO;
  }
  return ET_E2E_OK;
}

static et_e2e_status read_file_nofollow(const char *path, size_t max_bytes,
                                        uint8_t **out_data, size_t *out_len) {
  int fd;
  struct stat st;
  uint8_t *buf = NULL;
  size_t total = 0u;
  size_t got = 0u;

  fd = open(path, O_RDONLY | O_NOFOLLOW);
  if (fd < 0) {
    return ET_E2E_ERR_IO;
  }
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
      (uint64_t)st.st_size > (uint64_t)max_bytes) {
    (void)close(fd);
    return ET_E2E_ERR_IO;
  }
  total = (size_t)st.st_size;
  buf = (uint8_t *)calloc(1u, total > 0u ? total : 1u);
  if (buf == NULL) {
    (void)close(fd);
    return ET_E2E_ERR_ALLOC;
  }
  while (got < total) {
    ssize_t r = read(fd, buf + got, total - got);
    if (r < 0) {
      if (errno == EINTR) {
        continue;
      }
      free(buf);
      (void)close(fd);
      return ET_E2E_ERR_IO;
    }
    if (r == 0) {
      free(buf);
      (void)close(fd);
      return ET_E2E_ERR_IO;
    }
    got += (size_t)r;
  }
  (void)close(fd);
  *out_data = buf;
  *out_len = total;
  return ET_E2E_OK;
}

static void owned_corpus_free(owned_corpus *c) {
  if (c == NULL) {
    return;
  }
  for (size_t i = 0u; i < c->doc_count; ++i) {
    free(c->docs[i]);
    c->docs[i] = NULL;
    c->lens[i] = 0u;
  }
  c->doc_count = 0u;
}

static et_e2e_status save_clean_corpus_bundle(const char *path,
                                              const uint8_t *const *docs,
                                              const size_t *lens,
                                              size_t doc_count,
                                              uint8_t out_sha256[32]) {
  size_t payload_bytes = 8u + 4u;
  size_t total_bytes;
  uint8_t *buf;
  size_t off = 0u;
  et_e2e_status st;

  if (doc_count == 0u || doc_count > ET_E2E_MAX_DOCS) {
    return ET_E2E_ERR_ARGUMENT;
  }
  for (size_t i = 0u; i < doc_count; ++i) {
    if (lens[i] > UINT32_MAX || payload_bytes > (SIZE_MAX - 4u - lens[i])) {
      return ET_E2E_ERR_ARGUMENT;
    }
    payload_bytes += 4u + lens[i];
  }
  total_bytes = payload_bytes + 32u;
  buf = (uint8_t *)calloc(1u, total_bytes);
  if (buf == NULL) {
    return ET_E2E_ERR_ALLOC;
  }
  memcpy(buf, ET_E2E_CORPUS_MAGIC, 8u);
  off = 8u;
  put_u32_le(buf + off, (uint32_t)doc_count);
  off += 4u;
  for (size_t i = 0u; i < doc_count; ++i) {
    put_u32_le(buf + off, (uint32_t)lens[i]);
    off += 4u;
    if (lens[i] > 0u) {
      memcpy(buf + off, docs[i], lens[i]);
      off += lens[i];
    }
  }
  if (et_data4_sha256(buf, payload_bytes, buf + payload_bytes) != ET_DATA4_OK) {
    free(buf);
    return ET_E2E_ERR_STAGE1_DATA4;
  }
  if (out_sha256 != NULL) {
    memcpy(out_sha256, buf + payload_bytes, 32u);
  }
  st = write_file_atomic(path, buf, total_bytes);
  free(buf);
  return st;
}

static et_e2e_status load_clean_corpus_bundle(const char *path,
                                              owned_corpus *out_corpus,
                                              uint8_t out_sha256[32]) {
  uint8_t *buf = NULL;
  size_t len = 0u;
  uint8_t computed[32];
  uint32_t doc_count;
  size_t off;
  et_e2e_status st;

  memset(out_corpus, 0, sizeof(*out_corpus));
  st = read_file_nofollow(path, 1048576u, &buf, &len);
  if (st != ET_E2E_OK) {
    return st;
  }
  if (len < 8u + 4u + 32u || memcmp(buf, ET_E2E_CORPUS_MAGIC, 8u) != 0) {
    free(buf);
    return ET_E2E_ERR_STAGE1_DATA4;
  }
  if (et_data4_sha256(buf, len - 32u, computed) != ET_DATA4_OK ||
      memcmp(computed, buf + (len - 32u), 32u) != 0) {
    free(buf);
    return ET_E2E_ERR_STAGE1_DATA4;
  }
  doc_count = get_u32_le(buf + 8u);
  if (doc_count == 0u || doc_count > ET_E2E_MAX_DOCS) {
    free(buf);
    return ET_E2E_ERR_STAGE1_DATA4;
  }
  off = 12u;
  for (uint32_t i = 0u; i < doc_count; ++i) {
    uint32_t dlen;
    if (off + 4u > len - 32u) {
      owned_corpus_free(out_corpus);
      free(buf);
      return ET_E2E_ERR_STAGE1_DATA4;
    }
    dlen = get_u32_le(buf + off);
    off += 4u;
    if (off + (size_t)dlen > len - 32u) {
      owned_corpus_free(out_corpus);
      free(buf);
      return ET_E2E_ERR_STAGE1_DATA4;
    }
    out_corpus->docs[i] = (uint8_t *)calloc(1u, dlen > 0u ? (size_t)dlen : 1u);
    if (out_corpus->docs[i] == NULL) {
      owned_corpus_free(out_corpus);
      free(buf);
      return ET_E2E_ERR_ALLOC;
    }
    if (dlen > 0u) {
      memcpy(out_corpus->docs[i], buf + off, (size_t)dlen);
    }
    out_corpus->lens[i] = (size_t)dlen;
    out_corpus->doc_count = (size_t)(i + 1u);
    off += (size_t)dlen;
  }
  if (off != len - 32u) {
    owned_corpus_free(out_corpus);
    free(buf);
    return ET_E2E_ERR_STAGE1_DATA4;
  }
  if (out_sha256 != NULL) {
    memcpy(out_sha256, computed, 32u);
  }
  free(buf);
  return ET_E2E_OK;
}

static void format_tokenizer_id(const uint8_t sha256[32], char out[64]) {
  char hex[65];
  sha256_to_hex(sha256, hex);
  (void)snprintf(out, 64u, "etbpe100-%.16s", hex);
}

/* ---- Stage 3 MOD5 + RoPE tiny transformer block ---- */

typedef struct model_layout {
  size_t V;
  size_t D;
  size_t H;
  size_t Dh;
  size_t D_ff;
  size_t off_emb;
  size_t off_wq;
  size_t off_wk;
  size_t off_wv;
  size_t off_wo;
  size_t off_wff1;
  size_t off_wff2;
  size_t off_wlm;
  size_t total_params;
} model_layout;

static model_layout make_model_layout(size_t vocab_size) {
  model_layout L;
  L.V = vocab_size;
  L.D = ET_E2E_MODEL_DIM;
  L.H = ET_E2E_MODEL_HEADS;
  L.Dh = ET_E2E_MODEL_HEAD_DIM;
  L.D_ff = ET_E2E_MODEL_FF_DIM;
  L.off_emb = 0u;
  L.off_wq = L.off_emb + L.V * L.D;
  L.off_wk = L.off_wq + L.D * L.D;
  L.off_wv = L.off_wk + L.D * L.D;
  L.off_wo = L.off_wv + L.D * L.D;
  L.off_wff1 = L.off_wo + L.D * L.D;
  L.off_wff2 = L.off_wff1 + L.D_ff * L.D;
  L.off_wlm = L.off_wff2 + L.D * L.D_ff;
  L.total_params = L.off_wlm + L.V * L.D;
  return L;
}

static uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

static float det_uniform_f32(uint64_t *state, double scale) {
  uint64_t bits = splitmix64_next(state);
  double u = ((double)(bits >> 11u) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
  return (float)(u * scale);
}

static void init_model_weights(const model_layout *L, float *params) {
  uint64_t rng = UINT64_C(0xE2E30001CAFEBABE);
  for (size_t i = 0u; i < L->V * L->D; ++i) {
    params[L->off_emb + i] = det_uniform_f32(&rng, 0.16);
  }
  for (size_t i = 0u; i < 4u * L->D * L->D; ++i) {
    params[L->off_wq + i] = det_uniform_f32(&rng, 0.12);
  }
  for (size_t i = 0u; i < L->D_ff * L->D; ++i) {
    params[L->off_wff1 + i] = det_uniform_f32(&rng, 0.12);
  }
  for (size_t i = 0u; i < L->D * L->D_ff; ++i) {
    params[L->off_wff2 + i] = det_uniform_f32(&rng, 0.12);
  }
  for (size_t i = 0u; i < L->V * L->D; ++i) {
    params[L->off_wlm + i] = det_uniform_f32(&rng, 0.08);
  }
}

/* Forward (and optional analytic backward accumulation) on a single sequence
 * x_ids[0..T-1] -> optional y_ids[0..T-1].
 * Uses native et_mod5_rope_forward_f32 and et_mod5_rope_backward_f32. */
static et_e2e_status model_forward_backward_window(
    const model_layout *L, const float *params, const uint32_t *x_ids,
    const uint32_t *y_ids, size_t T, double loss_scale,
    float *out_h1 /* optional [T, D] */, float *out_h2 /* optional [T, D] */,
    float *out_logits /* optional [T, V] */, double *out_loss,
    float *grads_accum /* optional [total_params] */) {
  const size_t D = L->D;
  const size_t H = L->H;
  const size_t Dh = L->Dh;
  const size_t D_ff = L->D_ff;
  const size_t V = L->V;
  const float *W_emb = params + L->off_emb;
  const float *W_q = params + L->off_wq;
  const float *W_k = params + L->off_wk;
  const float *W_v = params + L->off_wv;
  const float *W_o = params + L->off_wo;
  const float *W_ff1 = params + L->off_wff1;
  const float *W_ff2 = params + L->off_wff2;
  const float *W_lm = params + L->off_wlm;

  float h0[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
  float Q_lin[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
  float K_lin[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
  float V_lin[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
  float Q_in[ET_E2E_MODEL_HEADS * ET_E2E_SEQ_LEN * ET_E2E_MODEL_HEAD_DIM];
  float K_in[ET_E2E_MODEL_HEADS * ET_E2E_SEQ_LEN * ET_E2E_MODEL_HEAD_DIM];
  float V_in[ET_E2E_MODEL_HEADS * ET_E2E_SEQ_LEN * ET_E2E_MODEL_HEAD_DIM];
  float Q_rot[ET_E2E_MODEL_HEADS * ET_E2E_SEQ_LEN * ET_E2E_MODEL_HEAD_DIM];
  float K_rot[ET_E2E_MODEL_HEADS * ET_E2E_SEQ_LEN * ET_E2E_MODEL_HEAD_DIM];
  double attn_prob[ET_E2E_MODEL_HEADS * ET_E2E_SEQ_LEN * ET_E2E_SEQ_LEN];
  float ctx[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
  float h1[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
  float ff_pre[ET_E2E_SEQ_LEN * ET_E2E_MODEL_FF_DIM];
  float ff_sig[ET_E2E_SEQ_LEN * ET_E2E_MODEL_FF_DIM];
  float ff_act[ET_E2E_SEQ_LEN * ET_E2E_MODEL_FF_DIM];
  float h2[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
  int64_t positions[ET_E2E_SEQ_LEN];
  et_kernel_error kerr;
  double inv_sqrt_dh = 1.0 / sqrt((double)Dh);
  float *logits_buf = NULL;
  double *probs_row = NULL;
  double loss_sum = 0.0;

  if (T == 0u || T > ET_E2E_SEQ_LEN) {
    return ET_E2E_ERR_ARGUMENT;
  }
  for (size_t t = 0u; t < T; ++t) {
    if (x_ids[t] >= V) {
      return ET_E2E_ERR_ARGUMENT;
    }
    if (y_ids != NULL && y_ids[t] >= V) {
      return ET_E2E_ERR_ARGUMENT;
    }
    positions[t] = (int64_t)t;
    memcpy(&h0[t * D], &W_emb[(size_t)x_ids[t] * D], D * sizeof(float));
  }

  /* Q, K, V linear projections. */
  for (size_t t = 0u; t < T; ++t) {
    for (size_t o = 0u; o < D; ++o) {
      double sq = 0.0;
      double sk = 0.0;
      double sv = 0.0;
      for (size_t d = 0u; d < D; ++d) {
        double xval = (double)h0[t * D + d];
        sq += xval * (double)W_q[o * D + d];
        sk += xval * (double)W_k[o * D + d];
        sv += xval * (double)W_v[o * D + d];
      }
      Q_lin[t * D + o] = (float)sq;
      K_lin[t * D + o] = (float)sk;
      V_lin[t * D + o] = (float)sv;
    }
  }

  /* Reshape [T, H*Dh] -> [H, T, Dh]. */
  for (size_t h = 0u; h < H; ++h) {
    for (size_t t = 0u; t < T; ++t) {
      for (size_t j = 0u; j < Dh; ++j) {
        size_t dst = (h * T + t) * Dh + j;
        size_t src = t * D + h * Dh + j;
        Q_in[dst] = Q_lin[src];
        K_in[dst] = K_lin[src];
        V_in[dst] = V_lin[src];
      }
    }
  }

  /* Apply native MOD5 RoPE forward to Q and K. */
  memset(&kerr, 0, sizeof(kerr));
  if (et_mod5_rope_forward_f32(1u, H, T, Dh, Q_in, positions,
                               ET_MOD5_ROPE_DEFAULT_THETA,
                               ET_MOD5_ROPE_DEFAULT_SCALE, Q_rot, &kerr) != 0) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  if (et_mod5_rope_forward_f32(1u, H, T, Dh, K_in, positions,
                               ET_MOD5_ROPE_DEFAULT_THETA,
                               ET_MOD5_ROPE_DEFAULT_SCALE, K_rot, &kerr) != 0) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }

  /* Causal scaled dot-product attention. */
  memset(attn_prob, 0, sizeof(attn_prob));
  for (size_t h = 0u; h < H; ++h) {
    for (size_t t = 0u; t < T; ++t) {
      double scores[ET_E2E_SEQ_LEN];
      double max_s = -1e300;
      double sum_e = 0.0;
      for (size_t s = 0u; s <= t; ++s) {
        double dot = 0.0;
        for (size_t j = 0u; j < Dh; ++j) {
          dot += (double)Q_rot[(h * T + t) * Dh + j] *
                 (double)K_rot[(h * T + s) * Dh + j];
        }
        scores[s] = dot * inv_sqrt_dh;
        if (scores[s] > max_s) {
          max_s = scores[s];
        }
      }
      for (size_t s = 0u; s <= t; ++s) {
        double e = exp(scores[s] - max_s);
        attn_prob[(h * T + t) * T + s] = e;
        sum_e += e;
      }
      for (size_t s = 0u; s <= t; ++s) {
        attn_prob[(h * T + t) * T + s] /= sum_e;
      }
      for (size_t j = 0u; j < Dh; ++j) {
        double acc = 0.0;
        for (size_t s = 0u; s <= t; ++s) {
          acc += attn_prob[(h * T + t) * T + s] *
                 (double)V_in[(h * T + s) * Dh + j];
        }
        ctx[t * D + h * Dh + j] = (float)acc;
      }
    }
  }

  /* Output projection + residual -> h1. */
  for (size_t t = 0u; t < T; ++t) {
    for (size_t d = 0u; d < D; ++d) {
      double acc = (double)h0[t * D + d];
      for (size_t o = 0u; o < D; ++o) {
        acc += (double)ctx[t * D + o] * (double)W_o[d * D + o];
      }
      h1[t * D + d] = (float)acc;
    }
  }
  if (out_h1 != NULL) {
    memcpy(out_h1, h1, T * D * sizeof(float));
  }

  /* MLP (SiLU) + residual -> h2. */
  for (size_t t = 0u; t < T; ++t) {
    for (size_t f = 0u; f < D_ff; ++f) {
      double acc = 0.0;
      for (size_t d = 0u; d < D; ++d) {
        acc += (double)h1[t * D + d] * (double)W_ff1[f * D + d];
      }
      double sig = 1.0 / (1.0 + exp(-acc));
      ff_pre[t * D_ff + f] = (float)acc;
      ff_sig[t * D_ff + f] = (float)sig;
      ff_act[t * D_ff + f] = (float)(acc * sig);
    }
    for (size_t d = 0u; d < D; ++d) {
      double acc = (double)h1[t * D + d];
      for (size_t f = 0u; f < D_ff; ++f) {
        acc += (double)ff_act[t * D_ff + f] * (double)W_ff2[d * D_ff + f];
      }
      h2[t * D + d] = (float)acc;
    }
  }
  if (out_h2 != NULL) {
    memcpy(out_h2, h2, T * D * sizeof(float));
  }

  /* LM head + optional cross-entropy loss & backward. */
  if (out_logits != NULL || y_ids != NULL) {
    logits_buf =
        (out_logits != NULL) ? out_logits : (float *)calloc(T * V, sizeof(float));
    if (logits_buf == NULL) {
      return ET_E2E_ERR_ALLOC;
    }
    for (size_t t = 0u; t < T; ++t) {
      for (size_t v = 0u; v < V; ++v) {
        double acc = 0.0;
        for (size_t d = 0u; d < D; ++d) {
          acc += (double)h2[t * D + d] * (double)W_lm[v * D + d];
        }
        logits_buf[t * V + v] = (float)acc;
      }
    }
  }

  if (y_ids != NULL) {
    probs_row = (double *)calloc(V, sizeof(double));
    if (probs_row == NULL) {
      if (out_logits == NULL) {
        free(logits_buf);
      }
      return ET_E2E_ERR_ALLOC;
    }

    float dh2[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
    memset(dh2, 0, sizeof(dh2));

    for (size_t t = 0u; t < T; ++t) {
      double max_l = -1e300;
      double sum_e = 0.0;
      for (size_t v = 0u; v < V; ++v) {
        double lv = (double)logits_buf[t * V + v];
        if (lv > max_l) {
          max_l = lv;
        }
      }
      for (size_t v = 0u; v < V; ++v) {
        double e = exp((double)logits_buf[t * V + v] - max_l);
        probs_row[v] = e;
        sum_e += e;
      }
      double log_z = max_l + log(sum_e);
      double nll = log_z - (double)logits_buf[t * V + y_ids[t]];
      loss_sum += nll;

      if (grads_accum != NULL) {
        float *dW_lm = grads_accum + L->off_wlm;
        for (size_t v = 0u; v < V; ++v) {
          double p = probs_row[v] / sum_e;
          double dl = loss_scale * (p - (v == (size_t)y_ids[t] ? 1.0 : 0.0));
          for (size_t d = 0u; d < D; ++d) {
            dW_lm[v * D + d] += (float)(dl * (double)h2[t * D + d]);
            dh2[t * D + d] += (float)(dl * (double)W_lm[v * D + d]);
          }
        }
      }
    }

    if (out_loss != NULL) {
      *out_loss = loss_sum / (double)T;
    }

    if (grads_accum != NULL) {
      float *dW_emb = grads_accum + L->off_emb;
      float *dW_q = grads_accum + L->off_wq;
      float *dW_k = grads_accum + L->off_wk;
      float *dW_v = grads_accum + L->off_wv;
      float *dW_o = grads_accum + L->off_wo;
      float *dW_ff1 = grads_accum + L->off_wff1;
      float *dW_ff2 = grads_accum + L->off_wff2;

      float dh1[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
      float dh0[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
      float dctx[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
      float dQ_rot[ET_E2E_MODEL_HEADS * ET_E2E_SEQ_LEN * ET_E2E_MODEL_HEAD_DIM];
      float dK_rot[ET_E2E_MODEL_HEADS * ET_E2E_SEQ_LEN * ET_E2E_MODEL_HEAD_DIM];
      float dV_in[ET_E2E_MODEL_HEADS * ET_E2E_SEQ_LEN * ET_E2E_MODEL_HEAD_DIM];
      float dQ_in[ET_E2E_MODEL_HEADS * ET_E2E_SEQ_LEN * ET_E2E_MODEL_HEAD_DIM];
      float dK_in[ET_E2E_MODEL_HEADS * ET_E2E_SEQ_LEN * ET_E2E_MODEL_HEAD_DIM];

      /* MLP backward */
      memcpy(dh1, dh2, T * D * sizeof(float));
      for (size_t t = 0u; t < T; ++t) {
        double dff_pre_row[ET_E2E_MODEL_FF_DIM];
        for (size_t f = 0u; f < D_ff; ++f) {
          double dact = 0.0;
          for (size_t d = 0u; d < D; ++d) {
            dW_ff2[d * D_ff + f] +=
                (float)((double)dh2[t * D + d] * (double)ff_act[t * D_ff + f]);
            dact += (double)dh2[t * D + d] * (double)W_ff2[d * D_ff + f];
          }
          double s = (double)ff_sig[t * D_ff + f];
          double x = (double)ff_pre[t * D_ff + f];
          double dsilu = s * (1.0 + x * (1.0 - s));
          dff_pre_row[f] = dact * dsilu;
        }
        for (size_t f = 0u; f < D_ff; ++f) {
          double dp = dff_pre_row[f];
          for (size_t d = 0u; d < D; ++d) {
            dW_ff1[f * D + d] += (float)(dp * (double)h1[t * D + d]);
            dh1[t * D + d] += (float)(dp * (double)W_ff1[f * D + d]);
          }
        }
      }

      /* Output projection + residual backward */
      memcpy(dh0, dh1, T * D * sizeof(float));
      memset(dctx, 0, sizeof(dctx));
      for (size_t t = 0u; t < T; ++t) {
        for (size_t o = 0u; o < D; ++o) {
          double acc = 0.0;
          for (size_t d = 0u; d < D; ++d) {
            dW_o[d * D + o] +=
                (float)((double)dh1[t * D + d] * (double)ctx[t * D + o]);
            acc += (double)dh1[t * D + d] * (double)W_o[d * D + o];
          }
          dctx[t * D + o] = (float)acc;
        }
      }

      /* Causal attention backward */
      memset(dQ_rot, 0, sizeof(dQ_rot));
      memset(dK_rot, 0, sizeof(dK_rot));
      memset(dV_in, 0, sizeof(dV_in));
      for (size_t h = 0u; h < H; ++h) {
        for (size_t t = 0u; t < T; ++t) {
          double dprob[ET_E2E_SEQ_LEN];
          double dot_p = 0.0;
          for (size_t s = 0u; s <= t; ++s) {
            double dp = 0.0;
            double p = attn_prob[(h * T + t) * T + s];
            for (size_t j = 0u; j < Dh; ++j) {
              double dc = (double)dctx[t * D + h * Dh + j];
              dp += dc * (double)V_in[(h * T + s) * Dh + j];
              dV_in[(h * T + s) * Dh + j] += (float)(p * dc);
            }
            dprob[s] = dp;
            dot_p += p * dp;
          }
          for (size_t s = 0u; s <= t; ++s) {
            double p = attn_prob[(h * T + t) * T + s];
            double dscore = p * (dprob[s] - dot_p) * inv_sqrt_dh;
            for (size_t j = 0u; j < Dh; ++j) {
              dQ_rot[(h * T + t) * Dh + j] +=
                  (float)(dscore * (double)K_rot[(h * T + s) * Dh + j]);
              dK_rot[(h * T + s) * Dh + j] +=
                  (float)(dscore * (double)Q_rot[(h * T + t) * Dh + j]);
            }
          }
        }
      }

      /* Native MOD5 RoPE backward on dQ_rot and dK_rot. */
      if (et_mod5_rope_backward_f32(1u, H, T, Dh, dQ_rot, positions,
                                    ET_MOD5_ROPE_DEFAULT_THETA,
                                    ET_MOD5_ROPE_DEFAULT_SCALE, dQ_in,
                                    &kerr) != 0) {
        free(probs_row);
        if (out_logits == NULL) {
          free(logits_buf);
        }
        return ET_E2E_ERR_STAGE3_MOD5;
      }
      if (et_mod5_rope_backward_f32(1u, H, T, Dh, dK_rot, positions,
                                    ET_MOD5_ROPE_DEFAULT_THETA,
                                    ET_MOD5_ROPE_DEFAULT_SCALE, dK_in,
                                    &kerr) != 0) {
        free(probs_row);
        if (out_logits == NULL) {
          free(logits_buf);
        }
        return ET_E2E_ERR_STAGE3_MOD5;
      }

      /* Q, K, V projections backward & embedding backward. */
      for (size_t t = 0u; t < T; ++t) {
        for (size_t h = 0u; h < H; ++h) {
          for (size_t j = 0u; j < Dh; ++j) {
            size_t o = h * Dh + j;
            double dq = (double)dQ_in[(h * T + t) * Dh + j];
            double dk = (double)dK_in[(h * T + t) * Dh + j];
            double dv = (double)dV_in[(h * T + t) * Dh + j];
            for (size_t d = 0u; d < D; ++d) {
              double xval = (double)h0[t * D + d];
              dW_q[o * D + d] += (float)(dq * xval);
              dW_k[o * D + d] += (float)(dk * xval);
              dW_v[o * D + d] += (float)(dv * xval);
              dh0[t * D + d] += (float)(dq * (double)W_q[o * D + d] +
                                        dk * (double)W_k[o * D + d] +
                                        dv * (double)W_v[o * D + d]);
            }
          }
        }
        for (size_t d = 0u; d < D; ++d) {
          dW_emb[(size_t)x_ids[t] * D + d] += dh0[t * D + d];
        }
      }
    }
    free(probs_row);
  }

  if (out_logits == NULL) {
    free(logits_buf);
  }
  return ET_E2E_OK;
}

static et_e2e_status create_stage3_adamw(const model_layout *L,
                                         const char *tokenizer_id,
                                         et_mod5_adamw_state **out_state) {
  et_mod5_adamw_config_v1 cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.param_count = (uint64_t)L->total_params;
  cfg.beta1 = 0.9f;
  cfg.beta2 = 0.999f;
  cfg.eps = 1.0e-8f;
  cfg.weight_decay = 0.01f;
  cfg.max_grad_norm = 1.0f;
  cfg.schedule.kind = ET_MOD5_LR_LINEAR_WARMUP_COSINE_DECAY;
  cfg.schedule.base_lr = 0.06;
  cfg.schedule.min_lr = 0.015;
  cfg.schedule.warmup_steps = 2u;
  cfg.schedule.total_steps = ET_E2E_PRETRAIN_STEPS;
  if (et_mod5_adamw_sha256_bytes((const uint8_t *)ET_E2E_MODEL_TAG,
                                 strlen(ET_E2E_MODEL_TAG),
                                 cfg.model_config_hash) != ET_MOD5_ADAMW_OK) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  cfg.tokenizer_id = tokenizer_id;
  if (et_mod5_adamw_create(&cfg, out_state) != ET_MOD5_ADAMW_OK) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  return ET_E2E_OK;
}

static et_e2e_status run_stage3_step(const model_layout *L,
                                     et_mod5_adamw_state *state,
                                     const int64_t *shard_tokens,
                                     size_t shard_count, float *params_scratch,
                                     float *grads_scratch,
                                     double *out_step_loss) {
  et_mod5_adamw_step_info_v1 info;
  et_mod5_dataset_cursor_v1 cursor;
  uint64_t rng_val = 0u;
  double total_loss = 0.0;
  size_t stride;

  if (shard_count < ET_E2E_SEQ_LEN + 1u) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  if (et_mod5_adamw_get_params(state, params_scratch, L->total_params) !=
      ET_MOD5_ADAMW_OK) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  memset(grads_scratch, 0, L->total_params * sizeof(float));

  stride = (shard_count - (ET_E2E_SEQ_LEN + 1u)) / ET_E2E_NUM_TRAIN_WINDOWS;
  if (stride == 0u) {
    stride = 1u;
  }

  for (size_t w = 0u; w < ET_E2E_NUM_TRAIN_WINDOWS; ++w) {
    uint32_t x_ids[ET_E2E_SEQ_LEN];
    uint32_t y_ids[ET_E2E_SEQ_LEN];
    double wloss = 0.0;
    size_t start = (w * stride) % (shard_count - ET_E2E_SEQ_LEN);
    for (size_t t = 0u; t < ET_E2E_SEQ_LEN; ++t) {
      x_ids[t] = (uint32_t)shard_tokens[start + t];
      y_ids[t] = (uint32_t)shard_tokens[start + t + 1u];
    }
    et_e2e_status st = model_forward_backward_window(
        L, params_scratch, x_ids, y_ids, ET_E2E_SEQ_LEN,
        1.0 / (double)(ET_E2E_NUM_TRAIN_WINDOWS * ET_E2E_SEQ_LEN), NULL, NULL,
        NULL, &wloss, grads_scratch);
    if (st != ET_E2E_OK) {
      return st;
    }
    total_loss += wloss;
  }
  total_loss /= (double)ET_E2E_NUM_TRAIN_WINDOWS;
  if (!isfinite(total_loss)) {
    return ET_E2E_ERR_NONFINITE;
  }

  if (et_mod5_adamw_step(state, grads_scratch, L->total_params,
                         (uint64_t)(ET_E2E_NUM_TRAIN_WINDOWS * ET_E2E_SEQ_LEN),
                         &info) != ET_MOD5_ADAMW_OK) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  if (et_mod5_adamw_rng_next_u64(state, &rng_val) != ET_MOD5_ADAMW_OK) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  (void)rng_val;
  if (et_mod5_adamw_get_cursor(state, &cursor) != ET_MOD5_ADAMW_OK) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  cursor.token_offset += (uint64_t)(ET_E2E_NUM_TRAIN_WINDOWS * ET_E2E_SEQ_LEN);
  if (cursor.token_offset >= (uint64_t)shard_count) {
    cursor.token_offset %= (uint64_t)shard_count;
    cursor.epoch += 1u;
  }
  if (et_mod5_adamw_set_cursor(state, &cursor) != ET_MOD5_ADAMW_OK) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }

  *out_step_loss = total_loss;
  return ET_E2E_OK;
}

static et_e2e_status ensure_tokenizer_and_shard_loaded(et_e2e_aot_ctx *ctx,
                                                       const char *work_dir) {
  char tok_path[ET_E2E_PATH_MAX];
  char shard_path[ET_E2E_PATH_MAX];
  et_t2_bpe_model *model = NULL;
  et_t2_bpe_info info;
  uint8_t sha[32];
  int64_t read_count = 0;

  if (ctx->vocab_size != 0u && ctx->shard_token_count > 0u) {
    return ET_E2E_OK;
  }
  if (!build_path(tok_path, sizeof(tok_path), work_dir,
                  "stage2_tokenizer.etbpe") ||
      !build_path(shard_path, sizeof(shard_path), work_dir,
                  "stage3_train.ets")) {
    return ET_E2E_ERR_ARGUMENT;
  }
  if (et_t2_bpe_load_file(tok_path, &model, sha) != ET_T2_BPE_OK) {
    return ET_E2E_ERR_STAGE2_BPE;
  }
  if (et_t2_bpe_get_info(model, &info) != ET_T2_BPE_OK) {
    et_t2_bpe_destroy(model);
    return ET_E2E_ERR_STAGE2_BPE;
  }
  et_t2_bpe_destroy(model);

  ctx->vocab_size = info.vocab_size;
  ctx->bos_id = info.bos_id;
  ctx->eos_id = info.eos_id;
  memcpy(ctx->m.s2_tokenizer_sha256, sha, 32u);
  format_tokenizer_id(sha, ctx->tokenizer_id);

  if (et_mod5_adamw_aot_read_d1_shard(
          shard_path, 0, (int64_t)ctx->vocab_size, ctx->tokenizer_id,
          ctx->shard_tokens, (int64_t)ET_E2E_MAX_SHARD_TOKENS,
          &read_count) != ET_MOD5_ADAMW_OK ||
      read_count < (int64_t)(ET_E2E_SEQ_LEN + 1u)) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  ctx->shard_token_count = (size_t)read_count;
  ctx->m.s3_shard_tokens = (uint64_t)read_count;
  return ET_E2E_OK;
}

static et_e2e_status ensure_stage3_weights_loaded(et_e2e_aot_ctx *ctx,
                                                  const char *work_dir) {
  char snap_path[ET_E2E_PATH_MAX];
  et_mod5_adamw_state *state = NULL;
  model_layout L;
  et_e2e_status st;

  st = ensure_tokenizer_and_shard_loaded(ctx, work_dir);
  if (st != ET_E2E_OK) {
    return st;
  }
  if (ctx->pretrain_weights != NULL && ctx->live_weights != NULL) {
    return ET_E2E_OK;
  }
  L = make_model_layout((size_t)ctx->vocab_size);
  if (!build_path(snap_path, sizeof(snap_path), work_dir,
                  "stage3_resumed.etsnap")) {
    return ET_E2E_ERR_ARGUMENT;
  }
  st = create_stage3_adamw(&L, ctx->tokenizer_id, &state);
  if (st != ET_E2E_OK) {
    return st;
  }
  if (et_mod5_adamw_load_snapshot(state, snap_path) != ET_MOD5_ADAMW_OK) {
    et_mod5_adamw_destroy(state);
    return ET_E2E_ERR_STAGE3_MOD5;
  }

  free(ctx->pretrain_weights);
  free(ctx->live_weights);
  ctx->pretrain_weights = (float *)calloc(L.total_params, sizeof(float));
  ctx->live_weights = (float *)calloc(L.total_params, sizeof(float));
  if (ctx->pretrain_weights == NULL || ctx->live_weights == NULL) {
    et_mod5_adamw_destroy(state);
    return ET_E2E_ERR_ALLOC;
  }
  if (et_mod5_adamw_get_params(state, ctx->pretrain_weights, L.total_params) !=
      ET_MOD5_ADAMW_OK) {
    et_mod5_adamw_destroy(state);
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  memcpy(ctx->live_weights, ctx->pretrain_weights,
         L.total_params * sizeof(float));
  ctx->param_count = L.total_params;
  ctx->m.s3_param_count = (uint64_t)L.total_params;
  et_mod5_adamw_destroy(state);
  return ET_E2E_OK;
}

/* ---- Context lifecycle ---- */

int32_t et_e2e_aot_create(et_e2e_aot_ctx **out_ctx) {
  et_e2e_aot_ctx *ctx;
  if (out_ctx == NULL) {
    return ET_E2E_ERR_ARGUMENT;
  }
  ctx = (et_e2e_aot_ctx *)calloc(1u, sizeof(*ctx));
  if (ctx == NULL) {
    return ET_E2E_ERR_ALLOC;
  }
  *out_ctx = ctx;
  return ET_E2E_OK;
}

void et_e2e_aot_destroy(et_e2e_aot_ctx *ctx) {
  if (ctx == NULL) {
    return;
  }
  free(ctx->pretrain_weights);
  free(ctx->live_weights);
  free(ctx);
}

/* ---- Stage 1: DATA4 Corpus Hygiene & Near-Dedup ---- */

int32_t et_e2e_aot_run_stage1_data4(et_e2e_aot_ctx *ctx, const char *work_dir) {
  et_data4_pipeline_config cfg;
  et_data4_document docs[11];
  et_data4_doc_result results[11];
  et_data4_pipeline_summary summary;
  size_t report_len = 0u;
  char *report_buf = NULL;
  uint8_t report_sha[32];
  const uint8_t *kept_ptrs[ET_E2E_MAX_DOCS];
  size_t kept_lens[ET_E2E_MAX_DOCS];
  size_t kept_idx = 0u;
  char corpus_path[ET_E2E_PATH_MAX];
  char report_path[ET_E2E_PATH_MAX];
  et_e2e_status st;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  if (!build_path(corpus_path, sizeof(corpus_path), work_dir,
                  "stage1_clean_corpus.bin") ||
      !build_path(report_path, sizeof(report_path), work_dir,
                  "stage1_report.tsv")) {
    return ET_E2E_ERR_ARGUMENT;
  }

  if (et_data4_config_default(&cfg) != ET_DATA4_OK) {
    return ET_E2E_ERR_STAGE1_DATA4;
  }
  cfg.min_bytes = 40u;
  cfg.max_bytes = 8192u;
  cfg.min_codepoints = 40u;
  cfg.max_codepoints = 8192u;
  cfg.min_printable_ratio = 0.90;
  cfg.min_alnum_ratio = 0.50;
  cfg.max_dup_line_ratio = 0.50;
  cfg.max_repeated_ngram_ratio = 0.50;
  cfg.rep_ngram_size = 16u;
  cfg.near_dup_enabled = 1u;
  cfg.dedup_ngram_size = 5u;
  cfg.minhash_num_perms = 64u;
  cfg.minhash_bands = 16u;
  cfg.minhash_seed = UINT64_C(0xE2E0123456789ABC);
  cfg.near_dup_jaccard_threshold = 0.65;

  docs[0] = (et_data4_document){"doc-00-train", (const uint8_t *)k_doc0,
                                strlen(k_doc0)};
  docs[1] = (et_data4_document){"doc-01-bpe", (const uint8_t *)k_doc1,
                                strlen(k_doc1)};
  docs[2] = (et_data4_document){"doc-02-exact-dup", (const uint8_t *)k_doc0,
                                strlen(k_doc0)};
  docs[3] = (et_data4_document){"doc-03-lora", (const uint8_t *)k_doc2,
                                strlen(k_doc2)};
  docs[4] = (et_data4_document){"doc-04-near-dup",
                                (const uint8_t *)k_doc_near_dup,
                                strlen(k_doc_near_dup)};
  docs[5] = (et_data4_document){"doc-05-bad-utf8", k_doc_bad_utf8,
                                sizeof(k_doc_bad_utf8)};
  docs[6] = (et_data4_document){"doc-06-dpo", (const uint8_t *)k_doc3,
                                strlen(k_doc3)};
  docs[7] = (et_data4_document){"doc-07-too-short",
                                (const uint8_t *)k_doc_too_short,
                                strlen(k_doc_too_short)};
  docs[8] = (et_data4_document){"doc-08-eval", (const uint8_t *)k_doc4,
                                strlen(k_doc4)};
  docs[9] = (et_data4_document){"doc-09-low-quality",
                                (const uint8_t *)k_doc_low_quality,
                                strlen(k_doc_low_quality)};
  docs[10] = (et_data4_document){"doc-10-spec-io5", (const uint8_t *)k_doc5,
                                 strlen(k_doc5)};

  memset(results, 0, sizeof(results));
  memset(&summary, 0, sizeof(summary));
  if (et_data4_pipeline_run(docs, 11u, &cfg, results, 11u, &summary) !=
      ET_DATA4_OK) {
    return ET_E2E_ERR_STAGE1_DATA4;
  }

  /* Verify exact expected kept/dropped disposition counts. */
  if (summary.input_documents != 11u || summary.kept_documents != 6u ||
      summary.duplicate_exact_documents != 1u ||
      summary.duplicate_near_documents != 1u ||
      summary.rejected_utf8_documents != 1u ||
      summary.rejected_length_documents != 1u ||
      summary.rejected_quality_documents != 1u ||
      summary.rejected_repetition_documents != 0u) {
    return ET_E2E_ERR_INVARIANT;
  }

  /* Format and deterministically verify provenance report. */
  if (et_data4_report_format(&cfg, &summary, results, 11u, NULL, 0u,
                             &report_len, report_sha) != ET_DATA4_OK ||
      report_len == 0u) {
    return ET_E2E_ERR_STAGE1_DATA4;
  }
  report_buf = (char *)calloc(1u, report_len + 1u);
  if (report_buf == NULL) {
    return ET_E2E_ERR_ALLOC;
  }
  if (et_data4_report_format(&cfg, &summary, results, 11u, report_buf,
                             report_len + 1u, &report_len,
                             report_sha) != ET_DATA4_OK ||
      et_data4_report_verify(report_buf, report_len, NULL, NULL, NULL, 0u) !=
          ET_DATA4_OK) {
    free(report_buf);
    return ET_E2E_ERR_STAGE1_DATA4;
  }
  st = write_file_atomic(report_path, (const uint8_t *)report_buf, report_len);
  free(report_buf);
  if (st != ET_E2E_OK) {
    return st;
  }

  for (size_t i = 0u; i < 11u; ++i) {
    if (results[i].disposition == ET_DATA4_DISP_KEPT) {
      kept_ptrs[kept_idx] = docs[i].bytes;
      kept_lens[kept_idx] = docs[i].byte_count;
      ++kept_idx;
    }
  }
  if (kept_idx != 6u) {
    return ET_E2E_ERR_INVARIANT;
  }
  st = save_clean_corpus_bundle(corpus_path, kept_ptrs, kept_lens, kept_idx,
                                ctx->clean_corpus_sha256);
  if (st != ET_E2E_OK) {
    return st;
  }

  ctx->m.s1_input_docs = summary.input_documents;
  ctx->m.s1_kept_docs = summary.kept_documents;
  ctx->m.s1_dropped_exact = summary.duplicate_exact_documents;
  ctx->m.s1_dropped_near = summary.duplicate_near_documents;
  ctx->m.s1_dropped_utf8 = summary.rejected_utf8_documents;
  ctx->m.s1_dropped_length = summary.rejected_length_documents;
  ctx->m.s1_dropped_quality = summary.rejected_quality_documents;
  ctx->m.s1_kept_bytes = summary.kept_bytes;
  memcpy(ctx->m.s1_report_sha256, report_sha, 32u);
  return ET_E2E_OK;
}

/* ---- Stage 2a: T2 Deterministic BPE Train & Atomic Save ---- */

int32_t et_e2e_aot_run_stage2_train_save(et_e2e_aot_ctx *ctx,
                                         const char *work_dir) {
  char corpus_path[ET_E2E_PATH_MAX];
  char tok_path[ET_E2E_PATH_MAX];
  owned_corpus corpus;
  et_t2_bpe_document bpe_docs[ET_E2E_MAX_DOCS];
  et_t2_bpe_train_config cfg;
  et_t2_bpe_model *model = NULL;
  et_t2_bpe_info info;
  uint8_t sha[32];
  et_e2e_status st;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  if (!build_path(corpus_path, sizeof(corpus_path), work_dir,
                  "stage1_clean_corpus.bin") ||
      !build_path(tok_path, sizeof(tok_path), work_dir,
                  "stage2_tokenizer.etbpe")) {
    return ET_E2E_ERR_ARGUMENT;
  }
  st = load_clean_corpus_bundle(corpus_path, &corpus, ctx->clean_corpus_sha256);
  if (st != ET_E2E_OK) {
    return st;
  }
  for (size_t i = 0u; i < corpus.doc_count; ++i) {
    bpe_docs[i].bytes = corpus.docs[i];
    bpe_docs[i].byte_count = corpus.lens[i];
  }

  cfg.target_vocab_size = ET_E2E_BPE_TARGET_VOCAB;
  cfg.min_frequency = 2u;
  cfg.special_mask = ET_T2_BPE_SPECIAL_BOS | ET_T2_BPE_SPECIAL_EOS;

  if (et_t2_bpe_train(bpe_docs, corpus.doc_count, &cfg, &model) !=
      ET_T2_BPE_OK) {
    owned_corpus_free(&corpus);
    return ET_E2E_ERR_STAGE2_BPE;
  }
  if (et_t2_bpe_get_info(model, &info) != ET_T2_BPE_OK ||
      et_t2_bpe_save_file(model, tok_path, sha) != ET_T2_BPE_OK) {
    et_t2_bpe_destroy(model);
    owned_corpus_free(&corpus);
    return ET_E2E_ERR_STAGE2_BPE;
  }
  et_t2_bpe_destroy(model);
  owned_corpus_free(&corpus);

  if (info.merge_count == 0u || info.bos_id == ET_T2_BPE_TOKEN_NONE ||
      info.eos_id == ET_T2_BPE_TOKEN_NONE) {
    return ET_E2E_ERR_INVARIANT;
  }

  ctx->vocab_size = info.vocab_size;
  ctx->bos_id = info.bos_id;
  ctx->eos_id = info.eos_id;
  format_tokenizer_id(sha, ctx->tokenizer_id);

  ctx->m.s2_vocab_size = info.vocab_size;
  ctx->m.s2_merge_count = info.merge_count;
  ctx->m.s2_bos_id = info.bos_id;
  ctx->m.s2_eos_id = info.eos_id;
  memcpy(ctx->m.s2_tokenizer_sha256, sha, 32u);
  return ET_E2E_OK;
}

/* ---- Stage 2b: Fresh-Process Reload, Lossless Encode/Decode & D1 Shard ---- */

int32_t et_e2e_aot_run_stage2_reload_encode(et_e2e_aot_ctx *ctx,
                                            const char *work_dir) {
  char corpus_path[ET_E2E_PATH_MAX];
  char tok_path[ET_E2E_PATH_MAX];
  char shard_path[ET_E2E_PATH_MAX];
  owned_corpus corpus;
  et_t2_bpe_model *model = NULL;
  et_t2_bpe_info info;
  uint8_t sha[32];
  uint64_t total_raw_bytes = 0u;
  uint64_t total_encoded_tokens = 0u;
  size_t shard_pos = 0u;
  int64_t read_count = 0;
  et_e2e_status st;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  if (!build_path(corpus_path, sizeof(corpus_path), work_dir,
                  "stage1_clean_corpus.bin") ||
      !build_path(tok_path, sizeof(tok_path), work_dir,
                  "stage2_tokenizer.etbpe") ||
      !build_path(shard_path, sizeof(shard_path), work_dir,
                  "stage3_train.ets")) {
    return ET_E2E_ERR_ARGUMENT;
  }

  st = load_clean_corpus_bundle(corpus_path, &corpus, ctx->clean_corpus_sha256);
  if (st != ET_E2E_OK) {
    return st;
  }
  if (et_t2_bpe_load_file(tok_path, &model, sha) != ET_T2_BPE_OK ||
      et_t2_bpe_get_info(model, &info) != ET_T2_BPE_OK) {
    et_t2_bpe_destroy(model);
    owned_corpus_free(&corpus);
    return ET_E2E_ERR_STAGE2_BPE;
  }

  for (size_t i = 0u; i < corpus.doc_count; ++i) {
    size_t n_toks = 0u;
    size_t n_dec = 0u;
    uint32_t *toks = NULL;
    uint8_t *dec = NULL;

    if (et_t2_bpe_encode(model, corpus.docs[i], corpus.lens[i], NULL, 0u,
                         &n_toks) != ET_T2_BPE_OK ||
        n_toks == 0u) {
      et_t2_bpe_destroy(model);
      owned_corpus_free(&corpus);
      return ET_E2E_ERR_STAGE2_BPE;
    }
    toks = (uint32_t *)calloc(n_toks, sizeof(uint32_t));
    if (toks == NULL) {
      et_t2_bpe_destroy(model);
      owned_corpus_free(&corpus);
      return ET_E2E_ERR_ALLOC;
    }
    if (et_t2_bpe_encode(model, corpus.docs[i], corpus.lens[i], toks, n_toks,
                         &n_toks) != ET_T2_BPE_OK ||
        et_t2_bpe_decode(model, toks, n_toks, NULL, 0u, &n_dec) !=
            ET_T2_BPE_OK ||
        n_dec != corpus.lens[i]) {
      free(toks);
      et_t2_bpe_destroy(model);
      owned_corpus_free(&corpus);
      return ET_E2E_ERR_STAGE2_BPE;
    }
    dec = (uint8_t *)calloc(1u, n_dec > 0u ? n_dec : 1u);
    if (dec == NULL) {
      free(toks);
      et_t2_bpe_destroy(model);
      owned_corpus_free(&corpus);
      return ET_E2E_ERR_ALLOC;
    }
    if (et_t2_bpe_decode(model, toks, n_toks, dec, n_dec, &n_dec) !=
            ET_T2_BPE_OK ||
        n_dec != corpus.lens[i] ||
        memcmp(dec, corpus.docs[i], corpus.lens[i]) != 0) {
      free(dec);
      free(toks);
      et_t2_bpe_destroy(model);
      owned_corpus_free(&corpus);
      return ET_E2E_ERR_INVARIANT;
    }

    total_raw_bytes += (uint64_t)corpus.lens[i];
    total_encoded_tokens += (uint64_t)n_toks;

    if (shard_pos + n_toks + 1u > ET_E2E_MAX_SHARD_TOKENS) {
      free(dec);
      free(toks);
      et_t2_bpe_destroy(model);
      owned_corpus_free(&corpus);
      return ET_E2E_ERR_STAGE2_BPE;
    }
    for (size_t k = 0u; k < n_toks; ++k) {
      ctx->shard_tokens[shard_pos++] = (int64_t)toks[k];
    }
    ctx->shard_tokens[shard_pos++] = (int64_t)info.eos_id;

    free(dec);
    free(toks);
  }
  et_t2_bpe_destroy(model);
  owned_corpus_free(&corpus);

  double compression = (double)total_raw_bytes / (double)total_encoded_tokens;
  if (!(compression > 1.5)) {
    return ET_E2E_ERR_INVARIANT;
  }

  ctx->vocab_size = info.vocab_size;
  ctx->bos_id = info.bos_id;
  ctx->eos_id = info.eos_id;
  format_tokenizer_id(sha, ctx->tokenizer_id);

  /* Write validated D1 v1.0 .ets token shard and verify immediate readback. */
  if (et_mod5_adamw_aot_write_d1_shard(shard_path, 0, (int64_t)info.vocab_size,
                                       ctx->tokenizer_id, ctx->shard_tokens,
                                       (int64_t)shard_pos) !=
      ET_MOD5_ADAMW_OK) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  if (et_mod5_adamw_aot_read_d1_shard(
          shard_path, 0, (int64_t)info.vocab_size, ctx->tokenizer_id,
          ctx->shard_tokens, (int64_t)ET_E2E_MAX_SHARD_TOKENS,
          &read_count) != ET_MOD5_ADAMW_OK ||
      (size_t)read_count != shard_pos) {
    return ET_E2E_ERR_STAGE3_MOD5;
  }
  ctx->shard_token_count = shard_pos;

  ctx->m.s2_vocab_size = info.vocab_size;
  ctx->m.s2_merge_count = info.merge_count;
  ctx->m.s2_bos_id = info.bos_id;
  ctx->m.s2_eos_id = info.eos_id;
  ctx->m.s2_raw_bytes = total_raw_bytes;
  ctx->m.s2_encoded_tokens = total_encoded_tokens;
  ctx->m.s2_compression_ratio = compression;
  ctx->m.s2_lossless_verified = 1u;
  ctx->m.s2_reload_sha_match = 1u;
  memcpy(ctx->m.s2_tokenizer_sha256, sha, 32u);
  ctx->m.s3_shard_tokens = (uint64_t)shard_pos;
  return ET_E2E_OK;
}

/* ---- Stage 3a: Uninterrupted 8-step Reference + Split Part 1 (4 steps) ---- */

int32_t et_e2e_aot_run_stage3_pretrain_part1(et_e2e_aot_ctx *ctx,
                                             const char *work_dir) {
  char full_path[ET_E2E_PATH_MAX];
  char mid_path[ET_E2E_PATH_MAX];
  model_layout L;
  et_mod5_adamw_state *full_state = NULL;
  et_mod5_adamw_state *split_state = NULL;
  float *init_params = NULL;
  float *params_scratch = NULL;
  float *grads_scratch = NULL;
  static const uint64_t init_rng[4] = {
      UINT64_C(0x1122334455667788), UINT64_C(0x99AABBCCDDEEFF01),
      UINT64_C(0x0123456789ABCDEF), UINT64_C(0xFEDCBA9876543210)};
  et_mod5_dataset_cursor_v1 init_cursor = {0u, 0u, 0u};
  double prev_loss = 1e300;
  double step_loss = 0.0;
  et_e2e_status st;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  st = ensure_tokenizer_and_shard_loaded(ctx, work_dir);
  if (st != ET_E2E_OK) {
    return st;
  }
  if (!build_path(full_path, sizeof(full_path), work_dir,
                  "stage3_full.etsnap") ||
      !build_path(mid_path, sizeof(mid_path), work_dir,
                  "stage3_mid.etsnap")) {
    return ET_E2E_ERR_ARGUMENT;
  }

  L = make_model_layout((size_t)ctx->vocab_size);
  init_params = (float *)calloc(L.total_params, sizeof(float));
  params_scratch = (float *)calloc(L.total_params, sizeof(float));
  grads_scratch = (float *)calloc(L.total_params, sizeof(float));
  if (init_params == NULL || params_scratch == NULL || grads_scratch == NULL) {
    free(init_params);
    free(params_scratch);
    free(grads_scratch);
    return ET_E2E_ERR_ALLOC;
  }
  init_model_weights(&L, init_params);

  /* 1. Uninterrupted 8-step reference run. */
  st = create_stage3_adamw(&L, ctx->tokenizer_id, &full_state);
  if (st != ET_E2E_OK) {
    goto cleanup;
  }
  if (et_mod5_adamw_set_params(full_state, init_params, L.total_params) !=
          ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_set_rng(full_state, init_rng) != ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_set_cursor(full_state, &init_cursor) != ET_MOD5_ADAMW_OK) {
    st = ET_E2E_ERR_STAGE3_MOD5;
    goto cleanup;
  }
  for (uint32_t s = 0u; s < ET_E2E_PRETRAIN_STEPS; ++s) {
    st = run_stage3_step(&L, full_state, ctx->shard_tokens,
                         ctx->shard_token_count, params_scratch, grads_scratch,
                         &step_loss);
    if (st != ET_E2E_OK) {
      goto cleanup;
    }
    if (s == 0u) {
      ctx->m.s3_loss_step1 = step_loss;
    } else if (!(step_loss < prev_loss)) {
      st = ET_E2E_ERR_INVARIANT;
      goto cleanup;
    }
    if (s + 1u == ET_E2E_PRETRAIN_SPLIT_STEP) {
      ctx->m.s3_loss_step4 = step_loss;
    }
    if (s + 1u == ET_E2E_PRETRAIN_STEPS) {
      ctx->m.s3_loss_step8 = step_loss;
    }
    prev_loss = step_loss;
  }
  if (et_mod5_adamw_save_snapshot(full_state, full_path) != ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_sha256_file(full_path, ctx->m.s3_full_snap_sha256) !=
          ET_MOD5_ADAMW_OK) {
    st = ET_E2E_ERR_STAGE3_MOD5;
    goto cleanup;
  }

  /* 2. Split Part 1: 4 steps -> save midpoint snapshot ETM5TRN1. */
  st = create_stage3_adamw(&L, ctx->tokenizer_id, &split_state);
  if (st != ET_E2E_OK) {
    goto cleanup;
  }
  if (et_mod5_adamw_set_params(split_state, init_params, L.total_params) !=
          ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_set_rng(split_state, init_rng) != ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_set_cursor(split_state, &init_cursor) != ET_MOD5_ADAMW_OK) {
    st = ET_E2E_ERR_STAGE3_MOD5;
    goto cleanup;
  }
  for (uint32_t s = 0u; s < ET_E2E_PRETRAIN_SPLIT_STEP; ++s) {
    st = run_stage3_step(&L, split_state, ctx->shard_tokens,
                         ctx->shard_token_count, params_scratch, grads_scratch,
                         &step_loss);
    if (st != ET_E2E_OK) {
      goto cleanup;
    }
  }
  if (et_mod5_adamw_save_snapshot(split_state, mid_path) != ET_MOD5_ADAMW_OK) {
    st = ET_E2E_ERR_STAGE3_MOD5;
    goto cleanup;
  }

  ctx->param_count = L.total_params;
  ctx->m.s3_param_count = (uint64_t)L.total_params;
  ctx->m.s3_steps_full = ET_E2E_PRETRAIN_STEPS;
  ctx->m.s3_steps_split = ET_E2E_PRETRAIN_SPLIT_STEP;
  st = ET_E2E_OK;

cleanup:
  et_mod5_adamw_destroy(full_state);
  et_mod5_adamw_destroy(split_state);
  free(init_params);
  free(params_scratch);
  free(grads_scratch);
  return st;
}

/* ---- Stage 3b: Fresh-Process Snapshot Resume + Bitwise Parity Gate ---- */

int32_t et_e2e_aot_run_stage3_pretrain_resume(et_e2e_aot_ctx *ctx,
                                              const char *work_dir) {
  char full_path[ET_E2E_PATH_MAX];
  char mid_path[ET_E2E_PATH_MAX];
  char corrupt_path[ET_E2E_PATH_MAX];
  char resumed_path[ET_E2E_PATH_MAX];
  model_layout L;
  et_mod5_adamw_state *resumed_state = NULL;
  et_mod5_adamw_state *full_state = NULL;
  float *params_scratch = NULL;
  float *grads_scratch = NULL;
  float *full_params = NULL;
  float *resumed_params = NULL;
  uint8_t *mid_bytes = NULL;
  size_t mid_len = 0u;
  uint8_t full_sha[32];
  uint8_t resumed_sha[32];
  double prev_loss = 1e300;
  double step_loss = 0.0;
  et_e2e_status st;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  st = ensure_tokenizer_and_shard_loaded(ctx, work_dir);
  if (st != ET_E2E_OK) {
    return st;
  }
  if (!build_path(full_path, sizeof(full_path), work_dir,
                  "stage3_full.etsnap") ||
      !build_path(mid_path, sizeof(mid_path), work_dir, "stage3_mid.etsnap") ||
      !build_path(corrupt_path, sizeof(corrupt_path), work_dir,
                  "stage3_corrupt.etsnap") ||
      !build_path(resumed_path, sizeof(resumed_path), work_dir,
                  "stage3_resumed.etsnap")) {
    return ET_E2E_ERR_ARGUMENT;
  }

  L = make_model_layout((size_t)ctx->vocab_size);
  params_scratch = (float *)calloc(L.total_params, sizeof(float));
  grads_scratch = (float *)calloc(L.total_params, sizeof(float));
  full_params = (float *)calloc(L.total_params, sizeof(float));
  resumed_params = (float *)calloc(L.total_params, sizeof(float));
  if (params_scratch == NULL || grads_scratch == NULL || full_params == NULL ||
      resumed_params == NULL) {
    st = ET_E2E_ERR_ALLOC;
    goto cleanup;
  }

  st = create_stage3_adamw(&L, ctx->tokenizer_id, &resumed_state);
  if (st != ET_E2E_OK) {
    goto cleanup;
  }
  if (et_mod5_adamw_load_snapshot(resumed_state, mid_path) !=
          ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_get_step(resumed_state) != ET_E2E_PRETRAIN_SPLIT_STEP) {
    st = ET_E2E_ERR_STAGE3_MOD5;
    goto cleanup;
  }

  /* Negative check: 1-bit corrupted snapshot MUST fail with CHECKSUM_MISMATCH
   * and leave live state untouched. */
  st = read_file_nofollow(mid_path, 67108864u, &mid_bytes, &mid_len);
  if (st != ET_E2E_OK || mid_len <= 260u) {
    st = ET_E2E_ERR_STAGE3_MOD5;
    goto cleanup;
  }
  mid_bytes[260] ^= 0x01u;
  st = write_file_atomic(corrupt_path, mid_bytes, mid_len);
  free(mid_bytes);
  mid_bytes = NULL;
  if (st != ET_E2E_OK) {
    goto cleanup;
  }
  if (et_mod5_adamw_load_snapshot(resumed_state, corrupt_path) !=
      ET_MOD5_ADAMW_ERR_CHECKSUM_MISMATCH) {
    st = ET_E2E_ERR_INVARIANT;
    goto cleanup;
  }
  (void)unlink(corrupt_path);
  ctx->m.s3_corrupt_rejected = 1u;

  /* Finish steps 5..8 on resumed state. */
  for (uint32_t s = ET_E2E_PRETRAIN_SPLIT_STEP; s < ET_E2E_PRETRAIN_STEPS;
       ++s) {
    st = run_stage3_step(&L, resumed_state, ctx->shard_tokens,
                         ctx->shard_token_count, params_scratch, grads_scratch,
                         &step_loss);
    if (st != ET_E2E_OK) {
      goto cleanup;
    }
    if (s > ET_E2E_PRETRAIN_SPLIT_STEP && !(step_loss < prev_loss)) {
      st = ET_E2E_ERR_INVARIANT;
      goto cleanup;
    }
    prev_loss = step_loss;
  }
  ctx->m.s3_loss_step8 = step_loss;

  if (et_mod5_adamw_save_snapshot(resumed_state, resumed_path) !=
      ET_MOD5_ADAMW_OK) {
    st = ET_E2E_ERR_STAGE3_MOD5;
    goto cleanup;
  }

  /* Load full uninterrupted snapshot and verify bitwise weight & SHA parity. */
  st = create_stage3_adamw(&L, ctx->tokenizer_id, &full_state);
  if (st != ET_E2E_OK) {
    goto cleanup;
  }
  if (et_mod5_adamw_load_snapshot(full_state, full_path) != ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_get_params(full_state, full_params, L.total_params) !=
          ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_get_params(resumed_state, resumed_params, L.total_params) !=
          ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_sha256_file(full_path, full_sha) != ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_sha256_file(resumed_path, resumed_sha) !=
          ET_MOD5_ADAMW_OK) {
    st = ET_E2E_ERR_STAGE3_MOD5;
    goto cleanup;
  }

  if (memcmp(full_params, resumed_params, L.total_params * sizeof(float)) !=
          0 ||
      memcmp(full_sha, resumed_sha, 32u) != 0) {
    st = ET_E2E_ERR_INVARIANT;
    goto cleanup;
  }

  free(ctx->pretrain_weights);
  free(ctx->live_weights);
  ctx->pretrain_weights = resumed_params;
  resumed_params = NULL;
  ctx->live_weights = (float *)calloc(L.total_params, sizeof(float));
  if (ctx->live_weights == NULL) {
    st = ET_E2E_ERR_ALLOC;
    goto cleanup;
  }
  memcpy(ctx->live_weights, ctx->pretrain_weights,
         L.total_params * sizeof(float));

  ctx->param_count = L.total_params;
  ctx->m.s3_param_count = (uint64_t)L.total_params;
  ctx->m.s3_steps_full = ET_E2E_PRETRAIN_STEPS;
  ctx->m.s3_steps_split = ET_E2E_PRETRAIN_STEPS;
  ctx->m.s3_weights_bitwise_equal = 1u;
  ctx->m.s3_snapshot_sha_equal = 1u;
  memcpy(ctx->m.s3_full_snap_sha256, full_sha, 32u);
  memcpy(ctx->m.s3_resumed_snap_sha256, resumed_sha, 32u);
  st = ET_E2E_OK;

cleanup:
  et_mod5_adamw_destroy(resumed_state);
  et_mod5_adamw_destroy(full_state);
  free(mid_bytes);
  free(params_scratch);
  free(grads_scratch);
  free(full_params);
  free(resumed_params);
  return st;
}

/* ---- Stage 4a: LoRA Adapter Fine-Tuning & Frozen Base Invariance ---- */

int32_t et_e2e_aot_run_stage4_lora_train_save(et_e2e_aot_ctx *ctx,
                                              const char *work_dir) {
  char ckpt_path[ET_E2E_PATH_MAX];
  model_layout L;
  et_lora_config cfg;
  float x_in[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
  float w_base[ET_E2E_MODEL_FF_DIM * ET_E2E_MODEL_DIM];
  float w_base_orig[ET_E2E_MODEL_FF_DIM * ET_E2E_MODEL_DIM];
  float w_merged[ET_E2E_MODEL_FF_DIM * ET_E2E_MODEL_DIM];
  float w_restored[ET_E2E_MODEL_FF_DIM * ET_E2E_MODEL_DIM];
  float a[ET_E2E_LORA_RANK * ET_E2E_MODEL_DIM];
  float b[ET_E2E_MODEL_FF_DIM * ET_E2E_LORA_RANK];
  float da[ET_E2E_LORA_RANK * ET_E2E_MODEL_DIM];
  float db[ET_E2E_MODEL_FF_DIM * ET_E2E_LORA_RANK];
  float dx[ET_E2E_SEQ_LEN * ET_E2E_MODEL_DIM];
  float y_base[ET_E2E_SEQ_LEN * ET_E2E_MODEL_FF_DIM];
  float y_target[ET_E2E_SEQ_LEN * ET_E2E_MODEL_FF_DIM];
  float y_out[ET_E2E_SEQ_LEN * ET_E2E_MODEL_FF_DIM];
  float y_merged[ET_E2E_SEQ_LEN * ET_E2E_MODEL_FF_DIM];
  float dy[ET_E2E_SEQ_LEN * ET_E2E_MODEL_FF_DIM];
  uint32_t x_ids[ET_E2E_SEQ_LEN];
  uint8_t base_sha_before[32];
  uint8_t base_sha_after[32];
  uint8_t ckpt_sha[32];
  double prev_loss = 1e300;
  double max_merge_err = 0.0;
  double max_unmerge_err = 0.0;
  et_e2e_status st;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  st = ensure_stage3_weights_loaded(ctx, work_dir);
  if (st != ET_E2E_OK) {
    return st;
  }
  if (!build_path(ckpt_path, sizeof(ckpt_path), work_dir, "stage4_lora.etlr")) {
    return ET_E2E_ERR_ARGUMENT;
  }

  L = make_model_layout((size_t)ctx->vocab_size);
  cfg.m = ET_E2E_SEQ_LEN;
  cfg.k = ET_E2E_MODEL_DIM;
  cfg.n = ET_E2E_MODEL_FF_DIM;
  cfg.r = ET_E2E_LORA_RANK;
  cfg.alpha = 8.0f;

  /* Extract real hidden state h1 from Stage 3 model + real pretrained W_ff1. */
  for (size_t t = 0u; t < ET_E2E_SEQ_LEN; ++t) {
    x_ids[t] = (uint32_t)ctx->shard_tokens[t];
  }
  st = model_forward_backward_window(&L, ctx->pretrain_weights, x_ids, NULL,
                                     ET_E2E_SEQ_LEN, 1.0, x_in, NULL, NULL,
                                     NULL, NULL);
  if (st != ET_E2E_OK) {
    return st;
  }
  memcpy(w_base, ctx->pretrain_weights + L.off_wff1, sizeof(w_base));
  memcpy(w_base_orig, w_base, sizeof(w_base_orig));

  if (et_lora_base_weight_sha256(cfg.k, cfg.n, w_base, base_sha_before) !=
          ET_LORA_OK ||
      et_lora_init_adapter(&cfg, UINT64_C(4242), a, b) != ET_LORA_OK ||
      et_lora_base_forward(&cfg, x_in, w_base, y_base) != ET_LORA_OK) {
    return ET_E2E_ERR_STAGE4_LORA;
  }

  /* Construct deterministic adaptation target = base + structured offset. */
  uint64_t rng = UINT64_C(0x4A00CAFE12345678);
  for (size_t i = 0u; i < (size_t)(cfg.m * cfg.n); ++i) {
    y_target[i] = y_base[i] + det_uniform_f32(&rng, 0.20);
  }

  for (uint32_t step = 0u; step <= ET_E2E_LORA_STEPS; ++step) {
    double mse = 0.0;
    if (et_lora_forward(&cfg, x_in, w_base, a, b, y_out) != ET_LORA_OK) {
      return ET_E2E_ERR_STAGE4_LORA;
    }
    for (size_t i = 0u; i < (size_t)(cfg.m * cfg.n); ++i) {
      double diff = (double)y_out[i] - (double)y_target[i];
      mse += diff * diff;
      dy[i] = (float)((2.0 / (double)(cfg.m * cfg.n)) * diff);
    }
    mse /= (double)(cfg.m * cfg.n);
    if (!isfinite(mse)) {
      return ET_E2E_ERR_NONFINITE;
    }
    if (step == 0u) {
      ctx->m.s4_initial_loss = mse;
    } else if (!(mse < prev_loss)) {
      return ET_E2E_ERR_INVARIANT;
    }
    prev_loss = mse;
    if (step == ET_E2E_LORA_STEPS) {
      ctx->m.s4_final_loss = mse;
      break;
    }

    if (et_lora_backward(&cfg, x_in, w_base, a, b, dy, dx, da, db) !=
        ET_LORA_OK) {
      return ET_E2E_ERR_STAGE4_LORA;
    }
    const float lr = 0.35f;
    for (size_t i = 0u; i < (size_t)(cfg.r * cfg.k); ++i) {
      a[i] -= lr * da[i];
    }
    for (size_t i = 0u; i < (size_t)(cfg.n * cfg.r); ++i) {
      b[i] -= lr * db[i];
    }
  }

  /* Verify exact frozen base weight SHA-256 invariance. */
  if (et_lora_base_weight_sha256(cfg.k, cfg.n, w_base, base_sha_after) !=
          ET_LORA_OK ||
      memcmp(base_sha_before, base_sha_after, 32u) != 0 ||
      memcmp(w_base, w_base_orig, sizeof(w_base)) != 0) {
    return ET_E2E_ERR_INVARIANT;
  }

  /* Verify merge/unmerge parity <= 1e-6. */
  if (et_lora_merge(&cfg, w_base, a, b, w_merged) != ET_LORA_OK ||
      et_lora_base_forward(&cfg, x_in, w_merged, y_merged) != ET_LORA_OK ||
      et_lora_unmerge(&cfg, w_merged, a, b, w_restored) != ET_LORA_OK) {
    return ET_E2E_ERR_STAGE4_LORA;
  }
  for (size_t i = 0u; i < (size_t)(cfg.m * cfg.n); ++i) {
    double err = fabs((double)y_out[i] - (double)y_merged[i]);
    if (err > max_merge_err) {
      max_merge_err = err;
    }
  }
  for (size_t i = 0u; i < (size_t)(cfg.n * cfg.k); ++i) {
    double err = fabs((double)w_base[i] - (double)w_restored[i]);
    if (err > max_unmerge_err) {
      max_unmerge_err = err;
    }
  }
  if (!(max_merge_err <= 1.0e-6) || !(max_unmerge_err <= 1.0e-6)) {
    return ET_E2E_ERR_INVARIANT;
  }

  /* Save ETLR1000 adapter checkpoint atomically. */
  if (et_lora_checkpoint_save(ckpt_path, &cfg, w_base, a, b) != ET_LORA_OK ||
      et_mod5_adamw_sha256_file(ckpt_path, ckpt_sha) != ET_MOD5_ADAMW_OK) {
    return ET_E2E_ERR_STAGE4_LORA;
  }

  ctx->m.s4_rank = cfg.r;
  ctx->m.s4_steps = ET_E2E_LORA_STEPS;
  ctx->m.s4_merge_max_abs_err = max_merge_err;
  ctx->m.s4_unmerge_max_abs_err = max_unmerge_err;
  ctx->m.s4_base_frozen_equal = 1u;
  memcpy(ctx->m.s4_base_sha256, base_sha_after, 32u);
  memcpy(ctx->m.s4_ckpt_sha256, ckpt_sha, 32u);
  return ET_E2E_OK;
}

/* ---- Stage 4b: Fresh-Process LoRA Checkpoint Reload & Merge ---- */

int32_t et_e2e_aot_run_stage4_lora_reload(et_e2e_aot_ctx *ctx,
                                          const char *work_dir) {
  char ckpt_path[ET_E2E_PATH_MAX];
  model_layout L;
  et_lora_config cfg;
  et_lora_checkpoint_meta meta;
  float w_base[ET_E2E_MODEL_FF_DIM * ET_E2E_MODEL_DIM];
  float w_wrong[ET_E2E_MODEL_FF_DIM * ET_E2E_MODEL_DIM];
  float w_merged[ET_E2E_MODEL_FF_DIM * ET_E2E_MODEL_DIM];
  float a_loaded[ET_E2E_LORA_RANK * ET_E2E_MODEL_DIM];
  float b_loaded[ET_E2E_MODEL_FF_DIM * ET_E2E_LORA_RANK];
  uint8_t base_sha[32];
  uint8_t ckpt_sha[32];
  et_e2e_status st;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  st = ensure_stage3_weights_loaded(ctx, work_dir);
  if (st != ET_E2E_OK) {
    return st;
  }
  if (!build_path(ckpt_path, sizeof(ckpt_path), work_dir, "stage4_lora.etlr")) {
    return ET_E2E_ERR_ARGUMENT;
  }

  L = make_model_layout((size_t)ctx->vocab_size);
  cfg.m = ET_E2E_SEQ_LEN;
  cfg.k = ET_E2E_MODEL_DIM;
  cfg.n = ET_E2E_MODEL_FF_DIM;
  cfg.r = ET_E2E_LORA_RANK;
  cfg.alpha = 8.0f;

  memcpy(w_base, ctx->pretrain_weights + L.off_wff1, sizeof(w_base));
  if (et_lora_base_weight_sha256(cfg.k, cfg.n, w_base, base_sha) !=
      ET_LORA_OK) {
    return ET_E2E_ERR_STAGE4_LORA;
  }

  /* Negative check: mismatched base weight MUST fail with ET_LORA_ERR_BASE_HASH. */
  memcpy(w_wrong, w_base, sizeof(w_wrong));
  w_wrong[0] += 0.125f;
  if (et_lora_checkpoint_load(ckpt_path, cfg.k, cfg.n, cfg.r, w_wrong, &meta,
                              a_loaded, b_loaded) != ET_LORA_ERR_BASE_HASH) {
    return ET_E2E_ERR_INVARIANT;
  }

  /* Positive reload in fresh process. */
  memset(&meta, 0, sizeof(meta));
  if (et_lora_checkpoint_load(ckpt_path, cfg.k, cfg.n, cfg.r, w_base, &meta,
                              a_loaded, b_loaded) != ET_LORA_OK ||
      et_mod5_adamw_sha256_file(ckpt_path, ckpt_sha) != ET_MOD5_ADAMW_OK) {
    return ET_E2E_ERR_STAGE4_LORA;
  }
  if (memcmp(meta.base_weight_sha256, base_sha, 32u) != 0 ||
      meta.r != cfg.r || meta.k != cfg.k || meta.n != cfg.n) {
    return ET_E2E_ERR_INVARIANT;
  }

  /* Merge loaded LoRA adapter into live model's W_ff1 for Stages 5-8. */
  if (et_lora_merge(&cfg, w_base, a_loaded, b_loaded, w_merged) != ET_LORA_OK) {
    return ET_E2E_ERR_STAGE4_LORA;
  }
  memcpy(ctx->live_weights + L.off_wff1, w_merged, sizeof(w_merged));

  ctx->m.s4_rank = meta.r;
  ctx->m.s4_reload_equal = 1u;
  memcpy(ctx->m.s4_base_sha256, base_sha, 32u);
  memcpy(ctx->m.s4_ckpt_sha256, ckpt_sha, 32u);
  return ET_E2E_OK;
}

/* ---- Stage 5: DPO Preference Alignment ---- */

int32_t et_e2e_aot_run_stage5_dpo(et_e2e_aot_ctx *ctx, const char *work_dir) {
  model_layout L;
  et_dpo_config cfg;
  const size_t B = ET_E2E_DPO_BATCH;
  const size_t T = ET_E2E_DPO_SEQ;
  const size_t Vd = ET_E2E_DPO_VOCAB;
  const size_t D = ET_E2E_MODEL_DIM;

  uint32_t active_vocab[ET_E2E_DPO_VOCAB];
  float w_ref[ET_E2E_DPO_VOCAB * ET_E2E_MODEL_DIM];
  float w_ref_orig[ET_E2E_DPO_VOCAB * ET_E2E_MODEL_DIM];
  float w_policy[ET_E2E_DPO_VOCAB * ET_E2E_MODEL_DIM];
  float dw_policy[ET_E2E_DPO_VOCAB * ET_E2E_MODEL_DIM];
  float h_chosen[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ * ET_E2E_MODEL_DIM];
  float h_rejected[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ * ET_E2E_MODEL_DIM];
  float ref_chosen[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ * ET_E2E_DPO_VOCAB];
  float ref_rejected[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ * ET_E2E_DPO_VOCAB];
  float pol_chosen[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ * ET_E2E_DPO_VOCAB];
  float pol_rejected[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ * ET_E2E_DPO_VOCAB];
  float d_chosen[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ * ET_E2E_DPO_VOCAB];
  float d_rejected[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ * ET_E2E_DPO_VOCAB];
  int64_t chosen_targets[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ];
  int64_t rejected_targets[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ];
  uint8_t chosen_mask[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ];
  uint8_t rejected_mask[ET_E2E_DPO_BATCH * ET_E2E_DPO_SEQ];
  uint8_t ref_sha_before[32];
  uint8_t ref_sha_after[32];
  double prev_loss = 1e300;
  double prev_margin = -1e300;
  et_e2e_status st;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  st = ensure_stage3_weights_loaded(ctx, work_dir);
  if (st != ET_E2E_OK) {
    return st;
  }
  L = make_model_layout((size_t)ctx->vocab_size);

  /* Select 64 distinct active token rows from W_lm (covering merged + byte IDs). */
  for (size_t i = 0u; i < Vd; ++i) {
    active_vocab[i] = (uint32_t)((i * 5u + 32u) % ctx->vocab_size);
    /* Ensure uniqueness even if vocab_size shares a factor with 5. */
    for (size_t j = 0u; j < i; ++j) {
      if (active_vocab[j] == active_vocab[i]) {
        active_vocab[i] = (active_vocab[i] + 1u) % ctx->vocab_size;
        j = (size_t)-1;
      }
    }
    memcpy(&w_ref[i * D],
           ctx->live_weights + L.off_wlm + (size_t)active_vocab[i] * D,
           D * sizeof(float));
  }
  memcpy(w_ref_orig, w_ref, sizeof(w_ref_orig));
  memcpy(w_policy, w_ref, sizeof(w_policy));

  /* Extract real hidden states h2 from post-LoRA trunk on distinct token pairs. */
  for (size_t b = 0u; b < B; ++b) {
    uint32_t seq_w[ET_E2E_DPO_SEQ];
    uint32_t seq_l[ET_E2E_DPO_SEQ];
    for (size_t t = 0u; t < T; ++t) {
      size_t iw = (b * 11u + t) % ctx->shard_token_count;
      size_t il = (b * 17u + t + 23u) % ctx->shard_token_count;
      seq_w[t] = (uint32_t)ctx->shard_tokens[iw];
      seq_l[t] = (uint32_t)ctx->shard_tokens[il];
      chosen_targets[b * T + t] = (int64_t)((b * 9u + t * 3u + 1u) % Vd);
      rejected_targets[b * T + t] = (int64_t)((b * 9u + t * 3u + 33u) % Vd);
      chosen_mask[b * T + t] = (t == 0u) ? 0u : 1u;
      rejected_mask[b * T + t] = (t == 0u) ? 0u : 1u;
    }
    st = model_forward_backward_window(&L, ctx->live_weights, seq_w, NULL, T,
                                       1.0, NULL, &h_chosen[b * T * D], NULL,
                                       NULL, NULL);
    if (st != ET_E2E_OK) {
      return st;
    }
    st = model_forward_backward_window(&L, ctx->live_weights, seq_l, NULL, T,
                                       1.0, NULL, &h_rejected[b * T * D], NULL,
                                       NULL, NULL);
    if (st != ET_E2E_OK) {
      return st;
    }
  }

  /* Compute frozen reference logits once and hash W_ref. */
  for (size_t bt = 0u; bt < B * T; ++bt) {
    for (size_t v = 0u; v < Vd; ++v) {
      double acc_w = 0.0;
      double acc_l = 0.0;
      for (size_t d = 0u; d < D; ++d) {
        acc_w += (double)h_chosen[bt * D + d] * (double)w_ref[v * D + d];
        acc_l += (double)h_rejected[bt * D + d] * (double)w_ref[v * D + d];
      }
      ref_chosen[bt * Vd + v] = (float)acc_w;
      ref_rejected[bt * Vd + v] = (float)acc_l;
    }
  }
  if (et_dpo_weight_sha256(Vd * D, w_ref, ref_sha_before) != ET_DPO_OK) {
    return ET_E2E_ERR_STAGE5_DPO;
  }

  cfg.batch_size = (uint32_t)B;
  cfg.seq_len = (uint32_t)T;
  cfg.vocab_size = (uint32_t)Vd;
  cfg.reference_free = 0u;
  cfg.beta = 0.25;
  cfg.label_smoothing = 0.0;

  for (uint32_t step = 0u; step <= ET_E2E_DPO_STEPS; ++step) {
    et_dpo_telemetry tel;
    memset(&tel, 0, sizeof(tel));

    for (size_t bt = 0u; bt < B * T; ++bt) {
      for (size_t v = 0u; v < Vd; ++v) {
        double acc_w = 0.0;
        double acc_l = 0.0;
        for (size_t d = 0u; d < D; ++d) {
          acc_w += (double)h_chosen[bt * D + d] * (double)w_policy[v * D + d];
          acc_l += (double)h_rejected[bt * D + d] * (double)w_policy[v * D + d];
        }
        pol_chosen[bt * Vd + v] = (float)acc_w;
        pol_rejected[bt * Vd + v] = (float)acc_l;
      }
    }

    if (et_dpo_forward_backward(&cfg, pol_chosen, pol_rejected, ref_chosen,
                                ref_rejected, chosen_targets, rejected_targets,
                                chosen_mask, rejected_mask, &tel, d_chosen,
                                d_rejected) != ET_DPO_OK) {
      return ET_E2E_ERR_STAGE5_DPO;
    }

    /* Masked prompt positions (t == 0) MUST have exact +0.0f gradient. */
    const uint32_t pos_zero_bits = 0u;
    for (size_t b = 0u; b < B; ++b) {
      for (size_t v = 0u; v < Vd; ++v) {
        uint32_t bw = 0u;
        uint32_t bl = 0u;
        memcpy(&bw, &d_chosen[(b * T) * Vd + v], sizeof(bw));
        memcpy(&bl, &d_rejected[(b * T) * Vd + v], sizeof(bl));
        if (bw != pos_zero_bits || bl != pos_zero_bits) {
          return ET_E2E_ERR_INVARIANT;
        }
      }
    }

    if (step == 0u) {
      ctx->m.s5_initial_loss = tel.mean_loss;
      ctx->m.s5_initial_margin = tel.reward_margin_mean;
    } else {
      if (!(tel.mean_loss < prev_loss) ||
          !(tel.reward_margin_mean > prev_margin)) {
        return ET_E2E_ERR_INVARIANT;
      }
    }
    prev_loss = tel.mean_loss;
    prev_margin = tel.reward_margin_mean;

    if (step == ET_E2E_DPO_STEPS) {
      ctx->m.s5_final_loss = tel.mean_loss;
      ctx->m.s5_final_margin = tel.reward_margin_mean;
      ctx->m.s5_final_accuracy = tel.preference_accuracy;
      if (tel.preference_accuracy != 1.0) {
        return ET_E2E_ERR_INVARIANT;
      }
      break;
    }

    memset(dw_policy, 0, sizeof(dw_policy));
    for (size_t bt = 0u; bt < B * T; ++bt) {
      for (size_t v = 0u; v < Vd; ++v) {
        double gw = (double)d_chosen[bt * Vd + v];
        double gl = (double)d_rejected[bt * Vd + v];
        for (size_t d = 0u; d < D; ++d) {
          dw_policy[v * D + d] += (float)(gw * (double)h_chosen[bt * D + d] +
                                          gl * (double)h_rejected[bt * D + d]);
        }
      }
    }
    const float lr = 0.6f;
    for (size_t i = 0u; i < Vd * D; ++i) {
      w_policy[i] -= lr * dw_policy[i];
    }
  }

  /* Verify frozen reference invariance. */
  if (et_dpo_weight_sha256(Vd * D, w_ref, ref_sha_after) != ET_DPO_OK ||
      memcmp(ref_sha_before, ref_sha_after, 32u) != 0 ||
      memcmp(w_ref, w_ref_orig, sizeof(w_ref)) != 0) {
    return ET_E2E_ERR_INVARIANT;
  }

  /* Write aligned policy rows back into live model W_lm for Stages 6-8. */
  for (size_t i = 0u; i < Vd; ++i) {
    memcpy(ctx->live_weights + L.off_wlm + (size_t)active_vocab[i] * D,
           &w_policy[i * D], D * sizeof(float));
  }

  ctx->m.s5_steps = ET_E2E_DPO_STEPS;
  ctx->m.s5_ref_frozen_equal = 1u;
  memcpy(ctx->m.s5_ref_sha256, ref_sha_after, 32u);
  return ET_E2E_OK;
}

/* ---- Stage 6a: EVAL5 Contamination-Gated Benchmark ---- */

int32_t et_e2e_aot_run_stage6_eval5(et_e2e_aot_ctx *ctx, const char *work_dir) {
  char corpus_path[ET_E2E_PATH_MAX];
  char tok_path[ET_E2E_PATH_MAX];
  char contam_manifest_path[ET_E2E_PATH_MAX];
  char clean_manifest_path[ET_E2E_PATH_MAX];
  owned_corpus train_corpus;
  et_eval5_document train_docs[ET_E2E_MAX_DOCS];
  et_eval5_document contam_eval[2];
  et_eval5_document clean_eval[3];
  et_eval5_contamination_config ccfg;
  et_eval5_contamination_summary contam_sum;
  et_eval5_contamination_summary clean_sum;
  uint64_t contam_indices[3];
  et_eval5_status c_st;
  struct stat st_check;
  et_t2_bpe_model *bpe = NULL;
  uint8_t tok_sha[32];
  uint8_t eval_sha[32];
  uint8_t ckpt_sha[32];
  model_layout L;
  uint32_t eval_toks[128];
  size_t eval_tok_count = 0u;
  float *lm_logits = NULL;
  float *mc_logits = NULL;
  et_eval5_lm_metrics lm_metrics;
  et_eval5_mc_metrics mc_metrics;
  et_eval5_manifest manifest;
  et_eval5_manifest verify_manifest;
  et_e2e_status st;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  st = ensure_stage3_weights_loaded(ctx, work_dir);
  if (st != ET_E2E_OK) {
    return st;
  }
  if (!build_path(corpus_path, sizeof(corpus_path), work_dir,
                  "stage1_clean_corpus.bin") ||
      !build_path(tok_path, sizeof(tok_path), work_dir,
                  "stage2_tokenizer.etbpe") ||
      !build_path(contam_manifest_path, sizeof(contam_manifest_path), work_dir,
                  "stage6_contaminated.manifest") ||
      !build_path(clean_manifest_path, sizeof(clean_manifest_path), work_dir,
                  "stage6_clean.manifest")) {
    return ET_E2E_ERR_ARGUMENT;
  }
  (void)unlink(contam_manifest_path);

  st = load_clean_corpus_bundle(corpus_path, &train_corpus,
                                ctx->clean_corpus_sha256);
  if (st != ET_E2E_OK) {
    return st;
  }
  for (size_t i = 0u; i < train_corpus.doc_count; ++i) {
    train_docs[i] = (et_eval5_document){train_corpus.docs[i],
                                        train_corpus.lens[i], NULL, 0u};
  }

  ccfg.ngram_mode = ET_EVAL5_NGRAM_BYTES;
  ccfg.ngram_order = 8u;
  ccfg.forbid_exact_match = 1u;
  ccfg.min_ngram_overlap = 0.25;
  ccfg.max_allowed_rate = 0.0;

  /* 1. Fail-closed contaminated eval audit: MUST reject & write ZERO manifest. */
  contam_eval[0] = train_docs[0];
  contam_eval[1] = (et_eval5_document){(const uint8_t *)k_clean_eval_doc0,
                                       strlen(k_clean_eval_doc0), NULL, 0u};
  memset(&contam_sum, 0, sizeof(contam_sum));
  c_st = et_eval5_audit_contamination(train_docs, train_corpus.doc_count,
                                      contam_eval, 2u, &ccfg, contam_indices,
                                      2u, &contam_sum);
  if (c_st != ET_EVAL5_ERR_CONTAMINATED ||
      contam_sum.contaminated_documents != 1u || contam_indices[0] != 0u) {
    owned_corpus_free(&train_corpus);
    return ET_E2E_ERR_INVARIANT;
  }
  /* Fail-closed: do NOT emit manifest on contamination; verify absent on disk. */
  if (lstat(contam_manifest_path, &st_check) == 0 || errno != ENOENT) {
    owned_corpus_free(&train_corpus);
    return ET_E2E_ERR_INVARIANT;
  }

  /* 2. Clean held-out eval audit: MUST pass with 0 contaminated docs. */
  clean_eval[0] = (et_eval5_document){(const uint8_t *)k_clean_eval_doc0,
                                      strlen(k_clean_eval_doc0), NULL, 0u};
  clean_eval[1] = (et_eval5_document){(const uint8_t *)k_clean_eval_doc1,
                                      strlen(k_clean_eval_doc1), NULL, 0u};
  clean_eval[2] = (et_eval5_document){(const uint8_t *)k_clean_eval_doc2,
                                      strlen(k_clean_eval_doc2), NULL, 0u};
  memset(&clean_sum, 0, sizeof(clean_sum));
  if (et_eval5_audit_contamination(train_docs, train_corpus.doc_count,
                                   clean_eval, 3u, &ccfg, contam_indices, 3u,
                                   &clean_sum) != ET_EVAL5_OK ||
      clean_sum.contaminated_documents != 0u) {
    owned_corpus_free(&train_corpus);
    return ET_E2E_ERR_STAGE6_EVAL5;
  }
  owned_corpus_free(&train_corpus);

  /* Encode clean held-out eval doc0 with Stage 2 BPE and score LM NLL/PPL. */
  if (et_t2_bpe_load_file(tok_path, &bpe, tok_sha) != ET_T2_BPE_OK) {
    return ET_E2E_ERR_STAGE2_BPE;
  }
  if (et_t2_bpe_encode(bpe, clean_eval[0].bytes, clean_eval[0].byte_count,
                       eval_toks, 128u, &eval_tok_count) != ET_T2_BPE_OK ||
      eval_tok_count < ET_E2E_SEQ_LEN + 1u) {
    et_t2_bpe_destroy(bpe);
    return ET_E2E_ERR_STAGE6_EVAL5;
  }
  et_t2_bpe_destroy(bpe);

  L = make_model_layout((size_t)ctx->vocab_size);
  lm_logits = (float *)calloc(ET_E2E_SEQ_LEN * L.V, sizeof(float));
  mc_logits = (float *)calloc(4u * 4u * L.V, sizeof(float));
  if (lm_logits == NULL || mc_logits == NULL) {
    free(lm_logits);
    free(mc_logits);
    return ET_E2E_ERR_ALLOC;
  }

  st = model_forward_backward_window(&L, ctx->live_weights, eval_toks, NULL,
                                     ET_E2E_SEQ_LEN, 1.0, NULL, NULL, lm_logits,
                                     NULL, NULL);
  if (st != ET_E2E_OK) {
    free(lm_logits);
    free(mc_logits);
    return st;
  }
  et_eval5_sequence lm_seq;
  lm_seq.logits = lm_logits;
  lm_seq.target_ids = eval_toks + 1u;
  lm_seq.token_count = ET_E2E_SEQ_LEN;
  lm_seq.vocab_size = L.V;
  lm_seq.logit_count = ET_E2E_SEQ_LEN * L.V;
  if (et_eval5_score_lm(&lm_seq, (uint64_t)clean_eval[0].byte_count,
                        &lm_metrics) != ET_EVAL5_OK ||
      !isfinite(lm_metrics.mean_nll) || !isfinite(lm_metrics.perplexity) ||
      !(lm_metrics.perplexity > 1.0)) {
    free(lm_logits);
    free(mc_logits);
    return ET_E2E_ERR_STAGE6_EVAL5;
  }

  /* Score 2 deterministic held-out multiple-choice items with post-Stage-5 logits. */
  et_eval5_sequence mc_seqs[4];
  uint32_t mc_targets[4][4];
  et_eval5_choice_item mc_items[2];
  for (size_t c = 0u; c < 4u; ++c) {
    uint32_t seq_in[4];
    for (size_t t = 0u; t < 4u; ++t) {
      seq_in[t] = eval_toks[(c * 2u + t) % eval_tok_count];
      mc_targets[c][t] = eval_toks[(c * 2u + t + 1u) % eval_tok_count];
    }
    st = model_forward_backward_window(&L, ctx->live_weights, seq_in, NULL, 4u,
                                       1.0, NULL, NULL,
                                       mc_logits + c * 4u * L.V, NULL, NULL);
    if (st != ET_E2E_OK) {
      free(lm_logits);
      free(mc_logits);
      return st;
    }
    mc_seqs[c].logits = mc_logits + c * 4u * L.V;
    mc_seqs[c].target_ids = mc_targets[c];
    mc_seqs[c].token_count = 4u;
    mc_seqs[c].vocab_size = L.V;
    mc_seqs[c].logit_count = 4u * L.V;
  }
  mc_items[0] = (et_eval5_choice_item){&mc_seqs[0], 2u, 0u};
  mc_items[1] = (et_eval5_choice_item){&mc_seqs[2], 2u, 1u};
  if (et_eval5_score_mc(mc_items, 2u, 1u, &mc_metrics) != ET_EVAL5_OK ||
      !isfinite(mc_metrics.accuracy)) {
    free(lm_logits);
    free(mc_logits);
    return ET_E2E_ERR_STAGE6_EVAL5;
  }
  free(lm_logits);
  free(mc_logits);

  if (et_data4_sha256(clean_eval[0].bytes, clean_eval[0].byte_count,
                      eval_sha) != ET_DATA4_OK ||
      et_dpo_weight_sha256(L.total_params, ctx->live_weights, ckpt_sha) !=
          ET_DPO_OK) {
    return ET_E2E_ERR_STAGE6_EVAL5;
  }

  memset(&manifest, 0, sizeof(manifest));
  manifest.version_major = ET_EVAL5_VERSION_MAJOR;
  manifest.version_minor = ET_EVAL5_VERSION_MINOR;
  (void)snprintf(manifest.suite, sizeof(manifest.suite), "e2e_llm_lifecycle");
  (void)snprintf(manifest.task, sizeof(manifest.task), "heldout_lm_mc");
  (void)snprintf(manifest.split, sizeof(manifest.split), "validation");
  memcpy(manifest.train_dataset_sha256, ctx->clean_corpus_sha256, 32u);
  memcpy(manifest.eval_dataset_sha256, eval_sha, 32u);
  memcpy(manifest.tokenizer_sha256, tok_sha, 32u);
  memcpy(manifest.checkpoint_sha256, ckpt_sha, 32u);
  manifest.context_length = ET_E2E_SEQ_LEN;
  manifest.max_samples = 3u;
  manifest.contamination = ccfg;
  manifest.metric_mask = ET_EVAL5_METRIC_LM | ET_EVAL5_METRIC_MC;
  manifest.mc_normalize_by_length = 1u;
  manifest.lm = lm_metrics;
  manifest.mc = mc_metrics;
  manifest.audit = clean_sum;

  if (et_eval5_manifest_write_atomic(clean_manifest_path, &manifest) !=
          ET_EVAL5_OK ||
      et_eval5_manifest_read_file(clean_manifest_path, &verify_manifest) !=
          ET_EVAL5_OK) {
    return ET_E2E_ERR_STAGE6_EVAL5;
  }

  ctx->m.s6_contam_status = (int32_t)c_st;
  ctx->m.s6_contam_manifest_absent = 1u;
  ctx->m.s6_contam_docs_found = contam_sum.contaminated_documents;
  ctx->m.s6_clean_contam_docs = clean_sum.contaminated_documents;
  ctx->m.s6_lm_mean_nll = lm_metrics.mean_nll;
  ctx->m.s6_lm_perplexity = lm_metrics.perplexity;
  ctx->m.s6_mc_accuracy = mc_metrics.accuracy;
  memcpy(ctx->m.s6_manifest_sha256, verify_manifest.manifest_sha256, 32u);
  return ET_E2E_OK;
}

/* ---- Stage 6b: Fresh-Process Manifest Reload Verification ---- */

int32_t et_e2e_aot_run_stage6_verify_fresh(et_e2e_aot_ctx *ctx,
                                           const char *work_dir) {
  char clean_manifest_path[ET_E2E_PATH_MAX];
  char contam_manifest_path[ET_E2E_PATH_MAX];
  et_eval5_manifest loaded;
  struct stat st_check;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  if (!build_path(clean_manifest_path, sizeof(clean_manifest_path), work_dir,
                  "stage6_clean.manifest") ||
      !build_path(contam_manifest_path, sizeof(contam_manifest_path), work_dir,
                  "stage6_contaminated.manifest")) {
    return ET_E2E_ERR_ARGUMENT;
  }
  if (lstat(contam_manifest_path, &st_check) == 0 || errno != ENOENT) {
    return ET_E2E_ERR_INVARIANT;
  }
  memset(&loaded, 0, sizeof(loaded));
  if (et_eval5_manifest_read_file(clean_manifest_path, &loaded) !=
      ET_EVAL5_OK) {
    return ET_E2E_ERR_STAGE6_EVAL5;
  }
  if (strcmp(loaded.suite, "e2e_llm_lifecycle") != 0 ||
      strcmp(loaded.task, "heldout_lm_mc") != 0 ||
      loaded.audit.contaminated_documents != 0u ||
      !isfinite(loaded.lm.mean_nll) || !isfinite(loaded.lm.perplexity) ||
      !(loaded.lm.perplexity > 1.0) || !isfinite(loaded.mc.accuracy)) {
    return ET_E2E_ERR_INVARIANT;
  }

  ctx->m.s6_clean_contam_docs = loaded.audit.contaminated_documents;
  ctx->m.s6_lm_mean_nll = loaded.lm.mean_nll;
  ctx->m.s6_lm_perplexity = loaded.lm.perplexity;
  ctx->m.s6_mc_accuracy = loaded.mc.accuracy;
  ctx->m.s6_fresh_reload_verified = 1u;
  memcpy(ctx->m.s6_manifest_sha256, loaded.manifest_sha256, 32u);
  return ET_E2E_OK;
}

/* ---- Stage 7: Exact Lossless Speculative Decoding ---- */

static uint32_t argmax_row_f32(const float *row, size_t V) {
  uint32_t best = 0u;
  float best_val = row[0];
  for (size_t v = 1u; v < V; ++v) {
    if (row[v] > best_val) {
      best_val = row[v];
      best = (uint32_t)v;
    }
  }
  return best;
}

int32_t et_e2e_aot_run_stage7_spec_decode(et_e2e_aot_ctx *ctx,
                                          const char *work_dir) {
  const uint32_t K = ET_E2E_SPEC_HORIZON;
  model_layout L;
  float *draft_logits = NULL;
  float *target_logits = NULL;
  float *eos_target_logits = NULL;
  uint32_t seq_in[ET_E2E_SPEC_HORIZON + 1u];
  uint32_t draft_tokens[ET_E2E_SPEC_HORIZON];
  uint32_t eos_draft_tokens[ET_E2E_SPEC_HORIZON];
  uint32_t greedy_out[ET_E2E_SPEC_HORIZON + 1u];
  uint32_t stoch_out_a[ET_E2E_SPEC_HORIZON + 1u];
  uint32_t stoch_out_b[ET_E2E_SPEC_HORIZON + 1u];
  uint32_t eos_out[ET_E2E_SPEC_HORIZON + 1u];
  et_spec_verify_result_v1 greedy_res;
  et_spec_verify_result_v1 stoch_res_a;
  et_spec_verify_result_v1 stoch_res_b;
  et_spec_verify_result_v1 eos_res;
  et_spec_sampling_params_v1 sparams;
  et_spec_rng_state_v1 rng_init;
  et_e2e_status st;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  st = ensure_stage3_weights_loaded(ctx, work_dir);
  if (st != ET_E2E_OK) {
    return st;
  }
  L = make_model_layout((size_t)ctx->vocab_size);

  draft_logits = (float *)calloc((size_t)K * L.V, sizeof(float));
  target_logits = (float *)calloc((size_t)(K + 1u) * L.V, sizeof(float));
  eos_target_logits = (float *)calloc((size_t)(K + 1u) * L.V, sizeof(float));
  if (draft_logits == NULL || target_logits == NULL ||
      eos_target_logits == NULL) {
    free(draft_logits);
    free(target_logits);
    free(eos_target_logits);
    return ET_E2E_ERR_ALLOC;
  }

  for (size_t t = 0u; t < (size_t)(K + 1u); ++t) {
    seq_in[t] = (uint32_t)ctx->shard_tokens[t];
  }
  /* Draft logits from pre-LoRA/pre-DPO Stage 3 base weights; target logits
   * from post-Stage-5 aligned weights! */
  st = model_forward_backward_window(&L, ctx->pretrain_weights, seq_in, NULL,
                                     (size_t)K, 1.0, NULL, NULL, draft_logits,
                                     NULL, NULL);
  if (st != ET_E2E_OK) {
    goto cleanup;
  }
  st = model_forward_backward_window(&L, ctx->live_weights, seq_in, NULL,
                                     (size_t)(K + 1u), 1.0, NULL, NULL,
                                     target_logits, NULL, NULL);
  if (st != ET_E2E_OK) {
    goto cleanup;
  }

  for (uint32_t i = 0u; i < K; ++i) {
    draft_tokens[i] = argmax_row_f32(draft_logits + (size_t)i * L.V, L.V);
  }

  /* 1. Greedy 100% lossless token equivalence check. */
  memset(&greedy_res, 0, sizeof(greedy_res));
  if (et_spec_verify_greedy_v1(draft_tokens, target_logits, K, (uint32_t)L.V,
                               ET_SPEC_NO_EOS, greedy_out, K + 1u,
                               &greedy_res) != ET_SPEC_OK ||
      greedy_res.emitted_count == 0u || greedy_res.emitted_count > K + 1u) {
    st = ET_E2E_ERR_STAGE7_SPEC;
    goto cleanup;
  }
  for (uint32_t i = 0u; i < greedy_res.emitted_count; ++i) {
    uint32_t expected_tok =
        argmax_row_f32(target_logits + (size_t)i * L.V, L.V);
    if (greedy_out[i] != expected_tok) {
      st = ET_E2E_ERR_INVARIANT;
      goto cleanup;
    }
  }

  /* 2. Seeded stochastic replay bitwise determinism check. */
  sparams.temperature = 0.8;
  sparams.top_p = 0.9;
  sparams.top_k = 16u;
  sparams.eos_token_id = ET_SPEC_NO_EOS;
  rng_init.seed = UINT64_C(0x5EEDC0FFEE123456);
  rng_init.counter = 0u;

  memset(&stoch_res_a, 0, sizeof(stoch_res_a));
  memset(&stoch_res_b, 0, sizeof(stoch_res_b));
  if (et_spec_verify_sample_v1(draft_tokens, draft_logits, target_logits, K,
                               (uint32_t)L.V, &sparams, &rng_init, stoch_out_a,
                               K + 1u, &stoch_res_a) != ET_SPEC_OK ||
      et_spec_verify_sample_v1(draft_tokens, draft_logits, target_logits, K,
                               (uint32_t)L.V, &sparams, &rng_init, stoch_out_b,
                               K + 1u, &stoch_res_b) != ET_SPEC_OK) {
    st = ET_E2E_ERR_STAGE7_SPEC;
    goto cleanup;
  }
  if (stoch_res_a.emitted_count != stoch_res_b.emitted_count ||
      stoch_res_a.accepted_count != stoch_res_b.accepted_count ||
      stoch_res_a.rejected != stoch_res_b.rejected ||
      stoch_res_a.rng_draws != stoch_res_b.rng_draws ||
      stoch_res_a.rng_draws == 0u ||
      stoch_res_a.next_rng.seed != stoch_res_b.next_rng.seed ||
      stoch_res_a.next_rng.counter != stoch_res_b.next_rng.counter ||
      memcmp(stoch_out_a, stoch_out_b,
             stoch_res_a.emitted_count * sizeof(uint32_t)) != 0) {
    st = ET_E2E_ERR_INVARIANT;
    goto cleanup;
  }

  /* 3. EOS early termination check (using Stage 2 BPE EOS token ID). */
  memcpy(eos_target_logits, target_logits, (size_t)(K + 1u) * L.V * sizeof(float));
  eos_target_logits[1u * L.V + ctx->eos_id] = 25.0f;
  eos_draft_tokens[0] = argmax_row_f32(eos_target_logits, L.V);
  eos_draft_tokens[1] = ctx->eos_id;
  eos_draft_tokens[2] = draft_tokens[2];
  eos_draft_tokens[3] = draft_tokens[3];
  memset(&eos_res, 0, sizeof(eos_res));
  if (et_spec_verify_greedy_v1(eos_draft_tokens, eos_target_logits, K,
                               (uint32_t)L.V, ctx->eos_id, eos_out, K + 1u,
                               &eos_res) != ET_SPEC_OK ||
      eos_res.eos_hit != 1u || eos_res.emitted_count != 2u ||
      eos_out[1] != ctx->eos_id) {
    st = ET_E2E_ERR_INVARIANT;
    goto cleanup;
  }

  ctx->m.s7_horizon = K;
  ctx->m.s7_greedy_emitted = greedy_res.emitted_count;
  ctx->m.s7_greedy_accepted = greedy_res.accepted_count;
  ctx->m.s7_greedy_exact_match = 1u;
  ctx->m.s7_stoch_emitted = stoch_res_a.emitted_count;
  ctx->m.s7_stoch_replay_match = 1u;
  ctx->m.s7_eos_hit = eos_res.eos_hit;
  ctx->m.s7_eos_emitted = eos_res.emitted_count;
  st = ET_E2E_OK;

cleanup:
  free(draft_logits);
  free(target_logits);
  free(eos_target_logits);
  return st;
}

/* ---- Stage 8: Native IO5 F16 / BF16 / Q8_0 Codec Round-Trip ---- */

int32_t et_e2e_aot_run_stage8_io5_quant(et_e2e_aot_ctx *ctx,
                                        const char *work_dir) {
  size_t n;
  size_t f32_bytes;
  size_t half_bytes;
  size_t q8_bytes = 0u;
  uint8_t *f16_buf = NULL;
  uint8_t *bf16_buf = NULL;
  uint8_t *q8_buf = NULL;
  float *dec_buf = NULL;
  double f16_max = 0.0;
  double bf16_max = 0.0;
  double q8_max = 0.0;
  double q8_sq_err = 0.0;
  double ref_sq = 0.0;
  double q8_rel_l2 = 0.0;
  et_e2e_status st;

  if (ctx == NULL || work_dir == NULL || work_dir[0] == '\0') {
    return ET_E2E_ERR_ARGUMENT;
  }
  st = ensure_stage3_weights_loaded(ctx, work_dir);
  if (st != ET_E2E_OK) {
    return st;
  }

  n = ctx->param_count;
  if (n == 0u || (n % ET_IO5_QK8_0) != 0u) {
    return ET_E2E_ERR_INVARIANT;
  }
  f32_bytes = n * sizeof(float);
  half_bytes = n * 2u;

  if (et_io5_validate_f32_finite(ctx->live_weights, f32_bytes, n) !=
          ET_IO5_QUANT_OK ||
      et_io5_q8_0_byte_size(n, &q8_bytes) != ET_IO5_QUANT_OK) {
    return ET_E2E_ERR_STAGE8_IO5;
  }

  f16_buf = (uint8_t *)calloc(1u, half_bytes);
  bf16_buf = (uint8_t *)calloc(1u, half_bytes);
  q8_buf = (uint8_t *)calloc(1u, q8_bytes);
  dec_buf = (float *)calloc(n, sizeof(float));
  if (f16_buf == NULL || bf16_buf == NULL || q8_buf == NULL ||
      dec_buf == NULL) {
    st = ET_E2E_ERR_ALLOC;
    goto cleanup;
  }

  /* F16 round-trip. */
  if (et_io5_encode_f16(ctx->live_weights, f32_bytes, f16_buf, half_bytes, n) !=
          ET_IO5_QUANT_OK ||
      et_io5_decode_f16(f16_buf, half_bytes, dec_buf, f32_bytes, n) !=
          ET_IO5_QUANT_OK) {
    st = ET_E2E_ERR_STAGE8_IO5;
    goto cleanup;
  }
  for (size_t i = 0u; i < n; ++i) {
    if (!isfinite(dec_buf[i])) {
      st = ET_E2E_ERR_NONFINITE;
      goto cleanup;
    }
    double err = fabs((double)ctx->live_weights[i] - (double)dec_buf[i]);
    if (err > f16_max) {
      f16_max = err;
    }
  }

  /* BF16 round-trip. */
  if (et_io5_encode_bf16(ctx->live_weights, f32_bytes, bf16_buf, half_bytes,
                         n) != ET_IO5_QUANT_OK ||
      et_io5_decode_bf16(bf16_buf, half_bytes, dec_buf, f32_bytes, n) !=
          ET_IO5_QUANT_OK) {
    st = ET_E2E_ERR_STAGE8_IO5;
    goto cleanup;
  }
  for (size_t i = 0u; i < n; ++i) {
    if (!isfinite(dec_buf[i])) {
      st = ET_E2E_ERR_NONFINITE;
      goto cleanup;
    }
    double err = fabs((double)ctx->live_weights[i] - (double)dec_buf[i]);
    if (err > bf16_max) {
      bf16_max = err;
    }
  }

  /* Q8_0 round-trip. */
  if (et_io5_quantize_q8_0(ctx->live_weights, f32_bytes, q8_buf, q8_bytes, n) !=
          ET_IO5_QUANT_OK ||
      et_io5_dequantize_q8_0(q8_buf, q8_bytes, dec_buf, f32_bytes, n) !=
          ET_IO5_QUANT_OK) {
    st = ET_E2E_ERR_STAGE8_IO5;
    goto cleanup;
  }
  for (size_t i = 0u; i < n; ++i) {
    if (!isfinite(dec_buf[i])) {
      st = ET_E2E_ERR_NONFINITE;
      goto cleanup;
    }
    double ref = (double)ctx->live_weights[i];
    double diff = ref - (double)dec_buf[i];
    double aerr = fabs(diff);
    if (aerr > q8_max) {
      q8_max = aerr;
    }
    q8_sq_err += diff * diff;
    ref_sq += ref * ref;
  }
  q8_rel_l2 = (ref_sq > 0.0) ? sqrt(q8_sq_err / ref_sq) : 0.0;

  if (!(f16_max <= 1.0e-3) || !(bf16_max <= 1.0e-2) ||
      !(q8_rel_l2 <= 1.0e-2)) {
    st = ET_E2E_ERR_INVARIANT;
    goto cleanup;
  }

  ctx->m.s8_weight_count = (uint64_t)n;
  ctx->m.s8_q8_0_bytes = (uint64_t)q8_bytes;
  ctx->m.s8_f16_max_abs_err = f16_max;
  ctx->m.s8_bf16_max_abs_err = bf16_max;
  ctx->m.s8_q8_0_max_abs_err = q8_max;
  ctx->m.s8_q8_0_rel_l2_err = q8_rel_l2;
  st = ET_E2E_OK;

cleanup:
  free(f16_buf);
  free(bf16_buf);
  free(q8_buf);
  free(dec_buf);
  return st;
}

/* ---- Direct C11/C++17 APIs ---- */

et_e2e_status et_e2e_run_full_in_process(const char *work_dir,
                                         et_e2e_stage_metrics_v1 *out_metrics) {
  et_e2e_aot_ctx *ctx = NULL;
  et_e2e_status st;

  if (work_dir == NULL || work_dir[0] == '\0' || out_metrics == NULL) {
    return ET_E2E_ERR_ARGUMENT;
  }
  st = et_e2e_aot_create(&ctx);
  if (st != ET_E2E_OK) {
    return st;
  }

  st = et_e2e_aot_run_stage1_data4(ctx, work_dir);
  if (st == ET_E2E_OK) {
    st = et_e2e_aot_run_stage2_train_save(ctx, work_dir);
  }
  if (st == ET_E2E_OK) {
    st = et_e2e_aot_run_stage2_reload_encode(ctx, work_dir);
  }
  if (st == ET_E2E_OK) {
    st = et_e2e_aot_run_stage3_pretrain_part1(ctx, work_dir);
  }
  if (st == ET_E2E_OK) {
    st = et_e2e_aot_run_stage3_pretrain_resume(ctx, work_dir);
  }
  if (st == ET_E2E_OK) {
    st = et_e2e_aot_run_stage4_lora_train_save(ctx, work_dir);
  }
  if (st == ET_E2E_OK) {
    st = et_e2e_aot_run_stage4_lora_reload(ctx, work_dir);
  }
  if (st == ET_E2E_OK) {
    st = et_e2e_aot_run_stage5_dpo(ctx, work_dir);
  }
  if (st == ET_E2E_OK) {
    st = et_e2e_aot_run_stage6_eval5(ctx, work_dir);
  }
  if (st == ET_E2E_OK) {
    st = et_e2e_aot_run_stage6_verify_fresh(ctx, work_dir);
  }
  if (st == ET_E2E_OK) {
    st = et_e2e_aot_run_stage7_spec_decode(ctx, work_dir);
  }
  if (st == ET_E2E_OK) {
    st = et_e2e_aot_run_stage8_io5_quant(ctx, work_dir);
  }

  if (st == ET_E2E_OK) {
    *out_metrics = ctx->m;
  }
  et_e2e_aot_destroy(ctx);
  return st;
}

et_e2e_status et_e2e_stage3_check_fd_gradients(double *out_max_abs_err) {
  const size_t V = 32u;
  model_layout L = make_model_layout(V);
  float *params = (float *)calloc(L.total_params, sizeof(float));
  float *grads = (float *)calloc(L.total_params, sizeof(float));
  const uint32_t x_ids[ET_E2E_SEQ_LEN] = {1u, 5u, 9u, 13u, 17u, 21u, 25u, 29u};
  const uint32_t y_ids[ET_E2E_SEQ_LEN] = {2u, 6u, 10u, 14u, 18u, 22u, 26u, 30u};
  double base_loss = 0.0;
  double max_err = 0.0;
  const float eps = 1.0e-3f;
  size_t probes[16];
  et_e2e_status st;

  if (out_max_abs_err == NULL) {
    free(params);
    free(grads);
    return ET_E2E_ERR_ARGUMENT;
  }
  if (params == NULL || grads == NULL) {
    free(params);
    free(grads);
    return ET_E2E_ERR_ALLOC;
  }

  init_model_weights(&L, params);
  st = model_forward_backward_window(&L, params, x_ids, y_ids, ET_E2E_SEQ_LEN,
                                     1.0 / (double)ET_E2E_SEQ_LEN, NULL, NULL,
                                     NULL, &base_loss, grads);
  if (st != ET_E2E_OK) {
    free(params);
    free(grads);
    return st;
  }

  /* Two probes in each of the 8 parameter tensors (16 total). */
  probes[0] = L.off_emb + 1u * L.D + 3u;
  probes[1] = L.off_emb + 13u * L.D + 11u;
  probes[2] = L.off_wq + 2u * L.D + 5u;
  probes[3] = L.off_wq + 12u * L.D + 9u;
  probes[4] = L.off_wk + 1u * L.D + 7u;
  probes[5] = L.off_wk + 14u * L.D + 2u;
  probes[6] = L.off_wv + 3u * L.D + 4u;
  probes[7] = L.off_wv + 11u * L.D + 15u;
  probes[8] = L.off_wo + 0u * L.D + 6u;
  probes[9] = L.off_wo + 15u * L.D + 10u;
  probes[10] = L.off_wff1 + 4u * L.D + 1u;
  probes[11] = L.off_wff1 + 27u * L.D + 12u;
  probes[12] = L.off_wff2 + 5u * L.D_ff + 8u;
  probes[13] = L.off_wff2 + 13u * L.D_ff + 23u;
  probes[14] = L.off_wlm + 2u * L.D + 4u;
  probes[15] = L.off_wlm + 30u * L.D + 14u;

  for (size_t i = 0u; i < 16u; ++i) {
    size_t idx = probes[i];
    float orig = params[idx];
    double lp = 0.0;
    double lm = 0.0;
    params[idx] = orig + eps;
    st = model_forward_backward_window(&L, params, x_ids, y_ids, ET_E2E_SEQ_LEN,
                                       1.0 / (double)ET_E2E_SEQ_LEN, NULL, NULL,
                                       NULL, &lp, NULL);
    if (st != ET_E2E_OK) {
      free(params);
      free(grads);
      return st;
    }
    params[idx] = orig - eps;
    st = model_forward_backward_window(&L, params, x_ids, y_ids, ET_E2E_SEQ_LEN,
                                       1.0 / (double)ET_E2E_SEQ_LEN, NULL, NULL,
                                       NULL, &lm, NULL);
    params[idx] = orig;
    if (st != ET_E2E_OK) {
      free(params);
      free(grads);
      return st;
    }
    double fd = (lp - lm) / (2.0 * (double)eps);
    double err = fabs((double)grads[idx] - fd);
    if (err > max_err) {
      max_err = err;
    }
  }

  free(params);
  free(grads);
  *out_max_abs_err = max_err;
  return (max_err <= 2.0e-3) ? ET_E2E_OK : ET_E2E_ERR_INVARIANT;
}

/* ---- AOT Metric & Digest Accessors ---- */

int64_t et_e2e_aot_get_i64(const et_e2e_aot_ctx *ctx, int32_t metric_id) {
  if (ctx == NULL) {
    return -1;
  }
  switch (metric_id) {
    case ET_E2E_M_S1_INPUT_DOCS: return (int64_t)ctx->m.s1_input_docs;
    case ET_E2E_M_S1_KEPT_DOCS: return (int64_t)ctx->m.s1_kept_docs;
    case ET_E2E_M_S1_DROPPED_EXACT: return (int64_t)ctx->m.s1_dropped_exact;
    case ET_E2E_M_S1_DROPPED_NEAR: return (int64_t)ctx->m.s1_dropped_near;
    case ET_E2E_M_S1_DROPPED_UTF8: return (int64_t)ctx->m.s1_dropped_utf8;
    case ET_E2E_M_S1_DROPPED_LENGTH: return (int64_t)ctx->m.s1_dropped_length;
    case ET_E2E_M_S1_DROPPED_QUALITY: return (int64_t)ctx->m.s1_dropped_quality;
    case ET_E2E_M_S1_KEPT_BYTES: return (int64_t)ctx->m.s1_kept_bytes;

    case ET_E2E_M_S2_VOCAB_SIZE: return (int64_t)ctx->m.s2_vocab_size;
    case ET_E2E_M_S2_MERGE_COUNT: return (int64_t)ctx->m.s2_merge_count;
    case ET_E2E_M_S2_BOS_ID: return (int64_t)ctx->m.s2_bos_id;
    case ET_E2E_M_S2_EOS_ID: return (int64_t)ctx->m.s2_eos_id;
    case ET_E2E_M_S2_RAW_BYTES: return (int64_t)ctx->m.s2_raw_bytes;
    case ET_E2E_M_S2_ENCODED_TOKENS: return (int64_t)ctx->m.s2_encoded_tokens;
    case ET_E2E_M_S2_LOSSLESS_VERIFIED: return (int64_t)ctx->m.s2_lossless_verified;
    case ET_E2E_M_S2_RELOAD_SHA_MATCH: return (int64_t)ctx->m.s2_reload_sha_match;

    case ET_E2E_M_S3_PARAM_COUNT: return (int64_t)ctx->m.s3_param_count;
    case ET_E2E_M_S3_SHARD_TOKENS: return (int64_t)ctx->m.s3_shard_tokens;
    case ET_E2E_M_S3_STEPS_FULL: return (int64_t)ctx->m.s3_steps_full;
    case ET_E2E_M_S3_STEPS_SPLIT: return (int64_t)ctx->m.s3_steps_split;
    case ET_E2E_M_S3_WEIGHTS_BITWISE_EQUAL:
      return (int64_t)ctx->m.s3_weights_bitwise_equal;
    case ET_E2E_M_S3_SNAPSHOT_SHA_EQUAL:
      return (int64_t)ctx->m.s3_snapshot_sha_equal;
    case ET_E2E_M_S3_CORRUPT_REJECTED:
      return (int64_t)ctx->m.s3_corrupt_rejected;

    case ET_E2E_M_S4_RANK: return (int64_t)ctx->m.s4_rank;
    case ET_E2E_M_S4_STEPS: return (int64_t)ctx->m.s4_steps;
    case ET_E2E_M_S4_BASE_FROZEN_EQUAL:
      return (int64_t)ctx->m.s4_base_frozen_equal;
    case ET_E2E_M_S4_RELOAD_EQUAL: return (int64_t)ctx->m.s4_reload_equal;

    case ET_E2E_M_S5_STEPS: return (int64_t)ctx->m.s5_steps;
    case ET_E2E_M_S5_REF_FROZEN_EQUAL:
      return (int64_t)ctx->m.s5_ref_frozen_equal;

    case ET_E2E_M_S6_CONTAM_STATUS: return (int64_t)ctx->m.s6_contam_status;
    case ET_E2E_M_S6_CONTAM_MANIFEST_ABSENT:
      return (int64_t)ctx->m.s6_contam_manifest_absent;
    case ET_E2E_M_S6_CONTAM_DOCS_FOUND:
      return (int64_t)ctx->m.s6_contam_docs_found;
    case ET_E2E_M_S6_CLEAN_CONTAM_DOCS:
      return (int64_t)ctx->m.s6_clean_contam_docs;
    case ET_E2E_M_S6_FRESH_RELOAD_VERIFIED:
      return (int64_t)ctx->m.s6_fresh_reload_verified;

    case ET_E2E_M_S7_HORIZON: return (int64_t)ctx->m.s7_horizon;
    case ET_E2E_M_S7_GREEDY_EMITTED: return (int64_t)ctx->m.s7_greedy_emitted;
    case ET_E2E_M_S7_GREEDY_ACCEPTED: return (int64_t)ctx->m.s7_greedy_accepted;
    case ET_E2E_M_S7_GREEDY_EXACT_MATCH:
      return (int64_t)ctx->m.s7_greedy_exact_match;
    case ET_E2E_M_S7_STOCH_EMITTED: return (int64_t)ctx->m.s7_stoch_emitted;
    case ET_E2E_M_S7_STOCH_REPLAY_MATCH:
      return (int64_t)ctx->m.s7_stoch_replay_match;
    case ET_E2E_M_S7_EOS_HIT: return (int64_t)ctx->m.s7_eos_hit;
    case ET_E2E_M_S7_EOS_EMITTED: return (int64_t)ctx->m.s7_eos_emitted;

    case ET_E2E_M_S8_WEIGHT_COUNT: return (int64_t)ctx->m.s8_weight_count;
    case ET_E2E_M_S8_Q8_0_BYTES: return (int64_t)ctx->m.s8_q8_0_bytes;
    default: return -1;
  }
}

double et_e2e_aot_get_f64(const et_e2e_aot_ctx *ctx, int32_t metric_id) {
  if (ctx == NULL) {
    return NAN;
  }
  switch (metric_id) {
    case ET_E2E_F_S2_COMPRESSION_RATIO: return ctx->m.s2_compression_ratio;
    case ET_E2E_F_S3_LOSS_STEP1: return ctx->m.s3_loss_step1;
    case ET_E2E_F_S3_LOSS_STEP4: return ctx->m.s3_loss_step4;
    case ET_E2E_F_S3_LOSS_STEP8: return ctx->m.s3_loss_step8;
    case ET_E2E_F_S4_INITIAL_LOSS: return ctx->m.s4_initial_loss;
    case ET_E2E_F_S4_FINAL_LOSS: return ctx->m.s4_final_loss;
    case ET_E2E_F_S4_MERGE_MAX_ABS_ERR: return ctx->m.s4_merge_max_abs_err;
    case ET_E2E_F_S4_UNMERGE_MAX_ABS_ERR: return ctx->m.s4_unmerge_max_abs_err;
    case ET_E2E_F_S5_INITIAL_LOSS: return ctx->m.s5_initial_loss;
    case ET_E2E_F_S5_FINAL_LOSS: return ctx->m.s5_final_loss;
    case ET_E2E_F_S5_INITIAL_MARGIN: return ctx->m.s5_initial_margin;
    case ET_E2E_F_S5_FINAL_MARGIN: return ctx->m.s5_final_margin;
    case ET_E2E_F_S5_FINAL_ACCURACY: return ctx->m.s5_final_accuracy;
    case ET_E2E_F_S6_LM_MEAN_NLL: return ctx->m.s6_lm_mean_nll;
    case ET_E2E_F_S6_LM_PERPLEXITY: return ctx->m.s6_lm_perplexity;
    case ET_E2E_F_S6_MC_ACCURACY: return ctx->m.s6_mc_accuracy;
    case ET_E2E_F_S8_F16_MAX_ABS_ERR: return ctx->m.s8_f16_max_abs_err;
    case ET_E2E_F_S8_BF16_MAX_ABS_ERR: return ctx->m.s8_bf16_max_abs_err;
    case ET_E2E_F_S8_Q8_0_MAX_ABS_ERR: return ctx->m.s8_q8_0_max_abs_err;
    case ET_E2E_F_S8_Q8_0_REL_L2_ERR: return ctx->m.s8_q8_0_rel_l2_err;
    default: return NAN;
  }
}

int32_t et_e2e_aot_get_sha256_hex(const et_e2e_aot_ctx *ctx, int32_t digest_id,
                                  char *out_buf, int64_t capacity) {
  const uint8_t *src = NULL;
  if (ctx == NULL || out_buf == NULL || capacity < ET_E2E_SHA256_HEX_BYTES) {
    return ET_E2E_ERR_ARGUMENT;
  }
  switch (digest_id) {
    case ET_E2E_DIGEST_S1_REPORT: src = ctx->m.s1_report_sha256; break;
    case ET_E2E_DIGEST_S2_TOKENIZER: src = ctx->m.s2_tokenizer_sha256; break;
    case ET_E2E_DIGEST_S3_FULL_SNAP: src = ctx->m.s3_full_snap_sha256; break;
    case ET_E2E_DIGEST_S3_RESUMED_SNAP:
      src = ctx->m.s3_resumed_snap_sha256;
      break;
    case ET_E2E_DIGEST_S4_BASE_WEIGHT: src = ctx->m.s4_base_sha256; break;
    case ET_E2E_DIGEST_S4_LORA_CKPT: src = ctx->m.s4_ckpt_sha256; break;
    case ET_E2E_DIGEST_S5_REF_WEIGHT: src = ctx->m.s5_ref_sha256; break;
    case ET_E2E_DIGEST_S6_MANIFEST: src = ctx->m.s6_manifest_sha256; break;
    default: return ET_E2E_ERR_ARGUMENT;
  }
  sha256_to_hex(src, out_buf);
  return ET_E2E_OK;
}

int64_t et_e2e_aot_read_byte(const char *buf, int64_t index) {
  if (buf == NULL || index < 0) {
    return -1;
  }
  return (int64_t)(unsigned char)buf[index];
}

et_e2e_status et_e2e_aot_get_metrics(const et_e2e_aot_ctx *ctx,
                                     et_e2e_stage_metrics_v1 *out_metrics) {
  if (ctx == NULL || out_metrics == NULL) {
    return ET_E2E_ERR_ARGUMENT;
  }
  *out_metrics = ctx->m;
  return ET_E2E_OK;
}
