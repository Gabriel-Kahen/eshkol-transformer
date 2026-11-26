#include "constrained_decode_eshkol_bridge.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Built-in byte + multi-byte + adversarial token vocabulary layout:
 *   [0 .. 94]   : single printable ASCII bytes 0x20 (' ') .. 0x7E ('~')
 *   [95 .. 110] : multi-byte JSON structural / value / adversarial tokens:
 *     95  (tag 1): "{\"answer\":\""
 *     96  (tag 2): "yes"
 *     97  (tag 3): "no"
 *     98  (tag 4): "maybe"
 *     99  (tag 5): "\",\"score\":"
 *     100 (tag 6): ",\"ok\":"
 *     101 (tag 7): "true"
 *     102 (tag 8): "false"
 *     103 (tag 9): "42"
 *     104 (tag 10): "-42"
 *     105 (tag 11): "INVALID{{"
 *     106 (tag 12): "\"\"broken"
 *     107 (tag 13): "007"
 *     108 (tag 14): "null"
 *     109 (tag 15): "true}"
 *     110 (tag 16): "false}"
 *   [111]       : EOS token (tag 0)
 */

#define AOT_SINGLE_ASCII_COUNT 95u
#define AOT_MULTI_COUNT 16u
#define AOT_EOS_TOKEN_ID (AOT_SINGLE_ASCII_COUNT + AOT_MULTI_COUNT)
#define AOT_VOCAB_SIZE (AOT_EOS_TOKEN_ID + 1u)
#define AOT_MAX_OUTPUT_BYTES 4096u

static const char *const k_multi_tokens[AOT_MULTI_COUNT] = {
    "{\"answer\":\"", /* 95  tag 1 */
    "yes",           /* 96  tag 2 */
    "no",            /* 97  tag 3 */
    "maybe",         /* 98  tag 4 */
    "\",\"score\":", /* 99  tag 5 */
    ",\"ok\":",      /* 100 tag 6 */
    "true",          /* 101 tag 7 */
    "false",         /* 102 tag 8 */
    "42",            /* 103 tag 9 */
    "-42",           /* 104 tag 10 */
    "INVALID{{",     /* 105 tag 11 */
    "\"\"broken",    /* 106 tag 12 */
    "007",           /* 107 tag 13 */
    "null",          /* 108 tag 14 */
    "true}",         /* 109 tag 15 */
    "false}"         /* 110 tag 16 */
};

struct et_cd_aot_ctx {
  et_cd_dfa *dfa;
  et_cd_vocab_index *index;
  uint8_t single_ascii[AOT_SINGLE_ASCII_COUNT];
  const uint8_t *token_ptrs[AOT_VOCAB_SIZE];
  size_t token_lens[AOT_VOCAB_SIZE];
  float logits[AOT_VOCAB_SIZE];
  float masked_logits[AOT_VOCAB_SIZE];
  uint32_t mask_valid;
  uint16_t cur_state;
  uint32_t finished;
  uint8_t output[AOT_MAX_OUTPUT_BYTES];
  size_t output_len;
  uint8_t artifact_sha256[ET_CD_SHA256_BYTES];
  int has_artifact_sha256;
};

static int32_t init_ctx_with_dfa(et_cd_dfa *dfa, et_cd_aot_ctx **out_ctx) {
  if (dfa == NULL || out_ctx == NULL) {
    et_cd_dfa_destroy(dfa);
    return ET_CD_ERR_ARGUMENT;
  }

  et_cd_aot_ctx *ctx = (et_cd_aot_ctx *)calloc(1u, sizeof(et_cd_aot_ctx));
  if (ctx == NULL) {
    et_cd_dfa_destroy(dfa);
    return ET_CD_ERR_NOMEM;
  }

  for (uint32_t i = 0u; i < AOT_SINGLE_ASCII_COUNT; ++i) {
    ctx->single_ascii[i] = (uint8_t)(0x20u + i);
    ctx->token_ptrs[i] = &ctx->single_ascii[i];
    ctx->token_lens[i] = 1u;
  }
  for (uint32_t i = 0u; i < AOT_MULTI_COUNT; ++i) {
    ctx->token_ptrs[AOT_SINGLE_ASCII_COUNT + i] =
        (const uint8_t *)k_multi_tokens[i];
    ctx->token_lens[AOT_SINGLE_ASCII_COUNT + i] = strlen(k_multi_tokens[i]);
  }
  ctx->token_ptrs[AOT_EOS_TOKEN_ID] = NULL;
  ctx->token_lens[AOT_EOS_TOKEN_ID] = 0u;

  et_cd_vocab_index *idx = NULL;
  const et_cd_status st =
      et_cd_vocab_index_build(dfa, ctx->token_ptrs, ctx->token_lens,
                              AOT_VOCAB_SIZE, AOT_EOS_TOKEN_ID, &idx);
  if (st != ET_CD_OK) {
    free(ctx);
    et_cd_dfa_destroy(dfa);
    return st;
  }

  et_cd_dfa_info info;
  (void)et_cd_dfa_info_get(dfa, &info);
  ctx->dfa = dfa;
  ctx->index = idx;
  ctx->cur_state = info.start_state;
  ctx->finished = 0u;
  ctx->output_len = 0u;
  *out_ctx = ctx;
  return ET_CD_OK;
}

int32_t et_cd_aot_create_json_schema(et_cd_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_CD_ERR_ARGUMENT;
  static const char *const answer_choices[3] = {"yes", "no", "maybe"};
  const et_cd_json_field_spec fields[3] = {
      {"answer", ET_CD_JSON_ENUM, answer_choices, 3u},
      {"score", ET_CD_JSON_INTEGER, NULL, 0u},
      {"ok", ET_CD_JSON_BOOLEAN, NULL, 0u},
  };
  et_cd_dfa *dfa = NULL;
  const et_cd_status st = et_cd_dfa_compile_flat_json_schema(fields, 3u, &dfa);
  if (st != ET_CD_OK)
    return st;
  return init_ctx_with_dfa(dfa, out_ctx);
}

int32_t et_cd_aot_create_enum_grammar(et_cd_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_CD_ERR_ARGUMENT;
  const uint8_t *choices[3] = {(const uint8_t *)"yes", (const uint8_t *)"no",
                               (const uint8_t *)"maybe"};
  const size_t lengths[3] = {3u, 2u, 5u};
  et_cd_dfa *dfa = NULL;
  const et_cd_status st =
      et_cd_dfa_compile_literal_choices(choices, lengths, 3u, &dfa);
  if (st != ET_CD_OK)
    return st;
  return init_ctx_with_dfa(dfa, out_ctx);
}

void et_cd_aot_destroy(et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return;
  et_cd_vocab_index_destroy(ctx->index);
  et_cd_dfa_destroy(ctx->dfa);
  free(ctx);
}

int64_t et_cd_aot_vocab_size(const et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1;
  return (int64_t)AOT_VOCAB_SIZE;
}

int64_t et_cd_aot_eos_token_id(const et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1;
  return (int64_t)AOT_EOS_TOKEN_ID;
}

int64_t et_cd_aot_start_state(const et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1;
  et_cd_dfa_info info;
  if (et_cd_dfa_info_get(ctx->dfa, &info) != ET_CD_OK)
    return -1;
  return (int64_t)info.start_state;
}

int64_t et_cd_aot_num_states(const et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1;
  et_cd_dfa_info info;
  if (et_cd_dfa_info_get(ctx->dfa, &info) != ET_CD_OK)
    return -1;
  return (int64_t)info.num_states;
}

int64_t et_cd_aot_live_states(const et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1;
  et_cd_dfa_info info;
  if (et_cd_dfa_info_get(ctx->dfa, &info) != ET_CD_OK)
    return -1;
  return (int64_t)info.live_state_count;
}

int64_t et_cd_aot_current_state(const et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1;
  return (int64_t)ctx->cur_state;
}

int64_t et_cd_aot_is_finished(const et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1;
  return (int64_t)ctx->finished;
}

int64_t et_cd_aot_find_token_by_tag(const et_cd_aot_ctx *ctx, int64_t tag_id) {
  if (ctx == NULL)
    return -1;
  if (tag_id == 0)
    return (int64_t)AOT_EOS_TOKEN_ID;
  if (tag_id >= 1 && tag_id <= (int64_t)AOT_MULTI_COUNT)
    return (int64_t)(AOT_SINGLE_ASCII_COUNT + (uint32_t)(tag_id - 1));
  /* Tag 100..226 maps directly to single ASCII byte (tag_id - 100). */
  if (tag_id >= 132 && tag_id <= 226)
    return (int64_t)(tag_id - 132);
  return -1;
}

int32_t et_cd_aot_reset(et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_CD_ERR_ARGUMENT;
  et_cd_dfa_info info;
  const et_cd_status st = et_cd_dfa_info_get(ctx->dfa, &info);
  if (st != ET_CD_OK)
    return st;
  ctx->cur_state = info.start_state;
  ctx->finished = 0u;
  ctx->output_len = 0u;
  ctx->mask_valid = 0u;
  memset(ctx->logits, 0, sizeof(ctx->logits));
  memset(ctx->masked_logits, 0, sizeof(ctx->masked_logits));
  return ET_CD_OK;
}

int32_t et_cd_aot_set_logit(et_cd_aot_ctx *ctx, int64_t token_id,
                            double value) {
  if (ctx == NULL || token_id < 0 || token_id >= (int64_t)AOT_VOCAB_SIZE)
    return ET_CD_ERR_ARGUMENT;
  ctx->logits[(uint32_t)token_id] = (float)value;
  ctx->mask_valid = 0u;
  return ET_CD_OK;
}

int32_t et_cd_aot_apply_mask(et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_CD_ERR_ARGUMENT;
  ctx->mask_valid = 0u;
  const et_cd_status st = et_cd_apply_logit_mask_v1(
      ctx->index, ctx->cur_state, ctx->logits, ctx->masked_logits);
  if (st == ET_CD_OK)
    ctx->mask_valid = 1u;
  return st;
}

int64_t et_cd_aot_masked_is_allowed(const et_cd_aot_ctx *ctx,
                                    int64_t token_id) {
  if (ctx == NULL || !ctx->mask_valid || token_id < 0 ||
      token_id >= (int64_t)AOT_VOCAB_SIZE)
    return 0;
  return isfinite(ctx->masked_logits[(uint32_t)token_id]) ? 1 : 0;
}

double et_cd_aot_get_masked_logit(const et_cd_aot_ctx *ctx, int64_t token_id) {
  if (ctx == NULL || !ctx->mask_valid || token_id < 0 ||
      token_id >= (int64_t)AOT_VOCAB_SIZE)
    return 0.0;
  return (double)ctx->masked_logits[(uint32_t)token_id];
}

int32_t et_cd_aot_advance_token(et_cd_aot_ctx *ctx, int64_t token_id) {
  if (ctx == NULL || token_id < 0 || token_id >= (int64_t)AOT_VOCAB_SIZE)
    return ET_CD_ERR_ARGUMENT;
  uint16_t next_state = 0u;
  uint32_t is_finished = 0u;
  const et_cd_status st = et_cd_advance_token_v1(
      ctx->index, ctx->cur_state, (uint32_t)token_id, &next_state,
      &is_finished);
  if (st != ET_CD_OK)
    return st;

  if (!is_finished) {
    const size_t tlen = ctx->token_lens[(uint32_t)token_id];
    if (ctx->output_len + tlen > AOT_MAX_OUTPUT_BYTES)
      return ET_CD_ERR_LIMIT;
    memcpy(ctx->output + ctx->output_len, ctx->token_ptrs[(uint32_t)token_id],
           tlen);
    ctx->output_len += tlen;
  }
  ctx->cur_state = next_state;
  ctx->finished = is_finished;
  ctx->mask_valid = 0u;
  return ET_CD_OK;
}

int64_t et_cd_aot_output_len(const et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1;
  return (int64_t)ctx->output_len;
}

int64_t et_cd_aot_output_byte(const et_cd_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || (size_t)index >= ctx->output_len)
    return -1;
  return (int64_t)ctx->output[(size_t)index];
}

int64_t et_cd_aot_output_accepted_by_dfa(const et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1;
  uint32_t accepted = 0u;
  if (et_cd_dfa_accepts_bytes(ctx->dfa, ctx->output, ctx->output_len,
                              &accepted) != ET_CD_OK)
    return -1;
  return (int64_t)accepted;
}

int32_t et_cd_aot_verify_artifact_roundtrip(et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_CD_ERR_ARGUMENT;
  size_t need = 0u;
  et_cd_status st = et_cd_dfa_artifact_size(ctx->dfa, &need);
  if (st != ET_CD_OK)
    return st;

  uint8_t *buf = (uint8_t *)malloc(need);
  if (buf == NULL)
    return ET_CD_ERR_NOMEM;

  size_t written = 0u;
  st = et_cd_dfa_serialize(ctx->dfa, buf, need, &written);
  if (st != ET_CD_OK || written != need) {
    free(buf);
    return st != ET_CD_OK ? st : ET_CD_ERR_FORMAT;
  }

  et_cd_dfa *reloaded = NULL;
  uint8_t sha[ET_CD_SHA256_BYTES];
  st = et_cd_dfa_parse(buf, written, &reloaded, sha);
  if (st != ET_CD_OK) {
    free(buf);
    return st;
  }

  uint8_t *buf2 = (uint8_t *)malloc(need);
  if (buf2 == NULL) {
    et_cd_dfa_destroy(reloaded);
    free(buf);
    return ET_CD_ERR_NOMEM;
  }
  size_t written2 = 0u;
  st = et_cd_dfa_serialize(reloaded, buf2, need, &written2);
  if (st == ET_CD_OK && (written2 != written || memcmp(buf, buf2, written) != 0))
    st = ET_CD_ERR_CHECKSUM;

  if (st == ET_CD_OK) {
    memcpy(ctx->artifact_sha256, sha, ET_CD_SHA256_BYTES);
    ctx->has_artifact_sha256 = 1;
  }
  free(buf2);
  et_cd_dfa_destroy(reloaded);
  free(buf);
  return st;
}

int64_t et_cd_aot_artifact_sha256_word(const et_cd_aot_ctx *ctx,
                                       int64_t word_index) {
  if (ctx == NULL || !ctx->has_artifact_sha256 || word_index < 0 ||
      word_index >= 4)
    return -1;
  const uint8_t *p = ctx->artifact_sha256 + (size_t)word_index * 8u;
  uint64_t v = 0u;
  for (size_t i = 0u; i < 8u; ++i)
    v = (v << 8u) | (uint64_t)p[i];
  /* Return non-negative signed 63-bit slice or exact int64 bit pattern. */
  return (int64_t)v;
}

int32_t et_cd_aot_check_dead_end_dfa(void) {
  uint16_t trans[2u * ET_CD_ALPHABET_SIZE];
  memset(trans, 0xFF, sizeof(trans));
  trans[(size_t)0u * ET_CD_ALPHABET_SIZE + (uint8_t)'a'] = 1u;
  trans[(size_t)1u * ET_CD_ALPHABET_SIZE + (uint8_t)'b'] = 0u;
  et_cd_dfa *dfa = NULL;
  const et_cd_status st = et_cd_dfa_build_raw(2u, 0u, NULL, 0u, trans, &dfa);
  et_cd_dfa_destroy(dfa);
  return st;
}

int32_t et_cd_aot_check_illegal_advance(const et_cd_aot_ctx *ctx,
                                        int64_t token_id) {
  if (ctx == NULL || token_id < 0 || token_id >= (int64_t)AOT_VOCAB_SIZE)
    return ET_CD_ERR_ARGUMENT;
  uint16_t sentinel_state = 0x1234u;
  uint32_t sentinel_fin = 99u;
  const et_cd_status st =
      et_cd_advance_token_v1(ctx->index, ctx->cur_state, (uint32_t)token_id,
                             &sentinel_state, &sentinel_fin);
  if (st != ET_CD_OK && (sentinel_state != 0x1234u || sentinel_fin != 99u))
    return ET_CD_ERR_STATE;
  return st;
}

int32_t et_cd_aot_check_nan_logit_preserves_output(et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_CD_ERR_ARGUMENT;
  float bad_logits[AOT_VOCAB_SIZE];
  float out_buf[AOT_VOCAB_SIZE];
  for (uint32_t i = 0u; i < AOT_VOCAB_SIZE; ++i) {
    bad_logits[i] = 1.0f;
    out_buf[i] = 7.25f;
  }
  const uint32_t nan_bits = UINT32_C(0x7FC00001);
  memcpy(&bad_logits[AOT_VOCAB_SIZE / 2u], &nan_bits, sizeof(float));

  const et_cd_status st =
      et_cd_apply_logit_mask_v1(ctx->index, ctx->cur_state, bad_logits, out_buf);
  for (uint32_t i = 0u; i < AOT_VOCAB_SIZE; ++i) {
    if (out_buf[i] != 7.25f)
      return ET_CD_ERR_STATE;
  }
  return st;
}

int32_t et_cd_aot_check_inf_logit_preserves_output(et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_CD_ERR_ARGUMENT;
  float bad_logits[AOT_VOCAB_SIZE];
  float out_buf[AOT_VOCAB_SIZE];
  for (uint32_t i = 0u; i < AOT_VOCAB_SIZE; ++i) {
    bad_logits[i] = -0.5f;
    out_buf[i] = -3.5f;
  }
  const uint32_t inf_bits = UINT32_C(0x7F800000);
  memcpy(&bad_logits[3u], &inf_bits, sizeof(float));

  const et_cd_status st =
      et_cd_apply_logit_mask_v1(ctx->index, ctx->cur_state, bad_logits, out_buf);
  for (uint32_t i = 0u; i < AOT_VOCAB_SIZE; ++i) {
    if (out_buf[i] != -3.5f)
      return ET_CD_ERR_STATE;
  }
  return st;
}

int32_t et_cd_aot_check_dead_state_mask(et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_CD_ERR_ARGUMENT;
  float logits[AOT_VOCAB_SIZE];
  float out_buf[AOT_VOCAB_SIZE];
  for (uint32_t i = 0u; i < AOT_VOCAB_SIZE; ++i) {
    logits[i] = 0.25f;
    out_buf[i] = 1.5f;
  }
  const et_cd_status st =
      et_cd_apply_logit_mask_v1(ctx->index, ET_CD_DEAD_STATE, logits, out_buf);
  for (uint32_t i = 0u; i < AOT_VOCAB_SIZE; ++i) {
    if (out_buf[i] != 1.5f)
      return ET_CD_ERR_STATE;
  }
  return st;
}

int32_t et_cd_aot_check_tampered_artifact(const et_cd_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_CD_ERR_ARGUMENT;
  size_t need = 0u;
  et_cd_status st = et_cd_dfa_artifact_size(ctx->dfa, &need);
  if (st != ET_CD_OK)
    return st;
  uint8_t *buf = (uint8_t *)malloc(need);
  if (buf == NULL)
    return ET_CD_ERR_NOMEM;
  size_t written = 0u;
  st = et_cd_dfa_serialize(ctx->dfa, buf, need, &written);
  if (st != ET_CD_OK) {
    free(buf);
    return st;
  }
  /* Flip one byte in the payload region. */
  buf[ET_CD_ARTIFACT_HEADER_BYTES] ^= 0x01u;
  et_cd_dfa *bad = NULL;
  st = et_cd_dfa_parse(buf, written, &bad, NULL);
  et_cd_dfa_destroy(bad);
  free(buf);
  return st;
}
