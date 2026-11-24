#ifndef ESHKOL_TRANSFORMER_CONSTRAINED_DECODE_ABI_H
#define ESHKOL_TRANSFORMER_CONSTRAINED_DECODE_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Hard Bounds and Format Constants ---- */

#define ET_CD_MAX_STATES 4096u
#define ET_CD_ALPHABET_SIZE 256u
#define ET_CD_DEAD_STATE 0xFFFFu

#define ET_CD_MAX_VOCAB_SIZE 8192u
#define ET_CD_MAX_TOKEN_BYTES 256u

#define ET_CD_MAX_SCHEMA_FIELDS 64u
#define ET_CD_MAX_CHOICES 256u
#define ET_CD_MAX_CHOICE_BYTES 256u
#define ET_CD_MAX_KEY_BYTES 128u

#define ET_CD_ACCEPT_WORDS_MAX 64u /* (ET_CD_MAX_STATES + 63u) / 64u */
#define ET_CD_VOCAB_WORDS_MAX 128u /* (ET_CD_MAX_VOCAB_SIZE + 63u) / 64u */

#define ET_CD_ARTIFACT_MAGIC "ETCD1000"
#define ET_CD_ARTIFACT_MAGIC_BYTES 8u
#define ET_CD_ARTIFACT_VERSION_MAJOR 1u
#define ET_CD_ARTIFACT_VERSION_MINOR 0u
#define ET_CD_ARTIFACT_HEADER_BYTES 32u
#define ET_CD_SHA256_BYTES 32u

/* ---- Status Codes ---- */

typedef int32_t et_cd_status;

enum {
  ET_CD_OK = 0,
  ET_CD_ERR_ARGUMENT = -1,
  ET_CD_ERR_LIMIT = -2,
  ET_CD_ERR_EMPTY_LANGUAGE = -3,
  ET_CD_ERR_NO_VALID_TOKENS = -4,
  ET_CD_ERR_ILLEGAL_TOKEN = -5,
  ET_CD_ERR_NONFINITE = -6,
  ET_CD_ERR_STATE = -7,
  ET_CD_ERR_SCHEMA = -8,
  ET_CD_ERR_NOMEM = -9,
  ET_CD_ERR_FORMAT = -10,
  ET_CD_ERR_VERSION = -11,
  ET_CD_ERR_TRUNCATED = -12,
  ET_CD_ERR_CHECKSUM = -13,
  ET_CD_ERR_IO = -14,
  ET_CD_ERR_SYMLINK = -15,
  ET_CD_ERR_OVERFLOW = -16,
  ET_CD_ERR_FP_ENV = -17,
  ET_CD_ERR_ALIASING = -18
};

/* ---- JSON Flat-Object Schema Types ---- */

typedef enum et_cd_json_field_type {
  ET_CD_JSON_STRING = 1,
  ET_CD_JSON_INTEGER = 2,
  ET_CD_JSON_BOOLEAN = 3,
  ET_CD_JSON_ENUM = 4
} et_cd_json_field_type;

typedef struct et_cd_json_field_spec {
  const char *key;                /* NUL-terminated printable ASCII key (no '"' or '\\') */
  int32_t type;                   /* et_cd_json_field_type */
  const char *const *enum_values; /* Required iff type == ET_CD_JSON_ENUM */
  size_t num_enum_values;         /* Must be in [1, ET_CD_MAX_CHOICES] for ENUM, else 0 */
} et_cd_json_field_spec;

/* ---- Public POD Metadata Structures ---- */

typedef struct et_cd_dfa_info {
  uint16_t num_states;
  uint16_t start_state;
  uint16_t live_state_count;
  uint16_t accepting_state_count;
} et_cd_dfa_info;

typedef struct et_cd_vocab_index_info {
  uint16_t num_states;
  uint16_t start_state;
  uint16_t live_state_count;
  uint16_t reserved;
  uint32_t vocab_size;
  uint32_t eos_token_id;
  uint32_t mask_words;
} et_cd_vocab_index_info;

/* ---- Opaque Compiled Automaton and Vocabulary Index Handles ---- */

typedef struct et_cd_dfa et_cd_dfa;
typedef struct et_cd_vocab_index et_cd_vocab_index;

/* ---- DFA Construction and Built-In Deterministic Schema Compilers ---- */

/* Build a byte-level DFA from an explicit row-major transition table
 * transitions_row_major[state * 256 + byte] (each entry < num_states or
 * ET_CD_DEAD_STATE). Runs reverse-BFS co-accessibility pruning from
 * accept_states; rejects empty languages (dead start_state) with
 * ET_CD_ERR_EMPTY_LANGUAGE. */
et_cd_status et_cd_dfa_build_raw(uint16_t num_states, uint16_t start_state,
                                 const uint16_t *accept_states,
                                 size_t num_accept_states,
                                 const uint16_t *transitions_row_major,
                                 et_cd_dfa **out_dfa);

/* Compile an exact finite choice grammar (union of non-empty byte strings,
 * no duplicates) into a deterministic co-accessible byte-level DFA. */
et_cd_status et_cd_dfa_compile_literal_choices(const uint8_t *const *choices,
                                               const size_t *choice_lengths,
                                               size_t num_choices,
                                               et_cd_dfa **out_dfa);

/* Compile a deterministic compact JSON object grammar with ordered typed fields:
 *   {"<k0>":<v0>,"<k1>":<v1>,...,"<k{n-1}>":<v{n-1}>}
 * Field grammars:
 *   ET_CD_JSON_STRING  : RFC 8259 JSON string with valid escapes and UTF-8
 *   ET_CD_JSON_INTEGER : -?(0|[1-9][0-9]*)
 *   ET_CD_JSON_BOOLEAN : true|false
 *   ET_CD_JSON_ENUM    : JSON-escaped string matching one of enum_values
 */
et_cd_status
et_cd_dfa_compile_flat_json_schema(const et_cd_json_field_spec *fields,
                                   size_t num_fields, et_cd_dfa **out_dfa);

void et_cd_dfa_destroy(et_cd_dfa *dfa);

/* ---- Zero-Allocation DFA Inspection ---- */

et_cd_status et_cd_dfa_info_get(const et_cd_dfa *dfa, et_cd_dfa_info *out_info);
uint16_t et_cd_dfa_next_state(const et_cd_dfa *dfa, uint16_t state,
                              uint8_t byte);
uint32_t et_cd_dfa_is_accepting(const et_cd_dfa *dfa, uint16_t state);
uint32_t et_cd_dfa_is_live(const et_cd_dfa *dfa, uint16_t state);
et_cd_status et_cd_dfa_accepts_bytes(const et_cd_dfa *dfa, const uint8_t *bytes,
                                     size_t len, uint32_t *out_accepted);

/* ---- Checksummed Binary Grammar Artifact (ETCD1000 v1.0, SHA-256) ---- */

et_cd_status et_cd_dfa_artifact_size(const et_cd_dfa *dfa, size_t *out_size);
et_cd_status et_cd_dfa_serialize(const et_cd_dfa *dfa, uint8_t *out_bytes,
                                 size_t capacity, size_t *out_size);
et_cd_status et_cd_dfa_parse(const uint8_t *bytes, size_t size,
                             et_cd_dfa **out_dfa,
                             uint8_t out_sha256[ET_CD_SHA256_BYTES]);
et_cd_status et_cd_dfa_save(const char *path, const et_cd_dfa *dfa);
et_cd_status et_cd_dfa_load(const char *path, et_cd_dfa **out_dfa,
                            uint8_t out_sha256[ET_CD_SHA256_BYTES]);

/* ---- Vocabulary Index Precomputation and Inspection ---- */

/* Bind a byte-level DFA to a token vocabulary of size vocab_size <= 8192.
 * For each live state s and token t:
 *   - If t == eos_token_id: allowed iff s is accepting.
 *   - Else if token_lengths[t] == 0: disallowed in every state.
 *   - Else: walk token_bytes[t][0..len-1] from s; allowed iff every byte
 *     transitions to a live state (never ET_CD_DEAD_STATE), recording the
 *     resulting destination state dest_state[s][t].
 */
et_cd_status et_cd_vocab_index_build(const et_cd_dfa *dfa,
                                     const uint8_t *const *token_bytes,
                                     const size_t *token_lengths,
                                     uint32_t vocab_size, uint32_t eos_token_id,
                                     et_cd_vocab_index **out_index);

void et_cd_vocab_index_destroy(et_cd_vocab_index *index);

et_cd_status et_cd_vocab_index_info_get(const et_cd_vocab_index *index,
                                        et_cd_vocab_index_info *out_info);
et_cd_status et_cd_vocab_index_allowed_count(const et_cd_vocab_index *index,
                                             uint16_t state,
                                             uint32_t *out_count);
uint32_t et_cd_vocab_index_is_token_allowed(const et_cd_vocab_index *index,
                                            uint16_t state, uint32_t token_id);
et_cd_status et_cd_vocab_index_copy_mask_words(const et_cd_vocab_index *index,
                                               uint16_t state,
                                               uint64_t *out_words,
                                               size_t max_words);

/* ---- Hot-Path Runtime Operations (ZERO Heap Allocations) ---- */

/* Validate all input logits[0..V-1] are finite; write bit-exact logits[t] for
 * allowed tokens and -INFINITY for disallowed tokens into masked_logits[0..V-1].
 * Fails closed with ET_CD_ERR_NO_VALID_TOKENS if state has zero allowed tokens,
 * ET_CD_ERR_STATE if state is invalid/dead, ET_CD_ERR_NONFINITE if any input
 * logit is non-finite. Always preserves masked_logits untouched on any error. */
et_cd_status et_cd_apply_logit_mask_v1(const et_cd_vocab_index *index,
                                       uint16_t state, const float *logits,
                                       float *masked_logits);

/* Validate token_id is allowed from state and advance deterministically.
 * On eos_token_id from an accepting state: sets *out_is_finished = 1 and
 * *out_next_state = ET_CD_DEAD_STATE.
 * On a non-EOS allowed token: sets *out_is_finished = 0 and *out_next_state
 * to the precomputed live destination state.
 * Rejects disallowed tokens with ET_CD_ERR_ILLEGAL_TOKEN and preserves outputs
 * untouched on any error. */
et_cd_status et_cd_advance_token_v1(const et_cd_vocab_index *index,
                                    uint16_t state, uint32_t token_id,
                                    uint16_t *out_next_state,
                                    uint32_t *out_is_finished);

#ifdef __cplusplus
}
#endif

#endif
