#ifndef ESHKOL_TRANSFORMER_C2_CHECKPOINT_CODEC_H
#define ESHKOL_TRANSFORMER_C2_CHECKPOINT_CODEC_H

#include "c2_checkpoint_format.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_c2_checkpoint_encode_request_v1 {
  size_t struct_size;
  const uint8_t *tokenizer_fingerprint;
  size_t tokenizer_fingerprint_bytes;
  const uint8_t *x1_canonical;
  size_t x1_canonical_bytes;
  const uint8_t *current_cursor;
  size_t current_cursor_bytes;
  const uint8_t *epoch_cursor;
  size_t epoch_cursor_bytes;
  const uint8_t *model_c1;
  size_t model_c1_bytes;
  const uint8_t *optimizer_metadata;
  size_t optimizer_metadata_bytes;
  const uint8_t *optimizer_payload;
  size_t optimizer_payload_bytes;
  uint64_t total_tokens;
  uint64_t epochs;
  uint64_t rng_key_bits;
  uint64_t rng_counter_low_bits;
  uint64_t rng_counter_high_bits;
} et_c2_checkpoint_encode_request_v1;

#define ET_C2_CHECKPOINT_ENCODE_REQUEST_V1_SIZE \
  ((size_t)sizeof(et_c2_checkpoint_encode_request_v1))

/*
 * Inputs are borrowed for the call and must already have passed complete C2
 * semantic and cross-component validation, including canonical X1, both D2
 * cursors, C1 alias topology, and O2 configuration/path/shape relationships.
 * model_c1 is one complete canonical C1 container. optimizer_metadata is one
 * complete exact O2 wire-metadata span; the encoder ignores and recomputes both
 * 32-byte digest fields in every parameter record. The request, error, and
 * input spans may not overlap either output. This primitive does not replace
 * the validator; parse the assembled image before publication.
 * Any measure failure leaves file_bytes intact.
 */
int32_t et_c2_checkpoint_encode_measure_v1(
    const et_c2_checkpoint_encode_request_v1 *request, size_t *file_bytes,
    et_c2_checkpoint_format_error_v1 *error);

/* destination_bytes must equal the measured size. Any failure leaves it intact. */
int32_t et_c2_checkpoint_encode_v1(
    const et_c2_checkpoint_encode_request_v1 *request, uint8_t *destination,
    size_t destination_bytes, et_c2_checkpoint_format_error_v1 *error);

#ifdef __cplusplus
}
#endif

#endif
