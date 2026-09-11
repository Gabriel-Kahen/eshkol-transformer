#ifndef ESHKOL_TRANSFORMER_C2_X1_CANONICAL_H
#define ESHKOL_TRANSFORMER_C2_X1_CANONICAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_C2_X1_MAX_CANONICAL_BYTES ((size_t)16384u)

enum et_c2_x1_status {
  ET_C2_X1_OK = 0,
  ET_C2_X1_INVALID_ARGUMENT = 1,
  ET_C2_X1_CORRUPT_DATA = 2,
  ET_C2_X1_VERSION_MISMATCH = 3,
  ET_C2_X1_UNSUPPORTED = 4,
  ET_C2_X1_SHAPE_MISMATCH = 6,
  ET_C2_X1_DTYPE_MISMATCH = 7,
  ET_C2_X1_DEVICE_MISMATCH = 8
};

enum et_c2_x1_code {
  ET_C2_X1_CODE_NONE = 0,
  ET_C2_X1_CODE_ARGUMENT = 1,
  ET_C2_X1_CODE_LIMIT = 2,
  ET_C2_X1_CODE_UTF8 = 3,
  ET_C2_X1_CODE_CANONICAL = 4,
  ET_C2_X1_CODE_VERSION = 5,
  ET_C2_X1_CODE_FINGERPRINT = 6,
  ET_C2_X1_CODE_PROVENANCE = 7,
  ET_C2_X1_CODE_DIMENSION = 8
};

typedef struct et_c2_x1_projection_v1 {
  size_t struct_size;
  uint64_t vocabulary_size;
  uint64_t context_length;
  uint64_t hidden_size;
  uint64_t layer_count;
  uint64_t query_head_count;
  uint64_t kv_head_count;
  uint64_t head_size;
  uint64_t seed;
  uint64_t accumulation_steps;
} et_c2_x1_projection_v1;

typedef struct et_c2_x1_error_v1 {
  size_t struct_size;
  uint32_t category;
  uint32_t code;
  uint64_t offset;
} et_c2_x1_error_v1;

/*
 * Allocation-free validation of one already-retained canonical X1 image.
 * Projection remains unchanged on failure. Input, projection, and error spans
 * must not overlap.
 */
int32_t et_c2_private_x1_canonical_inspect_v1(
    const uint8_t *canonical, size_t canonical_bytes,
    const uint8_t *expected_fingerprint, size_t expected_fingerprint_bytes,
    et_c2_x1_projection_v1 *projection, et_c2_x1_error_v1 *error);

/* Eshkol-private bytevector bridge. Writes nine little-endian u64 fields. */
int64_t
et_c2_private_x1_canonical_inspect_bytevectors_v1(void *canonical_bytevector,
                                                  void *fingerprint_bytevector,
                                                  void *projection_bytevector);

#ifdef __cplusplus
}
#endif

#endif
