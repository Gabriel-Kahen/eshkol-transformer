#ifndef ESHKOL_TRANSFORMER_C2_CHECKPOINT_FORMAT_H
#define ESHKOL_TRANSFORMER_C2_CHECKPOINT_FORMAT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_C2_CHECKPOINT_FORMAT_ABI_MAJOR 1u
#define ET_C2_CHECKPOINT_FORMAT_ABI_MINOR 0u

#define ET_C2_WIRE_MAX_FILE_BYTES UINT64_C(1099511627776)
#define ET_C2_WIRE_MAX_METADATA_BYTES UINT64_C(268435456)
#define ET_C2_WIRE_MAX_TENSOR_BYTES UINT64_C(274877906944)
#define ET_C2_WIRE_MAX_TENSORS UINT32_C(6826)

#define ET_C2_PROFILE_MAX_FILE_BYTES UINT64_C(16777216)
#define ET_C2_PROFILE_MAX_METADATA_BYTES UINT64_C(524288)
#define ET_C2_PROFILE_MAX_TENSOR_BYTES UINT64_C(8388608)
#define ET_C2_PROFILE_MAX_TENSORS UINT32_C(64)

enum et_c2_checkpoint_format_status {
  ET_C2_FORMAT_OK = 0,
  ET_C2_FORMAT_INVALID_ARGUMENT = 1,
  ET_C2_FORMAT_CORRUPT_DATA = 2,
  ET_C2_FORMAT_VERSION_MISMATCH = 3,
  ET_C2_FORMAT_UNSUPPORTED = 4,
  ET_C2_FORMAT_DETERMINISM_UNAVAILABLE = 5,
  ET_C2_FORMAT_SHAPE_MISMATCH = 6,
  ET_C2_FORMAT_DTYPE_MISMATCH = 7,
  ET_C2_FORMAT_DEVICE_MISMATCH = 8,
  ET_C2_FORMAT_NONCONTIGUOUS = 9
};

enum et_c2_checkpoint_format_mode {
  ET_C2_FORMAT_INSPECT = 1,
  ET_C2_FORMAT_LOAD = 2
};

enum et_c2_checkpoint_format_code {
  ET_C2_FORMAT_CODE_NONE = 0,
  ET_C2_FORMAT_CODE_ARGUMENT = 1,
  ET_C2_FORMAT_CODE_TRUNCATED = 2,
  ET_C2_FORMAT_CODE_MAGIC = 3,
  ET_C2_FORMAT_CODE_VERSION = 4,
  ET_C2_FORMAT_CODE_FIXED_FIELD = 5,
  ET_C2_FORMAT_CODE_RESERVED = 6,
  ET_C2_FORMAT_CODE_LIMIT = 7,
  ET_C2_FORMAT_CODE_OVERFLOW = 8,
  ET_C2_FORMAT_CODE_SPAN = 9,
  ET_C2_FORMAT_CODE_CHECKSUM = 10,
  ET_C2_FORMAT_CODE_UTF8 = 11,
  ET_C2_FORMAT_CODE_IDENTITY = 12,
  ET_C2_FORMAT_CODE_ORDER = 13,
  ET_C2_FORMAT_CODE_SHAPE = 14,
  ET_C2_FORMAT_CODE_ALIAS = 15,
  ET_C2_FORMAT_CODE_CURSOR = 16,
  ET_C2_FORMAT_CODE_OPTIMIZER = 17,
  ET_C2_FORMAT_CODE_MOMENT = 18,
  ET_C2_FORMAT_CODE_PROFILE = 19
};

typedef struct et_c2_checkpoint_limits_v1 {
  size_t struct_size;
  uint64_t maximum_file_bytes;
  uint64_t maximum_metadata_bytes;
  uint64_t maximum_tensor_bytes;
  uint32_t maximum_tensors;
  uint32_t enforce_operational_profile;
} et_c2_checkpoint_limits_v1;

typedef struct et_c2_checkpoint_view_v1 {
  size_t struct_size;
  const uint8_t *bytes;
  uint64_t file_bytes;
  uint64_t outer_metadata_offset;
  uint64_t outer_metadata_bytes;
  uint64_t physical_payload_offset;
  uint64_t physical_payload_bytes;
  uint64_t model_container_offset;
  uint64_t model_container_bytes;
  uint64_t optimizer_metadata_offset;
  uint64_t optimizer_metadata_bytes;
  uint64_t optimizer_payload_offset;
  uint64_t optimizer_payload_bytes;
  uint64_t x1_offset;
  uint64_t x1_bytes;
  uint64_t current_cursor_offset;
  uint64_t epoch_cursor_offset;
  uint64_t total_tokens;
  uint64_t completed_updates;
  uint64_t epochs;
  uint64_t rng_key_bits;
  uint64_t rng_counter_low_bits;
  uint64_t rng_counter_high_bits;
  uint32_t model_tensor_count;
  uint32_t unique_parameter_count;
  uint32_t optimizer_tensor_count;
  uint32_t total_tensor_count;
  uint32_t optimizer_group_count;
  uint32_t tokenizer_fingerprint_bytes;
} et_c2_checkpoint_view_v1;

typedef struct et_c2_checkpoint_format_error_v1 {
  size_t struct_size;
  uint32_t category;
  uint32_t code;
  uint64_t offset;
} et_c2_checkpoint_format_error_v1;

#define ET_C2_CHECKPOINT_LIMITS_V1_SIZE                                    \
  ((size_t)sizeof(et_c2_checkpoint_limits_v1))
#define ET_C2_CHECKPOINT_VIEW_V1_SIZE                                      \
  ((size_t)sizeof(et_c2_checkpoint_view_v1))
#define ET_C2_CHECKPOINT_FORMAT_ERROR_V1_SIZE                              \
  ((size_t)sizeof(et_c2_checkpoint_format_error_v1))

/*
 * Validate one already-retained immutable byte image.  The function allocates,
 * frees, calls back, decodes, reconstructs, or performs filesystem I/O.
 * Success covers the outer digest and identities, canonical X1 projection and
 * fingerprint, both D2 cursor checksums/continuation invariants, the complete
 * embedded C1 table/digests/alias topology, and O2 groups/records/moments.
 * The returned spans borrow `bytes` and remain valid only while that retained
 * image is unchanged and alive.
 */
int32_t et_c2_checkpoint_parse_v1(
    const uint8_t *bytes, size_t byte_count,
    const et_c2_checkpoint_limits_v1 *limits, uint32_t mode,
    et_c2_checkpoint_view_v1 *view,
    et_c2_checkpoint_format_error_v1 *error);

#ifdef __cplusplus
}
#endif

#endif
