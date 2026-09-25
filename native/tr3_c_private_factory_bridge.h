#ifndef ET_TR3_C_PRIVATE_FACTORY_BRIDGE_H
#define ET_TR3_C_PRIVATE_FACTORY_BRIDGE_H

#include <stdint.h>
#include "tr3_c_private_initializer_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_tr3_c_private_trainer_handle_v1 et_tr3_c_handle_v1;
typedef struct {
  uint32_t size, major, minor, flags;
  const uint8_t *x1_json;
  uint32_t x1_len;
  const uint8_t *directory;
  uint32_t directory_len;
  uint8_t expected_manifest_sha256[32];
  int64_t batch_size, sequence_length, maximum_manifest_bytes;
  int64_t maximum_shard_bytes, maximum_total_tokens, maximum_batch_bytes;
  int64_t shuffle_seed, shuffle_window_rows;
  uint32_t adamw_f32_bits[5];
  uint8_t shuffle_seed_present, packing;
  uint8_t reserved[6];
} et_tr3_c_create_request_v1;

typedef struct {
  uint32_t status, stage, reason, original_category;
  et_tr3_c_handle_v1 *handle;
} et_tr3_c_result_v1;

enum {
  ET_TR3_C_OK = 0,
  ET_TR3_C_INVALID_ARGUMENT = 1,
  ET_TR3_C_SHAPE_MISMATCH = 2,
  ET_TR3_C_DTYPE_MISMATCH = 3,
  ET_TR3_C_DEVICE_MISMATCH = 4,
  ET_TR3_C_NONCONTIGUOUS = 5,
  ET_TR3_C_UNSUPPORTED = 6,
  ET_TR3_C_INVALID_STATE = 7,
  ET_TR3_C_IO = 8,
  ET_TR3_C_CORRUPT_DATA = 9,
  ET_TR3_C_VERSION_MISMATCH = 10,
  ET_TR3_C_DETERMINISM_UNAVAILABLE = 11,
  ET_TR3_C_INTERNAL = 12,
};
enum {
  ET_TR3_C_STAGE_NONE = 0,
  ET_TR3_C_STAGE_ADMISSION = 1,
  ET_TR3_C_STAGE_X1 = 2,
  ET_TR3_C_STAGE_T2 = 3,
  ET_TR3_C_STAGE_D2 = 4,
  ET_TR3_C_STAGE_CORPUS_IDENTITY = 5,
  ET_TR3_C_STAGE_M3T_INITIALIZER = 6,
  ET_TR3_C_STAGE_M3T_MODEL = 7,
  ET_TR3_C_STAGE_O2 = 8,
  ET_TR3_C_STAGE_LEASE = 9,
  ET_TR3_C_STAGE_CLOSE = 10,
};
enum {
  ET_TR3_C_REASON_RAISED_E1 = 0,
  ET_TR3_C_REASON_NOT_READY = 1,
  ET_TR3_C_REASON_BUSY = 2,
  ET_TR3_C_REASON_ATTEMPT_USED = 3,
  ET_TR3_C_REASON_BAD_HANDLE = 4,
  ET_TR3_C_REASON_ALREADY_CLOSED = 5,
  ET_TR3_C_REASON_DIGEST_MISMATCH = 6,
  ET_TR3_C_REASON_CLEANUP_FAILED = 7,
  ET_TR3_C_REASON_FOREIGN_EXCEPTION = 8,
  ET_TR3_C_REASON_MALFORMED_REQUEST = 9,
};

ET_TR3_C_PRIVATE_EXPORT et_tr3_c_result_v1
et_tr3_c_private_trainer_create_v1(const et_tr3_c_create_request_v1 *request);
ET_TR3_C_PRIVATE_EXPORT et_tr3_c_result_v1
et_tr3_c_private_trainer_close_v1(et_tr3_c_handle_v1 *handle);

#ifdef __cplusplus
}
#endif
#endif
