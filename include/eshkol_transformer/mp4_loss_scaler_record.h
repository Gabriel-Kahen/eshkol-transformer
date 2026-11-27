#ifndef ESHKOL_TRANSFORMER_MP4_LOSS_SCALER_RECORD_H
#define ESHKOL_TRANSFORMER_MP4_LOSS_SCALER_RECORD_H

#include "eshkol_transformer/mp4_mixed_precision_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MP4_LOSS_SCALER_RECORD_V1_BYTES 44u

enum {
  ET_MP4_LOSS_SCALER_RECORD_OK = 0,
  ET_MP4_LOSS_SCALER_RECORD_INVALID_ARGUMENT = 1,
  ET_MP4_LOSS_SCALER_RECORD_INVALID_RECORD = 2,
  ET_MP4_LOSS_SCALER_RECORD_UNSUPPORTED_VERSION = 3
};
typedef uint32_t et_mp4_loss_scaler_record_status;

/* Both calls require exactly 44 record bytes and disjoint record and scaler
 * storage. The scaler must be aligned; the byte record may be unaligned.
 * Callers must synchronize access to these spans. Failure leaves both spans
 * unchanged. See docs/MP4_MIXED_PRECISION.md. */
et_mp4_loss_scaler_record_status
et_mp4_loss_scaler_encode_v1(const et_mp4_loss_scaler_v1 *state,
                             uint8_t *record, size_t record_bytes);

/* On success, replaces every scaler field. Prior destination contents are
 * irrelevant. The record and its checksum are validated before any write. */
et_mp4_loss_scaler_record_status
et_mp4_loss_scaler_decode_v1(const uint8_t *record, size_t record_bytes,
                             et_mp4_loss_scaler_v1 *state);

#ifdef __cplusplus
}
#endif

#endif
