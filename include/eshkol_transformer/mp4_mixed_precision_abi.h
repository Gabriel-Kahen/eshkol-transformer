#ifndef ESHKOL_TRANSFORMER_MP4_MIXED_PRECISION_ABI_H
#define ESHKOL_TRANSFORMER_MP4_MIXED_PRECISION_ABI_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MP4_MIXED_PRECISION_ABI_MAJOR 1u
#define ET_MP4_MIXED_PRECISION_ABI_MINOR 0u

/* Explicit K1 provider. See docs/MP4_MIXED_PRECISION.md for tensor schemas. */
const et_kernel_provider_v1 *et_mp4_kernel_provider_v1(void);

typedef struct et_mp4_loss_scaler_v1 {
  float scale;
  float growth_factor;
  float backoff_factor;
  uint32_t growth_interval;
  uint32_t clean_steps;
  float min_scale;
  float max_scale;
} et_mp4_loss_scaler_v1;

enum {
  ET_MP4_LOSS_SCALER_STEP_ACCEPTED = 0,
  ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF = 1,
  ET_MP4_LOSS_SCALER_UNDERFLOW_FAULT = 2,
  ET_MP4_LOSS_SCALER_INVALID_ARGUMENT = 3,
  ET_MP4_LOSS_SCALER_UNSUPPORTED_ENVIRONMENT = 4
};
typedef uint32_t et_mp4_loss_scaler_status;

/* Initializes clean_steps to zero. STEP_ACCEPTED denotes successful init. */
et_mp4_loss_scaler_status
et_mp4_loss_scaler_init(et_mp4_loss_scaler_v1 *state, float scale,
                        float growth_factor, float backoff_factor,
                        uint32_t growth_interval, float min_scale,
                        float max_scale);

/* Only STEP_ACCEPTED and OVERFLOW_BACKOFF commit state. Only STEP_ACCEPTED
 * permits an optimizer step; skip the optimizer step on OVERFLOW_BACKOFF and
 * every fault or error status. UNDERFLOW_FAULT requires caller intervention. */
et_mp4_loss_scaler_status
et_mp4_loss_scaler_update(et_mp4_loss_scaler_v1 *state,
                          uint8_t overflow_detected);

#ifdef __cplusplus
}
#endif

#endif
