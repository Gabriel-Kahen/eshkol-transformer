#ifndef ESHKOL_TRANSFORMER_MP4_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_MP4_ESHKOL_BRIDGE_H

#include "eshkol_transformer/kernel_abi.h"
#include "eshkol_transformer/mp4_mixed_precision_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  ET_MP4_AOT_DTYPE_F32 = 0,
  ET_MP4_AOT_DTYPE_F16 = 1,
  ET_MP4_AOT_DTYPE_BF16 = 2,
  ET_MP4_AOT_DTYPE_BOOL = 3
};

/* Standard-layout, trivially-copyable dynamic loss-scaler wrapper for AOT
 * callers. Embeds the MP4 ABI 1.0 scaler state plus initialization and
 * overflow tracking with zero implicit padding on x86-64. */
typedef struct et_mp4_aot_scaler {
  et_mp4_loss_scaler_v1 state;
  uint32_t initialized;
  uint64_t overflow_count;
} et_mp4_aot_scaler;

#if defined(__cplusplus)
static_assert(sizeof(et_mp4_aot_scaler) == 40u,
              "MP4 AOT scaler layout changed");
static_assert(offsetof(et_mp4_aot_scaler, initialized) == 28u,
              "MP4 AOT scaler initialized offset changed");
static_assert(offsetof(et_mp4_aot_scaler, overflow_count) == 32u,
              "MP4 AOT scaler overflow_count offset changed");
#else
_Static_assert(sizeof(et_mp4_aot_scaler) == 40u,
               "MP4 AOT scaler layout changed");
_Static_assert(offsetof(et_mp4_aot_scaler, initialized) == 28u,
               "MP4 AOT scaler initialized offset changed");
_Static_assert(offsetof(et_mp4_aot_scaler, overflow_count) == 32u,
               "MP4 AOT scaler overflow_count offset changed");
#endif

/* Private AOT transport over the MP4 K1 ABI 1.0 provider.
 * Status is zero on success or an explicit K1 / MP4 error code; there are
 * no silent fallbacks. Pointer arguments accept void * so misaligned-pointer
 * negative tests never require undefined typed pointer casts in C11. */
int32_t et_mp4_aot_open(et_kernel_runtime **runtime);
void et_mp4_aot_close(et_kernel_runtime *runtime);

/* Storage sizing helpers. Return -1 on invalid dtype, count, or overflow. */
int64_t et_mp4_aot_element_bytes(int32_t dtype);
int64_t et_mp4_aot_tensor_bytes(int32_t dtype, int64_t count);
int64_t et_mp4_aot_scaler_bytes(void);

/* Narrowing and widening cast dispatch through K1 (verified 1D rows). */
int32_t et_mp4_aot_cast_f32_to_f16(const et_kernel_runtime *runtime,
                                   int64_t count, const void *src_f32,
                                   void *dst_f16);
int32_t et_mp4_aot_cast_f16_to_f32(const et_kernel_runtime *runtime,
                                   int64_t count, const void *src_f16,
                                   void *dst_f32);
int32_t et_mp4_aot_cast_f32_to_bf16(const et_kernel_runtime *runtime,
                                    int64_t count, const void *src_f32,
                                    void *dst_bf16);
int32_t et_mp4_aot_cast_bf16_to_f32(const et_kernel_runtime *runtime,
                                    int64_t count, const void *src_bf16,
                                    void *dst_f32);

/* Direct K1 gradient unscale dispatch (narrow[count], f32[] -> f32[count],
 * bool[]). On overflow, dispatch returns 0, sets *overflow_out = 1, and
 * zeroes dst_f32 to +0.0f. */
int32_t et_mp4_aot_grad_unscale_f16(const et_kernel_runtime *runtime,
                                    int64_t count, const void *src_f16,
                                    const void *inv_scale_f32, void *dst_f32,
                                    void *overflow_out);
int32_t et_mp4_aot_grad_unscale_bf16(const et_kernel_runtime *runtime,
                                     int64_t count, const void *src_bf16,
                                     const void *inv_scale_f32, void *dst_f32,
                                     void *overflow_out);

/* Dynamic loss-scaler wrapper around et_mp4_loss_scaler_init/update. */
int32_t et_mp4_aot_scaler_init(void *scaler, double init_scale,
                               double growth_factor, double backoff_factor,
                               int64_t growth_interval, double min_scale,
                               double max_scale);
double et_mp4_aot_scaler_scale(const void *scaler, double value);
int32_t et_mp4_aot_scaler_scale_f32(const void *scaler, int64_t count,
                                    const void *src_f32, void *dst_f32);
int32_t et_mp4_aot_scaler_unscale_f16(const et_kernel_runtime *runtime,
                                      const void *scaler, int64_t count,
                                      const void *src_f16, void *dst_f32,
                                      void *overflow_out);
int32_t et_mp4_aot_scaler_unscale_bf16(const et_kernel_runtime *runtime,
                                       const void *scaler, int64_t count,
                                       const void *src_bf16, void *dst_f32,
                                       void *overflow_out);
int32_t et_mp4_aot_scaler_step(void *scaler, int32_t overflow_detected);
double et_mp4_aot_scaler_get_scale(const void *scaler);
int64_t et_mp4_aot_scaler_get_growth_tracker(const void *scaler);
int64_t et_mp4_aot_scaler_get_overflow_count(const void *scaler);

/* Scalar buffer marshal helpers with explicit null, alignment, and bounds
 * checks. Stores return 0 or ET_KERNEL_ERROR_INVALID_ARGUMENT without writing.
 * Loads return NAN (f32) or -1 (integer bit accessors) on invalid input. */
int32_t et_mp4_aot_store_f32(void *buffer, int64_t capacity, int64_t index,
                             double value);
double et_mp4_aot_load_f32(const void *buffer, int64_t capacity, int64_t index);
int32_t et_mp4_aot_store_f32_bits(void *buffer, int64_t capacity, int64_t index,
                                  int64_t bits);
int64_t et_mp4_aot_load_f32_bits(const void *buffer, int64_t capacity,
                                 int64_t index);
int32_t et_mp4_aot_store_u16(void *buffer, int64_t capacity, int64_t index,
                             int64_t bits);
int64_t et_mp4_aot_load_u16(const void *buffer, int64_t capacity,
                            int64_t index);
int32_t et_mp4_aot_store_u8(void *buffer, int64_t capacity, int64_t index,
                            int64_t value);
int64_t et_mp4_aot_load_u8(const void *buffer, int64_t capacity, int64_t index);

#ifdef __cplusplus
}
#endif

#endif
