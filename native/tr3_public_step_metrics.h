#ifndef ET_TR3_PUBLIC_STEP_METRICS_H
#define ET_TR3_PUBLIC_STEP_METRICS_H

#include <stdint.h>

/* Private fixed-profile binary32 kernel. Buffers are Eshkol bytevectors:
 * accumulator [numerator, weight] (8 bytes), objective observation
 * [numerator, weight, mean] (12 bytes), all little-endian physical f32 words.
 * Each accumulation rounds once to binary32 in microbatch order. A negative
 * return is -1 for rejected/nonfinite input or result and -2 for unsupported
 * x87/MXCSR controls or floating-environment capture/restore failure. Success
 * is zero for add and the nonnegative finite f32 word for mean. Neither
 * rejection writes the accumulator. No trainer or metrics authority. */
int64_t et_tr3_step_metrics_accumulate_v1(void *accumulator, void *observation);
int64_t et_tr3_step_metrics_mean_bits_v1(void *accumulator);

#endif
