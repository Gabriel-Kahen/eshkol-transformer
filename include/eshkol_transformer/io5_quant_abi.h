#ifndef ESHKOL_TRANSFORMER_IO5_QUANT_ABI_H
#define ESHKOL_TRANSFORMER_IO5_QUANT_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_IO5_QUANT_ABI_MAJOR 1u
#define ET_IO5_QUANT_ABI_MINOR 0u

/* Standard GGML Q8_0 block geometry: 32 elements = 2-byte LE f16 scale + 32 i8. */
#define ET_IO5_QK8_0 32u
#define ET_IO5_Q8_0_BLOCK_BYTES 34u

enum {
  ET_IO5_QUANT_OK = 0u,
  ET_IO5_QUANT_ERR_INVALID_ARG = 1u,
  ET_IO5_QUANT_ERR_INVALID_LENGTH = 2u,
  ET_IO5_QUANT_ERR_ALIAS = 3u,
  ET_IO5_QUANT_ERR_NONFINITE = 4u,
  ET_IO5_QUANT_ERR_OVERFLOW = 5u
};
typedef uint32_t et_io5_quant_status;

/* Returns (ET_IO5_QUANT_ABI_MAJOR << 16) | ET_IO5_QUANT_ABI_MINOR. */
uint32_t et_io5_quant_abi_version(void);

/* Checked byte-size calculation for Q8_0 (requires element_count > 0 and % 32 == 0). */
et_io5_quant_status et_io5_q8_0_byte_size(size_t element_count,
                                          size_t *out_bytes);

/* Scalar IEEE-754 RNE finite-only conversions matching MP4 bit semantics.
 * All NaN/Inf inputs and out-of-range finite inputs (|x| > 65504.0 for f16,
 * or values rounding to infinity for bf16) are rejected explicitly. */
et_io5_quant_status et_io5_f32_bits_to_f16_bits(uint32_t f32_bits,
                                                uint16_t *out_f16_bits);
et_io5_quant_status et_io5_f16_bits_to_f32_bits(uint16_t f16_bits,
                                                uint32_t *out_f32_bits);
et_io5_quant_status et_io5_f32_bits_to_bf16_bits(uint32_t f32_bits,
                                                 uint16_t *out_bf16_bits);
et_io5_quant_status et_io5_bf16_bits_to_f32_bits(uint16_t bf16_bits,
                                                 uint32_t *out_f32_bits);

/* Validates that every little-endian binary32 element in src_f32_le is finite. */
et_io5_quant_status et_io5_validate_f32_finite(const void *src_f32_le,
                                               size_t src_bytes,
                                               size_t element_count);

/* Bulk byte-buffer codecs.
 * All bulk APIs operate on little-endian byte representations without pointer
 * alignment requirements (zero misaligned-pointer UB), reject NULL or
 * overlapping buffers, validate all inputs before writing any output bytes,
 * and perform zero heap allocations. */
et_io5_quant_status et_io5_encode_f16(const void *src_f32_le, size_t src_bytes,
                                      void *dst_f16_le, size_t dst_bytes,
                                      size_t element_count);
et_io5_quant_status et_io5_decode_f16(const void *src_f16_le, size_t src_bytes,
                                      void *dst_f32_le, size_t dst_bytes,
                                      size_t element_count);

et_io5_quant_status et_io5_encode_bf16(const void *src_f32_le, size_t src_bytes,
                                       void *dst_bf16_le, size_t dst_bytes,
                                       size_t element_count);
et_io5_quant_status et_io5_decode_bf16(const void *src_bf16_le,
                                       size_t src_bytes, void *dst_f32_le,
                                       size_t dst_bytes, size_t element_count);

/* Standard GGML Q8_0 block quantization and dequantization.
 * Rejects element_count == 0 or element_count % 32 != 0, mismatched buffer
 * lengths, NULL or aliased buffers, non-finite f32 inputs, scales exceeding
 * finite f16 max (65504.0), and non-finite f16 block scales on decode. */
et_io5_quant_status et_io5_quantize_q8_0(const void *src_f32_le,
                                         size_t src_bytes, void *dst_q8_0,
                                         size_t dst_bytes,
                                         size_t element_count);
et_io5_quant_status et_io5_dequantize_q8_0(const void *src_q8_0,
                                           size_t src_bytes, void *dst_f32_le,
                                           size_t dst_bytes,
                                           size_t element_count);

#ifdef __cplusplus
}
#endif

#endif
