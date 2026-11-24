#ifndef ESHKOL_TRANSFORMER_W4A32_QUANT_ABI_H
#define ESHKOL_TRANSFORMER_W4A32_QUANT_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_W4A32_ABI_MAJOR 1u
#define ET_W4A32_ABI_MINOR 0u

/* Bounded dimensions preventing size_t overflow while covering realistic
 * transformer projections and MLPs. */
#define ET_W4A32_MAX_M 4096u
#define ET_W4A32_MAX_N 4096u
#define ET_W4A32_MAX_K 4096u

/* Quantized weight formats and exact block geometries. */
#define ET_W4A32_Q4_0_GROUP_SIZE 32u
#define ET_W4A32_Q4_0_BLOCK_BYTES 18u

#define ET_W4A32_GROUP_SIZE_32 32u
#define ET_W4A32_GROUP_SIZE_64 64u
#define ET_W4A32_GROUP_SIZE_128 128u

#define ET_W4A32_ASYM_HEADER_BYTES 4u
#define ET_W4A32_ASYM_BLOCK_BYTES(group_size) \
  (ET_W4A32_ASYM_HEADER_BYTES + ((group_size) / 2u))

/* Checksummed quantized weight container format (ETQ41000 v1.0). */
#define ET_W4A32_SHA256_BYTES 32u
#define ET_W4A32_CONTAINER_MAGIC "ETQ41000"
#define ET_W4A32_CONTAINER_MAGIC_BYTES 8u
#define ET_W4A32_CONTAINER_VERSION_MAJOR 1u
#define ET_W4A32_CONTAINER_VERSION_MINOR 0u
#define ET_W4A32_CONTAINER_HEADER_BYTES 48u
#define ET_W4A32_CONTAINER_CHECKSUM_BYTES 32u
#define ET_W4A32_CONTAINER_MIN_BYTES \
  (ET_W4A32_CONTAINER_HEADER_BYTES + ET_W4A32_CONTAINER_CHECKSUM_BYTES)
/* Worst-case packed weight payload: MAX_N * (MAX_K / 32) * 20 bytes (Q4_ASYM G=32). */
#define ET_W4A32_CONTAINER_MAX_PAYLOAD_BYTES \
  ((size_t)ET_W4A32_MAX_N * ((size_t)ET_W4A32_MAX_K / 32u) * 20u)
#define ET_W4A32_CONTAINER_MAX_BYTES \
  (ET_W4A32_CONTAINER_MIN_BYTES + ET_W4A32_CONTAINER_MAX_PAYLOAD_BYTES)

typedef int32_t et_w4a32_status;
typedef uint32_t et_w4a32_format;

enum {
  ET_W4A32_FORMAT_Q4_0 = 1u,
  ET_W4A32_FORMAT_Q4_ASYM = 2u
};

enum {
  ET_W4A32_OK = 0,
  ET_W4A32_ERR_ARGUMENT = -1,
  ET_W4A32_ERR_SHAPE = -2,
  ET_W4A32_ERR_ALIGNMENT = -3,
  ET_W4A32_ERR_ALIAS = -4,
  ET_W4A32_ERR_NONFINITE = -5,
  ET_W4A32_ERR_OVERFLOW = -6,
  ET_W4A32_ERR_FP_ENV = -7,
  ET_W4A32_ERR_FORMAT = -8,
  ET_W4A32_ERR_VERSION = -9,
  ET_W4A32_ERR_TRUNCATED = -10,
  ET_W4A32_ERR_CHECKSUM = -11,
  ET_W4A32_ERR_SYMLINK = -12,
  ET_W4A32_ERR_IO = -13,
  ET_W4A32_ERR_NOMEM = -14
};

/* Metadata decoded from an ETQ41000 v1.0 quantized weight container. */
typedef struct et_w4a32_container_meta {
  uint32_t version_major;
  uint32_t version_minor;
  uint32_t format;
  uint32_t group_size;
  uint32_t rows_n;
  uint32_t cols_k;
  uint64_t payload_bytes;
  uint8_t sha256[ET_W4A32_SHA256_BYTES];
} et_w4a32_container_meta;

#if defined(__cplusplus)
#define ET_W4A32_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_W4A32_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_W4A32_STATIC_ASSERT(sizeof(et_w4a32_container_meta) == 64u &&
                           offsetof(et_w4a32_container_meta, payload_bytes) ==
                               24u &&
                           offsetof(et_w4a32_container_meta, sha256) == 32u,
                       "et_w4a32_container_meta ABI layout changed");
#undef ET_W4A32_STATIC_ASSERT

/* Exact packed byte size for row-aligned quantized weights [rows_n, cols_k].
 * Blocks never cross rows; cols_k must be a positive multiple of group_size.
 * For ET_W4A32_FORMAT_Q4_0, group_size must be 32.
 * For ET_W4A32_FORMAT_Q4_ASYM, group_size must be 32, 64, or 128. */
et_w4a32_status et_w4a32_packed_bytes(et_w4a32_format format,
                                      uint32_t group_size, uint32_t rows_n,
                                      uint32_t cols_k, size_t *out_bytes);

/* Exact serialized byte size of an ETQ41000 v1.0 container. */
et_w4a32_status et_w4a32_container_size(et_w4a32_format format,
                                        uint32_t group_size, uint32_t rows_n,
                                        uint32_t cols_k, size_t *out_bytes);

/* IEEE-754 binary32 <-> binary16 round-to-nearest-even conversion helpers.
 * Rejects non-finite inputs (ET_W4A32_ERR_NONFINITE) and binary16 overflow
 * (|value| >= 65520.0f -> ET_W4A32_ERR_OVERFLOW). */
et_w4a32_status et_w4a32_f32_to_f16(float value, uint16_t *out_bits);
et_w4a32_status et_w4a32_f16_to_f32(uint16_t bits, float *out_value);

/* Standalone SHA-256 over arbitrary byte buffer (zero heap allocations). */
et_w4a32_status et_w4a32_sha256_bytes(const uint8_t *data, size_t size,
                                      uint8_t out_digest[ET_W4A32_SHA256_BYTES]);

/* ---- GGML Q4_0 (group_size = 32, 18 bytes/block) ----
 * Block layout (18 bytes, little-endian, row-aligned, never crossing rows):
 *   bytes [0..1]:  IEEE-754 binary16 scale d (little-endian)
 *   bytes [2..17]: qs[16] split-half nibbles:
 *                  low nibble  (qs[j] & 0x0F)        -> element j       (j=0..15)
 *                  high nibble ((qs[j] >> 4) & 0x0F) -> element j + 16  (j=0..15)
 * Reconstruction:  w_hat = d * ((int32_t)q - 8), q in [0, 15].
 *
 * All Q4_0 kernels are two-pass validate-before-write with zero heap
 * allocations; any error leaves output buffers completely untouched. */
et_w4a32_status et_w4a32_q4_0_quantize_v1(uint32_t rows_n, uint32_t cols_k,
                                          const float *weights_f32,
                                          uint8_t *out_packed,
                                          size_t out_packed_bytes);

et_w4a32_status et_w4a32_q4_0_dequantize_v1(uint32_t rows_n, uint32_t cols_k,
                                            const uint8_t *packed,
                                            size_t packed_bytes,
                                            float *out_weights_f32);

/* Fused Q4_0 dequant-matmul:
 *   Y[m, n] = X[m, k] @ dequant(W_q4_0[n, k])^T + (bias ? bias[n] : 0)
 * Accumulates directly from packed 4-bit blocks in binary64 without
 * materializing an [n, k] binary32 weight buffer.
 * Supports M = 1 (decode GEMV) and M > 1 (prefill GEMM). */
et_w4a32_status et_w4a32_q4_0_gemm_v1(uint32_t m, uint32_t n, uint32_t k,
                                      const float *x, const uint8_t *w_q4_0,
                                      size_t w_q4_0_bytes, const float *bias,
                                      float *y);

/* ---- Asymmetric Group-Wise Q4_ASYM (group_size in {32, 64, 128}) ----
 * Block layout (4 + group_size/2 bytes, little-endian, row-aligned):
 *   bytes [0..1]: IEEE-754 binary16 scale d (little-endian)
 *   bytes [2..3]: IEEE-754 binary16 min/offset m (little-endian)
 *   bytes [4..4+half-1]: qs[half] split-half nibbles (half = group_size / 2):
 *                        low nibble  (qs[j] & 0x0F)        -> element j
 *                        high nibble ((qs[j] >> 4) & 0x0F) -> element j + half
 * Reconstruction:  w_hat = d * (float)q + m, q in [0, 15].
 *
 * All Q4_ASYM kernels are two-pass validate-before-write with zero heap
 * allocations; any error leaves output buffers completely untouched. */
et_w4a32_status et_w4a32_asym_quantize_v1(uint32_t rows_n, uint32_t cols_k,
                                          uint32_t group_size,
                                          const float *weights_f32,
                                          uint8_t *out_packed,
                                          size_t out_packed_bytes);

et_w4a32_status et_w4a32_asym_dequantize_v1(uint32_t rows_n, uint32_t cols_k,
                                            uint32_t group_size,
                                            const uint8_t *packed,
                                            size_t packed_bytes,
                                            float *out_weights_f32);

/* Fused Q4_ASYM dequant-matmul:
 *   Y[m, n] = X[m, k] @ dequant(W_asym[n, k])^T + (bias ? bias[n] : 0)
 * Accumulates directly from packed 4-bit blocks in binary64 without
 * materializing an [n, k] binary32 weight buffer. */
et_w4a32_status et_w4a32_asym_gemm_v1(uint32_t m, uint32_t n, uint32_t k,
                                      uint32_t group_size, const float *x,
                                      const uint8_t *w_asym,
                                      size_t w_asym_bytes, const float *bias,
                                      float *y);

/* Dense binary32 reference matmul (binary64 inner accumulation, zero heap
 * allocations, two-pass validate-before-write):
 *   Y[m, n] = X[m, k] @ W_f32[n, k]^T + (bias ? bias[n] : 0) */
et_w4a32_status et_w4a32_f32_gemm_v1(uint32_t m, uint32_t n, uint32_t k,
                                     const float *x, const float *w_f32,
                                     const float *bias, float *y);

/* ---- Checksummed Quantized Weight Container (ETQ41000 v1.0) ----
 * Buffer-level serialize and parse (zero heap allocations, output untouched
 * on error). Validates all packed f16 scales/mins as finite. */
et_w4a32_status et_w4a32_container_serialize(
    et_w4a32_format format, uint32_t group_size, uint32_t rows_n,
    uint32_t cols_k, const uint8_t *packed, size_t packed_bytes,
    uint8_t *out_bytes, size_t capacity, size_t *out_size);

et_w4a32_status et_w4a32_container_parse(
    const uint8_t *bytes, size_t size, et_w4a32_format expected_format,
    uint32_t expected_group_size, uint32_t expected_rows_n,
    uint32_t expected_cols_k, et_w4a32_container_meta *out_meta,
    uint8_t *out_packed, size_t out_packed_capacity);

/* Atomic file save (mkstemp + write + fsync + close + rename, mode 0600,
 * non-executable) and O_NOFOLLOW + S_ISREG verified file load. */
et_w4a32_status et_w4a32_container_save(const char *path,
                                        et_w4a32_format format,
                                        uint32_t group_size, uint32_t rows_n,
                                        uint32_t cols_k, const uint8_t *packed,
                                        size_t packed_bytes);

et_w4a32_status et_w4a32_container_load(
    const char *path, et_w4a32_format expected_format,
    uint32_t expected_group_size, uint32_t expected_rows_n,
    uint32_t expected_cols_k, et_w4a32_container_meta *out_meta,
    uint8_t *out_packed, size_t out_packed_capacity);

#ifdef __cplusplus
}
#endif

#endif
