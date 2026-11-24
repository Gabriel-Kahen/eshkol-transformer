#include "eshkol_transformer/io5_quant_abi.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

static_assert(ET_IO5_QUANT_ABI_MAJOR == 1u, "expected ABI major 1");
static_assert(ET_IO5_QUANT_ABI_MINOR == 0u, "expected ABI minor 0");
static_assert(ET_IO5_QK8_0 == 32u, "expected QK8_0 = 32");
static_assert(ET_IO5_Q8_0_BLOCK_BYTES == 34u, "expected Q8_0 block = 34 bytes");
static_assert(std::is_same<et_io5_quant_status, std::uint32_t>::value,
              "status typedef must be uint32_t");

int main() {
  if (et_io5_quant_abi_version() != ((1u << 16u) | 0u)) {
    std::fprintf(stderr, "ABI version mismatch\n");
    return 1;
  }

  std::size_t q8_bytes = 0u;
  if (et_io5_q8_0_byte_size(64u, &q8_bytes) != ET_IO5_QUANT_OK ||
      q8_bytes != 68u) {
    std::fprintf(stderr, "q8_0_byte_size mismatch\n");
    return 1;
  }

  std::uint16_t f16 = 0u;
  std::uint32_t f32 = 0u;
  if (et_io5_f32_bits_to_f16_bits(0x3f800000u, &f16) != ET_IO5_QUANT_OK ||
      f16 != 0x3c00u) {
    std::fprintf(stderr, "f32->f16 1.0f mismatch\n");
    return 1;
  }
  if (et_io5_f16_bits_to_f32_bits(f16, &f32) != ET_IO5_QUANT_OK ||
      f32 != 0x3f800000u) {
    std::fprintf(stderr, "f16->f32 1.0f mismatch\n");
    return 1;
  }

  std::uint16_t bf16 = 0u;
  if (et_io5_f32_bits_to_bf16_bits(0x3f800000u, &bf16) != ET_IO5_QUANT_OK ||
      bf16 != 0x3f80u) {
    std::fprintf(stderr, "f32->bf16 1.0f mismatch\n");
    return 1;
  }
  if (et_io5_bf16_bits_to_f32_bits(bf16, &f32) != ET_IO5_QUANT_OK ||
      f32 != 0x3f800000u) {
    std::fprintf(stderr, "bf16->f32 1.0f mismatch\n");
    return 1;
  }

  float src[32];
  for (std::size_t i = 0u; i < 32u; ++i) {
    src[i] = static_cast<float>(static_cast<int>(i) - 16) * 0.25f;
  }
  std::uint8_t q8[34];
  float dst[32];
  if (et_io5_quantize_q8_0(src, sizeof(src), q8, sizeof(q8), 32u) !=
      ET_IO5_QUANT_OK) {
    std::fprintf(stderr, "C++17 Q8_0 quantize failed\n");
    return 1;
  }
  if (et_io5_dequantize_q8_0(q8, sizeof(q8), dst, sizeof(dst), 32u) !=
      ET_IO5_QUANT_OK) {
    std::fprintf(stderr, "C++17 Q8_0 dequantize failed\n");
    return 1;
  }
  if (et_io5_validate_f32_finite(dst, sizeof(dst), 32u) != ET_IO5_QUANT_OK) {
    std::fprintf(stderr, "C++17 validate_f32_finite failed\n");
    return 1;
  }
  return 0;
}
