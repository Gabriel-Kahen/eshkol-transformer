#include "eshkol_transformer/w4a32_quant_abi.h"
#include "w4a32_quant_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_w4a32_container_meta>);
static_assert(std::is_same_v<et_w4a32_status, int32_t>);
static_assert(std::is_same_v<et_w4a32_format, uint32_t>);

int main() {
  size_t q4_bytes = 0u;
  if (et_w4a32_packed_bytes(ET_W4A32_FORMAT_Q4_0, 32u, 2u, 32u, &q4_bytes) !=
          ET_W4A32_OK ||
      q4_bytes != 36u)
    return 1;

  size_t asym_bytes = 0u;
  if (et_w4a32_packed_bytes(ET_W4A32_FORMAT_Q4_ASYM, 64u, 2u, 64u,
                            &asym_bytes) != ET_W4A32_OK ||
      asym_bytes != 72u)
    return 2;

  float w[64];
  float x[32];
  float bias[2] = {0.25f, -0.5f};
  for (uint32_t i = 0; i < 64u; ++i)
    w[i] = static_cast<float>(static_cast<int32_t>(i % 15u) - 7) * 0.125f;
  for (uint32_t i = 0; i < 32u; ++i)
    x[i] = static_cast<float>(static_cast<int32_t>(i % 7u) - 3) * 0.25f;

  uint8_t packed[36] = {0u};
  float deq[64] = {0.0f};
  float y_fused[2] = {0.0f};
  float y_ref[2] = {0.0f};

  if (et_w4a32_q4_0_quantize_v1(2u, 32u, w, packed, sizeof(packed)) !=
      ET_W4A32_OK)
    return 3;
  if (et_w4a32_q4_0_dequantize_v1(2u, 32u, packed, sizeof(packed), deq) !=
      ET_W4A32_OK)
    return 4;
  if (et_w4a32_q4_0_gemm_v1(1u, 2u, 32u, x, packed, sizeof(packed), bias,
                            y_fused) != ET_W4A32_OK)
    return 5;
  if (et_w4a32_f32_gemm_v1(1u, 2u, 32u, x, deq, bias, y_ref) != ET_W4A32_OK)
    return 6;
  for (uint32_t i = 0; i < 2u; ++i) {
    if (std::fabs(y_fused[i] - y_ref[i]) > 1e-6f)
      return 7;
  }

  et_w4a32_aot_ctx *ctx = nullptr;
  if (et_w4a32_aot_create(2, 32, 64, 32, ET_W4A32_FORMAT_Q4_0, 32, &ctx) !=
          ET_W4A32_OK ||
      ctx == nullptr)
    return 8;
  et_w4a32_aot_destroy(ctx);
  return 0;
}
