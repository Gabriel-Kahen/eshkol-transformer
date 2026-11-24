#include "eshkol_transformer/fp8_gemm_abi.h"
#include "fp8_gemm_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_fp8_delayed_scale_state_v1>);
static_assert(std::is_trivially_copyable_v<et_fp8_delayed_scale_state_v1>);
static_assert(std::is_same_v<et_fp8_status, int32_t>);
static_assert(std::is_same_v<et_fp8_format, uint32_t>);
static_assert(std::is_same_v<et_fp8_overflow_mode, uint32_t>);

int main() {
  uint8_t q = 0u;
  float f = 0.0f;
  if (et_fp8_e4m3_encode_scalar_v1(448.0f, ET_FP8_OVERFLOW_REJECT, &q) !=
          ET_FP8_OK ||
      q != 0x7Eu)
    return 1;
  if (et_fp8_e4m3_decode_scalar_v1(0x7Eu, &f) != ET_FP8_OK || f != 448.0f)
    return 2;
  if (et_fp8_e4m3_decode_scalar_v1(0x7Fu, &f) != ET_FP8_ERR_NONFINITE)
    return 3;

  if (et_fp8_e5m2_encode_scalar_v1(57344.0f, ET_FP8_OVERFLOW_REJECT, &q) !=
          ET_FP8_OK ||
      q != 0x7Bu)
    return 4;
  if (et_fp8_e5m2_decode_scalar_v1(0x7Bu, &f) != ET_FP8_OK || f != 57344.0f)
    return 5;
  if (et_fp8_e5m2_decode_scalar_v1(0x7Cu, &f) != ET_FP8_ERR_NONFINITE)
    return 6;

  et_fp8_delayed_scale_state_v1 state;
  if (et_fp8_delayed_scale_init(&state, ET_FP8_FORMAT_E4M3FN, 4u, 1u, 1e-6f,
                                1e-4f, 65536.0f) != ET_FP8_OK)
    return 7;
  if (et_fp8_delayed_scale_update(&state, 16.0f) != ET_FP8_OK)
    return 8;
  float scale = 0.0f;
  float hist_amax = 0.0f;
  if (et_fp8_delayed_scale_compute(&state, &scale, &hist_amax) != ET_FP8_OK)
    return 9;
  if (std::fabs(scale - 14.0f) > 1e-6f || hist_amax != 16.0f)
    return 10;

  float x[16];
  float w[16];
  for (int i = 0; i < 16; ++i) {
    x[i] = static_cast<float>(i - 8) * 0.125f;
    w[i] = static_cast<float>(7 - i) * 0.25f;
  }
  uint8_t xq[16];
  uint8_t wq[16];
  float sx[1];
  float sw[1];
  float y[1] = {0.0f};
  if (et_fp8_block_quantize_e4m3_v1(x, 1u, 16u, 1u, 16u, xq, sx) != ET_FP8_OK)
    return 11;
  if (et_fp8_block_quantize_e4m3_v1(w, 1u, 16u, 1u, 16u, wq, sw) != ET_FP8_OK)
    return 12;
  if (et_fp8_block_gemm_e4m3_v1(1u, 1u, 16u, 16u, 1u, xq, sx, wq, sw, nullptr,
                                y) != ET_FP8_OK)
    return 13;

  const float dy[1] = {1.0f};
  float dx[16];
  float dw[16];
  float db[1];
  if (et_fp8_block_gemm_e4m3_backward_v1(1u, 1u, 16u, 16u, 1u, dy, xq, sx, wq,
                                         sw, dx, dw, db) != ET_FP8_OK)
    return 14;
  if (db[0] != 1.0f)
    return 15;

  /* Bridge C++17 linkage check. */
  if (et_fp8_aot_delayed_scale_state_bytes() !=
      static_cast<int64_t>(sizeof(et_fp8_delayed_scale_state_v1)))
    return 16;
  uint8_t bridge_q = 0u;
  if (et_fp8_aot_e4m3_encode_scalar(448.0, ET_FP8_OVERFLOW_REJECT, &bridge_q) !=
          ET_FP8_OK ||
      bridge_q != 0x7Eu)
    return 17;
  return 0;
}
