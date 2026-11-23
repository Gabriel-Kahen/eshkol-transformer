#include "mp4_eshkol_bridge.h"

#include <cstdint>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_mp4_aot_scaler>);
static_assert(std::is_trivially_copyable_v<et_mp4_aot_scaler>);
static_assert(sizeof(et_mp4_aot_scaler) == 40u);
static_assert(offsetof(et_mp4_aot_scaler, initialized) == 28u);
static_assert(offsetof(et_mp4_aot_scaler, overflow_count) == 32u);

static_assert(std::is_same_v<decltype(&et_mp4_aot_open),
                             int32_t (*)(et_kernel_runtime **)>);
static_assert(std::is_same_v<decltype(&et_mp4_aot_close),
                             void (*)(et_kernel_runtime *)>);
static_assert(std::is_same_v<
              decltype(&et_mp4_aot_cast_f32_to_f16),
              int32_t (*)(const et_kernel_runtime *, int64_t, const void *,
                          void *)>);
static_assert(std::is_same_v<
              decltype(&et_mp4_aot_cast_f32_to_bf16),
              int32_t (*)(const et_kernel_runtime *, int64_t, const void *,
                          void *)>);
static_assert(std::is_same_v<
              decltype(&et_mp4_aot_scaler_unscale_f16),
              int32_t (*)(const et_kernel_runtime *, const void *, int64_t,
                          const void *, void *, void *)>);
static_assert(std::is_same_v<
              decltype(&et_mp4_aot_scaler_step),
              int32_t (*)(void *, int32_t)>);

int main() {
  et_kernel_runtime *runtime = nullptr;
  if (et_mp4_aot_open(&runtime) != 0 || runtime == nullptr)
    return 1;

  et_mp4_aot_scaler scaler{};
  if (et_mp4_aot_scaler_init(&scaler, 32.0, 2.0, 0.5, 2, 1.0, 256.0) !=
      static_cast<int32_t>(ET_MP4_LOSS_SCALER_STEP_ACCEPTED)) {
    et_mp4_aot_close(runtime);
    return 2;
  }

  float in[8] = {0.5f, -1.0f, 1.5f, -2.0f, 0.25f, -0.5f, 0.75f, -1.0f};
  uint16_t f16[8] = {};
  uint16_t bf16[8] = {};
  float out[8] = {};
  uint8_t overflow = 0xffu;

  if (et_mp4_aot_cast_f32_to_f16(runtime, 8, in, f16) != 0 ||
      et_mp4_aot_cast_f16_to_f32(runtime, 8, f16, out) != 0 ||
      out[0] != 0.5f ||
      et_mp4_aot_cast_f32_to_bf16(runtime, 8, in, bf16) != 0 ||
      et_mp4_aot_scaler_unscale_bf16(runtime, &scaler, 8, bf16, out,
                                     &overflow) != 0 ||
      overflow != 0u || out[1] != -1.0f / 32.0f) {
    et_mp4_aot_close(runtime);
    return 3;
  }

  et_mp4_aot_close(runtime);
  return 0;
}
