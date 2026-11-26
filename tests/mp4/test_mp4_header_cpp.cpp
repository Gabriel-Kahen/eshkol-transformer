#include "eshkol_transformer/mp4_mixed_precision_abi.h"

#include <type_traits>

static_assert(ET_MP4_MIXED_PRECISION_ABI_MAJOR == 1u);
static_assert(ET_MP4_MIXED_PRECISION_ABI_MINOR == 0u);
static_assert(std::is_standard_layout_v<et_mp4_loss_scaler_v1>);
static_assert(std::is_trivially_copyable_v<et_mp4_loss_scaler_v1>);
static_assert(sizeof(et_mp4_loss_scaler_v1) == 28u);
static_assert(offsetof(et_mp4_loss_scaler_v1, growth_interval) == 12u);
static_assert(offsetof(et_mp4_loss_scaler_v1, max_scale) == 24u);
static_assert(std::is_same_v<decltype(&et_mp4_kernel_provider_v1),
                             const et_kernel_provider_v1 *(*)()>);
static_assert(std::is_same_v<decltype(&et_mp4_loss_scaler_init),
                             et_mp4_loss_scaler_status (*)(
                                 et_mp4_loss_scaler_v1 *, float, float, float,
                                 uint32_t, float, float)>);
static_assert(std::is_same_v<
              decltype(&et_mp4_loss_scaler_update),
              et_mp4_loss_scaler_status (*)(et_mp4_loss_scaler_v1 *, uint8_t)>);

int main() {
  const auto *provider = et_mp4_kernel_provider_v1();
  et_mp4_loss_scaler_v1 state{};
  return provider && provider->abi_major == ET_KERNEL_ABI_MAJOR &&
                 provider->abi_minor == ET_KERNEL_ABI_MINOR &&
                 et_mp4_loss_scaler_init(&state, 8.0f, 2.0f, 0.5f, 1u, 1.0f,
                                         32.0f) ==
                     ET_MP4_LOSS_SCALER_STEP_ACCEPTED &&
                 et_mp4_loss_scaler_update(&state, 1u) ==
                     ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF &&
                 state.scale == 4.0f
             ? 0
             : 1;
}
