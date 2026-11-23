#include "eshkol_transformer/mod5_rope_abi.h"
#include "mod5_rope_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

static_assert(ET_MOD5_ROPE_ABI_MAJOR == 1u);
static_assert(ET_MOD5_ROPE_ABI_MINOR == 0u);
static_assert(ET_MOD5_ROPE_MIN_BATCH == 1u && ET_MOD5_ROPE_MAX_BATCH == 2u);
static_assert(ET_MOD5_ROPE_MIN_HEADS == 1u && ET_MOD5_ROPE_MAX_HEADS == 4u);
static_assert(ET_MOD5_ROPE_MIN_TOKENS == 1u && ET_MOD5_ROPE_MAX_TOKENS == 64u);
static_assert(ET_MOD5_ROPE_MIN_HEAD_DIM == 2u &&
              ET_MOD5_ROPE_MAX_HEAD_DIM == 16u);
static_assert(std::is_same_v<decltype(&et_mod5_rope_kernel_provider_v1),
                             const et_kernel_provider_v1 *(*)()>);

int main() {
  const auto *provider = et_mod5_rope_kernel_provider_v1();
  if (provider == nullptr || provider->abi_major != ET_KERNEL_ABI_MAJOR)
    return 1;

  et_kernel_runtime *runtime = nullptr;
  if (et_mod5_rope_aot_open(&runtime) != 0 || runtime == nullptr)
    return 2;

  float x[8] = {1.0f, -0.5f, 0.25f, 0.75f, -1.25f, 0.5f, 0.125f, -0.875f};
  float y[8] = {0.0f};
  float dx[8] = {0.0f};
  int64_t pos[2] = {0, 3};
  et_kernel_error error{};

  if (et_mod5_rope_forward_f32(1u, 1u, 2u, 4u, x, pos,
                               ET_MOD5_ROPE_DEFAULT_THETA,
                               ET_MOD5_ROPE_DEFAULT_SCALE, y, &error) != 0) {
    et_mod5_rope_aot_close(runtime);
    return 3;
  }
  if (et_mod5_rope_aot_backward(runtime, 1, 1, 2, 4, y, pos, 10000.0, 1.0,
                                dx) != 0) {
    et_mod5_rope_aot_close(runtime);
    return 4;
  }
  for (int i = 0; i < 8; ++i) {
    if (std::fabs(dx[i] - x[i]) > 1e-6f) {
      et_mod5_rope_aot_close(runtime);
      return 5;
    }
  }
  et_mod5_rope_aot_close(runtime);
  return 0;
}
