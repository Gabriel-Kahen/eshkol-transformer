#include "eshkol_transformer/mod5_moe_abi.h"
#include "mod5_moe_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

static_assert(ET_MOD5_MOE_ABI_MAJOR == 1u);
static_assert(ET_MOD5_MOE_ABI_MINOR == 0u);
static_assert(ET_MOD5_MOE_MIN_BATCH == 1u && ET_MOD5_MOE_MAX_BATCH == 2u);
static_assert(ET_MOD5_MOE_MIN_TOKENS == 1u && ET_MOD5_MOE_MAX_TOKENS == 32u);
static_assert(ET_MOD5_MOE_MIN_MODEL_DIM == 2u &&
              ET_MOD5_MOE_MAX_MODEL_DIM == 32u);
static_assert(ET_MOD5_MOE_MIN_FFN_DIM == 2u && ET_MOD5_MOE_MAX_FFN_DIM == 64u);
static_assert(ET_MOD5_MOE_MIN_EXPERTS == 2u && ET_MOD5_MOE_MAX_EXPERTS == 8u);
static_assert(ET_MOD5_MOE_MIN_TOP_K == 1u && ET_MOD5_MOE_MAX_TOP_K == 4u);
static_assert(std::is_same_v<decltype(&et_mod5_moe_kernel_provider_v1),
                             const et_kernel_provider_v1 *(*)()>);

int main() {
  const auto *provider = et_mod5_moe_kernel_provider_v1();
  if (provider == nullptr || provider->abi_major != ET_KERNEL_ABI_MAJOR)
    return 1;

  et_kernel_runtime *runtime = nullptr;
  if (et_mod5_moe_aot_open(&runtime) != 0 || runtime == nullptr)
    return 2;

  /* N=1, T=2, D=2, F=2, E=2, K=1 smoke check through direct C11 & bridge. */
  float x[4] = {0.5f, -0.25f, -0.5f, 0.75f};
  float w_gate[4] = {1.0f, 0.0f, -1.0f, 0.0f};
  float w_gate_exp[8] = {0.5f, 0.25f, -0.25f, 0.5f,
                         0.25f, -0.5f, 0.5f, 0.25f};
  float w_up_exp[8] = {0.25f, 0.5f, 0.5f, -0.25f,
                       -0.25f, 0.5f, 0.25f, 0.5f};
  float w_down_exp[8] = {0.5f, -0.25f, 0.25f, 0.5f,
                         0.25f, 0.5f, -0.25f, 0.5f};
  float y_direct[4] = {0.0f};
  float y_bridge[4] = {0.0f};
  float aux_direct = 0.0f;
  float aux_bridge = 0.0f;
  int64_t counts_direct[2] = {0, 0};
  int64_t counts_bridge[2] = {0, 0};
  int64_t evals_direct = 0;
  int64_t evals_bridge = 0;
  et_kernel_error error{};

  if (et_mod5_moe_forward_f32(1u, 2u, 2u, 2u, 2u, 1u, x, w_gate, w_gate_exp,
                              w_up_exp, w_down_exp, 0.125, y_direct,
                              &aux_direct, counts_direct, &evals_direct,
                              &error) != 0) {
    et_mod5_moe_aot_close(runtime);
    return 3;
  }
  if (et_mod5_moe_aot_forward(runtime, 1, 2, 2, 2, 2, 1, x, w_gate, w_gate_exp,
                              w_up_exp, w_down_exp, 0.125, y_bridge,
                              &aux_bridge, counts_bridge,
                              &evals_bridge) != 0) {
    et_mod5_moe_aot_close(runtime);
    return 4;
  }
  if (evals_direct != 2 || evals_bridge != 2 ||
      counts_direct[0] != 1 || counts_direct[1] != 1 ||
      counts_bridge[0] != 1 || counts_bridge[1] != 1 ||
      std::fabs(aux_direct - aux_bridge) > 0.0f) {
    et_mod5_moe_aot_close(runtime);
    return 5;
  }
  for (int i = 0; i < 4; ++i) {
    if (std::fabs(y_direct[i] - y_bridge[i]) > 0.0f) {
      et_mod5_moe_aot_close(runtime);
      return 6;
    }
  }

  float dy[4] = {0.25f, -0.5f, 0.5f, 0.125f};
  float dx[4] = {0.0f};
  float dw_gate[4] = {0.0f};
  float dw_ge[8] = {0.0f};
  float dw_ue[8] = {0.0f};
  float dw_de[8] = {0.0f};
  if (et_mod5_moe_aot_backward(runtime, 1, 2, 2, 2, 2, 1, x, w_gate, w_gate_exp,
                               w_up_exp, w_down_exp, 0.125, dy, dx, dw_gate,
                               dw_ge, dw_ue, dw_de) != 0) {
    et_mod5_moe_aot_close(runtime);
    return 7;
  }

  et_mod5_moe_aot_close(runtime);
  return 0;
}
