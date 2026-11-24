#include "eshkol_transformer/o2_muon_abi.h"
#include "o2_muon_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_wsd_config>);
static_assert(std::is_standard_layout_v<et_muon_param_group>);
static_assert(std::is_standard_layout_v<et_muon_step_config>);
static_assert(std::is_same_v<et_muon_status, int32_t>);

int main() {
  const float g[6] = {0.5f, -0.25f, 0.75f, -0.5f, 0.125f, 0.625f};
  float u[6] = {0.0f};
  if (et_muon_newton_schulz5_f32(2u, 3u, g, ET_MUON_DEFAULT_NS_STEPS, 1e-7,
                                 u) != ET_MUON_OK)
    return 1;

  double ortho_err = -1.0;
  if (et_muon_orthogonality_error_f32(2u, 3u, u, &ortho_err) != ET_MUON_OK ||
      !std::isfinite(ortho_err) || ortho_err < 0.0 || ortho_err > 1.0)
    return 2;

  const et_wsd_config wsd = {4u, 5u, 14u, 0.04, 0.004, ET_WSD_DECAY_COSINE, 0u};
  double lr = 0.0;
  uint32_t phase = 99u;
  if (et_wsd_lr(&wsd, 4u, &lr, &phase) != ET_MUON_OK || lr != 0.04 ||
      phase != ET_WSD_PHASE_STABLE)
    return 3;

  float w2d[6] = {0.1f, -0.2f, 0.3f, -0.1f, 0.2f, -0.3f};
  float b2d[6] = {0.0f};
  float w1d[3] = {1.0f, 1.0f, 1.0f};
  const float g1d[3] = {0.0f, 0.0f, 0.0f};
  float m1d[3] = {0.0f};
  float v1d[3] = {0.0f};

  const et_muon_param_group groups[2] = {
      {ET_MUON_GROUP_MUON_2D, 2u, 3u, 1u, w2d, g, b2d, nullptr},
      {ET_MUON_GROUP_ADAMW_1D, 3u, 1u, 0u, w1d, g1d, m1d, v1d},
  };
  const et_muon_step_config step_cfg = {
      0.04, 1u, 0.9, 1e-7, 0.01, 5u, 1u, 0.9, 0.999, 1e-8, 0.01, 1.0};
  if (et_muon_hybrid_step_wsd(&wsd, 0u, &step_cfg, 2u, groups, &lr, &phase) !=
      ET_MUON_OK)
    return 4;
  if (w1d[0] != 1.0f || w1d[1] != 1.0f || w1d[2] != 1.0f)
    return 5;

  et_muon_aot_ctx *ctx = nullptr;
  if (et_muon_aot_create(0.9, 1e-7, 0.01, 5, 1, 0.9, 0.999, 1e-8, 0.01, 1.0, 4,
                         5, 14, 0.04, 0.004, ET_WSD_DECAY_COSINE, &ctx) !=
          ET_MUON_OK ||
      ctx == nullptr)
    return 6;
  et_muon_aot_destroy(ctx);
  return 0;
}
