#include "eshkol_transformer/multi_lora_sgmv_abi.h"
#include "multi_lora_sgmv_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_mlora_registry_config>);
static_assert(std::is_trivially_copyable_v<et_mlora_registry_config>);
static_assert(std::is_standard_layout_v<et_mlora_adapter_info>);
static_assert(std::is_trivially_copyable_v<et_mlora_adapter_info>);
static_assert(std::is_standard_layout_v<et_mlora_adamw_config>);
static_assert(std::is_trivially_copyable_v<et_mlora_adamw_config>);
static_assert(std::is_standard_layout_v<et_mlora_sgmv_stats>);
static_assert(std::is_trivially_copyable_v<et_mlora_sgmv_stats>);
static_assert(std::is_standard_layout_v<et_mlora_bundle_meta>);
static_assert(std::is_trivially_copyable_v<et_mlora_bundle_meta>);
static_assert(std::is_same_v<et_mlora_status, int32_t>);

int main() {
  const et_mlora_registry_config cfg = {4u, 3u, 4u, 2u, 1u, 0u};
  if (et_mlora_validate_registry_config_v1(&cfg) != ET_MLORA_OK)
    return 1;

  const float w_base[12] = {0.5f,  0.25f, -0.25f, 0.125f, -0.5f, 0.125f,
                            0.25f, -0.25f, 0.25f,  -0.125f, 0.5f, 0.25f};
  const float bias[3] = {0.125f, -0.25f, 0.0625f};

  et_mlora_registry *reg = nullptr;
  if (et_mlora_registry_create_v1(&cfg, w_base, bias, &reg) != ET_MLORA_OK ||
      reg == nullptr)
    return 2;

  et_mlora_adapter_info info1 = {};
  et_mlora_adapter_info info2 = {};
  if (et_mlora_register_seeded_adapter_v1(reg, 10, 1u, 2.0f, 11u, &info1) !=
      ET_MLORA_OK)
    return 3;
  if (et_mlora_register_seeded_adapter_v1(reg, 20, 2u, 4.0f, 22u, &info2) !=
      ET_MLORA_OK)
    return 4;

  const float x[12] = {0.25f, -0.5f,  0.75f,  0.125f, -0.25f, 0.5f,
                       -0.125f, 0.25f, 0.5f,  -0.25f, 0.125f, -0.5f};
  const int32_t row_ids[3] = {20, ET_MLORA_BASE_ONLY_ID, 10};
  float y_sgmv[9] = {0.0f};
  float y_ref[9] = {0.0f};
  et_mlora_sgmv_stats stats = {};

  if (et_mlora_sgmv_forward_v1(reg, 3u, x, row_ids, y_sgmv, &stats) !=
      ET_MLORA_OK)
    return 5;
  if (stats.base_only_rows != 1u || stats.adapted_rows != 2u ||
      stats.active_segments != 2u)
    return 6;

  for (uint32_t i = 0u; i < 3u; ++i) {
    if (et_mlora_single_adapter_forward_v1(reg, row_ids[i], 1u, &x[i * 4u],
                                           &y_ref[i * 3u]) != ET_MLORA_OK)
      return 7;
  }
  if (std::memcmp(y_sgmv, y_ref, sizeof(y_sgmv)) != 0)
    return 8;

  const float dy[9] = {0.1f, -0.2f, 0.3f,  -0.4f, 0.5f,
                       -0.6f, 0.2f, 0.1f, -0.3f};
  float dx[12] = {0.0f};
  if (et_mlora_sgmv_backward_v1(reg, 3u, x, row_ids, dy, dx, nullptr) !=
      ET_MLORA_OK)
    return 9;

  const et_mlora_adamw_config acfg = {0.05f, 0.9f, 0.999f, 1e-8f, 0.01f};
  if (et_mlora_adamw_step_adapter_v1(reg, 10, &acfg) != ET_MLORA_OK)
    return 10;

  uint32_t pin_count = 0u;
  if (et_mlora_pin_adapter_v1(reg, 10, &pin_count) != ET_MLORA_OK ||
      pin_count != 1u)
    return 11;
  if (et_mlora_unregister_adapter_v1(reg, 10) != ET_MLORA_ERR_BUSY)
    return 12;
  if (et_mlora_unpin_adapter_v1(reg, 10, &pin_count) != ET_MLORA_OK ||
      pin_count != 0u)
    return 13;
  if (et_mlora_unregister_adapter_v1(reg, 10) != ET_MLORA_OK)
    return 14;

  et_mlora_registry_destroy_v1(reg);

  et_mlora_aot_ctx *ctx = nullptr;
  if (et_mlora_aot_create(3, 4, 3, 4, 2, 1, &ctx) != ET_MLORA_OK ||
      ctx == nullptr)
    return 15;
  et_mlora_aot_destroy(ctx);
  return 0;
}
