#include "eshkol_transformer/lora_adapter_abi.h"
#include "lora_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_lora_config>);
static_assert(std::is_standard_layout_v<et_lora_checkpoint_meta>);
static_assert(std::is_same_v<et_lora_status, int32_t>);

int main() {
  const et_lora_config cfg = {2u, 4u, 3u, 2u, 4.0f};
  if (et_lora_validate_config(&cfg) != ET_LORA_OK)
    return 1;

  const float x[8] = {0.25f, -0.5f, 0.75f, 0.125f,
                      -0.25f, 0.5f, -0.125f, 0.25f};
  const float w[12] = {0.5f, 0.25f, -0.25f, 0.125f,
                       -0.5f, 0.125f, 0.25f, -0.25f,
                       0.25f, -0.125f, 0.5f, 0.25f};
  float a[8] = {0.0f};
  float b[6] = {0.0f};
  float y_base[6] = {0.0f};
  float y_lora[6] = {0.0f};
  float w_merged[12] = {0.0f};

  if (et_lora_init_adapter(&cfg, 99u, a, b) != ET_LORA_OK)
    return 2;
  if (et_lora_base_forward(&cfg, x, w, y_base) != ET_LORA_OK)
    return 3;
  if (et_lora_forward(&cfg, x, w, a, b, y_lora) != ET_LORA_OK)
    return 4;
  if (std::memcmp(y_base, y_lora, sizeof(y_base)) != 0)
    return 5;
  if (et_lora_merge(&cfg, w, a, b, w_merged) != ET_LORA_OK)
    return 6;
  if (std::memcmp(w, w_merged, sizeof(w)) != 0)
    return 7;

  et_lora_aot_ctx *ctx = nullptr;
  if (et_lora_aot_create(2, 4, 3, 2, 4.0, &ctx) != ET_LORA_OK ||
      ctx == nullptr)
    return 8;
  et_lora_aot_destroy(ctx);
  return 0;
}
