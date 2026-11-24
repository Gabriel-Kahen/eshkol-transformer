#include "eshkol_transformer/dpo_alignment_abi.h"
#include "dpo_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_dpo_config>);
static_assert(std::is_standard_layout_v<et_dpo_telemetry>);
static_assert(std::is_same_v<et_dpo_status, int32_t>);

int main() {
  const et_dpo_config cfg = {1u, 2u, 3u, 0u, 0.5, 0.1};
  if (et_dpo_validate_config(&cfg) != ET_DPO_OK)
    return 1;

  const float pi_w[6] = {0.5f, -0.25f, 0.75f, -0.5f, 1.0f, 0.25f};
  const float pi_l[6] = {0.25f, 0.5f, -0.75f, 0.75f, -0.25f, 0.5f};
  const float ref_w[6] = {0.25f, 0.0f, 0.5f, -0.25f, 0.5f, 0.25f};
  const float ref_l[6] = {0.25f, 0.5f, -0.5f, 0.5f, 0.0f, 0.25f};
  const int64_t tgt_w[2] = {0, 1};
  const int64_t tgt_l[2] = {1, 2};
  const uint8_t mask_w[2] = {0u, 1u};
  const uint8_t mask_l[2] = {0u, 1u};

  double logps[1] = {0.0};
  if (et_dpo_sequence_logps(1u, 2u, 3u, pi_w, tgt_w, mask_w, logps) !=
      ET_DPO_OK)
    return 2;
  if (!std::isfinite(logps[0]) || !(logps[0] < 0.0))
    return 3;

  et_dpo_telemetry fwd_telem = {0.0, 0.0, 0.0, 0.0, 0.0};
  if (et_dpo_forward(&cfg, pi_w, pi_l, ref_w, ref_l, tgt_w, tgt_l, mask_w,
                     mask_l, &fwd_telem) != ET_DPO_OK)
    return 4;

  et_dpo_telemetry fb_telem = {0.0, 0.0, 0.0, 0.0, 0.0};
  float dw[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float dl[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  if (et_dpo_forward_backward(&cfg, pi_w, pi_l, ref_w, ref_l, tgt_w, tgt_l,
                              mask_w, mask_l, &fb_telem, dw, dl) != ET_DPO_OK)
    return 5;
  if (std::memcmp(&fwd_telem, &fb_telem, sizeof(fwd_telem)) != 0)
    return 6;
  for (int v = 0; v < 3; ++v) {
    if (dw[v] != 0.0f || dl[v] != 0.0f)
      return 7;
  }

  uint8_t digest[ET_DPO_SHA256_BYTES] = {0u};
  if (et_dpo_weight_sha256(6u, ref_w, digest) != ET_DPO_OK)
    return 8;

  et_dpo_aot_ctx *ctx = nullptr;
  if (et_dpo_aot_create(1, 2, 3, 2, 0.5, 0.1, 0, &ctx) != ET_DPO_OK ||
      ctx == nullptr)
    return 9;
  et_dpo_aot_destroy(ctx);
  return 0;
}
