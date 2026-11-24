#include "eshkol_transformer/grpo_abi.h"
#include "grpo_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_grpo_loss_config>);
static_assert(std::is_standard_layout_v<et_grpo_loss_metrics>);
static_assert(std::is_standard_layout_v<et_grpo_verify_format_config>);
static_assert(std::is_standard_layout_v<et_grpo_verify_detail>);
static_assert(std::is_same_v<et_grpo_status, int32_t>);

int main() {
  const et_grpo_loss_config cfg = {2u, 3u, 4u, ET_GRPO_NORM_GLOBAL_TOKEN_MEAN,
                                   0.2f, 0.25f, 0.05f, 0u};
  if (et_grpo_validate_loss_config(&cfg) != ET_GRPO_OK)
    return 1;

  const float rewards[2] = {1.0f, 0.0f};
  float adv[2] = {0.0f, 0.0f};
  if (et_grpo_group_advantages_v1(1u, 2u, rewards, ET_GRPO_ADV_STANDARD, 1e-6f,
                                  adv) != ET_GRPO_OK)
    return 2;
  if (!(adv[0] > 0.99f && adv[1] < -0.99f))
    return 3;

  const float logits[24] = {
      0.1f, -0.2f, 0.3f, 0.0f,  0.5f, -0.1f, 0.2f, -0.3f,
      -0.2f, 0.4f, 0.1f, 0.2f,  0.0f, 0.1f, -0.2f, 0.3f,
      0.2f, 0.3f, -0.4f, 0.1f,  -0.3f, 0.2f, 0.4f, -0.1f};
  const int32_t targets[6] = {0, 1, 2, 3, 1, 0};
  const uint8_t mask[6] = {0u, 1u, 1u, 0u, 1u, 1u};
  float tok_lp[6] = {0.0f};
  float seq_lp[2] = {0.0f};
  if (et_grpo_token_logprobs_v1(2u, 3u, 4u, logits, targets, mask, tok_lp,
                                seq_lp) != ET_GRPO_OK)
    return 4;

  et_grpo_loss_metrics metrics;
  if (et_grpo_loss_forward_v1(&cfg, logits, targets, tok_lp, tok_lp, mask, adv,
                              rewards, &metrics) != ET_GRPO_OK)
    return 5;
  if (metrics.active_tokens != 4u || std::fabs(metrics.mean_kl) > 1e-6f)
    return 6;

  float grad_logits[24] = {0.0f};
  if (et_grpo_loss_backward_v1(&cfg, logits, targets, tok_lp, tok_lp, mask, adv,
                               rewards, grad_logits, nullptr,
                               &metrics) != ET_GRPO_OK)
    return 7;

  float verify_r = -1.0f;
  const char *comp = "<think>step</think><answer>7</answer>";
  const et_grpo_verify_format_config vcfg = {0.25f, 0.75f,
                                             ET_GRPO_VERIFY_TRIM_WS, 0u};
  if (et_grpo_verify_format_and_answer_v1(comp, std::strlen(comp), "7", 1u,
                                          &vcfg, &verify_r,
                                          nullptr) != ET_GRPO_OK)
    return 8;
  if (std::fabs(verify_r - 1.0f) > 1e-6f)
    return 9;

  et_grpo_aot_ctx *ctx = nullptr;
  if (et_grpo_aot_create(1, 2, 3, 4, 4, &ctx) != ET_GRPO_OK || ctx == nullptr)
    return 10;
  et_grpo_aot_destroy(ctx);
  return 0;
}
