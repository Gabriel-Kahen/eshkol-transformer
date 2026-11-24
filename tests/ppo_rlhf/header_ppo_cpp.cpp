#include "eshkol_transformer/ppo_rlhf_abi.h"
#include "ppo_rlhf_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_ppo_loss_config>);
static_assert(std::is_standard_layout_v<et_ppo_loss_metrics>);
static_assert(std::is_standard_layout_v<et_ppo_adamw_config>);
static_assert(std::is_standard_layout_v<et_ppo_trainer_config>);
static_assert(std::is_standard_layout_v<et_ppo_rollout_report>);
static_assert(std::is_standard_layout_v<et_ppo_checkpoint_meta>);
static_assert(std::is_same_v<et_ppo_status, int32_t>);

int main() {
  const et_ppo_loss_config cfg = {2u, 3u, 4u, 1u, 0.2f, 0.2f, 0.2f, 0.5f, 0.01f,
                                  0u};
  if (et_ppo_validate_loss_config(&cfg) != ET_PPO_OK)
    return 1;

  const float r_env[6] = {0.0f, 1.0f, 0.5f, 0.0f, -0.5f, 1.0f};
  const float old_lp[6] = {0.0f, -1.2f, -0.8f, 0.0f, -1.5f, -0.6f};
  const float ref_lp[6] = {0.0f, -1.0f, -0.9f, 0.0f, -1.3f, -0.7f};
  const uint8_t mask[6] = {0u, 1u, 1u, 0u, 1u, 1u};
  float r_shaped[6] = {0.0f};
  if (et_ppo_kl_shaped_rewards_v1(2u, 3u, 0.1f, r_env, old_lp, ref_lp, mask,
                                  r_shaped) != ET_PPO_OK)
    return 2;

  const float values[6] = {0.0f, 0.3f, 0.4f, 0.0f, -0.2f, 0.6f};
  float adv[6] = {0.0f};
  float ret[6] = {0.0f};
  if (et_ppo_gae_v1(2u, 3u, 0.99f, 0.95f, r_shaped, values, mask, adv, ret) !=
      ET_PPO_OK)
    return 3;

  float whitened[6] = {0.0f};
  if (et_ppo_whiten_advantages_v1(2u, 3u, 1e-6f, adv, mask, whitened) !=
      ET_PPO_OK)
    return 4;

  const float logits[24] = {
      0.1f, -0.2f, 0.3f, 0.0f,  0.5f, -0.1f, 0.2f, -0.3f,
      -0.2f, 0.4f, 0.1f, 0.2f,  0.0f, 0.1f, -0.2f, 0.3f,
      0.2f, 0.3f, -0.4f, 0.1f,  -0.3f, 0.2f, 0.4f, -0.1f};
  const int32_t actions[6] = {0, 1, 2, 3, 1, 0};
  float tok_lp[6] = {0.0f};
  if (et_ppo_token_logprobs_v1(2u, 3u, 4u, logits, actions, mask, tok_lp) !=
      ET_PPO_OK)
    return 5;

  et_ppo_loss_metrics metrics;
  std::memset(&metrics, 0, sizeof(metrics));
  if (et_ppo_loss_forward_v1(&cfg, logits, values, actions, tok_lp, values,
                             whitened, ret, mask, &metrics) != ET_PPO_OK)
    return 6;
  if (metrics.active_tokens != 4u || std::fabs(metrics.approx_kl) > 1e-6f)
    return 7;

  float d_logits[24] = {0.0f};
  float d_values[6] = {0.0f};
  if (et_ppo_loss_backward_v1(&cfg, logits, values, actions, tok_lp, values,
                              whitened, ret, mask, d_logits, d_values,
                              &metrics) != ET_PPO_OK)
    return 8;

  et_ppo_aot_ctx *ctx = nullptr;
  if (et_ppo_aot_create(4, 4, 8, 6, &ctx) != ET_PPO_OK || ctx == nullptr)
    return 9;
  et_ppo_aot_destroy(ctx);
  return 0;
}
