#include "eshkol_transformer/mtp_head_abi.h"
#include "mtp_head_eshkol_bridge.h"

#include <cstdio>
#include <cstring>
#include <type_traits>

static_assert(ET_MTP_ABI_MAJOR == 1u);
static_assert(ET_MTP_ABI_MINOR == 0u);
static_assert(ET_MTP_MIN_DEPTH == 1u);
static_assert(ET_MTP_MAX_DEPTH == 4u);
static_assert(ET_MTP_OK == 0);
static_assert(std::is_same_v<et_mtp_status_t, int32_t>);
static_assert(std::is_standard_layout_v<et_mtp_config_v1>);
static_assert(std::is_standard_layout_v<et_mtp_layout_v1>);
static_assert(std::is_standard_layout_v<et_mtp_loss_stats_v1>);
static_assert(std::is_standard_layout_v<et_mtp_step_stats_v1>);
static_assert(std::is_standard_layout_v<et_mtp_rng_state_v1>);
static_assert(std::is_standard_layout_v<et_mtp_verify_result_v1>);

int main() {
  et_mtp_config_v1 cfg{};
  cfg.struct_size = sizeof(cfg);
  cfg.vocab_size = 16u;
  cfg.hidden_dim = 8u;
  cfg.ffn_dim = 16u;
  cfg.mtp_depth = 3u;
  cfg.tie_weights = 0u;
  cfg.rms_eps = 1e-6;
  cfg.lambda_mtp = 0.5;
  cfg.lr = 0.08;
  cfg.beta1 = 0.9;
  cfg.beta2 = 0.999;
  cfg.eps = 1e-8;
  cfg.weight_decay = 0.01;
  cfg.max_grad_norm = 5.0;

  et_mtp_layout_v1 lay{};
  if (et_mtp_compute_layout_v1(&cfg, &lay) != ET_MTP_OK ||
      lay.total_params == 0u) {
    std::fprintf(stderr, "C++17 layout check failed\n");
    return 1;
  }

  et_mtp_model_t *model = nullptr;
  if (et_mtp_model_create_v1(&cfg, &model) != ET_MTP_OK || model == nullptr) {
    std::fprintf(stderr, "C++17 model create failed\n");
    return 1;
  }
  if (et_mtp_model_init_weights_v1(model, 42u) != ET_MTP_OK) {
    et_mtp_model_destroy_v1(model);
    std::fprintf(stderr, "C++17 init weights failed\n");
    return 1;
  }
  uint8_t digest[32]{};
  if (et_mtp_model_state_sha256_v1(model, digest) != ET_MTP_OK) {
    et_mtp_model_destroy_v1(model);
    std::fprintf(stderr, "C++17 state sha256 failed\n");
    return 1;
  }
  et_mtp_model_destroy_v1(model);

  et_mtp_aot_session_t *sess = nullptr;
  if (et_mtp_aot_open(16, 8, 16, 3, 0, 1, 10, 1e-6, 0.5, 0.08, 0.9, 0.999,
                      1e-8, 0.01, 5.0, &sess) != ET_MTP_OK ||
      sess == nullptr) {
    std::fprintf(stderr, "C++17 bridge open failed\n");
    return 1;
  }
  if (et_mtp_aot_total_params(sess) != static_cast<int64_t>(lay.total_params)) {
    et_mtp_aot_close(sess);
    std::fprintf(stderr, "C++17 bridge param count mismatch\n");
    return 1;
  }
  et_mtp_aot_close(sess);
  return 0;
}
