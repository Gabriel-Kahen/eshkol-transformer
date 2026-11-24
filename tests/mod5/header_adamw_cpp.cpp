#include "eshkol_transformer/mod5_adamw_abi.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <unistd.h>

static_assert(ET_MOD5_ADAMW_ABI_MAJOR == 1u);
static_assert(ET_MOD5_ADAMW_ABI_MINOR == 0u);
static_assert(ET_MOD5_ADAMW_SNAPSHOT_HEADER_BYTES == 224u);
static_assert(ET_MOD5_ADAMW_SNAPSHOT_CHECKSUM_BYTES == 32u);
static_assert(ET_MOD5_ADAMW_MAX_TOKENIZER_ID_BYTES == 128u);

static_assert(std::is_standard_layout_v<et_mod5_lr_schedule_config_v1>);
static_assert(std::is_trivially_copyable_v<et_mod5_lr_schedule_config_v1>);
static_assert(std::is_standard_layout_v<et_mod5_dataset_cursor_v1>);
static_assert(std::is_trivially_copyable_v<et_mod5_dataset_cursor_v1>);
static_assert(std::is_standard_layout_v<et_mod5_adamw_config_v1>);
static_assert(std::is_trivially_copyable_v<et_mod5_adamw_config_v1>);
static_assert(std::is_standard_layout_v<et_mod5_adamw_step_info_v1>);
static_assert(std::is_trivially_copyable_v<et_mod5_adamw_step_info_v1>);

static_assert(std::is_same_v<decltype(&et_mod5_adamw_create),
                             et_mod5_adamw_status (*)(
                                 const et_mod5_adamw_config_v1 *,
                                 et_mod5_adamw_state **)>);
static_assert(std::is_same_v<decltype(&et_mod5_adamw_destroy),
                             void (*)(et_mod5_adamw_state *)>);
static_assert(std::is_same_v<decltype(&et_mod5_adamw_step),
                             et_mod5_adamw_status (*)(
                                 et_mod5_adamw_state *, const float *, size_t,
                                 uint64_t, et_mod5_adamw_step_info_v1 *)>);
static_assert(std::is_same_v<decltype(&et_mod5_adamw_save_snapshot),
                             et_mod5_adamw_status (*)(
                                 const et_mod5_adamw_state *, const char *)>);
static_assert(std::is_same_v<decltype(&et_mod5_adamw_load_snapshot),
                             et_mod5_adamw_status (*)(et_mod5_adamw_state *,
                                                      const char *)>);
static_assert(std::is_same_v<decltype(&et_mod5_adamw_compute_lr),
                             et_mod5_adamw_status (*)(
                                 const et_mod5_lr_schedule_config_v1 *,
                                 uint64_t, double *)>);

int main(int argc, char **argv) {
  const char *path = (argc > 1 && argv[1][0] != '\0')
                         ? argv[1]
                         : "/tmp/et_mod5_adamw_cpp_smoke.etsnap";

  et_mod5_adamw_config_v1 cfg{};
  cfg.param_count = 4u;
  cfg.beta1 = 0.9f;
  cfg.beta2 = 0.999f;
  cfg.eps = 1e-8f;
  cfg.weight_decay = 0.01f;
  cfg.max_grad_norm = 1.0f;
  cfg.schedule.kind = ET_MOD5_LR_LINEAR_WARMUP_COSINE_DECAY;
  cfg.schedule.base_lr = 0.05;
  cfg.schedule.min_lr = 0.005;
  cfg.schedule.warmup_steps = 4u;
  cfg.schedule.total_steps = 16u;
  cfg.tokenizer_id = "mod5-cpp-smoke-v1";
  const char *tag = "mod5-cpp-model-v1";
  if (et_mod5_adamw_sha256_bytes(reinterpret_cast<const uint8_t *>(tag),
                                 std::strlen(tag),
                                 cfg.model_config_hash) != ET_MOD5_ADAMW_OK) {
    return 1;
  }

  et_mod5_adamw_state *s1 = nullptr;
  et_mod5_adamw_state *s2 = nullptr;
  if (et_mod5_adamw_create(&cfg, &s1) != ET_MOD5_ADAMW_OK || s1 == nullptr) {
    return 2;
  }
  if (et_mod5_adamw_create(&cfg, &s2) != ET_MOD5_ADAMW_OK || s2 == nullptr) {
    et_mod5_adamw_destroy(s1);
    return 3;
  }

  const float init_p[4] = {1.0f, -0.5f, 0.25f, 0.75f};
  const float grads[4] = {0.2f, -0.4f, 0.1f, 0.3f};
  const et_mod5_dataset_cursor_v1 cur{1u, 8u, 2u};
  et_mod5_adamw_step_info_v1 info{};
  if (et_mod5_adamw_set_params(s1, init_p, 4u) != ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_set_cursor(s1, &cur) != ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_step(s1, grads, 4u, 8u, &info) != ET_MOD5_ADAMW_OK ||
      info.step != 1u ||
      et_mod5_adamw_save_snapshot(s1, path) != ET_MOD5_ADAMW_OK ||
      et_mod5_adamw_load_snapshot(s2, path) != ET_MOD5_ADAMW_OK) {
    (void)unlink(path);
    et_mod5_adamw_destroy(s1);
    et_mod5_adamw_destroy(s2);
    return 4;
  }

  float p1[4] = {}, p2[4] = {}, m1[4] = {}, m2[4] = {}, v1[4] = {}, v2[4] = {};
  et_mod5_dataset_cursor_v1 cur2{};
  const int ok =
      et_mod5_adamw_get_params(s1, p1, 4u) == ET_MOD5_ADAMW_OK &&
      et_mod5_adamw_get_params(s2, p2, 4u) == ET_MOD5_ADAMW_OK &&
      et_mod5_adamw_get_moments(s1, m1, v1, 4u) == ET_MOD5_ADAMW_OK &&
      et_mod5_adamw_get_moments(s2, m2, v2, 4u) == ET_MOD5_ADAMW_OK &&
      et_mod5_adamw_get_cursor(s2, &cur2) == ET_MOD5_ADAMW_OK &&
      std::memcmp(p1, p2, sizeof(p1)) == 0 &&
      std::memcmp(m1, m2, sizeof(m1)) == 0 &&
      std::memcmp(v1, v2, sizeof(v1)) == 0 &&
      et_mod5_adamw_get_step(s2) == 1u &&
      et_mod5_adamw_get_cumulative_tokens(s2) == 8u &&
      cur2.shard_index == 1u && cur2.token_offset == 8u && cur2.epoch == 2u;

  (void)unlink(path);
  et_mod5_adamw_destroy(s1);
  et_mod5_adamw_destroy(s2);
  return ok ? 0 : 5;
}
