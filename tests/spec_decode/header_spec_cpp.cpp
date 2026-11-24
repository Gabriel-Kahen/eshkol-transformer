#include "eshkol_transformer/spec_decode_abi.h"
#include "spec_decode_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_spec_sampling_params_v1>);
static_assert(std::is_standard_layout_v<et_spec_rng_state_v1>);
static_assert(std::is_standard_layout_v<et_spec_verify_result_v1>);
static_assert(ET_SPEC_MIN_HORIZON == 1u);
static_assert(ET_SPEC_MAX_HORIZON == 16u);
static_assert(ET_SPEC_MIN_VOCAB == 2u);
static_assert(ET_SPEC_MAX_VOCAB == 4096u);
static_assert(ET_SPEC_AOT_ERR_NOMEM == -100);

int main() {
  constexpr uint32_t K = 2u;
  constexpr uint32_t V = 4u;
  const uint32_t draft_tokens[K] = {1u, 2u};
  const float draft_logits[K * V] = {
      0.1f, 2.0f, 0.2f, -0.5f,
      -0.3f, 0.4f, 1.8f, 0.0f,
  };
  const float target_logits[(K + 1u) * V] = {
      0.1f, 2.0f, 0.2f, -0.5f,
      -0.3f, 0.4f, 1.8f, 0.0f,
      0.0f, 0.1f, 0.2f, 2.5f,
  };
  uint32_t out_tokens[K + 1u] = {0u, 0u, 0u};
  et_spec_verify_result_v1 res{};

  if (et_spec_verify_greedy_v1(draft_tokens, target_logits, K, V,
                               ET_SPEC_NO_EOS, out_tokens, K + 1u,
                               &res) != ET_SPEC_OK) {
    std::fprintf(stderr, "C++17 greedy verify failed\n");
    return 1;
  }
  if (res.accepted_count != 2u || res.emitted_count != 3u ||
      out_tokens[0] != 1u || out_tokens[1] != 2u || out_tokens[2] != 3u) {
    std::fprintf(stderr, "C++17 greedy verify mismatch\n");
    return 2;
  }

  const et_spec_sampling_params_v1 params = {1.0, 1.0, V, ET_SPEC_NO_EOS};
  const et_spec_rng_state_v1 rng = {UINT64_C(42), UINT64_C(0)};
  if (et_spec_verify_sample_v1(draft_tokens, draft_logits, target_logits, K, V,
                               &params, &rng, out_tokens, K + 1u,
                               &res) != ET_SPEC_OK) {
    std::fprintf(stderr, "C++17 sample verify failed\n");
    return 3;
  }
  if (res.accepted_count != 2u || res.emitted_count != 3u ||
      res.rng_draws != 3u) {
    std::fprintf(stderr, "C++17 sample verify p==q mismatch\n");
    return 4;
  }

  et_spec_aot_context *ctx = nullptr;
  if (et_spec_aot_create(K, V, &ctx) != ET_SPEC_OK || ctx == nullptr) {
    std::fprintf(stderr, "C++17 bridge create failed\n");
    return 5;
  }
  for (uint32_t i = 0u; i < K; ++i) {
    if (et_spec_aot_set_draft_token(ctx, i, draft_tokens[i]) != ET_SPEC_OK)
      return 6;
    for (uint32_t v = 0u; v < V; ++v) {
      if (et_spec_aot_set_draft_logit(ctx, i, v, draft_logits[i * V + v]) !=
          ET_SPEC_OK)
        return 7;
    }
  }
  for (uint32_t i = 0u; i <= K; ++i) {
    for (uint32_t v = 0u; v < V; ++v) {
      if (et_spec_aot_set_target_logit(ctx, i, v, target_logits[i * V + v]) !=
          ET_SPEC_OK)
        return 8;
    }
  }
  if (et_spec_aot_verify_greedy(ctx, ET_SPEC_AOT_NO_EOS) != ET_SPEC_OK)
    return 9;
  if (et_spec_aot_accepted_count(ctx) != 2 ||
      et_spec_aot_emitted_count(ctx) != 3 ||
      et_spec_aot_emitted_token(ctx, 2) != 3)
    return 10;
  if (et_spec_aot_logits_to_probs(ctx, 1, 2, 1.0, V, 1.0) != ET_SPEC_OK)
    return 11;
  double sum = 0.0;
  for (uint32_t v = 0u; v < V; ++v)
    sum += et_spec_aot_prob(ctx, v);
  if (std::fabs(sum - 1.0) > 1e-12)
    return 12;

  et_spec_aot_destroy(ctx);
  return 0;
}
