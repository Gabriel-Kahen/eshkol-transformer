#include "eshkol_transformer/mod5_generate_abi.h"

#include <type_traits>

static_assert(ET_MOD5_GEN_ABI_MAJOR == 1u);
static_assert(ET_MOD5_GEN_ABI_MINOR == 0u);
static_assert(ET_MOD5_GEN_MAX_SEQ == 64u);
static_assert(ET_MOD5_GEN_MAX_DIM == 32u);
static_assert(ET_MOD5_GEN_MAX_FFN == 64u);
static_assert(ET_MOD5_GEN_MAX_LAYERS == 8u);
static_assert(std::is_same_v<et_mod5_gen_status, int32_t>);
static_assert(std::is_standard_layout_v<et_mod5_gen_config>);
static_assert(std::is_standard_layout_v<et_mod5_gen_layer_weights>);
static_assert(std::is_standard_layout_v<et_mod5_gen_model_weights>);
static_assert(std::is_standard_layout_v<et_mod5_gen_rng>);
static_assert(std::is_standard_layout_v<et_mod5_gen_sampling_config>);
static_assert(std::is_standard_layout_v<et_mod5_gen_result>);

int main() {
  double max_diff = -1.0;
  if (et_mod5_gen_verify_provider(&max_diff) != ET_MOD5_GEN_OK ||
      max_diff != 0.0) {
    return 1;
  }

  et_mod5_gen_rng rng{};
  if (et_mod5_gen_rng_init(&rng, 77u) != ET_MOD5_GEN_OK) {
    return 2;
  }

  const float logits[4] = {0.25f, 1.5f, 1.5f, -0.5f};
  et_mod5_gen_sampling_config sc{};
  sc.mode = ET_MOD5_GEN_SAMPLE_GREEDY;
  sc.eos_token_id = -1;
  int64_t tok = -1;
  if (et_mod5_gen_sample(logits, 4u, &sc, &rng, &tok) != ET_MOD5_GEN_OK) {
    return 3;
  }
  /* Greedy tie-breaks to lowest index (1) and must not advance RNG. */
  if (tok != 1 || rng.counter != 0u) {
    return 4;
  }
  return 0;
}
