#include "eshkol_transformer/beam_search_abi.h"
#include "beam_search_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

static_assert(ET_BEAM_ABI_MAJOR == 1u);
static_assert(ET_BEAM_ABI_MINOR == 0u);
static_assert(ET_BEAM_OK == 0);
static_assert(std::is_same_v<et_beam_status, int32_t>);
static_assert(std::is_standard_layout_v<et_beam_config_v1>);
static_assert(std::is_standard_layout_v<et_beam_hypothesis_v1>);
static_assert(std::is_standard_layout_v<et_beam_state_v1>);

static void require(bool cond, const char *msg) {
  if (!cond) {
    std::fprintf(stderr, "header_beam_cpp FAIL: %s\n", msg);
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  et_beam_config_v1 cfg{};
  cfg.struct_size = sizeof(et_beam_config_v1);
  cfg.num_beams = 2u;
  cfg.vocab_size = 4u;
  cfg.eos_token_id = 0u;
  cfg.max_new_tokens = 2u;
  cfg.min_new_tokens = 0u;
  cfg.no_repeat_ngram_size = 2u;
  cfg.early_stopping = 1u;
  cfg.reserved = 0u;
  cfg.length_penalty = 1.0;
  cfg.repetition_penalty = 1.25;

  require(et_beam_validate_config_v1(&cfg, 1u) == ET_BEAM_OK,
          "validate_config");

  auto *state = static_cast<et_beam_state_v1 *>(
      std::calloc(1u, sizeof(et_beam_state_v1)));
  require(state != nullptr, "alloc state");

  const uint32_t prompt[1] = {1u};
  require(et_beam_init_v1(state, &cfg, prompt, 1u) == ET_BEAM_OK, "beam_init");
  require(et_beam_active_count_v1(state) == 1u, "active_count == 1");

  const double step0[4] = {-2.0, -1.0, 2.0, 1.5};
  require(et_beam_step_v1(state, step0, 1u) == ET_BEAM_OK, "step 0");
  require(et_beam_active_count_v1(state) == 2u, "active_count == 2");

  const double step1[8] = {
      3.0, -1.0, -1.0, -1.0,
      3.0, -1.0, -1.0, -1.0,
  };
  require(et_beam_step_v1(state, step1, 2u) == ET_BEAM_OK, "step 1");
  require(et_beam_is_done_v1(state) == 1u, "beam done");

  et_beam_hypothesis_v1 hyps[2];
  size_t count = 0u;
  require(et_beam_finalize_v1(state, hyps, 2u, &count) == ET_BEAM_OK,
          "finalize");
  require(count == 2u, "2 hypotheses");
  require(hyps[0].finished == 1u && hyps[0].tokens[1] == 2u &&
              hyps[0].tokens[2] == 0u,
          "top hypothesis");
  require(hyps[0].score > hyps[1].score, "strictly ranked");

  /* Verify bridge linkage from C++17. */
  require(et_beam_aot_config_bytes() ==
              static_cast<int64_t>(sizeof(et_beam_config_v1)),
          "bridge config_bytes");
  require(et_beam_aot_state_bytes() ==
              static_cast<int64_t>(sizeof(et_beam_state_v1)),
          "bridge state_bytes");
  require(et_beam_aot_hyp_bytes() ==
              static_cast<int64_t>(sizeof(et_beam_hypothesis_v1)),
          "bridge hyp_bytes");

  std::free(state);
  return 0;
}
