#include "eshkol_transformer/rope_yarn_abi.h"
#include "rope_yarn_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_rope_yarn_config_t>);
static_assert(std::is_trivially_copyable_v<et_rope_yarn_config_t>);
static_assert(std::is_standard_layout_v<et_rope_yarn_invariance_result_t>);
static_assert(std::is_trivially_copyable_v<et_rope_yarn_invariance_result_t>);
static_assert(std::is_same_v<et_rope_yarn_status, int32_t>);

int main() {
  et_rope_yarn_config_t cfg{};
  cfg.mode = ET_ROPE_SCALE_YARN;
  cfg.layout = ET_ROPE_LAYOUT_NEOX_SPLIT_HALF;
  cfg.head_dim = 12u;
  cfg.rotary_dim = 8u;
  cfg.orig_max_pos = 2048u;
  cfg.base = ET_ROPE_YARN_DEFAULT_BASE;
  cfg.scale_factor = 4.0;
  cfg.beta_fast = ET_ROPE_YARN_DEFAULT_BETA_FAST;
  cfg.beta_slow = ET_ROPE_YARN_DEFAULT_BETA_SLOW;

  if (et_rope_yarn_validate_config_v1(&cfg) != ET_ROPE_YARN_OK) {
    std::fprintf(stderr, "C++17 validate_config failed\n");
    return EXIT_FAILURE;
  }

  double inv_freq[4] = {};
  double mscale = 0.0;
  if (et_rope_yarn_compute_freqs_v1(&cfg, inv_freq, 4u, &mscale) !=
      ET_ROPE_YARN_OK) {
    std::fprintf(stderr, "C++17 compute_freqs failed\n");
    return EXIT_FAILURE;
  }
  const double expected_mscale = 0.1 * std::log(4.0) + 1.0;
  if (std::fabs(mscale - expected_mscale) > 1e-14) {
    std::fprintf(stderr, "C++17 mscale mismatch: %.17g vs %.17g\n", mscale,
                 expected_mscale);
    return EXIT_FAILURE;
  }

  float x[12] = {0.5f, -0.25f, 0.75f, 0.125f, -0.5f, 0.375f,
                 0.25f, -0.75f, 1.25f, -2.5f, 3.75f, -4.0f};
  float y[12] = {};
  float dx[12] = {};
  int64_t pos[1] = {4096};
  if (et_rope_yarn_forward_v1(&cfg, 1u, 1u, 1u, x, pos, y) != ET_ROPE_YARN_OK ||
      et_rope_yarn_backward_v1(&cfg, 1u, 1u, 1u, y, pos, dx) !=
          ET_ROPE_YARN_OK) {
    std::fprintf(stderr, "C++17 forward/backward failed\n");
    return EXIT_FAILURE;
  }
  /* Unrotated tail [8..11] must pass through identically. */
  for (int d = 8; d < 12; ++d) {
    if (y[d] != x[d] || dx[d] != x[d]) {
      std::fprintf(stderr, "C++17 partial rotary tail mismatch at d=%d\n", d);
      return EXIT_FAILURE;
    }
  }

  et_rope_yarn_invariance_result_t inv{};
  if (et_rope_yarn_qk_dot_invariance_check_v1(&cfg, x, y, 17, 5, 8175, &inv) !=
          ET_ROPE_YARN_OK ||
      inv.shift_abs_err > 1e-5 || inv.relative_abs_err > 1e-5) {
    std::fprintf(stderr, "C++17 invariance check failed\n");
    return EXIT_FAILURE;
  }

  /* Verify AOT bridge C++17 linkage as well. */
  double aot_freqs[4] = {};
  double aot_mscale = 0.0;
  if (et_rope_yarn_aot_compute_freqs(
          ET_ROPE_SCALE_YARN, ET_ROPE_LAYOUT_NEOX_SPLIT_HALF, 12, 8, 2048,
          10000.0, 4.0, 32.0, 1.0, aot_freqs, 4, &aot_mscale) !=
      ET_ROPE_YARN_OK) {
    std::fprintf(stderr, "C++17 bridge compute_freqs failed\n");
    return EXIT_FAILURE;
  }
  if (aot_mscale != mscale || aot_freqs[0] != inv_freq[0]) {
    std::fprintf(stderr, "C++17 bridge mismatch\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
