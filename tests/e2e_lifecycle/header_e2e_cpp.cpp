#include "eshkol_transformer/e2e_lifecycle_abi.h"
#include "e2e_lifecycle_eshkol_bridge.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_e2e_stage_metrics_v1>);
static_assert(std::is_same_v<et_e2e_status, int32_t>);
static_assert(ET_E2E_OK == 0);
static_assert(ET_E2E_MODEL_DIM == 16u);
static_assert(ET_E2E_LORA_RANK == 4u);
static_assert(ET_E2E_SPEC_HORIZON == 4u);

int main(int argc, char **argv) {
  if (argc != 2 || argv[1] == nullptr || argv[1][0] == '\0') {
    std::fprintf(stderr, "usage: header_e2e_cpp <work_dir>\n");
    return 2;
  }

  et_e2e_aot_ctx *ctx = nullptr;
  if (et_e2e_aot_create(&ctx) != ET_E2E_OK || ctx == nullptr) {
    std::fprintf(stderr, "C++17 linkage: et_e2e_aot_create failed\n");
    return 1;
  }
  et_e2e_aot_destroy(ctx);

  double fd_err = 1.0;
  if (et_e2e_stage3_check_fd_gradients(&fd_err) != ET_E2E_OK ||
      !std::isfinite(fd_err) || fd_err > 2.0e-3) {
    std::fprintf(stderr, "C++17 linkage: FD check failed (err=%g)\n", fd_err);
    return 1;
  }

  et_e2e_stage_metrics_v1 metrics{};
  if (et_e2e_run_full_in_process(argv[1], &metrics) != ET_E2E_OK) {
    std::fprintf(stderr, "C++17 linkage: full in-process run failed\n");
    return 1;
  }
  if (metrics.s1_kept_docs != 6u || metrics.s2_lossless_verified != 1u ||
      metrics.s3_weights_bitwise_equal != 1u ||
      metrics.s4_base_frozen_equal != 1u ||
      metrics.s5_ref_frozen_equal != 1u ||
      metrics.s6_fresh_reload_verified != 1u ||
      metrics.s7_greedy_exact_match != 1u || metrics.s8_weight_count == 0u) {
    std::fprintf(stderr, "C++17 linkage: metric verification failed\n");
    return 1;
  }
  return 0;
}
