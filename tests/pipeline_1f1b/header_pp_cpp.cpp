#include "eshkol_transformer/pipeline_1f1b_abi.h"
#include "pipeline_1f1b_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

static_assert(std::is_standard_layout_v<et_pp_frame_header_t>);
static_assert(std::is_standard_layout_v<et_pp_schedule_step_t>);
static_assert(std::is_standard_layout_v<et_pp_schedule_t>);
static_assert(std::is_standard_layout_v<et_pp_config_t>);
static_assert(std::is_standard_layout_v<et_pp_run_telemetry_t>);
static_assert(std::is_same_v<et_pp_status_t, int32_t>);

int main() {
  // Verify schedule generation linkage from C++17.
  et_pp_schedule_t sched{};
  if (et_pp_build_schedule_v1(4u, 0u, 8u, ET_PP_SCHEDULE_1F1B, &sched) !=
      ET_PP_OK) {
    return 1;
  }
  if (sched.num_warmup_steps != 3u || sched.num_steady_pairs != 5u ||
      sched.num_cooldown_steps != 3u || sched.peak_live_activations != 4u ||
      sched.step_count != 16u) {
    return 2;
  }

  // Verify Eshkol AOT bridge linkage from C++17.
  int64_t summary[5] = {};
  double bubble = 0.0;
  if (et_pp_aot_build_schedule(4, 3, 8, ET_PP_SCHEDULE_1F1B, summary,
                               &bubble) != ET_PP_OK) {
    return 3;
  }
  if (summary[0] != 0 || summary[1] != 8 || summary[2] != 0 ||
      summary[3] != 16 || summary[4] != 1) {
    return 4;
  }
  if (std::fabs(bubble - (3.0 / 11.0)) > 1e-15) {
    return 5;
  }

  // Verify real multi-process P=2 1F1B linkage from C++17.
  et_pp_config_t cfg{};
  cfg.abi_major = ET_PP_ABI_MAJOR;
  cfg.abi_minor = ET_PP_ABI_MINOR;
  cfg.num_stages = 2u;
  cfg.layers_per_stage = 1u;
  cfg.num_microbatches = 4u;
  cfg.tokens_per_mb = 2u;
  cfg.d_model = 4u;
  cfg.d_ff = 6u;
  cfg.out_dim = 4u;
  cfg.schedule_kind = ET_PP_SCHEDULE_1F1B;
  cfg.loss_type = ET_PP_LOSS_CROSS_ENTROPY;
  cfg.timeout_ms = 5000u;
  cfg.rms_eps = 1e-5;

  size_t pcount = 0;
  if (et_pp_param_count_v1(&cfg, &pcount) != ET_PP_OK || pcount == 0) {
    return 6;
  }
  std::vector<double> params(pcount, 0.05);
  std::vector<double> inputs(cfg.num_microbatches * cfg.tokens_per_mb *
                                 cfg.d_model,
                             0.1);
  std::vector<int32_t> targets(cfg.num_microbatches * cfg.tokens_per_mb, 1);
  std::vector<double> grads_ref(pcount, 0.0);
  std::vector<double> grads_pp(pcount, 0.0);
  double loss_ref = 0.0;
  et_pp_run_telemetry_t telem{};

  if (et_pp_run_reference_step_v1(&cfg, inputs.data(), targets.data(), nullptr,
                                  params.data(), grads_ref.data(),
                                  &loss_ref) != ET_PP_OK) {
    return 7;
  }
  if (et_pp_run_multiprocess_step_v1(&cfg, inputs.data(), targets.data(),
                                     nullptr, params.data(), grads_pp.data(),
                                     &telem) != ET_PP_OK) {
    return 8;
  }
  if (telem.distinct_pid_count != 2u || telem.stage_peak_live[0] != 2u ||
      telem.stage_peak_live[1] != 1u) {
    return 9;
  }
  if (std::fabs(telem.total_loss - loss_ref) > 1e-12) {
    return 10;
  }
  for (size_t i = 0; i < pcount; ++i) {
    if (std::fabs(grads_pp[i] - grads_ref[i]) > 1e-12) {
      return 11;
    }
  }
  return 0;
}
