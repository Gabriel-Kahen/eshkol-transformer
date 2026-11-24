#include "eshkol_transformer/fsdp_zero_abi.h"
#include "fsdp_zero_eshkol_bridge.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_fsdp_config_v1>,
              "et_fsdp_config_v1 must be standard layout");
static_assert(std::is_standard_layout_v<et_fsdp_memory_report_v1>,
              "et_fsdp_memory_report_v1 must be standard layout");
static_assert(std::is_standard_layout_v<et_fsdp_comm_stats_v1>,
              "et_fsdp_comm_stats_v1 must be standard layout");
static_assert(std::is_standard_layout_v<et_fsdp_step_info_v1>,
              "et_fsdp_step_info_v1 must be standard layout");
static_assert(std::is_same_v<et_fsdp_status_t, int32_t>,
              "et_fsdp_status_t must be int32_t");
static_assert(std::is_same_v<et_zero_stage_t, uint32_t>,
              "et_zero_stage_t must be uint32_t");

int main() {
  et_fsdp_comm_t *comm = nullptr;
  if (et_fsdp_comm_create_fds(ET_FSDP_ABI_MAJOR, ET_FSDP_ABI_MINOR, 0u, 1u,
                              nullptr, 0u, &comm) != ET_FSDP_OK ||
      comm == nullptr) {
    std::fprintf(stderr, "C++17 comm_create_fds failed\n");
    return 1;
  }

  uint64_t num_params = 0u;
  if (et_fsdp_mlp_num_params(8u, 17u, 5u, &num_params) != ET_FSDP_OK ||
      num_params != 243u) {
    std::fprintf(stderr, "C++17 mlp_num_params failed\n");
    et_fsdp_comm_destroy(comm);
    return 1;
  }

  et_fsdp_config_v1 cfg{};
  cfg.stage = ET_ZERO_STAGE_3_FSDP;
  cfg.reserved = 0u;
  cfg.num_params = num_params;
  cfg.lr = 0.05;
  cfg.beta1 = 0.9;
  cfg.beta2 = 0.999;
  cfg.eps = 1e-8;
  cfg.weight_decay = 0.01;
  cfg.max_grad_norm = 0.5;

  et_fsdp_engine_t *engine = nullptr;
  if (et_fsdp_engine_create(comm, &cfg, &engine) != ET_FSDP_OK ||
      engine == nullptr) {
    std::fprintf(stderr, "C++17 engine_create failed\n");
    et_fsdp_comm_destroy(comm);
    return 1;
  }

  et_fsdp_memory_report_v1 rep{};
  if (et_fsdp_engine_get_memory_report(engine, &rep) != ET_FSDP_OK ||
      rep.padded_params != 243u || rep.shard_len != 243u) {
    std::fprintf(stderr, "C++17 memory_report failed\n");
    et_fsdp_engine_destroy(engine);
    et_fsdp_comm_destroy(comm);
    return 1;
  }

  et_fsdp_engine_destroy(engine);
  et_fsdp_comm_destroy(comm);

  et_fsdp_aot_session_t *sess = nullptr;
  if (et_fsdp_aot_open("", 0, 1, ET_ZERO_STAGE_3_FSDP, 8, 17, 5, 8, 0.05, 0.9,
                       0.999, 1e-8, 0.01, 0.5, 5000, &sess) != ET_FSDP_OK ||
      sess == nullptr) {
    std::fprintf(stderr, "C++17 aot_open failed\n");
    return 1;
  }
  if (et_fsdp_aot_init_params(sess, 42) != ET_FSDP_OK ||
      et_fsdp_aot_train_step(sess, 0, 100, 5000) != ET_FSDP_OK ||
      !std::isfinite(et_fsdp_aot_last_loss(sess)) ||
      et_fsdp_aot_step(sess) != 1) {
    std::fprintf(stderr, "C++17 aot_train_step failed\n");
    et_fsdp_aot_close(sess);
    return 1;
  }
  et_fsdp_aot_close(sess);
  return 0;
}
