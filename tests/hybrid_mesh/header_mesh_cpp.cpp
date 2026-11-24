#include "eshkol_transformer/hybrid_mesh_abi.h"
#include "hybrid_mesh_eshkol_bridge.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_mesh_coords_v1>,
              "et_mesh_coords_v1 must be standard layout");
static_assert(std::is_standard_layout_v<et_mesh_comm_stats_v1>,
              "et_mesh_comm_stats_v1 must be standard layout");
static_assert(std::is_standard_layout_v<et_mesh_model_config_v1>,
              "et_mesh_model_config_v1 must be standard layout");
static_assert(std::is_standard_layout_v<et_mesh_optim_config_v1>,
              "et_mesh_optim_config_v1 must be standard layout");
static_assert(std::is_standard_layout_v<et_mesh_memory_report_v1>,
              "et_mesh_memory_report_v1 must be standard layout");
static_assert(std::is_standard_layout_v<et_mesh_step_info_v1>,
              "et_mesh_step_info_v1 must be standard layout");
static_assert(std::is_same_v<et_mesh_status_t, int32_t>,
              "et_mesh_status_t must be int32_t");

int main() {
  et_mesh_coords_v1 coords{};
  if (et_mesh_validate_topology(2u, 2u, 3u, &coords) != ET_MESH_OK ||
      coords.dp_rank != 1u || coords.tp_rank != 1u || coords.world_size != 4u) {
    std::fprintf(stderr, "C++17 validate_topology failed\n");
    return 1;
  }

  et_mesh_t *mesh = nullptr;
  if (et_mesh_create_fds(ET_MESH_ABI_MAJOR, ET_MESH_ABI_MINOR, 0u, 1u, 1u,
                         nullptr, 0u, nullptr, 0u, &mesh) != ET_MESH_OK ||
      mesh == nullptr) {
    std::fprintf(stderr, "C++17 create_fds(1x1) failed\n");
    return 1;
  }

  et_mesh_model_config_v1 mcfg{};
  mcfg.d_model = 8u;
  mcfg.d_ff = 16u;
  mcfg.num_classes = 6u;
  mcfg.global_batch = 8u;
  mcfg.rms_eps = 1e-5f;
  mcfg.reserved = 0u;

  et_mesh_optim_config_v1 ocfg{};
  ocfg.lr = 0.05;
  ocfg.beta1 = 0.9;
  ocfg.beta2 = 0.999;
  ocfg.eps = 1e-8;
  ocfg.weight_decay = 0.01;
  ocfg.max_grad_norm = 0.5;

  et_mesh_engine_t *engine = nullptr;
  if (et_mesh_engine_create(mesh, &mcfg, &ocfg, &engine) != ET_MESH_OK ||
      engine == nullptr) {
    std::fprintf(stderr, "C++17 engine_create failed\n");
    et_mesh_destroy(mesh);
    return 1;
  }

  et_mesh_memory_report_v1 rep{};
  if (et_mesh_engine_get_memory_report(engine, &rep) != ET_MESH_OK ||
      rep.full_total_params != 454u || rep.full_sharded_weight_bytes != 1536u ||
      rep.replicated_weight_bytes != 280u) {
    std::fprintf(stderr, "C++17 memory_report failed\n");
    et_mesh_engine_destroy(engine);
    et_mesh_destroy(mesh);
    return 1;
  }

  et_mesh_engine_destroy(engine);
  et_mesh_destroy(mesh);

  et_mesh_aot_session_t *sess = nullptr;
  if (et_mesh_aot_open("", 0, 1, 1, 8, 16, 6, 8, 1e-5, 0.05, 0.9, 0.999, 1e-8,
                       0.01, 0.5, 5000, &sess) != ET_MESH_OK ||
      sess == nullptr) {
    std::fprintf(stderr, "C++17 aot_open failed\n");
    return 1;
  }
  if (et_mesh_aot_init_params(sess, 20261124) != ET_MESH_OK ||
      et_mesh_aot_train_step(sess, 0, 100, 5000) != ET_MESH_OK ||
      !std::isfinite(et_mesh_aot_last_global_loss(sess)) ||
      et_mesh_aot_step(sess) != 1) {
    std::fprintf(stderr, "C++17 aot_train_step failed\n");
    et_mesh_aot_close(sess);
    return 1;
  }
  et_mesh_aot_close(sess);
  return 0;
}
