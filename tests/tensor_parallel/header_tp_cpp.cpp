#include "eshkol_transformer/tensor_parallel_abi.h"
#include "tensor_parallel_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_tp_collective_stats_t>);
static_assert(std::is_standard_layout_v<et_tp_swiglu_config_t>);
static_assert(std::is_standard_layout_v<et_tp_gqa_config_t>);
static_assert(std::is_standard_layout_v<et_tp_block_config_t>);
static_assert(std::is_standard_layout_v<et_tp_block_memory_stats_t>);
static_assert(std::is_standard_layout_v<et_tp_block_weights_t>);
static_assert(std::is_standard_layout_v<et_tp_block_grads_t>);
static_assert(std::is_standard_layout_v<et_tp_block_cache_t>);
static_assert(std::is_same_v<et_tp_status_t, int32_t>);

int main() {
  const et_tp_block_config_t cfg = {1u, 2u, 16u, 32u, 8u, 4u, 2u, 1e-5f};
  if (et_tp_block_validate_config(&cfg, 1u) != ET_TP_OK)
    return 1;
  if (et_tp_block_validate_config(&cfg, 2u) != ET_TP_OK)
    return 2;
  if (et_tp_block_validate_config(&cfg, 4u) != ET_TP_OK)
    return 3;
  if (et_tp_block_validate_config(&cfg, 3u) != ET_TP_ERR_INVALID_ARG)
    return 4;

  et_tp_block_memory_stats_t m1{}, m2{}, m4{};
  if (et_tp_block_memory_stats_v1(&cfg, 1u, &m1) != ET_TP_OK)
    return 5;
  if (et_tp_block_memory_stats_v1(&cfg, 2u, &m2) != ET_TP_OK)
    return 6;
  if (et_tp_block_memory_stats_v1(&cfg, 4u, &m4) != ET_TP_OK)
    return 7;
  if (m2.rank_sharded_weight_bytes * 2u != m1.full_sharded_weight_bytes)
    return 8;
  if (m4.rank_sharded_weight_bytes * 4u != m1.full_sharded_weight_bytes)
    return 9;

  et_tp_group_t *g = nullptr;
  if (et_tp_group_create(ET_TP_ABI_MAJOR, ET_TP_ABI_MINOR, 0u, 1u, nullptr, 0u,
                         1000, &g) != ET_TP_OK ||
      g == nullptr)
    return 10;

  const float x[4] = {0.5f, -0.25f, 0.75f, 0.125f};
  const float w[8] = {0.2f, -0.1f, 0.3f, 0.4f, -0.5f, 0.6f, 0.1f, -0.2f};
  const float b[2] = {0.05f, -0.05f};
  float y[2] = {0.0f, 0.0f};
  if (et_tp_column_linear_forward_v1(g, 1u, 4u, 2u, x, w, b, y) != ET_TP_OK)
    return 11;
  if (!std::isfinite(y[0]) || !std::isfinite(y[1]))
    return 12;
  et_tp_group_destroy(g);

  et_tp_aot_session_t *s = nullptr;
  if (et_tp_aot_session_create(1, 1, 2, 16, 32, 8, 4, 2, 1e-5, &s) !=
          ET_TP_OK ||
      s == nullptr)
    return 13;
  if (et_tp_aot_forward(s) != ET_TP_OK)
    return 14;
  const double loss = et_tp_aot_mse_loss(s);
  if (!std::isfinite(loss) || !(loss >= 0.0))
    return 15;
  if (et_tp_aot_backward(s) != ET_TP_OK)
    return 16;
  et_tp_aot_session_destroy(s);
  return 0;
}
