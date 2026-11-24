#include "eshkol_transformer/radix_kv_abi.h"
#include "radix_kv_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_radix_kv_config>);
static_assert(std::is_standard_layout_v<et_radix_kv_stats>);
static_assert(std::is_standard_layout_v<et_radix_kv_match_result>);
static_assert(std::is_same_v<et_radix_kv_status, int32_t>);
static_assert(std::is_same_v<et_radix_kv_handle, uint64_t>);

int main() {
  const et_radix_kv_config cfg = {
      32u, /* max_kv_tokens */
      16u, /* max_nodes */
      2u,  /* n_layers */
      2u,  /* n_kv_heads */
      4u,  /* head_dim */
      0u   /* reserved */
  };
  if (et_radix_kv_validate_config(&cfg) != ET_RADIX_OK)
    return 1;

  size_t pool_floats = 0u;
  if (et_radix_kv_pool_float_count(&cfg, &pool_floats) != ET_RADIX_OK ||
      pool_floats != 32u * 2u * 2u * 2u * 4u)
    return 2;

  et_radix_kv_tree *tree = nullptr;
  if (et_radix_kv_create(&cfg, &tree) != ET_RADIX_OK || tree == nullptr)
    return 3;

  const uint32_t seq_a[6] = {10u, 20u, 30u, 40u, 50u, 60u};
  float k_a[6 * 2 * 2 * 4];
  float v_a[6 * 2 * 2 * 4];
  for (size_t i = 0u; i < 6u * 16u; ++i) {
    k_a[i] = 0.0625f * static_cast<float>(static_cast<int>(i % 17u) - 8);
    v_a[i] = 0.0625f * static_cast<float>(static_cast<int>((i * 3u) % 19u) - 9);
  }

  et_radix_kv_handle h_a = ET_RADIX_KV_HANDLE_INVALID;
  uint32_t matched_ins = 99u;
  if (et_radix_kv_insert(tree, seq_a, 6u, k_a, v_a, &h_a, &matched_ins) !=
          ET_RADIX_OK ||
      matched_ins != 0u)
    return 4;

  const uint32_t seq_b[5] = {10u, 20u, 30u, 77u, 88u};
  et_radix_kv_match_result mres = {};
  uint32_t slots[5] = {};
  if (et_radix_kv_match_prefix(tree, seq_b, 5u, &mres, slots, 5u) !=
          ET_RADIX_OK ||
      mres.matched_len != 3u || mres.did_split != 1u)
    return 5;

  if (et_radix_kv_pin(tree, mres.handle) != ET_RADIX_OK)
    return 6;

  et_radix_kv_stats st = {};
  if (et_radix_kv_stats_get(tree, &st) != ET_RADIX_OK ||
      st.used_tokens != 6u || st.pinned_tokens != 3u ||
      st.evictable_tokens != 3u || st.total_edge_splits != 1u)
    return 7;

  /* Verify bitwise GQA parity between cached 3-token prefix + 3-token suffix
   * and uncached 6-token full prefill on layer 0. */
  float q_full[6 * 4 * 4];
  for (size_t i = 0u; i < 6u * 16u; ++i)
    q_full[i] = 0.0625f * static_cast<float>(static_cast<int>((i * 5u) % 15u) - 7);

  float k_l0[6 * 2 * 4];
  float v_l0[6 * 2 * 4];
  for (uint32_t t = 0u; t < 6u; ++t) {
    std::memcpy(k_l0 + t * 8u, k_a + t * 16u, 8u * sizeof(float));
    std::memcpy(v_l0 + t * 8u, v_a + t * 16u, 8u * sizeof(float));
  }

  float out_full[6 * 4 * 4] = {};
  float out_cached[3 * 4 * 4] = {};
  const et_radix_kv_handle root_h = et_radix_kv_root_handle(tree);
  if (et_radix_kv_attend_gqa_f32(tree, root_h, 0u, 4u, 6u, 6u, 0.5f, q_full,
                                 k_l0, v_l0, out_full) != ET_RADIX_OK)
    return 8;
  if (et_radix_kv_attend_gqa_f32(tree, mres.handle, 0u, 4u, 3u, 3u, 0.5f,
                                 q_full + 3u * 16u, k_l0 + 3u * 8u,
                                 v_l0 + 3u * 8u, out_cached) != ET_RADIX_OK)
    return 9;
  if (std::memcmp(out_cached, out_full + 3u * 16u, sizeof(out_cached)) != 0)
    return 10;

  if (et_radix_kv_unpin(tree, mres.handle) != ET_RADIX_OK)
    return 11;
  if (et_radix_kv_verify_invariants(tree) != ET_RADIX_OK)
    return 12;

  et_radix_kv_destroy(tree);

  et_radix_kv_aot_ctx *ctx = nullptr;
  if (et_radix_kv_aot_create(32, 16, 2, 2, 4, &ctx) != ET_RADIX_OK ||
      ctx == nullptr)
    return 13;
  et_radix_kv_aot_destroy(ctx);
  return 0;
}
