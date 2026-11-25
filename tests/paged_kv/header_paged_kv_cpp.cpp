#include "eshkol_transformer/paged_kv_abi.h"
#include "paged_kv_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_paged_kv_config>);
static_assert(std::is_standard_layout_v<et_paged_kv_stats>);
static_assert(std::is_same_v<et_paged_kv_status, int32_t>);

int main() {
  const et_paged_kv_config cfg = {
      8u, /* num_blocks */
      2u, /* num_layers */
      2u, /* num_kv_heads */
      4u, /* block_size */
      4u, /* head_dim */
      4u, /* max_seqs */
      4u, /* max_blocks_per_seq */
      0u  /* reserved */
  };
  if (et_paged_kv_validate_config(&cfg) != ET_PAGED_KV_OK)
    return 1;

  size_t pool_floats = 0u;
  if (et_paged_kv_pool_float_count(&cfg, &pool_floats) != ET_PAGED_KV_OK ||
      pool_floats != 8u * 2u * 2u * 2u * 4u * 4u)
    return 2;

  et_paged_kv_pool *pool = nullptr;
  if (et_paged_kv_create(&cfg, nullptr, 0u, &pool) != ET_PAGED_KV_OK ||
      pool == nullptr)
    return 3;

  uint32_t s0 = 0u;
  uint32_t s1 = 0u;
  if (et_paged_kv_seq_alloc(pool, &s0) != ET_PAGED_KV_OK)
    return 4;

  const float k_tok[16] = {0.25f, -0.5f, 0.5f, 0.125f, -0.25f, 0.5f,
                           0.25f, -0.125f, 0.125f, 0.25f, -0.25f, 0.5f,
                           0.5f, -0.25f, 0.125f, 0.25f};
  const float v_tok[16] = {1.0f, -0.5f, 0.25f, 0.75f, -0.5f, 0.25f,
                           0.5f, 0.125f, 0.25f, 0.5f, -0.75f, 0.25f,
                           0.75f, -0.25f, 0.5f, -0.5f};
  if (et_paged_kv_append_token(pool, s0, k_tok, v_tok) != ET_PAGED_KV_OK)
    return 5;
  if (et_paged_kv_seq_fork(pool, s0, &s1) != ET_PAGED_KV_OK)
    return 6;

  et_paged_kv_stats st = {};
  if (et_paged_kv_stats_get(pool, &st) != ET_PAGED_KV_OK ||
      st.used_blocks != 1u || st.shared_blocks != 1u || st.cow_copies != 0u)
    return 7;

  const float q[16] = {0.5f, 0.25f, -0.25f, 0.125f, 0.25f, -0.5f,
                       0.125f, 0.5f, -0.25f, 0.5f, 0.25f, -0.125f,
                       0.125f, 0.25f, -0.5f, 0.25f};
  float out0[16] = {0.0f};
  float out1[16] = {0.0f};
  if (et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, 0.5f, q, out0) !=
      ET_PAGED_KV_OK)
    return 8;
  if (et_paged_kv_decode_attention_v1(pool, s1, 0u, 4u, 0.5f, q, out1) !=
      ET_PAGED_KV_OK)
    return 9;
  if (std::memcmp(out0, out1, sizeof(out0)) != 0)
    return 10;

  if (et_paged_kv_seq_free(pool, s1) != ET_PAGED_KV_OK ||
      et_paged_kv_seq_free(pool, s0) != ET_PAGED_KV_OK)
    return 11;
  if (et_paged_kv_stats_get(pool, &st) != ET_PAGED_KV_OK ||
      st.free_blocks != st.total_blocks)
    return 12;

  et_paged_kv_destroy(pool);

  et_paged_kv_aot_ctx *ctx = nullptr;
  if (et_paged_kv_aot_create(8, 2, 2, 4, 4, 4, 4, &ctx) != ET_PAGED_KV_OK ||
      ctx == nullptr)
    return 13;
  et_paged_kv_aot_destroy(ctx);
  return 0;
}
