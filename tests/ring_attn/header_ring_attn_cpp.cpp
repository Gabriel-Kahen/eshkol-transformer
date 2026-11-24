#include "eshkol_transformer/ring_attn_abi.h"
#include "ring_attn_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

static_assert(ET_RING_ATTN_ABI_MAJOR == 1u);
static_assert(ET_RING_ATTN_ABI_MINOR == 0u);
static_assert(ET_RING_ATTN_OK == 0);
static_assert(ET_RING_ATTN_MAX_WORLD_SIZE == 4u);
static_assert(std::is_same_v<decltype(&et_ring_attn_multiprocess_forward_v1),
                             et_ring_attn_status_t (*)(
                                 const void *, const void *, const void *,
                                 const void *, void *, void *,
                                 et_ring_attn_stats_v1 *)>);
static_assert(std::is_same_v<decltype(&et_ring_attn_multiprocess_backward_v1),
                             et_ring_attn_status_t (*)(
                                 const void *, const void *, const void *,
                                 const void *, const void *, const void *,
                                 const void *, void *, void *, void *,
                                 et_ring_attn_stats_v1 *)>);

int main() {
  /* N=1, T=8, H=4, Hkv=2 (GQA), Dh=4, W=4, causal=1. */
  et_ring_attn_config_v1 cfg{};
  if (et_ring_attn_aot_set_config(&cfg, 1, 8, 4, 2, 4, 4, 1, 0.0, 10000) !=
      ET_RING_ATTN_OK)
    return 1;

  double eff_scale = 0.0;
  if (et_ring_attn_validate_config_v1(&cfg, &eff_scale) != ET_RING_ATTN_OK ||
      std::fabs(eff_scale - 0.5) > 1e-12)
    return 2;

  constexpr size_t q_elems = 1u * 8u * 4u * 4u;
  constexpr size_t kv_elems = 1u * 8u * 2u * 4u;
  constexpr size_t lse_elems = 1u * 8u * 4u;

  float q[q_elems], k[kv_elems], v[kv_elems], do_buf[q_elems];
  float o_ref[q_elems], o_mp[q_elems];
  double lse_ref[lse_elems], lse_mp[lse_elems];
  float dq_ref[q_elems], dk_ref[kv_elems], dv_ref[kv_elems];
  float dq_mp[q_elems], dk_mp[kv_elems], dv_mp[kv_elems];

  for (size_t i = 0; i < q_elems; ++i) {
    q[i] = static_cast<float>((static_cast<int>(i * 7u) % 15) - 7) * 0.125f;
    do_buf[i] =
        static_cast<float>((static_cast<int>(i * 11u) % 13) - 6) * 0.0625f;
  }
  for (size_t i = 0; i < kv_elems; ++i) {
    k[i] = static_cast<float>((static_cast<int>(i * 5u) % 15) - 7) * 0.125f;
    v[i] = static_cast<float>((static_cast<int>(i * 9u) % 17) - 8) * 0.0625f;
  }

  et_ring_attn_stats_v1 fwd_stats{}, bwd_stats{};
  if (et_ring_attn_aot_dense_forward(&cfg, q, k, v, o_ref, lse_ref) !=
      ET_RING_ATTN_OK)
    return 3;
  if (et_ring_attn_aot_mp_forward(&cfg, q, k, v, o_mp, lse_mp, &fwd_stats) !=
      ET_RING_ATTN_OK)
    return 4;
  if (et_ring_attn_aot_dense_backward(&cfg, q, k, v, o_ref, lse_ref, do_buf,
                                      dq_ref, dk_ref, dv_ref) !=
      ET_RING_ATTN_OK)
    return 5;
  if (et_ring_attn_aot_mp_backward(&cfg, q, k, v, o_mp, lse_mp, do_buf, dq_mp,
                                   dk_mp, dv_mp, &bwd_stats) !=
      ET_RING_ATTN_OK)
    return 6;

  if (fwd_stats.total_block_steps != 16u ||
      fwd_stats.skipped_causal_blocks != 6u ||
      fwd_stats.executed_blocks != 10u || fwd_stats.ring_rounds != 3u ||
      bwd_stats.total_block_steps != 16u ||
      bwd_stats.skipped_causal_blocks != 6u ||
      bwd_stats.executed_blocks != 10u || bwd_stats.ring_rounds != 4u)
    return 7;

  for (size_t i = 0; i < q_elems; ++i) {
    if (std::fabs(static_cast<double>(o_ref[i]) -
                  static_cast<double>(o_mp[i])) > 1e-6 ||
        std::fabs(static_cast<double>(dq_ref[i]) -
                  static_cast<double>(dq_mp[i])) > 1e-6)
      return 8;
  }
  for (size_t i = 0; i < lse_elems; ++i) {
    if (std::fabs(lse_ref[i] - lse_mp[i]) > 1e-6)
      return 9;
  }
  for (size_t i = 0; i < kv_elems; ++i) {
    if (std::fabs(static_cast<double>(dk_ref[i]) -
                  static_cast<double>(dk_mp[i])) > 1e-6 ||
        std::fabs(static_cast<double>(dv_ref[i]) -
                  static_cast<double>(dv_mp[i])) > 1e-6)
      return 10;
  }
  return 0;
}
