#include "eshkol_transformer/flash_attn2_abi.h"
#include "flash_attn2_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

static_assert(ET_FA2_ABI_MAJOR == 1u);
static_assert(ET_FA2_ABI_MINOR == 0u);
static_assert(ET_FA2_OK == 0);
static_assert(ET_FA2_MAX_BATCH == 4u);
static_assert(ET_FA2_MAX_SEQ_LEN == 256u);
static_assert(ET_FA2_MAX_HEADS == 16u);
static_assert(ET_FA2_MAX_HEAD_DIM == 64u);
static_assert(std::is_standard_layout_v<et_fa2_config_v1>);
static_assert(std::is_standard_layout_v<et_fa2_stats_v1>);
static_assert(std::is_same_v<decltype(&et_fa2_tile_scratch_bytes_v1),
                             size_t (*)(size_t, size_t)>);
static_assert(std::is_same_v<decltype(&et_fa2_validate_config_v1),
                             et_fa2_status_t (*)(const void *, double *)>);
static_assert(
    std::is_same_v<decltype(&et_fa2_forward_v1),
                   et_fa2_status_t (*)(const void *, const void *, const void *,
                                       const void *, void *, void *,
                                       et_fa2_stats_v1 *)>);
static_assert(
    std::is_same_v<decltype(&et_fa2_backward_v1),
                   et_fa2_status_t (*)(const void *, const void *, const void *,
                                       const void *, const void *, const void *,
                                       const void *, void *, void *, void *,
                                       et_fa2_stats_v1 *)>);

int main() {
  /* GQA + Causal + Sliding Window + Gemma-2 Softcap across two tile configs. */
  et_fa2_config_v1 cfg8{}, cfg16{};
  if (et_fa2_aot_set_config(&cfg8, 1, 4, 2, 24, 24, 8, 8, 8, 1, 10, 0.0, 20.0) !=
      ET_FA2_OK)
    return 1;
  if (et_fa2_aot_set_config(&cfg16, 1, 4, 2, 24, 24, 8, 16, 16, 1, 10, 0.0,
                            20.0) != ET_FA2_OK)
    return 2;

  double eff_scale = 0.0;
  if (et_fa2_validate_config_v1(&cfg8, &eff_scale) != ET_FA2_OK ||
      std::fabs(eff_scale - (1.0 / std::sqrt(8.0))) > 1e-12)
    return 3;

  if (et_fa2_tile_scratch_bytes_v1(8u, 8u) != 1024u ||
      et_fa2_tile_scratch_bytes_v1(16u, 16u) != 4096u ||
      et_fa2_tile_scratch_bytes_v1(32u, 16u) != 8192u ||
      et_fa2_tile_scratch_bytes_v1(12u, 16u) != 0u)
    return 4;

  constexpr size_t q_elems = 1u * 4u * 24u * 8u;
  constexpr size_t kv_elems = 1u * 2u * 24u * 8u;
  constexpr size_t lse_elems = 1u * 4u * 24u;

  float q[q_elems], k[kv_elems], v[kv_elems], do_buf[q_elems];
  float o8[q_elems], o16[q_elems];
  double lse8[lse_elems], lse16[lse_elems];
  float dq8[q_elems], dk8[kv_elems], dv8[kv_elems];
  float dq16[q_elems], dk16[kv_elems], dv16[kv_elems];

  for (size_t i = 0; i < q_elems; ++i) {
    q[i] = static_cast<float>((static_cast<int>(i * 7u) % 15) - 7) * 0.125f;
    do_buf[i] =
        static_cast<float>((static_cast<int>(i * 11u) % 13) - 6) * 0.0625f;
  }
  for (size_t i = 0; i < kv_elems; ++i) {
    k[i] = static_cast<float>((static_cast<int>(i * 5u) % 15) - 7) * 0.125f;
    v[i] = static_cast<float>((static_cast<int>(i * 9u) % 17) - 8) * 0.0625f;
  }

  et_fa2_stats_v1 f8{}, b8{}, f16{}, b16{};
  if (et_fa2_aot_forward(&cfg8, q, k, v, o8, lse8, &f8) != ET_FA2_OK ||
      et_fa2_aot_backward(&cfg8, q, k, v, o8, lse8, do_buf, dq8, dk8, dv8,
                          &b8) != ET_FA2_OK)
    return 5;
  if (et_fa2_aot_forward(&cfg16, q, k, v, o16, lse16, &f16) != ET_FA2_OK ||
      et_fa2_aot_backward(&cfg16, q, k, v, o16, lse16, do_buf, dq16, dk16, dv16,
                          &b16) != ET_FA2_OK)
    return 6;

  if (f8.skipped_tiles == 0u || f8.skipped_tiles != b8.skipped_tiles ||
      f8.evaluated_tiles + f8.skipped_tiles != f8.total_tiles ||
      f8.tile_scratch_bytes != 1024u || f16.tile_scratch_bytes != 4096u)
    return 7;

  for (size_t i = 0; i < q_elems; ++i) {
    if (std::fabs(static_cast<double>(o8[i]) - static_cast<double>(o16[i])) >
            1e-6 ||
        std::fabs(static_cast<double>(dq8[i]) - static_cast<double>(dq16[i])) >
            1e-6)
      return 8;
  }
  for (size_t i = 0; i < lse_elems; ++i) {
    if (std::fabs(lse8[i] - lse16[i]) > 1e-12)
      return 9;
  }
  for (size_t i = 0; i < kv_elems; ++i) {
    if (std::fabs(static_cast<double>(dk8[i]) - static_cast<double>(dk16[i])) >
            1e-6 ||
        std::fabs(static_cast<double>(dv8[i]) - static_cast<double>(dv16[i])) >
            1e-6)
      return 10;
  }
  return 0;
}
