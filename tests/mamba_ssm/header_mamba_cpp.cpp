#include "eshkol_transformer/mamba_ssm_abi.h"
#include "mamba_ssm_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

static_assert(ET_MAMBA_SSM_ABI_MAJOR == 1u);
static_assert(ET_MAMBA_SSM_ABI_MINOR == 0u);
static_assert(ET_MAMBA_OK == 0);
static_assert(ET_MAMBA_MAX_BATCH == 4u);
static_assert(ET_MAMBA_MAX_SEQ_LEN == 64u);
static_assert(ET_MAMBA_MAX_D_INNER == 32u);
static_assert(ET_MAMBA_MAX_D_STATE == 16u);
static_assert(ET_MAMBA_MAX_K_CONV == 4u);

static_assert(std::is_same_v<decltype(&et_mamba_causal_conv1d_silu_forward_v1),
                             et_mamba_status_t (*)(const void *, const void *,
                                                   const void *, const void *,
                                                   void *)>);
static_assert(std::is_same_v<decltype(&et_mamba_selective_scan_forward_v1),
                             et_mamba_status_t (*)(
                                 const void *, const void *, const void *,
                                 const void *, const void *, const void *,
                                 const void *, const void *, void *)>);
static_assert(std::is_same_v<decltype(&et_mamba_cache_step_v1),
                             et_mamba_status_t (*)(
                                 void *, const void *, const void *,
                                 const void *, const void *, const void *,
                                 const void *, const void *, const void *,
                                 const void *, void *)>);

int main() {
  /* N=2, T=8, D=4, S=3, K=3 */
  et_mamba_config_v1 cfg{};
  if (et_mamba_aot_set_config(&cfg, 2, 8, 4, 3, 3) != ET_MAMBA_OK)
    return 1;
  if (et_mamba_validate_config_v1(&cfg) != ET_MAMBA_OK)
    return 2;

  constexpr size_t N = 2u, T = 8u, D = 4u, S = 3u, K = 3u;
  const size_t conv_bytes = et_mamba_cache_conv_state_bytes_v1(&cfg);
  const size_t ssm_bytes = et_mamba_cache_ssm_state_bytes_v1(&cfg);
  const size_t total_bytes = et_mamba_cache_total_state_bytes_v1(&cfg);
  if (conv_bytes != N * (K - 1u) * D * sizeof(float) ||
      ssm_bytes != N * D * S * sizeof(double) ||
      total_bytes != conv_bytes + ssm_bytes)
    return 3;

  float u[N * T * D], conv_w[D * K], conv_b[D];
  float x[N * T * D], dt_raw[N * T * D], a_log[D * S];
  float b_in[N * T * S], c_in[N * T * S], d_skip[D], z_in[N * T * D];
  float out_full[N * T * D];

  for (size_t i = 0; i < N * T * D; ++i) {
    u[i] = static_cast<float>(static_cast<int>((i * 7u) % 13u) - 6) * 0.125f;
    dt_raw[i] =
        static_cast<float>(static_cast<int>((i * 5u) % 11u) - 5) * 0.15f;
    z_in[i] = static_cast<float>(static_cast<int>((i * 11u) % 15u) - 7) * 0.1f;
  }
  for (size_t i = 0; i < D * K; ++i)
    conv_w[i] =
        static_cast<float>(static_cast<int>((i * 13u) % 9u) - 4) * 0.125f;
  for (size_t i = 0; i < D; ++i) {
    conv_b[i] = static_cast<float>(static_cast<int>(i) - 1) * 0.05f;
    d_skip[i] = 0.25f + 0.05f * static_cast<float>(i);
  }
  for (size_t i = 0; i < D * S; ++i)
    a_log[i] = -0.5f + 0.2f * static_cast<float>(i % S);
  for (size_t i = 0; i < N * T * S; ++i) {
    b_in[i] = static_cast<float>(static_cast<int>((i * 3u) % 11u) - 5) * 0.1f;
    c_in[i] = static_cast<float>(static_cast<int>((i * 9u) % 11u) - 5) * 0.1f;
  }

  if (et_mamba_aot_conv1d_silu_forward(&cfg, u, conv_w, conv_b, x) !=
      ET_MAMBA_OK)
    return 4;
  if (et_mamba_aot_selective_scan_forward(&cfg, x, dt_raw, a_log, b_in, c_in,
                                          d_skip, z_in, out_full) !=
      ET_MAMBA_OK)
    return 5;

  et_mamba_cache_v1 cache{};
  float conv_state[N * (K - 1u) * D];
  double ssm_state[N * D * S];
  if (et_mamba_aot_cache_init(&cache, &cfg, conv_state,
                              static_cast<int64_t>(sizeof(conv_state)),
                              ssm_state,
                              static_cast<int64_t>(sizeof(ssm_state))) !=
      ET_MAMBA_OK)
    return 6;

  float u_step[N * D], dt_step[N * D], z_step[N * D], out_step[N * D];
  float b_step[N * S], c_step[N * S];

  for (size_t t = 0; t < T; ++t) {
    for (size_t n = 0; n < N; ++n) {
      for (size_t d = 0; d < D; ++d) {
        u_step[n * D + d] = u[(n * T + t) * D + d];
        dt_step[n * D + d] = dt_raw[(n * T + t) * D + d];
        z_step[n * D + d] = z_in[(n * T + t) * D + d];
      }
      for (size_t s = 0; s < S; ++s) {
        b_step[n * S + s] = b_in[(n * T + t) * S + s];
        c_step[n * S + s] = c_in[(n * T + t) * S + s];
      }
    }
    if (et_mamba_aot_cache_step(&cache, u_step, conv_w, conv_b, dt_step, a_log,
                                b_step, c_step, d_skip, z_step, out_step) !=
        ET_MAMBA_OK)
      return 7;
    for (size_t n = 0; n < N; ++n) {
      for (size_t d = 0; d < D; ++d) {
        const double diff = std::fabs(
            static_cast<double>(out_step[n * D + d]) -
            static_cast<double>(out_full[(n * T + t) * D + d]));
        if (diff > 1e-6)
          return 8;
      }
    }
  }

  if (et_mamba_aot_cache_step_count(&cache) != static_cast<int64_t>(T))
    return 9;
  if (et_mamba_aot_cache_reset(&cache) != ET_MAMBA_OK ||
      et_mamba_aot_cache_step_count(&cache) != 0)
    return 10;

  return 0;
}
