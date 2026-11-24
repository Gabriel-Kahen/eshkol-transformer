#include "eshkol_transformer/mod5_mla_abi.h"
#include "mod5_mla_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

static_assert(ET_MOD5_MLA_ABI_MAJOR == 1u);
static_assert(ET_MOD5_MLA_ABI_MINOR == 0u);
static_assert(ET_MOD5_MLA_MIN_BATCH == 1u && ET_MOD5_MLA_MAX_BATCH == 2u);
static_assert(ET_MOD5_MLA_MIN_TOKENS == 1u && ET_MOD5_MLA_MAX_TOKENS == 32u);
static_assert(ET_MOD5_MLA_MIN_MODEL_DIM == 4u &&
              ET_MOD5_MLA_MAX_MODEL_DIM == 32u);
static_assert(ET_MOD5_MLA_MIN_HEADS == 1u && ET_MOD5_MLA_MAX_HEADS == 4u);
static_assert(ET_MOD5_MLA_MIN_HEAD_DIM == 2u &&
              ET_MOD5_MLA_MAX_HEAD_DIM == 16u);
static_assert(ET_MOD5_MLA_MIN_LATENT_DIM == 1u &&
              ET_MOD5_MLA_MAX_LATENT_DIM == 16u);
static_assert(std::is_same_v<decltype(&et_mod5_mla_kernel_provider_v1),
                             const et_kernel_provider_v1 *(*)()>);

int main() {
  const auto *provider = et_mod5_mla_kernel_provider_v1();
  if (provider == nullptr || provider->abi_major != ET_KERNEL_ABI_MAJOR)
    return 1;

  et_kernel_runtime *runtime = nullptr;
  if (et_mod5_mla_aot_open(&runtime) != 0 || runtime == nullptr)
    return 2;

  /* N=1, T=4, D=4, H=2, Dh=4, dc=4 -> HD=8, 4x KV compression. */
  et_mod5_mla_dims dims{1u, 4u, 4u, 2u, 4u, 4u};
  if (et_mod5_mla_latent_kv_bytes(1u, 4u, 4u) != 64u ||
      et_mod5_mla_full_kv_bytes(1u, 4u, 2u, 4u) != 256u ||
      std::fabs(et_mod5_mla_kv_compression_ratio(2u, 4u, 4u) - 4.0) > 1e-12) {
    et_mod5_mla_aot_close(runtime);
    return 3;
  }

  float x[16], w_q[32], w_dkv[16], w_uk[32], w_uv[32], w_o[32];
  float o_direct[16] = {0.0f}, o_aot[16] = {0.0f}, o_cached[16] = {0.0f};
  float dy[16], dx[16], dw_q[32], dw_dkv[16], dw_uk[32], dw_uv[32], dw_o[32];
  float c_kv[16] = {0.0f};

  for (int i = 0; i < 16; ++i) {
    x[i] = static_cast<float>((i % 7) - 3) * 0.125f;
    w_dkv[i] = static_cast<float>(((i * 3) % 7) - 3) * 0.125f;
    dy[i] = static_cast<float>(((i * 5) % 9) - 4) * 0.0625f;
  }
  for (int i = 0; i < 32; ++i) {
    w_q[i] = static_cast<float>(((i * 5) % 7) - 3) * 0.125f;
    w_uk[i] = static_cast<float>(((i * 7) % 9) - 4) * 0.0625f;
    w_uv[i] = static_cast<float>(((i * 11) % 9) - 4) * 0.0625f;
    w_o[i] = static_cast<float>(((i * 13) % 7) - 3) * 0.125f;
  }

  et_kernel_error error{};
  if (et_mod5_mla_forward_f32(&dims, x, w_q, w_dkv, w_uk, w_uv, w_o, o_direct,
                              &error) != 0) {
    et_mod5_mla_aot_close(runtime);
    return 4;
  }
  if (et_mod5_mla_aot_forward(runtime, &dims, x, w_q, w_dkv, w_uk, w_uv, w_o,
                              o_aot) != 0) {
    et_mod5_mla_aot_close(runtime);
    return 5;
  }
  if (et_mod5_mla_aot_backward(runtime, &dims, x, w_q, w_dkv, w_uk, w_uv, w_o,
                               dy, dx, dw_q, dw_dkv, dw_uk, dw_uv, dw_o) != 0) {
    et_mod5_mla_aot_close(runtime);
    return 6;
  }

  et_mod5_mla_kv_cache cache{};
  if (et_mod5_mla_cache_init(&cache, &dims, c_kv, sizeof(c_kv), &error) != 0 ||
      et_mod5_mla_cache_prefill_f32(&cache, 2u, x, w_q, w_dkv, w_uk, w_uv, w_o,
                                    o_cached, &error) != 0 ||
      et_mod5_mla_cache_step_f32(&cache, x + 8, w_q, w_dkv, w_uk, w_uv, w_o,
                                 o_cached + 8, &error) != 0 ||
      et_mod5_mla_cache_step_f32(&cache, x + 12, w_q, w_dkv, w_uk, w_uv, w_o,
                                 o_cached + 12, &error) != 0) {
    et_mod5_mla_aot_close(runtime);
    return 7;
  }

  for (int i = 0; i < 16; ++i) {
    if (o_direct[i] != o_aot[i] ||
        std::fabs(o_direct[i] - o_cached[i]) > 1e-6f) {
      et_mod5_mla_aot_close(runtime);
      return 8;
    }
  }
  if (et_mod5_mla_cache_reset(&cache, &error) != 0 || cache.seq_len != 0u) {
    et_mod5_mla_aot_close(runtime);
    return 9;
  }

  et_mod5_mla_aot_close(runtime);
  return 0;
}
