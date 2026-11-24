#include "eshkol_transformer/kv_cache_quant_abi.h"
#include "kv_cache_quant_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

static_assert(std::is_standard_layout_v<et_kvq_config>);
static_assert(std::is_trivially_copyable_v<et_kvq_config>);
static_assert(std::is_standard_layout_v<et_kvq_memory_stats>);
static_assert(std::is_trivially_copyable_v<et_kvq_memory_stats>);
static_assert(std::is_same_v<et_kvq_format, uint32_t>);
static_assert(std::is_same_v<et_kvq_status, int32_t>);
static_assert(sizeof(et_kvq_config) == 24u);
static_assert(sizeof(et_kvq_memory_stats) == 48u);

static void require(bool cond, const char *msg) {
  if (!cond) {
    std::fprintf(stderr, "header_kvq_cpp failed: %s\n", msg);
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  const et_kvq_config cfg{ET_KVQ_FMT_FP8_E4M3FN, 1u, 2u, 16u, 8u, 0u};
  require(et_kvq_validate_config(&cfg) == ET_KVQ_OK, "validate_config");

  size_t req_bytes = 0u;
  require(et_kvq_storage_bytes_required(&cfg, &req_bytes) == ET_KVQ_OK,
          "storage_bytes_required");
  require(req_bytes == 1u * 2u * 8u * 2u * (16u + 4u), "exact fp8 storage size");

  et_kvq_cache *cache = nullptr;
  require(et_kvq_cache_create(&cfg, nullptr, 0u, &cache) == ET_KVQ_OK,
          "cache_create");

  std::vector<float> k(32u, 0.25f);
  std::vector<float> v(32u, -0.5f);
  k[0] = 1.0f;
  v[1] = 2.0f;
  require(et_kvq_append_token_v1(cache, 0u, k.data(), v.data()) == ET_KVQ_OK,
          "append_token");

  std::vector<float> dk(32u, 0.0f);
  std::vector<float> dv(32u, 0.0f);
  require(et_kvq_dequant_slice_v1(cache, 0u, 0u, dk.data(), dv.data()) ==
              ET_KVQ_OK,
          "dequant_slice");

  std::vector<float> q(64u, 0.125f);
  std::vector<float> out(64u, 0.0f);
  require(et_kvq_decode_gqa_v1(cache, 0u, 4u, 0.25f, 0u, q.data(),
                               out.data()) == ET_KVQ_OK,
          "decode_gqa");
  require(std::fabs(out[1] - dv[1]) < 1e-6f, "single-token decode equals V");

  et_kvq_memory_stats stats{};
  require(et_kvq_cache_memory_stats(cache, &stats) == ET_KVQ_OK,
          "memory_stats");
  require(stats.fp32_baseline_bytes == 256u, "fp32 baseline active bytes");
  require(stats.quantized_bytes == 80u, "fp8 quantized active bytes");

  et_kvq_cache_destroy(cache);

  /* Also verify C++ linkage against the Eshkol AOT bridge archive. */
  et_kvq_aot_ctx *ctx = nullptr;
  require(et_kvq_aot_create(ET_KVQ_FMT_INT8_SYM, 1, 2, 16, 4, &ctx) ==
              ET_KVQ_OK,
          "aot_create");
  require(et_kvq_aot_set_k(ctx, 0, 0, 1.5) == ET_KVQ_OK, "aot_set_k");
  require(et_kvq_aot_set_v(ctx, 0, 0, -0.75) == ET_KVQ_OK, "aot_set_v");
  require(et_kvq_aot_append_token(ctx, 0) == ET_KVQ_OK, "aot_append");
  require(et_kvq_aot_layer_seq_len(ctx, 0) == 1, "aot_seq_len");
  et_kvq_aot_destroy(ctx);
  return 0;
}
