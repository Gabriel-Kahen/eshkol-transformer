#include "eshkol_transformer/qlora_nf4_abi.h"
#include "qlora_nf4_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

static_assert(std::is_standard_layout_v<et_qlora_quant_spec>);
static_assert(std::is_standard_layout_v<et_qlora_quant_footprint>);
static_assert(std::is_standard_layout_v<et_qlora_linear_config>);
static_assert(std::is_standard_layout_v<et_qlora_adamw_config>);
static_assert(std::is_standard_layout_v<et_qlora_mlp_config>);
static_assert(std::is_standard_layout_v<et_qlora_checkpoint_meta>);
static_assert(std::is_same_v<et_qlora_status, int32_t>);
static_assert(std::is_same_v<et_qlora_quant_mode, uint32_t>);

int main() {
  if (ET_NF4_CODEBOOK[0] != -1.0f || ET_NF4_CODEBOOK[7] != 0.0f ||
      ET_NF4_CODEBOOK[15] != 1.0f)
    return 1;

  et_qlora_quant_spec spec{};
  spec.rows_out = 4u;
  spec.cols_in = 64u;
  spec.block_size = ET_QLORA_BLOCK_SIZE_64;
  spec.dq_block_size = ET_QLORA_DQ_BLOCK_SIZE_16;
  spec.mode = ET_QLORA_QUANT_NF4_DQ;

  et_qlora_quant_footprint fp{};
  if (et_qlora_packed_footprint(&spec, &fp) != ET_QLORA_OK)
    return 2;
  if (fp.f32_bytes != 1024u || fp.nibble_bytes != 128u ||
      fp.packed_bytes != 140u)
    return 3;

  std::vector<float> w(256u);
  for (size_t i = 0; i < w.size(); ++i)
    w[i] = ET_NF4_CODEBOOK[i % 16u] * 0.5f;

  std::vector<uint8_t> packed(static_cast<size_t>(fp.packed_bytes), 0u);
  if (et_qlora_nf4_quantize(&spec, w.data(), packed.data(), packed.size()) !=
      ET_QLORA_OK)
    return 4;

  et_qlora_linear_config cfg{};
  cfg.quant = spec;
  cfg.m = 2u;
  cfg.r = 2u;
  cfg.alpha = 4.0f;

  std::vector<float> x(128u, 0.25f);
  std::vector<float> a(128u, 0.0f);
  std::vector<float> b(8u, 0.0f);
  std::vector<float> y_base(8u, 0.0f);
  std::vector<float> y_lora(8u, 0.0f);

  if (et_qlora_init_lora(64u, 4u, 2u, 123u, a.data(), b.data()) != ET_QLORA_OK)
    return 5;
  if (et_qlora_linear_forward(&cfg, x.data(), packed.data(), packed.size(),
                              nullptr, nullptr, nullptr,
                              y_base.data()) != ET_QLORA_OK)
    return 6;
  if (et_qlora_linear_forward(&cfg, x.data(), packed.data(), packed.size(),
                              nullptr, a.data(), b.data(),
                              y_lora.data()) != ET_QLORA_OK)
    return 7;
  if (std::memcmp(y_base.data(), y_lora.data(), y_base.size() * sizeof(float)) !=
      0)
    return 8;

  et_qlora_aot_ctx *ctx = nullptr;
  if (et_qlora_aot_create(2, 64, 64, 32, 4, 64, 16, ET_QLORA_QUANT_NF4_DQ, 8.0,
                          &ctx) != ET_QLORA_OK ||
      ctx == nullptr)
    return 9;
  et_qlora_aot_destroy(ctx);
  return 0;
}
