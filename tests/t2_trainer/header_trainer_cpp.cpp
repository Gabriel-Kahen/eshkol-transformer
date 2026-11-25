#include "eshkol_transformer/t2_bpe_trainer_abi.h"
#include "t2_bpe_trainer_eshkol_bridge.h"

#include <cstring>
#include <type_traits>

static_assert(std::is_same_v<et_t2_bpe_status, int32_t>);
static_assert(std::is_standard_layout_v<et_t2_bpe_document>);
static_assert(std::is_standard_layout_v<et_t2_bpe_train_config>);
static_assert(std::is_standard_layout_v<et_t2_bpe_merge>);
static_assert(std::is_standard_layout_v<et_t2_bpe_info>);
static_assert(ET_T2_BPE_HEADER_BYTES == 44u);
static_assert(ET_T2_BPE_MERGE_RECORD_BYTES == 12u);
static_assert(ET_T2_BPE_SHA256_BYTES == 32u);

int main() {
  const char corpus[] = "alpha beta alpha beta alpha beta";
  const et_t2_bpe_document doc = {
      reinterpret_cast<const uint8_t *>(corpus),
      std::strlen(corpus),
  };
  const et_t2_bpe_train_config cfg = {
      280u,
      2u,
      ET_T2_BPE_SPECIAL_BOS | ET_T2_BPE_SPECIAL_EOS,
  };

  et_t2_bpe_model *model = nullptr;
  if (et_t2_bpe_train(&doc, 1u, &cfg, &model) != ET_T2_BPE_OK ||
      model == nullptr) {
    return 1;
  }

  int64_t vocab = 0;
  int64_t merges = 0;
  if (et_t2_bpe_aot_vocab_size(model, &vocab) != ET_T2_BPE_OK ||
      et_t2_bpe_aot_merge_count(model, &merges) != ET_T2_BPE_OK ||
      vocab != 256 + merges + 2 || merges <= 0) {
    et_t2_bpe_destroy(model);
    return 2;
  }

  uint32_t toks[64] = {0};
  int64_t tok_n = 0;
  uint8_t decoded[64] = {0};
  int64_t dec_n = 0;
  if (et_t2_bpe_aot_encode_cstr(model, "alpha beta", toks, 64, &tok_n) !=
          ET_T2_BPE_OK ||
      tok_n >= 10 ||
      et_t2_bpe_aot_decode_bytes(model, toks, tok_n, decoded, 64, &dec_n) !=
          ET_T2_BPE_OK ||
      dec_n != 10 || std::memcmp(decoded, "alpha beta", 10u) != 0) {
    et_t2_bpe_destroy(model);
    return 3;
  }

  et_t2_bpe_aot_destroy(model);
  return 0;
}
