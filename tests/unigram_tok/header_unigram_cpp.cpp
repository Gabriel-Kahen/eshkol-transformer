#include "eshkol_transformer/unigram_tok_abi.h"
#include "unigram_tok_eshkol_bridge.h"

#include <cmath>
#include <cstring>
#include <type_traits>

static_assert(std::is_same_v<et_unigram_status, int32_t>);
static_assert(std::is_standard_layout_v<et_unigram_document_v1>);
static_assert(std::is_standard_layout_v<et_unigram_piece_spec_v1>);
static_assert(std::is_standard_layout_v<et_unigram_train_config_v1>);
static_assert(std::is_standard_layout_v<et_unigram_train_stats_v1>);
static_assert(std::is_standard_layout_v<et_unigram_info_v1>);
static_assert(ET_UNIGRAM_HEADER_BYTES == 48u);
static_assert(ET_UNIGRAM_PIECE_OVERHEAD_BYTES == 10u);
static_assert(ET_UNIGRAM_SHA256_BYTES == 32u);

int main() {
  const char corpus[] =
      "alpha beta gamma alpha beta gamma alpha beta gamma alpha beta";
  const et_unigram_document_v1 doc = {
      reinterpret_cast<const uint8_t *>(corpus),
      std::strlen(corpus),
  };
  const et_unigram_train_config_v1 cfg = {
      static_cast<uint32_t>(sizeof(et_unigram_train_config_v1)),
      272u,
      512u,
      12u,
      2u,
      2u,
      4u,
      ET_UNIGRAM_MSTEP_MLE,
      ET_UNIGRAM_SPECIAL_BOS | ET_UNIGRAM_SPECIAL_EOS,
      0u,
      0.75,
      1e-3,
  };

  et_unigram_model *model = nullptr;
  et_unigram_train_stats_v1 stats = {};
  if (et_unigram_train_v1(&doc, 1u, &cfg, &model, &stats) != ET_UNIGRAM_OK ||
      model == nullptr) {
    return 1;
  }

  int64_t vocab = 0;
  int64_t active = 0;
  int64_t learned = 0;
  if (et_unigram_aot_vocab_size(model, &vocab) != ET_UNIGRAM_OK ||
      et_unigram_aot_active_count(model, &active) != ET_UNIGRAM_OK ||
      et_unigram_aot_learned_count(model, &learned) != ET_UNIGRAM_OK ||
      vocab != active + 2 || active != 256 + learned || learned <= 0) {
    et_unigram_destroy_v1(model);
    return 2;
  }

  uint32_t toks[64] = {0};
  int64_t tok_n = 0;
  double vscore = 0.0;
  uint8_t decoded[64] = {0};
  int64_t dec_n = 0;
  double log_z = 0.0;
  if (et_unigram_aot_encode_cstr(model, "alpha beta", toks, 64, &tok_n,
                                 &vscore) != ET_UNIGRAM_OK ||
      !std::isfinite(vscore) || tok_n >= 10 ||
      et_unigram_aot_decode_bytes(model, toks, tok_n, decoded, 64, &dec_n) !=
          ET_UNIGRAM_OK ||
      dec_n != 10 || std::memcmp(decoded, "alpha beta", 10u) != 0 ||
      et_unigram_aot_forward_backward_cstr(model, "alpha beta", &log_z, nullptr,
                                           0) != ET_UNIGRAM_OK ||
      !std::isfinite(log_z) || log_z < vscore - 1e-12) {
    et_unigram_destroy_v1(model);
    return 3;
  }

  et_unigram_aot_destroy(model);
  return 0;
}
