#include "eval5_eshkol_bridge.h"

#include <cstring>
#include <type_traits>

static_assert(std::is_same_v<et_eval5_status, int32_t>);
static_assert(ET_EVAL5_AOT_AUDIT_SUMMARY_LEN == 6);
static_assert(ET_EVAL5_AOT_LM_METRICS_LEN == 4);
static_assert(ET_EVAL5_AOT_MANIFEST_COUNTS_LEN == 12);
static_assert(ET_EVAL5_AOT_MANIFEST_METRICS_LEN == 6);

int main() {
  et_eval5_aot_corpus *train = nullptr;
  et_eval5_aot_corpus *eval = nullptr;
  if (et_eval5_aot_corpus_create(&train) != ET_EVAL5_OK || train == nullptr) {
    return 1;
  }
  if (et_eval5_aot_corpus_create(&eval) != ET_EVAL5_OK || eval == nullptr) {
    et_eval5_aot_corpus_destroy(train);
    return 2;
  }

  const uint32_t t_toks[4] = {1u, 2u, 3u, 4u};
  const uint32_t e_toks[4] = {5u, 6u, 7u, 8u};
  const uint8_t t_bytes[4] = {'a', 'b', 'c', 'd'};
  const uint8_t e_bytes[4] = {'w', 'x', 'y', 'z'};
  int64_t indices[1] = {-1};
  int64_t summary[ET_EVAL5_AOT_AUDIT_SUMMARY_LEN] = {0};
  double rate = -1.0;

  int rc = 0;
  if (et_eval5_aot_corpus_add_doc(train, t_bytes, 4, t_toks, 4) != ET_EVAL5_OK ||
      et_eval5_aot_corpus_add_doc(eval, e_bytes, 4, e_toks, 4) != ET_EVAL5_OK) {
    rc = 3;
  } else if (et_eval5_aot_audit_contamination(
                 train, eval, ET_EVAL5_NGRAM_TOKENS, 2, 1, 0.25, 0.0, indices,
                 1, summary, &rate) != ET_EVAL5_OK ||
             rate != 0.0 || summary[ET_EVAL5_AOT_AUDIT_CONTAMINATED_DOCS] != 0) {
    rc = 4;
  } else {
    float logits[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t targets[1] = {0u};
    double lm[ET_EVAL5_AOT_LM_METRICS_LEN] = {0};
    if (et_eval5_aot_score_lm(logits, 4, targets, 1, 4, 2, lm) != ET_EVAL5_OK ||
        lm[ET_EVAL5_AOT_LM_PERPLEXITY] != 4.0) {
      rc = 5;
    }
  }

  et_eval5_aot_corpus_destroy(eval);
  et_eval5_aot_corpus_destroy(train);
  return rc;
}
