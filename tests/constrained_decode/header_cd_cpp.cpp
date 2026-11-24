#include "eshkol_transformer/constrained_decode_abi.h"
#include "constrained_decode_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_cd_json_field_spec>);
static_assert(std::is_standard_layout_v<et_cd_dfa_info>);
static_assert(std::is_standard_layout_v<et_cd_vocab_index_info>);
static_assert(std::is_same_v<et_cd_status, int32_t>);

static void require(bool cond, const char *msg) {
  if (!cond) {
    std::fprintf(stderr, "header_cd_cpp FAIL: %s\n", msg);
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  const char *answers[3] = {"yes", "no", "maybe"};
  const et_cd_json_field_spec fields[3] = {
      {"answer", ET_CD_JSON_ENUM, answers, 3u},
      {"score", ET_CD_JSON_INTEGER, nullptr, 0u},
      {"ok", ET_CD_JSON_BOOLEAN, nullptr, 0u},
  };

  et_cd_dfa *dfa = nullptr;
  require(et_cd_dfa_compile_flat_json_schema(fields, 3u, &dfa) == ET_CD_OK,
          "compile_flat_json_schema");
  require(dfa != nullptr, "dfa non-null");

  et_cd_dfa_info info{};
  require(et_cd_dfa_info_get(dfa, &info) == ET_CD_OK, "dfa_info_get");
  require(info.num_states > 0u && info.live_state_count == info.num_states,
          "all schema states live");

  const char *valid_json = "{\"answer\":\"yes\",\"score\":-42,\"ok\":true}";
  uint32_t accepted = 0u;
  require(et_cd_dfa_accepts_bytes(dfa,
                                  reinterpret_cast<const uint8_t *>(valid_json),
                                  std::strlen(valid_json), &accepted) == ET_CD_OK,
          "accepts_bytes call");
  require(accepted == 1u, "valid JSON accepted");

  const char *bad_json = "{\"answer\":\"yes\",\"score\":01,\"ok\":true}";
  require(et_cd_dfa_accepts_bytes(dfa,
                                  reinterpret_cast<const uint8_t *>(bad_json),
                                  std::strlen(bad_json), &accepted) == ET_CD_OK,
          "reject bad JSON call");
  require(accepted == 0u, "leading-zero integer rejected");

  et_cd_dfa_destroy(dfa);

  et_cd_aot_ctx *ctx = nullptr;
  require(et_cd_aot_create_json_schema(&ctx) == ET_CD_OK,
          "aot_create_json_schema");
  require(ctx != nullptr, "ctx non-null");
  require(et_cd_aot_verify_artifact_roundtrip(ctx) == ET_CD_OK,
          "aot artifact roundtrip");
  et_cd_aot_destroy(ctx);
  return 0;
}
