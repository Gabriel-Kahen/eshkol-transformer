#include "eshkol_transformer/data4_pipeline_abi.h"
#include "data4_pipeline_eshkol_bridge.h"

#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_same_v<et_data4_status, int32_t>);
static_assert(std::is_same_v<et_data4_disposition, uint32_t>);
static_assert(std::is_standard_layout_v<et_data4_document>);
static_assert(std::is_trivially_copyable_v<et_data4_document>);
static_assert(std::is_standard_layout_v<et_data4_pipeline_config>);
static_assert(std::is_trivially_copyable_v<et_data4_pipeline_config>);
static_assert(std::is_standard_layout_v<et_data4_doc_result>);
static_assert(std::is_trivially_copyable_v<et_data4_doc_result>);
static_assert(std::is_standard_layout_v<et_data4_pipeline_summary>);
static_assert(std::is_trivially_copyable_v<et_data4_pipeline_summary>);

static_assert(std::is_same_v<decltype(&et_data4_config_default),
                             et_data4_status (*)(et_data4_pipeline_config *)>);
static_assert(std::is_same_v<decltype(&et_data4_disposition_name),
                             const char *(*)(et_data4_disposition)>);
static_assert(std::is_same_v<decltype(&et_data4_validate_utf8),
                             et_data4_status (*)(const uint8_t *, size_t,
                                                 uint64_t *, uint64_t *,
                                                 uint64_t *)>);
static_assert(std::is_same_v<decltype(&et_data4_sha256),
                             et_data4_status (*)(const uint8_t *, size_t,
                                                 uint8_t *)>);
static_assert(std::is_same_v<
              decltype(&et_data4_compute_minhash),
              et_data4_status (*)(const uint8_t *, size_t, uint32_t, uint32_t,
                                  uint64_t, uint64_t *, size_t, uint64_t *)>);
static_assert(std::is_same_v<
              decltype(&et_data4_exact_jaccard),
              et_data4_status (*)(const uint8_t *, size_t, const uint8_t *,
                                  size_t, uint32_t, uint64_t *, uint64_t *)>);
static_assert(std::is_same_v<
              decltype(&et_data4_pipeline_run),
              et_data4_status (*)(const et_data4_document *, size_t,
                                  const et_data4_pipeline_config *,
                                  et_data4_doc_result *, size_t,
                                  et_data4_pipeline_summary *)>);
static_assert(std::is_same_v<
              decltype(&et_data4_report_format),
              et_data4_status (*)(const et_data4_pipeline_config *,
                                  const et_data4_pipeline_summary *,
                                  const et_data4_doc_result *, size_t, char *,
                                  size_t, size_t *, uint8_t *)>);
static_assert(std::is_same_v<
              decltype(&et_data4_report_verify),
              et_data4_status (*)(const char *, size_t,
                                  et_data4_pipeline_config *,
                                  et_data4_pipeline_summary *,
                                  et_data4_doc_result *, size_t)>);

int main() {
  et_data4_pipeline_config cfg{};
  if (et_data4_config_default(&cfg) != ET_DATA4_OK) return 1;

  const char *text = "Deterministic corpus pipeline check for C++17 linkage.";
  const et_data4_document docs[2] = {
      {"doc-a", reinterpret_cast<const uint8_t *>(text), std::strlen(text)},
      {"doc-b", reinterpret_cast<const uint8_t *>(text), std::strlen(text)},
  };
  et_data4_doc_result results[2]{};
  et_data4_pipeline_summary summary{};
  if (et_data4_pipeline_run(docs, 2u, &cfg, results, 2u, &summary) !=
      ET_DATA4_OK)
    return 2;
  if (summary.kept_documents != 1u || summary.duplicate_exact_documents != 1u ||
      results[0].disposition != ET_DATA4_DISP_KEPT ||
      results[1].disposition != ET_DATA4_DISP_DUPLICATE_EXACT)
    return 3;

  size_t report_size = 0u;
  uint8_t digest[ET_DATA4_SHA256_BYTES]{};
  if (et_data4_report_format(&cfg, &summary, results, 2u, nullptr, 0u,
                             &report_size, digest) != ET_DATA4_OK ||
      report_size == 0u)
    return 4;

  char *buf = new char[report_size + 1u];
  size_t written = 0u;
  if (et_data4_report_format(&cfg, &summary, results, 2u, buf,
                             report_size + 1u, &written, digest) !=
          ET_DATA4_OK ||
      written != report_size ||
      et_data4_report_verify(buf, written, nullptr, nullptr, nullptr, 0u) !=
          ET_DATA4_OK) {
    delete[] buf;
    return 5;
  }
  delete[] buf;

  void *session = et_data4_aot_session_create();
  if (session == nullptr) return 6;
  if (et_data4_aot_add_doc(session, "s0", 2, text,
                           static_cast<int64_t>(std::strlen(text))) !=
          ET_DATA4_OK ||
      et_data4_aot_run(session) != ET_DATA4_OK ||
      et_data4_aot_get_stat(session, ET_DATA4_AOT_STAT_KEPT_DOCS) != 1) {
    et_data4_aot_session_destroy(session);
    return 7;
  }
  et_data4_aot_session_destroy(session);
  return 0;
}
