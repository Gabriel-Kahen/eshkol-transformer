/* Test-only interposition around the real bridge, with its ownership intact. */
#include "data4_pipeline_eshkol_bridge.h"

void *et_data4_quality_test_real_create(void);
void et_data4_quality_test_real_destroy(void *session);
int32_t et_data4_quality_test_real_copy_report(const void *session, void *out,
                                              int64_t capacity);
int32_t et_data4_quality_test_real_copy_digest(const void *session, void *out,
                                              int64_t capacity);

static int32_t fault;
static int32_t live;

void et_data4_quality_test_fault(int32_t mode) { fault = mode; }
int32_t et_data4_quality_test_live(void) { return live; }

void *et_data4_aot_session_create(void) {
  void *session = et_data4_quality_test_real_create();
  if (session != NULL) ++live;
  return session;
}

void et_data4_aot_session_destroy(void *session) {
  if (session != NULL) --live;
  et_data4_quality_test_real_destroy(session);
}

int32_t et_data4_aot_copy_report(const void *session, void *out,
                                 int64_t capacity) {
  const int32_t status = et_data4_quality_test_real_copy_report(session, out, capacity);
  if (status == ET_DATA4_OK && fault == 1) ((uint8_t *)out)[0] ^= 1u;
  return status;
}

int32_t et_data4_aot_copy_report_sha256(const void *session, void *out,
                                        int64_t capacity) {
  const int32_t status = et_data4_quality_test_real_copy_digest(session, out, capacity);
  if (status == ET_DATA4_OK && fault == 2) ((uint8_t *)out)[0] ^= 1u;
  return status;
}
