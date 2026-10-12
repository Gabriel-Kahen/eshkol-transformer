/* Test-only snapshot of the original idle G3 generator. */
#if defined(ET_G3R_G3T_SAVE_LEDGER_PRIVATE) && \
    defined(ET_G3R_SAVE_IDLE_LEDGER_TESTING)
#include "../../src/eshkol_transformer/g3t_transport.h"
#include <stdint.h>
#include <string.h>

int64_t et_g3r_test_save_idle_ledger_copy_v1(void *generator,
                                               void *bytevector_header,
                                               int64_t byte_count) {
  et_g3r_g3t_idle_snapshot_words_v1 words;
  int64_t encoded = -1;
  if (!generator || !bytevector_header ||
      byte_count != (int64_t)sizeof(words)) return -1;
  memcpy(&encoded, bytevector_header, sizeof(encoded));
  if (encoded != byte_count) return -1;
  void *snapshot = et_g3r_private_g3t_idle_snapshot_create_v1(generator);
  if (!snapshot) return -1;
  int64_t copied = et_g3r_private_g3t_idle_snapshot_copy_v1(
      snapshot, &words, (int64_t)sizeof(words));
  int64_t released = et_g3r_private_g3t_idle_snapshot_release_v1(snapshot);
  if (copied != 0 || released != 0) return -1;
  memcpy((unsigned char *)bytevector_header + sizeof(encoded),
         &words, sizeof(words));
  return 0;
}
#else
typedef int et_g3r_test_save_idle_ledger_feature_off;
#endif
