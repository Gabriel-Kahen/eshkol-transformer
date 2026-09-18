#ifndef ESHKOL_TRANSFORMER_C2_CHECKPOINT_LOAD_BRIDGE_H
#define ESHKOL_TRANSFORMER_C2_CHECKPOINT_LOAD_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_C2_CHECKPOINT_LOAD_RESULT_BYTES INT64_C(192)
#define ET_C2_CHECKPOINT_LOAD_RESULT_MAGIC UINT64_C(0x43324c4f41443130)

/*
 * Private K2-independent staging bridge. Every pointer addresses a pinned
 * Eshkol bytevector header (signed i64 length followed by exact bytes).
 *
 * Measure and stage each perform their own authoritative same-descriptor LOAD
 * validation. Measure publishes allocation sizes only. Stage uses those sizes
 * to validate every caller span before I/O, then requires its independently
 * validated retained image to have the same sizes before copying any byte.
 * The second image is authoritative for all content and scalars. No pointer or
 * handle is retained across either call.
 */
int64_t et_c2_private_checkpoint_load_measure_v1(
    const void *path, int64_t maximum_file_bytes,
    int64_t maximum_metadata_bytes, int64_t maximum_tensor_bytes,
    int64_t maximum_tensors, int64_t enforce_operational_profile,
    void *result);

int64_t et_c2_private_checkpoint_load_stage_v1(
    const void *path, int64_t maximum_file_bytes,
    int64_t maximum_metadata_bytes, int64_t maximum_tensor_bytes,
    int64_t maximum_tensors, int64_t enforce_operational_profile,
    const void *measurement, void *tokenizer_fingerprint,
    void *config_fingerprint, void *x1_canonical, void *current_cursor,
    void *epoch_cursor, void *model_c1, void *optimizer_metadata,
    void *optimizer_payload, void *result);

#ifdef ET_C2_CHECKPOINT_LOAD_TESTING
void et_c2_checkpoint_load_test_reset_v1(void);
void et_c2_checkpoint_load_test_fail_after_validate_v1(int64_t enabled);
uint64_t et_c2_checkpoint_load_test_copy_count_v1(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
