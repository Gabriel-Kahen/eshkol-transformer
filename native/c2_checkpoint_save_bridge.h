#ifndef ESHKOL_TRANSFORMER_C2_CHECKPOINT_SAVE_BRIDGE_H
#define ESHKOL_TRANSFORMER_C2_CHECKPOINT_SAVE_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_C2_CHECKPOINT_SAVE_RESULT_BYTES INT64_C(32)
#define ET_C2_CHECKPOINT_SAVE_RESULT_MAGIC UINT64_C(0x4332534156453130)

enum et_c2_checkpoint_save_phase {
  ET_C2_CHECKPOINT_SAVE_PHASE_NONE = 0,
  ET_C2_CHECKPOINT_SAVE_PHASE_MEASURE = 1,
  ET_C2_CHECKPOINT_SAVE_PHASE_ENCODE = 2,
  ET_C2_CHECKPOINT_SAVE_PHASE_PARSE = 3,
  ET_C2_CHECKPOINT_SAVE_PHASE_POSTCONDITION = 4
};

/*
 * Every pointer addresses a pinned-Eshkol bytevector header: signed i64
 * length followed by exact bytes. Calls retain no pointer and perform no
 * filesystem I/O. Result is exactly 32 bytes. Success publishes magic at 0
 * and file bytes at 8; failure publishes phase/category/code/offset.
 */
int64_t et_c2_private_checkpoint_save_measure_v1(
    const void *tokenizer_fingerprint, const void *x1_canonical,
    const void *current_cursor, const void *epoch_cursor, const void *model_c1,
    const void *optimizer_metadata, const void *optimizer_payload,
    int64_t total_tokens, int64_t epochs, int64_t rng_key_bits,
    int64_t rng_counter_low_bits, int64_t rng_counter_high_bits,
    int64_t maximum_file_bytes, int64_t maximum_metadata_bytes,
    int64_t maximum_tensor_bytes, int64_t maximum_tensors,
    int64_t enforce_operational_profile, void *result);

int64_t et_c2_private_checkpoint_save_encode_validate_v1(
    const void *tokenizer_fingerprint, const void *x1_canonical,
    const void *current_cursor, const void *epoch_cursor, const void *model_c1,
    const void *optimizer_metadata, const void *optimizer_payload,
    int64_t total_tokens, int64_t epochs, int64_t rng_key_bits,
    int64_t rng_counter_low_bits, int64_t rng_counter_high_bits,
    int64_t maximum_file_bytes, int64_t maximum_metadata_bytes,
    int64_t maximum_tensor_bytes, int64_t maximum_tensors,
    int64_t enforce_operational_profile, void *destination, void *result);

#ifdef ET_C2_CHECKPOINT_SAVE_TESTING
void et_c2_checkpoint_save_test_fail_stage_v1(int64_t stage);
#endif

#ifdef __cplusplus
}
#endif

#endif
