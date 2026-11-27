#ifndef ET_EVAL5_MOD5_V256_SCORE_BRIDGE_H
#define ET_EVAL5_MOD5_V256_SCORE_BRIDGE_H

#include <stdint.h>

/* Test-only bridge: four [8,256] CPU-f32 live MOD5 contexts and the exact
 * 18 T1 byte IDs from two D1 evaluation documents. Results are untouched on
 * error. No manifest, checkpoint, or dataset identity is inferred here. */
int32_t et_eval5_v256_score(const float *doc0, const float *doc1,
                            const float *context_a, const float *context_b,
                            const uint32_t *eval_tokens, double metrics_out[5],
                            int64_t counts_out[2], float probes_out[8]);

/* Compare the one-shot runner with expected_size legacy canonical bytes
 * and their trailing NUL. Fingerprints are T1, train D1, and eval D1 (3 x 96).
 * Digests are train D1, eval D1, and checked C1 (3 x 32). Documents hold the
 * original train and eval bytes in order (2 x 18), matched by their ID spans.
 * All inputs are borrowed reads. The comparison performs no I/O. */
int32_t et_eval5_v256_compare_runner(
    const uint8_t fingerprints[288], const uint8_t digests[96],
    const uint8_t documents[36], const uint32_t train_tokens[18],
    const uint32_t eval_tokens[18], const float *doc0, const float *doc1,
    const float *context_a, const float *context_b,
    const char *expected, uint64_t expected_size);

/* Two owner-validated nine-byte documents per split, with their ordered raw
 * T1/D1 IDs. Token 3-grams, overlap >= 0.25, rate <= 0, and no exact match.
 * summary_out holds six audit counts followed by two matching eval indices;
 * all outputs remain untouched on errors other than CONTAMINATED. */
int32_t et_eval5_v256_audit(const uint8_t train_bytes[18],
                            const uint32_t train_tokens[18],
                            const uint8_t eval_bytes[18],
                            const uint32_t eval_tokens[18],
                            uint64_t summary_out[8]);

/* Fixed-size, checked f64 readback for the Eshkol FFI. Invalid access returns
 * NaN, which the mandatory score comparator rejects. */
double et_eval5_v256_metric(const double metrics[5], int64_t index);

#endif
