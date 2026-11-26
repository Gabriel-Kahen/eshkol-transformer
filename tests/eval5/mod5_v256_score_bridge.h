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

#endif
