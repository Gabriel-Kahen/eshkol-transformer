#ifndef ET_EVAL5_MOD5_V256_MANIFEST_BRIDGE_H
#define ET_EVAL5_MOD5_V256_MANIFEST_BRIDGE_H

#include "eshkol_transformer/eval5_benchmark_abi.h"

/* Development-only fixed fixture input. Fingerprints and digests must come
 * from the same validated T1/D1/C1 bytevectors used by the AOT witness;
 * the standalone bridge cannot authenticate their source. Audit counts use
 * the eight-field order of et_eval5_v256_audit. Metrics and MC counts use
 * the et_eval5_v256_score output order. */
typedef struct {
  const uint8_t *tokenizer_fingerprint;
  size_t tokenizer_length;
  const uint8_t *corpus_fingerprint;
  size_t corpus_length;
  uint8_t train_manifest_sha256[ET_EVAL5_SHA256_BYTES];
  uint8_t eval_manifest_sha256[ET_EVAL5_SHA256_BYTES];
  uint8_t checkpoint_sha256[ET_EVAL5_SHA256_BYTES];
  et_eval5_status audit_status;
  uint64_t audit_counts[8];
  double metrics[5];
  int64_t mc_counts[2];
} et_eval5_v256_manifest_input;

/* Format and parse in memory only. On every failure, out and out_size stay
 * untouched. No file I/O or publication occurs. */
et_eval5_status et_eval5_v256_manifest_roundtrip(
    const et_eval5_v256_manifest_input *input,
    char out[ET_EVAL5_MANIFEST_MAX_BYTES + 1u], size_t capacity,
    size_t *out_size);

/* Eight-pointer FFI seam for the fixed Eshkol witness. Each fingerprint has
 * exactly 96 bytes; digests are train D1, eval D1, then C1 (3 x 32 bytes).
 * audit_record is status then eight counts, all i64. The output allocation
 * must hold ET_EVAL5_MANIFEST_MAX_BYTES + 1 bytes. On failure, output and
 * out_size are untouched. */
et_eval5_status et_eval5_v256_manifest_roundtrip_live(
    const uint8_t tokenizer_fingerprint[96],
    const uint8_t corpus_fingerprint[96],
    const uint8_t digests[96], const int64_t audit_record[9],
    const double metrics[5], const int64_t mc_counts[2],
    char out[ET_EVAL5_MANIFEST_MAX_BYTES + 1u], uint64_t *out_size);

#endif
