# EVAL5 benchmark manifests and contamination audits

EVAL5 is a standalone C11 library. It accepts benchmark data and model outputs
from a caller and produces an exact contamination audit, evaluation metrics,
and a checksummed provenance record. The public header also compiles as C++17.
Build the archive with `bash scripts/build-eval5.sh` and link
`build/eval5/libeshkol_eval5.a` with `-lm`. The library requires POSIX.1-2008
file operations, IEEE binary32 and binary64, and round-to-nearest arithmetic
with masked floating-point exceptions and gradual underflow. On x86-64 the
library checks the x87 and MXCSR controls and returns `ET_EVAL5_ERR_FP_ENV`
for unsupported settings without changing them. On other architectures it
checks the rounding mode, and callers must ensure the remaining settings.
The build disables fast math and contraction.

## Audit and scoring

Call `et_eval5_audit_contamination` with the training and evaluation document
arrays. Supply each document's original bytes and, in token mode, its
pretokenized uint32 IDs from the intended tokenizer. Empty spans are valid.
The audit accepts an empty training array and requires a nonempty evaluation
array. Input buffers stay immutable for the duration of each call, including
when input and output storage are supplied by different threads. Keep output
storage separate from inputs and from other outputs.

Exact document identity uses SHA-256 of the original bytes, followed by
length and byte comparison. This also matches empty documents. The n-gram
index uses a rolling polynomial (base 257, input unit plus one, modulo
2^64). Sort and dedup compare equal hashes by the full byte or token span.
A lookup therefore requires actual equality even when two distinct spans
share a hash. Windows stay within document boundaries. The index includes
each distinct training n-gram once. Evaluation counts every window
occurrence, including repeated occurrences in one document.

An evaluation document meets the n-gram criterion when it has at least one
matched window and its matched-window fraction is at least
`min_ngram_overlap`. A threshold of zero means any match qualifies.
Documents shorter than `ngram_order` have zero windows. Exact document and
n-gram counts can intersect, and `contaminated_documents` counts their union.
The reported rate divides that union count by `eval_documents`. The caller
supplies space for one index per evaluation document. The written prefix
contains zero-based indices in ascending evaluation order.

The audit returns `ET_EVAL5_ERR_CONTAMINATED` when the reported rate is
strictly greater than `max_allowed_rate` or when an exact match is present
and `forbid_exact_match` is set. Equality with the maximum rate passes
unless the exact-match rule applies. The summary and indices are available
for either gate result. A benchmark runner should proceed to scoring only
on `ET_EVAL5_OK`. The serializer can also record rejected audits so their
counts and policy remain available for inspection.

- `et_eval5_score_lm` consumes contiguous binary32 logits of shape
  `[token_count, vocab_size]` and one uint32 target per row. It sums
  max-shifted log-sum-exp losses in binary64 in token order and reports total
  NLL, mean NLL, `exp(mean_nll)`, and bits per original byte. The bits-per-byte
  denominator for mean NLL is `ln(2) * (byte_count / token_count)`.
  The caller provides the original byte count for the target rows.
- `et_eval5_score_mc` takes each item's ordered candidate continuations.
  The lowest total NLL wins in raw mode, and the lowest mean NLL wins in
  length-normalized mode. An exact binary64 tie selects the lowest original
  choice index. One-token choices support classification. The adapter reports
  exact counts and their binary64 ratio.

Both adapters reject non-finite logits anywhere in a supplied row and
out-of-range targets. Language modeling also rejects a zero byte count and
perplexity overflow. Dimensions and element counts are checked before a
sequence is read. The caller supplies buffers with the declared extents.

## Canonical manifest v1.0

Initialize `et_eval5_manifest` with version 1.0, the benchmark suite, task
and split identifiers, a positive context length, and the requested sample
cap. A zero cap denotes the full split. Provide the SHA-256 digests of the
training and evaluation datasets. For a sharded dataset, use the digest of
the canonical shard inventory that binds shard contents and order. Supply
the tokenizer fingerprint and checkpoint or model digest from the same
evaluation run. The audit summary covers the supplied audit corpus, which
can include the full evaluation split when the scoring run is capped.

The document starts with the two LF-terminated lines `EVAL5_MANIFEST_V1`
and `version=1.0`. The remaining keys follow the fixed order below. Every
key appears once as `key=value` followed by LF.

| Keys in order | Encoding |
| --- | --- |
| `suite`, `task`, `split` | 1–127 ASCII characters, starting with a letter or digit, then letters, digits, `-`, `_`, or `.` |
| `train_dataset_sha256`, `eval_dataset_sha256`, `tokenizer_sha256`, `checkpoint_sha256` | 64 lowercase hexadecimal digits per digest |
| `context_length`, `max_samples` | Unsigned decimal |
| `contamination.ngram_mode`, `contamination.ngram_order`, `contamination.forbid_exact_match` | Unsigned decimal, with mode 1 for bytes and 2 for tokens |
| `contamination.min_ngram_overlap`, `contamination.max_allowed_rate` | Binary64 |
| `metric_mask`, `mc_normalize_by_length` | Unsigned decimal, with bit 1 for LM and bit 2 for MC |
| `lm.token_count`, `lm.byte_count` | Unsigned decimal |
| `lm.total_nll`, `lm.mean_nll`, `lm.perplexity`, `lm.bits_per_byte` | Binary64 |
| `mc.correct_count`, `mc.total_count` | Unsigned decimal |
| `mc.accuracy` | Binary64 |
| `audit.eval_documents`, `audit.contaminated_documents`, `audit.exact_doc_matches`, `audit.ngram_contaminated_docs`, `audit.total_eval_ngrams`, `audit.matched_eval_ngrams` | Unsigned decimal |
| `audit.contamination_rate` | Binary64 |
| `manifest_sha256` | 64 lowercase hexadecimal digits |

Unsigned decimal has no leading zeroes except for zero itself. Binary64 uses
16 lowercase hexadecimal digits of the IEEE bit pattern in most-significant
digit first order. Metric fields for disabled adapters use positive zero.
The SHA-256 covers the entire prefix through the LF immediately preceding the
checksum line. There are no trailing bytes or NUL terminators in the file.
The decoder checks the fixed schema, length, digest, value ranges, and count
relationships. Enabled LM metrics require positive token and byte counts.
If either NLL is zero, both NLLs and bits per byte must be zero and perplexity
must be one. Otherwise the mean must equal total NLL divided by token count.
Perplexity and bits per byte must match their derived values within
`1e-9 * max(1, expected)`. Validation rejects a positive subnormal total NLL
whose mean rounds to zero. The record is data only. The runner supplies
trusted input digests and is responsible for associating them with the data
it actually scored. The manifest stores the audit summary. The audit API
returns the individual contaminated indices.

`et_eval5_manifest_format` supports a size query, then emits the complete
record and a convenience NUL outside the record length.
`et_eval5_manifest_write_atomic` writes a private temporary file in the
destination directory, fsyncs and closes it, then atomically replaces the
destination entry. Failed writes remove the temporary file. Only the
temporary file receives an fsync. `et_eval5_manifest_read_file` opens with
`O_NOFOLLOW`, checks for a regular file, reads up to 4096 manifest bytes
with one extra byte to detect growth, and verifies the record. The final
path component must be a regular file. The caller selects the parent
directory. `et_eval5_manifest_parse` provides the same verification for an
in-memory record.

## Validation and limits

`EVAL5_ASAN_DETECT_LEAKS=1 bash scripts/test-eval5.sh` builds normal and
instrumented archives, checks their repeat-build SHA-256s and exports,
runs the same driver under both builds, compares the resulting metrics and
manifests byte for byte, and runs a stdlib Python oracle from `tests/eval5/`.
The driver includes an intentional uint64 n-gram collision, malformed input
fixtures, and an end-to-end audit, scoring, publication, and reload.

The audit indexes documents and n-gram occurrences in memory before
deduplication. Memory scales with the training document and window counts.
Evaluation metrics consume caller-provided logits and use the platform's
`libm`, so last-bit agreement across different math library implementations
requires separate validation. The manifest stores the exact emitted bits.
Integration with dataset readers, tokenizers, model inference, and external
benchmark runners is outside this standalone library.
