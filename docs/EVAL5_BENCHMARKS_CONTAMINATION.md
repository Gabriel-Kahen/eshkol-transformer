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
the tokenizer identity digest and checkpoint or model digest from the same
evaluation run. For an accepted T1 or T2 tokenizer, `tokenizer_sha256` is the
32 bytes decoded from the 64-hex-digit suffix of its validated public
fingerprint. T1 uses `sha256:eshkol-byte-tokenizer-v1:...` (96 bytes), and T2
uses `sha256:eshkol-bpe-tokenizer-v1:...` (95 bytes).
The suffix is already domain separated by the tokenizer format. The runner
calls `et_eval5_bind_tokenizer_identity` with the validated T1/T2 public
fingerprint and the authenticated D1 corpus-summary fingerprint, each with an
explicit byte length and vocabulary size. The helper requires the canonical
family prefix, lowercase hex suffix, T1 vocabulary 256–4352 or T2 vocabulary
256–4608, and exact equality of both complete fingerprints and vocabulary
sizes. It returns `ET_EVAL5_ERR_FORMAT` for malformed tokenizer identity or
`ET_EVAL5_ERR_IDENTITY` for a D1 mismatch; failures leave the output digest
untouched. The runner copies the returned digest into `tokenizer_sha256`.
The caller must source these values from the same validated tokenizer and D1
owners used for scoring; the standalone helper cannot authenticate those
owners or prove that a later manifest used its output.
For any other tokenizer, the runner must define and verify a canonical
tokenizer identity digest before using this field. The audit summary covers
the supplied audit corpus, which can include the full evaluation split when
the scoring run is capped.

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
Mean NLL must equal total NLL divided by token count in binary64. Zero
total NLL requires zero bits per byte and perplexity of exactly one. A
positive subnormal total NLL can yield a mean that rounds to zero. The
validator accepts this scorer result when perplexity is exactly one and
bits per byte agrees with `total_nll / (ln(2) * byte_count)` within
`1e-9 * max(1, expected)`. For a positive mean, the same tolerance applies
to both perplexity and bits per byte. The record is data only. The runner
supplies trusted input digests and is responsible for associating them with
the data it actually scored. The manifest stores the audit summary. The audit API
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

## Fixed MOD5/D1 reference oracle (development only)

`tests/eval5/mod5_reference_oracle.py` accepts a CPU-F32 C1 checkpoint and
two validated D1 corpus directories, using the existing IO5 C1 decoder,
MOD5 mapping checker, and D1 reference parser. It checks the fixed one-block
`[V=16,D=16,F=32,Hq=4,Hkv=2,Dh=4]` MOD5 parameter roster and computes
independent binary64 reference logits, token 3-gram and exact-document audit,
LM metrics, and four length-normalized MC decisions. Run
`python3 tests/eval5/mod5_reference_oracle.py model.etcp train-dir eval-dir`.
It emits JSON for comparison with a future live AOT run; it writes no EVAL5
manifest. `python3 -m unittest tests.eval5.test_mod5_reference_oracle` checks
the parser boundary, clean and contaminated fixture outcomes, and rejection.

This oracle uses a deliberately synthetic V=16 tokenizer representation: each
token is written as two ASCII hex bytes solely for fixture audit and the
32-byte LM denominator. It does not prove original-text provenance or the
canonical T1/T2 tokenizer identity required by EVAL5 admission. The older
`codex/eval5-live-mod5-runner` candidate hashes a fixture identity string
for its manifest and hashes a single shard for each dataset; those fields
must be replaced by owner-authenticated tokenizer fingerprints and canonical
D1 inventory digests before a live manifest can be accepted. A live AOT gate
must also compare its scored logits and metrics with this independent oracle
on the exact C1 checkpoint and D1 corpora. The oracle performs development
calculations in Python binary64 and does not assert last-bit parity with
Eshkol or the EVAL5 C scorer.
The current MOD5 fixture has vocabulary 16, while accepted T1/T2 tokenizer
identities require at least 256 IDs. A real T1/T2 benchmark therefore needs
a corresponding 256-or-larger-vocabulary live MOD5 model and matching D1
corpora, or a separately specified and validated non-T1/T2 tokenizer owner.

### V=256 provenance-correct reference fixture

`python3 -m tests.eval5.mod5_v256_fixture build DIR` creates a fresh
development fixture without an Eshkol compiler. `verify DIR` checks it;
`python3 -m unittest -v tests.eval5.test_mod5_reference_oracle
tests.eval5.test_mod5_v256_fixture` runs the small negative and scoring cases.
The generator reuses the accepted T1 reference artifact and byte encoder,
the existing D1 reference writer/validator, and the IO5 C1 encoder/decoder.
It stores two original nine-byte documents per train/clean split and encodes
those exact bytes as raw T1 IDs. Three ordered six-token D1 shards form each
split. `fixture.json` binds the canonical tokenizer fingerprint and artifact,
the untrained deterministic C1 checkpoint, each original document, the full
D1 manifest, and every ordered shard by SHA-256. The complete D1 manifest is
the dataset inventory whose own records bind shard order and digests. The C1
head and token embedding each have shape `[256,16]`; the block dimensions
remain `[D=16,F=32,Hq=4,Hkv=2,Dh=4,T=8]`.

The fixed reference hashes are: T1 artifact
`62e2aa0feef161ec7115f50f172bc24b3b4c51847150d6a663ab71ce9cb001cd`,
C1 checkpoint `505358be27f676998816c9d2ed194358edb017be4ccb2050069c1e22d9279eae`,
train D1 manifest `71e4c68f07866a317aa091575ef555ecda37c04347381dbc795adcb5371023b6`,
and clean D1 manifest `b2e31837e5094b79d95c5ac30a799a54e1c117f60e12511d3853f5ef5bfd69e2`.
The accepted T1 fingerprint is
`sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704`.
The Python reference on this untrained model finds zero contamination matches
among 14 token three-grams, LM total NLL `90.78076731811535` over 16 targets
and 16 original target bytes, and MC accuracy `2/4`. Python computes model
logits in binary64 and rounds each logit to binary32 before scoring; the
values are a tolerance target, not a measured compiled-model result.

The tested native CLI from PR #255 independently generated the same T1
artifact, both D1 manifests, and all six shards byte for byte from the
original documents on pinned, network-disabled Ubuntu 22.04.
[Hosted executable provenance](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/45773952998)

The opt-in MOD5 provider advertises the fixed block shapes used here and has
no vocabulary dimension. The existing MOD5-C and older EVAL5 AOT fixtures
hardcode V=16; the test-only source specialization below executes V=256
without changing them. The reference fixture itself publishes no benchmark
manifest and provides no external-corpus claim.

The read-only hosted CLI receipt at
`/home/gabe/.codex/evidence/eshkol-transformer/eval5-v256-native-input-replay-20261126.w6iFhy`
contains the accepted T1 artifact and both native D1 corpora. Its
`SHA256SUMS` verifies, and all nine T1/D1 artifact files are byte-identical
to fresh compiler-free fixture output. This establishes input byte parity;
the receipt does not execute the V=256 model.

`tests/eval5/build_mod5_v256_witness.py` is the bounded source preparation
for the next gate. It requires SHA-256
`593d9de4dc03521359941bacece2d159c7e022f96f4b0752943b83fc6ae3c31c`
for the accepted MOD5-C E2E source, reuses that source's decoder, K1 calls,
and trusted C1/P1/I2 loader, and only specializes the head/embedding roster
to `[256,16]`, vocabulary to 256, and f32 scratch span to cover the largest
weight. Its appended test-only entry point checks the T1 owner identity,
three authenticated D1 shards, original document bytes, and full manifest
and C1 hashes before scoring. The source-only gate checks these extraction
contracts and balanced generated Eshkol forms; it is not an AOT build result.

The test-only C bridge `tests/eval5/mod5_v256_score_bridge.c` passes four
`[8,256]` binary32 live logit matrices to the existing EVAL5 LM and MC C
scorers. It validates 18 D1 token IDs and finite logits, leaves outputs
untouched on failure, and returns total/mean NLL, perplexity, bits per byte,
MC accuracy/counts, and eight binary32 logit probes. The standalone C test
passes under normal Clang 22 and ASan/UBSan/LSan; this is a bridge test, not a
model execution. The comparator checks live metrics against the fixed Python
reference with relative tolerance `5e-4` and absolute tolerance `5e-3`, and
the eight f32 probes with relative tolerance `5e-4` and absolute tolerance
`2e-3`; MC counts must match exactly. These numerical acceptance bounds were
declared before the AOT run. A mismatched or nonfinite value fails closed.

`scripts/test-eval5-v256-aot.sh` prepares one strict CPU-f32 AOT build using
the same pinned MOD5-C native bridge, trusted C1/P1/I2 archives, and EVAL5
scorer. It compares the live score and probes, then exercises a correctly
checksummed C1 shape mutant and swapped D1 split identity. Its explicit
`EVAL5_V256_AOT_SLOT=1` guard prevents an uncoordinated full AOT build.
Set `EVAL5_V256_EVIDENCE_DIR` to a fresh directory to retain its compiled
source, compiler log, score line, fixture, and negative-path outputs.

The first pinned Ubuntu 22.04/Clang 21 replay on `a13051c5` stopped at AOT
source generation before model execution: Eshkol rejected `volatile-load f64`
and `cddddr`. The retained attempt directory is
`/home/gabe/.codex/evidence/eshkol-transformer/eval5-v256-aot-a13051c5-20261126T190906Z`.
The focused successor uses an FFI f64 readback helper and the already used
`cdddr` list primitive; source and normal/sanitized C checks pass locally.

### Measured V=256 compiled scoring witness

The one-slot replay of commit `a0f2f967d9dccd434faa9058131b26b0e034fa70`
(tree `05224e53cd1de987f17073290c4584439ee9dcfd`) passed on Ubuntu
22.04.5, Clang/LLVM 21.1.8, Eshkol
`fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`, and the pinned
`eshkol-r0-debug:20260928` image. The AOT ELF SHA-256 is
`82be011d931c5143b1963ead2ab0a105e8dba80200657a999fdd7b95ba0022ef`.
The compiled Eshkol decoder loaded the exact C1 checkpoint, validated the
T1 owner fingerprint and all three D1 shards in each split, matched the
original byte documents and complete D1 inventory hashes, and passed live
binary32 logits into the EVAL5 C LM/MC scorers. The score was total NLL
`90.780767317731659`, mean NLL `5.6737979573582287`, perplexity
`291.13816763515604`, bits per byte `8.1855601760866463`, and MC `2/4`.
All eight f32 logit probes and the metrics passed the predeclared comparator;
the largest metric absolute difference from the Python reference was
`6.98173608e-09`. Fresh-process C1 reload passed; a correctly checksummed
head shape mutant and swapped train/eval D1 split both failed before scoring.

The retained evidence directory is
`/home/gabe/.codex/evidence/eshkol-transformer/eval5-v256-aot-a0f2f967-20261126T191053Z`;
its `SHA256SUMS` digest is
`748e7287fc8d97b3194812bb545423c450d152e6ac57094862106381a0b79181`.
It includes the generated source, compile log, executable, fixture, live
score, and negative outputs. A second fresh, network-disabled Ubuntu
container replayed that sealed executable without rebuilding it; its score
and both rejected-path outputs matched byte for byte, with both rejects
returning status 2. The independent receipt is
`/home/gabe/.codex/evidence/eshkol-transformer/eval5-v256-independent-replay-20261126.7DQDU2`;
its `SHA256SUMS` digest is
`0ad31dc93046564ce63be086200dfe0d62bb470812fb5011709953a7eb9e2164`.
That exact source has one AOT build and one deterministic fixture, not a
second-build reproducibility proof, training run, external corpus evaluation, live C
contamination audit, or EVAL5 manifest publication.
A full benchmark runner still needs a live contamination gate and admitted
manifest publication over its scored data and owner-sourced identities.

The measured AOT receipt above remains evidence for its exact source commit.
A subsequent witness-source correction removes two independent path reopens:
the D1 manifest digest now checks the validated `metadata[6]` bytevector,
and the C1 digest checks the one bytevector passed to the accepted internal
decoder. Its state and weight release path matches the existing MOD5 loader;
the separate `--check-c1` path still tests a validly checksummed wrong-shape
checkpoint. The AOT script makes its staged fixture files read-only after
source generation and requires separate C1 shape, C1 identity, and D1 split
identity failures. Source and fixture negative tests cover the correction;
the corrected source passed a separately pinned AOT replay at commit
`ab088449c159cabcf60fc6d1df30cc8722fe8046` (tree
`9f2b763eb2c8985052db3271ed2e203d3a67b29b`) on the same pinned
Ubuntu 22.04.5, Clang/LLVM 21.1.8, Eshkol commit, and image. Its AOT ELF
SHA-256 is
`ac18b4025652809c43f637198b40b36a8b3303cfaa3b3d665487525a45cf23ac`.
The live score line, including eight f32 probes, was byte-identical to the
earlier one, and the comparator again passed with maximum metric absolute
difference `6.98173608e-09`. The well-formed wrong-shape C1 produced
`unexpected MOD5 parameter metadata` in `--check-c1` mode; supplying that
C1 to scoring produced `C1 checkpoint SHA-256 mismatch`. Swapping the D1
splits produced `D1 manifest SHA-256 mismatch`. All three exited with status
2 before any `SCORE` line. The fresh evidence directory is
`/home/gabe/.codex/evidence/eshkol-transformer/eval5-v256-aot-ab088449-20261126T192020Z.E7a9YZ`;
its verified `SHA256SUMS` digest is
`79d1560bf66580162819455dba100cd6664e1873d9252ae473e0c771c22cdf49`.
This replay proves the corrected fixed development witness, not an independent
second build of the same source, live C contamination audit, benchmark manifest,
external evaluation, training, or production runtime behavior.

A second network-disabled Ubuntu container independently replayed the
corrected executable. Its score and all three rejection outputs matched
the sealed run byte for byte. The invalid C1 shape, C1 identity and D1
manifest identity cases each exited with status 2.

### Development-only V=256 live audit candidate

The successor witness retains the same validated T1 artifact, canonical D1
manifest and ordered shards, and original nine-byte documents. Its test-only
C bridge calls `et_eval5_audit_contamination` with those original bytes and
their raw-byte T1/D1 token IDs. The fixed policy is token 3-grams, minimum
overlap `0.25`, maximum contamination rate `0.0`, and exact matches
forbidden. The normal train/eval pair must produce status `0`, two evaluation
documents, 14 windows, and zero exact, n-gram, or matched-window counts
before the C1 load or scoring call. A separate `--audit-control` invocation
uses the already validated training documents as evaluation input; it must
produce status `ET_EVAL5_ERR_CONTAMINATED` (`-7`), two exact and n-gram
matches, all 14 windows matched, and matching indices `0,1`. That control
returns without loading C1, computing logits, or writing a manifest.

The audit bridge passed direct C11 and ASan/UBSan tests. Generated-source
checks enforce audit before C1 load. The compiled composition below runs the
same audit and retains the canonical manifest in the test evidence stream.

### In-memory V=256 manifest round-trip candidate

The successor test-only C bridge `et_eval5_v256_manifest_roundtrip` accepts
the validated T1 and D1 summary fingerprint spans, SHA-256 of each canonical
D1 manifest and the checked C1 bytes, the live audit status and counts, and
the LM/MC score outputs. It calls the accepted
`et_eval5_bind_tokenizer_identity`, formats a fixed development suite/task/
split manifest in local memory, parses it, and requires byte-identical
canonical reformatting. It copies text and size to the caller only after all
checks pass. Contamination, an inconsistent clean audit, tokenizer identity
rejection, invalid metrics, and insufficient output capacity leave both
outputs untouched. The bridge performs no file I/O or publication.

The focused C test uses exact fixture digests, a fresh direct EVAL5 audit,
and score values recorded by the earlier sealed V=256 AOT run. It passes
normal and sanitizer builds, but it does not run the model again or prove
a compiled same-run Eshkol call. No EVAL5 manifest has been published.

The next opt-in source candidate connects this bridge to the same compiled
V=256 witness through a fixed eight-pointer test-only FFI seam. The Eshkol
path retains the parsed T1 fingerprint, both D1 summary fingerprints and
SHA-256 digests of the validated manifest bytevectors, and the SHA-256 of
the same C1 bytes passed to decoding. After a clean live audit and LM/MC
score, it passes those values and the native score buffers to the in-memory
round-trip bridge. The canonical text is emitted between `MANIFEST_BEGIN`
and `MANIFEST_END` on stdout and retained only in the test evidence stream.
The comparator independently checks canonical field order, owner-sourced
digests, exact binary64 metric bits, and final SHA-256. The contaminated
control and T1/D1/C1 identity failures have no manifest-output call. This
source at `b0593e4b` on tree `27b4c78d` passed a network-disabled Ubuntu
22.04 and Clang/LLVM 21.1.8 AOT run using the pinned Eshkol compiler. The
clean audit counted two documents and 14 windows with zero matches. Live
LM and MC scores agreed with the reference within `6.982e-09` absolute error.
The compiled manifest matched the independent canonical comparator. The
contaminated control matched all 14 windows and emitted only its audit
record. The wrong T1 owner, D1 split, and C1 shape and digest cases exited
with status 2 before scoring or manifest output. The test-only manifest
is retained in the local evidence stream.

The sealed receipt is
`/home/gabe/.codex/evidence/eshkol-eval5-live-manifest-20261127.LMcYvd/live.SHA256SUMS`.
