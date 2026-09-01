# T1 byte tokenizer contract

Status: **accepted**. The format direction was proposed in issue #17 and PR #26.
The public-runtime topology, X1 interpretation, persistence policy, and I1 lifetime
rules were accepted for implementation in
[issue #1 comment 5480447613](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5480447613).
PR #40 subsequently passed independent exact-head review, supported CI, merge, and
merged-main retest; the evidence is recorded in `docs/INTEGRATION_LOG.md`.

## Public boundary

T1 implements exactly the eight A0 tokenizer operations without changing a name,
arity, error category, shape, dtype, device, or ownership rule:

- `tokenizer-byte config`
- `tokenizer-load path policy`
- `tokenizer-save! tokenizer path policy`
- `tokenizer-encode tokenizer text`
- `tokenizer-decode tokenizer ids`
- `tokenizer-vocab-size tokenizer`
- `tokenizer-fingerprint tokenizer`
- `tokenizer-special-token-id tokenizer name`

Tokenizer values are unforgeable process-local identities whose complete metadata is
deep-owned by the trusted Eshkol registry. `tokenizer-encode` returns a fresh sealed
rank-1 I1 tensor with exact signed-`i64` storage, shape `[U]`, CPU device, dense
row-major layout, zero offset, and byte stride 8. `tokenizer-decode` accepts only a
sealed tensor shell made by the same Wave-1 aggregate. It performs no implicit cast,
copy from another carrier, materialization, transfer, scalar carrier substitution,
or fallback.

`tokenizer-byte` accepts only an X1 resolved configuration created by the same
aggregate, revalidates it, and requires `model.vocabulary-size = 256`. It constructs
the canonical baseline: byte IDs `0..255`, normalization `none`, UTF-8 policy `raw`,
no special tokens, and empty prefix and suffix lists. X1 schema 1.0 is unchanged.
Explicit specials and `raw|strict` policy are available only through validated T1
artifacts in version 1.

## Token semantics

Byte value `b` has ID `b` for every `0 <= b <= 255`. Special IDs uniquely and
contiguously cover `256..(255 + special-count)`. Special names are unique ASCII
strings matching `[a-z][a-z0-9._-]{0,63}` and are serialized by ascending ID.

The version-1 special decode policies are:

- `omit`: emit no bytes;
- `error`: reject decoding that ID with `invalid-argument`.

Encoding never recognizes a special name or spelling in input bytes. Special IDs are
inserted only by the artifact's ordered prefix and suffix lists. Lists may repeat an
ID and may reference only configured `omit` specials. There is no implicit BOS, EOS,
padding, or other default insertion.

Normalization is exactly `none`. Policy `raw` accepts bytevectors, including embedded
NUL and malformed UTF-8, and returns exact bytes. Strings contribute their original
UTF-8 encoding. Policy `strict` validates input bytes before encoding and emitted
bytes after decoding. It rejects overlong sequences, surrogates, isolated
continuations, truncation, and values above U+10FFFF; it never replaces, drops,
normalizes, or repairs bytes. Valid UTF-8 round-trips byte-for-byte.

## Canonical artifact bytes

An artifact is strict seven-bit ASCII TSV. Field separators are one byte `09`, every
record ends with one byte `0a`, and no byte follows the checksum record. BOM, CR,
trailing space, blank lines, non-ASCII, or an extra final LF are corrupt.

Version 1.0 records occur in this exact order:

```text
format<HT>eshkol-byte-tokenizer<LF>
version<HT>1<HT>0<LF>
required-features<HT>0<LF>
optional-fields<HT>0<LF>
limit-file-bytes<HT>1048576<LF>
limit-specials<HT>4096<LF>
limit-name-bytes<HT>64<LF>
limit-header-items<HT>4096<LF>
limit-optional-value-bytes<HT>64<LF>
limit-prefix<HT>4096<LF>
limit-suffix<HT>4096<LF>
payload-bytes<HT><PAYLOAD_BYTES><LF>
kind<HT>byte<LF>
normalization<HT>none<LF>
utf8-policy<HT><raw-or-strict><LF>
byte-ids<HT>0<HT>255<LF>
special-count<HT><N><LF>
special<HT><ID><HT><NAME><HT><omit-or-error><LF>  (N records)
prefix-count<HT><P><LF>
prefix<HT><INDEX><HT><ID><LF>                    (P records)
suffix-count<HT><S><LF>
suffix<HT><INDEX><HT><ID><LF>                    (S records)
checksum-algorithm<HT>sha256<LF>
checksum<HT><DIGEST><LF>
```

Integers are unsigned base-10 ASCII with no sign or leading zero; zero is exactly
`0`. Special records are strictly ascending by ID. Prefix/suffix indices are exactly
`0..count-1`; their ID order is semantic and is not sorted.

`PAYLOAD_BYTES` counts from the `k` of `kind` through the LF ending the final suffix
record, or the `suffix-count` record when the suffix is empty. It excludes the
envelope, `checksum-algorithm`, and `checksum` records and must equal the observed
count.

The required-feature count is followed by sorted unique
`required-feature<HT><NAME><LF>` records. The optional-field count is followed by
sorted unique `optional-field<HT><NAME><HT><VALUE><LF>` records. Names use the
special-name grammar. Optional values are nonempty, even-length lowercase hexadecimal
encoding of at most 64 bytes. Both counts are zero in 1.0. A reader rejects a major
other than 1 with `version-mismatch` and rejects a noncanonical version integer as
`corrupt-data`. A higher version-1 minor is accepted only when required features are
empty and every extension is declared inert as an optional field. A semantic
extension must be required.

## Integrity and identity

Let `A` be exact artifact bytes from `format` through the LF ending
`checksum-algorithm`. Let `H` be exact envelope bytes from `format` through the LF
ending `limit-suffix`, including declared features/optional fields but excluding
`payload-bytes`. Let `P` be the exact declared payload.

The artifact checksum is lowercase SHA-256 of:

```text
"eshkol-byte-tokenizer-checksum-v1\n" || A
```

The identity digest is lowercase SHA-256 of:

```text
"sha256:eshkol-byte-tokenizer-v1\n" || H || P
```

The public fingerprint is exactly:

```text
sha256:eshkol-byte-tokenizer-v1:<64 lowercase hexadecimal digits>
```

The digest binds the format/canonicalization version, fixed byte mapping,
normalization and UTF-8 policy, complete special table, and ordered prefix/suffix
behavior. Reserializing a valid 1.0 artifact reproduces the same bytes. SHA-256 is an
unkeyed integrity digest, not authentication; no alternate digest is substituted.

## Limits and strict validation

Version 1.0 hard limits are:

- file bytes: 1,048,576;
- semantic payload/metadata bytes: 1,048,576;
- specials: 4,096;
- special-name bytes: 1..64;
- record bytes before LF: at most 256;
- required-feature and optional-field count fields: at most 4,096 each; nonzero
  required-feature counts remain unsupported, while v1.0 requires both counts zero;
- optional-field decoded bytes: 1..64;
- prefix and suffix entries: 4,096 each; and
- token IDs: nonnegative signed `i64`, further constrained by the byte/special rules.

These are exact canonical-format ceilings and supported per-artifact runtime
admission ceilings in a fresh process on the supported lane; T1
does not silently substitute a lower special, prefix, suffix, or name limit. A caller
policy may lower only the effective file and payload limits described below; it has
no count-specific field. RSS and elapsed-time thresholds used by the quality gate are
evidence budgets, not hidden input limits or new serialized fields. Admission at a
format ceiling does not promise memory proportional to serialized bytes,
constant-time processing, or an unlimited number of admissions in one process. The
trusted representation deep-owns parsed metadata, and the process-lifetime retention
rules below make cumulative resource use a separate operational concern from whether
one artifact is valid.

Readers bound file and line lengths before allocation, then validate ASCII and record
order, numbers/counts, version/features, declared limits/size, checksum, and all
semantic invariants before registering a tokenizer. Duplicate or unknown records,
names, or IDs; gaps/collisions; bad references or policies; truncation/trailing bytes;
noncanonical numbers/order; overflow; and declared-size disagreement never expose a
partial tokenizer. Artifact data never selects code, callbacks, providers, native
libraries, paths, or capability evidence.

## Persistence policy and atomic I/O

For Wave 1, `transformer.persistence` implements only the five-argument A0
`persistence-policy`. It returns an unforgeable observationally immutable shell with
privately copied fields validated by C1. The C2-owned checkpoint operations remain
unavailable.

Every T1 load/save revalidates the shell. Device must be `cpu`. The effective T1
file limit is `min(max-file-bytes, 1048576)` and the semantic payload/metadata limit
is `min(max-metadata-bytes, 1048576)`. C1 `max-tensor-bytes` and `max-tensors` remain
validated and retained but do not apply to this tensor-free artifact and cannot
increase T1 limits. Forged, copied, mutated, foreign-aggregate, or malformed policies
reject before filesystem I/O.

T1 reuses C1 ABI 1.0 exact reads and atomic writes. Save validates and serializes all
bytes before I/O, creates a same-directory mode-0600 unpredictable temporary, checks
short/EINTR/zero writes, fsyncs and closes it, atomically replaces the destination,
then fsyncs the parent directory. Rename is the publication commit. Pre-commit failure
leaves the prior target unchanged and attempts exact-temp cleanup. A post-rename
directory-sync/close failure is `io` with `published? = #t` and durability `unknown`.
No universal power-loss, NFS, FUSE, hostile-parent, or concurrent-writer guarantee is
claimed.

## Wave-1 aggregate and I1 lifetime

One canonical aggregate is compiled from trusted E1, P1, D1, X1, C1, and T1 sources
and localized exactly once. It exports exactly 46 reviewed globals: six E1
accessors, seventeen P1 wrappers, eight D1 wrappers, six X1 wrappers, one policy
wrapper, and eight T1 wrappers. Module source surfaces remain narrow. All constructors,
raise/dispatch seams, registries, trusted helpers, generated companions, C1/T1
internals, P1 privilege, D1 private operations, raw I1 calls, and C1 I/O calls are
local. Already-localized registry artifacts are never aggregate inputs.

The private T1 shell ABI has fixed-arity create, length, write, read, seal, and
unpublished-only abort operations. Registry lookup by raw pointer value occurs before
every dereference. One shell owns exactly one rank-1 I1 tensor plus its live borrow.
Writes are construction-only. Seal is the publication commit; no fallible operation
occurs afterward before the public result returns. Abort ends the borrow, destroys
the tensor, and unregisters only an unpublished shell. Decode accepts sealed shells
from the same aggregate only. Native code transports exact values and manages
lifetime; Eshkol owns tokenization, UTF-8, special handling, parsing, serialization,
checksums, fingerprints, policy selection, and errors.

## Process lifetime, complexity, and operational use

The aggregate has three strong append-only process-lifetime identity registries:

- every successful `tokenizer-byte` or `tokenizer-load` retains its tokenizer shell,
  complete deep-owned core, canonical artifact bytes, and parsed metadata;
- every successful `persistence-policy` retains its shell and private copy of all
  five validated fields; and
- every successful `tokenizer-encode` retains the Eshkol admission entry, native
  sealed shell, owned I1 tensor storage, and live borrow/view.

Each registry prepends new entries and uses a linear identity scan. Lookup is
therefore O(N) in successfully registered values, and retained memory grows
monotonically with their count and size. Dropping an application reference does not
remove the strong registry reference. The pinned runtime has no reviewed finalizer,
weak-identity facility, explicit public release operation, or automatic reclamation
path for these values. Failed operations before publication do not register a
tokenizer or policy, and unpublished encoded shells are aborted and destroyed, but
successful published values remain live until process exit.

Registry mutation and lookup are not synchronized, and concurrent or reentrant use
is unverified. Callers must serialize T1 construction, load/save, encode/decode, and
accessor calls. Long-running code should construct a policy once, construct or load a
tokenizer once, and reuse those identities. It must not repeatedly recreate equal
policies/tokenizers or assume temporary encode results are reclaimed. Where bounded
retention cannot be arranged, a bounded worker process and process exit are the only
currently reviewed reclamation boundary. T2/D2/training must not adopt repeated
ephemeral encoding without a separately reviewed reclamation/reuse mechanism.

The shell is not a K1 capability, general tensor interoperability surface, numerical
kernel, cast, transfer, fallback, or performance claim. The exact format limits are
not a promise that cumulative append-only registry growth is bounded.
