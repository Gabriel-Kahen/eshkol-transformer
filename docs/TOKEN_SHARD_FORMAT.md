# D1 token corpus format

Status: **D1 v1.0 direction accepted with conditions through issues #1 and #18;
implementation review-ready**.

## Scope

D1 stores a flat ordered sequence of token IDs that has already been produced by a
tokenizer. It records the tokenizer identity but never parses, normalizes, recomputes,
or follows it. It defines no document boundary, text tokenization, dataset iteration,
shuffle, packing, batch, target, mask, cursor, or resume behavior; those remain T1/T2
or D2 work.

The first-release writer borrows a proper, acyclic Eshkol list and validates every
element as an exact signed-`i64` value. This is an explicitly CPU control-plane,
complete-list-in-memory authoring interface—not an A0 tensor, `tensor.i64`
capability, T1-output substitute, conversion, or fallback. It is not a streaming-
scale ingestion interface. A later streaming/I1 integration must preserve these
exact bytes and rerun D1 determinism, publication, and corruption gates.

A published corpus view consists of one `manifest.etm` plus the shards referenced by
its ordered records. Shard paths are derived as
`shard-%016d.ets`, with a zero-padded decimal record index. No path is serialized or
followed. The v1 validator does not enumerate the directory and ignores unrelated
files and unreferenced orphan shards; duplicate, repeated, gapped, or out-of-order
records in the manifest are corrupt. Files use little-endian fixed-width integers
and have no executable fields.

## Common version and integrity rules

- Format version is `(major=1, minor=0)`, stored as two unsigned 16-bit integers.
  Version 1.0 readers reject any other major or minor with `version-mismatch`; they
  never guess compatibility.
- Required-feature count is an unsigned 32-bit field. It is zero in v1.0. A nonzero
  value is `unsupported`. Version 1.0 has no optional field or feature that may be
  ignored. All reserved bytes are zero; nonzero reserved bytes are `corrupt-data`.
  A future minor format must assign its extension bytes and optional/required rules
  explicitly before a reader may ignore them.
- Token encoding ID `1` is `i64-le`: exactly eight little-endian bytes per token.
  Although the bits occupy an unsigned field on disk, v1 tokens are nonnegative
  signed-`i64` values and must satisfy `0 <= token < vocab_size`. Other encoding IDs
  are `unsupported`; there is no narrowing or dtype fallback.
- Checksum ID `1` is SHA-256 and every digest is exactly 32 raw bytes. Other checksum
  IDs are `unsupported`; there is no shell, Python, library, or alternate-algorithm
  fallback in the production path.
- Every semantic count, size, index, and vocabulary value is at most
  `2^63-1`. The implementation additionally caps total encoded tokens at
  `1152921504606846900` for signed-`i64` total arithmetic. SHA-256's 64-bit bit-
  length field independently limits the bytes covered by any one file digest to
  `2^61-1` (`2305843009213693951`). Thus a shard must satisfy
  `80 + fingerprint_bytes + 8*token_count <= 2^61-1`, and a manifest must satisfy
  `96 + fingerprint_bytes + 56*shard_count <= 2^61-1`, before allocation. Shard
  count is at most `10^16` so every derived name has exactly 16 decimal digits.
- Before multiplying a count by its serialized width, the reader requires that count
  to be in the signed-`i64` domain and below the relevant format bound. Before the
  writer allocates its record table or any output bytevector, it verifies that the
  exact manifest size and the sum of all derived shard sizes fit signed `i64`.
- The tokenizer fingerprint is an opaque canonical UTF-8 string. Its encoded byte
  length is `1..192`; a reader decodes and re-encodes it and requires byte equality.
  It is then compared byte-for-byte across the manifest and all shards. D1 imposes no
  prefix, digest length, or lexical interpretation. T1's concurrently proposed
  lexical form is not an accepted D1 requirement.

Readers bound each file bytevector allocation using the actual `file-size` and the
caller's manifest-file or shard-file limit. After the bounded manifest load, they
validate serialized counts, the total-token policy, and all signed-`i64`
multiplication/addition before any shard read or offset use. Exact declared sizes
reject truncation and trailing bytes.
`maximum-total-tokens=0` accepts only a corpus whose manifest declares zero tokens;
zero is not an unlimited policy.

## Shard layout

The fixed shard header is 80 bytes. `header_bytes` adds the tokenizer fingerprint.
The payload immediately follows the fingerprint, and the checksum is the last 32
bytes.

| Offset | Width | Type | Field / required v1.0 value |
|---:|---:|---|---|
| 0 | 8 | bytes | magic / format ID `ESHKTSH1` |
| 8 | 2 | `u16-le` | major = `1` |
| 10 | 2 | `u16-le` | minor = `0` |
| 12 | 4 | `u32-le` | `header_bytes = 80 + fingerprint_bytes` |
| 16 | 4 | `u32-le` | required-feature count = `0` |
| 20 | 4 | `u32-le` | token encoding ID = `1` (`i64-le`) |
| 24 | 4 | `u32-le` | checksum ID = `1` (SHA-256) |
| 28 | 4 | bytes | reserved, all zero |
| 32 | 8 | `u64-le` | canonical shard index, semantic signed-`i64` |
| 40 | 8 | `u64-le` | token count, semantic signed-`i64` |
| 48 | 8 | `u64-le` | `payload_bytes = token_count * 8` |
| 56 | 8 | `u64-le` | positive vocabulary size, semantic signed-`i64` |
| 64 | 4 | `u32-le` | tokenizer fingerprint byte count, `1..192` |
| 68 | 4 | bytes | reserved, all zero |
| 72 | 8 | `u64-le` | exact file bytes |
| 80 | variable | bytes | canonical UTF-8 tokenizer fingerprint |
| `header_bytes` | `payload_bytes` | `i64-le[]` | flat token payload |
| `file_bytes-32` | 32 | bytes | SHA-256 of every preceding shard byte |

`file_bytes` must equal `header_bytes + payload_bytes + 32`. The manifest record
stores the same digest as the shard trailer: SHA-256 over the shard header,
fingerprint, and payload, excluding the trailer. Binding the index, version, identity,
vocabulary, counts, and payload prevents a valid shard from being silently moved to
another record.

## Manifest layout

The fixed manifest header is 96 bytes. `header_bytes` adds the tokenizer fingerprint.
It is followed by exactly `shard_count` fixed 56-byte records and a 32-byte checksum.

| Offset | Width | Type | Field / required v1.0 value |
|---:|---:|---|---|
| 0 | 8 | bytes | magic / format ID `ESHKTCM1` |
| 8 | 2 | `u16-le` | major = `1` |
| 10 | 2 | `u16-le` | minor = `0` |
| 12 | 4 | `u32-le` | `header_bytes = 96 + fingerprint_bytes` |
| 16 | 4 | `u32-le` | required-feature count = `0` |
| 20 | 4 | `u32-le` | token encoding ID = `1` (`i64-le`) |
| 24 | 4 | `u32-le` | checksum ID = `1` (SHA-256) |
| 28 | 4 | bytes | reserved, all zero |
| 32 | 8 | `u64-le` | shard count, semantic signed-`i64`, at most `10^16` |
| 40 | 8 | `u64-le` | total token count, semantic signed-`i64` |
| 48 | 8 | `u64-le` | sum of exact shard file bytes |
| 56 | 8 | `u64-le` | positive vocabulary size, semantic signed-`i64` |
| 64 | 8 | `u64-le` | positive canonical tokens-per-shard limit |
| 72 | 4 | `u32-le` | tokenizer fingerprint byte count, `1..192` |
| 76 | 4 | `u32-le` | record bytes = `56` |
| 80 | 8 | `u64-le` | exact manifest file bytes |
| 88 | 8 | bytes | reserved, all zero |
| 96 | variable | bytes | canonical UTF-8 tokenizer fingerprint |
| `header_bytes` | `56*shard_count` | records | ordered shard table |
| `manifest_bytes-32` | 32 | bytes | SHA-256 of every preceding manifest byte |

Each record is:

| Record offset | Width | Type | Field |
|---:|---:|---|---|
| 0 | 8 | `u64-le` | shard index |
| 8 | 8 | `u64-le` | token count |
| 16 | 8 | `u64-le` | exact shard file bytes |
| 24 | 32 | bytes | shard digest defined above |

Records are unique, ordered, and contiguous from zero. An empty corpus has zero
tokens, zero shards, and zero total shard bytes. Otherwise every nonfinal record has
exactly `shard_token_limit` tokens and the final record has `1..shard_token_limit`.
No empty shard is valid. Record token and byte sums must equal the manifest totals.

## Writer and publication contract

`token-corpus-write!` validates the complete supplied token list and configuration
before mutation, then atomically creates `.d1-writer-lock` as a directory. An existing
lock is an explicit concurrent-or-stale-writer rejection. After acquiring the lock,
the writer rejects preexisting canonical final and deterministic `.tmp` targets.

Each complete shard is written to its same-directory deterministic temporary and
renamed to its final name. After every shard is final, the complete manifest is
written to `manifest.etm.tmp`. That complete temporary remains an exclusion claim
while the writer releases the lock, so even an empty corpus has no unclaimed
concurrent-writer window. It is then renamed to `manifest.etm`; that final rename is
the publication commit point and nothing fallible remains afterward. A directory
without the final manifest is not a corpus, so a failure or crash before commit
cannot expose a partially valid corpus. A handled failure removes this invocation's
derived shards, temporaries, and lock. A crash may leave incomplete/orphan shards, a
manifest temporary, and/or the lock; an operator must establish that no writer is
live, remove those incomplete derived files and lock, then retry.

This is manifest-last publication, not whole-directory atomicity. Close-plus-rename
is executed directly by compiled Eshkol and is tested, but v1 provides no `fsync`
primitive and therefore makes no power-loss durability claim. Overwrite is not
supported. Preexisting targets are never replaced.

## Validation and errors

The validator checks bounded actual size, magic, exact version, required features,
encoding/checksum IDs, zero reserved bytes, header/count arithmetic, canonical UTF-8
identity, checksum, record order/partition/totals, derived filename, exact shard size,
repeated shard metadata, shard checksum, manifest digest binding, and every token
range. It exposes no tokens and returns no summary until every shard passes.

The canonical compiler's compiled `directory-list` probe did not return a usable
list. Consequently v1 validates the closed path set derived from the checksummed
manifest but does not claim rejection of unrelated directory entries or unreferenced
orphan files. No directory-enumeration substitute or Python/shell fallback is used.

No verified canonical primitive distinguishes a regular file from a symlink without
following it. D1 therefore treats the caller-supplied directory and its filesystem
namespace as trusted. It does not claim confinement against a hostile directory,
symlink replacement, or non-regular referenced target. This limitation never
authorizes embedded paths: none are serialized, and only canonical names derived
from checksummed record indices are opened.

- Caller argument/range, preexisting-target, and writer-lock failures:
  `invalid-argument`.
- Missing/unreadable/unwritable files and rename failures: `io`.
- Malformed magic/header/UTF-8, policy-exceeding stored file, truncation, trailing
  bytes, nonzero reserved bytes, duplicate/gapped/out-of-order shard indices,
  checksum failure, identity/count/size/total mismatch, or invalid stored token:
  `corrupt-data`.
- Unknown major or minor version: `version-mismatch`.
- Unknown required features, token encoding, or checksum algorithm: `unsupported`.

No error is converted to an empty corpus, partial summary, skipped shard, Python or
shell implementation, alternate checksum, scalar numeric path, dtype conversion,
device transfer, or other fallback.
