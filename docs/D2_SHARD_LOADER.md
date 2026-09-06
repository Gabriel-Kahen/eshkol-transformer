# D2 memory-bounded shard loader

Status: **accepted contract; implementation in review**. The binding decision is
[integration issue #1 comment 5562461427](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5562461427),
mirrored on [D2 issue #42](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/42#issuecomment-5562461448).
This status does not claim independent D2-R approval or merge.

## Public surface and configuration

D2 implements the ten A0 dataset/batch operations and adds unary
`token-batch-release!`:

```text
token-dataset-open/2          token-dataset-next-batch/1
token-dataset-cursor/1        token-dataset-end?/1
token-dataset-seek!/2         token-dataset-close!/1
token-batch-inputs/1          token-batch-targets/1
token-batch-loss-mask/1       token-batch-validate/1
token-batch-release!/1
```

`token-dataset-open` borrows an acyclic proper alternating option list. It must
contain every key below exactly once and no other key:

| Key | Accepted value |
|---|---|
| `:directory` | nonempty UTF-8 string with no NUL |
| `:batch-size` | positive exact signed i64 `N` |
| `:sequence-length` | positive exact signed i64 `T` |
| `:maximum-manifest-bytes` | positive exact signed i64 |
| `:maximum-shard-bytes` | positive exact signed i64 |
| `:maximum-total-tokens` | nonnegative exact signed i64 |
| `:maximum-batch-bytes` | positive exact signed i64 |
| `:shuffle-seed` | `#f` or nonnegative exact signed i64 |
| `:shuffle-window-rows` | positive exact signed i64 `B` |
| `:packing?` | boolean |

Open deep-copies and normalizes the directory and values, retains neither the
caller list/string nor the tokenizer, and validates the bounded manifest and every
referenced shard before publishing a dataset. This procedural carrier is not an X1
schema extension. Device is exactly CPU.

## Rows, shift, mask, and ordering

For a token stream of length `L`, row count is zero for `L < 2` and otherwise
`1 + floor((L - 2) / T)`. Row `r` starts at `r*T`. Its true positions contain
inputs from source positions `s...` and targets from `s+1...`; unused positions and
unused rows of the fixed final `[N,T]` batch contain token zero and false loss mask.
Every published batch has at least one true mask bit.

Packed mode treats ordered D1 shards as one flat stream and may cross any shard
boundary. Unpacked mode applies the formula independently to each shard and never
crosses a shard for a target. D1 shards are not document boundaries. D2 exposes no
causal mask and does not reset causal attention at a shard crossing.

Optional shuffle permutes logical row ordinals, not row contents. Consecutive
windows of at most `B` rows use descending Fisher-Yates. Draw `x` is digest bytes
0..7 interpreted as u64-le from:

```text
SHA-256("eshkol-d2-window-shuffle-v1\n" ||
        u64-le(seed) || u64-le(window-index) || u64-le(draw-counter))
```

For `m=i+1`, a draw is accepted only below `floor(2^64/m)*m`, then index
`x mod m` is used. Rejections consume counters. Counter exhaustion is
`unsupported` and never wraps. This removes modulo bias within each window
conditional on the digest stream; it is not a global-uniform permutation claim.
`B=1` and seed `#f` are canonical identity orders. D2 defines one finite order,
not epochs or epoch reseeding.

## Carrier and lifetime

Each live batch owns exactly two CPU dense row-major `i64[N,T]` planes and one CPU
dense row-major one-byte `bool[N,T]` loss-mask plane. Offset is zero, alignment is
natural, and semantic payload is exactly `17*N*T` bytes. There is no scalar-list,
f32-mask, cast, transfer, alternate precision/device, allocation, or provider
fallback.

The three accessors return stable read-only state-backed opaque identities; repeated
calls return the same identities and allocate/copy nothing. A reviewed
same-aggregate consumer may resolve one identity to its unchanged K1-v1 view only
during one synchronous begin/use/end call. The raw view and pointer cannot escape.
Release or close rejects before mutation while that borrow is active, and callback
failure still ends the borrow.

One dataset owns at most one live batch. Batch and tensor shell copies are aliases
to the current authenticated generation. First release invalidates the batch and all
three tensor identities before the fixed nonrecoverable native destruction tail.
Exact already-released authentic generation aliases release idempotently; other
stale use is `invalid-state`. Forged and wrong-kind values are `invalid-argument`.
No native or Eshkol per-generation tombstone or carrier data remains. Generation
exhaustion is `unsupported` before publication and never wraps. Calls are serialized
and nonreentrant.

The final batch may commit the cursor ordinal to `M` while live. Another next call
is still `invalid-state`; stable EOS begins only after release. Close best-effort
releases the live batch, invalidates dependent access, unregisters the one native
dataset entry, drops config/manifest/cursor/shuffle state, and is idempotent.

## Canonical cursor `ESHKDCU1` 1.0

`token-dataset-cursor` returns newly owned, detached, mutable Eshkol bytevector
storage. Its exact size is `208+F`, where `F` is the canonical tokenizer fingerprint
UTF-8 length in `1..192`:

| Offset | Width | Field |
|---:|---:|---|
| 0 | 8 | magic `ESHKDCU1` |
| 8 | 2 | major `1` |
| 10 | 2 | minor `0` |
| 12 | 4 | `header_bytes = 176+F` |
| 16 | 4 | required feature count `0` |
| 20 | 4 | checksum id `1` (SHA-256) |
| 24 | 4 | shuffle id: `0` identity or `1` bounded-window SHA-256 |
| 28 | 4 | packing: `0` or `1` |
| 32 | 4 | seed present: `0` or `1` |
| 36 | 4 | fingerprint byte count `F` |
| 40 | 8 | vocabulary size |
| 48 | 8 | batch size `N` |
| 56 | 8 | sequence length `T` |
| 64 | 8 | maximum manifest bytes |
| 72 | 8 | maximum shard bytes |
| 80 | 8 | maximum total tokens |
| 88 | 8 | maximum batch bytes |
| 96 | 8 | shuffle seed, zero iff absent |
| 104 | 8 | shuffle window rows `B` |
| 112 | 8 | logical row count `M` |
| 120 | 8 | next ordinal in `0..M` |
| 128 | 32 | stored D1 manifest SHA-256 trailer |
| 160 | 8 | cursor bytes `208+F` |
| 168 | 8 | reserved zero |
| 176 | `F` | canonical fingerprint UTF-8 |
| `176+F` | 32 | domain-separated cursor SHA-256 |

The checksum is
`SHA-256("eshkol-token-dataset-cursor-checksum-v1\n" || bytes[0..175+F])`.
All numeric semantic values fit nonnegative signed i64. The only canonical
`(shuffle-id, seed-present)` pairs are `(0,0)` and `(1,1)`. Directory spelling is
excluded, so an identical relocated corpus is admissible.

Seek validates, in order: carrier and physical size; magic/header/declared-size
arithmetic; version; feature/checksum/shuffle identifiers; remaining cross-field
canonicality; checksum; dataset/options identity; recomputed `M`; and ordinal. It
commits one receiver field only after all checks pass. Caller cursor mutation cannot
affect the dataset and normally makes seek reject. This cursor is D2's only state
contract for C2; D2 defines no C2 container field.

## Resource and error contract

The admitted semantic working set is exactly:

```text
maximum_manifest_bytes + maximum_shard_bytes +
17*N*T + 8*min(B,M)
```

Arithmetic is checked before the corresponding read or allocation. The runtime
retains at most one bounded manifest, one bounded shard, one batch, and one bounded
shuffle window; cache replacement drops the old shard before loading the next.
It retains no whole corpus, token list, row table, or unbounded record table.
Allocator/runtime RSS, stacks, shared libraries, directory/config bytes, and fixed
control/view/hash state are measured separately from semantic payload.

Missing or unreadable manifest/shard data is `io`; malformed, noncanonical,
truncated, trailing, digest-corrupt, record-inconsistent, out-of-range-token, or D1
policy-excess data is `corrupt-data`; unsupported versions are `version-mismatch`;
unknown feature/encoding/checksum/shuffle and counter/generation exhaustion are
`unsupported`; tokenizer or cursor identity/options mismatch is `invalid-argument`;
and closed/live-batch/active-borrow conflicts are `invalid-state`. Invalid config
precedes tokenizer and filesystem work. An admitted allocation failure is `internal`
with `allocation-failed` detail and never selects a smaller path.

D1 manifest-last publication remains the commit point. Orphan or temporary shards
without `manifest.etm` do not form a corpus. On-demand reads revalidate exact shard
metadata, digest, and token ranges; failure destroys unpublished carrier state and
leaves the cursor unchanged.

## Packaging and limitations

The review aggregate source-composes the T2 and I2 trusted closures with their
shared E1/P1L/D1/X1/C1/T1 roots exactly once. It exports exactly 58 globals and 52
package operations: the accepted 47/41 base plus D2's eleven operations. It does not
link localized T2/I2/O2 archives, add a K1 provider, or authorize future C2/TR3
composition.

D2 is CPU-only, serialized, nonreentrant, finite, and loss-mask-only. It does not
claim GPU or alternate dtype support, causal/document-boundary masks, global-uniform
shuffle, epochs, C2/O2 behavior, a public C ABI, or power-loss durability. D1's
trusted-directory, symlink, and concurrent namespace mutation limitations remain.
The accepted config has no separate directory-byte ceiling: successful Linux paths
are bounded by the host pathname/syscall limits, while an overlong rejected caller
string incurs one transient normalized copy but publishes no dataset or native
registry entry.
Python references and fixture generators are development-only and are absent from
the delivered runtime and training path.
