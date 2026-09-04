# T2 deterministic BPE tokenizer contract

Status: **proposed; implementation candidate under review**. This contract extends tokenizer capability without changing
the accepted T1 `eshkol-byte-tokenizer` format, its fingerprint domain, or any of
the eight public tokenizer names/arities. The pre-freeze decision is coordinated in
issue #1 and recorded in `docs/INTEGRATION_LOG.md`.

## Boundary and composition

The existing public tokenizer operations accept either an accepted T1 byte tokenizer
or a validated T2 BPE tokenizer. T2 adds no installed public procedure. BPE training,
streaming encode/decode state, and D1 token extraction are fixed-arity build-only
Eshkol contracts for trusted later CLI/data composition. They are localized in the
canonical Wave-2 aggregate and are not independently linkable capabilities.

The Wave-2 aggregate is compiled from the accepted E1/P1/D1/X1/C1/T1 trusted sources
plus T2 and localized exactly once. Its public global surface remains the accepted 47
definitions. It must not be linked with the already-localized Wave-1 aggregate.
Public whole-input encode still returns exactly the accepted sealed rank-1 CPU I1
`i64` shell and therefore retains the accepted process-lifetime behavior. Streaming
feeds use private i64-le staging bytevectors and never publish one I1 shell per feed.

## Vocabulary and deterministic training

Byte value `b` has ID `b` for every `0 <= b <= 255`. A model has `M` learned merges.
Merge rank `r`, for `0 <= r < M`, has result ID `256+r`; both operands are less than
the result ID. Special IDs uniquely and contiguously cover
`256+M .. 255+M+special-count`. T1 special-name and `omit|error` decode-policy rules
are unchanged. Prefix and suffix lists may repeat only configured `omit` specials.
Encoding never recognizes special spellings.

Training input is a bounded proper acyclic list of documents. Each document is a
bounded proper acyclic list of string or bytevector chunks. Strings contribute their
original UTF-8 bytes; chunks concatenate within a document. Explicit document
boundaries never contribute adjacent pairs. Empty documents and chunks are valid.
`strict` validates each complete document; `raw` preserves every byte.

Training begins with byte tokens and repeats:

1. count every adjacent token-ID pair independently within every document;
2. discard pairs whose concatenated decoded spelling would exceed 256 bytes;
3. choose the pair with greatest count that meets `minimum-frequency`; ties use
   ascending unsigned `(left-id,right-id)`;
4. assign the next contiguous merge ID and replace that pair left-to-right without
   overlap in every document.

It stops when `maximum-merges` is reached or no eligible pair meets the minimum.
`maximum-merges` is a bound, not a demand to invent unreachable merges. Count
addition is checked before increment. Because document contributions are summed and
the tie-break is total, admitted document order is non-semantic. Chunk partition
inside a document is also non-semantic.

The build-only constructor is:

```scheme
(t2-bpe-train-core documents maximum-merges minimum-frequency utf8-policy
                   special-specs prefix-names suffix-names)
```

`special-specs` is in strict ascending special-name order and contains `(name
decode-policy)` records. Prefix/suffix names resolve only to `omit` specials. The
constructor returns a newly owned private core. Caller errors are `invalid-argument`;
checked internal invariant failures are `internal`.

## Canonical artifact

The artifact is strict seven-bit ASCII TSV with one LF ending every record and no
byte after the checksum LF. Unsigned integers use canonical base-10 spelling. Version
1.0 has no optional fields and requires exactly this record order:

```text
format<HT>eshkol-bpe-tokenizer<LF>
version<HT>1<HT>0<LF>
required-features<HT>0<LF>
limit-file-bytes<HT>1048576<LF>
limit-merges<HT>256<LF>
limit-token-bytes<HT>256<LF>
limit-specials<HT>4096<LF>
limit-name-bytes<HT>64<LF>
limit-prefix<HT>4096<LF>
limit-suffix<HT>4096<LF>
payload-bytes<HT><PAYLOAD_BYTES><LF>
kind<HT>byte-bpe<LF>
normalization<HT>none<LF>
utf8-policy<HT><raw-or-strict><LF>
byte-ids<HT>0<HT>255<LF>
merge-count<HT><M><LF>
merge<HT><RANK><HT><RESULT-ID><HT><LEFT-ID><HT><RIGHT-ID><LF> (M records)
special-count<HT><N><LF>
special<HT><ID><HT><NAME><HT><omit-or-error><LF>               (N records)
prefix-count<HT><P><LF>
prefix<HT><INDEX><HT><SPECIAL-ID><LF>                         (P records)
suffix-count<HT><S><LF>
suffix<HT><INDEX><HT><SPECIAL-ID><LF>                         (S records)
checksum-algorithm<HT>sha256<LF>
checksum<HT><DIGEST><LF>
```

`PAYLOAD_BYTES` covers `kind` through the LF ending the last suffix record, or the
`suffix-count` record for an empty suffix. Merge ranks and result IDs are exactly
`0..M-1` and `256..255+M`. Operands reference only byte or earlier merge IDs. Operand
pairs are unique. Reconstructed result spellings must be at most 256 bytes. Special,
prefix, suffix, UTF-8, line, ASCII, and canonical-order rules otherwise match T1.

Let `A` be exact bytes from `format` through the LF ending `checksum-algorithm`, `H`
the exact header from `format` through `limit-suffix` excluding `payload-bytes`, and
`P` the declared payload. The checksum is lowercase SHA-256 of:

```text
"eshkol-bpe-tokenizer-checksum-v1\n" || A
```

The identity digest is lowercase SHA-256 of:

```text
"sha256:eshkol-bpe-tokenizer-v1\n" || H || P
```

The public fingerprint is
`sha256:eshkol-bpe-tokenizer-v1:<64 lowercase hexadecimal digits>`. It is 95 UTF-8
bytes and therefore fits D1's unchanged opaque 1..192-byte fingerprint field.

Readers bound the physical file and declared payload before allocation, validate the
checksum before publishing a tokenizer, then validate every merge and reconstructed
spelling, special, insertion, and canonical invariant. Truncation, trailing bytes,
duplicate/gapped/out-of-order records, forward/self references, collisions,
noncanonical integers, count/size disagreement, checksum failure, and invalid stored
IDs are `corrupt-data`. Unsupported version or algorithm follows the existing public
tokenizer error categories. No partial tokenizer is registered.

The production-linked compiled parser matrix freezes every v1 format, ASCII/LF/line,
envelope, payload, order, arity, checksum, canonical-integer, count, merge, special,
insertion, and persistence-policy invariant. It contains 360 deterministic,
byte-distinct malformed artifacts, asserts exact public category and `tokenizer-load`
operation for each, and
immediately reloads the canonical model after every rejection to prove that no failed
parse publishes a caller-visible tokenizer or poisons later admission. Separate valid
artifacts exercise the exact 256-merge and 4,096-special/prefix/suffix parser ceilings;
one-over records reject. Python creates bytes only; all parsing and assertions execute
in the delivered Eshkol AOT.

## Exact operational limits

Version 1.0 admits exactly:

- artifact bytes and semantic payload: at most 1,048,576, further lowered by the
  accepted T1/C1 persistence policy;
- learned merges: 0..256;
- decoded bytes in one learned token: 1..256;
- specials, prefix entries, and suffix entries: 0..4,096 each;
- special names: 1..64 bytes and the accepted T1 grammar;
- training input: at most 65,536 aggregate bytes, 4,096 documents, and 4,096 chunks;
- one streaming logical input/output: at most 65,536 input/decoded bytes and
  73,728 decoder token IDs (589,824 i64-le staging bytes). The token ceiling is
  65,536 possible byte tokens plus 4,096 prefix and 4,096 suffix insertions.

The focused gate admits every constructible semantic/count maximum and rejects its
one-over case before unbounded allocation; the physical file gate separately rejects
1,048,577 bytes. Training/stream and delivered public-runtime processes are bounded
at 60 seconds and 524,288 KiB peak RSS; the gate also rejects the pinned runtime's
heap-pressure warning. Decoder push validates and sizes the complete current token
chunk before mutating state, then allocates exactly that chunk's decoded-byte count;
it never reserves the remaining logical-stream budget. Across successful nonempty
output pushes this scratch is therefore at most the unchanged 65,536-byte decoded
stream ceiling. Omit-only pushes allocate only bounded zero-length result objects,
with their count capped by the unchanged 73,728-ID ceiling. The focused compiled
gate exercises both one 73,728-ID chunk and the adversarial maximum partition of
73,728 one-ID chunks under the same time, RSS, and no-warning admission budget.
Other runtime probes have a 60-second timeout. These
thresholds are measured evidence budgets, not
additional hidden format or runtime limits. No GPU, alternate dtype, normalization,
dropout, stochastic training, approximate count, scalar fallback, Python runtime, or
unbounded corpus claim is made.

## Whole and streaming semantics

Whole encode initializes the byte sequence, applies merge rules in ascending rank,
then adds prefix and suffix once. Whole decode recursively reconstructs merge bytes,
omits `omit` specials, rejects `error` specials and unknown IDs, and applies the
accepted `raw|strict` policy.

The private streaming contracts are:

```scheme
(t2-stream-encoder-open core)
(t2-stream-encoder-push! state chunk)
(t2-stream-encoder-finish! state)
(t2-stream-decoder-open core)
(t2-stream-decoder-push! state i64-le-token-chunk)
(t2-stream-decoder-finish! state)
```

Push returns a newly owned private staging chunk. Encoder state composes one
left-to-right transducer per merge rank. A stage retains at most one pending input
token; it replaces its configured pair without overlap and forwards every other
token to the next stage. Composition in ascending rank is exactly whole-input merge
application, while preserving pending stage tokens across feeds makes chunk
partition non-semantic. Finalization flushes stages in ascending rank and adds suffix
specials. The bound is exactly one pending token per learned merge, not an assumed
raw-byte lookbehind: later ranks can otherwise propagate a boundary effect farther
left than the longest learned token. Prefix specials occur only in the first logical
output. Decoder state retains at most an incomplete UTF-8 scalar in strict mode.
Empty chunks are valid. Feed after finish, double finish, forged/wrong-kind state,
malformed staging, out-of-range ID, and total-limit excess are explicit errors; a
failed state cannot be resumed. Decoder admission depends only on aggregate decoded
bytes and token IDs, not on chunk partition. Build-only cores and states are uniquely
owned and
must not be mutated except through these operations; arbitrary in-place mutation of
their private vector representation is outside the contract. Registry use is
serialized.

## D1 composition

The private T2 D1 seam fully invokes the accepted D1 validator under caller limits,
compares summary fingerprint byte-for-byte and vocabulary exactly with the supplied
same-aggregate tokenizer, and only then returns bounded private i64-le token staging.
D1 bytes, public operations, publication, and corruption categories are unchanged.
A self-consistent corpus paired with the wrong tokenizer is `invalid-argument` under
operation `t2-token-corpus-read`; malformed corpus bytes remain D1 `corrupt-data`.
Concurrent namespace mutation and hostile-directory behavior retain D1's explicit
limitations. The accepted D1 summary registry is append-only: every successful
internal validation publishes one retained summary until process exit, including a
validated corpus whose subsequent tokenizer fingerprint/vocabulary comparison
fails. Callers load and reuse stable corpus/tokenizer identities in a bounded worker.
