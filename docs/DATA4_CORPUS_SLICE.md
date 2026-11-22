# DATA4-A bounded corpus selection

Status: implementation candidate with focused local native evidence. This
build-only Eshkol helper is not a public library API or accepted as the full
DATA4 workstream. An opt-in [installed command and report candidate](DATA4_INSTALLED_REPORT.md)
is separately staged for review.

## Input and selection

`(data4-corpus-prepare documents minimum-bytes maximum-bytes)` accepts a proper
list of 1..128 explicit documents. Each document is exactly `(source-id
bytevector)`. Source IDs are unique, 1..64 seven-bit ASCII characters from
`[A-Za-z0-9._-]`; the caller assigns them before this call. Aggregate raw bytes
must be at most 65,536. The inclusive filter bounds are exact integers with
`0 <= minimum-bytes <= maximum-bytes <= 65536`.

Every document is hashed with SHA-256 over its unmodified raw bytes. Filtering
uses raw byte length; then exact byte equality removes repeats among documents
that passed the filter. The first passing occurrence wins. Input order is
semantic, and neither Unicode normalization nor fuzzy/near deduplication occurs.
Input bytevectors and IDs are copied into the prepared value. That value and all
of its descendants are build-only, uniquely owned, and must not be mutated by a
caller. Its shape check detects accidental wrong-kind use; it is not a public
tamper-resistant token.

The prepared value contains kept documents, an ordered row for every input
document `(ordinal, source-id, byte-count, sha256-hex, status,
duplicate-of-or-empty)`, the input and kept counts and byte totals, and the
filter bounds. Status is exactly `keep`, `below-min`, `above-max`, or
`duplicate`. A duplicate names the first kept source ID with the same raw bytes.
Filtered documents cannot suppress a later passing one.

## D1/T2 composition and report

`(data4-corpus-write! tokenizer directory shard-token-limit prepared)` accepts
the trusted same-aggregate T2 tokenizer identity and the prepared value. It
encodes each kept document separately through T2's private i64-le staging seam,
so configured prefix and suffix specials occur once per kept document. It
concatenates the resulting IDs in kept input order, rejects more than 65,536
encoded IDs, and passes the flat list plus T2 fingerprint and vocabulary to the
accepted D1 writer. D1's format and publication semantics remain unchanged.
The shard-token limit is 1..65,536.

The result is `#(D1-summary report-bytes)`. Report bytes are canonical UTF-8
ASCII TSV with LF records. They start with `format`, `version 1 0`, filter and
dedup policy identifiers, bounds, tokenizer fingerprint/vocabulary, input and
kept counts/bytes, shard-token limit, derived shard count, and total tokens.
One `document` record follows for every input document in ordinal order, with
the six row fields above. The final two records are
`checksum-algorithm<TAB>sha256` and `checksum<TAB><64 lowercase hex>`. The
digest is SHA-256 of every report byte preceding the `checksum` record. The
report is constructed before D1 publication; after D1 returns, only assignment
of its summary into the preallocated result remains.

The report is returned in memory. The caller must persist it if needed; this
slice provides no sidecar publication or reader, no independent manifest-digest
binding, and no claim that the report and D1 directory are atomically published
together. D1 has no document boundaries, so the report is the only record of
which retained source contributed each flat segment. Token offsets are not
recorded in this slice. Content SHA-256 records support exact contamination
matching only; benchmark-aware contamination checks remain EVAL5 work.

## Limits and next gate

The 128-document ceiling bounds quadratic exact comparison cost. The path is a
CPU control-plane corpus authoring step; it makes no GPU or streaming-scale
claim. T2's strict/raw behavior and per-document prefix/suffix semantics apply
unchanged. D1's trusted directory, checked-write, no-fsync, and summary-lifetime
limits apply unchanged.

The pinned local source gate passed 28 Eshkol checks, using genuine T2 BPE
encoding and a D1 writer stub. From clean commit `6b885150`, the corrected
one-command same-aggregate native gate passed in the pinned offline Ubuntu
22.04/LLVM 21.1.8 image: compiler 1:12.86, whole gate 1:13.97, peak compiler
and gate RSS 3,460,012 KiB. It produced byte-identical reports, manifests,
and shards on two fresh runs, read back the six expected T2 IDs through D1,
rejected a preexisting target without emitting a report, and classified a
flipped shard byte as D1 `corrupt-data`. Sixteen five-second samples observed
at most 140,393,345 bytes of gate output files; this is a sampled high-water
value, not an exact disk peak. Local logs and hashes are retained at
`/tmp/et-data4-clean-6b885150-20261114/`. The preceding manual object-only
P1 link repair and strict source compile remain historical evidence, not the
basis for this clean-gate result. These are local measurements, not release
acceptance.

The required shared CI engine now registers the dedicated `data4-corpus` suite
with `make test-ci-data4-after-build`. It runs the unchanged source gate first
(including malformed shape, limit, cycle, and token-overflow negatives), then
the unchanged same-aggregate native gate. Its prerequisite plan is clean and
configuration only: both gates build their own inputs and consume no canonical
package artifacts. Both commands also remain mandatory in the full test and
acceptance-predecessor entry points. The engine and exact-evidence inventories
now require 22 suites and 38 commands; prior evidence without DATA4 is rejected.
DATA4 code, gate, and oracle changes retain fail-closed full-suite selection,
while allowlisted prose-only changes retain the existing selection rules.

The exact `562cb353` DATA4-A successor passed hosted full-suite run
`43743963935`, including the required `data4-corpus` suite. Installed CLI3
wiring, report persistence/validation and full DATA4 admission remain separate
gates; that hosted result does not validate the later DATA4-B candidate.
