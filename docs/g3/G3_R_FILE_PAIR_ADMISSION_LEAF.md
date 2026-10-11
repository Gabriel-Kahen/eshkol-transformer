# G3-R private file-pair read and data-only admission candidate

This source-only candidate composes the accepted G3RCV1 in-memory codec with
the existing owning C2/SHA receipt. It adds no public LOAD, SAVE, generation,
model restore, RNG import, cache replay, or pair writer. It does not change
either predecessor API or wire format.

`g3r-file-pair-admit-internal` takes `c2-path`, `companion-path`, the accepted
C2 persistence `policy`, a real K2 capability `report`, and an exact nine-slot
mutable general `destination`. The C2 path goes through
`g3r-c2-load-with-sha-internal`, including its authoritative second-stage
whole-image SHA and reconstructed detached owner. The companion is then read
through the existing C1 native exact reader into a private 288-byte
bytevector. That reader requires a regular file of exactly 288 bytes, checks
EOF on the opened descriptor, rejects a final symlink, and copies only after
read and close succeed. The codec validates the record checksum, bounded
fields, and equality with the SHA borrowed from the exact receipt. While the
receipt remains borrowed, the leaf compares the record's exact raw-V256
fingerprint and vocabulary 256 with the loaded C2 borrow's tokenizer controls.

On success the destination receives one live C2/SHA receipt followed by the
codec's eight data-only fields: policy, four RNG words, history, prompt length,
generated count, manual count, fingerprint and C2 SHA. The caller owns and
must release the receipt with `g3r-c2-receipt-release-internal!`. All file
reads, decode checks, owner borrows, control comparisons, allocations and
test fault cuts finish before one checked nine-slot `vector-copy!`. Under the
pinned fe9 tagged-vector implementation, promotion of the whole range
precedes any destination write. On failure the private guard ends any active
borrow and releases the receipt; a cleanup defect fails stop. Caller slots
remain unchanged for ordinary rejection and failed whole-range publication.
Caller-owned destinations must not be mutated concurrently.

`tests/g3r/file_pair_test.esk` covers orphan C2/companion files, short and
extended companions, a torn checksum, a digest mismatch, a validly checksummed
wrong tokenizer fingerprint, post-receipt and prepublication faults, failed
whole-vector promotion, and successful release. Its independent Python
fixture generator creates deterministic records from a supplied valid C2
image. The source-only local gate in `tests/g3r/test_file_pair_source.py`
checks composition and publication order, independent record vectors, and
the already accepted C1 native exact reader's failure atomicity. The Eshkol
fixture has not run here because this worktree has no `eshkol-run` binary.

This leaf only authenticates a data-only pair and the loaded C2 tokenizer
controls. It does not prove same-aggregate live T1 tokenizer identity, exact
M3T model/schema and parameter restoration, no-draw replay, or continuation.
The predecessor receipt's published evidence covers native sanitizer
instrumentation and a narrow compiled private path; all-path Eshkol ownership
and this new composition remain unaccepted pending a supported runtime gate
and independent review.
