# G3-R fixed-record in-memory codec leaf

This source-private prerequisite implements only the 288-byte `G3RCV1`
companion layout in the accepted [G3-R continuation contract](G3_R_DATA_ONLY_CONTINUATION_CONTRACT.md).
It reuses the checked-in C1 `c1-sha256` implementation for the record digest.
No C2 file is opened, saved or loaded; no generator, model, tokenizer, cache,
RNG owner or registry entry is created. The caller-supplied 32-byte C2 digest
is an **already authenticated immutable-image digest precondition**. Equality
with that value checks the pair only; the codec cannot authenticate the image
or its model/tokenizer identity. The 96-byte fingerprint is checked for the
exact raw-V256 prefix and lowercase SHA syntax, not against a live T1 owner.

`g3r-record-encode!` takes nine arguments: an exact 288-byte destination,
policy vector of six exact signed i64 words, G3-S vector of four signed i64
words, one- or two-byte raw history, origin prompt length, prior generated
count, prior manual append count, exact 96-byte T1 fingerprint, and 32-byte
C2 digest. `g3r-record-decode!` takes the exact 288-byte source, expected
32-byte C2 digest, and an exact eight-slot mutable general vector. On success
the slots contain newly owned data-only policy, RNG, history, prompt length,
generated count, manual count, fingerprint and C2 digest, in that order.
Neither function grants authority to use those words as a live RNG or model.

Both operations validate exact dimensions, canonical little-endian header,
version/profile/algorithm, zero padding/unused history, history partition,
greedy/categorical f32 bit domains, budget/EOS, RNG version/seed/full signed
counter words including exhaustion, and checksum. Encode builds and hashes
a private staged record before its sole fixed destination copy. Decode hashes,
validates and allocates fresh slot values into a separate eight-slot vector,
then uses one checked whole-range `vector-copy!` publication. Pinned fe9
`eshkol_region_copy_tagged_checked` stages every tagged promotion before any
destination slot is written; a late promotion failure cannot leave early
slots published. A numeric tensor-backed vector cannot hold these data-only
objects and is rejected by the same checked publication.
Rejected inputs leave the caller's destination bytes or slot identities
unchanged. Calls assume serialized access to caller-owned bytevectors; there
is no claim under concurrent mutation or process failure during a copy.

The separate feature-off fixture loads only C1 SHA and has no G3-R symbols.
The enabled fixture tests exact greedy, generated-EOS/exhausted categorical,
carry, high-bit counter and manual EOS-match/zero-budget vectors; an
independent Python `hashlib`/`struct` oracle compares all 288 bytes. Malformed
input, truncated/extended records, repaired checksums over invalid semantics,
reserved bytes, wrong pair digest, and failed whole-range publication with
destination preservation are negative cases. The pinned supported normal,
repeat, ASan+UBSan+LSan Eshkol gate and exact symbol/source closure remain
the acceptance gate for this leaf. No C2 pair publication, typed live-RNG
import, live model restore, full-history no-draw replay, continuation or
larger-profile G3-R claim follows from this codec.
