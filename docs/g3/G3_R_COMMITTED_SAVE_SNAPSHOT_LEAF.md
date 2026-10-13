# G3-R private committed-save snapshot candidate

This bounded candidate extends the accepted native committed-history owner at
`f20cd277` with a feature-gated same-owner copy boundary. It adds no SAVE
writer, file publication, restore, replay or public generation entry.

`g3r-committed-save-snapshot!` accepts an exact registered G3-T generator and
a caller-owned bytevector of at least 136 bytes. It returns a detached
seven-field vector: policy[6], RNG[4], ordered raw history, origin prompt
length, generated count, manual count and generated-EOS terminal bit. On
success it also copies the 136-byte native record to the destination. The
record is staging data, not G3RCV1 file bytes. The destination is unchanged
on rejection.

Under the shared M3 guard, the Eshkol wrapper checks the canonical 12-slot
generator entry, its idle call links, exact live M3T model entry, raw V256 T1
tokenizer and fingerprint. The private native helper checks the exact native
generator/model pair and both bytevector header spans before acquiring all 14
model parameter pins and an A2 cache read borrow. It checks the committed
cache length, bound parameter image, native policy/RNG/history, and G1 pending
provenance while those leases are held. Both leases end before native scratch
publication. Eshkol compares the six native policy words to its registered
policy (mapping absent EOS to -1), validates the capacity-two partition and
terminal classification, then publishes detached data and copies the scratch
to the original destination. No native pointer escapes.

The focused local runner checks feature-off symbol closure, C owner success
for P1, P2, manual and generated partitions, EOS and no-EOS cases, plus wrong
owner, busy/pending/held leases, G1 provenance, malformed cache length view,
capacity/overlap, allocation and borrow failure cuts. It runs normal, repeat
and ASan/UBSan builds. `scripts/test-g3r-committed-history-owner.sh` remains
green. These C tests use controlled model/A2 stubs. Compiled Eshkol aggregate,
real A2/model lease behavior, current-main union and full supported build
gates remain prerequisites before source acceptance or use by a writer.
The idle generator retains no I1 input owner: a generator-associated I1 borrow
exists only inside an active call/frame, which this helper rejects. An unrelated
detached I1 owner has no same-owner history authority and is outside this copy.
