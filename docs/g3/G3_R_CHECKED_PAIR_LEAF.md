# G3-R checked pair source-private leaf

This bounded candidate composes the accepted G3RCV1 in-memory codec with the
owning C2/SHA receipt in one private source closure. It adds no file-pair
loader, public ABI, candidate ledger, restore, or continuation claim.
`native/g3r_checked_pair_root.esk` is that private composition entrypoint.
`native/g3r_checked_pair_source_closure.txt` pins the receipt predecessor and
the added codec/pair sources for a later compiled source review.

`g3r-checked-pair-admit-internal(record, receipt-borrow)` requires an exact
active receipt borrow. It copies the caller's exact 288 bytes, takes the
receipt's independently measured complete-C2-image SHA, runs the accepted
decoder (including record checksum and all semantic fields), and registers a
fresh opaque shell. The private registry retains the exact receipt and C2
owner identities, the copied record, SHA-256 of all 288 record bytes, and the
copied 32-byte complete-image digest. No caller-owned record or decoded vector
is retained.

`g3r-checked-pair-recheck-internal(pair, receipt)` authenticates the exact
registry shells and C2 owner, non-dead receipt phase, private full-record SHA,
record image-digest field, and receipt image SHA. It can be called after the
receipt borrow ends and again inside a future exclusive restore reservation.
`g3r-checked-pair-copy-fields-internal!(pair, receipt, destination)` publishes
fresh decoded field copies after that recheck. The returned mutable fields
are data for inspection; later replay must use the registered pair shell and
repeat the identity recheck before TR3. A pair tied to a released receipt
rejects. A malformed, changed, or cross-image record rejects before registry
publication.

`g3r-checked-pair-retire-internal!(pair)` requires the exact shell, marks it
dead, zeroes the retained record and both SHA bytevectors, and clears its
receipt/C2 owner references. It is idempotent for that exact shell. The
registry retains only the opaque tombstone identity. Recheck and copyout
reject after retirement. Future candidate retirement must call this operation
after its last pair use.

Admission of record bytes from an actual companion file and a same-reservation
candidate ledger remain separate prerequisites. The focused test uses a
receipt-registry double to exercise mutation and foreign-identity rejection;
it is not evidence of live C2 loading. The Eshkol runtime test must run on a
host with the pinned `eshkol-run` toolchain before this leaf can be accepted.
