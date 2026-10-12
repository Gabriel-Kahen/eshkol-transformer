# E3 N1 synthetic broker candidate

Status: **source-only HOLD**. `scripts/e3_n1_broker.py` is a noninstalled
library with no CLI, service, production pin, or real F layer route. It
implements the separate-UID broker's proposed attempt protocol against tiny
generated source files. It does not establish an actual UID, ACL, ancestor,
mount, `/proc`, inherited-fd, or immutable target boundary.

The broker constructor takes a frozen `Config`: protected parent, reviewed pin
digest, target basename, ordered `Source` descriptors, positive byte/time/RSS
caps, a private receipt key, an independent seal verifier key and its identity,
each source's reviewed resolved path, and the
device/inode/mode chain from filesystem root through the source. The tiny test
fixture freezes these values locally; **no production source identities or
ancestor policy have been reviewed**. `attempt(pin, target)` accepts only that
exact pin and basename. It creates a unique checksummed `HOLD` intent, fsyncs record
and parent before stage creation, opens and compares the stage inode, then
fsyncs an initial stage `HOLD.json` before the first source read. Each source
must be a nonsymlink regular file whose pinned path chain, pre-open `lstat`,
opened `fstat`, and post-read path/fd identity agree; copied bytes must meet
its exact size and SHA-256. A same-byte inode replacement is rejected. The broker retains
the stage fd through no-replace publication, target readback and final sync.

An ordinary copy or no-replace failure keeps the broker-owned partial stage
and checksummed `HOLD.json` with observed sizes and SHA-256 for every present
copied file. Marker/sync/cleanup/retraction faults produce an
`INDETERMINATE` result without a receipt. A failed retraction may leave a
visible broker-owned target. A verified target receives a checksummed
`VERIFIED.json`; only after that and its parent sync does the broker return a
versioned HMAC receipt bound to attempt, pin, target device/inode and exact
file list. The receipt is returned out of band and is **not** stored in the
export; a crash before return has no consumer authority. The protected intent
stays HOLD for recovery and cannot itself promote a target.

`verify_receipt(config, receipt, seal)` requires the HMAC receipt and a seal
authenticated with the verifier key pinned in `Config`. The seal signs its
key identity and the exact receipt body. Configuration validation rejects a
missing verifier or a verifier key equal to the broker receipt key. Verification
then checks the live target inode,
checksummed VERIFIED record and every file's hash/size through nofollow fd
walks. This is a synthetic consumer check. A deployment must separately prove
the seal's read-only custody and key separation; HMAC bytes alone do not make
a writable target immutable. There is no automated recovery promotion.

`scripts/e3_n1_reconcile.py` adds a read-only synthetic crash check.
`reconcile(config, attempt, receipt=None, seal=None)` returns
`Reconciliation(state, reason, attempt)`, where `state` is `HOLD` or `MATCHED`.
It checks the protected intent, named stage/target inode at open, checksummed
HOLD/VERIFIED markers, complete ordered source hashes and target layout, and
duplicate VERIFIED targets for the attempt. `MATCHED` requires the out-of-band
broker HMAC receipt and separately keyed matching seal. A reconciliation caller
cannot choose a seal key; only the reviewed frozen config identifies the trusted
verifier. A seal made with the broker key, a different seal identity, or a config
that reuses the broker key remains HOLD. Missing or unreadable
evidence remains HOLD. It never writes, repairs, issues a receipt, or promotes
a crash state from marker bytes alone. The stage's original inode is not
durably pinned in the intent, so a stage replacement cannot be proven from
this record alone; such an attempt remains HOLD regardless. This source does
not establish external receipt issuance, durable receipt custody, separate UID
or ACLs, real release pins, or independent read-only seal custody. `MATCHED`
is only a synthetic consistency result, not N1 admission.

Generated tiny fixtures test hard exits at 15 publication checkpoints,
including after readable VERIFIED bytes and the last parent sync but before
receipt return; missing intent/marker, duplicate target, changed target inode,
and changed file checksums all remain HOLD. No real f31 bytes are read.

## Source-only durable receipt successor (v2)

`scripts/e3_n1_durable.py` leaves the broker and reconciler v1 contracts
unchanged. `release_descriptor(config, receipt_store, image_id, compiler_head,
compiler_tree, source_head, source_tree)` freezes the ordered source paths,
resolved paths, device/inode/mode chain, sizes, SHA-256 values, caps, target,
parent and its ancestor chain, receipt-store absolute path and ancestor chain,
seal key identity, and the five release identifiers. The canonical JSON SHA-256 must
equal `config.pin`. This v2 candidate accepts only explicitly `synthetic:`
identifiers. It cannot accidentally treat an arbitrary 64-hex request pin or
invented f31/compiler identity as an admitted release.

After `Broker.attempt` has returned VERIFIED with its out-of-band receipt,
`issue(config, release, outcome, store)` checks the signed receipt, protected
intent, live target identity, exact marker and ordered file hashes. It uses
attempt-keyed no-replace names in an out-of-export receipt directory. It
writes and fsyncs the receipt file and directory, reads it back, then writes
and fsyncs a commit record bound to that file's SHA-256 and device/inode. A
fresh 32-byte issuance token is returned only after the commit readback and
directory sync. The commit stores only its SHA-256. Pending temp files from
write/sync faults remain quarantine evidence and never count as a receipt.
An uncertain write or sync returns HOLD without a token. No recovery process
manufactures a token from a visible receipt or marker.

`reconcile_durable(config, release, attempt, store, receipt, token, seal_ref)`
is read-only. It requires the exact externally held broker receipt, returned
issuance token, matching durable receipt/commit and a caller-provided path,
SHA-256 and device/inode pin for a separately stored seal. It reads that seal
with nofollow and calls the existing v1 reconciliation, including its HMAC
check against the frozen verifier identity. Missing, stale, replaced or
unreadable evidence remains HOLD. A copied receipt store with a rewritten
commit inode also remains HOLD because the store path and opened directory
inode must match the frozen release descriptor. A visible commit after a
process crash before final sync/return cannot promote without the unreturned
token.

This is a synthetic consistency protocol, not independent seal authority.
`SealRef` can be constructed by the same login, and v1 `Config` still holds
symmetric HMAC verification bytes. No separate UID, read-only seal custody,
protected ancestors, `/proc`/fd isolation, crash-safe external delivery, or
real f31 release descriptor has been established. A production successor
needs those independently reviewed boundaries and authentic image/compiler/
source identities before N1/M1 admission. The generated fixture tests use
tiny files only; no Docker or real OCI layer is read.

Before any real acquisition, independently review this source and tests;
freeze the exact release config, source device/inode, resolved path, ancestor
policy and bounds; deploy a
broker under a distinct UID; prove parent and all writable ancestors reject
caller rename/unlink/write, including ACL and mount behavior; prove `/proc`
and inherited fd isolation; and review receipt key custody plus an independent
read-only export seal. The currently available login has UID 1000 and no
passwordless elevation, so the cross-UID gate is untested. Exporter D, real
f31 layers, capture, N1/M1 and the numerical gate remain HOLD.
