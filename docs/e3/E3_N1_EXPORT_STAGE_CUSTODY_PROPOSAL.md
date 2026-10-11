# E3 N1 export stage custody proposal

Status: **design candidate, HOLD**. This does not release the inert D source,
install `RELEASED_PIN_SHA256`, authorize a layer read, or change the exact-input
plan. It addresses the independent D feasibility HOLD at
`e3-n1-export-acquisition-feasibility-independent-d-a416db2e-20261011-a`
(seal SHA-256 `d56d1e57319adcfb925fddedf0cdda88b30829e2a4308959385f230abf930f93`).

## Authority boundary

D creates a directory by `mkdirat` and subsequently opens it by name. With an
unrestricted same-UID writer in the parent, the original inode can be moved
between those calls and an empty replacement can be removed instead. A held
directory fd protects only the interval *after* its successful open. Random
names, `O_NOFOLLOW`, `0700`, inode checks, and rollback do not close that gap.
The same writer can later remove a HOLD marker. D therefore does not meet the
plan's unconditional marked-partial-failure requirement.

The proposed minimum authority change is a **separate-UID output broker**
which exclusively owns the export parent, stage directories, attempt records,
and final target. The export parent and every writable ancestor must deny
rename, unlink, and write operations to the acquisition caller and all other
untrusted same-UID processes. The broker must run with a distinct UID, no
shared writable group/ACL, no inherited stage or parent fd in the caller, and
no caller access to broker `/proc` fds or process memory. A private `0700`
directory alone is insufficient when the broker shares the caller's UID.
The broker must not expose a filesystem write capability to the caller.

The broker itself should perform the existing bounded source validation and
copy, with narrowly granted read access to the pinned sources. If a caller
passes source fds instead, that is a separate protocol requiring proof of
source path, inode, namespace, and pre/post identity against the reviewed pin;
it is not part of this proposal. Neither a caller-supplied stage fd nor a
caller-writable stage satisfies this custody boundary. The broker accepts only
an exact reviewed pin digest and target basename; the destination parent,
limits, and expected F descriptors remain frozen by the reviewed release.

## Attempt protocol and evidence

1. Before `mkdirat`, the broker creates a unique attempt ID and durably writes
   a protected, versioned `HOLD` intent record containing the pin digest,
   target basename, stage basename, and reason `in-progress`. Sync the record
   and its parent. If this fails, create no stage and return HOLD. The record
   is authoritative for admission; it is not an OCI export.
2. Under the protected parent fd, create the stage, open it with
   `O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`, and compare the named inode to `fstat`.
   Keep this fd until the attempt is complete. Write and sync an initial
   `HOLD.json` in the stage before copying bytes. Do not start a layer read
   until both the protected intent and initial marker are durable. If opening
   or marking the stage fails, attempt cleanup through the parent fd and keep
   the intent at HOLD. Any remaining stage is quarantined under the broker,
   never eligible for publication or reuse.
3. Copy and hash against the exact ordered descriptors using broker-owned
   regular files and existing caps. On a failure, preserve partial files,
   replace the initial marker with a checked final HOLD record and checksum
   list when the medium permits, sync the stage and parent, and keep the
   protected intent at HOLD. A marker or sync failure is an explicit
   `custody-failure` with an INDETERMINATE, quarantined outcome. The protected
   intent and inaccessible stage remain for recovery; it is not reported as a
   successfully sealed partial stage.
4. After all bytes, identities, limits, and stage syncs pass, use
   `renameat2(RENAME_NOREPLACE)` under the retained parent fd. Post-rename
   checks and parent sync remain fallible. Only a successfully checked and
   synced target may receive a final protected VERIFIED record. The broker
   issues an authenticated completion receipt, bound to the attempt ID,
   target inode, pin digest, and recorded file hashes, only after the final
   record and its parent have synced successfully. A readable VERIFIED record
   whose write, rename, or sync returned an error grants no authority. An
   error triggers retraction and HOLD while the broker still owns the parent.
   If retraction or a later marker write fails, the target remains
   quarantined. A consumer requires both that receipt and a separately
   verified read-only export seal; directory presence, `ACQUISITION.json`, or
   readable broker-record bytes alone are not proof. Receipt authentication
   and seal verification are part of the broker implementation review.
5. A separate read-only recovery/reconciliation gate compares protected
   attempt records, stage and target inode identities, markers, and checksums.
   It never promotes a HOLD attempt automatically. A crash before the receipt
   leaves the attempt quarantined even if final record bytes are visible. A
   crash after the receipt does not revoke an independently verified seal.
   Missing receipts or records, duplicate targets, mismatched inodes, failed
   syncs, and inaccessible stages remain HOLD for admission until reviewed.

This protocol can guarantee that untrusted same-UID callers cannot move or
alter the broker's stage and that failed attempts cannot acquire final VERIFIED
authority. It cannot guarantee a `HOLD.json` beside every partial file after
arbitrary filesystem write, sync, or cleanup failure, nor guarantee physical
removal of a named target if the filesystem refuses retraction. The exact-input
plan currently requires both a marked partial and no published export on every
failure. Before implementation or release, ROOT must resolve that literal
requirement: either accept the protected intent plus quarantine as the
failure artifact for storage-failure cases, or provide a stronger storage and
publication mechanism with evidence for those cases. No source-only D patch
should claim the existing requirement is met.

## Review gate without real f31 layers

Use generated tiny OCI blobs and fault injection. Required independent
negatives include pre-open move/replacement attempts by an actual caller-UID
process, attempted marker deletion and parent rename, failed stage open,
failed cleanup, failed marker write and sync, crash at each transaction
boundary, late copy/hash/cap failure, target move/retraction failure, final
record write/rename/sync failure, crashes before and after the receipt,
independent-seal failure, and duplicate target. Check the protected record,
receipt, and stage/target inode after each case, and show that no failure has
final VERIFIED authority. Verify
actual UID, directory ACLs, ancestor permissions, `/proc` isolation, and fd
inheritance in the test environment; mocks alone do not prove the boundary.
Test a successful tiny export and independent seal. No real f31 layer, Docker,
guest execution, capture, or production pin is needed for this gate.

Next gate: review this authority and failure-artifact contract, then implement
and independently test the broker in a new isolated workstream. D stays HOLD.
