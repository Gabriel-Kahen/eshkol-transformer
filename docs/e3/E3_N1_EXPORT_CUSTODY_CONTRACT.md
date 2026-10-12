# E3 N1 export custody contract candidate

Status: **proposed contract correction; HOLD**. This amends only the failure
artifact and publication authority in the [exact-input plan](E3_N1_EXACT_INPUT_PLAN.md).
It does not accept exporter D `a416db2e`, release its absent production pin,
authorize a real layer read, or admit N1/M1. The independent D feasibility
audit is sealed at
`e3-n1-export-acquisition-feasibility-independent-d-a416db2e-20261011-a`
(`d56d1e57319adcfb925fddedf0cdda88b30829e2a4308959385f230abf930f93`).
The earlier protected-stage proposal is `05c56530a2d39a06fcd1dafb1e88c9b75faede2d`.

## Authority boundary

D creates a directory by `mkdirat` and subsequently opens it by name. With an
unrestricted same-UID writer in the parent, the original inode can be moved
between those calls and an empty replacement can be removed instead. A held
directory fd protects only the interval *after* its successful open. Random
names, `O_NOFOLLOW`, `0700`, inode checks, and rollback do not close that gap.
The same writer can later remove a HOLD marker. D therefore does not meet the
plan's unconditional marked-partial-failure requirement.
Arbitrary write, sync, and cleanup failure also precludes a literal universal
marker or physical absence of a target after every failure. That wording came
from the internal exact-input plan at `27ea4402`; it is not stated in the
original E3 §9 evaluation proposal. This candidate makes the achievable
admission guarantee explicit while preserving the other review gates.

The proposed minimum authority change is a **separate-UID output broker**
which exclusively owns the export parent, stage directories, attempt records,
and final target. The export parent and every writable ancestor must deny
rename, unlink, and write operations to the acquisition caller and all other
untrusted same-UID processes. The broker must run with a distinct UID, no
shared writable group/ACL, no inherited stage or parent fd in the caller, and
no caller access to broker `/proc` fds or process memory. A private `0700`
directory alone is insufficient when the broker shares the caller's UID.
The broker must not expose a filesystem write capability to the caller.
Review its actual UID, ACLs, mount namespace, ancestor permissions, and fd
isolation before relying on this boundary.

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
   is the durable failure artifact even when stage-local evidence cannot be
   written; it is not an OCI export.
2. Under the protected parent fd, create the stage, open it with
   `O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`, and compare the named inode to `fstat`.
   Keep this fd until the attempt is complete. Write and sync an initial
   `HOLD.json` in the stage before copying bytes. Do not start a layer read
   until both the protected intent and initial marker are durable. If opening
   or marking the stage fails, attempt cleanup through the parent fd and keep
   the intent at HOLD. Any remaining stage is quarantined under the broker,
   never eligible for admission or reuse. No stage name or fd is reused.
3. Copy and hash against the exact ordered descriptors using broker-owned
   regular files and existing caps. On a failure, preserve partial files,
   replace the initial marker with a checked final HOLD record and checksum
   list when the medium permits, sync the stage and parent, and keep the
   protected intent at HOLD. A marker or sync failure is an explicit
   `custody-failure` with an INDETERMINATE, quarantined outcome. The protected
   intent and inaccessible stage remain for recovery; it is not reported as a
   successfully sealed partial stage. Recovery must account for its inode.
4. After all bytes, identities, limits, and stage syncs pass, use
   `renameat2(RENAME_NOREPLACE)` under the retained parent fd. Post-rename
   checks and parent sync remain fallible. The target can become *visible*
   before those checks finish; visibility is not admission. Only a checked
   and synced target may receive a final protected VERIFIED record. The broker
   issues an authenticated completion receipt, bound to the attempt ID,
   target inode, pin digest, and recorded file hashes, only after the final
   record and its parent have synced successfully. A readable VERIFIED record
   whose write, rename, or sync returned an error grants no authority. An
   error triggers retraction and HOLD while the broker still owns the parent.
   If retraction or a later marker write fails, the target remains
   broker-owned and INDETERMINATE quarantined, without a receipt. A consumer
   requires both that receipt and a separately verified read-only export seal;
   directory presence, `ACQUISITION.json`, or
   readable broker-record bytes alone are not proof. Receipt authentication
   and seal verification are part of the broker implementation review.
5. A separate read-only recovery/reconciliation gate compares protected
   attempt records, stage and target inode identities, markers, and checksums.
   It never promotes an attempt lacking its receipt. A crash before the receipt
   leaves the attempt quarantined even if final record bytes are visible. A
   crash after the receipt does not revoke an independently verified seal.
   Missing receipts or records, duplicate targets, mismatched inodes, failed
   syncs, and inaccessible stages remain HOLD for admission until reviewed.

This protocol gives a fail-closed admission rule: no failed attempt has final
VERIFIED authority. It cannot guarantee a `HOLD.json` beside every partial
file after arbitrary filesystem write, sync, or cleanup failure, nor guarantee
physical removal of a named target if the filesystem refuses retraction. The
corrected plan therefore requires no **admitted** export after failure; a
visible but quarantined target may remain. Receipt authentication, durable
issuance, recovery, and immutable target custody require implementation
review. No source-only D patch can establish this authority.

## Review gate without real f31 layers

Use generated tiny OCI blobs only. In every negative verify the protected
intent, stage/target inode, receipt state, and consumer denial. Before issuance
the receipt must be absent; after issuance a stale receipt must be rejected.
Inability to read evidence is itself HOLD, never inferred success.

| Injected event | Required result |
| --- | --- |
| Intent write or parent sync failure | No stage, no receipt, HOLD. |
| Caller-UID parent/ancestor rename, pre-open stage move/replacement, or marker deletion | Denied by actual authority boundary; otherwise contract failure and HOLD. |
| Stage open, initial marker write/sync, or cleanup failure | No layer read; protected intent HOLD; remaining inode quarantined. |
| Late source drift, wrong hash/size, or time/RSS/disk cap | No receipt; partial bytes quarantined; final HOLD marker if storage permits. |
| Partial marker/checksum write or sync failure | INDETERMINATE quarantine; never claim a sealed partial stage. |
| Duplicate target or no-replace rename failure | No receipt; attempt HOLD; existing target never adopted. |
| After rename: identity/readback, parent sync, final record write/rename/sync, or retraction failure | No receipt; visible target only in broker quarantine. |
| Crash at each boundary before receipt, including after readable VERIFIED bytes | No receipt; recovery cannot promote. |
| Crash after receipt, then changed inode or independent seal failure | Consumer denies; stale receipt cannot be reused. |
| Tiny successful attempt with independent seal | Exact target, receipt, and record identities agree before admission. |

Verify actual UID, ACLs, ancestor permissions, `/proc` isolation, and fd
inheritance with a real separate-UID caller; mocks alone cannot prove custody.
No real f31 layer, Docker, guest execution, capture, compiler ABI guess, or
production release pin belongs to this gate.

Next gate: independently and ROOT-review this contract, then implement and
independently test a new isolated broker. Only after its exact source,
deployment authority, metadata pin, and destination pass separate review may
a first bounded real-layer acquisition be considered. D stays HOLD.
