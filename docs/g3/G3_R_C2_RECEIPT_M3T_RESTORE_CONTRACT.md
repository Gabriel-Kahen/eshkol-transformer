# G3-R C2 receipt to fresh M3T restore boundary

**Proposed private contract and current blocker. No receipt-driven restore is
implemented or accepted here.** Scope is the fixed CPU-f32 `diagnostic-c2`,
batch-one, raw-V256, context-two candidate in the
[data-only continuation contract](G3_R_DATA_ONLY_CONTINUATION_CONTRACT.md).
There is no public API, C2 field, C4 conversion, or generation replay.

## Exact ownership boundary

The owning receipt stores one detached C2 owner and the ordinary SHA-256 of
the same authoritative second LOAD image
(`native/g3r_c2_sha_receipt_extension.esk:120-211`). Its borrow entry returns
an authenticated **C2 borrow**, not the owner; release rejects a borrowed
receipt (`:226-287`). TR3 joint restore accepts a detached C2 **owner** and
begins its own borrow (`native/tr3_c_joint_restore_extension.esk:230-239,613-641`).
Passing a receipt shell or its borrow to `trainer-load-state!` is invalid;
holding a receipt borrow while TR3 enters makes that owner busy. The test-only
CLI3/G3-G path instead uses public C2 LOAD, fresh M3T, TR3 restore, trainer
release and model eval (`tests/cli3_generate/public_runtime.esk:106-149`;
`native/cli3_generate_extension.esk:59-81`). It proves one-registry live-model
reachability, not receipt consumption or same-image SHA binding.

The smallest new feature-local bridge must authenticate the exact live receipt
and expected G3RCV1 digest under its borrow, inspect/copy the C2 controls and
fixed schema, then **end that borrow before TR3**. It must protect the same
receipt record against release, reuse and reentry while passing its exact
owned C2 shell to the existing joint restore. TR3 retains its normal C2 borrow
and transaction. On success or recoverable failure, the bridge returns the
receipt to a releasable state; a defect after irreversible mutation fails
stop. This is a proposed source-private seam, not an existing callable API.
No reopened path, second LOAD, unattached SHA copy, raw owner escape, or
public LOAD/TR3 ABI change is allowed.

## Proposed closed ownership bridge

The existing interfaces do **not** suffice as a caller composition API. The
receipt's `g3r-c2-receipt-borrow-c2-internal` returns its active C2 borrow,
while `tr3-c-joint-restore-internal!` immediately calls
`c2-training-state-borrow-begin-internal` on its `source` argument. C2 permits
one active borrow: it changes the exact registered owner's phase from `live`
to that borrow before provider callbacks and rejects another borrow or release
until the first ends (`native/c2_training_state_extension.esk:346-405,477-512,
632-643`). The private receipt keeps the exact owner in record slot 2 and
rejects another receipt borrow or release while its phase is not `live`
(`native/g3r_c2_sha_receipt_extension.esk:167-170,226-266`). Thus a direct
call with a receipt shell or its borrow is invalid, and extracting slot 2 in a
caller would bypass the receipt's ownership authority.

Propose one feature-local, unexported **ownership-only** operation, with this
bounded signature sketch to review before implementation:

```scheme
(g3r-c2-receipt-restore-m3t-internal!
  receipt expected-image-sha trainer) -> #t
```

`receipt` must be an exact live receipt from the same owning aggregate;
`expected-image-sha` is the exact 32-byte digest from an already validated
G3RCV1 companion, not a standalone authority; `trainer` is a fresh,
unpublished, idle M3T/TR3 candidate. The bridge checks the digest against a
copy obtained under its own exact receipt borrow. It may copy C2 controls
while that borrow is active, but may not retain a projection, owner token,
P1/O2 child, or borrowed pointer after the borrow ends. The three arguments
cannot establish the candidate's X1/T1/D2/schema/capability identity against
the receipt; therefore this bridge alone cannot admit or publish a resumed
candidate. The full candidate compositor needs a separately reviewed exact
input/witness contract that performs those comparisons under this same
reserved receipt authority before TR3 enters. No caller-side precheck followed
by this three-argument call is claimed to close that identity gap.

Preallocate the exception/cleanup record and reserve the required handler
capacity before changing receipt phase. After the last successful C2 borrow
end, publish a private `restoring` phase into the exact receipt record with an
assignment-only transition; no fallible call or callback may occur between
ending that borrow and publishing the phase. Only then pass the lexically read
record-slot-2 **owner shell** to the existing joint restore. The bridge never
transfers shell ownership: the receipt remains the sole owner before, during
and after TR3, while TR3 owns its temporary C2 borrow and receiver
transaction (`native/tr3_c_joint_restore_extension.esk:230-280,613-641`).
Receipt borrow/release and reentrant restore must reject during `restoring`.
No new public getter, second C2 LOAD, path reopen, or raw-shell result is
permitted. Same-aggregate feature-on source closure must prove that the
bridge sees these exact registries and TR3 implementation.

On successful TR3 return, its C2 borrow is closed and trainer lease is idle;
the bridge changes the receipt back to `live` by assignment and returns `#t`.
On a recoverable TR3 failure, TR3 first performs its own abort and closes its
C2 borrow (`native/tr3_c_joint_restore_extension.esk:457-559,619-635`); the
bridge then restores the receipt to `live` and rethrows the original error.
The caller's candidate ledger retains the receipt and releases it exactly
once through `g3r-c2-receipt-release-internal!` after any required C2-bit
comparison, on both success and failure. That release uses the receipt's
existing C2 release record and zeros the copied SHA
(`native/g3r_c2_sha_receipt_extension.esk:257-288`). A pre-admission failure
may leave the receipt live for an explicit caller retry; after the caller
abandons the candidate, its ledger releases it. The bridge must never
silently consume a receipt that the ledger still owns.

If a borrow-end or TR3 cleanup raises after ownership may have been consumed,
the bridge must determine exact C2 and trainer phases before treating the
failure as recoverable. An uncertain phase, second cleanup defect or failure
after TR3's irreversible tail is fail-stop (exit 134); it may not publish a
retryable receipt, call borrow-end twice on spent authority, or replace the
first error with a cleanup error. The existing receipt borrow-end updates its
own phase only after C2 end returns (`native/g3r_c2_sha_receipt_extension.esk:
250-255`), so its post-consumption failure case needs an explicit reviewed
certainty witness before implementation. A simple outer `guard` that blindly
sets `live` is insufficient.

## Fresh candidate and bit identity

Before TR3 restore, compare the companion digest, profile, canonical
X1/config fingerprint, T1 fingerprint, D2 corpus/cursor, resource limits and
CPU-f32 capability with the receipt's C2 borrow and same-aggregate inputs.
Open any candidate D2 dataset and enroll it in the cleanup ledger before this
comparison. The fresh candidate owns its new M3T model, 14-handle P1 tree,
O2 optimizer and TR3 trainer; its ledger separately records whether the T1
binding, D2 dataset and input initializer are candidate-owned or borrowed
from the caller. The initializer and equal seed are scaffolding, not proof of
restored weights. Restore into this trainer only, release its idle lease, then
call `module-eval!`. Do not touch the caller's model, optimizer, generator,
cache, RNG or old output.

The 14 primary paths and shapes are ordered exactly as follows
(`native/m3t_transport_extension.esk:307-320,365-393`;
`native/tr3_c_joint_restore_extension.esk:7-27,112-178`):

| Index | Path | Shape |
|---:|---|---|
| 0 | `blocks/0/attention/key/weight` | `[4,4]` |
| 1 | `blocks/0/attention/output/weight` | `[4,4]` |
| 2 | `blocks/0/attention/query/weight` | `[4,4]` |
| 3 | `blocks/0/attention/value/weight` | `[4,4]` |
| 4 | `blocks/0/ffn/down/weight` | `[4,8]` |
| 5 | `blocks/0/ffn/up/weight` | `[8,4]` |
| 6 | `blocks/0/norm1/bias` | `[4]` |
| 7 | `blocks/0/norm1/weight` | `[4]` |
| 8 | `blocks/0/norm2/bias` | `[4]` |
| 9 | `blocks/0/norm2/weight` | `[4]` |
| 10 | `head/weight` | `[256,4]` |
| 11 | `norm_final/bias` | `[4]` |
| 12 | `norm_final/weight` | `[4]` |
| 13 | `position_embedding/weight` | `[2,4]` |

All 14 primary handles must be distinct. The fifteenth path,
`token_embedding/weight`, must be the **same handle and storage** as index 10,
not a copy. Every primary requires the exact authenticated path, rank, shape,
f32 dtype, CPU device and dense row-major layout. TR3 already checks source
topology and stages 14 C2-owned clones
(`native/tr3_c_joint_restore_extension.esk:112-205`); M3T validates its
constructed topology (`native/m3t_transport_extension.esk:365-393,439-453`).
The receipt-driven gate must additionally compare each restored value's exact
binary32 bits to its authenticated C2 primary and prove gradient values/counts
unchanged. Logits and equal-seed initialization are insufficient. An exact
typed copyout/comparison witness needs review before this claim.

Index 13 is a hard profile boundary: C4 owns `[4,4]`
(`src/eshkol_transformer/g3c4_model_owner.c:72-74`), while M3T/TR3 require
`[2,4]`. C2 storage admits both shapes. Reject a C4 image as schema/profile
mismatch before restore; never relabel its owner as M3T.

## Cleanup and next gate

Preallocate a candidate ledger and enroll each returned child before the next
fallible call. Releasing an idle TR3 trainer does not retire the sealed model
or O2 native payloads. The required unpublished teardown is specified in
[fresh-candidate retirement](G3_R_FRESH_CANDIDATE_RETIREMENT_CONTRACT.md):
close child leases; preflight all owners; retire O2 moments; revoke sealed
P1/I2 handles; retire 14 M3T parameters; tombstone M3T and exclusively owned
initializers. The G3-T live-generator retire path is a separate replay cleanup
prerequisite. These paths remain pending in
[the roadmap](../ROADMAP.md) and the linked retirement contracts.
No restore candidate can be published or claim failure-atomic rollback across
post-construction cuts until teardown exists. Recoverable failures must leave
the existing receiver and outputs unchanged, retire its C2 owner once,
and permit retry.

The next gate is a reviewed private receipt-to-TR3 bridge and complete
candidate retirement. A same-registry, fresh-process producer/consumer test
must use a real C2 image plus G3RCV1 digest, compare all 14 value bit images
and tied identity, check unchanged gradients/counts and eval mode, and exercise
wrong-schema, forged, busy, borrowed and corrupt negatives with rollback/retry
under allocation/provider cuts. Run supported normal, repeat, sanitizer and
source/symbol-closure gates. Replay, pair publication and full G3-R follow.

For the bounded ownership bridge gate, first prove exact pair SHA and one
same-registry receipt with a real C2 image; compare the digest before one
fresh M3T restore. Assert one C2 borrow at a time, no borrow at return,
receipt still owning the same shell, and exact 14 value bits/tied alias with
unchanged gradients. Negative cases must include wrong 32-byte digest,
malformed digest length, forged/cross-aggregate/dead receipt, forged or busy
trainer, receipt already borrowed or restoring, direct TR3 call with a receipt
or C2 borrow, nested callback attempts to release/borrow/restore, wrong C4
position shape, and fault cuts before/after the receipt reservation and at
TR3 stage/abort. Each recoverable case preserves the existing caller and
permits a valid retry; each receipt is then released once and its SHA cleared.
Test the uncertain borrow-end and post-commit cleanup paths as explicit
fail-stop subprocesses. Until complete candidate retirement is accepted,
these remain contract/source checks rather than a claimed reusable compiled
candidate gate: an idle trainer release alone leaves model/O2 payloads live.
The later full-composition gate adds same-reservation control/schema
authentication before restore; no result of the ownership bridge by itself
proves that authentication.
