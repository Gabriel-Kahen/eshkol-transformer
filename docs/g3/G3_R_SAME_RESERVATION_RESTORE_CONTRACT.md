# G3-R same-reservation input and restore contract

**Proposed source-private transaction; no implementation or continuation admission.**
This refines the three-argument ownership sketch in the
[receipt/M3T boundary](G3_R_C2_RECEIPT_M3T_RESTORE_CONTRACT.md). That sketch
cannot authenticate candidate inputs: a caller-side inspection consumes the
receipt's only C2 borrow before TR3 starts its own borrow. The fixed scope is
`diagnostic-c2`, CPU f32, batch one, raw V256 and context two. This contract
adds no public operation, checkpoint field, C4 conversion or replay.

## Authority and transaction input

One same-aggregate compositor operation takes an **exact live receipt**, an
exact checked G3RCV1 pair record, and an **unpublished candidate ledger**. Its
return is only a status to that ledger; it never returns the C2 owner, a borrow,
or a publishable model. These are proposed roles, not currently callable
signatures. The pair record must have independently validated all 288 bytes,
its own SHA and the 32-byte complete-C2-image digest. Compare that digest
with the receipt's independently copied digest under its borrow. The current
receipt tombstones its stage on publication and exposes no stage-identity
comparison; stronger pair/receipt authority needs a future checked-pair
contract. The ledger must retain exact candidate
M3T, O2, TR3, X1, T1 and D2 identities, ownership flags for input/initializer,
and all children already created. A bare digest, caller-built vector, trainer
shell, or separate precheck is not authority. The checked pair parser and
candidate ledger/retirement are still prerequisites, so no ABI or function
name is installed by this document. The checked pair must be sealed and
immutable under its own registry, or every authenticated profile, policy,
history and G3-S RNG field must be copied into ledger-owned storage during
this reservation and its exact pair identity rechecked before TR3. A mutable
caller-owned record cannot be the later replay authority.

The compositor must authenticate the receipt through its registry, its exact
record-slot-2 C2 owner and SHA, and the ledger/trainer through their own
registries in the **same compiled source closure**. It must preallocate its
exception/cleanup state and reserve handler capacity before changing phases.
Only the ledger may designate the trainer and its bound candidate components;
the caller cannot swap a trainer or supply copied identity fields. The ledger
enters an exclusive `verifying` state before receipt inspection; its own
operations, release and publication reject while verifying/restoring. Existing
TR3 admission remains mandatory; the new ledger phase is proposed, not an
existing TR3 lease mode. No callback may acquire a second receipt borrow or
release this receipt during the transaction.

## Evidence while the sole receipt borrow is active

Begin one receipt borrow; obtain its C2 borrow through
`g3r-c2-receipt-borrow-c2-internal`, its copied SHA through the receipt API,
and an authenticated C2 projection through
`c2-training-state-borrow-projection-internal`. Compare the SHA with the
checked pair's image digest **inside this reservation**. The C2 projection
contains P1, O2, controls, topology and O2 projection. Its controls are
`[rng, tokenizer, D2 cursor, resolved X1, counters]`; C2 has already checked
the cursor pair and canonical X1 against tokenizer vocabulary, CPU/f32 and
seed (`native/c2_training_state_extension.esk:151-229,407-423`). Inspect the
source P1/O2 schema through their C2-owned tokens while the borrow is valid.
The O2 options require the existing authenticated `o2-require-c2-owned-state`
route; raw O2 record fields are not a substitute. Copy the exact typed bits
of all 14 source parameter values here for the later post-restore comparison.
That bounded copyout itself needs review. Every bytevector or mutable leaf
needed later must be copied into private ledger-owned storage **before**
ending the borrow. No projection header,
borrow, owner token, P1/O2 child, tensor access or pointer survives as usable
authority after it ends.

The same operation compares these authenticated source fields to the exact
candidate identities, rather than trusting a caller assertion:

* G3RCV1 profile `1`, algorithm, raw-V256 fingerprint and image digest;
  C2 tokenizer fingerprint equals the pair's fingerprint and the registered
  candidate T1 fingerprint. C2 vocabulary `256` equals the candidate T1
  vocabulary; the pair carries no vocabulary-size field. Check the T1
  identity in its own registry, not only by matching text.
* C2 resolved X1 canonical bytes and fingerprint equal the candidate's
  registered resolved X1; CPU, f32, context and seed satisfy the fixed profile.
  Exact X1 fields must match, while any separate caller resource limit may
  be stricter than C2's bound but never wider. The C2
  training RNG is **not** the G3-S RNG in the pair.
* The C2 current/epoch D2 cursor pair must be internally valid and carry the
  same immutable corpus/options/tokenizer identity as the ledger's exact open
  candidate dataset. Enroll that dataset in the ledger before comparison.
  The candidate's mutable next ordinal need not equal the saved ordinal;
  TR3's D2 restore checks immutable identity and installs the saved position.
  Matching cursor bytes alone does not authenticate the D2 owner.
* C2 P1/O2 topology is exactly the 14 ordered M3T paths and shapes, 14
  distinct primaries, one exact tied token-embedding alias, CPU f32 dense
  storage and the supported O2 configuration. The candidate's M3T/O2 graph
  must independently match the same profile. In particular, C4's `[4,4]`
  position table rejects before TR3; the required M3T table is `[2,4]`.

TR3 already rechecks source topology and compares its detached X1/T1 controls
to the trainer's X1/T1 record, then prepares D2 restore
(`native/tr3_c_joint_restore_extension.esk:230-280,292-360,363-432`). These
checks are retained but do not replace the pair, profile, exact candidate
registry or resource-limit comparison above. The current code does not supply
a single authenticated candidate ledger or a checked G3RCV1 pair input;
implementing this operation depends on those contracts being accepted.

## Borrow handoff and receiver mutation

After all source comparisons, end the receipt borrow **once**. Its current
implementation calls C2 end first and changes receipt slots 4 and 1 only
after C2 returns (`native/g3r_c2_sha_receipt_extension.esk:249-255`). On a
normal return, publish `restoring` in the exact receipt record immediately by
assignment, before any fallible call or callback. No other agent can acquire
that receipt during the gap: the receipt still has its active-borrow phase
until this store, and the interval contains only primitive assignments. The
receipt remains the sole owner of the exact C2 shell throughout.

With the receipt reserved and the ledger still exclusive, reauthenticate the
candidate ledger and exact trainer/component identities against the private
copies. This recheck is necessary because C2 borrow-end may call a provider.
Any failure here releases the reservation to `live` only after C2 is known
live and no TR3 restore has begun. Pass the lexically retained exact C2 owner
from receipt slot 2 to `tr3-c-joint-restore-internal!`; TR3 then enters its own
C2 borrow and receiver transaction. Its own source/control/D2 checks and
abort are not bypassed. A normal return must leave C2 live, TR3 trainer idle
and no outstanding source access before the receipt becomes `live`. A
recoverable TR3 failure may make the receipt live only after TR3's abort has
closed its C2 borrow and returned the candidate to an authenticated idle
state. The first-error rule needs a composed TR3 cleanup witness: the present
TR3 guard may replace its caught error with an internal cleanup error
(`native/tr3_c_joint_restore_extension.esk:619-635`). The compositor cannot
claim to preserve the first error in that case. Uncertain C2 or trainer state,
cleanup failure, or failure after TR3's irreversible commit tail exits 134. The
receipt's original owner is released exactly once by the candidate ledger
after all needed C2 bit checks, never silently consumed by this operation.

**Certainty rule for receipt borrow-end failure:** a raised end is not proof
that the borrow remains active. C2 revokes P1, then O2, then clears its owner
backlink and publishes `live` before the receipt updates its own slots
(`native/c2_training_state_extension.esk:477-512`). The current receipt API
has no authenticated complete-consumption witness or safe one-shot recovery
for every intermediate cut. Therefore **every borrow-end exception in this
proposed gate is fail-stop (134)**, including a wrapper raising after C2
consumed the borrow. Never call end twice, set the receipt `live` on a guess,
retry TR3, or treat an internal cleanup error as the original recoverable
failure. A future narrower recovery rule requires a reviewed same-registry
witness proving the exact C2, P1 and O2 phases and spent/intact authority;
it cannot be inferred from the receipt phase alone.

After TR3, a separate exact typed copyout must compare all 14 restored f32
value bit images to the source bits copied under the original receipt borrow,
preserve gradient
values/counts and prove the tied handle/storage alias; then release the idle
trainer and switch the model to eval. Equal-seed initialization or matching
logits is insufficient. These are acceptance checks, not operations supplied
by this contract.

## Deterministic negative and completion gates

Source tests should assert one receipt borrow at a time, exact same owner
through reservation/TR3, no raw-owner escape or second LOAD, and one
assignment-only borrow-end-to-`restoring` gap. A fresh-process producer and
consumer must use a real C2 image and G3RCV1 companion from the same staged
bytes. Inject the following before any publication:

| Cut or substitution | Required result |
|---|---|
| Wrong 32-byte digest; short/long digest; forged/dead/cross-aggregate receipt or pair | Reject before TR3; no owner escape; valid retry from an intact receipt where applicable. |
| Mismatched X1 canonical bytes/fingerprint, T1 registry/fingerprint/vocabulary, immutable D2 corpus/options identity, malformed or cross-paired current/epoch cursors, limits or CPU/f32 capability | Reject under the active receipt borrow; leave candidate and existing receiver unchanged. |
| C4 `[4,4]` position table, reordered/duplicate 14 paths, broken tied alias, O2 option mismatch | Reject before restore; no relabeling as M3T. |
| Forged, stale, busy or borrowed trainer/dataset; nested callback borrow/release/restore, candidate swap or checked-pair field mutation | Reject or preserve the exclusive reservation; no second C2 borrow or changed pair authority. |
| Fault before receipt borrow, during field copy, or before C2 end | End the borrow exactly once only if acquired; leave receipt releasable after successful cleanup. Candidate teardown/reusable retry waits for retirement. |
| Fault **inside** receipt borrow-end, before or after C2 consumption | Fail-stop subprocess exit 134; no retry, second end or `live` receipt publication. |
| TR3 stage fault followed by successful abort before commit | Receipt may become live only with authenticated idle trainer and live C2; candidate teardown/reusable retry waits for retirement. |
| TR3 abort cleanup fault | Exit 134 unless a separately reviewed exact cleanup witness proves complete rollback. |
| TR3 commit-tail or later cleanup defect | Fail-stop subprocess exit 134; never publish candidate. |

On successful source/restore gate, compare all 14 binary32 value images,
their path/shape/dtype/device/layout, tied alias, unchanged gradients/counts
and eval mode; release the receipt once and verify its SHA zeroed. Repeat
normal and sanitizer runs plus feature-off and source/symbol-closure checks.
Until the [fresh-candidate retirement](G3_R_FRESH_CANDIDATE_RETIREMENT_CONTRACT.md)
and its P1/I2 and O2/M3T prerequisites are implemented, failures after
candidate construction cannot be called failure-atomic or reusable: idle
trainer unenrollment leaves model and optimizer payloads live. The G3-T
generator retire path is a further replay prerequisite. No full G3-R
continuation claim follows from this data-only gate.
