# TR3-C live trainer snapshot and joint restore contract

Status: **accepted design contract; implementation and acceptance pending**.
Source base: `19f404cf21632944e1f2d5d4959e1240f9a79f5f`.
Issue: [#120](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/120).

The source-grounded mapping from this contract to the accepted D2/I2/O2/C2
interfaces and the pending lease is frozen in
[the joint restore composer implementation plan](TR3_C_JOINT_RESTORE_IMPLEMENTATION_PLAN.md).

This document refines sections 10 and 11 of the TR3 trainer proposal. It adds
no public operation, checkpoint byte, persistence policy, compiler identity,
provider identity, or E3 frame. C2 1.0 remains accepted and complete. The
eventual public spellings remain `trainer-state` and `trainer-load-state!`, but
this contract alone adds no implementation or export. Implementation remains
gated by the companion plan's prerequisites and acceptance evidence.

The selected design deep-stages every detached tensor and control while one
genuine C2 borrow is active, then closes that borrow before preparing any live
receiver mutation. The irreversible tail therefore retains no authority into
the input C2 owner and needs no new C2 sealed-borrow lifecycle.

## 1. Fixed ownership decisions and prerequisites

The following decisions are normative for this contract.

1. C2 remains the sole detached-state policy and ownership lineage. Snapshot
   calls the existing P1 and O2 snapshot operations and the existing authentic
   C2 composer. Restore calls the existing authentic C2 borrow and never
   consumes or adopts the input owner.
2. Restore deep-owns 42 source tensors before ending the C2 borrow: 14 P1 model
   values and 28 O2 moments. No live receiver write occurs while the C2 borrow,
   a P1 state-tensor access, or an O2 detached-state access remains active.
3. One I2 transaction covers all 42 live destinations. Public
   `module-load-state-dict!`, `optimizer-load-state!`, and
   `token-dataset-seek!` are not called or sequenced.
4. O2 owns the only optimizer exclusion. TR3-O [#119](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/119)
   and TR3-C may use distinct typed plans, but both use the existing optimizer
   busy field and one exact active-operation backreference. Update-clear and
   restore preparations mutually reject. C2 and TR3-C add no second optimizer
   receiver registry or competing control owner.
5. The D2 current cursor targets the live D2 iterator. Epoch-start cursor bytes
   are trainer-owned. D2 does not acquire a second epoch-start field.
6. Canonical train modes are validated read-only. Restore writes no mode and
   uses no E3 frame or E3 evaluation restore authority. A fixed trainer-private
   checker walks the trainer's retained 17 exact module identities and requires
   `module-mode-internal == 'train` for each; it exposes no module or mode setter.
7. SHARED-R2 [#114](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/114)
   is required for public C2 reconstruction of the five fixed rank-two shapes.
   It is not live-restore authority and its provider 1.1 identity/evidence must
   not be changed by TR3-C.
8. C2 guard-order repair [#121](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/121)
   is a hard final-acceptance and supported-adoption prerequisite. Its unchanged
   API permits bounded implementation after root freezes this contract, but no
   snapshot/restore acceptance claim may precede its merged runtime evidence.
   #121 owns the existing borrow-begin, compose, SAVE, LOAD/release, and upstream
   P1 rollback-handler ordering fixes. TR3-C does not duplicate those changes or
   add a sealed C2 borrow API.

## 2. Fixed profile and staged data

The receiver and detached state must describe this exact canonical topology.
The 15th logical path, `token_embedding/weight`, aliases `head/weight`; it does
not create a fifteenth destination.

| Index | Canonical unique path | Shape | Bytes |
|---:|---|---|---:|
| 0 | `blocks/0/attention/key/weight` | `[4,4]` | 64 |
| 1 | `blocks/0/attention/output/weight` | `[4,4]` | 64 |
| 2 | `blocks/0/attention/query/weight` | `[4,4]` | 64 |
| 3 | `blocks/0/attention/value/weight` | `[4,4]` | 64 |
| 4 | `blocks/0/ffn/down/weight` | `[4,8]` | 128 |
| 5 | `blocks/0/ffn/up/weight` | `[8,4]` | 128 |
| 6 | `blocks/0/norm1/bias` | `[4]` | 16 |
| 7 | `blocks/0/norm1/weight` | `[4]` | 16 |
| 8 | `blocks/0/norm2/bias` | `[4]` | 16 |
| 9 | `blocks/0/norm2/weight` | `[4]` | 16 |
| 10 | `head/weight` | `[256,4]` | 4096 |
| 11 | `norm_final/bias` | `[4]` | 16 |
| 12 | `norm_final/weight` | `[4]` | 16 |
| 13 | `position_embedding/weight` | `[2,4]` | 32 |

The model total is 1,184 binary32 values / 4,736 bytes. O2 contributes an
`exp_avg` and `exp_avg_sq` tensor with the same shape for each row: 28 tensors /
9,472 bytes. The complete restore stage is exactly 42 tensors / 14,208 bytes.
Every tensor is CPU, f32, dense row-major, zero-offset, and owns distinct source
storage. All 42 live destinations are distinct from every staged source and
from each other. The tie is represented by one destination at index 10, never
by duplicate writes.

## 3. Operations and status mapping

All private source operations use the eventual public operation symbol
`trainer-state` or `trainer-load-state!`; callers do not supply an operation,
category, callback, selector, pointer, index, or registry token.

The contract's logical source-private responsibilities are:

```scheme
(tr3-c-trainer-state-internal trainer result-cell)
(tr3-c-trainer-load-state-internal! trainer detached-state result-cell)

(tr3-c-restore-stage-p1-internal! plan c2-borrow c2-projection)
(tr3-c-restore-stage-controls-internal! plan c2-projection)
(tr3-c-restore-prepare-receiver-internal! plan)
(tr3-c-restore-seal-internal! plan)
(tr3-c-restore-commit-tail-internal! plan)
(tr3-c-restore-abort-internal! plan)
```

These names describe the accepted parent transaction decomposition; they are
not a requirement to recreate component operations that now have accepted exact
entries. Any implementation names are lexical to the source-composed TR3
package. They are not installed, exported, serialized, or callable through an
arbitrary vector. The exact accepted component binding is recorded in the
companion implementation plan.
`result-cell` is an exact caller-owned one-slot cell initialized to `#f` and
promoted before admission. Snapshot writes the exact C2 shell only on success;
restore writes `#t` only after the trainer is idle again.

The accepted native I2 participant is feature-gated private package-bridge
surface, not public I2/provider ABI. It includes the authority-bound fixed
restore42 create/append/verify/prepare/checked commit/checked abort/terminal
entries recorded in `native/tr3_c_i2_restore_internal.h`, plus the checked owned
release entry:

```c
void et_tr3_c_i2_owned_release_checked_v1(void *owned);
```

The checked restore42 commit and abort reject an unauthenticated builder before
mutation, but after authenticating a TR3-C-owned builder they check every child
status and `_Exit(134)` on an impossible commit/release invariant. The
owned-release entry is sealed-tail-only and likewise fail-stops on a nonzero raw
release status.

TR3-specific failures use the accepted A0 categories:

| Condition | Category |
|---|---|
| malformed/foreign trainer, state, or option | `invalid-argument` |
| busy trainer, live batch/view, present gradient, active plan, same-thread reentry | `invalid-state` |
| path, tie, shape, count, or cross-counter mismatch | `shape-mismatch`, except lifecycle relations use `invalid-state` |
| non-f32 / non-CPU / non-dense or offset storage | `dtype-mismatch` / `device-mismatch` / `noncontiguous` |
| fixed profile or provider unavailable | `unsupported` |
| unavailable deterministic controls or compiler mismatch at load | `determinism-unavailable` |
| malformed/checksum-invalid cursor or detached data | `corrupt-data` |
| nonexact accepted format/component version | `version-mismatch` |
| allocation/provider/native preparation defect | existing dependency category or `internal` |
| admitted invariant after the first live write | fail-stop; never a recoverable return |

## 4. Snapshot admission and authority ledger

`trainer-state` admits only an exact registered trainer in `idle` with:

- canonical fixed-17 train modes;
- no live batch, tensor view, model frame/output, state borrow, optimizer plan,
  or restore plan;
- absent gradients and zero accumulated microbatches/weight;
- O2 completed updates equal trainer completed updates; and
- no same-thread entry already active.

The outer exception handler, result cell, three authority cells, cleanup
outcome, and copied-control envelope are allocated and promoted before the
trainer phase changes from `idle` to `snapshotting`. A canonical readback from
the trainer-owned active record follows the write barrier; no regional identity
is used afterward.

The exact authority cells are:

```text
p1-cell = [#f]
o2-cell = [#f]
c2-cell = [#f]
```

Snapshot performs this fixed order:

1. Revalidate the trainer, modes, absent gradients, D2 idle state, and counter
   equality after publication of `snapshotting`.
2. Call the real P1 `module-state-dict`; immediately place its exact detached
   owner in `p1-cell`.
3. Call the real O2 `optimizer-state`; immediately place its exact detached
   owner in `o2-cell`.
4. Obtain fresh current D2 cursor bytes. Deep-copy the trainer-owned epoch-start
   cursor, RNG vector, tokenizer vector, X1 canonical/fingerprint pair, and
   counters into the already-rooted control envelope. Validate the complete
   relations in section 7.
5. Form the existing exact six-slot C2 compose request
   `[c2-training-state-compose-tag,p1,o2,controls,'trainer-state,#f]` and call
   `c2-training-state-compose-internal` unchanged.
6. Determine transfer from the request, not from a wrapper return. When request
   slots 1 and 2 are `#f` and slot 5 is an exact live C2 owner, clear `p1-cell`
   and `o2-cell` and place that shell in `c2-cell`, even if a trusted wrapper
   raises or substitutes its return.
7. Publish the exact shell from `c2-cell` to the promoted result cell, reread it,
   clear `c2-cell`, set trainer phase to `idle`, and return that canonical shell.

On any prepublication failure, cleanup clears each nonempty authority cell
*before* invoking its release. Cleanup order is C2, O2, P1; owning C2 excludes
owning either component cell. Cleanup continues after the first cleanup defect,
never retries consumed authority, restores trainer phase to `idle`, and then
raises the original error unless cleanup reported an admitted provider defect,
which becomes `internal`. The live trainer's parameters, optimizer, D2 cursor,
RNG, modes, and counters never change during snapshot.

Issue #121 must repair the current compose handler-allocation prefix before this
path is implemented: P1 claim cannot precede installation of the O2-claim
rollback handler. TR3-C relies on the repaired authentic composer rather than
forking its policy.

## 5. Restore plan identity and phase machine

One source-private parent plan is allocated per restore call and installed only
in the authenticated trainer's active-plan slot. A copied vector cannot grant
authority: every helper first reauthenticates the trainer, requires
`eq? plan trainer.active-plan`, requires the plan self slot, and checks the
expected phase.

The outer exception handler, independently rooted result cell, all child
authority cells, and cleanup-progress/first-error record are allocated,
promoted, and armed before the trainer phase or active-plan slot changes and
before C2 borrow begin. Later preparation adds no new handler needed to recover
an O2 publication, clone/decode output, or source-borrow failure.

The logical plan fields are fixed:

```text
[private-tag, self, trainer, phase,
 c2-borrow-or-#f, 14-P1-stage-cells, O2-stage-cell, O2-plan-cell,
 copied-controls, D2-restore-plan, I2-copy-builder,
 cleanup-flags, promoted-result-cell]
```

`phase` transitions are:

```text
constructing -> source-staged -> source-closed -> receiver-prepared
             -> sealed -> committing -> committed -> dead
          \-----------------------------------------> aborting -> dead
```

Only `sealed -> committing` crosses the commit point. `aborting` is reachable
only before `committing`. The trainer phase is `restoring` for the whole plan
lifetime and returns to `idle` on abort or success. Publication into the
trainer slot promotes the complete graph; every later operation rereads the
canonical plan from that slot. Copied O2 configuration, epoch-start bytes, RNG,
counters, D2 plan, staging carriers, and result cell are therefore in trainer
lifetime before the first live write.

Installing this parent plan and the transient `restoring` phase is only
control-plane exclusion for the restore call. It changes no persistent trainer
control, tensor, optimizer, or dataset value and reserves no O2, D2, I2, or P1
component authority. "No live receiver reservation before `source-closed`"
means no component plan, busy flag, endpoint pin, or destination authority;
the parent exclusion is installed earlier so same-thread reentry rejects while
the detached source is staged. Every earlier failure removes that exclusion
and restores `idle`.

## 6. Detached-source staging and C2 borrow shutdown

Restore first admits the receiver as exact, `idle`, fixed-profile, canonical
train mode, D2-idle, absent-gradient, and plan-free. It then begins the existing
genuine C2 borrow with operation `trainer-load-state!`, obtains the existing
projection and exact access token, and runs the source/admission subset of
section 7 before constructing staged owners. Checks that depend on a prepared
O2, D2, or I2 receiver plan run only in the final precommit pass in section 9.

### 6.1 P1 staging

The existing C2-owned P1 projection supplies canonical entry shells and aliases.
For each of the 14 canonical unique parameter entries, in table order:

1. begin the existing authenticated C2-owned tensor access;
2. validate the exact path, kind, shape, dtype, device, layout, byte span, and
   tie topology;
3. call the accepted private `i2-clone` provider operation on the synchronous
   carrier through a pre-rooted request whose output slot is initially `#f`;
4. after either return or catch, inspect the request output, move any published
   owner into its exact parent authority cell once, clear the request slot, and
   only then propagate the original failure; promote and reread the installed
   owner's canonical identity;
5. end the tensor access before visiting the next entry.

No state-backed carrier, view, tensor shell, or access token escapes into the
stage. Failure releases every installed clone after ending the current access.

### 6.2 O2 staging and shared exclusion

No new detached O2 access authority is needed. The source wrapper authenticates
the exact C2-owned O2 state with the current C2 access token, copies its logical
config/path/alias/update projection, and uses the accepted fixed request:

```scheme
[O2-state,index,moment-kind,destination-bytevector,elements,
 C2-access-token,'trainer-load-state!]
```

with `o2-state-copy-moment-bits-c2-owned-internal` for each canonical index and
moment kind. Each destination is a preallocated exact-size bytevector. The
wrapper immediately decodes those bits through a pre-rooted accepted private
I2 decode request whose output slot is initially `#f`. After return or catch it
moves any published owner into the exact parent authority cell once, clears the
request slot, and then propagates an original failure. It promotes/rereads that
owner and drops the temporary bytes. Exactly 28 independent owned clones exist
before the outer C2 borrow ends. No O2 handle, native state pointer, or borrowed
moment carrier enters the parent plan. Clone/decode return substitution is
ignored; only the request output slot transfers authority.

After source shutdown, O2 owns a typed private receiver restore plan. The fixed
native core for the following logical responsibilities is accepted in
`native/tr3_c_o2_restore_internal.h`. One source-private adapter remains required
to realize the two-cell transfer, source topology/config validation, native
status mapping, logical slot-2 publication, and authority cleanup. It may use
these names or exact lexical equivalents, but must not invent a competing native
restore authority:

```scheme
(o2-optimizer-restore-prepare-internal
  optimizer trainer-lease copied-config copied-paths copied-aliases
  target-completed-updates expected-receiver-completed-updates
  staged-moment-authority-cell plan-authority-cell operation)
  -> exact-plan-or-raises

(o2-optimizer-restore-append-internal!
  o2-restore-plan exact-i2-copy-builder first-index operation) -> #t

(o2-optimizer-restore-check-internal o2-restore-plan operation) -> #t
(o2-optimizer-restore-commit-controls-internal! o2-restore-plan) -> #t
(o2-optimizer-restore-abort-internal! o2-restore-plan operation) -> #t
```

Prepare authenticates the live optimizer and all 28 I2 owned-clone carriers,
requires absent gradients, validates exact 14-entry topology, finite moments,
options/configuration, provider identity, target update `U`, and captured live
receiver update `R`, and only then publishes the existing optimizer busy field
plus exact active-operation
backreference. The parent supplies two promoted one-slot authority cells. The
stage cell initially owns the fixed 28-cell stage vector and the plan cell is
`#f`. Prepare publishes the exact plan into the plan cell and clears the stage
cell only after the common busy/backreference is established. On every return
or raise, including a trusted wrapper that raises after publication or
substitutes its return, the parent ignores the returned value and rereads both
cells plus the authenticated optimizer backreference. A nonempty stage cell
means the parent still owns all 28 stages and no restore plan was published; an
empty stage cell means the exact plan cell/common backreference owns them.
Every other combination is an internal invariant failure before any persistent
receiver value changes; a published exclusion is still recoverably abortable.
Thus cleanup can never release both authorities or lose a
publish-then-raise plan.
The O2 source wrapper also owns the promoted deep-copied logical configuration
needed for the accepted Eshkol optimizer vector slot 2. Append is callable only
by the localized TR3-C compositor; it
requires `first-index == 14` and appends exactly 28 fixed destination/staged
source pairs. It exposes no assignment array, moment pointer, parameter index,
or generic builder mutation seam. The O2 plan and TR3-O update-clear plan use
the same optimizer busy/backreference and mutually reject.

The accepted #119 receiver invariant is reused verbatim whenever the common
private-operation feature is compiled: idle is `busy == 0 && active_operation ==
NULL`; a long-lived plan is `busy == 1 && active_operation == exact-plan`;
accepted legacy synchronous O2 work may transiently be `busy == 1 &&
active_operation == NULL`. Restore defines no second field or helper. Its typed
registry/magic/lifecycle authenticate the operation kind, and every check,
commit, and abort requires `optimizer->active_operation == plan`.

Commit-controls is callable only from the sealed TR3 tail. While the exact
common busy/backreference remains installed, the source helper writes the
prevalidated native config/options/completed-update count, assignment-writes
the promoted logical config into the optimizer's existing Eshkol slot 2, and
drains its 28 staged clones after I2 has unpinned them. A native no-error
finalizer then clears `active_operation`, clears `busy`, and retires the typed
plan last. The slot-2 store mirrors current `optimizer-load-state!`, cannot
allocate or fail, and therefore cannot escape after native and source-visible
configuration diverge. Abort is precommit-only; it drains retained stages and
then clears `active_operation` and `busy` without changing the optimizer. The
typed restore plan has a mandatory private registry, magic, lifecycle, and
tombstones, but creates no second optimizer receiver registry, busy field, or
detached-state authority.

### 6.3 Control staging and borrow end

The parent deep-copies the five C2 control fields: RNG, tokenizer, cursor pair,
resolved X1, and counters. It retains no list, bytevector, or vector owned only
by the borrowed state. It validates and promotes those copies before shutdown.

After all 42 sources and all controls are independently owned:

1. require no active P1 tensor access;
2. end the existing C2 borrow through the unchanged
   `c2-training-state-borrow-end-internal`;
3. clear the plan's C2-borrow slot;
4. require the returned operation succeeded before any receiver plan is built.

There is no separate callable nonmutating borrow-end preflight in main and this
contract does not invent one. The repaired #121 path must prove that the
unchanged combined operation completes every recoverable outer/P1/O2 marker
check before its first subordinate clear. A pre-admission rejection leaves the
whole borrow intact; any defect after the first clear fail-stops instead of
returning a detached owner that might be partially busy. The admitted tail
allocates nothing and calls no provider/callback. #121's acceptance evidence
must cover this exact ordering. There is no persistent receiver mutation before
step 4, and the O2 receiver restore plan does not yet exist. On any recoverable
earlier failure, all 42 staged tensors are released, the receiver returns to
`idle`, and the error is raised. Successful shutdown is the proof
boundary: the detached C2 owner is again live, byte-unchanged, saveable,
releasable, and independently
retryable. The later transaction retains no C2/P1/O2 detached-state carrier,
token, shell, view, or pointer.

“Unchanged” means identical logical projection and canonical SAVE bytes with no
active borrow/access. The first authenticated P1 tensor access may lazily
materialize C2-owned state-tensor shell identities; that monotonic private cache
is not serialized mutation. Retention-flat evidence therefore prewarms those
identities once, then measures the 1,024/8,192 steady-state horizons separately.

## 7. Complete precommit validation

All arithmetic is checked in signed nonnegative i64 space before multiplication
or addition. Let:

- `A` be the fixed accumulation count from X1;
- `U` be target completed updates from C2, detached O2, and copied controls;
- `R` be the captured pre-restore completed updates of the live receiver;
- `E` be completed epochs;
- `T` be total active tokens;
- `M` be D2 total rows;
- `S` be the epoch-start next ordinal; and
- `C` be the current next ordinal.

The following are mandatory:

- exact library, compiler, provider, X1 canonical bytes/fingerprint, tokenizer
  fingerprint/V256, and fixed M3 profile;
- exactly 15 P1 paths, 14 unique parameters, and the one
  `head/weight`–`token_embedding/weight` tie;
- all 42 exact shapes and storage properties in section 2, finite moment values,
  valid O2 group/options/schedule fields, and
  `detached-O2.completed_updates == C2.completed-updates == U`;
- before mutation, live O2 and live trainer completed updates both equal `R`;
  `R` may be less than, equal to, or greater than `U` and must remain unchanged
  through seal;
- RNG `[philox4x32-10,1,key,low,high]`, `key == X1 run.seed`, and, because this
  fixed no-dropout profile never advances training RNG, `low == high == 0`;
- both D2 cursors decode canonically, have identical fields 1 through 14, match
  the live dataset identity/options, and satisfy `S < M` and `S <= C <= M`;
- `U*A == E*(M-S)+(C-S)`;
- `U*A <= T <= 2*U*A` for fixed `[1,2]` batches; and
- receiver is still `restoring`, modes are still the exact 17 train modes,
  gradients are absent, D2 is idle, parameter/moment identities are unchanged,
  and O2 still owns this exact restore-plan backreference.

The fixed read-only mode check reuses
`tr3-p1-fixed-recheck-internal` through a trainer-lexical wrapper. It
authenticates the trainer lease and its 17 retained module identities, reads
only the existing P1 mode state, and accepts only all-train. It is not installed
and grants no E3 or mode-mutation authority.

The receiver's current tensors, controls, and captured count `R` are not required
to equal the checkpoint targets or `U`; only topology, configuration,
identities, and lifecycle must be compatible. Successful commit publishes `U`
to both O2 and trainer counters in the fixed tail.

## 8. Trainer-specific D2 binding

Public `token-dataset-seek!` is not used. The implemented trainer-private source
operations are:

```scheme
(tr3-d2-restore-prepare-internal dataset current-cursor epoch-start-cursor)
(tr3-d2-restore-check-internal plan)
(tr3-d2-restore-commit-internal! plan)
(tr3-d2-restore-abort-internal! plan)
```

Prepare authenticates the exact open D2 state and native owner, requires no live
batch or trusted view (`state[10] != 'live`, borrow count zero, native current
batch null), decodes and checks both cursors, and captures exact state/core
identities, the receiver's precommit ordinal, target `C`, `M`, and dataset
generation/status witnesses. It allocates and parses only before sealing. The
epoch-start bytevector is copied into the parent trainer controls; D2 plan does
not store or publish it as D2 state.

Check is read-only and allocation-free. It requires the same authenticated
dataset/state/core identities, same precommit ordinal and generation witnesses,
no batch/view, and the exact prepared target.

Commit has no recoverable branch and performs only:

```scheme
(vector-set! core-state 15 target-current-ordinal)
(vector-set! d2-state 10 'none)
(vector-set! d2-state 16 #f)
```

It does not rewrite native generations, shuffle storage, total rows, dataset
metadata, manifest, config, or epoch-start state. Abort clears only the private
plan and never touches D2.

Source fields alone cannot prove that native `current_batch` is null. The
trainer-specific D2 translation unit therefore adds exactly one fixed idle
probe, localized and uninstalled:

```c
int64_t et_tr3_c_d2_dataset_idle_preflight_v1(const void *owner);
```

It compares `owner` through D2's actual private registry without dereferencing
foreign memory. It returns 1 for null/unknown owner, 2 for an admitted dataset
with nonnull `current_batch`, and 0 for an admitted dataset with null
`current_batch`; the corresponding D2 private last-status is
invalid-argument/invalid-state/ok. It allocates nothing and mutates no dataset
field or generation. Prepare and check require status 0. This is a separate
trainer binding, not an E3 frame or a generic D2 pointer export.

## 9. One receiver transaction

After source shutdown, receiver preparation runs in this fixed order:

1. Prepare the trainer-specific D2 plan.
2. Prepare the O2 typed restore plan through the two-cell transfer ledger, then
   reread and check its canonical identity and shared busy/backreference.
3. Create one existing I2 copy builder for exactly 42 assignments.
4. A fixed TR3 model compositor authenticates the receiver trainer/model/tree
   and appends the 14 canonical live parameter destinations with the 14 staged
   sources at indices 0 through 13. It exposes no handle list or raw pointer.
5. Call the fixed O2 compositor to append indices 14 through 41.
6. Preinstall the remaining sealed-tail invariant handlers, promote and reread
   the final result graph, then run the complete checks in section 7 plus the
   D2 and O2 checks. The outer recovery handler and authority ledgers were
   already armed before source staging.
7. Prepare the existing I2 copy plan as the final recoverable operation. This
   proves distinct destinations and sources, exact shape equality, no
   source/destination cross-alias, and pins all 84 tensor endpoints without
   changing bytes. The already-installed cleanup path can still call the native
   builder abort.
8. Assignment-mark the parent `sealed`. No allocation, promotion, parsing,
   recursive data comparison, provider call, callback, error construction, or
   recoverable check occurs after this point.

The fixed model compositor is source-private to the authenticated trainer/M3/P1
aggregate. It is not the E3 frame binding and cannot be called with a caller-
selected model, handle, path, tensor, or index. The ordinary M3 call pins are
not used: their read-only value-borrow sentinel conflicts with I2 destination
plan preparation.

## 10. Commit tail proof

The first live write is the first destination `memcpy` inside the already-
prepared I2 plan. The tail is exactly:

1. set parent phase `committing`;
2. invoke the fixed TR3-C I2 builder commit wrapper, which marks the child plan
   consumed, copies all 42 tensors, checks successful child-plan release and
   pin removal, then frees the assignment array and retires the builder without
   allocation or callback; an impossible child-release failure fail-stops and
   is never returned as a recoverable partial restore;
3. invoke O2 `commit_controls` to publish native config/options/completed
   updates and Eshkol logical slot 2, drain the 28 O2 stages, clear the shared
   active-operation backreference and `busy`, then retire its typed plan;
4. invoke D2 commit to publish current ordinal and clear status/cache;
5. publish trainer-owned epoch-start bytes, RNG words, total tokens, completed
   updates, and completed epochs from the promoted control record;
6. drain the remaining 14 P1 staged owned tensors through fixed private
   no-error checked release after the I2 plan has removed their pins; clear each
   Eshkol carrier raw-pointer slot and then its parent authority cell;
7. while the canonical parent remains rooted in `trainer.active-plan`, clear
   its now-empty child authority slots and mark it `committed` then `dead`;
   clear `trainer.active-plan`, set trainer phase `idle`, and finally write
   immediate `#t` through the independently outer-rooted promoted result cell.
   No parent field is accessed after unlink.

The parent deliberately retains the accepted native builder rather than the
Eshkol `i2-prepare-load` result because the latter exposes no abort operation.
The current bridge commit and abort discard the status of
`et_f32_tensor_copy_plan_release_v1`; TR3-C must not call those unchecked paths.
Feature-gated, same-translation-unit TR3-C commit and abort entries authenticate
the exact builder, observe both child commit and release status, and
unlink/free/retire only after successful release has removed every pin. Any
failure after authenticated child commit/release admission invokes
`_Exit(134)`; it is never translated into success or an allocating/retryable E1
error. This changes no `f32_tensor.c` behavior, public I2 ABI, provider row, or
SHARED-R2 identity. I2 prepare remains the final recoverable validator before
seal. Staging cleanup preflights
every exact owner before sealing. Its post-copy drain uses the fixed localized
raw I2 owned-release entry, not the general P1 provider release callback. It
receives only authenticated owned tensors with zero borrows/pins, constructs no
error, and fail-stops on an impossible nonzero release. `free`/retirement is
allowed; allocation is not. No change to `native/f32_tensor.c`, I2 ABI, provider
row, or SHARED-R2 source is required.

No tensor storage identity changes. Parameters and O2 moments retain their live
owners and receive exact binary32 bytes. There is no rollback after step 2 and
no path that reports a retryable partial restore.

## 11. Precommit failure matrix and abort order

| Failure point | Required state after abort |
|---|---|
| receiver admission / plan allocation | receiver and input untouched; no plan |
| C2 borrow begin/projection | receiver idle; input live after repaired #121 cleanup |
| any P1 access/clone/end | current access ended; staged prefix released; input live |
| O2 source staging | staged prefix released; no live optimizer reservation exists |
| control copy or cursor decode | no receiver value changed; copied graph released |
| C2 borrow-end pre-admission rejection | no receiver value changed; no O2 receiver plan exists; repaired cleanup ends the intact borrow and releases stages |
| impossible post-admission borrow-end defect | fail-stop before persistent receiver mutation; never return a partially busy input as retryable |
| O2 receiver prepare | shared busy/backreference cleared; all stages released |
| D2 prepare/check | D2 unchanged; O2 reservation and stages released |
| model/O2 compositor or I2 prepare | zero live destination writes; all pins drained |
| final identity/reachability check | receiver byte/control snapshot unchanged; input already live and unchanged |

Abort order is phase-dependent. If a P1 tensor access is active, end it first.
If the C2 borrow is active, end it and clear its authority before releasing any
staged clone. After source shutdown, abort order is: checked I2 builder/plan
abort, D2 plan, O2 restore plan (or the still-parent-owned 28-stage cell), staged
P1 tensors, copied controls, trainer active-plan graph, trainer phase. After
source shutdown, each terminal child authority is taken locally and its parent
cell clears before its cleanup call; C2 borrow authority instead clears only
after successful borrow end. The checked native I2 abort retires its builder
only after child-plan release succeeds. Cleanup
continues after the first recoverable defect and never retries consumed
authority. The original operation error remains primary unless cleanup finds an
admitted ownership/provider invariant defect, which is reported as `internal`.
An impossible authenticated checked-release failure fail-stops rather than
leaking pins and reporting successful cleanup. No abort operation is called
after `committing`.

Every recoverable failure before commit must prove:

- all 14 live parameters, 28 live moments, O2 config/count, D2 ordinal/cache,
  trainer epoch-start/RNG/counters, and 17 modes are bit/identity unchanged;
- detached input C2 bytes and projections are unchanged;
- after any begun borrow is ended, the same input can be inspected, saved,
  released, or used by a retry; and
- no batch, view, tensor access, plan pin, optimizer backreference, or regional
  graph survives.

## 12. Lifetime diagram

```text
detached C2 owner (caller; remains live after call)
  -> genuine C2 borrow
       -> serial P1 state-tensor access -> owned P1 stage[14]
       -> O2 authenticated source       -> owned O2 stage[28] + controls
       -> copied C2 controls
  -> all accesses end
  -> existing C2 borrow ends  <===== hard precommit boundary

trainer (caller-owned, restoring)
  -> canonical promoted parent plan
       -> owned stage[42]
       -> O2 typed restore reservation (shared optimizer busy/backreference)
       -> trainer-private D2 plan
       -> one prepared I2 plan (42 destination/source pairs)
  -> seal
  -> I2 bytes -> O2 controls -> D2 current -> trainer controls
  -> stage/plan drain -> trainer idle -> #t publication
```

The detached owner and trainer lifetimes are independent. The parent plan never
owns the C2 owner; it owns only deep stages and copied data after source shutdown.

## 13. Acceptance evidence

Proposal acceptance requires an independent Sol/high exact-head review. Later
implementation acceptance requires focused source/native tests before root
integration or full CI dispatch.

### Snapshot

- Inject every P1 snapshot, O2 snapshot, cursor/control copy, compose claim,
  publication, and release prefix.
- Prove the three-cell ledger never double-releases or leaks P1/O2/C2 authority,
  including trusted-wrapper raise/return substitution after compose transfer.
- Prove snapshot leaves all live trainer state unchanged and returns the exact
  C2 object accepted by current `checkpoint-save!` and `trainer-state-release!`.
- Repeated snapshots at one update boundary serialize byte-identically.

### Restore

- Verify all 42 staged tensors and all 42 live destinations against independent
  exact bit fixtures; mutate each source/destination shape/path/alias in turn.
- Inject every allocation and provider/native preparation prefix. Before the
  first write, compare complete receiver bits/controls and detached-state SAVE
  bytes, then retry successfully.
- Inject raise and return substitution after every I2 clone/decode output
  publication and after O2 plan publication. Prove each output/transfer ledger
  moves or releases every owner exactly once and clears both O2 exclusion fields.
- Assert the C2 borrow and every P1 tensor access have ended before I2 receiver
  prepare/commit. Assert input SAVE/release works while the receiver plan is
  prepared.
- Verify tensor owner/data identities survive successful restore.
- Cover malformed current and epoch-start cursors, cross-dataset pairs,
  `S==M`, `C<S`, `C>M`, arithmetic overflow, both reachability equations,
  nonzero fixed-profile RNG counters, counter disagreement, NaN/Inf moments,
  present gradients, live D2 batch/view, wrong mode, and competing O2 plan.
- Restore `U>0` into a fresh compatible receiver captured at `R=0`, and restore
  a rewind target with `R>U`; in both cases require precommit preservation of
  `R`, commit publication of `U` to both O2/trainer, and exact continuation.
- Instrument the sealed tail: zero allocations, handlers, callbacks, providers,
  parser calls, error construction, or recoverable exits; exact fixed write order.
- Inject the private I2 child-release defect hooks in checked abort and commit;
  require `_Exit(134)` rather than ignored status, leaked pins, success, or a
  retryable partial restore.
- Include #121 evidence that combined C2 borrow-end completes every recoverable
  marker check before its first clear and never returns after a partial clear.
- After one documented P1 identity prewarm, run ASan/UBSan/LSan focused tests
  and retained-control snapshots for 1,024 and 8,192 snapshot/restore/failure
  cycles without growing live authorities.

### Exact continuation

Run uninterrupted `K+R` and checkpoint/resume `K` then `R`, with `R >= 3`, in
fresh processes. Cover `A=1` and `A>=2` with unequal active-token counts, and
place `K` so the resumed suffix crosses D2 EOS. Require bit-identical:

- 14 parameters and 28 moments;
- next-step metrics and effective learning rate;
- O2/trainer update counts, RNG words, current/epoch-start cursors, token/epoch
  counters; and
- final canonical C2 bytes.

Inject one failure immediately before seal, abort, retry, and require the whole
remaining trajectory and final C2 file to equal uninterrupted execution.

## 14. Unsupported scope

Mid-accumulation snapshot/restore, present gradients, live D2 batch/view,
evaluation mode, active model frames/outputs, arbitrary model profiles, devices
or dtypes other than fixed CPU f32, and competing optimizer operations reject
before mutation.

Same-thread reentry is detected and returns `invalid-state` before mutation.
The underlying I2/P1 registries require caller serialization; racing calls from
multiple threads, signal handlers, or forked processes are unsupported and have
no structured-error or data-race-safety guarantee. No test may reinterpret that
unsupported scope as a lock or transactional concurrency claim.

## 15. Accepted decisions and implementation prerequisites

Root accepted the exact O2 restore-plan/shared-exclusion boundary, the checked
private I2 commit/abort and staging-drain ownership, and the trainer-specific D2
seam. The native I2, O2, and D2 participants now exist as bounded private leaves.
The fixed model compositor remains source-private to TR3 and does not belong to
E3.

Implementation must use the exact accepted component entry points and authority
identities recorded in the companion implementation plan. It must not recreate
the earlier abstract operations in this document as competing component seams.
The lease registry and source binding/compositor remain pending. Merged, verified
#121 remains mandatory before final snapshot/restore acceptance and supported
adoption.
