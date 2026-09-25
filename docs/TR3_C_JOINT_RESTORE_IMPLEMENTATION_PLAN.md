# TR3-C joint restore composer implementation plan

Status: **lease successor and private I2/O2 binding partition accepted;
composer remains pending**.

The normative restore protocol is the accepted
[TR3-C live-state contract](TR3_C_LIVE_STATE_CONTRACT.md). This companion maps
that protocol to the accepted component interfaces. Its original source mapping
used integration head `ca26c3c` and lease candidate `f602a66`; the reviewed
22-slot lease and private I2/O2 binding leaf are integrated at `b428da3`.
Exact accepted source pins and evidence are recorded in
[the integration log](INTEGRATION_LOG.md).

This is not a public trainer, joint restore, or exact-resume acceptance claim.
The reviewed 22-slot lease successor and rooted D2 allocation repair are now
integrated, with byte-identical passing O0/O2 allocation-prefix matrices. The
private binding partition adds only fixed feature-local adapters over the
accepted I2/O2 restore entries; final composition and trajectory evidence remain
separate work.

## 1. Readiness verdict

The accepted #120 protocol is complete enough for bounded implementation and
needs no cross-component redesign. The accepted D2, I2, and O2 restore leaves
and existing C2 borrowed projection cover the required native and detached-state
authority. No new public API, tensor-provider row, native tensor operation,
generic handle export, scheduler owner, D2 epoch-start field, or second optimizer
registry is required.

The dependency state for the source-level prerequisites is:

1. the accepted lease successor is selectively composed with the current
   integration head;
2. trainer-owned restore controls and authenticated restore enter/finish
   transitions are present in the sole lease authority;
3. feature-local Eshkol bindings for the accepted native I2 and O2 restore
   entries are accepted in the integration branch; and
4. final source composition, composer failure-prefix evidence, and mandatory #121
   C2 handler-order acceptance evidence.

The lease allocation prerequisite is closed by the rooted D2 u64-limit repair
and the full exact production-OFF O0/O2 failure-prefix matrices. This does not
make a broader claim about unsupported runtime allocation paths.

## 2. Exact existing component boundaries

### 2.1 Accepted lease successor: sole trainer authority

`native/tr3_lease_core_extension.esk` owns one enrollment root and an exact
private token. Its accepted 22-slot record is:

```text
0 tag                       9 exact M3T model entry
1 self                     10 retained modules[17]
2 public trainer shell     11 retained P1 handles[14]
3 phase                    12 retained parameter carriers[14]
4 active plan              13 optimizer shell
5 private lease token      14 exact O2 record
6 resolved X1 object       15 dataset shell
7 tokenizer identity       16 exact D2 state
8 model shell             17 epoch-start cursor bytes
                          18 RNG [algorithm, version, key, low, high]
                          19 total active tokens T
                          20 completed updates U
                          21 completed epochs E
```

`tr3-lease-authenticate` revalidates exact record, token, M3T, P1, O2, and D2
identity. `tr3-lease-restore-internal` admits only an already-published
`restoring` record whose active plan is the exact parent. It does not publish or
clear that state. The lease is logical authority, not a tensor pin, model frame,
D2 borrow, I2 plan, or optimizer busy owner.

The accepted successor preserves one registry and the exact private token. It
adds no public token, generic phase setter, second trainer registry, or
component lifetime pin.

### 2.2 C2 detached source: already source-callable

The existing operations are:

```scheme
(c2-training-state-borrow-begin-internal owner 'trainer-load-state!)
(c2-training-state-borrow-projection-internal borrow)
(c2-training-state-borrow-owner-token-internal borrow 'trainer-load-state!)
(c2-training-state-borrow-end-internal borrow)
```

The borrowed projection is exactly:

```text
[P1-owner, O2-owner, controls, P1-topology, O2-projection]
```

`controls` has canonical order:

```text
0 RNG             [philox4x32-10, 1, key, low, high]
1 tokenizer       [fingerprint, vocabulary]
2 D2 cursor pair  decoded canonical pair retaining current and epoch-start bytes
3 resolved X1     [canonical bytes, fingerprint, decoded projection]
4 counters        [total active tokens, completed updates, completed epochs]
```

The P1 source access operations are
`state-dict-c2-owned-projection-internal`,
`state-dict-c2-owned-tensor-borrow-begin-internal`, and
`state-dict-c2-owned-tensor-borrow-end-internal`. The existing `i2-clone`
request deep-owns each of the 14 unique parameter values. The O2 source
projection and `o2-state-copy-moment-bits-c2-owned-internal` provide the 28
moment images, which the existing `i2-checkpoint-decode` converts to independent
owned clones.

The C2 owner remains caller-owned. A successful borrow end is the hard source
boundary: no C2, P1 state-tensor, or detached O2 authority may enter receiver
preparation.

### 2.3 D2 receiver: complete source-private seam

`native/tr3_c_d2_restore_extension.esk` already defines:

```scheme
(tr3-d2-restore-prepare-internal dataset current-cursor epoch-start-cursor)
(tr3-d2-restore-check-internal plan)
(tr3-d2-restore-commit-internal! plan)
(tr3-d2-restore-abort-internal! plan)
```

Prepare parses both cursors, authenticates dataset identity, requires a native
and source-level idle receiver, captures the old ordinal and generation/status
witnesses, and retains only the target current ordinal. Check is recoverable and
read-only. Commit is the sealed assignment-only tail: it writes the current
ordinal and clears D2 status/cache. Epoch-start bytes remain trainer-owned. Abort
retires only the private plan.

The native idle probe is
`et_tr3_c_d2_dataset_idle_preflight_v1`; no additional D2 pointer export or
public seek operation is needed.

### 2.4 I2 receiver transaction: complete native seam

`native/tr3_c_i2_restore_internal.h` freezes the 42-assignment transaction:

```c
et_tr3_c_i2_copy_builder_create_restore42_v1
et_tr3_c_i2_copy_builder_append_parameter_v1
et_tr3_c_i2_copy_builder_append_restore42_v1
et_tr3_c_i2_copy_builder_verify_restore42_v1
et_tr3_c_i2_copy_builder_prepare_restore42_v1
et_tr3_c_i2_copy_builder_commit_restore42_checked_v1
et_tr3_c_i2_copy_builder_abort_restore42_checked_v1
et_tr3_c_i2_copy_builder_require_terminal_restore42_v1
```

The checked raw owned-release declaration is in
`native/tr3_c_o2_restore_internal.h`; its implementation shares the same I2
package bridge and fail-stop policy:

```c
et_tr3_c_i2_owned_release_checked_v1
```

The builder owns exactly 42 assignment slots. Indices 0 through 13 are appended
one at a time from the fixed lease-retained parameter carriers and P1 handles.
The O2 participant appends the fixed suffix at index 14. Prepare is the final
recoverable allocation/pin operation. Checked commit performs the first live
write, copies all 42 tensors, removes all pins, and retires the builder. Checked
abort removes pins and retires the builder without writing. An admitted child
commit/release defect fail-stops with `_Exit(134)`.

The exact O2 native plan pointer is the I2 restore authority. The composer must
create the I2 builder only after O2 prepare and pass that same pointer to every
parameter append and I2 prepare/commit/abort call. A parent-plan pointer is not a
substitute for this authority.

### 2.5 O2 receiver: complete native participant, source binding missing

`native/tr3_c_o2_restore_internal.h` and
`native/tr3_c_o2_restore.inc` define:

```c
et_o2_tr3_c_restore_prepare_v1
et_o2_tr3_c_restore_append_v1
et_o2_tr3_c_restore_check_v1
et_o2_tr3_c_restore_commit_native_v1
et_o2_tr3_c_restore_finalize_v1
et_o2_tr3_c_restore_abort_v1
```

The fixed 728-byte request contains target and expected receiver update counts,
clip policy, scheduler kind/minimum ratio/warmup/total updates, and fourteen
per-parameter option/stage rows. Each row carries learning rate, beta1, beta2,
epsilon, weight decay, `exp_avg`, and `exp_avg_sq`.

Prepare authenticates the live optimizer, validates absent gradients, all
receiver identities, all 28 stages, target configuration, target update count
`U`, and captured receiver count `R`, then acquires the existing optimizer
active-operation authority. Append supplies assignments 14 through 41 and binds
the exact I2 builder. Check is the final recoverable O2 validation.

After I2 is terminal-committed, `commit_native` assignment-writes native config,
per-entry options, and `U`. The source adapter must then assignment-write the
already-promoted logical configuration into the existing optimizer record slot
2. `finalize` checked-releases the 28 stages, clears the existing optimizer
active-operation/busy authority, and retires the typed plan. No fallible source
operation may occur between native commit, logical slot-2 publication, and
finalize.

The Eshkol I2 carrier records are source ownership wrappers, not the native O2
plan's authority. On successful O2 prepare, the adapter must clear each of the
28 transferred carrier raw-pointer slots exactly once after the native plan is
canonically published. On prepare failure, those slots remain intact and the
parent still owns the stages. O2 finalize/abort releases the transferred raw
owners; the cleared wrappers are then only dead bookkeeping.

Abort requires the I2 builder either absent or terminal-aborted, releases all 28
stages, and clears the same optimizer authority. It is precommit-only.

## 3. Minimal source seams still required

Names and layouts for these seams must be frozen in the lease/composer
implementation task. This plan fixes responsibilities, not speculative public
or native APIs.

### 3.1 Lease-owned trainer state and transitions

The accepted successor must give the sole canonical trainer record durable
ownership of:

```text
epoch-start cursor bytes
RNG [algorithm, version, key, low, high]
total active tokens T
completed updates U
completed epochs E
```

The immutable lease-bound tokenizer identity and resolved X1 identity remain
validation witnesses; restore never replaces them. Scheduler state remains in
O2 configuration and is not duplicated into the trainer record.

The same authority needs narrow private operations for:

- authenticating a public trainer shell and obtaining the canonical record and
  private token without exporting either;
- allocation-safe publication of `idle/#f -> restoring/exact-parent` after the
  complete parent graph and recovery machinery are promoted;
- reauthentication of `restoring/exact-parent` during preparation and seal; and
- assignment-only success/abort unlink back to `idle/#f`.

These may be lexical composer stores if their exact ownership is frozen and
tested. They must not become caller-selectable phase mutation.

### 3.2 Fixed I2/O2 Eshkol bindings

The accepted C APIs use native structs and output pointers and are not directly
safe source interfaces. Bounded feature-local package ABI adapters must turn the
I2/O2 out-parameters into authenticated pointer-return/status operations, and
one lexical source adapter must:

- translate the authenticated O2 projection and 28 owned stage carriers into
  the fixed request without exposing a generic request builder;
- distinguish parent-owned O2 stages from a successfully published native O2
  plan using promoted authority cells and canonical readback, and clear the 28
  transferred carrier raw-pointer slots exactly once;
- expose only the fixed prepare/append/check/commit/finalize/abort sequence;
- bind restore42 builder create, 14 fixed prefix appends, prepare, checked commit,
  checked abort, and checked P1-stage release; and
- map recoverable native statuses through the existing dependency categories.

No new native restore authority or storage primitive is required. The only new
native-callable symbols are fixed ABI adapters over the accepted entries; the
source adapter is an ownership ledger over those bindings.

### 3.3 Fixed model and mode composition

The fixed prefix compositor uses lease record slots 11 and 12—the exact 14 P1
handles and parameter carriers—in their accepted canonical order. It appends
those destinations to the I2 builder without accepting caller-selected indices,
handles, carriers, or models.

`tr3-p1-fixed-recheck-internal` already validates the exact model, 17 modules,
14 handles/carriers, topology, provider, and all-train mode. A narrow lexical
wrapper may give it the trainer operation/error spelling. No new P1/E3 mode
authority or mode setter is required.

## 4. Exact restore sequence

### 4.1 Prepublication admission

1. Authenticate the exact trainer and private lease token. Require `idle`, no
   active plan, fixed train mode, M3/D2 idle, absent gradients, and live
   O2/trainer count equality.
2. Allocate the result cell, parent plan, all authority cells, copied-control
   envelope, cleanup progress, and first-error record. Install every recovery
   handler needed by later publication or clone/decode failures.
3. Promote the complete graph. Publish it as the trainer active plan and change
   phase to `restoring`. Reread the canonical graph from the trainer record.
4. Run lease restore admission against that exact canonical parent.

This publication is control-plane exclusion only. It owns no D2 plan, O2 busy
authority, I2 builder/pin, C2 borrow, or P1 tensor access.

### 4.2 Detached-source staging

1. Begin the genuine C2 borrow and retain the exact borrow and access token in
   the parent.
2. Validate the complete P1 projection and canonical 14-unique/15-path tie
   topology.
3. For each unique P1 entry in fixed table order, begin one C2-owned tensor
   access, call `i2-clone` through a pre-rooted output-slot request, move any
   published clone into its parent authority cell before propagating a failure,
   and end the access.
4. For each O2 row in canonical order, copy `exp_avg` then `exp_avg_sq` bits and
   decode each through a pre-rooted output-slot request. Move published ownership
   before propagating a failure. The second moment must satisfy finite and
   nonnegative policy during receiver preparation.
5. Deep-copy controls in this order: RNG, tokenizer identity, current cursor,
   epoch-start cursor, X1 canonical bytes/fingerprint, then counters `T/U/E`.
6. Mark `source-staged`, require no active P1 access, end the C2 borrow, clear its
   authority cell after success, then mark `source-closed`.

No receiver component plan is created before `source-closed`. On success the
detached owner is live, unchanged, independently saveable/releasable/retryable,
and no source carrier or access token survives in the parent.

### 4.3 Receiver preparation and seal

Use this fixed order:

1. D2 prepare from current and epoch-start cursor bytes.
2. O2 native prepare from the promoted target config, `U`, captured live `R`,
   and 28 stage carriers. Transfer stage ownership to the exact published O2
   plan only when native prepare has published it; then clear the 28 source
   carrier raw-pointer slots before dropping their parent authority.
3. Create the restore42 I2 builder with that exact O2 plan as authority.
4. Append 14 fixed live parameter destinations and P1 stages at indices 0..13.
5. O2 append the 28 moment pairs at fixed index 14; it fills indices 14..41.
6. Reauthenticate lease/parent and run all cross-component validation below.
7. Run D2 check and O2 check.
8. I2 prepare last. This proves the 42 complete, disjoint, shape-compatible
   assignments and pins all 84 endpoints without changing bytes.
9. Assignment-mark `sealed`.

The final validation covers:

- the immutable library/compiler/provider/fixed-profile witnesses;
- exact P1 topology and tie, O2 paths/options/clip/scheduler, finite moments,
  target `U`, and unchanged receiver count `R`;
- tokenizer fingerprint/vocabulary equal to the lease and X1 identities;
- exact X1 canonical bytes/fingerprint and seed;
- RNG algorithm/version/key and fixed-profile zero counters;
- current/epoch-start cursor identity, `S < M` and `S <= C <= M`;
- `U*A == E*(M-S)+(C-S)` with checked signed-i64 arithmetic;
- `U*A <= T <= 2*U*A` for the fixed `[1,2]` batch profile;
- exact train modes, absent gradients, idle D2, stable parameter/moment
  identities, and exact O2 active-operation ownership.

After `sealed`, there is no allocation, promotion, parse, recursive comparison,
provider/callback invocation, error construction, or recoverable branch.

### 4.4 Irreversible tail

The fixed write order is:

1. assignment-mark parent `committing`;
2. checked I2 commit: first live write, 42 tensor copies, unpin, retire builder;
3. O2 native commit: native config/options and update count `U`;
4. assignment-write the promoted logical O2 config to optimizer record slot 2;
5. O2 finalize: drain its 28 stages and clear active-operation/busy;
6. D2 commit: current ordinal plus status/cache clear;
7. trainer stores: epoch-start bytes, RNG, total active tokens `T`, completed
   updates `U`, completed epochs `E`, in that order;
8. in canonical order, checked-release each remaining P1 raw owner,
   assignment-clear its Eshkol carrier raw-pointer slot, then clear/drop that
   stage authority cell;
9. clear empty child authority cells, mark parent `committed` then `dead`, unlink
   it, set trainer `idle/#f`, and publish immediate `#t` to the independently
   rooted result cell last.

Every defect after the checked I2 commit is admitted and fail-stops. There is no
rollback and no retryable partial-restore result after the first tensor copy.

## 5. Precommit abort and error precedence

Abort is phase-dependent. Terminal child cleanup takes the exact authority into
a local, clears its parent cell, and then invokes cleanup. C2 borrow-end is the
deliberate exception: its cell clears only after successful end, because a
recoverable preflight rejection leaves the borrow intact for the same cleanup
path.

1. If a P1 tensor access is active, end it first.
2. If the C2 borrow is active, end it and clear the borrow cell before releasing
   any staged clone. A recoverable borrow-end preflight rejection leaves the
   intact borrow available to the same cleanup path; an impossible defect after
   subordinate clearing fail-stops under #121.
3. Clear/take I2 authority and call checked abort, if a builder exists.
4. Abort the D2 plan, if published.
5. Abort the O2 plan, or first clear/take the parent O2-stage authority cell and
   then, in canonical order, checked-release each still-parent-owned stage and
   clear its corresponding Eshkol carrier raw-pointer slot exactly once if
   native prepare did not transfer it.
6. For each P1 stage, take the exact authority, clear its parent cell,
   checked-release the raw owner, and clear its carrier raw-pointer slot.
7. Drop copied controls.
8. Clear/unlink the parent, then set the trainer to `idle/#f`.

Cleanup continues after the first recoverable cleanup defect and never retries
consumed authority. The original operation error remains primary unless cleanup
finds an admitted ownership/provider invariant defect; that defect is reported
as `internal`. Checked I2 release or checked owned-release failure is impossible
under admitted state and fail-stops instead of becoming a recoverable cleanup
error.

No abort helper is called after phase `committing`.

## 6. Reproducibility ownership

| State | Authority before restore | Validation | Successful publication |
|---|---|---|---|
| 14 parameter values | detached C2/P1 | exact path/shape/provider/tie | I2 commit to existing owners |
| 28 moments | detached C2/O2 | exact row/shape, finite; second moment nonnegative | I2 commit to existing owners |
| O2 per-row options | detached O2 projection | exact paths/groups and binary32 options | O2 native commit |
| clip and scheduler | detached O2 projection | kind, maximum/minimum, warmup/total, target `U` | O2 native commit plus logical slot 2 |
| RNG | detached C2 controls | Philox version/key and fixed zero counters | trainer store after D2 |
| tokenizer | detached C2 controls + lease | fingerprint and vocabulary equality | validation only; never replaced |
| X1 config | detached C2 controls + lease | canonical bytes/fingerprint/projection equality | validation only; never replaced |
| current cursor | detached C2 controls | D2 identity/range/reachability | D2 commit |
| epoch-start cursor | detached C2 controls | D2 identity/range/reachability | trainer store after D2 |
| `T/U/E` counters | detached C2 controls/O2 | cross-count and reachability equations | O2 publishes `U`; trainer publishes `T/U/E` |
| 17 modes/profile/compiler/provider | live lease/component identity | exact equality/all-train | validation only |

Accumulation state, gradients, a live batch/view, evaluation mode, model frames
or outputs, arbitrary profiles, non-CPU/non-f32 tensors, and competing optimizer
operations remain unsupported and reject before mutation.

## 7. Implementation partitions after lease acceptance

The minimal dependency order is:

1. **Lease successor (accepted):** repair allocation behavior, extend canonical
   trainer
   controls, freeze authenticated restore transitions, selectively compose with
   the accepted integration head, and pass the full failure-prefix matrix.
2. **Private source binding (accepted):** fixed I2/O2 extern bindings and status
   mapping pass public/native symbol isolation and exact ownership-transfer
   evidence; no composer or public entry is implied.
3. **Composer:** implement detached staging, fixed 14-prefix composition,
   cross-state validation, seal, tail, and phase-dependent abort.
4. **Focused transaction evidence:** exhaustive allocation/publication prefixes,
   forged/stale authority negatives, complete precommit preservation, sealed-tail
   instrumentation, and 1,024/8,192 lifetime horizons.
5. **Trajectory evidence:** uninterrupted versus checkpoint/resume across D2 EOS
   for `A=1` and `A>=2`, including abort/retry immediately before seal.
6. **Supported integration:** #121 acceptance, affected package gates, exact
   native coexistence, full supported CI, independent exact-head review, and only
   then public trainer/facade adoption.

No partition may claim full trainer restore, exact resume, flat total retention,
or public training before the corresponding later evidence exists. The accepted
I2 restore retains a 64-byte terminal tombstone per restore and O2 retains 1,464
bytes per terminal restore; total retained control is linear.
