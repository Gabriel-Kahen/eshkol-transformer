# TR3-O optimizer update-and-clear transaction proposal

Status: proposal for root acceptance; no runtime authority is implemented here.

Source baseline: `19f404cf21632944e1f2d5d4959e1240f9a79f5f`.
Related work: issue #119, TR3-C issue #120, SHARED-R2 issue #114, and the
broader trainer proposal at `3e28db6c9d1f97bd9ce9d6f2d05be51182a8bd9d`.

## Decision

TR3-O can use the two accepted I2 plans without a new I2 mutation API.

The existing copy plan pins only its `3N` value/moment destinations and staged
sources.  The existing reset plan pins only the same `N` parameter controls and
their gradient tensors.  Actual source inspection and the native witness in
`tests/o2/test_tr3o_i2_plan_coexistence.c` prove that both plans can be prepared
and retained together for one parameter.  The witness also proves mutation-free
abort, retry, allocation-disabled copy-then-reset commit, unchanged parameter
value identity, canonical absent gradient metadata, exact-zero gradient backing,
and return of every live I2 count to baseline.

The second I2 commit and cleanup helpers retain defensive status branches.  They
are nevertheless sufficient because their handles never leave the registered
outer O2 plan, caller serialization is required, the optimizer remains busy, the
exact plan is its active operation, and the two pin sets are disjoint.  Before
the outer plan enters `COMMITTING`, every recoverable check is complete.  From
that transition onward, any nonzero child status is a provider invariant defect
and invokes the fixed fail-stop path; it is never formatted, promoted, ignored,
or returned as an Eshkol condition.  This is stronger than the required rule
that no recoverable failure follows the first live write.

No I2 ABI, K1 provider row, O2 public operation, C2 byte, optimizer algorithm, or
logical optimizer-state field changes.

## Exact private interfaces

The future TR3 aggregate compiles these declarations only under its private O2
transaction feature gate.  They do not enter an installed header.

```c
typedef struct et_o2_trainer_step_clear_plan
    et_o2_trainer_step_clear_plan;

int32_t et_o2_trainer_step_clear_prepare_v1(
    et_o2_optimizer *optimizer,
    uint64_t expected_completed_updates,
    uint64_t expected_contribution_count,
    uint32_t expected_normalization_weight_bits,
    et_o2_trainer_step_clear_plan **plan,
    et_o2_error_v1 *error);

int32_t et_o2_trainer_step_clear_commit_v1(
    et_o2_trainer_step_clear_plan *plan,
    et_o2_error_v1 *error);

int32_t et_o2_trainer_step_clear_abort_v1(
    et_o2_trainer_step_clear_plan *plan,
    et_o2_error_v1 *error);
```

The localized native-to-Eshkol bridge has exactly these signatures:

```c
void *et_tr3_private_o2_trainer_step_clear_prepare_v1(
    void *optimizer, int64_t expected_completed_updates,
    int64_t expected_contribution_count,
    int64_t expected_normalization_weight_bits);
int64_t et_tr3_private_o2_trainer_step_clear_commit_v1(void *plan);
int64_t et_tr3_private_o2_trainer_step_clear_abort_v1(void *plan);
```

The exact source-private logical calls are:

```text
o2-trainer-step-clear-prepare-internal/4
  (optimizer expected-completed-updates expected-contributions
             expected-normalization-weight-bits)
o2-trainer-step-clear-commit-internal!/1 (plan)
o2-trainer-step-clear-abort-internal!/1  (plan)
```

The weight argument is the positive finite binary32 bit pattern produced by the
trainer's accepted ordered microbatch-weight accumulation.  It is not a later
cast or a separately rounded sum.  Every bound I2 gradient must have that exact
bit pattern and the exact positive contribution count.  The update argument is
the current pre-update `completed-updates` value; a successful commit publishes
exactly `expected_completed_updates + 1`.

The source layer authenticates the exact O2 optimizer shell and argument ranges,
then holds only the returned opaque native plan in its lexical transaction.  It
does not create another registry, expose a raw child plan, or publish a plan in
an installed facade.  The native plan registry is the sole plan identity owner.

## Receiver ownership shared with TR3-C

The future private build adds one conditional `active_operation` backreference
to the opaque native optimizer alongside its existing `busy` bit.  Successful
TR3-O prepare sets both and retains them through commit or abort.  Public step,
zero-grad, state, load, and inspection therefore keep their accepted busy
rejection behavior.

TR3-C restore uses a distinct typed plan and registry, but it must acquire this
same O2 `busy` plus `active_operation` ownership.  Update-clear and restore plans
are mutually exclusive.  Neither workstream may add a second optimizer receiver
registry or a generic operation union exposed to callers.  The backreference is
process-local control state and is absent from O2 logical snapshots and C2 bytes.

## Prepare contract

Prepare performs all work that may allocate, borrow, validate, or return a
recoverable error:

1. Validate the output slot, error storage, exact registered optimizer/provider,
   idle receiver, parameter bound `1..1365`, and arguments.  Contributions are
   in `1..INT64_MAX`; expected weight bits are positive finite binary32; the
   expected current update is in `0..INT64_MAX-1`.
2. Allocate an unpublished outer shell.  Acquire the optimizer's existing busy
   bit and exact active-operation backreference.
3. In canonical unique-parameter order, revalidate P1/I2 identity and require a
   present gradient whose count and weight bits exactly equal the caller's
   values.  Validate finite numerator, parameter, and moments.
4. Compute the next successful-update schedule value, global norm, clipping
   factor, AdamW bias corrections, decoupled decay, and all finite next values
   in the accepted O2 order.
5. Allocate and fill `3N` ordinary I2 staged tensors and ordered assignments
   `[parameter value, exp_avg, exp_avg_sq]` per canonical entry.
6. Prepare one accepted I2 copy plan over those assignments, then one accepted
   I2 reset plan over the same canonical parameters.  The copy plan pins values,
   moments, and stages; the reset plan pins parameters and gradients.  No borrow
   or raw view survives prepare.
7. Register the outer plan as `PREPARED` and publish it only after both child
   plans and every payload are complete.

A failed prepare leaves the output null, every live byte and counter unchanged,
all child pins and temporary storage released, and the optimizer idle with no
active operation.  Any impossible cleanup defect fail-stops rather than returning
a partially cleaned receiver.  Allocation/admission failpoints must cover every
outer table, staged tensor, child-plan allocation, and publication boundary.

The plan exclusively owns:

- its exact optimizer busy lease and active backreference;
- the accepted I2 copy and reset plans;
- all `3N` staged tensors and assignment/parameter tables;
- the prevalidated next update and lifecycle state.

It owns no P1 state, C2 state, provider selector, callback, caller-selected
destination, or generic tensor mutation authority.

## Commit and failure tail

Commit first authenticates the exact registered `PREPARED` plan, its owner,
provider identity, busy bit, active backreference, unchanged current update, and
all retained child handles.  Forged, stale, consumed, aborted, or reentrant input
returns before mutation.  It then changes the outer lifecycle to `COMMITTING`.
No caller error buffer is touched after this transition.

The fixed tail is:

1. Commit the I2 copy plan with a null child error operand.
2. Commit the I2 reset plan with a null child error operand.
3. Store the prevalidated next `completed_updates` value.
4. Release the reset and copy plans, destroy every staged tensor, and free every
   payload table.
5. Clear the optimizer active-operation backreference and busy bit, mark the
   inert outer shell `COMMITTED`, and return the already fixed success status.

The first nonempty copy destination write is the physical commit point.  For an
all-empty tensor set, the first reset metadata write is the physical commit
point.  Conservatively, no recoverable result is possible after entering
`COMMITTING`, even before either physical case occurs.

Every tail helper is accepted nonallocating native code.  Exclusive ownership
makes a child stale/consumed/foreign status, retained borrow, residual pin, or
release failure unreachable on an admitted plan.  An unexpected nonzero status
from either child commit, either child release, or any staged-tensor destruction
calls a private `_Noreturn` fail-stop implemented with `_Exit(134)`.  It performs
no allocation, callback, formatting, E1 raise, result promotion, retry, or
continuation.  Silently ignoring such a status is forbidden.

Commit never clears a caller error record or constructs an error after entering
`COMMITTING`.  The source wrapper receives either a precommit categorized error,
successful integer zero, or process termination.  Its success path returns the
preexisting literal `#t`; it performs no state mutation or publication.

## Abort and tombstones

Abort is valid only for an exact registered `PREPARED` plan.  It authenticates
the complete outer owner before cleanup, marks the plan `ABORTING`, releases both
child plans, destroys stages, frees payload tables, clears the optimizer active
operation and busy bit, and marks the retained shell `ABORTED`.  It never changes
parameters, moments, gradients, or counters.  Impossible cleanup defects use the
same fail-stop rule.

Outer plan addresses are capabilities.  `COMMITTED` and `ABORTED` shells remain
registered inert tombstones so an address is never recycled into a new live
plan.  Their payload pointers are null.  Null, malformed, wrong-kind, foreign,
unregistered, or cross-aggregate tokens are `invalid-argument`.  Exact registered
committed/aborted tokens, double commit/abort, commit-after-abort,
abort-after-commit, or a plan in `COMMITTING`/`ABORTING` are `invalid-state`.
Only one commit or one abort can consume a plan.

Commit and abort deliberately accept the opaque plan value rather than a caller
slot.  This avoids a caller-memory write after the live commit and makes the
native registry lifecycle authoritative.  A copied lexical pointer has no extra
authority: after the first consume, every copy resolves to the same inert
tombstone and is rejected as `invalid-state`.

Live counters distinguish live plans, child plans, stages, and borrows from
cumulative inert tombstones.  Acceptance measures the per-update retired-control
slope at 1,024 and 8,192 updates and reports it; it must not describe reachable
tombstones as leaked live payload.

## Errors and counters

The source layer preserves the existing E1 category mapping.  The native private
contract uses the existing O2 categories and codes, plus no public error kind:

| Case | Category / code |
|---|---|
| malformed argument, null/nonempty output, invalid count/weight range | `invalid-argument / INVALID_OPTION` (or `NULL_ARGUMENT`) |
| source-level unknown optimizer shell or foreign plan token | `invalid-argument / INVALID_HANDLE` |
| recognized busy optimizer or reentrant live plan | `invalid-state / ACTIVE_BORROW` |
| exact consumed/aborted plan | `invalid-state / CONSUMED` |
| expected current update differs | `invalid-state / COUNTER_MISMATCH` (new private code 18) |
| gradient absent | `invalid-state / GRADIENT_ABSENT` |
| contribution or expected-weight mismatch | `invalid-state / GRADIENT_METADATA` |
| nonfinite input/result | `invalid-state / NONFINITE` |
| update would exceed `INT64_MAX` | `invalid-state / COUNTER_OVERFLOW` |
| unsupported float environment | `determinism-unavailable / FLOAT_ENVIRONMENT` |
| allocation failure | `internal / ALLOCATION_FAILED` |
| provider/identity defect before `COMMITTING` | existing `internal` provider code |
| impossible defect after `COMMITTING` | fail-stop; no returned category |

Only successful commit advances `completed-updates` and the derived schedule.
Abort, every prepare failure, and every precommit commit rejection leave it
unchanged.  Gradients clear only in the successful private commit tail; accepted
public `optimizer-step!` continues to retain them, and public
`optimizer-zero-grad!` remains separate.

## Packaging and aggregate closure

Standalone O2 remains exactly six public wrappers, 53 global definitions, 47
package exports, six public renames, one aggregate member, and its accepted source
closure.  The six accepted operations and C2 optimizer-state bytes are unchanged.

Implementation belongs only to the future one-member TR3 aggregate:

- `native/tr3_o2_step_clear_internal.h` and
  `native/tr3_o2_step_clear.inc`, with the include compiled from the same
  `o2_optimizer.c` translation unit only under
  `ET_TR3_O2_STEP_CLEAR_NATIVE`, so it reuses the one actual optimizer registry
  and private arithmetic;
- `native/tr3_o2_step_clear_extension.esk`, loaded after the accepted O2 source;
- the three bridge helpers behind `ET_TR3_O2_STEP_CLEAR_BRIDGE` in the existing
  `native/o2_wave2_package_bridge.c`; the future TR3 package defines the macro
  before its single inherited O2-bridge inclusion and never includes that bridge
  twice;
- required `LOCAL` assertions for every transaction native/bridge symbol;
- no transaction symbol in public exports, public strings, archive index, facade
  renames, K1 discovery, or an installed header;
- negative external-link and public-source-resolution tests;
- exactly one `f32_tensor.c` and `o2_optimizer.c` lineage in source closure, never
  a linked I2/O2/C2 sibling archive.

The feature-gated optimizer layout contains the shared active-operation pointer;
the standalone O2 build does not.  TR3-C #120 must use that same ownership seam.

The predecessor manifests have zero delta: O2 remains 53 defined globals, 47
exports, 53 public strings, six facade renames, 150 undefined entries, 15 Eshkol
closure entries, 29 native closure entries, and one `o2_wave2.o` member; C2
remains 81/75/81 with six renames and 32/53 source/native closure entries.  I2
remains 49 defined and 23 public symbols.  Future TR3 closure alone adds the
three native operations, three `et_tr3_private_o2_trainer_step_clear_*` bridge
helpers, internal header/include, and source extension, all localized.

SHARED-R2 #114 owns the later provider-1.1 shape matcher/evidence block in
`native/f32_tensor.c`.  TR3-O changes no provider code or strings.  Because this
proposal needs no I2 implementation change, it has no source collision with
#114 beyond ordinary future integration against its merged tree.

## Required acceptance evidence after implementation authorization

1. Public predecessor parity: accepted `optimizer-step!` plus
   `optimizer-zero-grad!` is bit-identical to private update-clear for constant
   and linear schedules, clipping below/equal/above boundary, present zero,
   unequal microbatch weights, multiple groups, and final admitted counter.
2. Exact expected-weight negative: all 14 gradients have valid matching counts
   and an internally common weight different from the caller's expected bits;
   rejection changes no byte/counter/identity, then correct-weight retry matches
   a clean run.
3. Metadata negatives at first/middle/last entry: absent, wrong count, wrong weight,
   nonfinite numerator, and nonfinite staged result.
4. Allocation/admission sweep through every prepare allocation and both child
   preparations, with null output, unchanged state, baseline live counts, and
   successful retry.
5. Forged/wrong-kind/cross-aggregate/unregistered, exact stale/consumed, double
   consume, commit-after-abort, abort-after-commit, and reentrant public/private
   operations.
6. Pin attempts against every value, moment, gradient, stage, and parameter class
   while prepared; abort must release all of them.
7. Post-prepare allocation-disabled success and test-only terminal subprocess
   witnesses for impossible defects at copy/reset commit and first/middle/last
   cleanup site; none may return a recoverable error.
8. Exact parameter, gradient, and moment storage identities before/after; gradient
   bytes are positive zero with absent/count-zero/weight-zero metadata.
9. N=14 actual-model coexistence in addition to the N=1 feasibility witness;
   normal, ASan/UBSan/LSan, float-environment, strict-AOT, aggregate privacy, and
   closure gates on the final exact tree.
10. Measured live and retired-control counts at ordinary, 1,024, and 8,192 update
    horizons, with no retained stage payload or active pin after consume.

No full repository CI, supported-platform claim, public trainer claim, restore
claim, end-to-end overfit claim, or merge is part of this proposal checkpoint.

## Independent review disposition

Three independent Sol/high reviews examined the exact current source and the
native witness.

- Source and adversarial review agree that the two existing I2 plans suffice
  when the outer O2 plan has exclusive ownership and every impossible status
  after `COMMITTING` fail-stops.  Their source proof covers child admission,
  disjoint pins, release of consumed plans, stage destruction, and tombstones.
- Packaging review withdrew its initial paired-I2 recommendation after reviewing
  the exclusivity proof.  It confirms zero predecessor manifest delta and the
  future-TR3-only compile gate and localization scheme above.
- Two reviewer drafts repeated the earlier three-input logical prepare.  Root's
  accepted weight-binding correction supersedes that text; this proposal freezes
  the exact four-input source call and four-value native/bridge prepare.
- Reviewer suggestions to pass a plan slot to commit/abort are not adopted.  The
  direct opaque value avoids a caller-slot write in the fixed tail, while the
  retained native tombstone provides the same stale/copy rejection.

No reviewer found a remaining need for an I2 API/provider change.  Runtime
authority still requires root acceptance of this exact proposal.
