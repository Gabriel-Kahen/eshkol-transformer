# TR3 model, optimizer, and dataset lease proposal

Status: **decision-ready proposal only**. This document adds no public API,
runtime authority, or completion claim. Every interface marked **PROPOSED** must
be implemented, independently reviewed, and accepted before the public trainer
may consume it.

Source baseline is `origin/main` at
`33a54ef7256f43d9e1ce82152915f7bded51bc24` (PR #112 merge). The accepted TR3-C
contract is `da872f04e60aa998e33cd3fcd490c2708b9cddaf`; the accepted TR3-O proposal
is `6872807b20dc3ef384cf4a8a4ad2d44fbffd243e`. This proposal refines only the
exclusive lease assumption in section 4 of `TR3_TRAINER_PROPOSAL.md`. It does not
change either accepted transaction contract or broaden TR3 completion.

## 1. Decision

Use one source-level logical lease owned by the eventual source-composed TR3
aggregate. Do **not** hold an I2 borrow, scoped guard, copy plan, gradient plan,
reset plan, M3 frame, or O2 busy bit for the trainer lifetime.

The minimal implementation is one append-only private trainer registry. Each
authenticated trainer record is also the lease record for exactly one M3/P1
model, one O2 optimizer, one D2 dataset, and the model's 14 unique P1 parameter
handles. It retains a fresh unexported lease-token identity. A single final
registry-head assignment enrolls the whole set after every fallible allocation,
promotion, validation, and component-idle check has completed.

The final TR3 aggregate replaces the exported mutating component entries with
lease-aware wrappers. The accepted component implementations remain localized
and unchanged behind those wrappers. Fixed trainer-private adapters authenticate
the exact trainer record, token, component identities, and phase before invoking
those implementations. No caller-selected callback, operation symbol, raw
pointer, parameter selector, or generic mutation function is added.

This design closes the important preexisting-handle gap. A caller may retain a
parameter tree and every handle before `trainer-create`. Current O2 accepts that
tree again and can construct another optimizer over the same live parameters.
Therefore the lease enrolls the handle identities, `optimizer-create` rejects
any overlap with an enrolled handle set, and acquisition rejects a competing
already-created optimizer sharing any enrolled handle.

No public arity, C2 byte, provider identity, O2 state field, or D2 cursor changes.
There is no trainer destroy in A0, so successful enrollment is process-lifetime.

## 2. Actual authority and mutation inventory

The table records behavior present on the source baseline. “Gate” is a proposed
TR3 aggregate adaptation, not current behavior.

| Receiver and operation | Current source/native mutation path | Current exclusion | Required lease gate |
|---|---|---|---|
| P1 `module-load-state-dict!` | `module-load-state-dict-scoped!` builds assignments, calls provider prepare/commit, then publishes canonical paths/finalization (`internal/p1/lib/transformer/module.esk:3948-4013`) | State-owner use marker and provider plan; no trainer owner | Reject public call for leased root; TR3-C uses its accepted 42-destination transaction |
| P1 `module-train!`, `module-eval!` | `finalization-plan` walks the root and `set-mode-modules!` writes every module mode (`module.esk:2313-2328`, `4015-4028`) | Root authentication only | Reject public call; trainer/E3 fixed adapter requires the exact lease |
| I2-backed `module-zero-grad!` | I2 enumerates `module-parameters`, prepares and commits one gradient reset builder (`native/i2_wave2_root.esk:852-904`) | Transient reset-plan pins only | Reject public call; update clear is owned by accepted TR3-O |
| P1 `module-parameters` | Produces a tree containing the existing handle identities (`module.esk:2612-2643`) | Read-only handle metadata; no optimizer exclusivity | Observation may remain public, but every O2 create must check handle enrollment |
| P1 `module-state-dict` | Clones model state into a detached owner (`module.esk:2909-2931`) | Detached P1 state ownership | May remain public outside unsupported racing calls; it exposes no live storage |
| M3 `model-forward` | Acquires/reuses an M3 workspace, runs the accepted schedule, captures an owned graph, then resets the active frame (`native/m3_model_extension.esk:121-150`) | Global M3 call guard plus model frame slot | Reject public call for leased model; fixed trainer adapter invokes the localized implementation |
| M3 `diagnostic-output-vjp!` | Restores graph, reruns forward/reverse, then commits one 14-row I2 gradient plan (`m3_model_extension.esk:199-239`; `src/eshkol_transformer/m3_model.c:314-349`) | M3 call guard, model/workspace active slots, transient gradient plan | Reject public call for leased graph owner; TR3-B leased adapter supplies authority |
| M3 output/logits operations | Create/release graph leases and logits owners (`m3_model_extension.esk:152-197`; `m3_model.c:269-308`) | Exact M3 registry entry/lifecycle | Gate every graph-bearing operation by its model owner; acquisition requires no live external result |
| M3T diagnostic workspace operations | Create a workspace tied to model entry, set owner frame state, run fixed roles, reset/release (`native/m3t_transport_extension.esk:478-564`) | Exact workspace owner and owner frame slot | Reject public operations for leased owner; no retained external workspace at acquisition |
| O2 `optimizer-create` | Resolves all P1 handles, allocates moments, and stores the handles in source/native optimizer records (`native/o2_wave2_extension.esk:615-677`; `native/o2_optimizer.c:67-78`) | Rejects duplicates only inside the new optimizer | Before native builder creation, reject any handle overlapping a trainer lease |
| O2 `optimizer-step!` | Stages next parameter/moment tensors, commits a `3N` I2 copy plan, then advances `completed_updates` (`o2_optimizer.c:1410-1569`) | Native `busy` during the synchronous call | Reject public call; accepted TR3-O uses exact lease plus its typed plan |
| O2 `optimizer-zero-grad!` | Prepares/commits one reset plan over bound parameters (`o2_optimizer.c:1572-1619`) | Native `busy` during the call | Reject public call; accepted TR3-O owns reset in the update tail |
| O2 `optimizer-load-state!` | Copies `2N` moments, then stores config/update/options (`o2_wave2_extension.esk:716-735`; `o2_optimizer.c:1799-1880`) | Optimizer busy plus detached-state borrow | Reject public call; accepted TR3-C owns joint restore |
| O2 `optimizer-state` | Clones a detached optimizer state (`o2_wave2_extension.esk:693-714`) | Synchronous native busy, detached result | May remain public outside unsupported racing calls; it cannot mutate the optimizer |
| D2 `token-dataset-next-batch` | Allocates/fills/seals a native batch, advances core ordinal, stores generation and `live` (`internal/d2/lib/d2_dataset.esk:747-828`) | One current batch and one synchronous view | Reject public call; trainer-private D2 adapter owns traversal |
| D2 `token-dataset-seek!` | Writes cursor ordinal and clears status/cache (`d2_dataset.esk:839-849`) | Rejects while a batch is live | Reject public call; rollback/restore use fixed trainer-private plans |
| D2 batch release | Invalidates source capabilities and destroys current native batch (`d2_dataset.esk:925-954`) | Exact dataset/generation and no active view | Gate by owning dataset; trainer cleanup invokes localized raw release |
| D2 `token-dataset-close!` | Releases a live batch if needed, clears retained state, closes native owner (`d2_dataset.esk:956-983`) | Exact dataset shell | Reject public call for leased dataset |
| D2 cursor and batch accessors | Cursor returns a detached snapshot; tensor views exist only in fixed synchronous callback (`d2_dataset.esk:830-923`) | Exact shell/generation and one live view | Cursor observation may remain public; batch/view access is reachable only through trainer-owned batch identity |
| TR3-C snapshot/restore | Accepted contract snapshots through real P1/O2/C2 owners and restores through one 42-destination plan plus typed O2/D2 controls | Trainer phase/active plan assumed; O2 owns shared busy/backreference | Consume this lease token; do not add another receiver owner |

The public P1 handle accessors do not expose the underlying carrier. The carrier
is available only through `parameter-handle-tensor-internal` (`module.esk:2660-2667`).
Nevertheless, a retained public tree is mutation-enabling because public
`optimizer-create` consumes those same handles. The create gate is mandatory;
blocking only P1 mutator names is insufficient.

Current O2 records have no cross-optimizer exclusivity. Source records retain
their handle list at slot 5 (`o2_wave2_extension.esk:649-657`), and native records
retain both the P1 handle and parameter identity (`o2_optimizer.c:43-51`). The
existing duplicate check applies within one builder, not across `et_o2_optimizers`.

## 3. Proposed record and private interfaces

### 3.1 One record, one commit store

**PROPOSED** logical record fields are:

```text
[tr3-trainer-tag, self, public-shell, phase, active-plan,
 exact-lease-token,
 resolved-config, tokenizer-identity,
 model-shell, exact-M3T-model-entry, fixed-17-P1-modules,
 canonical-parameter-tree, canonical-handles[14],
 optimizer-shell, exact-O2-source-record,
 dataset-shell, exact-D2-state,
 reusable owners and trainer controls ...]
```

`tr3-trainers` is one root-owned append-only list. The complete record, public
shell, token, fixed module vector, handle vector, trainer resources, and the
future cons cell are allocated and promoted before enrollment. Commit is:

```scheme
(vector-set! tr3-trainers 0 preallocated-next-registry)
```

The public shell is returned from the now-rooted canonical record with no later
allocation, promotion, provider call, callback, or validation. Every failure
before that store releases only newly created trainer resources and leaves the
model, optimizer, and dataset publicly usable and unleased. There is no abort
after successful enrollment because A0 has no destroy operation.

The token is never returned, placed in metrics/checkpoints, or accepted through
a public wrapper. Exact `eq?` identity with the token stored in the canonical
trainer record is necessary but not sufficient: each internal admission also
requires the exact record, expected phase, component identities, and no competing
active plan. The public trainer shell is not itself a lease token.

### 3.2 Fixed private admissions

The minimal same-aggregate logical seam is **PROPOSED** as:

```scheme
(tr3-lease-model-internal trainer-record lease-token expected-phase)
(tr3-lease-optimizer-internal trainer-record lease-token expected-phase)
(tr3-lease-dataset-internal trainer-record lease-token expected-phase)
(tr3-lease-fixed-modes-internal trainer-record lease-token expected-phase)
```

Each returns only its already-retained exact component record or `#t`. There is
no generic kind, operation, selector, handle, or destination argument. These
helpers are lexical, uninstalled, and absent from public strings and headers.

Fixed consumers are adapters rather than new component algorithms:

- `tr3-m3-forward-internal` authenticates the model lease and calls the accepted
  localized M3 forward implementation.
- the full-trainer wrapper around accepted TR3-B `/6` authenticates the same
  model lease and current trainer step before invoking TR3-B; TR3-B's objective,
  M3 schedule, VJP, and release protocol remain unchanged;
- `tr3-o2-step-clear-prepare-internal` authenticates the optimizer lease before
  calling accepted TR3-O `/4`; commit/abort still authenticate TR3-O's own typed
  plan and shared O2 backreference;
- accepted `o2-optimizer-restore-prepare-internal` receives this exact token in
  its existing `optimizer trainer-lease ...` position. TR3-C's fixed model
  compositor and `tr3-fixed-train-modes-check-internal` authenticate this same
  record; and
- fixed D2 next/release/rollback/restore adapters authenticate the dataset lease
  before invoking the localized accepted operation or typed plan.

An outer admission covers only one closed, callback-free internal operation. It
does not set a globally forgeable “trusted” flag or disable component busy checks.
The adapter calls fixed lexical names directly. Raw component entry names are
localized and cannot be resolved from the installed package.

### 3.3 Public wrapper gates

The eventual aggregate must export the existing public C ABI names through
lease-aware wrappers. It must localize the accepted raw component spellings and
all lease helpers.

For an enrolled component, external calls reject `invalid-state` before the
underlying operation changes state:

- P1: `module-load-state-dict!`, `module-train!`, `module-eval!`,
  `module-zero-grad!`;
- M3: `model-forward`, `diagnostic-output-vjp!`, and every output/logits action
  whose exact M3 entry names an enrolled model owner;
- M3T: workspace create/begin/reset/release and every fixed forward/reverse role
  whose exact workspace entry names an enrolled model owner;
- O2: `optimizer-step!`, `optimizer-zero-grad!`, `optimizer-load-state!`;
- D2: `token-dataset-next-batch`, `token-dataset-seek!`,
  `token-dataset-close!`, and `token-batch-release!` for an enrolled dataset.

`optimizer-create` is special: after its existing exact tree/unique-handle
validation and before provider/native builder staging, it scans every enrolled
14-handle set and rejects if any identity overlaps. This catches a tree produced
before enrollment as well as a new tree returned afterward.

Detached observations may remain available when no unsupported concurrent call
is racing the trainer: P1 metadata and `module-state-dict`, O2
`optimizer-state`, D2 cursor, model profile/initializer, and detached-state
release. They expose no live tensor carrier. This permission depends on the
optimizer-create overlap gate and on the final symbol audit proving that
`parameter-handle-tensor-internal` and raw native mutators are not exported.

## 4. Acquisition preflight

`trainer-create` validates and prepares in this order, without enrolling a
partial lease:

1. Authenticate the exact M3T model entry for the P1 root shell, accepted fixed
   profile, resolved config, tokenizer, O2 shell, and open D2 shell.
2. Use P1's existing bounded `finalization-plan` to retain the exact 17 module
   identities; require every mode `train`. Use the existing M3T schedule data to
   require the 15 paths, 14 unique handles, fixed shapes, and the one head/token
   tie. The 14 identities must equal M3T model entry slot 8 in order.
3. Require the repaired M3 global call guard idle, model entry frame slot 10
   false, and no live M3 `model-output` or `model-logits` entry owned by this
   model. Reject every live caller-created M3T workspace for this owner. An
   existing M3-private cached workspace is allowed only when its cache row is
   live, its source/native workspace is idle, and it has no graph association.
4. Require the exact supplied O2 source record to contain the same 14 handle
   identities in canonical order, completed updates zero, absent gradients,
   and the shared #119 invariant `busy == 0 && active_operation == NULL`. Scan
   every other exact O2 source record and reject if any handle overlaps.
5. Require D2 open, `state[10] == 'none'`, `state[15] == 0`, no unpublished/live
   batch, and native `current_batch == NULL`. Reuse the accepted TR3-C trainer
   D2 idle binding once implemented; this lease contract adds no third native
   idle probe. Current E3-D2 evidence proves the underlying registry fact but
   does not grant trainer authority.
6. Allocate/promote every trainer-lifetime object, lease record, token, result
   shell, error/cleanup cell, and preallocated registry successor. Recheck the
   component identities and idle facts under the serialized source model.
7. Perform the one registry-head assignment and return the canonical shell.

The M3 source registry makes the graph/workspace checks possible without a new
native lease ABI. Entries already retain kind, lifecycle, model entry, and frame
state (`native/m3t_transport_extension.esk:85-89,184-202`). M3's own workspace
cache separately records the owning model and cache lifecycle
(`native/m3_model_extension.esk:13-78`).

O2's current source record authenticates the optimizer shell and handle list,
while current native `et_o2_require_optimizer` proves provider validity and
`busy == 0` (`native/o2_optimizer.c:620-645`). The #119 implementation owns the
additional exact `active_operation` invariant. Lease acquisition observes that
shared invariant; it does not set either field.

The D2 source check alone cannot prove native `current_batch` is null. The
accepted TR3-C contract already assigns the required fixed native registry probe.
Lease implementation is dependency-gated on that seam rather than inventing a
second one.

## 5. Why a permanent component pin is wrong

M3 VJP and O2 mutation are compatible only because their I2 ownership is
transient and ordered.

M3 graph capture clones saved primals and retains the 14 parameter and handle
identities; it does not pin live parameter-value storage for the graph lifetime
(`src/eshkol_transformer/m3_model.c:187-218`). Before contribution, graph checking
opens scoped guards on live values and saved clones, then closes all guards
before returning (`m3_model.c:242-267`). Contribution prepares a 14-row gradient
plan, commits and releases it, then resets the M3 frame before success returns
(`m3_model.c:314-349`).

Accepted TR3-B closes its own scoped numerical views before it calls VJP. Its
caller then releases the accepted output/logits owners and D2 batch. At that
point there is no live frame, view, gradient plan, or graph lease.

Accepted TR3-O then prepares its copy plan over parameter values/moments and its
reset plan over parameter controls/gradients. Its coexistence witness proves
those transient pin sets can be retained together. O2's current synchronous
step releases its copy plan before clearing `busy` (`native/o2_optimizer.c:
1541-1569`); zero-grad likewise releases its reset plan (`1604-1619`). #119
extends that ordering under one typed operation without adding a lifetime pin.

Accepted TR3-C expressly avoids ordinary M3 call pins while preparing its 42
destination copy transaction. Its fixed model compositor authenticates the
trainer/model/handles under this logical lease, appends the 14 destinations,
and lets the one I2 plan own the only live pins.

A lifetime parameter pin would conflict with O2 and TR3-C destination-plan
preparation. A lifetime O2 `busy` value would also make #119 and TR3-C impossible.
The proposed logical source lease therefore controls who may start an operation
without occupying the operation's native borrow/plan mechanism.

## 6. Reentry, failure, and restore rules

The lease never bypasses an existing guard. Every internal adapter requires:

- exact registered trainer record and exact stored token;
- expected trainer phase and active-plan relation;
- exact retained component/source identities;
- component-local idle or exact typed-plan state required by that operation; and
- no same-thread trainer entry already active except the exact nested fixed call
  authorized by its parent phase.

Public mutation rejects solely because the component is enrolled; trainer idle
does not make it externally mutable. Internal mutation with a copied token,
another trainer's token, wrong phase, substituted component, stale graph/batch,
or competing typed plan rejects before calling the raw component operation.

TR3-O and TR3-C remain mutually exclusive through O2's one `busy` plus
`active_operation` backreference. `busy=1, active_operation=NULL` remains the
accepted legacy synchronous state; `busy=1, active_operation=exact-plan` is a
long-lived TR3-O or TR3-C reservation; idle requires both clear. The logical
trainer lease is neither state.

TR3-C may restore checkpoint update `U` over live receiver update `R`, including
`U != R`, exactly as its accepted contract states. The lease authenticates the
receiver set only. It does not require checkpoint values or counters to equal
the current receiver, does not add rollback storage, and does not change the
TR3-C write order. Before the first write, abort preserves `R`; the fixed tail
publishes `U` to O2 and trainer.

Every recoverable trainer operation restores its phase/active-plan and component
local exclusions before returning. It never clears the process-lifetime lease.
After an irreversible TR3-O or TR3-C write, their accepted fail-stop rules apply;
the lease layer must not catch and translate that condition into retryable state.

The project registries require caller serialization. Racing calls from multiple
threads, signal handlers, or forked processes remain unsupported; the lease is
not a mutex and makes no data-race-safety claim.

## 7. Aggregate privacy requirements

The lease works only in the single TR3 source aggregate already required by
TR3-A. Linking a completed M3, O2, D2, or C2 sibling archive with an independent
registry would make source identity scans meaningless.

The package gate must prove:

1. each P1/I2/M3/M3T/O2/D2/C2 lineage is source-composed once;
2. existing public component C ABI spellings resolve to the lease-aware wrappers;
3. raw component mutators, trainer registry, record tags, token helpers, fixed
   admissions, component source records, and native pointers are localized and
   absent from headers/public strings;
4. no public vector/facade contains a private lease helper;
5. C2 remains the single persistence-policy authority and the accepted C2 policy
   projection remains unchanged; and
6. TR3-C model/O2/D2 compositors and TR3-O plans consume this registry rather
   than creating a second trainer or component owner.

Separate already-built component archives may still exist as independent
products. Their shells cannot authenticate in this aggregate. The public TR3
package must not link them into its runtime closure.

## 8. Required evidence before trainer implementation

The smallest dependency-ready implementation unit is the logical lease registry,
acquisition preflight, public mutation gates, and fixed internal admissions. It
does not implement a training step, objective, optimizer transaction, snapshot,
or restore.

Focused acceptance must cover:

- retain the exact parameter tree and all 14 handles before enrollment; after
  enrollment, a second `optimizer-create` rejects before moment/builder
  publication and all component bytes/counts remain unchanged;
- create a second optimizer before enrollment; `trainer-create` rejects without
  enrolling anything, and the original model/optimizers/dataset remain usable;
- live M3 output, logits, caller workspace, active frame, D2 batch, D2 view, O2
  busy operation, present gradient, wrong topology/tie/mode, copied shell, and
  foreign token all reject with complete preflight preservation;
- released M3 results and an idle private M3 cache admit acquisition, proving the
  check does not require an unused model when accepted idle state is sufficient;
- every gated public P1/M3/M3T/O2/D2 mutator rejects on an enrolled receiver,
  including batch release and optimizer load, without changing bytes, cursor,
  modes, gradients, moments, or counters;
- each fixed internal adapter succeeds only with the exact token and phase;
  token substitution and component substitution reject before raw entry;
- one genuine accepted M3/TR3-B contribution followed by TR3-O prepare/abort and
  prepare/commit proves no lifetime pin conflict; competing TR3-O/TR3-C plans
  reject through the shared O2 exclusion;
- allocation/promotion failure at every acquisition prefix leaves no registry
  entry and preserves ordinary public mutation; no injected fallible point exists
  after the final registry assignment;
- source/native live counts and component identities before/after failed
  acquisition and public rejection; ASan/UBSan/LSan on the focused native closure;
  and
- exact archive member, exported symbol, undefined symbol, public string,
  localization, hostile-link, duplicate-registry, and fresh-AOT gates.

The evidence must report process-lifetime trainer/lease controls separately from
transient M3/O2/D2/I2 owners and accepted cumulative tombstones. It must not claim
flat total memory or thread synchronization.

## 9. Dependency disposition

Root must freeze this source-level seam before public trainer implementation.
Implementation depends on:

- the repaired M3 call guard runtime adoption; this proposal does not duplicate
  guard ordering;
- the accepted #119 shared O2 `busy`/`active_operation` invariant;
- the accepted #120 trainer D2 idle binding and fixed model/O2/D2 compositors;
- the accepted TR3-B bridge for the unchanged objective/VJP path; and
- one reviewed aggregate/export manifest proving all raw authorities private.

No additional native lease ABI, P1 handle mutation API, I2 plan kind, optimizer
state field, D2 cursor field, or C2 representation is required. If implementation
cannot install the public wrapper gates while keeping raw entries localized in
one aggregate, it must return that concrete build/symbol conflict to root rather
than weaken handle overlap, add a public token, or hold a permanent tensor pin.
