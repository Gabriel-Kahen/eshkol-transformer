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

Enrollment is globally exclusive, not merely unique inside one record. Both the
first identity admission and the final pre-store recheck reject if any existing
trainer record shares the model/P1 root, any of the fixed 17 module identities,
any of the 14 handles or carrier identities, the optimizer source record, or the
dataset state. Exact repeated `trainer-create` and partial overlap therefore
cannot mint a second token for one receiver.

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
| O2 `optimizer-create` | Resolves all P1 handles, allocates moments, and stores the handles in source/native optimizer records (`native/o2_wave2_extension.esk:615-677`; `native/o2_optimizer.c:67-78`) | Rejects duplicates only inside the new optimizer | The aggregate-installed private hook calls the lease overlap check immediately after exact unique-handle validation and before projection-budget, provider, or builder work |
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
 canonical-handles[14], parameter-carrier-identities[14],
 optimizer-shell, exact-O2-source-record,
 dataset-shell, exact-D2-state,
 reusable owners and trainer controls ...]
```

`tr3-trainers` is one root-owned append-only list. A separate immediate boolean
`tr3-acquisition-active?` is set only around acquisition. The outer recovery
handler is installed before the boolean changes. A reentrant `trainer-create`
rejects while it is true; the boolean retains no record, token, component, or
resource, grants no component authority, and is cleared on every failure and
after successful enrollment.

The complete trainer record, public shell, token, fixed module/handle/carrier
vectors, trainer resources, and the future cons cell are allocated and promoted
before enrollment. The future cons cell retains the exact registry head observed
after initial overlap admission. The final recheck requires that head to remain
identical, repeats every component/overlap/idle check, and then commit is:

```scheme
(vector-set! tr3-trainers 0 preallocated-next-registry)
```

The public shell is read back from the now-rooted canonical record, the
acquisition guard is cleared, and the shell is returned. Those are fixed
nonallocating stores/reads; no later promotion, provider call, callback, or
recoverable validation exists. Every failure before the registry store clears
the acquisition guard, releases only newly created trainer resources, and leaves
the model, optimizer, and dataset publicly usable and unleased. There is no
abort after successful enrollment because A0 has no destroy operation.

The token is never returned, placed in metrics/checkpoints, or accepted through
a public wrapper. Exact `eq?` identity with the token stored in the canonical
trainer record is necessary but not sufficient: each internal admission also
requires the exact record, its fixed phase, component identities, and no competing
active plan. The public trainer shell is not itself a lease token.

### 3.2 P1 fixed-set capture without a parameter tree

`trainer-create` must not call public `module-parameters`. The current P1 wrapper
creates and publishes a fresh append-only tree shell on every call
(`internal/p1/lib/transformer/module.esk:4255-4262`), and O2 retains paths,
aliases, and handles rather than that originating tree
(`native/o2_wave2_extension.esk:649-657`). Such a shell has no release operation
and would make a later failed acquisition grow P1 controls.

The TR3 build instead coordinates one append to the canonical P1 trusted
surface. The E3-P1 contract reserves slots 64 through 68; this lease/C2 seam
adds read-only slots 69 and 70 after them, for a 71-slot surface. If the E3-P1
slot disposition changes, root must refreeze one joint numbering before either
consumer implements. There is no copied P1 source, alternate module, or lexical
load (`docs/e3/E3_P1_MODES_CONTRACT.md:10-25`). The two calls are:

```text
(tr3-p1-fixed-capture-internal
  model-root exact-m3t-handles module-slots handle-slots carrier-slots) -> i64
(tr3-p1-fixed-recheck-internal
  model-root module-slots handle-slots carrier-slots) -> i64
```

Capture requires exact preallocated destination vectors of lengths 17, 14, and
14. Both calls return an immediate exact i64: 0 `OK`, 1 `INVALID_IDENTITY`,
2 `INVALID_STATE`, 3 `UNSUPPORTED_PROFILE`, or 4 `INVARIANT`; neither allocates
a list, vector, shell, exception, or registry cell. The outer trainer boundary
maps a nonzero status before enrollment. Capture scans the existing P1 shell
registry for the root and existing module/handle shells; it never calls
allocating `shell-for-raw`,
`finalization-plan`, flatten/sort, or a public observation.

The exact module order is:

```text
0 root; 1 blocks; 2 blocks/0; 3 blocks/0/attention;
4 blocks/0/attention/key; 5 blocks/0/attention/output;
6 blocks/0/attention/query; 7 blocks/0/attention/value;
8 blocks/0/ffn; 9 blocks/0/ffn/down; 10 blocks/0/ffn/up;
11 blocks/0/norm1; 12 blocks/0/norm2; 13 head; 14 norm_final;
15 position_embedding; 16 token_embedding.
```

The parent indices are `#(-1 0 1 2 3 3 3 3 2 8 8 2 2 0 0 0 0)`. Capture
requires all 17 distinct existing module shells/raw nodes, exact child names and
edges, no extra or missing child, root node count 17, every finalized flag set,
and every mode exactly `train` (`docs/e3/E3_P1_MODES_CONTRACT.md:62-68`). It checks the
14 unique M3T handles in model-entry slot 8 against these parameter paths, in
order:

```text
blocks/0/attention/{key,output,query,value}/weight;
blocks/0/ffn/{down,up}/weight;
blocks/0/norm1/{bias,weight}; blocks/0/norm2/{bias,weight};
head/weight; norm_final/{bias,weight}; position_embedding/weight.
```

`token_embedding/weight` must be the same handle as index 10 `head/weight`; all
other handle pairs are distinct. The 14 shapes in that order are
`#((4 4) (4 4) (4 4) (4 4) (4 8) (8 4) (4) (4) (4) (4) (256 4) (4) (4) (2 4))`.
Every row must also have `f32`, `cpu`, and the P1 provider identity. For each
exact existing handle, capture records its existing
`parameter-handle-tensor-identity-internal` carrier identity
(`module.esk:2666-2667`).
It publishes no tree/state shell, changes no P1 registry or module field, and
grants no tensor mutation or raw carrier operation.

Recheck authenticates the same root, existing shell/raw bindings, finalized
17-module parent/child identities, retained handle shells,
carrier identities, and `train` modes directly against the captured vectors.
Finalized P1 topology cannot acquire another child or leaf. TR3-C's accepted
`tr3-fixed-train-modes-check-internal trainer` first authenticates the canonical
trainer record, token, phase, and active plan, then invokes this recheck; it does
not create a second P1 token or registry. The one trainer record and its lease
token remain the only cross-component authority.

Slots 69 and 70 are validators only. They do not use or extend E3's mode ledger,
retain a caller, or create a P1 lease. Every inheriting P1 archive receives the
same canonical append and reviewed private-manifest delta; the public P1 provide
surface and C ABI remain unchanged.

### 3.3 Fixed private admissions

The minimal same-aggregate logical seam is **PROPOSED** as:

```scheme
(tr3-lease-step-model-internal trainer-record lease-token)
(tr3-lease-step-optimizer-internal trainer-record lease-token)
(tr3-lease-step-dataset-internal trainer-record lease-token)
(tr3-lease-commit-optimizer-internal trainer-record lease-token o2-plan)
(tr3-lease-evaluation-modes-internal trainer-record lease-token evaluation-frame)
(tr3-lease-snapshot-internal trainer-record lease-token)
(tr3-lease-restore-internal trainer-record lease-token restore-record)
```

Each has a phase literal and exact active-plan relation compiled into its body and
returns only its already-retained exact component record or `#t`. There is no
caller-supplied expected phase, generic kind, operation, selector, handle, or
destination argument. These helpers are lexical, uninstalled, and absent from
public strings and headers. TR3-C's fixed one-argument mode checker remains its
accepted facade: it obtains the canonical record and stored token internally and
delegates to the fixed snapshot or restore admission rather than accepting a
public token.

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

The exact trainer phase vocabulary stored in the canonical record is `idle`,
`step`, `commit`, `evaluate`, `snapshotting`, and `restoring`. The accepted
trainer proposal's table labels `snapshot` and `restore` map to the later
accepted TR3-C source states `snapshotting` and `restoring`; the shorter labels
are never stored. Admission is fixed as follows:

| Fixed consumer | Required trainer phase | Required `active-plan` relation |
|---|---|---|
| M3 forward, TR3-B prepare/VJP/release, D2 next/stage/release/rollback | `step` | `#f` |
| TR3-O prepare | `step` | `#f`; O2 is idle with `busy == 0` and `active_operation == NULL` |
| TR3-O abort/check/commit | `commit` | `eq?` the exact opaque native O2 plan published by prepare; O2's active-operation/backreference must agree |
| E3 fixed mode transitions/checks | `evaluate` | `#f`; the fixed adapter separately authenticates the exact E3 frame/token |
| TR3-C snapshot P1/O2/D2 observations and mode check | `snapshotting` | `#f` |
| TR3-C source staging, model/O2/D2 prepare/check/abort and mode check | `restoring` | `eq?` the canonical restore parent in `trainer.active-plan` |
| TR3-C sealed model/O2/D2 commit tail | `restoring` | same exact restore parent in its accepted `sealed`/`committing` child phase |

`idle` admits only a public trainer operation's outer transition; no component
mutation adapter admits `idle`. Every nested call reauthenticates the exact
record, stored token, active plan, component identities, and the child phase
shown above. A copied active plan or a token from another canonical trainer
cannot satisfy these relations.

The outer operation changes `idle/#f` to `step/#f`, `evaluate/#f`,
`snapshotting/#f`, or `restoring/exact-parent`. Successful TR3-O prepare first
publishes its exact native plan, then stores that same identity as
`trainer.active-plan` and changes `step` to `commit`. TR3-O commit or abort clears
that plan before returning to `idle/#f`. No row substitutes an Eshkol wrapper or
step record for the accepted native plan identity.

An outer admission covers only one closed, callback-free internal operation. It
does not set a globally forgeable “trusted” flag or disable component busy checks.
The adapter calls fixed lexical names directly. Raw component entry names are
localized and cannot be resolved from the installed package.

### 3.4 Public wrapper gates

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

`optimizer-create` is special: immediately after its existing exact unique-handle
validation and before `o2-state-projection-budget!`, provider preflight, or
native builder staging, it invokes one private hook with the already-derived
unique handle list. In the standalone O2 package the hook cell is uninstalled
and the invocation is a no-op. During final TR3 aggregate initialization, before
any public facade can be reached, a one-time private installer changes that cell
from `uninstalled` to the fixed lexical
`tr3-o2-create-overlap-check-internal/1`. That closure returns immediate i64 0
for no overlap or 1 for overlap; the O2 call site maps 1 to `invalid-state`
before projection-budget work. A missing installation prevents TR3
facade publication; a second installation rejects.

The concrete source seam is one private
`o2-install-tr3-create-overlap-check-internal!/1` plus the one-argument hook
invocation. The installer is callable only from the source-composed aggregate
initializer and accepts the exact aggregate-fixed closure; it is never installed
through a public operation or selected by a runtime caller.

The fixed check scans `tr3-trainers` directly and returns 1 if any identity
overlaps. It accepts no token, registry, selector, or operation kind,
performs no allocation or publication, and is never returned to a caller. The
cell is only a source-composition boundary needed because a thin outer C-ABI
wrapper cannot reach local `handles`; it is not a caller-selected callback or a
lease authority. The installer, cell, and fixed closure remain absent from
public provides, headers, symbols, and public strings. This catches a tree
produced before enrollment as well as a new tree returned afterward. The exact
insertion point is after current `native/o2_wave2_extension.esk:631` and before
line 632.

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
2. Allocate the fixed module/handle/carrier vectors and call
   `tr3-p1-fixed-capture-internal`; require the exact fixed set described in
   section 3.2 and M3T model-entry slot 8 in order. Publish no parameter tree.
   Scan the current trainer registry and reject any model/root/module/handle/
   carrier/O2/D2 overlap.
3. Require the repaired M3 global call guard idle, model entry frame slot 10
   false, and no live M3 `model-output` or `model-logits` entry owned by this
   model. Reject every live caller-created M3T workspace for this owner. An
   existing M3-private cached workspace is allowed only when its cache row is
   live, its source/native workspace is idle, and it has no graph association.
4. Require the exact supplied O2 source record to contain the same 14 handle
   identities in canonical order, completed updates zero, absent gradients,
   and the shared #119 invariant `busy == 0 && active_operation == NULL`. Scan
   every other exact O2 source record and reject if any handle overlaps.
5. Require D2 open, `state[10]` exactly `'none'` or `'released'`,
   `state[15] == 0`, no unpublished/live batch, and native
   `current_batch == NULL`. Every other status, including `'live'`, rejects.
   Both the fresh `'none'` state and the normal post-release `'released'` state
   are idle; closed is rejected by open-state authentication. Reuse the accepted TR3-C trainer
   D2 idle binding once implemented; this lease contract adds no third native
   idle probe. Current E3-D2 evidence proves the underlying registry fact but
   does not grant trainer authority.
6. Allocate/promote every trainer-lifetime object, lease record, token, result
   shell, error/cleanup cell, and preallocated registry successor. Run the
   allocation-free P1 recheck; recheck every component/idle fact, every O2
   sibling, every trainer overlap, and exact registry-head identity under the
   serialized source model.
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
- its compiled-in trainer phase and exact active-plan relation from section 3.3;
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

The canonical P1 delta appends only trusted-surface slots 69 and 70 after the
reserved E3-P1 slots and updates every affected private manifest. The canonical
O2 delta adds only the uninstalled hook cell, one-time installer, and exact call
site from section 3.4. Packaging evidence must pin the predecessor/source hashes,
prove the final slot indices and initialization order, and show that no duplicate
module, shadow source, or alternate registry enters the closure.

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
- repeat `trainer-create` with the exact same tuple and with each partial overlap
  (model/root, one fixed module, one handle/carrier, O2, and D2); initial and
  final recheck both reject without minting a record/token. Inject same-thread
  acquisition reentry and registry-head substitution; the outer call must not
  overwrite or unlink the inner/current registry head;
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
  cover every row of the phase/active-plan matrix. Token, active-record, child
  plan, phase, and component substitution reject before raw entry;
- P1 capture/recheck publishes no tree/state shell, changes no P1 live/retired
  count, and both calls are allocation-free. They detect first/middle/last module,
  handle, carrier, finalized-topology, and mode substitution; TR3-C snapshot and
  restore use that exact retained fixed set and the one trainer token;
- acquisition succeeds from both D2 `'none'` and normal idle `'released'`, and
  rejects live/borrowed/native-current-batch and closed states with cursor,
  generation, cache, and source/native counts unchanged;
- retained pre-enrollment P1 trees still let the installed TR3 O2 hook reject
  overlap before provider preflight, moment/builder allocation, or O2 registry
  publication. With the hook uninstalled, standalone O2 preserves its public ABI
  and predecessor operation results; review records its exact private-manifest
  delta;
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
  localization, hostile-link, duplicate-registry, canonical P1 slot numbering,
  O2 hook initialization/private-manifest, and fresh-AOT gates.

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
- one reviewed aggregate/export manifest proving all raw authorities private;
- the jointly reviewed canonical P1 slots 64 through 70 and the reviewed O2
  one-time private hook proving the exact capture and pre-projection-budget
  overlap boundaries without changing either public ABI.

No additional native lease ABI, P1 handle mutation API, I2 plan kind, optimizer
state field, D2 cursor field, or C2 representation is required. If implementation
cannot install the public wrapper gates while keeping raw entries localized in
one aggregate, it must return that concrete build/symbol conflict to root rather
than weaken handle overlap, add a public token, or hold a permanent tensor pin.
