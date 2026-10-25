# Private C4 trainable workspace ownership contract

**Proposed next gate; no workspace API or runtime is admitted here.** This
contract depends on compiled admission of the tagged C4 constructor, P1 slot
74, sealed native owner topology and exact per-index P1/I2/native association.
The source candidates `5503bb2b`, `9772674e`, `dd0cb754` and `95f44968`
provide the proposed authorities, not a completed workspace. It adds no
public or seeded-model route.

## Exact owner and lifetime

The input is the exact live `trainable-c4` model shell in the sole C4 model
registry, with its rooted nine-field X1/T1/initializer metadata, fourteen
ordered P1 handles, native owner at entry slot 3 and no construction or active
call at entry slots 9/10. Create requires the model in `train` mode. The new
workspace identity may have a separate workspace registry, but it must not
create another model registry or copy the model into a seeded owner. One model
entry has at most one live workspace. Its slot 10 is the workspace edge while
that workspace lives; the native owner has the matching exclusive active edge.
The current generation call-entry and native context paths must reject those
edges before borrowing parameters. A second workspace or any live same-owner
generation context rejects without changing the first, even when that context
is idle: an idle context retains a cache while the native owner `active` field
and C4 entry slot 10 are both empty. It must be closed before workspace
creation; training cannot leave its cache live under changed parameters.

The proposed private operations are `g3r-c4-workspace-create-internal(model)`,
`g3r-c4-workspace-reset-internal!(workspace)` and
`g3r-c4-workspace-release-internal!(workspace)`. They return an owned opaque
workspace shell, or a typed error, never a parameter array, raw pointer,
borrowed tensor or numerical result. The source-private native owner accepts
only its exact sealed owner and may retain the fourteen existing parameter
identities; its workspace record is distinct from a G3 generation context.

| Authority | Create requirement | While live | Release result |
| --- | --- | --- | --- |
| C4 registry | exact trainable entry, slot 10 empty | slot 10 identifies the sole workspace | slot 10 empty |
| P1/I2 | slot-74 train topology and fourteen exact retained carriers | no observation commit or replacement binding | original handles unchanged |
| native C4 owner | sealed, ready, idle, exact scalar association | exclusive active edge to this workspace | active edge empty |
| generation contexts | no live context bound to this owner, including idle | new context creation rejects | no old cache becomes resumable |
| workspace | fresh shell and native allocation | idle phase only in this gate | removed from the live registry; repeated release rejects as stale |

## Transaction and exclusion

Create validates the complete registry tuple, P1 topology, I2 live carriers,
native owner topology and fourteen scalar bindings before publication. It
also requires no held parameter or tensor lease, no live same-owner native
generation context, no active construction and no prior workspace. The native
context-registry exclusion needs a reviewed read-only same-owner scan; merely
testing `owner->active == NULL` is insufficient. Recheck this exclusion and
all identities after allocation and before commit under the same serialized
aggregate. Stage every fallible allocation under a cleanup ledger; abort
only the candidate on failure. The final publication of the native and C4
registry edges must be a reviewed no-failure tail. Failure leaves the model,
parameter bits, gradients, RNG words, handles, registry entry and any other
workspace byte-for-byte unchanged.

Reset is an idle, same-owner operation in this gate: it reauthenticates both
edges and clears only workspace-owned scratch, leaving the exclusive edge and
model bytes intact. No forward frame, graph, gradient or optimizer state is
created yet, so reset must not pretend to clear those future states. Release
requires idle state and no genuine held lease; it preflights all fallible
native cleanup before clearing both edges and removing the workspace from the
live registry. Foreign, forged and stale shells reject with a typed
`invalid-state` error, including repeated release of a formerly live shell.
The live registry compares exact shell identity before reading its row; it
must never invoke a caller-supplied procedure or reveal a shared seal. Dead
rows are not retained so repeated create/release has bounded retained bytes.
No route may clear an unrelated owner's edge to recover from an error.

The current association status requires model entry slot 10 to be empty, so
it applies to create preflight only. Reset and release need a distinct
read-only live-workspace recheck of the exact workspace shell/registry entry,
model slot-10 edge, native owner `active` edge, fourteen bindings and their
unchanged ownership. They must not bypass that check or pretend the empty-slot
status admits a published workspace. Existing generation call-entry code
treats slot 10 as an active call and admits only the seeded profile; future
trainable generation must explicitly reject a workspace-tagged slot on every
entry path rather than reinterpret it as a call.

P1's existing `module-eval!` and `module-train!` can change a finalized
model's mode without consulting this proposed workspace. This ownership-only
gate therefore permits release after mode drift by rechecking identity and
both exact edges without requiring `train`; reset rejects eval mode and leaves
the workspace releasable. No forward or VJP may run after mode drift. Before
a C4 compute gate, either P1 mode transitions must reject a live C4 workspace
through a reviewed interlock or every compute admission must prove the full
train-mode topology under the active workspace edge. A successful create
alone cannot assert a permanent train-mode invariant.

The lease check must be a reviewed C4-specific authority. The existing TR3
lease predicate covers a context-two trainer and does not close this C4
boundary. If a future C4 trainer introduces an additional lease or active
graph state, workspace admission and release must be extended and regated
before that trainer can use them. This gate does not expose a forward, VJP,
loss, optimizer, SAVE/LOAD or generation operation.

## Proof boundary

The first compiled fixture must use a genuine freshly constructed trainable
C4 model. It checks exact fourteen same-owner bindings and tied alias; rejects
seeded/C2/foreign/stale models, swapped or replaced handles, wrong `[2,4]`
position shape, changed live carrier, eval mode, active construction/call,
an idle live same-owner generator with retained cache, second workspace,
held native/P1 leases, forged closure capture, repeated stale release and
each allocation/edge cut. Test eval-mode drift
after create: reset rejects while release still clears both exact edges. Test
independent model coexistence, unchanged parameter and
gradient bits, and no partial publication. Feature-off symbols remain absent.
Normal/repeat and native ASan+UBSan+LSan must use pinned source/toolchain and
have an independently reviewed post-exit seal. Repeated create/release must
show bounded retained bytes rather than rely on a leak-free assertion alone.

Passing this gate establishes ownership and exclusion only. A subsequent
reviewed shape/lifetime schedule must bind the workspace to the genuine
four-token forward provider and all VJPs; only then can TR3/D2/O2 and C2
training/restore be attempted for the same checkpoint.
