# Closed CPU4 factory child custody

`native/cpu4_candidate_children_extension.esk` is source-private and loads after
O2, the CPU4 candidate/model binding/control extensions and the exact trainer
detach extension (`3580136f`). Control byte comparisons require `502f0671`.
It creates and owns a new canonical C4 candidate, optimizer, binding and trainer.
Tokenizer and dataset remain caller-owned. It adds no public facade or native
adoption escape. The model retains position `[4,4]`, fourteen unique parameters,
the head/token tie and entry slot 7 equal to `#f`.

The fixed private procedures are:

- `cpu4-owned-candidate-create-internal(resolved)` returns a registered ticket.
- `cpu4-owned-model-internal(ticket)` returns its same canonical model.
- `cpu4-owned-optimizer-create-internal!(ticket,config)` creates its own O2
  optimizer from that model's parameter tree. Foreign enrollment is unsupported.
- `cpu4-owned-binding-create-internal!(ticket)` enters TRAIN on its own exact
  candidate model, then creates its strict binding after O2.
- `cpu4-owned-trainer-create-internal!(ticket,tokenizer,dataset)` returns its
  authentic trainer; that trainer remains on an UNPUBLISHED candidate.
- `cpu4-owned-candidate-retire-internal!(ticket)` drains and retires its owned
  children/model. Completed retirement is idempotent; malformed/foreign tickets
  reject. Constructor failure retains the ticket for explicit cleanup/retry.

The closed O2 constructor reuses O2 configuration/group/provider/native-builder
helpers. It installs its cleanup handler before publishing any pending row and
roots the exact six-slot envelope before native builder allocation. This avoids
the public constructor's possible active-region global promotion after native
finish. The native builder is retained in the rooted ticket; finish consumption
is recorded before fixed native-pointer stores. No global publication, callback
or allocating operation follows successful finish. Public O2 source is unchanged.
The owned binding helper rejects foreign/duplicate tickets or a missing owned O2
before mode changes. Under its installed cleanup handler and busy claim, it
resolves its own exact candidate model, calls `module-train!` on that model, then
calls the unchanged strict binding constructor. TRAIN failure clears busy and
rethrows the original error. The tested guarded binding/context rejection paths
clear busy, retain owned authority and leave TRAIN for exact retry; mode rollback
is not attempted. Retry is not established for the lower constructor's
pre-handler registry-publication/handler-allocation window. This transition
changes mode, not model parameter values, and never transfers caller
tokenizer/dataset ownership. The direct binding API keeps
its existing strict TRAIN precondition. The P1 operation sets all seventeen mode
flags; it does not consult the native context registry. An existing native
context is rejected later by strict binding's native lease creation, after the
mode transition. That context is preserved, but its model remains TRAIN; restore
EVAL explicitly before reusing the existing generator, and close the caller-owned
context before retrying training construction. No earlier context preflight or
mode rollback is added. The authenticated public generator close path does not
require EVAL, so it can close the preserved context while the model remains
TRAIN. EVAL restoration is required only to reuse that generator for calls.

Binding and trainer constructors already root their own canonical publication
records before their native success tails. Their returned identities are retained
by fixed stores before another fallible factory operation.

Retirement first authenticates the exact O2 record/native pointer, fourteen
original ordered handles, unique registry membership and a preallocated cons-cell
splice. The genuine O2 retirement preflight runs with a NULL diagnostic before
trainer/binding release; its direct status is classified without stale bridge
error data. Any live O2 state, borrow or builder blocks cleanup conservatively,
including unrelated states/builders. Release those resources and retry.

The trainer detach operation is single-shot. Its success is immediately recorded
in the ticket, so a later base candidate preflight rejection can be retried
without detaching twice. Before a trainer exists, the captured binding is released
directly. Base retirement then performs the original native/I2/P1 checks and
ordered O2→P1→Scheme original-node unlink→C4 payload destruction. Its added fixed
private fifth operation accepts only an optional native optimizer pointer; the
existing four operations retain their behavior. Native admission checks the
original fourteen-parameter/handle roster. After base success, only the exact
O2 cons splice, six fixed record invalidations and ticket stores remain. Unexpected
failure after destruction is fail-stop. Independent optimizer cons nodes remain
identical in both cleanup orders.

Training/restore and evaluation use the same live binding. Finish and destroy the
E3 evaluation frame before trainer detach releases that binding; then create a
fresh generator and perform native adoption. This layer does not implement that
handoff, generation or a full checkpoint factory. Caller-owned dataset/tokenizer
are never destroyed. Native tombstones remain identity guards after payloads die.

`tests/cpu4_candidate_children/host_test.scm` executes the actual new orchestration
with explicitly nonnative doubles: 334 checks cover constructor cuts, blocked
preflight, single-shot drain retry, wrong rosters, duplicate membership and both
cons splice orders. Binding cases also cover all seventeen TRAIN flags, the
exact owned model and ordering, rejection before mode changes, TRAIN/binding
fault identity and retry, retained TRAIN on late native-context rejection,
explicit EVAL recovery and caller-owned T2/D2 survival. The compiled fixture/driver sweeps actual O2/F32 allocation
failures inside an Eshkol region, observes native resource counts, checks a live
snapshot blocks before detach, and checks final owned cleanup plus caller D2/T2
survival. Compiled/sanitizer results remain NOT RUN until a sealed supported gate
executes them. Sanitizer mode instruments native providers, not generated Eshkol
IR or the prebuilt runtime. Host source guards and doubles do not establish those
runtime claims.

`binding_mode_close_host_test.scm` additionally executes the actual source-extracted
public G2 constructor/profile/close and lower auth/close procedures, with explicit
native doubles. `CPU4_PUBLIC_G2_SOURCE` must name the separately reviewed public
G2 dispatcher from root pin `9b7b271c`; that source is not present on the `7db`
base. The control checks forged/policy-mutated close rejection, close in retained
TRAIN, explicit EVAL restoring actual call readiness, exact native context close
and same-ticket/optimizer/model binding retry. It proves host source orchestration,
not native context execution. No production source closure or public API is added.
