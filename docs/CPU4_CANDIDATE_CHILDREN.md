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
- `cpu4-owned-binding-create-internal!(ticket)` creates its binding after O2.
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
with explicitly nonnative doubles: 219 checks cover constructor cuts, blocked
preflight, single-shot drain retry, wrong rosters, duplicate membership and both
cons splice orders. The compiled fixture/driver sweeps actual O2/F32 allocation
failures inside an Eshkol region, observes native resource counts, checks a live
snapshot blocks before detach, and checks final owned cleanup plus caller D2/T2
survival. Compiled/sanitizer results remain NOT RUN until a sealed supported gate
executes them. Sanitizer mode instruments native providers, not generated Eshkol
IR or the prebuilt runtime. Host source guards and doubles do not establish those
runtime claims.
