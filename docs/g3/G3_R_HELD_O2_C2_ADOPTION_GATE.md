# Held O2 state adoption into C2: source gate

Status: **native owner-match plus source-private held O2 enrollment prerequisite**.
The adopter source is not yet loaded into a compiled aggregate. This leaf adds
no G3/O2 coordinator, C2 claim, SAVE pair, Eshkol owner, or file publication. Its
native negatives preserve the distinction between a bit-matching O2 state and
the exact state issued under the original-owner hold.

## Observed source boundary

`et_g3r_private_o2_save_state_v1` in `native/o2_optimizer.c` returns a raw
`et_o2_optimizer_state *` while its original optimizer is held. The companion
`et_g3r_private_o2_save_issued_match_v1` checks that a registered live state's
configuration, options, and moment bits match the held optimizer. It does not
record or require the address returned by `save_state`. The focused native
fixture now snapshots another equal state **before** the hold, issues the held
state, and verifies that both distinct pointers pass `issued_match`. A second
registered optimizer with the same parameter roster, options, and zero moment
bits produces a foreign snapshot that also passes `issued_match`. The new
feature-private `et_g3r_private_o2_save_owner_match_v1` checks the live hold's
exact registered optimizer pointer and then its held invariants; it rejects
the foreign optimizer, stale hold, and altered original optimizer. This query
does not identify the particular state returned by held `save_state`. A future
caller must retain that issued address itself.

`o2-public-optimizer-state` in `native/o2_wave2_extension.esk` preallocates an
O2 Scheme shell and enrolls an ordinary native snapshot in `o2-states`. That
snapshot call rejects the busy held optimizer. The separate
`c2_o2_reconstruct_extension.esk` enrolls a native state reconstructed from
checkpoint bytes after its builder commit; it does not accept a held original
optimizer or the pointer returned by held `save_state`. The C2
`c2-training-state-compose-internal` request takes the
registered Scheme O2 state shell, validates its topology/projection, and
claims it with a preallocated owner token. It cannot take a raw native state
pointer. The current `tr3-c-snapshot-compose!` calls the ordinary O2 snapshot,
so it does not bridge this gap. No current Eshkol source invokes the native
G3 and O2 held functions together.

## Minimum next interface

A feature-private Eshkol adoption helper must allocate and root its exact
Scheme shell, state record, registry cons, and native pointer cell **before**
calling the held `save_state` function. The native coordinator must hold the
original G3/M3/A2 and O2 owners, retain the exact returned pointer in its own
registered record, and check pointer identity as well as `issued_match` before
any C2 claim. It must also call `owner_match` on the exact O2 optimizer pointer
from the registered Scheme optimizer shell before held issue or adoption;
neither state bits nor an unregistered Scheme shell prove this relationship.
The native query authenticates the O2 hold-to-optimizer link only. Binding that
optimizer to the original G3/M3/A2 owner still requires the held roster and
Scheme owner checks in a future coordinator. The helper must put the returned
pointer into the preallocated root cell immediately; enrollment then uses
assignments only. It may publish
the Scheme state shell only after the native pointer and logical O2 projection
are checked against the same registered optimizer. Existing ordinary O2
snapshot code is a lifecycle reference, not an authorized held adopter.

On a preclaim exception, the helper releases exactly that rooted native state
once and leaves the original holds live until release is known. If release is
uncertain, it retains rooted authority and fails stop; it must not return a
recoverable error. After C2 composition starts, its request slots and exact
owner token are the transfer ledger. A partial P1/O2 claim must be rolled back
or quarantined with the exact child shell, and no receipt or file I/O can begin
while the original holds remain live. The existing C2 rollback only covers its
own claimed Scheme children; it cannot repair an unregistered raw O2 pointer.
Native `et_o2_optimizer_state_release_v1` may return a provider defect **after**
marking the state DEAD. The adopter must inspect the exact native lifecycle
after a nonzero release: a still-LIVE state retains its root for a controlled
retry, DEAD means its authority was consumed even though the defect is
reported, and RELEASING or an unprovable lifecycle fails stop. The existing
Scheme `o2-public-optimizer-state-release!` distinguishes its wrapped `-1`
and `-2` outcomes; a raw adopter cannot collapse every nonzero status into
"unchanged state." Preserve the first typed error across this inspection.

The first combined source/runtime gate should prove: distinct bit-equal state
rejected by pointer identity, one held issue and one claim, pre-enrollment
exception, post-enrollment exception, O2 release failure, partial C2 claim,
first typed error retention, reverse-order hold release, original model/O2
unchanged on recoverable cuts, and zero publication before all holds drain.
Then the 17-word G3 snapshot and original tokenizer/cursor/trainer controls
can be bound to the detached C2 owner and its exact image receipt. No current
source or test establishes that combined transaction.

## Joint-issued C2 claim source successor

`native/g3r_joint_c2_claim_extension.esk` is an unlinked private prerequisite.
It accepts the exact enrolled joint-issued O2 ledger, an independently prepared
detached P1 state, and C2 controls. It preallocates a 136-byte held-history
destination, C2 request and result ledger before the C2 claim. While both
original holds are live, it checks the registered optimizer and exact issued
state, then copies the held 17 words. Its guard covers the C2 composition call.
On a preclaim exception, C2's internal guard rolls back any token-matched O2
claim before the P1 claim; the outer guard requires the O2 shell to be LIVE,
releases the exact issued native state with one controlled retry only for
proved LIVE, and drains O2 before G3. An unknown release or hold lifecycle
fails stop, while a successful cleanup rethrows the first caught error.

After a successful C2 composition, the source verifies the exact registered
C2 P1/O2 children and owner token. The private native
`et_g3r_joint_hold_transfer_issued_v1` then checks pointer identity and the
held bits before clearing the joint record's issued custody. Only after that
does the source end the original O2 and G3 holds. A post-drain defect is
recorded in the rooted result; a hold that cannot be proved DEAD fails stop.
The native transfer itself cannot prove a Scheme claim, so it is deliberately
called only after the exact C2 registry check in this same source path.

This leaf has source mutation and C syntax evidence only. It does not prepare
or authenticate the P1 clone or original tokenizer/cursor/trainer controls,
bind the 17 words to a serialized pair, authorize receipt I/O, or establish
replay equivalence. The aggregate compilation, runtime fault cuts and
independent review are still gates before an original-owner C2 claim is
accepted.

## Bounded original Scheme owner and control binding

`native/g3r_joint_original_owner_binding_extension.esk` checks the exact
registered G3 generator, live M3T model, raw V256 T1 tokenizer and Scheme O2
optimizer shell against the native joint hold's original generator/model/O2
pointer triple. The native `owner_match` rejects a foreign or DEAD joint before
rechecking both holds. It copies C2 controls through the existing validator,
then requires the C2 tokenizer fingerprint/vocabulary to match the registered
T1 raw tokenizer and the C2 X1 canonical bytes/fingerprint to match the
original M3T model's retained config. Its wrapper passes those copied controls
and the exact joint-issued ledger into the earlier private C2 claim helper.
This is a source-only original-owner identity and two-control binding; it
does not derive or authenticate the detached P1 state.

The accepted live TR3 trainer cannot supply the remaining original controls
for this same G3 model without another reviewed mode contract. G3 admission
requires `module-mode-internal model` to be `eval` in
`native/g3t_model_admission_extension.esk:57-59`, while the trusted TR3 fixed
module recheck requires `e3-mode-train` in
`internal/p1/lib/transformer/module.esk:5251-5260`. The TR3 snapshot lease
therefore cannot be presented as a same-owner trainer-control witness for an
already-live G3 generator. T1 and TR3's T2 tokenizer shells are separate
identities; matching fingerprint values does not claim they are the same
shell. D2 cursor pair, trainer RNG, counters, P1 provenance, paired receipt
and replay all remain open. No mode contract was changed here.

## Bounded adopter source leaf

`native/g3r_held_o2_adopter_extension.esk` defines `g3r-held-o2-adopt!` for a
caller that already owns the live original O2 hold and the registered Scheme
optimizer. It calls the feature-private C bridge's owner match on the exact
native pointer in that optimizer record. The helper allocates the Scheme shell,
state record, registry cons, pointer cell, and ledger before held issue. It
stores the exact returned pointer in the cell immediately, then checks the
same pointer through owner match and issued match before publishing the Scheme
state. It installs the cleanup handler before native issue; its handler uses
non-raising native release statuses, with no nested handler installation after
ownership transfers. The returned ledger retains the hold, original optimizer,
native pointer, state shell, registry cons, and phase for a future C2
coordinator.

`g3r-held-o2-adopter-release!` releases only an unclaimed issued/enrolled
state. It uses the O2 bridge's lifecycle-aware status: `0` consumes it; `-2`
marks it dead while retaining the native provider defect; `-1` proves it is
still live and leaves its exact ledger rooted for retry; indeterminate status
fails stop. An exception after issue makes one controlled retry of a proved-live
release, then fails stop if authority remains live. The original exception is
re-raised after successful cleanup. The hold remains the coordinator's
authority and is never released by this helper.

This is a source prerequisite only. The future coordinator must prove the
original G3/M3/A2 to O2 binding, prepare a P1 clone and controls, transfer the
ledger's exact shell into C2, reconcile partial claims, release both holds in
reverse order, and gate receipt I/O. A compiled aggregate/runtime test and
independent review are required before any held C2 owner is claimed.

## Detached P1 provenance source successor

`native/g3r_joint_p1_provenance_extension.esk` is an unlinked private leaf.
Given the registered original G3/M3T/T1/O2 binding and an enrolled joint O2
ledger, it checks the live native joint owner, roots the exact return of
`module-state-dict model` immediately, and validates the P1 C2 topology against
all fourteen fixed M3T parameter paths and shapes plus the head/token tie.
It then rechecks the original owner and copied controls while the joint hold
remains live. Its preclaim ledger contains the state shell and first binding,
while a separate private registry record retains the exact shell and an
independent owner witness. The witness copies all original binding identities
and makes a separate validated deep copy of the controls; it shares no mutable
binding vector or control leaf with the returned ledger. The record is never
returned or stored inside that ledger. The claim wrapper first finds the
record by ledger identity, then installs cleanup protection before topology
or owner validation. It compares both the ledger binding and fresh claim
binding with the private witness, and passes only the registry-rooted P1 shell
to C2. A recoverable preclaim failure releases that
exact shell using `state-dict-release!` and rethrows the first caught value;
the P1 helper never ends the original G3/O2 holds. An uncertain P1 release
fails stop.

The source mutation gate covers omitted owner checks, topology and tied-alias
changes, cleanup omission, first-error loss, claim bypass, stale owner,
bitwise G3 history comparison, caller-side P1 substitution, in-place
cross-owner binding mutation, and validation moved before the cleanup guard.
The original binding source gate remains green.
These are source checks, not live negative tests. In particular, no compiled
aggregate has shown that P1's provider can clone under the live G3 model pins,
or that all provider failure and post-tombstone release cuts retain authority.
Until that fixture and independent review pass, this leaf is not an admitted
same-owner P1 clone or C2 claim. D2 cursor and trainer controls, pair/receipt,
SAVE publication and replay remain open.

## Joint begin source prerequisite

`native/g3r_joint_begin_extension.esk` adds a private, unlinked Eshkol entry
that can pass the exact registered G3 generator, live M3T model, raw T1
tokenizer and O2 optimizer pointers to the existing native joint hold begin.
It rejects mismatched or busy Scheme generator roots before native acquisition.
The native begin authenticates the fourteen-parameter original M3T/O2 roster
and returns one held owner; a native failure follows the existing joint
error mapping. A nine-mutation source gate checks shell-to-pointer wiring and
feature-off closure. No compiled aggregate or held-pin clone test has run on
this entry, so it does not establish a claim or a recoverable failure path.
The next gate must link the begin entry, joint O2 adopter, original binding,
P1 provenance and C2 claim in one pinned aggregate, then exercise the real
CPU-f32 clone and the specified failure cuts while both holds remain live.

## Feature-private held P1/C2 aggregate fixture

`tests/g3r/held_p1_c2_gate.esk` now loads the real joint begin, issued O2
adopter, P1 provenance and original-bound C2 claim in one D2/T2/I2/P1/M3T/G3
registry. It loads the accepted M3 schedule and model extension before the
G3 adapters. Its runner compiles the private native joint and provider flags only
for this fixture. The source scenarios create the diagnostic CPU-f32 fourteen
parameter model and raw T1 generator, with a distinct T2 byte tokenizer used
only to obtain canonical D2 cursor controls. A test-only native witness captures
all fourteen original value and absent-gradient bit baselines under the live
pins and checks both immediately after the P1 clone. After C2 transfer, the
fixture compares all fourteen original value payloads, gradient metadata and
the held 17-word G3 history. The fixture also specifies
foreign owner, malformed controls, injected topology, an authentic I2/F32
provider-clone allocation failure, caller-side shell substitution, stale owner
and post-O2 partial claim cuts. The provider failure arms the F32 test
allocator immediately before the clone while the original G3/O2 holds and
issued O2 are live. It requires a typed `module-state-dict` error, unchanged
owned-clone count, no P1 record or C2 child, all fourteen original held bit
witnesses and seventeen history words intact, then resets the allocator and
requires an exact same-owner P1 retry and release. The fixture archive now
includes the F32 test allocator implementation and checks its symbol. On recoverable
preclaim cuts it checks clone-count rollback and explicit issued-O2 then
G3 hold drain; after C2 transfer it checks the exact registered child and
owner through the claim helper and releases the resulting C2 owner.

The fixture, runner and seventeen-mutation source closure pass static checks.
The added regression rejects a runner that precreates the corpus directory:
the D2 resource builder creates that directory itself. The first pinned gate
on `2bcdda47` stopped at this setup error before native or Eshkol compilation;
the runner is repaired, but the aggregate has **not yet compiled or run**.
Post-tombstone P1 release
defect and uncertain native release are fail-stop
contracts, not exercised recoverable fixture outcomes. A successful compiled
run and independent review are required before this can be an admitted
same-owner P1/O2 C2 claim. TR3 train versus G3 eval, T1/T2 shell identity,
original trainer/cursor controls, pair/receipt and replay remain open.
