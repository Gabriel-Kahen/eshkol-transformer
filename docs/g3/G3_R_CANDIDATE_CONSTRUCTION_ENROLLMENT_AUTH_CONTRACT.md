# G3-R private construction enrollment authentication prerequisite

**Private source contract; compiled proof and integration pending.** This narrows the
[P1/I2 sealed-graph producer contract](G3_R_P1_I2_SEALED_GRAPH_RETIREMENT_CONTRACT.md)
at the first candidate native P1 token. It changes only the noninstalled
`ET_G3R_CANDIDATE_RETIRE_PRIVATE` source-composed aggregate. The installed
`transformer.module`, ordinary M3T/I2 paths, feature-off P1 objects and public
tuple remain byte-for-byte unchanged. It adds no native API and does not claim
graph, O2 or M3T retirement.

## Authority and exact call order

The generated P1 overlay creates one rooted registry holder **inside its one
authentic P1 lexical closure**. The fixed candidate ledger creator, I2 begin,
P1 begin and M3T constructor are closures in that same private lexical scope;
they capture the holder directly. The private surface exposes only their fixed
operations. It exposes no holder, registry row, registrar, arm/consume/reset
operation, identity selector or accessor returning any of those, and creates no
top-level `g3r-candidate-ledgers` binding. Neither the verifier nor native
begin is passed as a callback, stored in a candidate cell or chosen by a
caller. The test-only fault selector accepts a fixed numeric stage and never
accepts or returns an identity; its cross-ledger cut selects another authentic
row inside the closure. The private wrappers capture each fixed operation once while the P1 module is
initialized. They do not reread mutable `p1-trusted-surface` slots on each
call; replacing those appended slots later cannot redirect ledger, I2, P1 or
M3T authority. No wrapper returns a row, holder, admission setter or verifier.
The ledger creator registers one source-private row whose original identity fields
contain the exact ledger object, original config and initializer, preallocated owner cell, P1 enrollment cell,
P1 sidecar and I2 sidecar. The row's separate admission state and native-owner
identity advance only through the fixed constructor. The row
and all cells are rooted before the candidate constructor or any native owner
creation. The row is never returned or stored in a caller-owned object; its
original references are not rewritten after registration. "Immutable" here
means those private original references remain fixed, not that Scheme vectors
are physically immutable. Mutable values in returned ledger/cells are
observations, not authority; registry row identity and the lexical holder are
authoritative.

The holder has one active-ledger identity. The fixed candidate M3T constructor
authenticates the ledger by exact registry identity and the original inputs,
rejects an already active candidate, and activates that exact ledger before
native owner creation. The returned native owner is held in a predeclared
guarded lexical slot before any potentially fallible cell or stage write; its
exact pointer is then stored in the rooted owner cell and the row's dedicated
owner-identity field. Only after this owner
enrollment may the I2 candidate begin authenticate its exact sidecar and P1
cell against the active registry row. Immediately before calling P1, I2 arms
that row's one-use P1 admission state. The existing
`(module-candidate-construction-begin-internal provider enrollment-cell)`
signature is unchanged.

P1 begin scans the captured registry holder by **exact P1-cell identity** and
requires: the row's ledger is the active ledger; the original owner cell holds
the exact newly enrolled owner; the exact I2 sidecar is in its open admission
phase; the row is armed but unconsumed; the P1 cell/sidecar are pristine; and
the provider is the fixed `i2-dense-cpu-f32-v1` identity. It consumes the row's
one-use admission **before** invoking
`et_p1_private_candidate_construction_begin_v1`. No module/handle token may be
created before that check. I2 arms and immediately calls P1 in the same trusted
stack frame, with no caller, provider or user callback between those steps.
Before native begin, a predeclared lexical attempt flag is set. Native begin
clears the authentic context's prior result before any allocation and, on
success, retains the exact new pointer in that context. The existing read-only
`et_p1_private_result_ptr_v1` getter assigns it to a predeclared lexical
identity slot, before any fallible rooted write. The cleanup guard reads the
same context-held result for any error after native begin is attempted,
including a fault between successful status and lexical assignment. A failed
native begin leaves no allocated ledger and no context result. The getter must
remain total, allocation-free and free of
Scheme errors for the private authentic context; emitted IR must prove the
pointer return is packed without a runtime call or intervening fallible edge.
The frozen native source's getter only scans context pointers and returns its
result field; successful begin stores that field before returning status zero.
Pinned Eshkol `fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9` lowers an `extern
ptr` result through LLVM `ptrtoint` and SSA `insertvalue` packing, with no
allocation call in that return packing. This source inspection is not a
substitute for the fixture's emitted full IR review.
The pointer is enrolled in the canonical P1 record and original
P1 cell before creating the first root module token. If any enrollment write
fails, that lexical identity drives native/P1 abort; cleanup must not depend
on a cell write that may have failed.

## Consumption, abort and seal

The one-use state advances only `issued → owner-enrolled → armed → consumed →
sealed` or to terminal `aborted`; it never returns to `issued` or `armed` after
native begin is attempted. Pre-native validation failures leave all native
authority unchanged. A native begin failure or any later preseal failure
aborts the open/prepared P1 scope, restores I2's original registry/count,
aborts the exact native M3T owner, marks the row and ledger aborted, and clears
the active-ledger identity while preserving the first cause. A cleanup defect
is fail-stop 134. Successful P1/I2/M3T seals retain the exact sidecars and a
live sealed candidate, mark the row sealed, and clear active construction.
After the first successful native P1 seal, an unexpected error is fail-stop
134; it cannot use construction abort. No row state, cell or sidecar grants
retirement authority by itself.

## Required negative evidence

The composed fixture must reject, before native candidate begin or a token:
foreign and structurally forged P1 cells; a copied cell with identical fields;
an authentic cell from another ledger while one ledger is active; a stale cell
after consumption, abort or seal; a copied/foreign ledger; a mismatched I2
sidecar; an owner pointer substituted before I2 admission; and a provider other
than the fixed I2 identity. Each rejection preserves the active authentic
candidate, prior I2 count and native live counts. Faults raised as `#f` at
owner-return, P1-return and preseal enrollment cuts must release exactly the
owned native authority and preserve the first cause. Source and emitted IR
must demonstrate the lexical return-to-guard cuts, the root/registry order,
and no fallible gap between successful native P1 return and lexical ownership.
Fixed internal test cuts must exercise wrong-provider and copied-cell rejection
while an authentic row is armed, then continue the successful P1 admission;
owner-cell substitution must occur after native owner enrollment and be
restored before the primary I2 call. Distinct raised-`#f` cuts must land after
native begin status but before result-pointer assignment, and after assignment
but before the first rooted P1 write. Counts and the active authentic row must
survive each negative probe; fault cleanup must preserve the first cause.
The fixture also replaces every appended mutable P1 surface slot before using
the fixed wrappers to detect a regression to per-call dispatch. Its fixed
test-only numeric roster-byte readout uses an existing P1 native test hook;
faults must restore the baseline even when no P1 token was created.
The generated overlay, its raw depfile and exact feature-on/off symbols must
remain in the later supported normal/repeat/sanitizer evidence.
