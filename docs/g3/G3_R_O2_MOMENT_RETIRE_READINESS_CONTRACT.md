# G3-R native O2 moment retirement prerequisite

This is a **proposed private contract**, not an implemented API or a release
result. It narrows the O2 portion of [fresh-candidate retirement](G3_R_FRESH_CANDIDATE_RETIREMENT_CONTRACT.md)
and precedes [sealed P1/I2 graph revocation](G3_R_P1_I2_SEALED_GRAPH_RETIREMENT_CONTRACT.md).
The source baseline is main `90f3c420`; no public O2, f32 or Eshkol surface
changes here. The candidate's closed Eshkol ledger must separately authenticate
its exact native O2 receiver and 14 ordered M3T/P1 bindings. A bare native
optimizer pointer cannot prove that association or exclude external Scheme
owners.

## Existing boundary and missing producer

`native/o2_optimizer.c` publishes a registry-linked live optimizer with an
entry array and two ordinary f32 moment clones per entry. Its parameters,
P1 handles and canonical parameter storage are borrowed references. Builder
abort and optimizer-state release do not retire that live optimizer.
`et_o2_destroy_entry_moments` discards every tensor-destroy result; it is a
builder-cleanup helper, not a safe live-retirement tail. The ordinary
`et_f32_tensor_destroy_v1` authenticates a live tensor and rejects an active
borrow or plan pin before freeing data, strides and shape and retaining a dead
control shell. It has no read-only production destroy-readiness entry point.
Borrow-begin allocates and changes authority, so it cannot fill this gap.

More fundamentally, f32's live registry proves a control address and magic,
but its current mutable control fields alone do not independently prove that
`shape`, `strides` and `data` still name the exact allocations owned by that
tensor. The allocation envelope is an overlap accelerator, not a per-tensor
backing ledger. A new production-private readiness seam therefore requires
immutable creation/clone-time backing provenance for currently live
allocations, or an equivalently reviewed per-control ownership record,
before O2 can claim that all 28 frees have a
non-failing tail. A broad address-range test or a shape check is insufficient.

## Proposed f32 seam

Only under `ET_G3R_CANDIDATE_RETIRE_PRIVATE`, declare in
`native/f32_parameter_internal.h` and implement in `native/f32_tensor.c`:

```c
int32_t et_f32_tensor_private_destroy_ready_v1(
    const et_f32_tensor *candidate, et_f32_tensor_error *error);
```

Return zero only when one **non-null** tensor can be destroyed by the existing
ordinary tensor destructor without changing any other live authority. The
call allocates nothing, takes no borrow, invokes no provider, and changes no
tensor, plan or registry state. It may write only a caller-supplied, disjoint
error record. O2 supplies its own stack record; an error-output span inside a
current live owned allocation or a retained control/tombstone span rejects
before writing it. Historical freed payload spans are not reserved: their
addresses may back new legitimate allocations. Failure leaves the candidate
and all its backing bytes intact. The implementation must match
the existing `et_f32_tensor_error` category/code conventions: null rejects as
`INVALID_ARGUMENT/NULL_ARGUMENT`; foreign or stale registry identity as
`INVALID_STATE/INVALID_HANDLE`; active borrow as
`INVALID_STATE/ACTIVE_BORROW`; plan pin or wrong ownership/association as
`INVALID_STATE/INVALID_HANDLE`; corrupt or unproved backing as
`INTERNAL/PROVIDER_REJECTED`. It must not dereference `candidate` before an
exact live-registry match.

After that match, require the ordinary-owned tensor kind used by O2's clone
path, no borrow or prepared-plan pin, internally consistent rank/count/byte
length/shape/stride metadata, and exact recorded ownership of each currently
live control, shape, stride and data allocation. Empty spans must obey the same
create-time representation. Reject a moment that is any live parameter's
value or gradient, an I2 private-owned clone, or whose backing/control span
aliases another live owner's allocation or a retained control. These checks
do not establish Eshkol ledger provenance or discover unregistered external
raw-pointer holders; those are separate whole-candidate admissions. The seam
does not accept a caller-provided backing extent or a claimed owner token.

## O2 admission and irreversible tail

The proposed feature-local O2 preflight/commit signatures remain those in
the parent contract. Preflight first finds the optimizer in O2's registry
before dereference and admits its live provider, `busy == 0`, no active
operation, exactly 14 initialized entries and exact cached parameter,
P1-handle and canonical storage identities. It checks all 14 parameters live
and their cached P1 bindings valid. The **aggregate Eshkol root**, not the
pointer-only native O2 call, compares that ordered graph with its closed
ledger and M3T witness. Its 28 non-null moments must be pairwise distinct and
separate from every parameter value/gradient and every other live O2 entry;
each must pass the f32 readiness seam. Scan other live optimizers for shared
parameter, P1 handle, storage or moment identity. Do not infer the one tied
P1 alias from native O2's distinct entries; the Eshkol/M3T graph witness
must prove that topology. A failed check reports the first error and leaves
the complete live optimizer, entries and moments available for retry.

The aggregate performs all child, O2, P1/I2, M3T and initializer preflights
while this exact optimizer is still live, then repeats the read-only O2/f32
checks immediately before the first irreversible write under caller
serialization. Commit destroys each admitted moment using a local slot,
checks its return and null result, and frees the entry array only after all
28 succeeded. It may not call `et_o2_destroy_entry_moments`, ignore a result,
allocate, invoke a provider or raise into Scheme. Any unexpected failure
after the first destroy is fail-stop exit 134, never a retryable partial
return. It then marks this optimizer dead and retains its registry-recognized
control tombstone; the exact retired identity remains distinguishable from a
foreign or newly substituted pointer. Its Scheme enrollment changes once.
Only that authenticated O2 tombstone authorizes the later graph revocation;
M3T parameter/f32 retirement follows it. Neither the caller-owned input
initializer nor the caller's existing receiver is touched.

## First implementation gate

Implement and independently review the f32 backing-provenance and read-only
readiness producer **before** native O2 retirement. A focused feature-on/off
f32 gate must reject forged, stale, parameter-owned and private-clone tensor
pointers; aliased or corrupted backing; and active borrow/plan pins, while
preserving the exact feature-off symbol inventory. Only after that producer
passes may the bounded O2 gate test all 28 cross-entry aliases, provider or
13/15-entry mismatch, busy/active operation, and a foreign or changed
14-binding witness. It must verify precommit failure/retry and first-error
preservation, then genuine allocation-free destruction and postcommit
fail-stop fault injection. Count live O2 entries/moment payload and f32 live
bytes separately from retained O2/f32 control tombstones. No native O2
release, complete candidate cleanup, sealed-graph revocation or replay
rollback is accepted by this document.
