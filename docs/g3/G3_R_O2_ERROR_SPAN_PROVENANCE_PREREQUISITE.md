# G3-R provenance-backed error-span query prerequisite

This is a proposed source-private contract, not an implemented API or an O2
retirement acceptance. It gates the [native O2 moment retirement
leaf](G3_R_O2_MOMENT_RETIRE_READINESS_CONTRACT.md) after the accepted f32
destroy-readiness producer at `9757c160`. That accepted f32 source remains
unchanged by this proposal.

## Concrete gap

O2's preflight and commit accept a caller-owned `et_o2_error_v1 *`. The caller
must supply either null or a properly aligned, writable extent of
`sizeof(et_o2_error_v1) == 264` bytes. A rejected preflight may write that
record. Before its first write, O2 must prove the output is disjoint from
known live or retained allocations it could otherwise alter, including f32
moment backing. The existing
`et_f32_tensor_private_storage_overlap_v1` is read-only, but it scans mutable
`tensor->shape`, `strides`, `data` and `byte_length`. It is not a provenance
query. A test-only metadata restoration can move a live tensor's mutable
`data` pointer to a separate valid decoy allocation while its immutable
creation-time record still owns the original data. On the exact `9757c160`
source, querying a 264-byte span at the original data then returns `0`
(disjoint), while `et_f32_tensor_private_destroy_ready_v1` returns
`INTERNAL/PROVIDER_REJECTED`. A later O2 failure report could overwrite the
real backing before detecting the corruption. Local Clang 22 reproduction:
`recorded-live-payload overlap=0 destroy-ready=6`.

## Minimal private producer

Under `ET_G3R_CANDIDATE_RETIRE_PRIVATE`, add only an internal-header
declaration and production f32 implementation:

```c
int32_t et_f32_tensor_private_owned_span_overlap_v1(
    const void *span, size_t bytes);
```

The input is an opaque, nonempty, representable address span. The function
does not dereference or write through `span`, allocate, borrow, pin, invoke a
provider, or change any registry. It returns `0` only when it can prove
disjointness from the known f32-owned spans below; it does not prove that an
arbitrary pointer is writable or has the caller's claimed backing extent.
It returns `1` for overlap; `2` for null, zero-length or overflowing span;
and `3` when any relevant live or retained backing cannot be authenticated.
All nonzero results are fail-closed to O2. A forged pointer cannot be treated
as a request to inspect that memory; the query does not dereference it.
The producer must use immutable
creation/clone-time f32 backing provenance for the exact currently live
control, shape, stride and data allocations, including parameter value and
gradient children, ordinary tensors and private clones. It must also cover
live parameter, borrow and plan controls and their owned arrays, and retained
dead f32 controls. Plan-owned array addresses and lengths need their own
immutable authenticated provenance; if the producer has only mutable plan
pointers or counts, it returns `3` rather than claiming disjointness. A
current live allocation cannot disappear from this scan
because its mutable descriptor is corrupt. Historical freed payload spans
are not reserved; the allocator may legitimately reuse them.

O2 checks its own current and retained controls and owned arrays separately.
For a nonnull caller error record satisfying the caller precondition, it calls
this f32 query on the exact 264-byte span, then rejects every nonzero result
**before** writing the record. A null O2 error pointer skips the query and
causes no output write. A zero result proves only disjointness from known
owned spans; valid caller storage remains an independent precondition. The
native optimizer lookup remains registry-first,
and the aggregate Eshkol ledger/M3T/P1 witness remains a separate admission;
neither this query nor a caller-provided extent proves those associations.

## Acceptance gate

First review the exact private signature, status values, source/FFI symbol
allowlist and default-off parity. Focused tests must show read-only,
allocation-free results for null, zero, overflow, ordinary live payload,
parameter value/gradient payload, metadata and control, plan/borrow storage,
retained control and historical freed payload reuse. Mutating a tensor's
public data/shape/stride/length away from its recorded backing must never
yield disjoint for the original live allocation. Malformed provenance returns
`3` without writing caller memory; restoration permits a successful retry.
Run normal, repeat and ASan/UBSan/LSan modes on the supported pinned compiler,
then independently review the exact source and tests. Only after this gate
may the O2 leaf claim a preflight that leaves component, provider and allocator
state unchanged, with the disjoint caller error record as its only possible
write, and a non-failing retirement tail.

## Bounded producer candidate

The guarded f32 producer candidate implements the signature above using the
existing creation-time tensor backing records. It returns `3` for a live copy,
gradient or reset plan unless the queried span overlaps the authenticated
plan control itself: those plans have no immutable owned-array extent yet.
This conservatively blocks O2 admission while any such plan is live, including
an unrelated plan. It does not change the accepted destroy-readiness API or
feature-off f32 object bytes.

The exact independently reviewed source/tests head `af987fe2` (tree
`47d41967`) passed the pinned fe9/f31 Clang 21 gate on 2026-10-04. Normal,
repeat and ASan/UBSan/LSan each report 104 checks, including a real 320-byte
backing allocation, decoy pointer, 264-byte O2 error span, malformed public
metadata and backing record, retry, plans, retained controls and freed payload.
The existing f32 readiness gate passes 451 checks per mode. Positive compiler
and runtime stderr are empty, mode outputs are identical, and default-off
objects match the accepted baseline. Relative to that readiness baseline, the
only added production feature-on symbol is
`et_f32_tensor_private_owned_span_overlap_v1`; public/off header consumers
reject the private symbol with the exact expected diagnostic. Independent
review accepted the complete 164-file run seal
`g3r-f32-owned-span-af987fe2-fe9-f31-20261003-prepared-b/SHA256SUMS-RUN`
(SHA256 `4ef814b7e0268b8a1c2606e4e07da3e641856841208e49546173e55c7b2635e4`)
and raw dependency, symbol and pre/post pin evidence. All eleven native gate
inputs are byte-identical in the current-main integration. Hosted integration
and the dependent O2 retirement proof remain pending; this gate proves no
Eshkol candidate cleanup or full replay.
