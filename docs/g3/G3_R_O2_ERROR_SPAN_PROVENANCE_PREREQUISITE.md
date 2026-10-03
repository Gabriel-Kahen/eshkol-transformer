# G3-R provenance-backed error-span query prerequisite

This is a proposed source-private contract, not an implemented API or an O2
retirement acceptance. It gates the [native O2 moment retirement
leaf](G3_R_O2_MOMENT_RETIRE_READINESS_CONTRACT.md) after the accepted f32
destroy-readiness producer at `9757c160`. That accepted f32 source remains
unchanged by this proposal.

## Concrete gap

O2's preflight and commit accept a caller-owned `et_o2_error_v1 *`. A rejected
preflight may write that 264-byte record. Before its first write, O2 must prove
the output is disjoint from every live or retained allocation it could
otherwise alter, including f32 moment backing. The existing
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
disjointness; `1` for overlap; `2` for null, zero-length or overflowing span;
and `3` when any relevant live or retained backing cannot be authenticated.
All nonzero results are fail-closed to O2. A forged pointer cannot be treated
as a request to inspect that memory. The producer must use immutable
creation/clone-time f32 backing provenance for the exact currently live
control, shape, stride and data allocations, including parameter value and
gradient children, ordinary tensors and private clones. It must also cover
live parameter, borrow and plan controls and their owned arrays, and retained
dead f32 controls. A current live allocation cannot disappear from this scan
because its mutable descriptor is corrupt. Historical freed payload spans
are not reserved; the allocator may legitimately reuse them.

O2 checks its own current and retained controls and owned arrays separately.
It calls this f32 query on the exact 264-byte error span, then rejects every
nonzero result **before** writing the error record. A null O2 error pointer
requires no span query. The native optimizer lookup remains registry-first,
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
may the O2 leaf claim read-only preflight and a non-failing retirement tail.
