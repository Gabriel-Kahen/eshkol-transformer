# Private C2 checkpoint LOAD staging

This implementation is deliberately split at the K2 dependency boundary.
`c2-checkpoint-load-stage-internal` performs two independent LOAD validations,
each using one descriptor throughout its own probe/read/revalidation sequence.
The first returns allocation sizes. The second validates the current file
again and copies exact inert components only
after every destination header, length, span, and pairwise disjointness check
passes. There is no immutable snapshot or general change-detection claim
across the two opens; a component-size change rejects as `corrupt-data`, while
same-sized replacement bytes are represented coherently by the authoritative
second pass. It then parses and deep-copies the complete O2 reconstruction input and
revalidates X1 and both D2 cursors before publishing an opaque staging owner.
No tensor codec runs in this phase.

The eventual public `checkpoint-load/3` compositor must run the real K2
runtime/capability admission after staging and before calling
`c2-checkpoint-load-reconstruct-internal`. There is no public facade, fixture
capability report, substitute validator, or K2 claim here. Focused private
tests call reconstruction directly only to prove the separable component
transaction.

Reconstruction decodes C1 through the fixed I2 provider, reconstructs O2 with
its prepare/commit transaction, and transfers both owners into the existing C2
owner. The staging owner is one-shot. Failure consumes it, clears its large
references, and releases every created P1/O2 owner exactly once; cleanup
continues after the first provider defect and a cleanup defect is `internal`.

## Measured and unmeasured resources

The reader applies hard, caller, and optional operational-target file,
artifact-metadata, tensor-count, and per-tensor checks in the accepted order.
The operational tuple remains an unverified target. LOAD currently performs
two full reads and validations. During the second pass the retained native
image coexists with eight exact Eshkol destination bytevectors. During
reconstruction, detached model/O2 staging, C1 descriptor payloads, I2 decode
builders/clones, O2 request copies, and native O2 builders/clones can coexist.
These allocations are bounded by admitted bytes and counts, but no exact whole
heap/RSS ceiling, timing bound, or supported operational tuple is claimed.

Successful LOAD/release is not retired-control-flat: I2, P1, and O2 lifecycle
tombstones are intentional. Tests distinguish storage-bearing live counts from
retired/dead controls and require exact rollback baselines on failed loads.
No component owner is constructed in a region requiring identity promotion.
