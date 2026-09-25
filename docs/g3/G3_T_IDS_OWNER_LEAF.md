# G3-T private generated-ID owner mapping

This leaf implements only the accepted source-private
`g3t-generation-output-ids/1` accessor. It authenticates an exact live,
independently published output, then calls the accepted native
`output_ids_clone(ptr out)` stem. The result is a fresh one-element Eshkol
list containing a distinct kind-5 shell whose native owner holds rank-one
CPU I1[G], with G zero or one. A second call creates a different list, shell
and I1 owner; neither retains the parent output or generator.

The accessor preallocates and roots the shell, pending registry entry, ledger,
registry cons cell and result list before requesting the native clone. Native
construction allocates and copies the I1 before enrollment. A native failure
leaves only an inert Eshkol tombstone; if a later wrapper operation raises,
the guard releases the exact new kind-5 owner, tombstones the pending entry
and rethrows the first error. The existing
`g3t-generation-tensor-release!/1` now also admits an exact `ids` shell while
preserving its `input` route. Native typed release checks an active I1 borrow
before destruction; exact dead release is idempotent.

The focused witness covers forged, wrong-kind, pending, dead and borrowed
output rejection; native header and I1 allocation cuts through the wrapper;
fresh list/owner identity; borrowed kind-5 release; and detached P2/G0,
P1/G0 and P1/G1 results surviving output and generator release. The source
checker pins the wrapper's preallocation and no-orphan order. The focused
runtime does not inject Eshkol allocator failures before native clone; those
operations precede any native publication. No public G3-G facade, package,
other result accessor or CLI is added.
