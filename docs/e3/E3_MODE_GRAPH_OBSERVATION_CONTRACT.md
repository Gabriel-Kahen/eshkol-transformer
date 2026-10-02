# E3 in-call mode and graph observation contract

This is a **design-only, noninstalled test contract** for the remaining mode
and graph obligations in [E3 §9](../E3_EVALUATION_PROPOSAL.md). It adds no
runtime observer, source mutation, native API, accepted result, or edge-corpus
claim. This read-only design audit uses main `d3988abd` and the separately
reviewed, unmerged E3 test candidate `c78682ae`/tree `853f83e7`; its retention
and numerical candidate files are not promoted by this document. The existing
[P1 mode contract](E3_P1_MODES_CONTRACT.md) and those two test candidates
address different parts of §9. None reads all 17 modes *inside* the evaluator
or counts transient graph creation. Implementation must wait for the two
source/ownership decisions identified below.

## Exact mode witness

`m3t-entry` for the authentic M3 model stores the fixed 17 P1 module shells in
slot 4. Their order is the accepted P1 order: root; blocks; block 0; attention;
key, output, query, value; FFN; down, up; norm1, norm2; head; final norm;
position embedding; token embedding. The unmerged candidate's
`tests/e3_private/retention_driver.esk` uses this vector and
`module-mode-internal` to snapshot mixed train/eval symbols before a call and
compare all 17 after success, later D2 read failure, and same-authority retry.
That postcall witness cannot detect a child left in train mode during forward
if restoration later repairs it.

The in-call test seam must receive the **preauthenticated canonical 17-shell
vector**, captured before the protected call. The private evaluator invokes it
only after successful `p1-e3-mode-enter!` and entry-bit publication in
`e3-evaluate-closed!`, immediately before `e3-traverse!`, then at the start of
each entered batch before its first forward role. It reads every shell's actual
mode through the existing `module-mode-internal` and records a fixed 17-bit
eval mask, call phase, and batch ordinal in a preallocated test-owned cell.
Every recorded in-call mask must be all-eval. Use an actual mixed17 starting
state and a compiled missed-child-mode mutation that leaves one nonroot child
in train during traversal; the in-call mask, not a postcall proxy, must kill
that mutant. A later read fault must still observe the entered mode state before
restoration, followed by exact mixed17 restoration and a clean retry.

The observer is read-only with respect to modes, model, frame, D2, destinations
and provider state. It performs no public mode setter, graph/VJP/reset,
provider call, allocation, callback, exception construction, or printing while
the E3 guard is held. It reports only immediate bits into its preallocated
test cell; assertions and output occur after the guarded call. If observation
itself cannot complete without reentering protected authority, fail the test
tuple rather than treating an unobserved call as a pass. A production failure
retains its first E3 error and cleanup path; a test observation must not replace
the original category/stage or turn a recoverable failure into a new rollback
branch. Successful publication and all six caller outputs remain production
owned. Parameter and present/absent gradient bytes, typed RNG/initializer
successor, cursor, four destination bytes, two counters, and mode restoration
retain the candidate's success/failure/retry comparisons.

The present `.esk` build does not C-preprocess Scheme and has no accepted
conditional in-call hook in `native/e3_private_extension.esk`. Before code,
review an **exact noninstalled source-composition method** that inserts these
calls without changing canonical E3 execution or duplicating a diverging
evaluator. A postcall wrapper, P1 public setter, or test-only callback invoked
from the protected region does not satisfy this seam.

## Real graph event, and current reachability blocker

The same E3 aggregate includes `src/eshkol_transformer/m3_model.c`. The real
`et_m3_private_graph_capture_v1` creates an `m3_graph` and registers its
`M3_GRAPH` record at `enroll(&g->r, M3_GRAPH)`. The unmerged retention
candidate's guarded probe in `src/eshkol_transformer/e3_frame.c` scans the
registry for *live* graph records at selector 21; a graph created and released
within a call can disappear before that postcall scan. A future exact test tuple
may add a monotonic test-only event counter at that actual enrollment and a
separate capture-attempt counter at the real entry. Snapshot both after
model/frame setup, then require no
per-call delta at the in-call observation points and after success, later fault,
and same-authority retry. Also require live graph owners return to baseline.
Keep attempt, successful creation, and live-owner values distinct; no fabricated
registry insertion or field-layout cast is an acceptable graph witness.

**Creation-mutant reachability is presently unproved.** E3 acquisition sets
`frame->model->active = frame` in `e3_frame.c`. Real M3 graph capture admits an
M3 workspace and `m3_workspace_eligible` requires an active workspace to be
that model's active owner, as well as a ready forward result. A direct capture
on the E3 frame is not a valid M3 workspace; an M3 workspace on the same
actively held model cannot meet that owner check. A deliberately rejected
capture attempt would test the attempt counter but would *not* prove a genuine
graph-create/release mutant. The prerequisite is an independently reviewed
test-only source mutation/owner schedule that executes the real M3 forward,
capture and release while preserving the same E3 model's guard and authority.
If that schedule is impossible under the accepted owner contract, report the
graph-producing mutation obligation as blocked; do not relax M3 admission,
switch to an unrelated model, or claim a transient-creation kill from a
rejected attempt. A structural absence audit can supplement, not replace, a
reachable compiled mutation witness.

## Isolation and acceptance gate

Any implementation uses a distinct exact noninstalled test tuple with its own
root, bridge, source/native closure, undefined/defined/export/string inventories
and reviewed compiler flags. The test mode/graph observers and mutant flags
must be absent from canonical E3 private, diagnostic-private, installed public,
retention, and numerical artifacts; compare their exact inventories and test
near-match tuple rejection. No new installed symbol or general M3/P1 authority
is implied. Test-only native counters must be grounded in actual owner
operations, compiled only for this exact tuple, and reconciled with existing
live-owner counts. Run the genuine success, later D2 failure, and same-authority
retry trajectories under normal, poisoned optimized AOT and supported
ASan/UBSan/LSan with leak detection, with two fresh source/object/AOT builds
and deterministic output comparison. Keep edge corpora and other §9 numerical
mutants outside this leaf until their genuine traversal reachability is defined.
