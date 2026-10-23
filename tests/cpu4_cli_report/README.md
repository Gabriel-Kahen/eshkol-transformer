# Private CLI scalar/report producer candidate

Implements the accepted v3 contract in one proposed CLI package shim,
`native/cpu4_cli_package_bridge.c`, its source-private header, and the closed
`native/cpu4_cli_report_extension.esk` leaf. The common CPU4 root, base35 roster,
training/control/E3 producers, installed allowlists and roadmap are unchanged.

Five native helpers create/read/claim/release canonical ordinary rank0 CPU f32
owners and query three TLS diagnostic words. Nine Scheme interfaces authenticate
private rows; snapshot actual controls under WRITE=2, capture the unchanged full
step once, detach final CE numerator/weight/mean and active-target token counts,
and evaluate the same binding into four claimed scalars. Evaluation capture and
cleanup retry their exact phases; a failed evaluator requires explicit release.
A failed final report preserves its committed result/U and blocks another step.
The command and cumulative update limits are250; accumulation must be1. Command
exclusivity excludes an arbitrary private restore to equal U from staleness proof.

## Host checks executed

- `guile --no-auto-compile tests/cpu4_cli_report/host_test.scm`:53 checks PASS.
  All native/F32/control/training/E3 calls are explicitly labeled doubles. Guard
  forms are instrumented only in the test evaluator to fault handler installation.
  Coverage includes exact shell/copies, control-end custody, stale-before-step,
  report corruption/advanced-U/no double-step, cleanup blocking, cumulative U250,
  heldout retry identity, diagnostics before cleanup and reentrant busy ownership.
- `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/cpu4_cli_report -p 'test_*.py'`:
  seven source-contract checks PASS.
- Guile reader parsed the production leaf and both runtime Scheme inputs.

## Root-dispatched inputs; execution NOT RUN

- `native.c`: actual native scalar unit against base35+one production shim,
  covering raw-bit patterns, rank0 extent, foreign/interior/NULL/DEAD custody,
  claimed release, real borrow/scoped/pin conflicts, TLS reset and repeat release.
- `native_fault.c`: standalone C logic fixture including the production shim
  with explicitly renamed F32/allocation doubles. Compile this TU alone, without
  linking a second shim. Covers sidecar/native-create failures, every read-call
  cut, zeroed error sinks, absent/inconsistent/out-of-range diagnostics falling
  back to INVARIANT, scope drainage and rejected-destroy same-owner retry.
- `runtime.esk`: executable argv `train_directory heldout_directory`. Loads one
  unchanged common root and the new leaf, runs two genuine full updates then
  same-binding E3, checks report copies and releases report before candidate/D2
  retirement. `preservation.esk` reuses the exact existing C2/P1/O2 snapshot/copy
  functions from `tests/cpu4_resume/body.esk`, covering all14 value planes and28
  moment planes, metadata/topology/controls; modes, cursor, absent gradients and
  original component identities are checked separately. No observer TU or native
  testing flags are needed. No hardcoded unmeasured loss expectation exists.

Compiler/native/AOT/normal-repeat/SAN execution, malformed native metadata,
exhaustive allocation cuts, package/export inventory, exact retained-byte census,
installed CLI/JSON/teardown and continuation admission remain NOT RUN. These
inputs establish no learning, performance, installed or hardware claim.

## Resource delta awaiting CLI3 census

The C shim allocates exactly one32-byte LP64 sidecar BEFORE each native scalar
create; an unpublished sidecar is freed on create failure. Published sidecars
remain LIVE/DEAD tombstones, including after release. Each actual scalar also
charges its existing F32 owner/retired-control/index bookkeeping; the four-byte
payload is not the total ownership charge. Each E3 report creates four such
scalars. Scheme allocations include scalar row7/shell1/diagnostic3 and report
row28/shell1/scalars4/recovery cell, detached controls and staged report vectors.

Every actual controls getter, initial bind, per-step preflight, final copy and
before/after E3 snapshot creates an existing control parent that stays closed in
its upstream registry. Native validation snapshots add their own real closed
parents/images and must be separately inventoried. Command bounds do not bound
unlimited caller getter/evaluation retries; the command integration must census
its exact call schedule. No StageA observer or provisional resource API is used.
