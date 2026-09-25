# TR3-C private trainer load-state result cell

`tr3-c-trainer-load-state-internal!` accepts an enrolled live trainer, an exact
detached C2 owner, and a caller-owned, promoted one-slot result cell initialized
to `#f`. It rejects malformed or occupied cells before entering restore, then
calls the accepted `tr3-c-joint-restore-internal!` transaction unchanged.
Failure leaves the cell empty; success publishes `#t` only after the joint
transaction has returned the receiver to idle. The detached C2 owner remains
with the caller on both paths.

The supported strict-O0 f31/LLVM 21 private gate uses the genuine joint
aggregate and its injected last recoverable O2 check. It rejects nonvector,
short, long, and occupied result cells; rejects a forged and a busy detached
source; proves failed calls leave the cell empty, receiver image unchanged,
source owner live, and receiver idle; then retries successfully and compares
all 42 tensor byte images and controls with the advanced source.

Public `trainer-state` and `trainer-load-state!` remain blocked by the absent
public trainer constructor and package identity. The installed
`transformer.trainer` facade exports only `trainer-state-release!`; the only
live trainer constructor is source-private `tr3-lease-create-internal`. The
current C2 public archive composes D2/O2/K2/C2 without M3/TR3 and has no
TR3 export or bridge. This leaf installs no public operation or C ABI.
