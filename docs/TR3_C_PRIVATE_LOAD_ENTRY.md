# TR3-C private trainer load-state result cell

`tr3-c-trainer-load-state-internal!` accepts an enrolled live trainer, an exact
detached C2 owner, and a caller-owned, promoted one-slot result cell initialized
to `#f`. It rejects malformed or occupied cells before entering restore, then
calls the accepted `tr3-c-joint-restore-internal!` transaction unchanged.
Failure leaves the cell empty; success publishes `#t` only after the joint
transaction has returned the receiver to idle. The detached C2 owner remains
with the caller on both paths.

The supported strict-O0 fe9/f31/LLVM 21 private gate uses the genuine joint
aggregate and its injected last recoverable O2 check. It rejects nonvector,
short, long, and occupied result cells; rejects a forged and a busy detached
source; proves failed calls leave the cell empty, receiver image unchanged,
source owner live, and receiver idle; then retries successfully and compares
all 42 tensor byte images and controls with the advanced source.

This leaf installs no public operation or C ABI. A bounded installed public
trainer state/load candidate now exists; full exact-resume acceptance and
public checkpoint I/O packaging remain pending.
