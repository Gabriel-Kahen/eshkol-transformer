# TR3 explicit trainer release candidate

Status: non-installed source candidate; the linked aggregate gate is pending.
The accepted A0 amendment makes `trainer-release! trainer` an arity-one
operation. Successful release returns `#t`, ends the exact trainer's exclusive
lease, and leaves the resolved configuration, tokenizer, dataset, model, and
optimizer caller-owned and live. Dropping a trainer reference alone does not
release the strong registry edge. This contract does not promise physical
reclamation of the caller-region trainer shell.

The new root loads the unchanged non-installed constructor candidate and calls
the accepted `tr3-lease-unenroll-internal!` directly in the same registry-owning
aggregate. Its single unlink is the release commit point. Before unlink, the
private operation authenticates the exact shell and checks registry, trainer,
M3T, D2 and O2 idle state. Forged/repeated shells raise `invalid-argument`;
busy release raises `invalid-state` without ending the lease. The candidate
re-maps a structured error's category, message, detached details, and existing
cause to public operation `trainer-release!`, following the accepted D2 pattern.
It never exposes the private unenroll error as a new cause. A foreign raise is
`internal/trainer-release!` with no invented public cause.

The release bridge includes the existing constructor bridge once and adds only
release and accepted D2 next-batch/release boxed entries. The caller constructs
all five real operands through the same archive, proves forged/copied/busy
release preserves the lease, releases a genuine live batch, then releases,
re-leases, checks the old shell tombstone, releases again, and closes the
caller-owned D2 receiver. The candidate gate pins the single archive member,
source/native depfiles, exact private renames, dynamic C exports, unresolved
runtime symbols, fixed compiler/runtime and poisoned linked caller. It does
not install a public facade or change the production initializer-only export
boundary. Step/resume and full trainer acceptance remain separate work.
