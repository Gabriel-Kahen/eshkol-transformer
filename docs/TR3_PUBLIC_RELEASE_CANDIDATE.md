# TR3 explicit trainer release candidate

Status: supported non-installed linked candidate. The exact source/test commit
is `38329f9` (tree `0c471f84`); no public facade is installed.
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

The fixed f31/LLVM21 network-disabled gate passed on the exact candidate head:
48 Eshkol sources, 68 native source/header paths, 30 native objects, 4,144 raw
defined symbols, 21 localized public C exports, and 173 unresolved runtime
symbols. The poisoned genuine caller printed
`TR3-PUBLIC-RELEASE-CANDIDATE-PASS` with empty stderr. Source compilation took
3:51.98, peaked at 3,589,176 KiB RSS and used no swap. The immutable evidence
is `/home/gabe/.codex/evidence/eshkol-transformer/tr3-public-release-candidate-38329f9-20260925/`;
`SHA256SUMS` hashes to
`cd7872e863a3c632c44386758860d78a46f146647f9db1a5dd3216bda58f7cad`
and passed `sha256sum -c` with exit 0. A0 declaration and arity-negative
compile-only checks passed separately. This does not prove release during an
active trainer step, physical shell reclamation, or full public trainer
installation.
