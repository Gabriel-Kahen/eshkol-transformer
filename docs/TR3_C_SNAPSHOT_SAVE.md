# TR3-C private snapshot-to-checkpoint publication

`tr3-c-snapshot-save-internal!` takes a live trainer, path, exact C2 policy,
and C2 SAVE options. It composes one detached owner with the accepted private
TR3-C snapshot operation, passes that exact owner to
`c2-checkpoint-save-internal!`, and releases it with the accepted C2 release
operation. It returns `#t` only after the C2 writer succeeds and release
finishes. No public trainer operation or checkpoint format is added.

The one-slot result cell is the sole authority for the detached owner. A
guard is installed before snapshot composition; after SAVE returns or raises,
the cell is cleared before release. A C2 writer error is re-raised unchanged,
including `published?` and `durability`. A release defect is reported as
`internal` under `trainer-state`. Clearing before release prevents a second
release if a trusted callback raises after consuming the owner. The live
trainer remains idle and independent of the detached owner.

The focused compiled gate uses the genuine snapshot composer, C2 encoder,
full parse validation, and C1 atomic writer in one identity universe. It tests
repeat byte identity, malformed trainer/path/options, no-overwrite, injected
precommit and postcommit writer errors, post-publication wrapper failure,
foreign raised-value preservation, and post-release callback failure. The gate
uses the pinned f31/LLVM 21 runtime at strict O0. It does not prove public
trainer wiring, checkpoint LOAD, joint restore, or interrupted/resumed
trajectory equivalence.
