# TR3-C private checkpoint-to-live restore

`tr3-c-checkpoint-restore-internal!` accepts an enrolled live trainer, path,
exact C2 persistence policy, and caller-owned K2 capability report. It uses
the accepted `c2-public-checkpoint-load` compositor, which validates and
stages the file, admits every tensor through that report, and reconstructs a
detached C2 owner. It passes the exact owner to the accepted private
`tr3-c-joint-restore-internal!` transaction, then releases the owner with C2.
There is no public trainer facade or new checkpoint format.

The private wrapper installs a guard before LOAD. Its one-slot cell owns the
detached shell from LOAD return through joint restore and is cleared before
release. C2 LOAD and recoverable TR3 errors propagate unchanged after cleanup.
A release defect becomes `internal` under `trainer-load-state!`. The joint
restore borrows the owner; it does not consume it. C2's own LOAD compositor
continues to own and retire staging on prepublication failure.

The focused strict-O0 gate composes one private C2/K2/TR3 identity universe
under the pinned supported f31/LLVM 21 runtime. It creates a genuine
checksummed file via private snapshot SAVE, then checks checksum corruption,
truncation, malformed path, forged K2 report, and an authentic receiver with
different canonical X1 run seed. Each rejection preserves the receiver and
live source trainer; failures after LOAD return release the exact detached
owner once. The valid path compares all 42 restored tensor byte images and
controls against the saved source and confirms release and idle state.

This bounded seam does not establish public trainer wiring, a nontrivial
interrupted/resumed trajectory, all allocation-failure cuts, or general file
change detection across C2 LOAD's two reads.
