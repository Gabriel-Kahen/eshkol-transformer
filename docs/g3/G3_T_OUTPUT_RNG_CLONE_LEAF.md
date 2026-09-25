# G3-T private output RNG clone leaf

This source-private leaf implements only the accepted
`output_rng_clone(ptr out)` and `rng_release(ptr rng)` stems. The clone is a
detached kind-8 native owner with four aligned inline i64 words, not an I1
tensor. Its immutable snapshot is the final output RNG: P1/G0 and P2/G0 give
`[1,1729,0,0]` in the focused witness, and categorical P1/G1 gives
`[1,1729,1,0]` after its single committed draw. No Eshkol result accessor,
public facade, package export or CLI is introduced.

Clone authenticates the exact live, completed output and checks the C2 P/G
length relation before allocation. It reads only inline output RNG words, so
an active borrow of the output's I1 IDs does not block it. The complete
record is allocated and copied before registry enrollment. A failed header
allocation enrolls nothing and leaves output, cache and RNG unchanged.
`rng_release` rejects the wrong owner kind before lifecycle inspection; a
live release zeros all four words and retains an idempotent dead tombstone.
The clone survives output release and generator close. Generic tensor release
never accepts kind 8.

The P2-enabled private witness covers pending/dead/wrong-kind rejection,
borrowed-ID independence, the allocation cut, exact words on all three
accepted P/G paths, typed release, parent survival, zeroization and
idempotence. Production test observers remain excluded. Public G3-G and
P2/G1 capacity-three work remain separate dependencies.

The pinned Ubuntu 22.04/LLVM 21, network-disabled focused gate passes 345
checks each in normal, repeat and ASan/UBSan/LSan modes with identical stdout
and empty compile/runtime stderr. Six inherited source contracts and Q0 4/4
pass. The sanitizer launch uses `detect_leaks=1`, `halt_on_error=1` and arena
poisoning. This is private candidate evidence, not public package acceptance.
