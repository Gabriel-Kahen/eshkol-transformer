# G3-T private RNG constructor leaf

This source-private leaf implements the accepted
`generator_rng(model_owner, rng, mode, temperature_bits, k, p_bits, max_new,
eos)` native stem. It authenticates the exact live M3T model owner and G3-T
kind-8 RNG clone before reading the four source words. A pending, dead, busy,
forged, or wrong-kind source cannot construct a context. The existing C2
policy checks and capacity-two A2 cache construction are identical to the
seeded constructor. All four RNG words are copied into the new idle generator
before registry enrollment. The new generator holds no pointer or borrow to
the source RNG; the seeded constructor path is unchanged.

The focused witness uses final P1/G0 and P2/G0 snapshots
`[1,1729,0,0]` and a categorical P1/G1 snapshot `[1,1729,1,0]`.
The constructor preserves each word without a draw, starts with an empty
cache and no pins, and remains usable after its source RNG, output, and
generator are released. Forged model/source, wrong kind, dead source, busy
source, invalid policy, header allocation failure, and A2 allocation failure
leave the registry and source unchanged. The busy toggle is test-only and
excluded from the production native object.

The pinned, network-disabled Ubuntu 22.04/LLVM 21 focused gate passes
46,491 checks each in normal, repeat, and ASan/UBSan/LSan modes with
identical stdout and empty compile/runtime stderr. Seven inherited G3-T
source contracts and Q0 4/4 pass. The sanitizer run uses
`detect_leaks=1`, `halt_on_error=1`, and arena poisoning. The production
native object excludes testing hooks.

No Eshkol RNG-owner entry exists in the current G3-T registry, so this leaf
adds no Eshkol wrapper. That mapping, public G3-G facade, package exports,
and CLI remain separate dependencies. This gate proves only construction and
no-draw state; generated output from the new context is not asserted here.
