# G3-T private generated-length clone leaf

This source-private leaf implements only the accepted
`output_lengths_clone(ptr out)` stem: a detached kind-6 I1[1] tensor whose
value is the generated-token count G, zero for P1/G0 and P2/G0 and one for
P1/G1. It extends the accepted kind-5 ID clone and typed tensor release,
without adding an Eshkol accessor, public facade, package export or CLI.

The native call authenticates an independently live, completed output and
preflights its ID owner for an active borrow before allocating. It creates and
fills a new I1[1], then enrolls the kind-6 record; header and every I1
allocation failure leave the source output, cache, RNG and live-clone
membership unchanged. Typed release rejects a borrowed clone before mutation,
destroys its I1 payload, and retains an idempotent tombstone. Clones remain
valid after output and generator release.

The P2-enabled source witness tests all three P/G cases, zero/one exact
values, pending and borrowed source rejection, both allocation families,
borrowed-clone release rejection, survival and tombstones. The production
object must exclude all development observers. High-level result wrapping,
cache-length/RNG clones, public G3-G and retention remain separate gates.

The pinned Ubuntu 22.04/LLVM 21, network-disabled focused gate passes 281
checks each in normal, repeat and ASan/UBSan/LSan modes with identical stdout
and empty compile/runtime stderr. Six inherited source contracts and Q0 4/4
pass. The sanitizer launch uses `detect_leaks=1`, `halt_on_error=1` and arena
poisoning. This is candidate evidence, not public package acceptance.
