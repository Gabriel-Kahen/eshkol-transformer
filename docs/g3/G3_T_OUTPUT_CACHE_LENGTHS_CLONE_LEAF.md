# G3-T private cache-length clone leaf

This source-private leaf implements only the accepted
`output_cache_lengths_clone(ptr out)` stem: a detached kind-7 I1[1] tensor
whose value is P+G. The three admitted paths yield P1/G0=1, P1/G1=2 and
P2/G0=2. It extends the accepted kind-5 ID and kind-6 generated-length
clones and typed tensor release, without an Eshkol accessor, public facade,
package export or CLI.

The native call authenticates an independently live, completed output and
checks its exact P/G/cache-length relation and unborrowed source ID owner
before allocation. It fills a fresh I1[1] and enrolls the kind-7 record only
after full success. Header and each I1 allocation cut leave output, cache,
RNG and live-clone membership unchanged. Typed release rejects an active clone
borrow before mutation, destroys its I1 payload, and retains an idempotent
tombstone. Clones survive output release and generator close.

The P2-enabled source witness checks exact values across all three paths,
pending and borrowed source rejection, header and four I1 allocation cuts on
G0 and G1, borrowed-clone release rejection, detached survival and tombstones.
The production object excludes development observers. High-level result
wrapping, RNG clones, public G3-G and retention remain separate gates.

The pinned Ubuntu 22.04/LLVM 21, network-disabled focused gate passes 320
checks each in normal, repeat and ASan/UBSan/LSan modes with identical stdout
and empty compile/runtime stderr. Six inherited source contracts and Q0 4/4
pass. The sanitizer launch uses `detect_leaks=1`, `halt_on_error=1` and arena
poisoning. This is candidate evidence, not public package acceptance.
