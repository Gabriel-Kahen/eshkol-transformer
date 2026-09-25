# G3-T source-private P2/G0 zero-budget leaf

The private pair input owns two exact byte IDs. A read-only prompt preflight
rejects malformed length and `P+G>2` before call acquisition or model pinning.
Only P2/G0 reaches this leaf; P2/G1 is rejected before pins. The existing P1
and G1 paths retain their accepted contracts.

The P2 prefill runs one 21-role T2 schedule. N3K handles token and position
embeddings, residuals, linear rows, head layout, and GELU; N2 handles layer
norm; A2 attends over an unpublished two-position cache transaction with the
exact causal mask `{1,0;1,1}`. It publishes the true second-position logits,
both K/V positions, cache length two, empty ID/raw-text output, and unchanged
RNG only after binding, borrow, old-cache, and pin preflight.

The focused witness compares all 256 second-row logits bitwise with the
independent M3T two-token forward. It covers invalid pair IDs, length above
two, P2/G1 before pins, A2 creation and attention cuts, stale binding,
borrowed output, empty T1 decode, live output release, and rollback. The
same P2-enabled native object also passes a complete G1 route. Supported
f31/LLVM21 network-none normal, repeat, and ASan/UBSan/LSan each pass 201
checks with identical output and empty stderr; the inherited G1 publication
witness passes 118 checks on this checkout. This leaf adds no public
generation facade, continuation, accessor, or CLI.
