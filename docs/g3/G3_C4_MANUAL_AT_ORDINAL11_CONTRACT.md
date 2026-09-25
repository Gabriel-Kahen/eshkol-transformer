# G3-C4 private manual attention merge, ordinal 11

This step adds only the internal ordinal 11 precursor. No Eshkol entry or public role_step is added, and ordinals 12–20 remain absent. The accepted 21-role G3-T schedule is authoritative.

After ordinal 10 has left a frame-owned capacity-2 A2 candidate cache and pending transaction, ordinal 11 merges AH `[1,2,T,2]` to AT `[1,T,4]`. T1 prefill and T1 decode use the accepted G3-N `g3n.head-layout-forward` / `g3n.heads.merge.forward` row `[1,1,2,2]`. T2 prefill uses N3K `n3k.head-layout` / `n3k.heads.merge.forward` row `[1,2,2,2]`. One provider call writes local scratch; only success copies AT and advances the frame to ordinal 12.

Provider failure leaves AT and ordinal 11 unchanged for retry. The committed cache, pending logits, binding, RNG, AH, and pending A2 candidate remain unchanged on both success and failure. Call abort still owns and discards the pending A2 transaction and candidate after pending logits can be destroyed. This seam neither commits the A2 candidate nor publishes logits.
