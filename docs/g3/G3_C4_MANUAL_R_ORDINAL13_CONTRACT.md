# G3-C4 private manual attention residual, ordinal 13

This step adds only the internal ordinal 13 precursor. No Eshkol entry or public role_step is added, and ordinals 14–20 remain absent. The accepted 21-role G3-T schedule is authoritative.

After ordinal 12 produces AO `[1,T,4]`, ordinal 13 adds the accepted X and AO operands, in that order, to R `[1,T,4]`. T1 prefill and T1 decode use G3-N `g3n.residual-forward` / `g3n.residual.forward`; T2 prefill uses N3K `n3k.residual` / `n3k.residual.forward`. Both request row `[1,T,4]`. One provider call writes local scratch; only success copies R and advances the frame to ordinal 14.

Provider failure leaves R and ordinal 13 unchanged for retry. X, AO, the committed cache, pending logits, binding, RNG, and pending capacity-2 A2 candidate remain unchanged on both success and failure. Call abort still discards the candidate after pending logits can be destroyed. This seam neither commits A2 nor publishes logits.
