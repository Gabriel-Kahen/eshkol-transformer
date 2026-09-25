# G3-C4 private manual attention-out projection, ordinal 12

This step adds only the internal ordinal 12 precursor. No Eshkol entry or public role_step is added, and ordinals 13–20 remain absent. The accepted 21-role G3-T schedule is authoritative.

After ordinal 11 produces AT `[1,T,4]`, ordinal 12 projects it through pinned attention-out weight W_o at `context->pins.views[1]` to AO `[1,T,4]`. T1 prefill and T1 decode use accepted G3-N `g3n.linear-forward` / `g3n.linear.forward-no-bias`; T2 prefill uses N3K `n3k.linear` / `n3k.linear.forward-no-bias`. Both request row `[1,T,4,4]`. A single provider call writes local scratch; only success copies AO and advances the frame to ordinal 13.

Provider failure leaves AO and ordinal 12 unchanged for retry. AH, AT, the committed cache, pending logits, binding, RNG, and pending capacity-2 A2 candidate remain unchanged on both success and failure. Call abort still discards the candidate after pending logits can be destroyed. This seam neither commits A2 nor publishes logits.
