# G3-C4 private manual A2 ordinal 10

This step adds only the internal ordinal 10 precursor. No Eshkol entry or public role_step is added, and it does not claim the remaining 11 roles. The accepted 21-role schedule and G3-T private contract remain authoritative.

At ordinal 10, prefill T1/T2 or decode T1 creates a frame-owned capacity-2 A2 candidate cache with distinct K/V. Decode copies the committed length-1 K/V prefix under a read borrow and commits it only on the candidate. It stages the current projected KH/VH in a pending candidate transaction. A nested full-capacity view supplies K/V and validity for the exact A2 `kernel.causal-attention` / `causal-attention.forward` row `[1,2,2,T,2,2]`, with explicit positions and bool `[1,T,2]` mask. The view ends before AH and candidate handles enter the frame at ordinal 11.

On ordinal failure, the view and candidate unwind and the frame stays at ordinal 10 for retry. Committed cache, pending logits, binding, and RNG do not change. Call abort destroys pending logits, aborts the frame-owned transaction, destroys its candidate, then frees the frame; a borrowed pending logits tensor blocks that cleanup and permits retry. No RoPE is applied.
