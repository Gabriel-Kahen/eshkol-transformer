# G3-C4 private manual numerical tail, ordinals 14–20

This step adds only seven internal numerical precursors. No Eshkol entry or public role_step is added; frame commit, sampling, and generation remain absent. The accepted 21-role G3-T schedule is authoritative.

| Ordinal | Source and result | T1 prefill/decode route | T2 prefill route |
|---|---|---|---|
| 14 | R, gamma[9], beta[8], epsilon 0x3727c5ac → N2 `[1,T,4]` | G3-N layer norm | N2 `kernel.norm` |
| 15 | N2, W_up[5] → FU `[1,T,8]` | G3-N linear | N3K linear |
| 16 | FU → FG `[1,T,8]` | N2 `kernel.activation` / `gelu.forward` `[1,1,8]` | N3K `n3k.gelu` / `n3k.gelu.forward` `[1,2,8]` |
| 17 | FG, W_down[4] → FD `[1,T,4]` | G3-N linear | N3K linear |
| 18 | R, FD → Y `[1,T,4]` | G3-N residual | N3K residual |
| 19 | Y, gamma[12], beta[11], same epsilon → NF `[1,T,4]` | G3-N layer norm | N2 `kernel.norm` |
| 20 | NF, tied embedding weight[10] → Z `[1,T,256]` | G3-N linear | N3K linear |

Every ordinal makes one exact provider dispatch into separate local scratch, then copies only its result and advances the frame on success. A failed call leaves the current ordinal and its scratch unchanged and retryable. Earlier numerical slots, the committed cache, pending logits, binding, RNG, and pending capacity-2 A2 candidate remain unchanged. Ordinal 20 leaves a frame at next ordinal 21 with staged logits but no publication or commit. Call abort still owns and discards the candidate after pending logits can be destroyed.
