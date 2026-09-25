# G3-T private sampled-token frame staging leaf

This bounded successor to the P1/G1 prefill and speculative-sample leaf executes
the candidate token at absolute position 1. `frame_begin(ctx, NULL, 2)` accepts
only a sampled generate call with a committed P1 prompt and an exact 14-parameter
binding match. It copies the sampled ID into the frame, reuses the held pins,
and runs the fixed 21-role transcript. Ordinal 10 stages one token's K/V in a
capacity-2 A2 transaction on the committed prompt cache. The transaction view
supplies effective length 2, both visible keys and absolute query position 1.
The two-token M3T forward schedule independently checks all 256 resulting
position-1 logits bitwise.

The append remains speculative. A role failure leaves the ordinal unchanged;
the A2 transaction abort scrubs any staged tail. Call abort preserves the
committed prompt cache, binding and original generator RNG, and tombstones the
pending output. A model profile/eval change is checked by the private Eshkol
wrapper; the native begin and final preparation recheck parameter identities
and all 4,736 value bytes. Wrong input, frame kind, duplicate begin and ordinal
order reject before publication. The source-private wrapper continues to abort
at call exit under the shared M3 guard.

Final `frame_prepare` and `frame_commit` reject this sampled frame until the
accepted output readiness, raw T1 decode, cleanup preflight and atomic
cache/ID/RNG/output publication tail are implemented. This leaf adds no public
G3 generation API, published token or RNG advance.
