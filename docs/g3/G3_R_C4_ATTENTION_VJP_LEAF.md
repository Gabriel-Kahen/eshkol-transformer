# Private C4 causal-attention VJP leaf

Status: the pinned native Clang 21 normal/repeat/ASan+UBSan leaf gate and
independent result review are accepted.

The accepted G3-C4 forward provider's fixed full-length self-attention row
has Q, K, V and output `[1,2,4,2]`, query/key positions `[1,4]`, and keep
mask `[1,4,4]`. The private VJP accepts the same six inputs plus upstream
`[1,2,4,2]`, and writes dQ, dK, dV of that same shape. CPU f32 dense views
and positions `(0,1,2,3)` are required; keep bytes are zero or one. A key is
admitted when keep is one and its position is no later than the query. A row
with no admitted key yields zero output and zero gradients.

For each admitted row, `p = softmax((Q K^T) / sqrt(2))`. With upstream U,
`dV_j += p_j U`, `g_j = U dot V_j`,
`ds_j = p_j (g_j - sum_l p_l g_l)`,
`dQ += (ds K) / sqrt(2)`, and `dK += (ds^T Q) / sqrt(2)`.
All three gradients are staged until every view, value, mask, position,
overlap and finite result check succeeds. It rejects a nonfinite
score-minus-maximum before `expf`, matching the forward provider even when
both scores themselves are finite. The function retains no state.
Its fixed loops are a native numerical leaf and have no scalar fallback.

The focused test checks all 48 central differences for both a full causal
mask and a sparse mask with one empty row, through the genuine forward
provider. It also checks malformed views, shape, dtype, device, layout,
overlap, positions, mask, nonfinite inputs, overflowing score and unchanged
outputs after rejection. Normal/repeat/native ASan+UBSan, feature-off and
ordinary-provider symbol isolation are in the focused runner. This leaf is
not a composed trainable C4 backward graph or trained checkpoint.

Source `38c3ff7c`/tree `b7728c7d` passed 178 checks per mode with byte
parity, empty runtime stderr, feature-off rejection, and exact 2,607-file
source custody. The 49-file sealed receipt is
`g3r-c4-attention-vjp-38c3ff7c-attempt2/SHA256SUMS-RUN` (SHA-256
`8a43a5f7c69b9ba2a9c99674d712ad7be228f1637f92b16c28f53349b68b58af`).
Independent result review is sealed at
`g3r-c4-vjp-leaves-independent-result-review-20261025/SHA256SUMS-REVIEW`
(SHA-256 `09d4a732d6734b1fc8dcd55b816027755bd5c09e2bf146138ea393209cdfc24f`).
The earlier v1 recipe remains held for its finite-score softmax-shift parity
defect; it was never run.
