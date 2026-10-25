# Private C4 token-embedding VJP leaf

Status: pinned native Clang 21 normal/repeat/ASan+UBSan leaf gate and
independent result review accepted.

The genuine G3-C4 `g3c4.embedding.forward` row has raw V256 IDs `[1,4]`,
the tied token/head table `[256,4]`, and output `[1,4,4]`. This private
backward leaf accepts IDs and a `[1,4,4]` upstream gradient and stages the
full `[256,4]` token-table gradient. For each token `t` and channel `d`,
`dTable[ID[t],d] += upstream[t,d]`. Repeated IDs accumulate; all unused rows
are exactly positive zero. There is no derivative for integer IDs. The
table gradient must later be summed with the tied output-head gradient by a
separate reviewed graph composition.

The leaf requires exact dense CPU i64/f32 views, in-range IDs, finite
upstream and gradient sums, and nonoverlapping output storage. It returns
status only and leaves all output bytes unchanged on failure. Its fixed
native scatter is a numerical kernel with no fallback.

The focused test checks all 1,024 gradient words, including a repeated ID,
and central differences on every touched coordinate plus distributed unused
coordinates through the genuine C4 forward provider. It also checks shape,
dtype, device, layout, alias, invalid IDs, nonfinite values, accumulation
overflow, null views and feature-off isolation. This is not a composed C4
backward graph, trained model or checkpoint continuation.

Source `f58b4b85`/tree `0dd3e23e` passed 1,131 checks per mode with byte
parity, empty runtime stderr, feature-off rejection and exact 2,607-file
source custody. The 49-file sealed receipt is
`g3r-c4-embedding-vjp-f58b4b85-attempt1/SHA256SUMS-RUN` (SHA-256
`6e5429895cf05d535945eb88b4bea4d767669ab43df338c8a83de1a4cdc2f03c`).
Independent result review is sealed at
`g3r-c4-vjp-leaves-independent-result-review-20261025/SHA256SUMS-REVIEW`
(SHA-256 `09d4a732d6734b1fc8dcd55b816027755bd5c09e2bf146138ea393209cdfc24f`).
