# Private C4 token-embedding VJP leaf

Status: bounded source candidate. Independent review and pinned compiled
admission remain pending.

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
