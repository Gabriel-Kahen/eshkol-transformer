# Private C4 no-bias linear/projection VJP leaf

This bounded leaf follows the accepted C4 position, residual, and GELU VJPs.
The genuine G3-C4 provider exposes forward-only
`g3c4.linear.forward-no-bias` at four-token rows `(I,O)` of `(4,4)`, `(4,8)`,
`(8,4)`, and `(4,256)`. The forward operands are `X:f32[1,4,I]` and
`W:f32[O,I]`, producing `Y:f32[1,4,O]` by serial row-major sums.

Under `ET_M3T_C4_LINEAR_VJP_PRIVATE`, the synchronous borrowed-view entry
`et_m3t_c4_linear_vjp_private_v1(X,W,U,dX,dW)` accepts exact dense CPU f32
K1 v1 views. The upstream is `U:f32[1,4,O]`; outputs are `dX:f32[1,4,I]`
and `dW:f32[O,I]`. The equations are `dX[t,i]=sum_o U[t,o]*W[o,i]` and
`dW[o,i]=sum_t U[t,o]*X[t,i]`. All views have offset zero, exact byte
lengths, aligned storage, and an admitted row. Both destinations must be
disjoint from every source and each other. Input, weight, upstream, forward
products/sums, and both VJP products/sums must be finite. The implementation
stages at most 32 input-gradient and 1,024 weight-gradient floats on the
stack and publishes both only after complete validation. Any rejection leaves
both full destinations byte-identical. No heap allocation, retained owner,
public provider capability, or fallback is added; feature-off builds lack the
entry.

The focused test checks all `dX` entries for every admitted row and all `dW`
entries for the small rows, plus 32 distributed weights across the 256-wide
row, by central finite differences through the genuine C4 forward provider.
It also checks source preservation, malformed shape/rank/length/dtype/device/
layout, unaligned storage, nonfinite and overflow inputs, null views, exact
and partial output aliases, and unchanged destinations on rejection. This
single private primitive does not wire a trainable C4 M3T graph or admit a
trained checkpoint, SAVE/LOAD, generation, or persistence.
