# Private C4 LayerNorm VJP leaf

Status: bounded private native leaf accepted by independent source/test and
compiled result review. The exact receipts are recorded in `docs/ROADMAP.md`.

The existing `g3c4.layer-norm.forward` provider admits `[1,3,4]` and
`[1,4,4]`. This feature-private leaf covers only the four-token row. The
native entry point `et_m3t_c4_layernorm_vjp_private_v1` accepts borrowed,
dense, row-major CPU f32 K1 views for `X,U,dX[1,4,4]`, scale `gamma[4]`,
offset `beta[4]`, scalar `epsilon`, and `dGamma,dBeta[4]`. It returns a status
only, retains no borrow, and writes none of the three destinations on failure.
All destinations must be disjoint from each other and every input.

For each token independently, the provider computes the mean over four
channels, biased variance `sum((x-mean)^2)/4`, `rstd=1/sqrt(variance+epsilon)`,
and `Y=(X-mean)*rstd*gamma+beta`. Epsilon is positive finite f32. With
`v=U*gamma`, this leaf computes
`dX=rstd*(v-mean(v)-(X-mean)*rstd^2*mean(v*(X-mean)))`,
`dGamma=sum_tokens(U*(X-mean)*rstd)`, and `dBeta=sum_tokens(U)`.
It checks finite input and intermediate values, including the full forward
value, and stages all three gradients before publication. It does not add a
public backward provider or an Eshkol trainable-model API.

`tests/g3c4/test_layernorm_vjp_private.c` uses the genuine G3-C4 forward
provider for central differences of all 16 `dX`, four `dGamma`, and four
`dBeta` entries. It checks source preservation, malformed rank/shape/length,
dtype/device/layout, output overlap, nonpositive/nonfinite epsilon,
nonfinite inputs, finite-input overflow, null views, and unchanged
destinations on rejection. The focused script runs normal, repeat, native
ASan+UBSan, and feature-off isolation. Clang 22 local compatibility and
pinned Clang 21 compiled evidence are separate receipts.

This leaf is not composed into C4 M3T, attention VJP, training, C2 SAVE/LOAD,
or persisted generation. Those remain separately reviewed dependencies.
