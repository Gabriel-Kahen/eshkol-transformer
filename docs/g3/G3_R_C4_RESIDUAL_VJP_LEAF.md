# Private C4 residual VJP leaf

This is one upstream primitive for the proposed trained context-four model,
not a trainable C4 graph or generation checkpoint. The accepted C4 forward
provider's `g3c4.residual.forward` adds two `f32[1,4,4]` tensors into a
disjoint `f32[1,4,4]` result. The four-token full-prefix path uses it for
token plus position embeddings and later residual branches.

Under `ET_M3T_C4_RESIDUAL_VJP_PRIVATE`, the direct private function
`et_m3t_c4_residual_vjp_private_v1(left, right, upstream, left_gradient,
right_gradient)` takes five synchronous borrowed K1 v1 views. Every view must
be dense, row-major, CPU, aligned `f32[1,4,4]` with offset zero and exactly
64 bytes. Source views may overlap each other. Both gradient destinations
must be disjoint from every source and each other. Inputs, upstream and their
forward sums must be finite. On success each gradient is an exact bitwise
copy of the upstream tensor. Every rejection leaves both 64-byte destinations
unchanged; the call allocates and retains nothing. Feature-off builds have no
entry, and the ordinary C4 provider/export manifest is untouched.

The focused test compares all 32 gradient entries against central finite
differences of the actual C4 residual forward dispatch, using nontrivial
dyadic operands and upstream weights. Negative tests cover malformed rank,
shape, length, dtype, device, layout, offsets, alignment, invalid values,
overflowing forward sums, null views, exact and partial aliasing, and complete
preservation of both destinations. This leaf is not yet wired into M3T. The
other C4 forward operations and trainable shape schedule still need their
own backward and integration gates before a trained C4 C2 image can exist.
