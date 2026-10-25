# Private C4 position-embedding VJP leaf

**Bounded implementation target, not a trainable C4 model.** The accepted
G3-C4 provider admits forward position embedding at `[1,4,4,4]` with IDs
`i64[1,4]`, weight `f32[4,4]`, and output `f32[1,4,4]`.
`native/n3k_primitives_provider.c` admits embedding backward for two tokens,
while `native/g3c4_primitives_provider.c` has no backward capability. The
trainable M3T position parameter is `[2,4]`. This leaf supplies one genuine
reverse operation needed by a later four-token M3T schedule; it does not alter
any of those existing providers or their manifests.

Under `ET_M3T_C4_POSITION_VJP_PRIVATE`, the sole direct entry
`et_m3t_c4_position_vjp_private_v1(ids, upstream, gradient)` accepts K1 v1
views with exact dense CPU shapes `i64[1,4]`, `f32[1,4,4]`, `f32[4,4]`,
respectively; exact byte lengths `32/64/64`, offset zero and aligned storage.
IDs must equal `[0,1,2,3]` in order. Upstream elements must all be finite.
The destination must not overlap either input. On success it copies the
upstream row for each position into its corresponding gradient row, without
allocation or floating arithmetic. It returns a private status code. Every
rejection leaves the complete destination byte-identical; feature-off builds
have no entry. Source views and output must stay alive for the synchronous
call. This direct private ABI is not a public K1 provider capability and must
not be used by M3T until a separately reviewed integration contract exists.

The focused gate compares all sixteen gradient entries with central finite
differences of the accepted C4 forward embedding on nontrivial weight and
upstream values, plus exact algebraic copies. Negative tests cover malformed
rank/extent/length/layout/device/dtype/IDs, nonfinite upstream, nulls,
unaligned storage, overlapping spans and output preservation. Compile with
sanitizers and repeat the valid and invalid cases. The ordinary C4 and M3T
feature-off source/exports remain unchanged. Full trainable C4 still requires
four-token gradients for every other differentiable primitive, M3T tensor and
graph shape changes, and fresh compiled end-to-end evidence.
