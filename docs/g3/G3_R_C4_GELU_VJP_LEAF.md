# Private C4 GELU VJP leaf

This leaf follows the accepted private C4 position and residual VJPs. The
accepted C4 provider's four-token `g3c4.gelu.forward` row accepts and emits
`f32[1,4,8]`. Its scalar forward formula evaluates
`0.5f * x * (1.0f + erff(x * 0x1.6a09e6p-1f))` with finite checks.

Under `ET_M3T_C4_GELU_VJP_PRIVATE`,
`et_m3t_c4_gelu_vjp_private_v1(input, upstream, gradient)` takes three
synchronous borrowed K1 v1 views. Every view must be dense row-major CPU
`f32[1,4,8]`, offset zero, aligned, and exactly 128 bytes. The destination
must not overlap either source. Input and upstream values must be finite, and
the analytic gradient product must remain finite. The scalar derivative is
`0.5*(1+erf(x/sqrt(2))) + x*exp(-x*x/2)/sqrt(2*pi)`; the implementation uses
the provider's exact binary32 `1/sqrt(2)` constant and a binary32
`1/sqrt(pi)` constant. It computes into a 32-element stack staging buffer and
publishes only after all elements pass, so any rejection leaves the complete
destination unchanged. No heap allocation, retained owner, or public provider
capability is added. Feature-off builds have no entry.

The focused test compares every one of the 32 gradient entries with central
finite differences through the genuine C4 forward provider using varied
negative and positive inputs and nonzero upstream values. It also checks
source preservation, malformed rank/shape/length/dtype/device/layout,
misalignment, null views, nonfinite values, gradient overflow, exact and
partial aliases, and unchanged destination on every rejection. This is one
private backward primitive; it does not wire a trainable C4 M3T graph or
establish a trained checkpoint, SAVE/LOAD, or generation.
