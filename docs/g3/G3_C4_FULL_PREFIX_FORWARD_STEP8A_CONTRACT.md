# G3-C4 Step 8A fixed full-prefix forward contract

This bounded native leaf consumes the runtime authority admitted by Step 7A to
execute one fixed T=4 C4 numerical forward pass. It adds one source-private C
function under `ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE`; the macro is valid only
with `ET_G3C4_PROVIDER_ROUTES_PRIVATE`. No package or public API exposes it.

The synchronous function accepts an authenticated active generator context,
four already-copied T1 byte token IDs, and caller-owned f32 logits
`[1,4,256]`. It accepts manual-prefill and generate calls, rejects decode calls,
retains no borrowed pointer, and copies logits only after complete success.

The schedule is exactly the accepted 21 roles: token and learned-position
embedding; embedding residual; norm1; Q/K/V projections and head splits;
causal attention over candidate A2 K/V; head merge and attention projection;
attention residual; norm2; FFN up, exact-erf GELU and down; FFN residual; final
norm; and the tied token/head projection. Parameter selection is fixed by the
accepted 14-pin order. Because the provider accepts `(gamma,beta)` while the
model stores `(beta,gamma)`, the norm calls use pins `(7,6)`, `(9,8)`, and
`(12,11)`. Epsilon bits are exactly `0x3727c5ac`.

K/V `[1,2,4,2]` are staged into a one-layer A2 transaction with append count 4.
The function opens the staged full-capacity view, derives the exact `[1,4,4]`
causal keep mask from A2 validity and positions 0..3, dispatches the accepted
G3-C4 attention row `[1,2,2,4,4,2]`, and ends the view before continuing. The
candidate transaction is always aborted, including after numerical success, so
this leaf publishes no cache mutation. Frame ownership and cache commit remain
downstream.

Any validation, allocation, A2, or K1 dispatch failure leaves caller logits and
the committed cache byte-for-byte unchanged. Cleanup preserves the first error.
The focused gate wraps K1 dispatch and proves the exact 21-entry capability,
operation, and row transcript; deterministic repeat; token sensitivity; finite
logits; invalid token and decode rejection; A2 allocation cuts; cache and output
atomicity; and normal plus sanitizer execution.

This leaf does not construct a T1 owner, retain a T1 shell, commit a frame or
cache, select or invoke G3-S, extract last-position logits, sample a token,
publish a result, persist state, register packaging, or claim public generation.
The next gate may add explicit G3-S sampler transport over owned last-position
logits, or attach this numerical kernel to reviewed frame/cache ownership.
