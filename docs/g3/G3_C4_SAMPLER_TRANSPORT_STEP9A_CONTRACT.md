# G3-C4 Step 9A private sampler transport contract

This bounded native leaf consumes the actual G3-S runtime admitted by Step 7A
after the fixed full-prefix numerical boundary from Step 8A. It adds one
source-private synchronous function under `ET_G3C4_SAMPLER_TRANSPORT_PRIVATE`,
which requires `ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE`. No package or public API
exposes the function.

The function authenticates an active generate call with budget one, copies the
last `[1,256]` row from caller-owned finite full-prefix logits into private stack
storage, and materializes the generator's stored policy and numeric RNG into the
exact accepted G3-S views. Greedy dispatch uses two inputs and returns an
unchanged successor RNG. Categorical dispatch uses five inputs, with temperature
and top-p materialized by exact bit copy and top-k as an i64 scalar, and returns
the one-block successor. Both use two disjoint local outputs and deterministic
request row `[1,256]`.

The generator's RNG is copied, never passed as writable output, and never
advanced by this function. The cache, full logits, policy, active-call tuple and
any result owner remain unchanged. Only after complete K1/G3-S success and
candidate invariant checks does the function copy the token and successor RNG to
disjoint caller outputs. Output aliases with logits, each other, or the generator
context reject before dispatch. A failed dispatch, nonfinite last row, exhausted
categorical state, invalid call kind, or alias leaves all candidate outputs and
generator state unchanged.

This numerical transport does not own or authenticate a frame/result envelope
and cannot establish that prefill was committed; downstream reviewed frame
state must enforce that ordering before calling it. It does not publish the
candidate token or successor RNG, mutate or commit cache, begin a token forward,
apply EOS, extract text, persist state, register packaging, or claim public
generation. The next gate is reviewed candidate ownership and joint
token/cache/RNG commit, or a frame boundary that supplies those authorities.
