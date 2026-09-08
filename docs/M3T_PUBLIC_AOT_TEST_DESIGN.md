# M3T public AOT transport witness design

Design only; no production or test implementation. Based on proposal #1 comment
5587828007 and `docs/M3T_TRANSPORT_PROPOSAL.md` at the current worktree.

## Scope decision

Keep the proposed API and readiness checks unchanged. A successful public logits
copy requires all 21 forward calls because the only public readout is `z`. VJP begin
also requires `z`; reaching A2 VJP requires a further 12-call reverse prefix.
There is no shorter successful public numerical readout/VJP witness under this
surface. Forward-only prefixes can test admission and failure handling but cannot
export intermediate numerical values.

Suggested clarifying scope sentence for integration:

> M3T's test callers may execute literal public transport call sequences, including
> the full forward order required to admit VJP, solely as transport/readiness
> witnesses. M3T adds no production function executing the complete model schedule
> and makes no full-model forward/gradient parity or model-training claim.

This needs no new API or ABI. Do not introduce partial-frame tensor injection,
arbitrary ready-bit setters, primitive-specific bypasses, or a production self-test
operation merely to shorten this gate. If integration forbids even a test caller's
full forward sequence, the current requirement for positive public VJP execution
is incompatible with the proposed readiness contract and must be explicitly
revised; it cannot be fulfilled by an alleged short fragment.

## Production-only public caller closure

The AOT test source requires `transformer.config`, `transformer.module`,
`transformer.tokenizer`, `transformer.error_public`, and
`transformer.diagnostic_transport` from the installed facade root. It links only
the completed M3T aggregate and normal pinned Eshkol runtime. No trusted Eshkol
root, private include path, test C transport, private symbol extern or native
fixture carrier is available to this caller.

Create once outside reclaimed iteration regions: seed-1729 initializer, exact
resolved profile config, genuine P1 model, input, workspace, logits receiver, and
independent upstream logits receiver. Input is [65,66]. Use a finite nonzero
upstream byte fixture (e.g. alternating exact +1/-0.5) through explicit public
bit ingress; initialize it once. T1 can additionally encode `AB` once and copy it
through the public tokenizer adapter. Scalar fixture-byte construction is test
setup, not a production numerical operation or steady-state claim.

## Literal forward witness: 21 calls

After `(diagnostic-workspace-begin! workspace input)`:

```scheme
(diagnostic-embedding! workspace 'token)
(diagnostic-embedding! workspace 'position)
(diagnostic-residual! workspace 'embeddings)
(diagnostic-layer-norm! workspace 'block-input)
(diagnostic-linear! workspace 'q)
(diagnostic-linear! workspace 'k)
(diagnostic-linear! workspace 'v)
(diagnostic-heads-split! workspace 'q)
(diagnostic-heads-split! workspace 'k)
(diagnostic-heads-split! workspace 'v)
(diagnostic-attention! workspace)
(diagnostic-heads-merge! workspace)
(diagnostic-linear! workspace 'attention-output)
(diagnostic-residual! workspace 'attention)
(diagnostic-layer-norm! workspace 'ffn-input)
(diagnostic-linear! workspace 'ffn-up)
(diagnostic-gelu! workspace)
(diagnostic-linear! workspace 'ffn-down)
(diagnostic-residual! workspace 'ffn)
(diagnostic-layer-norm! workspace 'final)
(diagnostic-linear! workspace 'tied-head)
(diagnostic-workspace-copy-logits! workspace logits)
```

The copy is a lifecycle operation, not one of the 21 primitive calls. This executes
N3K primitive rows, N2 LayerNorm, and A2 attention using owned public receivers and
the reviewed fixed dispatch path. Assert each call returns #t. Outside memory
measurement, inspect the detached 2048-byte result for finite values, a nonzero
result, and deterministic equality on repeated reset/begin runs with equal input.
These checks establish transport behavior, not an independent model-value oracle.

## Minimal three-provider VJP witness: 12 calls

After the forward witness and
`(diagnostic-workspace-vjp-begin! workspace upstream)`:

```scheme
(diagnostic-linear-vjp! workspace 'tied-head)
(diagnostic-layer-norm-vjp! workspace 'final)
(diagnostic-residual-vjp! workspace 'ffn)
(diagnostic-linear-vjp! workspace 'ffn-down)
(diagnostic-gelu-vjp! workspace)
(diagnostic-linear-vjp! workspace 'ffn-up)
(diagnostic-layer-norm-vjp! workspace 'ffn-input)
(diagnostic-sum! workspace 'ffn-residual-input)
(diagnostic-residual-vjp! workspace 'attention)
(diagnostic-linear-vjp! workspace 'attention-output)
(diagnostic-heads-merge-vjp! workspace)
(diagnostic-attention-vjp! workspace)
```

This is sufficient for the focused positive public N3K/N2/A2 VJP transport gate.
Reset can discard the partial reverse frame. It does not accumulate gradients or
pretend to complete reverse-mode model acceptance.

To exercise every public reverse selector, append these 13 calls in a distinct
bounded selector-coverage case (25 reverse calls total):

```scheme
(diagnostic-heads-split-vjp! workspace 'q)
(diagnostic-heads-split-vjp! workspace 'k)
(diagnostic-heads-split-vjp! workspace 'v)
(diagnostic-linear-vjp! workspace 'q)
(diagnostic-linear-vjp! workspace 'k)
(diagnostic-linear-vjp! workspace 'v)
(diagnostic-sum! workspace 'qkv-input)
(diagnostic-layer-norm-vjp! workspace 'block-input)
(diagnostic-sum! workspace 'attention-residual-input)
(diagnostic-residual-vjp! workspace 'embeddings)
(diagnostic-embedding-vjp! workspace 'token)
(diagnostic-embedding-vjp! workspace 'position)
(diagnostic-sum! workspace 'tied-weight)
```

All sums are reverse-phase N3K forward sum operations; there is no extra
embedding-input sum or public sum VJP operation. No production helper executes
this sequence. The test's public API has no gradient-value accessor: numerical
VJP equality must be established by the focused internal role-wiring tests against
merged provider/reference evidence, not claimed from #t/readiness alone.

## Minimal test matrix

| Case | Public sequence/observation | Claim established |
|---|---|---|
| Owned construction | Initialize, inspect canonical P1 paths, 14 handles, one exact head/token tie, cached state words [1,1729,290,0] | Real shared P1/I2 ownership and seeded schedule identity |
| Continuation | Construct a second model with the cached successor; expect counter 580; repeat seed/state construction for equal parameter snapshot bytes using accepted detached state inspection paths | Typed continuation, no mutation of original state; finite construction count only |
| Exact input | List and once-encoded T1 `AB` routes produce equal logits after separate frames | Explicit exact rank-one to rank-two copy reaches numerical path |
| Forward witness | 21 primitives + owned logits copy | Public N3K/N2/A2 composition transport |
| VJP witness | 12-call reverse prefix after full forward | Positive public VJP transport for all three providers |
| Selector coverage | Append remaining 13 reverse calls | Every accepted reverse selector and all four sum routes admitted |
| Missing readiness | Before each producer, invoke one immediate dependent and expect invalid-state; then execute correct producer and continue | Dependency admission and failure does not poison retry |
| Duplicate write | Repeat a successfully completed forward/VJP selector; expect invalid-state | Single producer rule |
| Phase | VJP begin before z, second VJP begin, forward call in reverse, reverse call in forward, sum in forward | Exact phase contract |
| Identity/type | Wrong opaque kind, copied metadata, selector wrong type/unknown symbol; improper/cyclic input list and inexact/complex IDs | Tagged-value admission occurs before unsafe native conversion |
| Independent copies | Copy z, reset/reuse workspace with changed IDs, retain original logits bytes; mutate upstream after VJP begin and verify internal saved upstream in separate native role test | Logits and upstream snapshots have independent owners |
| Stale primals | Save P1 state once, mutate/load different exact parameter values after begin, check-primals rejects; restore exact original state, check passes | Current-value comparison policy, no epoch claim |
| Active ownership | Begin another workspace bound to same model; expect invalid-state; reset/release first then retry | One active frame per model |
| Release/reset | Reset forward and partial reverse; reset idle; double release; every later receiver operation rejects | Deterministic permanent invalidation |
| Failure atomicity | Inject each native constructor/guard/staging failure in isolated native/test artifact; public negative calls retain copied logits and permit retry | Private failpoints stay outside canonical public artifact |
| Repeated public AOT | Two fresh builds with source/aggregate-only closure; byte-identical binaries and stdout; swapped E1 import order | Actual pinned public AOT provenance and one error registry |

Some P1 state bytes have no public generic tensor-byte accessor. Where a matrix
asks for physical state or saved-gradient bytes, use the existing accepted native
I2/P1 inspection test boundary in a separate focused test; do not add a generic
public observer or pretend that an opaque-state identity proves byte equality.

## Memory lane

Run only success calls in reclaimed Eshkol regions. Create owners/input/upstream
outside those regions. For each iteration: begin, literal 21-call forward witness,
copy into the same logits receiver, optional 12-call VJP prefix, reset. Discard
all transient wrapper boxes inside the region; retain no new identity or detached
snapshot. Check output bytes and owner addresses between measured batches, not via
allocating snapshot accessors every iteration. Compare increasing post-warmup
iteration counts using current RSS and native payload/guard/retired-shell counts.

The mandatory public copy step means this lane is a full forward *test witness*;
calling it a short fragment would be inaccurate. A separate genuinely short
prefix/reset loop may diagnose transport allocation, but cannot replace the
copy/output-lifetime lane. Exclude E1 error creation, T1 encoding, fresh owners,
P1 snapshots/load/zero-grad plans, and future gradient plans from flat-memory
claims, exactly as the proposal specifies.
