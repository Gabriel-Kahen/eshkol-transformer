# M3 numerical/reference checkpoint

Development-only design checkpoint for issue #85. The initial gap analysis below
predates the accepted binding in issue #1 comment 5744129957; the resulting
contract is recorded in `docs/M3_COMPOSITION_PROPOSAL.md` and `docs/M3_MODEL.md`.
The production schedule is Eshkol-authored. Fixed-profile logits and raw-seed VJP
use existing numerical providers; optional loss and RNG trace are absent.
The independent reference is now frozen in `tests/m3/reference_manifest.json`;
`tests/m3/README.md` describes executable evidence and its limitations. This
checkpoint does not define additional ABI or confer supported-platform status.

## Actual merged numerical boundary

The profile is exactly N=1, T=2, V=256, D=4, Hq=Hkv=2, Dh=2, L=1, F=8,
CPU f32, learned positions, pre-LayerNorm plus final LayerNorm, bias-free
projections, exact-erf GELU, tied token embedding/head, and no dropout/cache.
LayerNorm epsilon is the f32 value with bits `0x3727c5ac`.

The literal role table in `src/eshkol_transformer/m3t_transport.c` supplies 21
forward and 25 reverse calls. Its rows use only N3K fixed numerical/layout/sum
operations, N2 LayerNorm, and A2 attention `[1,2,2,2,2,2]`. Native descriptors
are wiring, not an execution schedule. `native/m3t_transport_extension.esk`
defines `m3t-workspace-contributions-internal` (arity 1), returning a preallocated
14-row canonical handle/carrier vector only after every unique gradient is ready.
`et_m3t_private_workspace_gradient_v1(workspace,index)` independently requires
the active reverse frame and all 14 final slots ready. Neither operation commits
gradients or authorizes a later commit after an earlier primal check.

Weights use `[Dout,Din]`. Token-major hidden values are `[1,2,4]`; split/attention
values are `[1,2,2,2]`; FFN up/activation values are `[1,2,8]`; logits and seed
are `[1,2,256]`. Inputs/positions are exact i64 `[1,2]`. Attention uses fixed
positions `[0,1]`, four true keep bytes, and A2 causal position comparison.

## Complete Eshkol call schedule

After successful workspace begin, execute these forward calls, in this order.
Every name below has the existing `diagnostic-` prefix and terminal `!`.

| Step | Operation | Selector |
|---:|---|---|
| 1 | embedding | token |
| 2 | embedding | position |
| 3 | residual | embeddings |
| 4 | layer-norm | block-input |
| 5 | linear | q |
| 6 | linear | k |
| 7 | linear | v |
| 8 | heads-split | q |
| 9 | heads-split | k |
| 10 | heads-split | v |
| 11 | attention | none |
| 12 | heads-merge | none |
| 13 | linear | attention-output |
| 14 | residual | attention |
| 15 | layer-norm | ffn-input |
| 16 | linear | ffn-up |
| 17 | gelu | none |
| 18 | linear | ffn-down |
| 19 | residual | ffn |
| 20 | layer-norm | final |
| 21 | linear | tied-head |

After owned seed snapshot and successful workspace VJP begin, execute these
reverse calls. VJP names have the existing `diagnostic-` prefix and terminal
`-vjp!`; sum uses the existing `diagnostic-sum!` instead.

| Step | Operation | Selector |
|---:|---|---|
| 1 | linear | tied-head |
| 2 | layer-norm | final |
| 3 | residual | ffn |
| 4 | linear | ffn-down |
| 5 | gelu | none |
| 6 | linear | ffn-up |
| 7 | layer-norm | ffn-input |
| 8 | sum | ffn-residual-input |
| 9 | residual | attention |
| 10 | linear | attention-output |
| 11 | heads-merge | none |
| 12 | attention | none |
| 13 | heads-split | q |
| 14 | heads-split | k |
| 15 | heads-split | v |
| 16 | linear | q |
| 17 | linear | k |
| 18 | linear | v |
| 19 | sum | qkv-input |
| 20 | layer-norm | block-input |
| 21 | sum | attention-residual-input |
| 22 | residual | embeddings |
| 23 | embedding | token |
| 24 | embedding | position |
| 25 | sum | tied-weight |

The four sums are `(dn1q+dn1k)+dn1v`, `dxskip+dxattention`, `drskip+drffn`,
and `ghead+gtoken`, respectively. Residual reverse emits separate edges; no
implicit accumulation is present. In particular the final unique tied gradient
must be submitted once. The mathematical reference must encode the graph
independently rather than read this role table, transport descriptors, or native
expected outputs.

## Parameter and initialization evidence

Canonical P1 ordering has 15 logical paths and 14 unique tensors containing 1,184
f32 values. The tied head has the least path `head/weight`, and
`token_embedding/weight` must resolve to that exact handle.

| Unique index | Canonical path | Shape | Initial block interval |
|---:|---|---|---|
| 0 | blocks/0/attention/key/weight | [4,4] | [0,4) |
| 1 | blocks/0/attention/output/weight | [4,4] | [4,8) |
| 2 | blocks/0/attention/query/weight | [4,4] | [8,12) |
| 3 | blocks/0/attention/value/weight | [4,4] | [12,16) |
| 4 | blocks/0/ffn/down/weight | [4,8] | [16,24) |
| 5 | blocks/0/ffn/up/weight | [8,4] | [24,32) |
| 6 | blocks/0/norm1/bias | [4] | exact +0, no draw |
| 7 | blocks/0/norm1/weight | [4] | exact 1, no draw |
| 8 | blocks/0/norm2/bias | [4] | exact +0, no draw |
| 9 | blocks/0/norm2/weight | [4] | exact 1, no draw |
| 10 | head/weight | [256,4] | [32,288) |
| 11 | norm_final/bias | [4] | exact +0, no draw |
| 12 | norm_final/weight | [4] | exact 1, no draw |
| 13 | position_embedding/weight | [2,4] | [288,290) |

Intervals are offsets from the supplied initializer's counter. Eight random
matrices contain 1,160 values and consume 290 Philox blocks; six affine vectors
contain 24 values and consume none. The already independent integer oracle in
`tests/n3k/test_initializer_reference.py` can generate bits for this explicitly
written path schedule. Its old abstract eight-matrix sequence is differently
ordered and must not be reused as the model's expected schedule.

Check complete initial bytes, constants including zero sign, typed algorithm
`n3k.philox4x32-10.uniform-f32.v1`, original state, and successor state for seeds
0, 1729, and INT64_MAX. Also construct from a previous model's cached successor
and prove offsets start at 290 and finish at 580. Public seed admission is
nonnegative exact signed-i64; native N3K negative-seed tests do not widen it.

## Independent reference and mutation plan

Generate small deterministic test-local records and a canonical manifest with
generator/source/environment identities, checksums, shapes, path order, initial
bits, exact input IDs, seed values, logits, 14 unique VJPs and the two tied edge
VJPs. Store no executable checkpoint or large generated fixture in Git. A frozen
development PyTorch oracle should build the complete graph directly from tensor
equations using one shared differentiable embedding/head tensor and exact-erf
GELU. A separate binary64 mathematical/finite-difference check validates the
oracle; neither oracle may dispatch production providers. Python stays under
tests and never runs from the delivered AOT executable.

Use distinct token pairs `[3,197]` and `[0,255]`, swapped tokens, and repeated
tokens `[7,7]`. Use at least two distinguishable finite dense upstream tensors,
plus first-token-only, second-token-only, and exact-zero upstream. An example
deterministic dense seed is `(((i*29+salt*17)%101)-50)/64` rounded once to f32;
the salt and complete 512 bits belong in the record. Do not use only all-ones
or symmetric tensors. One fixture must exercise nonzero contributions from every
unique parameter path and separately nonzero head/embedding contributions.

Random profile weights are small and q/k gradients may be tiny. Measure errors
per path and per tied edge; a global gradient norm or loose absolute tolerance
can hide an omitted branch. Establish thresholds from independent conditioning
and observed f32-versus-f64 error before freezing them. Add a well-conditioned,
distinguishable parameter fixture through the accepted P1 exact-state load
boundary after proving initial model bits. This is a numerical fixture, not an
initializer change.

For the binary64 graph, central finite-difference the scalar
`sum(logits*upstream)` for every one of the 1,184 unique values, with scaled
step sizes and convergence across two steps. A perturbation of the tied tensor
must affect both uses at once. Also perturb its two uses independently in an
untied reference clone; their analytic gradients sum to the tied result. Keep
mathematical finite differences separate from practical f32 schedule finite
differences, whose representable steps and tolerances must account for rounding.
At least one native/public end-to-end gradient check must connect the actual
Eshkol schedule to the finite-difference/reference record.

Required sensitivity mutations include swapped q/k/v parameter roles; omitted
norm/residual branch; split as identity; swapped token/head axis; lost head,
lost embedding, duplicated head, duplicated embedding, and overwritten tied
sum; missing q/k/v edge in sum3; wrong FFN up/down operand; tanh GELU; wrong
normalization dimension; missing/relocated final norm; missing learned positions;
noncausal attention; and parameter-order/initializer double-draw mutations.
Each mutation must fail a named reference assertion, not merely a structural
lint. Ordered-f32-sum mutations may require separate literal bit witnesses since
ordinary mathematical tolerances need not distinguish association.

Change only the future token and require first-token logits invariant; seed only
the first-token logits and require no embedding input-path gradient into the
future token row. The tied head still legitimately contributes to every
vocabulary row: causal assertions must inspect the separate embedding edge,
not incorrectly demand a zero final tied gradient for that row.

## Preacceptance loss, seed, and accumulation gap analysis

A raw logits VJP needs no L2 operation: its caller supplies an already-weighted
finite f32 `[1,2,256]` numerator seed and a separate positive finite f32
normalization increment. I2 stores the 14 resulting unnormalized numerator
tensors and that shared metadata. One successful call adds one to every count,
including exact-zero VJPs, which must become present-zero. Dividing the seed by
the increment and also supplying the increment would normalize twice.

If A0 optional loss is implemented in this milestone, current M3T is insufficient:
L2 is not linked or routed, targets/weighted masks have no accepted diagnostic
ingress, and no owned scalar/reduction or loss-to-logits seed bridge exists.
L2 ABI 1.0 already supplies per-token forward and backward for `[1,2,256]`.
The narrow new numerical seam would be the reviewed fixed `[1,2]` mask reduction
and upstream materialization, with whole-output preflight and no native model
schedule. The decision must state whether mask values must be nonnegative and
finite: A0 states bool-or-f32 and the formula, but does not explicitly settle
all weighted-mask value admission details.

For accepted masked loss, report `L=sum(mask*CE)/W` with f32 `W=sum(mask)>0`.
For a numerator contribution seed use `mask`, not `mask/W`, through L2 backward;
store `W` separately in I2. A mathematical VJP of the reported mean scalar uses
`mask/W` instead. The public operation must distinguish these intentions and
never silently reinterpret the scalar-loss derivative as an unnormalized
numerator. Targets, masks, IDs, seed metadata and normalization weights do not
receive gradients.

A0 has no accepted generic backward name, seed owner, graph-release operation,
or M3 immutable output implementation. Existing diagnostic logits receivers are
mutable and cannot stand in for its immutable retained graph. Exact names,
arities, owned output versus copied-graph strategy, repeat-contribution policy,
release and plan reclamation must be approved on issue #1 before code freeze.
No JIT, arbitrary shapes, training loop, optimizer, checkpoint, accelerator,
full-training aggregate, or dual-registry linkage follows from this design.
