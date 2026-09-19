# M3 fixed-profile composition proposal

Status: **proposed, not accepted or frozen**. Decision checkpoint for
[issue #85](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/85) and
[the integration ledger](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1).
Base: main `21b1df13940e94adce998ab88cb45c329c956b58`; M3T implementation
`f50160968bd97677632f3539fd2e278a63624cf9` is accepted. The main update is
acceptance prose only. This document changes no runtime, ABI, manifest or status.

## Decision requested

Accept a bounded A0-compatible logits/explicit-VJP implementation with eight
additional public wrappers, owned immutable graph snapshots, Eshkol forward/reverse
replay, and one same-call I2 numerator contribution. Use ordinary I2 plans with
explicit cumulative control-shell accounting. Leave optional loss/RNG ingress,
flat training memory and full-training aggregation downstream. These are concrete
scope decisions, not a request to accept unimplemented numerical or lifetime claims.

The exact profile remains `eshkol-diagnostic-byte-decoder-v1`: N1/T2/V256/D4,
Hq=Hkv=2, Dh2/L1/F8, learned positions, pre-LayerNorm and final LayerNorm,
bias-free projections, exact-erf GELU, tied embedding/head, CPU f32, no dropout
or cache. Epsilon bits remain `0x3727c5ac`. Construction/configuration/typed
initializer stay at the accepted M3T entry points, including 14 unique tensors,
15 logical paths, 1,184 values and 290 Philox blocks per model. Graphs retain the
exact model/profile, original and successor initializer identities; initializer
metadata is not stochastic forward RNG authority.

## Exact public candidate

Add `transformer.model` with these eight bindings. All existing M3T names retain
their contracts. No new public native header or serialized format is proposed.

| Binding | Arity | Contract |
|---|---:|---|
| `model-forward` | 2 + flat opts | `(model input-ids . opts)` returns a newly owned opaque output/analytic graph. Input is the genuine same-aggregate M3T CPU i64[1,2] owner, synchronously borrowed and copied into graph-owned storage. |
| `model-output-logits` | 1 | Output to a **new**, independently owned read-only f32[1,2,256] tensor with its own graph lease. It is neither a bytevector nor a mutable diagnostic logits receiver. |
| `model-output-loss` | 1 | Live output to `#f` in this no-target subset. |
| `model-output-rng` | 1 | Live output to `#f` in this no-RNG subset. |
| `model-output-release!` | 1 | Idempotently release the exact output's graph lease; sibling logits copies remain valid. |
| `diagnostic-output-vjp!` | 4 | `(result seed normalization-weight-bits expected-ordinal)` executes analytic VJP and one whole-model contribution; returns `#t`. Result is the closed union of live output or graph-retaining model-logits tensor. |
| `diagnostic-model-logits-bits` | 1 | Model-logits tensor to a detached 2,048-byte little-endian exact-f32 copy. |
| `diagnostic-model-logits-release!` | 1 | Idempotently release that exact logits tensor and its graph lease. |

The VJP seed is an existing genuine mutable diagnostic logits owner. Snapshot its
512 finite f32 values before reverse execution; do not retain or mutate it.
Weight is an exact integer in `[0,2^32)`, interpreted as positive finite binary32
bits, without a floating cast. Ordinal is an exact nonnegative signed-i64 integer
equal to every current unique destination's contribution count. No argument names
arbitrary parameters, tensor shapes, providers, callbacks or native pointers.

`model-forward` keeps A0's four recognized option names and flat-option syntax.
First reject malformed/cyclic option lists, unknown/duplicate keys, missing values,
unpaired `:targets`/`:loss-mask`, and nonboolean `:deterministic?` as
`invalid-argument`. Then reject presence of `:targets`/`:loss-mask` or `:rng` as
`unsupported`, before dereferencing their unavailable carriers or allocating a
graph. Thus even `:rng #f` is outside the admitted option subset; omit it. This
is explicit feature rejection, not validation or acceptance of an arbitrary RNG
or mask identity. Both boolean deterministic values are admitted; execution still
uses the same reviewed deterministic path. Train and eval are admitted and captured.
All other shapes, dtypes, devices and cross-aggregate input carriers reject.

This implements the shape/ownership semantics of four A0 operations for a stated
subset; it does not claim first-release A0 coverage or compiler autodiff. The
existing explicit M3T integer/tokenizer adapters remain the only input ingress.
If integration elects to leave A0 names declaration-only, rename the first four
bindings to `diagnostic-forward`, `diagnostic-output-logits`,
`diagnostic-output-loss`, `diagnostic-output-rng`, and rename output release to
`diagnostic-output-release!`. Counts and ownership work are unchanged. The
recommended decision is the A0-name subset above.

## Owned graph and reusable workspace

Use one private reusable M3T workspace per model, created once outside transient
frame regions, with the existing one-active-frame exclusion shared with public
diagnostic workspaces. Forward executes the 21 existing fixed roles in Eshkol,
captures an immutable graph, and resets its workspace before publishing output.
It does not keep a workspace frame active merely because an output survives.
Separate outputs can coexist; each denotes its own exact primals/input/logits.

A graph owns 14 independent saved parameter tensors (4,736 bytes), exact I1
input IDs (16 bytes), and logits (2,048 bytes): **6,800 tensor payload bytes**.
It also records the exact fixed epsilon/positions/keep values (24 semantic bytes),
captured mode and model/profile/initializer/topology identity. Native headers,
shape/stride arrays and Eshkol identities are separate accounting, not included in
that payload figure. No graph payload aliases live parameters or reusable workspace.

VJP restores that graph's saved primals/IDs/constants into an idle workspace, then
executes the same 21-role Eshkol forward schedule followed by the 25-role analytic
reverse schedule. The only replay is deterministic Eshkol composition over reviewed
kernels; C owns fixed-shape copying, identity/lifetime and the contribution boundary.
Every VJP checks the live model against captured primals; mutation followed by exact
restoration may pass, as in M3T. Replay never silently substitutes current parameters.

`model-output-logits` clones the graph's original logits and adds one graph lease.
Releasing output preserves those copies, their bits and their ability to invoke VJP.
Releasing one copy affects no sibling. Final lease release frees graph tensor
payloads exactly once. All release calls authenticate exact kind/owner/liveness,
preflight active work/borrows/plans before invalidating authority, then perform a
nonallocating teardown. Registered dead receivers permit only idempotent release;
other use is `invalid-state`. Forged/copied/foreign/wrong-kind receivers are
`invalid-argument`. Expired Eshkol region storage has no stale-alias guarantee.

Creation stages all native allocations, Eshkol identities and ownership ledgers
before publication. Any failure leaves existing outputs/model/inputs unchanged,
reclaims unpublished tensor payloads, and resets private scratch. Failures never
publish a partial graph or retain an active model frame. The output and logits
shells must use proved owner-lifetime allocation and canonical promotion/readback;
raw captured-registry assignment is not a lifetime proof.

## Atomic contribution and normalization

Eshkol executes all role scheduling and the four exact ordered sums: Q+K then V,
skip+branch for each residual, and head+embedding for the tied tensor. The existing
private `m3t-workspace-contributions-internal/1` yields all 14 canonical
handle/carrier rows, with the tied result once. It grants no delayed commit token.

Inside the same serialized, nonreentrant public VJP call, validate the graph/model
association, immutable profile/topology, canonical P1 identities and current mode.
The final native contribution call repeats exact saved-versus-live checks for all
14 parameters/constants and the graph/restored-frame identity, validates complete
VJP readiness, and builds one fixed 14-entry I2 batch. Prepare exactly one
`et_f32_gradient_plan`, commit it once, release it on every path, then reset scratch.
No callback, public reentry or external mutation interval separates these checks
from prepare/commit. The model values never change.

The seed represents an already-weighted objective **numerator**. Do not multiply
by weight, divide by weight, or normalize gradients. Each success adds each unique
VJP once, adds the same supplied weight once, and increments each count once.
Zero seed turns absent gradients into present-zero with positive count and weight;
when numerators already exist it adds zero while still advancing metadata.
Repeated VJP is permitted only with the then-current expected ordinal; retrying
the old ordinal after success rejects. Only explicit `module-zero-grad!` resets
to absent/count-zero/weight-positive-zero. I2's finite, fenv, independence,
metadata agreement and overflow admission remain unchanged.

Every recoverable failure preserves **all** gradient bytes, states, counts and
weight bits. Plan preparation owns every fallible allocation/computation. Validate
the cleanup participants before commit; prove release and private frame reset form
an infallible nonallocating tail for these exact private owners after commit. A
committed result must never be returned as a retryable unchanged-gradient error.
Fault injection covers late-entry failure, every allocation, borrow/plan pin,
overflow, stale values/mode, cleanup eligibility, and retry after exclusions clear.

## Exact proposed private seams and package counts

Ten localized native additions, source-private contract version 1.0 and suffix
`v1`. Nine Eshkol-callable seams use i64 status and pointer-on-create conventions
matching M3T; the native-only I1 leaf returns i32 status and writes the existing
I1 error record. Prefix every name below with `et_m3_private_` and append `_v1`:

| Suffix | Ordered arguments | Arity |
|---|---|---:|
| `graph_capture` | workspace | 1 |
| `graph_restore` | workspace, graph | 2 |
| `graph_check_primals` | graph, current-mode | 2 |
| `graph_release` | graph's output lease | 1 |
| `model_logits_create` | graph | 1 |
| `model_logits_copy_to` | model-logits, byte span, exact count 2048 | 3 |
| `model_logits_release` | model-logits | 1 |
| `workspace_contribute` | workspace, graph, current-mode, weight-bits, ordinal | 5 |
| `workspace_reset` | exact M3-managed workspace | 1 |
| `i64_unborrowed` | registered I1 tensor, `et_i64_tensor_error *` | 2 |

Graph output-lease release is separately marked so repeat release cannot decrement
a surviving logits lease. Native logits owners retain their graph; trusted Eshkol
records authenticate the corresponding graph pointer for either VJP receiver kind.
No public pointer/graph accessor or generic revoker is added. Reuse M3T's three
localized error accessors and fixed domain/category/code mapping; snapshot errors
before cleanup. Internal Eshkol handlers are `m3-public-` plus each public name.
Forward's public facade is variadic with minimum arity 2; it passes the untouched
options list to a fixed arity-3 internal handler `(model input opts)`. Its boxed
C wrapper has four pointer arguments `(model input opts result-slot)`, not native
varargs. The other seven handlers preserve their public fixed arities, and their
boxed C wrappers add one result-slot pointer.

The M3-only source composition adds an exact restored-graph association and
owned-call guard to workspace control. Reset/release clear that association and
guard on every admitted exit. Internal preflight helpers distinguish their owning
transaction from public reentry; do not simply set M3T's busy flag and then call
an entry point that correctly rejects busy workspaces. This grants no generic
guard bypass and changes no standalone M3T public behavior.

Existing M3T reset preflight opens two **allocating I1 leases**, so it is not a
valid post-commit or allocation-failure cleanup tail. The M3-only reset admits
only its exact private model workspace and ended operation scopes, then uses
existing nonallocating I2 scoped guards and the new read-only I1 leaf to preflight
all storage before clearing active/phase/readiness/graph fields. The leaf signature
is `int32_t et_m3_private_i64_unborrowed_v1(const et_i64_tensor *, et_i64_tensor_error *)`;
it uses I1's existing error-output admission, scans `live_tensors` by pointer
equality before any candidate dereference, then validates the matched record's
magic and borrow state. Existing I1 `require_tensor` alone is insufficient because
it reads magic before registry membership. The leaf rejects an active borrow and
performs no allocation or owner
mutation. Compile it through a private source-including I1 translation unit in
the exact M3 tuple, replacing the ordinary I1 object once; standalone I1 is
unchanged and exports no new symbol. No arbitrary provider/tensor access is public.

`workspace_contribute/5` performs complete cleanup eligibility before prepare.
After commit, release removes I2 plan pins and a shared static helper performs
the fixed field reset without calling ordinary M3T reset or repeating fallible
preflight. The Eshkol coordinator follows only with preallocated stores. Guards
on forward/capture/replay failures invoke the new `workspace_reset/1` after every
owned synchronous borrow has ended. Require allocation-disabled rollback and
post-commit tail tests, active-borrow/pin rejection with unchanged frame state,
and retry after exclusions clear. Failure cannot strand a frame or disguise a
committed contribution as an unchanged-gradient error.

Create one M3 successor sibling, `libeshkol_transformer_m3.a`, one registry-owning
member `m3_package.o`, seven installed facades (the M3T six plus model), exactly
**93 globals / 87 exports / 93 public-name strings**: current 85/79 plus eight.
All new implementation/native seams are localized. Existing P1 18-public /
46-private /64-total and native identity 36-private counts stay unchanged.
Use an exact canonical source/native tuple; do not link the M3T archive beside
another registry owner. D2/O2/K2/C2 imports and interoperability remain excluded.
Standalone K1/I1/I2/N2/N3K/A2 public ABIs/provider identities stay unchanged; no L2
object or capability is added in this proposal.

## Release and measured-memory boundary

Ordinary I2 plans are sufficient for atomicity but do not have flat lifetime cost.
On the reviewed x86-64 layout, a successful gradient-plan prepare/release leaves
one 48-byte control tombstone; gradient reset leaves one 40-byte control tombstone.
Tensor release leaves an 88-byte I2 tensor control tombstone. These are source-layout
figures confirmed by local compile-only static assertions, separate from allocator overhead and Eshkol
registries. Graph creation therefore retires 15 such I2 controls (1,320 bytes) on
final release; each logits clone retires another 88 bytes. Published M3 graph/logits
and Eshkol identities have additional cumulative costs to measure exactly.

The recommended first implementation retains ordinary I2 semantics and reports
these trajectories honestly. It makes **no flat repeated-output/VJP/training-memory
claim**, even if all live tensor payloads return to baseline. Process exit is the
complete reclamation boundary for accumulated identity controls and live models.
Do not describe process-lifetime accumulation as bounded workspace storage.
If integration requires a flat contribution loop in M3, a separate accepted private
unpublished-plan reuse/reclamation seam is necessary before implementing that claim.

Measure forward/capture/release, surviving logits, repeated VJP, VJP+explicit reset,
and failure trajectories at 10/100/1000 plus compiled AOT 1024/8192 iterations:
native live/payload/retired counts, exact Eshkol retained-arena counters and separate
RSS. Use lexical regions for scratch; report deliberate retained graph/shell costs.
Keep the accepted M3T reusable-frame witness distinct from these new trajectories.

## Numerical evidence, downstream seams and handoff

The independent [reference design](m3/NUMERICAL_REFERENCE_DESIGN.md) specifies the
complete 21/25 schedules, all 1,184 parameter finite differences, multiple
initializer seeds/input/seed tensors, conditioned fixtures, per-path sensitivity,
causal tests and omitted/miswired/duplicate tied-edge mutations. AOT must call the
actual public model schedule. Private test-only gradient observation supplies bytes
without introducing an installed gradient-extraction API or Python runtime.

L2 already supplies carrier-neutral per-token CE/direct backward. Optional A0
loss additionally needs accepted target/mask ingress, fixed masked f32 reduction,
scalar ownership, and distinct mean-loss versus numerator-seed semantics. No such
seam is inferred here. Stochastic RNG, arbitrary shapes, trainer/optimizer,
checkpoint/generation and full-training aggregate decisions stay downstream.

Required implementation evidence is two fresh installed-facade AOT builds,
source/symbol/arity/hostile-import/duplicate-owner checks, numerical parity and
gradient checks, complete ownership/atomicity negatives and sanitizers, exact memory
trajectories, affected predecessor gates and one exact-head supported
Ubuntu22/LLVM21.1.8 full CI through the accepted engine. Separate M3-R, integration
merge/retest and acceptance remain mandatory. This proposal is not supported CI,
JIT, full-model execution or first-release evidence; M3 remains planned pending
the binding contract decision.

Independent checkpoint subreviews: [numerical/reference](m3/NUMERICAL_REFERENCE_DESIGN.md)
found no schedule/seed blocker; [ownership/atomicity](m3/ownership-design.md)
accepted the snapshot/replay and ordinary-plan design subject to implementation
proof of cleanup; [packaging](m3/packaging-design.md) found no boundary blocker
after fixing forward's exact boxed arity and selecting the ten-seam nonallocating
cleanup boundary. Local checks passed: 9 existing package
contract tests, 19 CI selector/topology tests, the 17-suite/24-command structural
checker, and `git diff --check`. Ownership compile-only assertions used GCC
16.1.1 and confirmed 88/48/40-byte controls. None are runtime M3 or supported-lane
acceptance evidence. The independent project M3-R review remains separate.
