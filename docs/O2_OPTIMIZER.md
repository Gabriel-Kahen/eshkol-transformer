# O2 optimizer logical contract (proposed)

## Status and dependency boundary

This is a pre-implementation proposal for O2. It freezes no native ABI, provider
identifier, archive topology, or serialized bytes. The binding dependency chain is
P1L (issue #51) to I2 (issue #49) to O2. A production optimizer and review-ready O2
pull request remain blocked until both prerequisites are independently approved and
merged in order.

The accepted O2 public optimizer surface preserves the five existing A0 names and
adds one exact arity-1 lifecycle operation when the post-P1L/I2 runtime is
implemented:

```scheme
(optimizer-create config parameter-tree)
(optimizer-step! optimizer)
(optimizer-zero-grad! optimizer)
(optimizer-state optimizer)
(optimizer-load-state! optimizer state)
(optimizer-state-release! state)
```

Integration accepted `optimizer-state-release!` with conditions in
[issue #1 comment 5493412398](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5493412398).
This accepts the logical public name, arity, ownership, and error contract only. It
does not freeze a runtime/private ABI or authorize implementation before P1L and I2
merge.

O2 does not change X1 schema 1.0. The `config` argument is the O2-specific data-only
logical value below. I2 owns the physical dense CPU f32 carrier, stable P1-handle
value/gradient identities, accumulated-contribution metadata, detached tensor
cloning, and whole-batch mutation transaction. O2 does not reinterpret or freeze
that boundary.

## Binary32 scalar spelling

Every floating configuration scalar is an eight-character lowercase hexadecimal
string containing one IEEE 754 binary32 bit pattern in big-endian display order.
This is a logical spelling, not a file byte order. NaNs, infinities, and negative
zero are rejected. Values are not parsed through locale-sensitive decimal text and
are never silently narrowed from an Eshkol runtime number.

The admitted scalar domains are:

- learning rate: finite and nonnegative;
- beta1 and beta2: finite and in `[0,1)`;
- epsilon: finite and positive;
- weight decay: finite and nonnegative;
- clipping maximum: finite and positive;
- linear-schedule minimum ratio: finite and in `[0,1]`.

## Optimizer configuration 1.0

The borrowed input is one bounded, proper, acyclic data-only list with exact grammar:

```text
(transformer-optimizer-config 1 0 ()
  adamw f32 cpu
  <clip>
  <schedule>
  (<group> ...))

<clip>     := (none) | (global-l2 <binary32>)
<schedule> := (constant)
            | (linear <warmup-updates> <total-updates> <minimum-ratio>)
<group>    := (group
                (<canonical-parameter-path> ...)
                <learning-rate> <beta1> <beta2> <epsilon> <weight-decay>)
```

Version fields and schedule counters are exact nonnegative signed-i64 values.
Version 1.0 requires an empty feature list. A linear schedule requires
`total-updates > 0` and `warmup-updates < total-updates`.

Raw validation is iterative and stops before traversing beyond depth 16 or 65,536
total data nodes. Every list spine is bounded by that same node budget. A config has
at most 4,096 groups, 4,096 total canonical paths, 4,096 paths in any group, and the
P1 bounds of 64 segments per path and 1..65,536 UTF-8 bytes per segment. The complete
config has an additional 16,777,216-byte aggregate UTF-8 budget. These are
pre-provider traversal bounds; an over-bound, improper, cyclic, malformed,
unordered, or range-invalid config is `invalid-argument` before P1 lookup or
allocation.

There are 1..4096 groups. Every group is nonempty. Paths within a group are in P1
UTF-8 byte order, and groups are ordered by their first path. Each path must be the
canonical path of one unique handle in the supplied P1 tree. Every unique handle
appears exactly once. Alias paths, duplicate handles or paths, missing handles,
unknown paths, empty groups, malformed ordering, and a foreign or stale parameter
tree are rejected before optimizer construction or I2 mutation.

`optimizer-create` validates the complete raw value, P1 topology, dtype, device, and
I2 capability before retaining anything. It also computes the exact logical-state
metadata projection described below; a topology/configuration whose future snapshot
would exceed that projection budget is `invalid-argument`. It deep-copies the
normalized logical configuration and retains the stable unique P1 handles, not the
transient parameter tree container. Successful optimizers are exclusive mutable
identities with no concurrency or reentrancy claim.

The only admitted algorithm is AdamW on dense, zero-offset, contiguous CPU f32
parameters and gradients. Sparse gradients, f64/f16/bf16 parameters, mixed or
master precision, accelerators, AMSGrad, maximize mode, fused/foreach variants,
loss scaling, token-based schedules, and every unlisted schedule are `unsupported`.
There is no scalar, cast, transfer, precision, device, finite-difference, or Python
fallback.

## Successful-update schedule semantics

`completed-updates` starts at zero and counts only committed optimizer updates. Let
`n = completed-updates + 1` for the update being prepared. A failed step does not
change the counter or schedule.

The constant schedule has factor 1. For a linear schedule with warmup `W`, total
updates `T`, and minimum ratio `R`:

```text
if W > 0 and n <= W: factor(n) = n / W
if n > W and n <= T: factor(n) = 1 - (1 - R) * (n - W) / (T - W)
if n > T:            factor(n) = R
```

For `W = 0`, the second branch applies from the first update. The mathematical
factor is rounded once to nearest-even binary32. Each group's effective learning
rate is its base learning rate times that factor, rounded once to nearest-even
binary32. No separately mutable scheduler cursor exists; the value is derived from
the immutable schedule definition and `completed-updates`.

Cosine and other transcendental schedules are intentionally not admitted in version
1 because their exact cross-host evaluation contract has not been established.
Token-count schedules are TR3 scope: the fixed A0 optimizer operations have no token
ingress.

## Accumulated-gradient and clipping semantics

The caller or TR3 decides when to call `optimizer-step!`. I2 owns the physical
accumulation slots and contribution metadata. For each unique parameter, the
effective gradient consumed by O2 is semantically:

```text
sum_i(contribution_numerator_i) / sum_i(mask_weight_i)
```

over one positive-total-weight accumulation set. This is the gradient of the global
weighted objective, not an unweighted mean of per-microbatch means. The exact I2
carrier/transaction API remains an I2 decision. O2 acceptance must compare unequal
microbatch weights against one reference backward of the combined objective.

Every unique bound handle must have one present, finite, same-shape dense CPU f32
effective gradient. A missing gradient is `invalid-state`; a present all-zero
gradient is valid. Tied aliases are inspected, normed, clipped, decayed, and updated
once through their canonical handle.

Clipping is disabled by `(none)`. `(global-l2 M)` computes one global L2 norm over
all elements of all unique effective gradients in canonical parameter-path and
row-major element order. I2 must use a reviewed scaled sum-of-squares implementation
that detects nonfinite input and unrepresentable output without avoidable overflow.
If `norm <= M`, gradients are unchanged, including at the exact boundary. If
`norm > M`, every effective gradient is multiplied once by `M / norm`. Clipping
occurs after accumulation normalization and exactly once before AdamW. The clipped
values are staged update inputs: neither clipping nor a successful step mutates the
retained I2 gradient slots or their contribution metadata. Only
`optimizer-zero-grad!` clears those slots.

## AdamW update and atomicity

For each unique parameter and successful-update index `n`:

```text
m       = beta1 * m + (1 - beta1) * g
v       = beta2 * v + (1 - beta2) * g * g
m_hat   = m / (1 - beta1^n)
v_hat   = v / (1 - beta2^n)
p_decay = p * (1 - effective_lr * weight_decay)
p_next  = p_decay - effective_lr * m_hat / (sqrt(v_hat) + epsilon)
```

Weight decay is decoupled. A present zero gradient still initializes/advances the
moments, applies decoupled decay, and commits the successful-update counter. O2
creates no gradient graph.

Each previously unseen moment tensor is mathematically initialized to positive-zero
f32 with the parameter shape before the first prepared update. The equations above
define reference mathematics, not a frozen Eshkol evaluation order: binary32
operation grouping, power, square-root, norm accumulation, and rounding points will
be specified only after merged I2 exposes the real tensor path and compiled Eshkol
probes validate it. Until then the PyTorch fixture is tolerance-based development
parity, not bit-determinism evidence.

Before the first write, O2 and I2 must validate every gradient and prepared
parameter/moment value, counter increment, alias relation, storage identity, and
scratch requirement. NaN, infinity, norm overflow, bias-correction failure, or any
nonfinite prepared result rejects the entire step. The commit point is the first I2
write. From that point the admitted whole-batch commit is one-shot, nonallocating,
nonraising, and preserves every live parameter storage identity. A recoverable
failure leaves parameters, moments, gradients, accumulation metadata,
`completed-updates`, and schedule values unchanged.

A successful step does not implicitly clear I2 gradient slots. The caller uses
`optimizer-zero-grad!`, which clears every unique slot and its accumulation metadata
exactly once without changing parameters, moments, groups, schedule, or counters.
Repeated zeroing is idempotent.

## Logical optimizer state 1.0

`optimizer-state` returns a deep-owned, data-only opaque value with this exact
logical field order (the angle-bracketed tensors are owned logical dense-tensor
values, not a frozen carrier or serialization):

```text
(transformer-optimizer-state 1 0 ()
  adamw f32 cpu
  <inert-p1-provider-identity>
  <transformer-optimizer-config-1.0>
  (<canonical-path> ...)
  (<canonical-p1-alias-group> ...)
  <completed-updates>
  ((parameter-state <canonical-path> <shape> f32 cpu row-major
                    <owned-exp-avg-f32-tensor>
                    <owned-exp-avg-sq-f32-tensor>) ...)
  none)
```

The outer identity, version `(1 0)`, and empty required-feature list are exact. It
contains:

- algorithm `adamw`, precision `f32`, and device `cpu`;
- inert P1 tensor-provider identity;
- the complete copied optimizer configuration, canonical P1 path set, and alias
  graph;
- nonnegative signed-i64 `completed-updates`;
- one entry per unique canonical parameter path, in P1 order, containing redundant
  shape/dtype/device/layout metadata plus independent owned dense CPU f32
  `exp-avg` and `exp-avg-sq` tensors;
- RNG policy `none`.

The accepted receiver/tensor lifetime contract makes each snapshot one explicit
O2 opaque owner with distinct O2 token kinds/ownership ledgers. Partial or rejected
construction records each successful clone as sole O2 ownership before any later
fallible action, cleans the exact unpublished ledger prefix once on failure, and
transfers the complete ledger once on publication. Load and future C2 use first
acquire O2 state-borrow authority, resolve non-owning moment handles synchronously,
end every I2/K1 borrow in a guaranteed tail, and retain no raw storage or carrier.

Release validates the exact registered O2 receiver, complete ownership ledger,
liveness, exact accepted I2 provider identity, serialization/nonreentrancy, and zero
active borrows before mutation. It then atomically closes all state/handle resolution
by entering an internal releasing state before the first callback. The fixed
provider-2.0 exact-once tail is nonallocating/nonraising; it publishes dead only after
all moment clones are released. There is no recoverable rollback after cleanup
starts. Repeating release on that exact registered dead token is an idempotent no-op;
other copied, forged, cross-owner, cross-aggregate, or stale tokens reject. Releasing
a snapshot must not change live optimizer moments, parameters, gradients, counters,
configuration, P1 state dictionaries, or another clone.

Each optimizer-state-backed moment handle is tied to its exact owner state,
entry/canonical path, and provider. Trusted resolution validates handle kind,
owner live/not-releasing state, entry membership, exact provider binding, and borrow
admission before exposing the owned carrier. The handle can never release, transfer,
or consume ownership. Provider metadata is inert and never selects executable code.

P1 `state-dict-release!`, P1 state tokens, and P1 state-backed handles remain
P1-specific and grant no O2 release authority. Process-lifetime tensor retention,
hidden finalizers, equality-triggered freeing, generic release dispatch, and a
second registry are forbidden. The only admitted release seam is a fixed O2-specific
private wrapper in the single trusted aggregate, statically bound to the exact
accepted I2 provider identity and exact O2 ownership-ledger entry. It exposes no
callback token and accepts no caller-selected provider, P1 state, live optimizer
moment, parameter, or gradient. No private symbol, token code, request layout, or
structure ABI is frozen before merged P1L/I2 provide the concrete substrate.

Parameter values are P1 model state and never occur in optimizer state. Schedule
values are derived rather than redundantly stored. The logical state contains no
parameter handles, addresses, callbacks, provider selection, native plans,
capability evidence, magic bytes, field numbers, byte order, checksum, filesystem
path/operation, or C1/C2 container decision.

Version 1 optimizer snapshots are update-boundary snapshots. Every I2 gradient slot
must be absent and its accumulation metadata empty; otherwise `optimizer-state`
raises `invalid-state`. Snapshot, load, and release additionally require that the
optimizer is not mutating and no moment/state borrow is live. Mid-accumulation
checkpointing is not represented by O2 1.0 and remains a later TR3/C2 contract
decision. This update-boundary call-phase rule grants the detached snapshot no live
optimizer/handle backreference or authority; implementation must enforce it through
the accepted caller/aggregate orchestration boundary or return to integration.

`optimizer-load-state!` validates the complete raw state, version/features, provider
identity, configuration, bound paths/aliases, counter, entry uniqueness,
metadata, tensor independence, finite moment values, and I2 capability before
mutation. State metadata has depth at most 32, at most 2,097,152 data nodes, at most
4,096 parameter entries, the P1 64-segment/per-segment UTF-8 bounds, and a
67,108,864-byte aggregate UTF-8 budget. The create-time projection counts every
logical occurrence in the embedded config, canonical path list, alias graph, and
parameter entries; tensor elements/bytes remain parameter-bound and await I2's
separate limits. Canonical paths, alias groups, and parameter entries are ordered by
P1 UTF-8 order; each unique canonical path occurs exactly once. Shapes have rank at
most 64 and exact nonnegative signed-i64 extents.

The receiver must already bind the identical algorithm, dtype, device, provider,
canonical unique path set, and alias topology. On success, load atomically replaces
the complete group assignment and hyperparameters, clipping rule, schedule
definition, `completed-updates`, and both moment tensors for every path. Thus config
and schedule are not independently mutable during ordinary stepping, but a strict
state load restores them. It never changes parameter values or I2 gradient slots. A
failed load preserves the old moments, counter, configuration, schedule, parameters,
and gradient slots exactly. The input is borrowed; successful load retains no
caller-mutable carrier. The receiver's I2 gradient slots and accumulation metadata
must be absent/empty before load; otherwise load is `invalid-state`. Load acquires
the input state-borrow authority before any moment-handle resolution, releases it on
every path, stages/copies into optimizer-owned moment storage, and never adopts an
input state carrier.

Unknown major versions, nonzero minor versions, and unknown required features are
`version-mismatch`. Malformed envelopes, duplicate entries, invalid counters, and
nonfinite moments are `corrupt-data`. Missing/unexpected paths or alias topology are
`shape-mismatch`; tensor metadata uses the matching A0 shape, dtype,
device, and noncontiguous categories. An unavailable or mismatched admitted provider
is `unsupported`.

## Exact public error mapping

- `optimizer-create`: malformed/boundedness/range/order failures, duplicate or alias
  paths, and unknown paths are `invalid-argument`; an incomplete unique-handle set is
  `shape-mismatch`; stale, foreign, or cross-aggregate trees/handles are
  `invalid-state`; tensor metadata maps to `shape-mismatch`, `dtype-mismatch`,
  `device-mismatch`, or `noncontiguous`; unavailable provider, algorithm, precision,
  device, layout, or verified I2 capability is `unsupported`.
- `optimizer-step!`: missing/stale gradients, nonfinite gradients or prepared
  results, nonpositive/nonfinite accumulated total weight, and update-counter
  overflow are `invalid-state`; gradient metadata uses the four matching tensor
  categories; unavailable I2 operations are `unsupported`; inability to satisfy a
  requested deterministic contract is `determinism-unavailable`.
- `optimizer-zero-grad!`: stale/foreign/closed optimizer or slots are
  `invalid-state`; an unavailable exact I2 clear operation is `unsupported`.
- `optimizer-state`: a stale/foreign/closed optimizer or any present/pending gradient
  state is `invalid-state`; unavailable detached clone/state capability is
  `unsupported`.
- `optimizer-load-state!`: malformed, over-bound, duplicate, nonfinite, or
  internally inconsistent logical state, including an incomplete, duplicate, or
  otherwise invalid state group assignment, is `corrupt-data`; unsupported versions
  or required features are `version-mismatch`; receiver path/alias/shape mismatches
  are `shape-mismatch`; a different valid receiver group partition is accepted and
  replaced; tensor dtype/device/layout use the matching categories; a stale receiver
  is `invalid-state`; an unavailable or mismatched provider/capability is
  `unsupported`.
- `optimizer-state-release!`: malformed, non-state, wrong-kind, forged, copied-token,
  unregistered, or cross-aggregate receivers are `invalid-argument`; a recognized
  live owner that is busy, reentrant, releasing, or has an owner-state conflict is
  `invalid-state`; the exact registered dead token succeeds idempotently. Unavailable
  exact provider/release capability is `unsupported` only before state construction
  or load. A post-admission provider invariant violation is `internal`, never a
  recoverable `unsupported` release branch.

Integration accepted `invalid-state` for a recognized, registered, exact-kind dead
optimizer-state passed to any non-release operation in
[comment 5493574404](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5493574404).
This includes load, trusted inspection/resolution, state-backed moment-handle
resolution, and future C2 serialization, all before native dereference. Only
`optimizer-state-release!` on that exact registered dead token succeeds
idempotently without a provider callback.

Future C2 serialization and trusted inspection acquire and end every state/I2/K1
borrow on every path and release every temporary optimizer-state owner they create.
They never serialize process-local owner tokens, callback identities, provider
authority, or capability evidence.

## Required continuation evidence

After N committed updates, tests snapshot one optimizer, construct a second optimizer
over independently owned parameters with identical values, load the detached owned
state (and later the equivalent trusted state reconstructed by C2), install identical
next accumulated gradients, and require bit-identical next Eshkol parameter/moment
tensors, effective learning rates, and counters. This is
separate from tolerance-based PyTorch parity and commits O2 to exact logical-state
continuation without committing C2 bytes.
