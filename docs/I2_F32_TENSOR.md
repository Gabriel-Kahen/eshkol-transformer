# I2 dense CPU-f32 tensor and parameter-gradient substrate

Status: **blocked on P1L**. I2 is a shared prerequisite of N2 and O2. The
implemented scope is a versioned carrier-local native storage, gradient, and
atomic-plan boundary. Production P1/Eshkol implementation, aggregate freeze, and a
review-ready transition wait for the independently merged release-capable P1L
contract. I2 adds no A0 public Eshkol name or arity, numerical layer, optimizer
algorithm, checkpoint format, mixed-precision policy, accelerator support, or
fallback.

## Evidence boundary

I2 owns physical IEEE-754 binary32 storage on CPU. The native ABI requires
eight-bit bytes and four-byte `float` with radix 2, 24-bit significand, and the
binary32 exponent range. Public copy ingress and egress use `uint32_t` bit
patterns and `memcpy`, so `+0`, `-0`, subnormals, finite extrema, infinities,
and NaN payloads round-trip without numerical conversion. There is no `double`
carrier, cast, materialization, transfer, scalar substitute, Python runtime, or
device fallback.

The public C11 header is `include/eshkol_transformer/f32_tensor.h`; ABI 1.0
exports only `_v1` entry points. The native artifact is
`build/i2/libeshkol_transformer_f32.a`. Its exact symbol and visibility policy is
produced by the I2 build gate. No production Eshkol aggregate or P1 provider is
built on this branch. I2 has no install or dynamic-library search contract.

## Owned tensor ABI 1.0

An `et_f32_tensor` exclusively owns its shape, byte strides, and naturally
aligned data allocation. Rank is `0..64`. Rank zero is one element and four
bytes. For positive ranks, an extent of zero canonically means zero elements,
zero bytes, null data, and zero byte strides in every dimension. Otherwise the
container computes dense row-major byte strides with a four-byte innermost
stride. Rank, extent products, byte counts, caller spans, address ends, and
allocations are checked before publication.

Creation requires a null output slot and deep-copies shape metadata. Destruction
is null-idempotent and nulls the caller slot only after success. Scalar and
pointer outputs are byte-preserved on failure. Exact-bit copies require the full
element count; empty storage accepts a null buffer. A nonempty caller buffer must
be naturally aligned, have a representable complete span, and be disjoint from
every process-local live I2 allocation.

Opaque handles are admitted by process-local registry membership before their
contents are dereferenced. Foreign, stale, copied-after-release, and wrong-kind
handles reject before mutation. The registries and receivers are unsynchronized;
callers must serialize all access. Arbitrary invalid native addresses, debugger
or process-memory mutation, and malicious native objects linked into the trusted
process are outside the ABI threat model.

Clones are detached owned tensors with the same shape and exact value bits.
Storage identity is object identity, while exact value equality requires the
same shape and byte-identical payload. Independent successful allocations are
storage-disjoint.

## Borrowed K1 views

One tensor permits at most one tracked borrow lease. The lease contains the
unchanged `et_kernel_tensor_view_v1` prefix with:

- dtype `f32`, device `cpu`, dense-row-major layout, and zero offset;
- the exact owned rank, shape, and byte span; and
- null data only for empty storage, otherwise naturally aligned stable data.

The tensor, shape, data, and view remain live and stable until explicit
`et_f32_tensor_borrow_end_v1`. An active borrow blocks owner-side mutation,
destruction, and prepared mutation plans. The borrower may read or write only as
authorized by the surrounding K1 call contract; I2 does not infer input/output
intent and retains no downstream view after a synchronous call.

This contract satisfies the ordinary N2/A2/L2 synchronous-view seam. A2 cache
storage remains A2-owned and must not be registered as I2 parameter or gradient
storage. A2's transaction-scoped full-capacity cache views, effective lengths,
and masks remain governed by A2. Exact i64 operands use the accepted I1 or
another separately reviewed exact carrier, and bool operands require a separate
exact bool carrier; numeric Eshkol vectors are not substitutes.

## K1 provider and composition

`et_f32_tensor_provider_v1()` returns an immutable explicit provider descriptor.
It verifies only this deterministic request family:

- capability `tensor.f32`;
- operation `storage.copy`;
- compute dtype `f32` and device `cpu`;
- rank zero or rank one with extent `0..SIZE_MAX/4`; and
- one dense zero-offset input and one disjoint caller-owned output of identical
  shape.

Validation is complete and non-mutating. Invoke performs only the admitted full
`memcpy`, allocates nothing, and cannot report a recoverable failure. I2 does not
define `eshkol_transformer_kernel_provider_v1`, mutate K1's provider-free
baseline, or claim `tensor.contiguous`, `autodiff.reverse`, any N2/A2/L2/O2
kernel, GPU, f16, or bf16.

Later Wave 2 composition must collect descriptors through their versioned accessors,
reject duplicate capability/operation ownership, and route an unchanged request and
unchanged view identities to exactly one provider. That aggregate is deferred until
P1L. N2, A2, and L2 retain ownership of their operation-specific validation and
numerical evidence. A private test transport or provider resolver is never
production capability evidence.

## Carrier-local parameter and gradient identity

The binding P1L 2.0 integration target is
`(transformer-tensor-provider 2 0 i2-dense-cpu-f32-v1)`. It is not registered by
the carrier-only artifact. Interface 1.x must not emulate the release callback or
load 2.0 state. An incompatible carrier, device, clone, gradient, or prepare/commit
semantic requires a different provider ID.

The private carrier-local parameter object contains stable live value storage,
preallocated same-shape gradient storage, one opaque identity binding, and gradient
metadata. Duplicate identities are rejected in whole-batch plans so a future P1
canonical tied handle can be updated exactly once. Independent objects have
disjoint value and gradient storage. This native identity is not a second P1
registry and does not authorize an Eshkol handle.

P1L provider interface 2.0 adds an eighth exact-once owned-carrier release callback
to describe, detached clone, storage identity, exact value equality, exact device
equality, whole-batch prepare, and one-shot commit. I2 will integrate only after that
layout merges. Each successful native clone has one owner and is destroyed exactly
once. A public `state-dict-tensor` will be only a read-only state-backed identity;
its private synchronous resolution must validate P1 owner/state/entry/provider
liveness, acquire/use/end any I2 K1 borrow wholly within one call, and retain no raw
pointer. State release is admitted only with no active resolution call. The current
artifact makes no production P1 snapshot/load claim.

## Accumulated gradients

A unique parameter gradient slot is exactly one of:

- **absent**: state absent, contribution count `0`, normalization-weight bits
  exact `+0` (`0x00000000`), and every backing gradient element exact `+0`; or
- **present**: state present, count in `1..INT64_MAX`, positive finite binary32
  normalization weight, and a same-shape dense CPU-f32 accumulated unnormalized
  weighted numerator.

Present all-zero remains distinguishable from absent. Read-only finite inspection
examines physical bits. Exact-positive-zero inspection requires every element to
be `0x00000000`; reset canonicalizes negative zero to positive zero. Gradient
snapshots are detached owned carrier-local inspection clones, not P1 state tensors
or public state accessors. Public P1 state access remains state-backed, read-only,
and synchronously borrowed after P1L. A live gradient borrow exposes the same K1
view contract as value storage and blocks mutation until released.

One contribution batch names each intended unique handle exactly once, supplies a
finite already-weighted numerator tensor per handle, one shared positive finite
binary32 normalization-weight increment, and the expected zero-based contribution
ordinal. Every destination must have the same pre-state count and normalization
metadata, and every ordinal must equal that count. A repeated tied handle is a
duplicate destination, not another contribution.

Preparation validates the complete handle set, identities, shapes, liveness,
metadata, source/destination independence, active borrows, finite current and
source values, count overflow, and finite positive weight addition. It allocates
all scratch and computes every next numerator and metadata value before mutation.
The operation order is fixed canonical-handle order and row-major element order
with binary32 arithmetic and contraction/excess precision disabled. The supported
x86-64 boundary also admits accumulation only while MXCSR selects round-to-nearest
ties-to-even, masks every floating-point exception, and disables both denormals-are-
zero and flush-to-zero. Preparation rejects any other environment before mutation;
each addition preserves the admitted control/status word. Any nonfinite result or
attempt to advance an `INT64_MAX` contribution count rejects before mutation.

O2 reads a present accumulated numerator, count, and normalization weight and may
stage normalized or clipped update values, but optimizer-step and clipping do not
clear or change the retained I2 gradient state. Only explicit
`optimizer-zero-grad!` or `module-zero-grad!` performs the atomic, idempotent
transition to canonical absent state. O2 state snapshot/load are update-boundary
operations and require every bound slot absent. O2 owns norm arithmetic, AdamW,
moments, config/state, clipping, counters, and successful-step schedules; TR3 owns
when to call optimizer-step and any token-based schedule.

## Whole-batch mutation

Value loads, gradient contributions, and gradient resets use opaque bounded plans.
All recoverable validation, alias checks, allocation, and result computation occur
during prepare. A successful prepare pins every participating live object until
plan release.

Commit marks the plan consumed before its first destination write, then performs a
fixed full-write sequence of staged bytes and metadata. It allocates nothing,
invokes no callback, takes no recoverable branch after mutation begins, and cannot
raise. A second commit rejects before mutation. Sources remain byte-identical and
destination storage identities remain stable. Malformed metadata, later-entry
failure, allocation failure, duplicate or aliased destinations, source/destination
overlap, active borrow, forged/stale/cross-aggregate identity, overflow, and
nonfinite arithmetic therefore leave every destination value, gradient byte,
gradient metadata field, and caller output byte-identical.

The carrier-local reset plan requires a unique identity set and commits exact
positive-zero bytes plus absent metadata once per object. After P1L integration,
`module-zero-grad!` must deduplicate canonical P1 handles before constructing this
plan. Other P1 providers remain outside I2's evidence.

## P1L integration and lifetime gate

Native tensor, borrow, parameter, and plan values have explicit release. Process-
lifetime tensor retention, hidden finalization, equality-triggered free, and a
generic privileged release dispatcher are rejected. P1L must merge provider
interface 2.0, public idempotent `state-dict-release!`, state-backed read-only tensor
handles, and fixed lifecycle identity entrypoints before I2 builds an Eshkol
aggregate.

The accepted serialized/nonreentrant rule distinguishes a state-backed identity
from an active I2 borrow. Private resolution acquires and ends a borrow inside one
call. `state-dict-release!` rejects before mutation if such a call is active; after
admission, exact-once I2 destruction is the nonallocating/nonraising/infallible tail.
Small identity tombstones may remain, but retain no tensor storage. Testing-only
live counts cover tensors, parameters, borrows, and every plan kind so later P1L
integration can prove return to baseline on success, rollback, and release.

The future aggregate must reuse the one accepted E1B/P1 registry-owning topology,
localize once, expose no generic privileged dispatcher or canonical K1 symbol, and
be an alternative to earlier registry-owning artifacts. None of those aggregate or
application-boundary properties is claimed by the current carrier archive.

## Native errors and future mapping

The 264-byte caller-owned native error record is independent of K1's error record.
A later P1L aggregate must map its categories to A0/E1 as follows:

| I2 category | E1/A0 category |
|---|---|
| `INVALID_ARGUMENT` | `invalid-argument` |
| `SHAPE_MISMATCH` | `shape-mismatch` |
| `NONCONTIGUOUS` | `noncontiguous` |
| `INVALID_STATE` | `invalid-state` |
| `VERSION_MISMATCH` | `version-mismatch` |
| `INTERNAL` or unknown | `internal` |

Allocation failure is explicit `INTERNAL/ALLOCATION_FAILED`; it is never relabeled
as a shape, state, or capability error. A live-storage alias cannot be used as an
error or output record.

## Explicit limitations

- Owned storage is CPU binary32 only. There is no f64, f16, bf16, accelerator,
  cast, transfer, materialization, or fallback path.
- The container admits ranks `0..64`, but its verified K1 `storage.copy`
  capability admits only rank zero and bounded rank one. Higher-rank ownership
  and borrowing are not a broader K1 capability claim.
- A tensor permits one active borrow, and all registries require caller
  serialization. Thread safety and concurrent mutation are not claimed.
- Native objects have explicit release. No Eshkol module/state lifetime claim is
  made until P1L's explicit release protocol merges and is integrated.
- I2 supplies no exact i64 or bool storage, neural operation, general reverse AD,
  optimizer, clipping/norm algorithm, schedule, checkpoint codec, serialization,
  performance, or accelerator evidence.
- A successful optimizer step does not consume or clear accumulated gradients.
  Callers must reset them explicitly at the accepted module/optimizer boundary.
- I2 supplies native clone/destroy mechanics but does not own O2 optimizer-state
  moment lifetimes. The accepted O2-specific release wrapper must validate the exact
  provider identity and its ownership-ledger entry before calling a fixed private
  same-aggregate I2 destroy seam. It cannot expose generic release authority or
  release live parameter, gradient, optimizer, or P1 state storage. C2/O2 inspection
  borrows acquire/use/end synchronously, and snapshot/release/load failpoints must
  return every I2 carrier, borrow, and plan count to baseline. I2 adds no O2 public
  name.

Run the focused gate with:

```sh
/usr/bin/bash -c 'make test-i2'
```

The current carrier gate warning-cleans C11/C++17 consumers, verifies the exact
one-member archive plus defined/undefined/visibility manifests, runs exact-bit,
storage, shape, alias, borrow, lifetime, gradient, plan, failpoint, failure-atomic,
and live-count tests, and exercises ASan/UBSan. It also rejects the canonical K1
provider symbol and production Python/PyTorch references.

Fresh-cache source/object/AOT application-boundary negatives, real P1/C1/T1/I2
state snapshot/load/release evidence, and the source-composed aggregate are deferred
until the independently approved P1L merge. Supported Ubuntu 22.04 x86-64 with
LLVM-Clang 21.1.8 is required at the exact final PR head before I2 can leave blocked
status or merge.
