# TR3-C private I2/O2 restore boundary proposal

Status: **proposal only; independent review and root freeze required before
implementation**.

Source commit: `3603ab4f090c57b826fbdc3fd545b2e1fc941a56`.
Source tree: `c17f953653575bdcc67daae1cd15bdfc183a1a55`.

This document closes the two native authority gaps discovered while attempting
the bounded O2 receiver portion of
[TR3-C](TR3_C_LIVE_STATE_CONTRACT.md). It amends only the private,
source-composed I2/O2 aggregate. It adds no public API, provider row, serialized
field, generic tensor-kind query, optimizer registry, optimizer exclusion,
trainer token, callback, or concurrency guarantee.

If root freezes this proposal, its exact shared-authorizer and native-boundary
terms replace the conflicting minimum/native wording in parent sections 6.2,
9 steps 3 through 7, and 10's three-function I2 list. Every other parent
requirement remains unchanged. Until that decision, neither document
authorizes implementation.

The existing `ET_TR3_O2_PRIVATE_OPERATIONS` implementation remains the sole
optimizer receiver-exclusion lineage. The existing `busy` field,
`active_operation` field, and static
`et_o2_active_operation_acquire/owns/release` helpers are reused byte-for-byte.
TR3-C adds one distinct typed restore-plan registry and tombstones, as already
required by section 6.2 of the parent contract; that registry authenticates
restore plans, not optimizers.

## 1. Actual-source gaps

The current sources cannot implement section 6.2 without an invented trust
assumption:

1. `et_i2_const_tensor` admits `ET_I2_CARRIER_TENSOR` and
   `ET_I2_CARRIER_OWNED_CLONE` identically. The existing live-carrier probe
   therefore cannot distinguish a genuine private owned clone from an ordinary
   live tensor. `ownership_kind` is private to `f32_tensor.c`, and the only
   current exact check is the destructive owned-release operation.
2. `et_i2_private_copy_builder_set_v1` admits any live builder index below its
   count. There is no operation that proves the TR3-C builder has exactly 42
   assignments, that the model compositor already filled exactly prefix
   `[0,14)`, or that an O2 operation exclusively filled suffix `[14,42)`.
3. The parent contract freezes the Scheme two-cell transfer and source-visible
   logical-config publication, but not the native request record and split
   no-failure control tail needed to preserve that ordering.

Implementation remains blocked until this proposal is reviewed and frozen.

## 2. Feature localization

The approved TR3-C and G3-R private aggregate closures define the shared
`ET_I2_PRIVATE_OWNED_CLONE_MATCH` conditional while compiling the one canonical
`f32_tensor.c` and every private translation unit that consumes its declaration
(TR3-C I2 bridge, TR3-C O2, and the approved G3-R consumer). That conditional
declares the one shared authorizer in `f32_parameter_internal.h`; only
`f32_tensor.c` defines it. No R-only, TR3-C-only, ordinary, public, or
predecessor closure owns a competing authorizer or redeclaration. TR3-C
additionally defines `ET_TR3_C_I2_RESTORE_PRIVATE` while compiling
`i2_wave2_package_bridge.c`, and defines `ET_TR3_C_O2_RESTORE_NATIVE` while
compiling `o2_optimizer.c`.
`ET_TR3_C_I2_RESTORE_PRIVATE` requires and enables the existing
`ET_TR3_C_I2_CHECKED_BRIDGE` in that same bridge object so restore retirement
and owned-stage draining use the accepted checked implementation. It never
links a second ordinary bridge or tensor object.

`ET_TR3_C_O2_RESTORE_NATIVE` causes `o2_optimizer.c` to enable the already
existing `ET_TR3_O2_PRIVATE_OPERATIONS` block and include exactly one new
`tr3_c_o2_restore.inc` after the shared acquire/owns/release helpers are
defined. A build that enables both update-clear and restore includes both typed
participants in the same O2 translation unit. Ordinary I2, ordinary O2, C2,
and the public packages define neither feature.

The fixed restore declarations live only in:

```text
native/tr3_c_i2_restore_internal.h
native/tr3_c_o2_restore_internal.h
```

Neither header is installed. The shared authorizer declaration lives only in
the existing uninstalled `native/f32_parameter_internal.h`, under
`ET_I2_PRIVATE_OWNED_CLONE_MATCH`. Feature and ordinary objects have exact
separate defined/undefined-symbol manifests. With the gates absent, ordinary
object bytes and the existing public/private manifests must remain unchanged.
The feature undefined-symbol manifests explicitly contain the one shared
owned-clone authorizer and checked release relation; ordinary I2/O2/C2
manifests contain neither restore symbol.

## 3. Exact owned-clone admission

`f32_tensor.c`, the only translation unit that owns the live-tensor registry
and `ownership_kind`, defines this uninstalled operation:

```c
enum {
  ET_I2_PRIVATE_OWNED_RESULT_OK = 0,
  ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT = 1,
  ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE = 2,
  ET_I2_PRIVATE_OWNED_RESULT_SHAPE_MISMATCH = 3,
  ET_I2_PRIVATE_OWNED_RESULT_NONFINITE = 4,
  ET_I2_PRIVATE_OWNED_RESULT_INVALID_VALUE = 5,
  ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT = 6,
  ET_I2_PRIVATE_OWNED_RESULT_VERSION_MISMATCH = 7,
  ET_I2_PRIVATE_OWNED_RESULT_ALLOCATION_FAILED = 8,
  ET_I2_PRIVATE_OWNED_RESULT_INTERNAL = 9
};

enum {
  ET_I2_PRIVATE_OWNED_VALUE_FINITE = 1,
  ET_I2_PRIVATE_OWNED_VALUE_FINITE_NONNEGATIVE = 2
};

int32_t et_i2_private_owned_clone_match_v1(
    const et_f32_tensor *candidate,
    const et_f32_tensor *destination,
    uint32_t value_policy);
```

The function is nonallocating, nonmutating, nonraising, and constructs no error
record. It performs only fixed registry scans and scalar/data reads:

1. Find both pointer values by comparison in the real live-tensor registry.
   Do not dereference either caller value before both lookups succeed.
2. Require both registered controls to have the live tensor magic.
3. Require the candidate's private field to equal
   `ET_F32_OWNERSHIP_PRIVATE_CLONE`. An ordinary clone or parameter value is
   `INVALID_ARGUMENT`, even when otherwise live and shape-compatible.
4. Require candidate and destination to have no active borrow and zero plan
   pins. A genuine owned clone or destination that is borrowed/pinned is
   `INVALID_STATE`.
5. Require exact rank/extents, canonical dense row-major strides, element count,
   and byte length equality. A mismatch is `SHAPE_MISMATCH`.
6. Require one of the two exact value policies; an unknown policy is
   `INVALID_ARGUMENT`. Scan the candidate's binary32
   payload by `memcpy` into `uint32_t` values and reject either NaN or infinity
   as `NONFINITE`. `FINITE_NONNEGATIVE` additionally rejects a set sign bit
   when the magnitude is nonzero as `INVALID_VALUE`; positive and negative zero
   are accepted.

The matcher never returns `OWNER_CONFLICT`, `VERSION_MISMATCH`,
`ALLOCATION_FAILED`, or `INTERNAL`; those shared results are reserved for the
restore-builder boundary.

The successful path has no write, callback, provider call, FFI call, error
construction, allocation, or borrow object. The operation is a private
source/destination match, not a general ownership-kind or metadata query. The
TR3-C model prefix, O2 `exp_avg`, and G3-R use `FINITE`; O2 `exp_avg_sq` uses
`FINITE_NONNEGATIVE`.

Genuine inputs are created by `et_f32_owned_tensor_clone_v1`; the accepted
source-composed operation remains `et_i2_private_owned_clone_v1`. Ordinary
`et_f32_tensor_clone_v1` does not create an admissible owner.

## 4. Fixed restore-42 builder boundary

The feature build extends the private `et_i2_copy_builder` control with these
fields:

```c
const void *tr3_c_restore_authority;
size_t tr3_c_next_index;
uint32_t tr3_c_restore_phase;
```

The exact phases are `ORDINARY=0`, `PREFIX=1`, `COMPLETE=2`, `PREPARED=3`,
`COMMITTED=4`, and `ABORTED=5`. Ordinary builders retain zero values. No API
reads or publishes the builder count, phase, assignments, or authority. A
restore builder is created only after the O2 plan exists, with that exact plan
as its nonnull authority; it always owns 42 zeroed assignments and begins at
`PREFIX` with `next_index == 0`. It is never an ordinary count-42 builder.

The new records and operations are fixed-size/fixed-arity:

```c
enum {
  ET_TR3_C_MODEL_DESTINATIONS = 14,
  ET_TR3_C_O2_MOMENT_ASSIGNMENTS = 28,
  ET_TR3_C_RESTORE_ASSIGNMENTS = 42,
  ET_TR3_C_O2_FIRST_INDEX = 14
};

enum {
  ET_TR3_C_I2_TERMINAL_COMMITTED = 1,
  ET_TR3_C_I2_TERMINAL_ABORTED = 2
};

typedef struct et_tr3_c_i2_restore42_append_v1 {
  size_t struct_size;
  const void *restore_authority;
  et_f32_parameter *parameters[ET_TR3_C_MODEL_DESTINATIONS];
  const void *p1_handles[ET_TR3_C_MODEL_DESTINATIONS];
  et_f32_tensor *moment_destinations[ET_TR3_C_O2_MOMENT_ASSIGNMENTS];
  const et_f32_tensor *moment_sources[ET_TR3_C_O2_MOMENT_ASSIGNMENTS];
} et_tr3_c_i2_restore42_append_v1;

#define ET_TR3_C_I2_RESTORE42_APPEND_V1_0_SIZE \
  ((size_t)sizeof(et_tr3_c_i2_restore42_append_v1))

int32_t et_tr3_c_i2_copy_builder_create_restore42_v1(
    const void *restore_authority,
    void **builder);

int32_t et_tr3_c_i2_copy_builder_append_parameter_v1(
    void *builder,
    const void *restore_authority,
    et_f32_parameter *parameter,
    const void *p1_handle,
    const et_f32_tensor *owned_source);

int32_t et_tr3_c_i2_copy_builder_append_restore42_v1(
    void *builder,
    const et_tr3_c_i2_restore42_append_v1 *request);

int32_t et_tr3_c_i2_copy_builder_verify_restore42_v1(
    const void *builder,
    const et_tr3_c_i2_restore42_append_v1 *request);

int32_t et_tr3_c_i2_copy_builder_prepare_restore42_v1(
    void *builder,
    const void *restore_authority);

int32_t et_tr3_c_i2_copy_builder_commit_restore42_checked_v1(
    void *builder,
    const void *restore_authority);

int32_t et_tr3_c_i2_copy_builder_abort_restore42_checked_v1(
    void *builder,
    const void *restore_authority);

int32_t et_tr3_c_i2_copy_builder_require_terminal_restore42_v1(
    const void *builder,
    uint32_t expected_terminal);
```

These symbols are feature-private C composition entries. They are not bound as
generic Scheme operations and are callable only by the typed O2 restore
participant, the fixed model compositor, the parent transaction, and focused
tests. All recoverable I2 operations return the shared numeric result enum in
section 3; callers translate it explicitly and never forward it as an O2
status.

On the supported LP64 target,
`sizeof(et_tr3_c_i2_restore42_append_v1) == 688`, with pointer arrays beginning
at offsets 16, 128, 240, and 464. The feature translation unit contains exact
`sizeof`/`offsetof` static assertions.

### 4.1 Admission

Create requires a nonnull authority and a naturally aligned writable output
slot whose initial value is null; the slot remains unchanged on failure. Its
checked byte range may not wrap or overlap a live/retired I2 builder shell and
its address may not equal the authority. Create performs the two fixed
builder/assignment allocations and returns `ALLOCATION_FAILED` if either fails.
Failure publishes no builder. Each prefix append first authenticates the live
restore builder by registry address before dereference, requires the same
authority, `PREFIX`, `next_index < 14`, and no child plan, authenticates the
exact live parameter plus its P1 handle, resolves its canonical value tensor,
and applies the finite owned-clone match. It rejects any parameter, P1 identity,
destination, or source repeated in the earlier prefix. It then writes exactly
the assignment at `next_index` and increments `next_index` last. There is no
caller-selected destination or index, so overwrite, skip, and reordering by
index are impossible.

Because the authority type is intentionally opaque, the private caller also
has the ABI precondition that the output slot is disjoint from the complete
authority object and all other live controls. The sole production caller uses
a dedicated aligned local slot, so this is structurally true and covered by a
focused local-address check. Unknown unreadable pointers and overlap
inside an unknown opaque object are outside this private C ABI; known I2
control/assignment overlap is rejected rather than treated as a precondition.

Suffix append requires a nonnull, naturally aligned request whose exact
688-byte readable range does not wrap. After authenticating the builder by
address and before reading request fields, it rejects overlap with the complete
builder shell or its full 42-assignment allocation. It copies the request into
one fixed stack snapshot; all later validation and publication reads only that
snapshot. It then completes every recoverable check without changing the
builder:

- request size is exact and `restore_authority` is nonnull;
- the builder is found in the real live-builder registry before dereference,
  has the exact magic, same authority, `count == 42`, phase `PREFIX`,
  `next_index == 14`, no prepared child plan, and entirely zero/unset suffix;
- for every `i` in `[0,14)`, the request parameter/P1 pair reauthenticates and
  resolves a canonical value tensor pointer-identical to the existing prefix
  destination;
- every prefix source is a genuine idle finite owned clone matching that exact
  destination through `et_i2_private_owned_clone_match_v1(..., FINITE)`;
- suffix index `14 + 2*i + kind` maps to the authenticated optimizer entry
  `i` and moment kind `exp_avg` then `exp_avg_sq`;
- every suffix source is a genuine idle finite owned clone matching its exact
  moment destination, using `FINITE` for `exp_avg` and
  `FINITE_NONNEGATIVE` for `exp_avg_sq`;
- all 42 destinations are distinct, all 42 sources are distinct, and no source
  aliases any destination.

Binding prefix destinations to the optimizer's exact 14 parameter value tensors
is mandatory. It proves that the model compositor and O2 participant describe
one receiver topology; count 42 and index 14 alone would permit an unrelated
model prefix. The O2 plan derives these identities only after authenticating
each optimizer entry's retained P1 identity.

Ordinary builders (including counts 41, 42, or 43), a prepared/consumed
builder, incomplete or wrong-order prefix, prefilled suffix, null/wrong
authority, ordinary source, borrowed/pinned source or destination, shape
mismatch, nonfinite source, negative nonzero second moment, duplicate
source/destination, and any cross-alias fail before a write.

### 4.2 Publication and verification

After complete admission, suffix append performs exactly 28 fixed assignment
writes, writes `next_index = 42`, and writes phase `COMPLETE` last. None can
fail. Authority was fixed at creation and never changes while live. There is no
heap allocation, callback, provider call, registry mutation, partial-return
path, or error construction in this publication suffix. Injected failures
exist only in the admission prefix; each must preserve the entire builder
byte-for-byte.

Verify repeats the exact builder, authority, prefix, suffix, owned-source,
shape, value-policy, uniqueness, and alias checks without mutation. It requires
phase `COMPLETE`, `next_index == 42`, no child plan, and the same authority
pointer. It is intentionally invalid after prepare because the owned matcher
rejects pinned tensors.

Prepare is the final recoverable pin/distinctness validator before seal. It
reauthenticates count/authority/phase/next-index and invokes the existing I2
copy prepare, which performs the two fixed child-plan/assignment-copy
allocations. Either allocation failure returns `ALLOCATION_FAILED` with the
builder byte-identical in `COMPLETE`, child plan null, and every endpoint
unpinned. It publishes phase `PREPARED` only after the exact child plan exists.
The child error translation is exact: allocation failure becomes
`ALLOCATION_FAILED`; active borrow/pin becomes `INVALID_STATE`; invalid shape or
buffer-size mismatch becomes `SHAPE_MISMATCH`; ABI mismatch becomes
`VERSION_MISMATCH`; and any other child defect after the complete restore
preflight becomes `INTERNAL`. All non-OK translations preserve the same
unchanged/unpinned state.
Commit requires `PREPARED`; abort accepts an authenticated restore builder in
`PREFIX`, `COMPLETE`, or `PREPARED`. They use the already accepted checked
child commit/release behavior and fail-stop after admitted mutation on any
impossible child defect. Commit can return a pre-mutation authentication status,
but its sealed source caller fail-stops on any nonzero result; abort propagates
pre-mutation authentication status recoverably. Retirement clears authority,
index, assignments, and child plan; preserves only the exact terminal phase
`COMMITTED` or `ABORTED`; and then links the inert tombstone. The terminal
probe compares the candidate address against the retired-builder registry
before any dereference and returns shared `OK` only when the requested terminal
disposition and exact tombstone invariants match. Unknown pointer or terminal
selector is `INVALID_ARGUMENT`; a recognized restore tombstone with the wrong
disposition is `INVALID_STATE`. Exact terminal invariants are `magic == 0`,
`count == 0`, null assignments/plan/authority, `next_index == 0`, and the
requested terminal phase. The O2 plan stores the exact builder identity as a
witness but never retires it. The terminal probe is a shared-result operation,
not a Boolean.

All existing generic set/prepare/commit/abort operations reject a restore
builder before mutation. The preexisting checked commit/abort entries remain
unchanged for D2 and other accepted builder counts; they do not authenticate
restore42. No existing wrapper is globally restricted to count 42.

## 5. Native O2 restore request

The O2 feature declares one exact fixed-profile request. It contains no paths,
aliases, trainer lease, callback, provider selector, builder, output cell, or
generic count. Those source-level relations remain the responsibility of the
accepted Scheme boundary and authenticated trainer lease.

```c
typedef struct et_o2_tr3_c_restore_entry_v1 {
  size_t struct_size;
  uint32_t learning_rate_bits;
  uint32_t beta1_bits;
  uint32_t beta2_bits;
  uint32_t epsilon_bits;
  uint32_t weight_decay_bits;
  uint32_t reserved;
  const et_f32_tensor *exp_avg_owned;
  const et_f32_tensor *exp_avg_sq_owned;
} et_o2_tr3_c_restore_entry_v1;

typedef struct et_o2_tr3_c_restore_request_v1 {
  size_t struct_size;
  uint64_t target_completed_updates;
  uint64_t expected_receiver_completed_updates;
  uint32_t clip_kind;
  uint32_t clip_max_bits;
  uint32_t schedule_kind;
  uint32_t minimum_ratio_bits;
  uint64_t warmup_updates;
  uint64_t total_updates;
  et_o2_tr3_c_restore_entry_v1 entries[ET_TR3_C_MODEL_DESTINATIONS];
} et_o2_tr3_c_restore_request_v1;

#define ET_O2_TR3_C_RESTORE_ENTRY_V1_0_SIZE \
  ((size_t)sizeof(et_o2_tr3_c_restore_entry_v1))
#define ET_O2_TR3_C_RESTORE_REQUEST_V1_0_SIZE \
  ((size_t)sizeof(et_o2_tr3_c_restore_request_v1))

typedef struct et_o2_tr3_c_restore_plan et_o2_tr3_c_restore_plan;

int32_t et_o2_tr3_c_restore_prepare_v1(
    et_o2_optimizer *optimizer,
    const et_o2_tr3_c_restore_request_v1 *request,
    et_o2_tr3_c_restore_plan **plan,
    et_o2_error_v1 *error);

int32_t et_o2_tr3_c_restore_append_v1(
    et_o2_tr3_c_restore_plan *plan,
    void *exact_i2_copy_builder,
    size_t first_index,
    et_o2_error_v1 *error);

int32_t et_o2_tr3_c_restore_check_v1(
    et_o2_tr3_c_restore_plan *plan,
    et_o2_error_v1 *error);

void et_o2_tr3_c_restore_commit_native_v1(
    et_o2_tr3_c_restore_plan *plan);

void et_o2_tr3_c_restore_finalize_v1(
    et_o2_tr3_c_restore_plan *plan);

int32_t et_o2_tr3_c_restore_abort_v1(
    et_o2_tr3_c_restore_plan *plan,
    et_o2_error_v1 *error);
```

Every entry requires `reserved == 0`. On the supported LP64 target the entry is
exactly 48 bytes with stage pointers at offsets 32 and 40. The request is
exactly 728 bytes with entries at offset 56. The feature translation unit
contains exact `sizeof`/`offsetof` assertions for every field boundary relied
on by the ABI.

Prepare requires nonnull optimizer, request, and output-slot pointers; `*plan`
must initially be null and remains byte-unchanged on every failure. The output
slot's bytes may not overlap the request or optional error record, and the
optional error record may not overlap the request. Every request and entry size
must equal the exact v1.0 size; smaller and larger records are both rejected.

All recoverable failures use literal operation `trainer-load-state!`. The exact
mapping is:

| Condition | O2 category / code |
|---|---|
| null required pointer | `INVALID_ARGUMENT / NULL_ARGUMENT` |
| nonnull prepare output or output owned elsewhere | `INVALID_ARGUMENT / OWNER_CONFLICT` |
| misaligned, wrapping, or overlapping request/output span | `INVALID_ARGUMENT / INVALID_HANDLE` |
| malformed/foreign/cross-kind object or ordinary I2 builder | `INVALID_ARGUMENT / INVALID_HANDLE` |
| wrong restore authority or builder owned by another plan | `INVALID_ARGUMENT / OWNER_CONFLICT` |
| wrong `first_index` | `INVALID_ARGUMENT / INVALID_OPTION` |
| wrong request/entry size or nonzero reserved field | `VERSION_MISMATCH / INVALID_OPTION` |
| `R` outside signed nonnegative i64 | `INVALID_ARGUMENT / COUNTER_OVERFLOW` |
| `U` outside signed nonnegative i64 | `CORRUPT_DATA / COUNTER_OVERFLOW` |
| invalid target config/option bits | `CORRUPT_DATA / INVALID_OPTION` |
| NaN or infinity in a detached moment | `CORRUPT_DATA / NONFINITE` |
| negative nonzero second moment | `CORRUPT_DATA / INVALID_MOMENT` |
| wrong shape | `SHAPE_MISMATCH / INVALID_MOMENT` |
| registered receiver provider invariant mismatch | `INTERNAL / PROVIDER_MISMATCH` |
| receiver count differs from 14 | `SHAPE_MISMATCH / MISSING_PARAMETER` |
| registered receiver has malformed internal storage | `INTERNAL / PROVIDER_DEFECT` |
| present gradient | `INVALID_STATE / GRADIENT_METADATA` |
| live borrow/pin, busy receiver, unretired builder, or wrong live phase | `INVALID_STATE / ACTIVE_BORROW` |
| consumed/aborted/committed typed plan | `INVALID_STATE / CONSUMED` |
| builder terminal disposition differs from required commit/abort disposition | `INVALID_STATE / CONSUMED` |
| live completed updates differ from captured `R` | `INVALID_STATE / COUNTER_MISMATCH` (private code 18) |
| captured config/option/parameter/P1/value/moment identity drift | `INVALID_STATE / OWNER_CONFLICT` |
| allocation failure | `INTERNAL / ALLOCATION_FAILED` |
| impossible child/internal defect before seal | `INTERNAL / PROVIDER_DEFECT` |

For I2 results, invalid argument/kind/count/alias translates to
`INVALID_ARGUMENT / INVALID_HANDLE`; authority mismatch translates to
`INVALID_ARGUMENT / OWNER_CONFLICT`; borrow/pin/phase translates to
`INVALID_STATE / ACTIVE_BORROW`; shape translates to
`SHAPE_MISMATCH / INVALID_MOMENT`; nonfinite and invalid value translate to
`CORRUPT_DATA / NONFINITE` and
`CORRUPT_DATA / INVALID_MOMENT`; allocation failure translates to
`INTERNAL / ALLOCATION_FAILED`; version mismatch translates to
`VERSION_MISMATCH / INVALID_OPTION`; and internal translates to
`INTERNAL / PROVIDER_DEFECT`. The two native `void` operations are
sealed-tail-only and `_Exit(134)` on an admitted invariant defect; they never
allocate, construct an error, or return a retryable partial restore.

## 6. Typed plan and prepare

The restore plan has its own magic, lifecycle, live registry, and retained
tombstones. Its fixed fields are:

```text
magic, registry_next, owner,
lifecycle, expected_receiver_completed_updates, target_completed_updates,
baseline receiver native config and 14 option records,
target copied native config and 14 option records,
exact 14 parameter identities and value tensors,
exact 28 moment destination identities,
exact 28 owned stage identities,
exact appended I2 builder identity or NULL
```

There is no dynamic assignment table. The plan shell is the only allocation.
Once published, its address is never recycled. Terminal plans clear every
parameter/P1/value/moment/stage/builder/owner pointer and remain inert
tombstones for stale/cross-kind rejection and measured retained-control
accounting.

Prepare requires an initially null output and completes all validation before
the shared exclusion is acquired:

1. Authenticate the optimizer through the sole existing optimizer registry;
   require provider 2.0 identity, exactly 14 entries, valid storage, and exact
   idle state `busy == 0 && active_operation == NULL`. This receiver check is
   against its fixed provider fields; a request cannot attest source provider
   identity.
2. Require exact request/entry sizes; nonnegative signed-i64-bounded `R` and
   `U`; valid accepted target clip/schedule and option fields. The target may
   differ from the live receiver. Capture the live receiver configuration and
   options separately as the immutable precommit baseline.
3. Require live `completed_updates == R`; `R` may be less than, equal to, or
   greater than `U`.
4. Require absent gradients for all 14 exact parameters and revalidate each
   parameter/P1-handle identity. Capture its canonical value tensor and both
   exact live moment destinations.
5. Map each fixed entry's two owned stages to those destinations and call the
   exact owned-clone match operation with `FINITE` for `exp_avg` and
   `FINITE_NONNEGATIVE` for `exp_avg_sq`. Require all 28 sources and
   destinations distinct and cross-disjoint.
6. Allocate and fill one plan shell. Allocation failure leaves the output,
   optimizer, stages, counters, and all receiver bytes unchanged.
7. Fill the unpublished plan, including magic and `PREPARED` lifecycle. As the
   last recoverable action, call the existing static
   `et_o2_active_operation_acquire(optimizer, plan)`. Then link it into the
   restore-plan registry and publish the exact output. The link/output suffix
   allocates nothing and cannot fail.

Any failure before output publication releases the shared exclusion if it was
acquired, frees the unpublished shell, and leaves stage ownership with the
caller's stage cell. No prepare failure releases a stage.

Before building this native request, the source helper authenticates the
detached C2/O2 source provider identity and proves its copied logical config is
consistent with the target native config/options. Native code authenticates
only the live receiver's fixed 2.0 provider fields. These are distinct checks.

On success, the native plan holds the exact stage identities and shared
backreference. The source-level `o2-optimizer-restore-prepare-internal` passes a
private C output slot initialized to null. After C success it enters an
unobservable no-failure suffix: write that exact output-slot value into the
already-promoted plan cell, reread and compare it, then clear the already-
promoted 28-stage cell. The wrapper never trusts or transfers authority through
its returned value. No allocation, callback, handler installation, provider
call, failpoint, user code, or raise boundary exists between C success and the
three stores/checks. A failpoint may raise immediately before the C call or
only after this entire transfer suffix. Outside that transient suffix, a
nonempty stage cell means the parent owns all stages and has no native plan;
an empty stage cell means the exact plan cell/shared backreference owns them.
No other stable pair is accepted by cleanup.

## 7. Append and check

Append authenticates the typed live plan through its registry/magic/lifecycle,
requires `first_index == 14`, exact shared `owns(owner, plan)`, and a null
builder witness. The builder must already have been created with this exact plan
as authority and contain the 14 prefix appends. O2 constructs the fixed I2
append record on the stack from the plan's captured parameter/P1, moment, and
stage identities, sets `restore_authority = plan`, and invokes the fixed
restore-42 suffix boundary. Only after that boundary succeeds does it store the
exact builder identity in the plan.

Check is read-only and allocation-free. It reauthenticates the typed plan and
requires:

- exact shared owner/backreference and `PREPARED` lifecycle;
- optimizer provider, count, config, option, parameter/P1 handle, parameter
  value, and moment destination identities unchanged from the captured receiver
  baseline (not equal to the target unless they happened to match);
- live completed updates still equal captured `R`;
- gradients absent;
- all 28 stages still exact idle finite owned clones matching the moments; and
- a nonnull exact builder witness for which fixed restore-42 verify succeeds
  with the same plan authority, while it is `COMPLETE` and unprepared.

Copied update-clear plans, restore tombstones, restore plans owned by another
optimizer, generic builders, and builders appended by another restore plan all
fail before mutation. Update-clear prepare while restore is live and restore
prepare while update-clear is live both reject through the same existing shared
fields.

## 8. Commit and abort

The last recoverable operations are restore check followed by the authority-
taking restore42 prepare. The parent then seals. The restore42 checked commit
performs the first live tensor writes, clears pins, and retires the builder.

`et_o2_tr3_c_restore_commit_native_v1` receives only the canonical plan retained
by the sealed parent. Before the first write it performs a fixed,
nonallocating, nonraising authentication: restore-plan registry lookup by
address before dereference; exact magic and `PREPARED` lifecycle; registered
owner with fixed provider 2.0 and count 14; exact shared
`owns(owner, plan)`; completed updates still `R`; native config/options still
equal the captured receiver baseline; all retained parameter/P1/value/moment
identities exact; and the retained builder witness recognized by the private
restore42 terminal probe as `COMMITTED`. A builder tombstone marked `ABORTED`
is rejected. Any defect `_Exit(134)`. It then writes, in fixed order:

1. plan lifecycle `COMMITTING`;
2. optimizer global target native config;
3. the 14 target native option records in canonical index order; and
4. optimizer `completed_updates = U`.

The source wrapper then performs the one accepted assignment write of the
promoted copied logical config into optimizer slot 2. That store cannot allocate
or fail.

`et_o2_tr3_c_restore_finalize_v1` first fail-stop-authenticates the plan through
its typed registry, exact magic/`COMMITTING`, registered owner/provider/count,
shared `owns(owner, plan)`, target config/options, `completed_updates == U`, and
the exact builder witness still terminal `COMMITTED`. Before releasing anything it also
reauthenticates every stage as a genuine unborrowed/unpinned private clone with
the correct shape and value policy. It then:

1. releases all 28 stages in canonical order through the existing checked
   TR3-C owned-release path, nulling each plan slot only after success;
2. calls the existing static `et_o2_active_operation_release(owner, plan)`;
3. clears every retained parameter, P1 handle, value, moment, stage, builder,
   and owner pointer; and
4. writes lifecycle `COMMITTED` last.

Every operation is fixed, nonallocating, noncallback, and nonrecoverable. An
impossible stage-release or shared-ownership defect fail-stops.

The sole source-private commit-controls helper enforces the complete order:
authority-taking checked I2 copy commit/unpin and retirement, authenticated
native controls commit, optimizer slot-2 assignment, then no-failure stage
drain/finalize. The split native tail symbols and fixed I2 request symbols are
unbound and unexported; no G3-R path can call them.

Abort is callable only while `PREPARED`. The parent must first abort any I2
restore builder/child plan so every stage pin is gone. If the O2 plan has a
nonnull builder witness, abort requires the exact restore42 terminal probe;
before append the witness is null and no builder relation exists. The probe
must report terminal `ABORTED`; a `COMMITTED` child tombstone makes O2 abort
fail recoverably before mutation. Abort
recoverably authenticates the typed plan, shared ownership, receiver baseline,
retained identities, and all 28 releasable stages before mutation. It then
marks `ABORTING`, releases the stages through the checked path, clears the
shared backreference/busy fields, clears every retained pointer, and writes
`ABORTED`. It changes no optimizer tensor, config, option, or counter. A defect
after `ABORTING` fail-stops; foreign, wrong-kind, copied, terminal, wrong-
authority, or unretired-builder candidates fail recoverably before mutation.

## 9. Required evidence

Focused acceptance must include all of the following before implementation can
be integrated.

### I2 boundary

- Null, forged, stale, ordinary live tensor, ordinary clone, parameter value,
  borrowed owned clone, pinned owned clone, malformed destination, wrong shape,
  NaN/Inf owned source, and negative nonzero second-moment negatives at every
  one of the 14 positions; both signs of zero remain accepted.
- Ordinary builders of size 41, 42, and 43; restore creation with null
  authority, nonnull output, misaligned/overlapping output, and both create
  allocation failures; prefix overwrite/out-of-order/skip attempts; wrong
  parameter/P1 or destination identity; incomplete prefix; prefilled suffix;
  prepared/consumed builder; null/wrong/cross-plan authority; second suffix
  append; request size/alignment/range overflow and builder-shell/assignment
  alias; every source and destination duplicate/cross-alias class.
- Every generic set/prepare/commit/abort attempt on a restore builder, and every
  restore-specific prepare/commit/abort attempt on a generic builder. Preserve
  the accepted count-1 checked paths used by D2. Cover restore abort at prefix
  lengths 0 through 14, complete, and prepared; verify after prepare is a
  defined `INVALID_STATE` rejection.
- Full-control byte snapshots proving every rejected prefix/suffix append
  preserves the builder. Inject both create allocations and both child-copy
  prepare allocations; each prepare failure leaves phase `COMPLETE`, child
  null, endpoints unpinned, and all builder bytes unchanged. Counters prove
  match/prefix/suffix/verify/commit/abort/terminal-probe allocate nothing.
- Correct exact-42 append and verify, authority-taking prepare, checked abort,
  and checked commit. The exact prefix 14 must be bound through authenticated
  parameter/P1 identities to the optimizer's value tensors, not merely
  shape-compatible tensors. Retirement clears authority and payload pointers
  and preserves only exact `COMMITTED` or `ABORTED` disposition. Commit after
  builder abort and O2 abort after builder commit must reject before O2
  mutation.

### O2 participant

- Every request/entry size, reserved field, output-slot initial state and alias
  class, config/option bit, provider/count, `R`/`U`, present
  gradient, parameter/P1 identity, moment destination, stage ownership,
  finite/shape, source/destination alias, and allocation/publication prefix.
- Successful `R=0,U>0`, rewind `R>U`, and target config/options deliberately
  different from the receiver baseline; precommit byte/control snapshots prove
  `R` and every receiver byte/config/option unchanged.
- Separate source-provider corruption and live receiver-provider corruption;
  prove that the C request itself cannot attest source provider identity.
- Mutual exclusion in both orders with the real #119 update-clear plan; exact
  shared backreference checks on append/check/abort/finalize.
- Wrong first index, wrong builder kind/count/phase, another plan's authority,
  copied/cross-kind/stale/tombstone plans, append twice, and builder mutation.
- Abort before append, after append, and after I2 prepare/abort; child processes
  prove out-of-order pinned-stage abort and injected checked-release defects
  fail-stop rather than return a retryable partial cleanup.
- Instrumented sealed tail proving zero allocations, handlers, callbacks,
  provider/parser calls, error construction, or recoverable exits, with exact
  native-controls/logical-slot/finalize order.
- Exact category/code/operation assertions for every mapping in section 5,
  including explicit I2-result translation and literal `trainer-load-state!`.
- Source failpoints immediately before the C call and after the complete
  two-cell transfer, substituted wrapper return values, and all four stable
  stage-cell/plan-cell/backreference combinations. Prove no failpoint or raise
  boundary exists inside the C-success-to-cell-transfer suffix.
- Allocation counters proving append/check/commit/finalize are zero-allocation;
  child-process fail-stop evidence for forged/stale/cross-kind plan pointers
  reaching either sealed native operation; and terminal snapshots proving all
  retained parameter/value/moment/stage/builder/owner pointers are cleared.
- 1,024 and 8,192 prepare/abort and commit cycles with live-authority counts
  returning to baseline and a measured, documented tombstone slope.

### Packaging and regression

- Strict GCC and Clang native builds, ASan/UBSan/LSan, exact ordinary/feature
  defined and undefined symbols, no public export/string/binding changes, and
  gate-absent ordinary I2/O2 object parity.
- Single-registry linkage: exactly one feature-gated canonical `f32_tensor.c`
  object and one I2 bridge object in each TR3-C/G3-R private aggregate, never an
  ordinary plus feature tensor lineage.
- One object with both #119 update-clear and TR3-C restore features; predecessor
  O2, #119 adversarial, I2 coexistence, checked-I2, and fixed restore tests.
- No Python/PyTorch dependency in delivered paths and no full trainer, lease,
  source compositor, resume, or public completion claim.

## 10. Explicit non-goals

This proposal does not authorize implementation yet. It does not define the
orthogonal O2 source create-overlap hook owned by #119, the trainer lease, the
model/O2 Scheme compositors, C2 staging, D2 integration, the parent restore
machine, checkpoint bytes, or public `trainer-load-state!` installation. It
does not relax genuine native authentication into a trusted-source assertion.
