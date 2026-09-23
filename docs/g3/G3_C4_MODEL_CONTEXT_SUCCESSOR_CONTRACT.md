# G3-C4 model construction and native context successor contract

**Frozen contract only.** This successor is based on the accepted selective
integration `fc924df9df03c568421673f117c114981e8aa701`, tree
`e13e1c873b9e9d9aa77e088a8ff46fa0e261e3e8`. It narrows the earlier proposed
model/transport design at `0c67676` to the next dependency-complete unit: a real
P1/I2 prepared state, one pending/live/dead C4 model authority, an atomic seeded
P1/I2/C4-owner construction transaction, and the native C4 context/cache/active
call substrate. It authorizes no implementation, public API, package inventory,
forward schedule, persistence, or generation behavior. Restricted sampler source
and implementation are outside this contract.

## Grounded authority and missing seams

The contract composes these accepted implementations rather than replacing or
copying them:

- P1 owns the sole native identity context, token registry, Eshkol construction
  registry, and active construction. Native construction currently has numeric
  states 0 `OPEN`, 1 `SEALED`, and 2 `ABORTED`; its seal is one-shot.
- I2 owns the sole Eshkol construction registry, module-count rollback ledger,
  carrier registry, and error record. Its current construction route is fixed to
  the two-argument M3T preflight.
- The C4 owner owns the sole native C4 owner registry, the singleton
  `staged_c4_owner`, fourteen parameters, and states `OPEN`, `PREPARED`,
  `SEALED`, and `ABORTED`. Its accepted prepare/commit/abort functions already
  provide the native owner half of a construction transaction.
- The accepted I2 bridge provides
  `et_i2_private_g3c4_construction_available_v1` and
  `et_i2_private_g3c4_construction_parameter_preflight_v1`. It authenticates the
  exact staged owner, member parameter, handle, and whole-owner idle state, but no
  Eshkol caller selects it yet.
- A2 owns the sole cache, transaction, view, and read-borrow registries. C4 owns
  no cache implementation.
- The accepted C4 fixed-14 pin implementation owns the sole C4 pin lifecycle.
  It is reused unchanged.
- `m3-call-state` is the aggregate outer-call exclusion authority. A C4 source
  tuple reuses it and does not create an independent aggregate guard.

The P1 prepared split, C4-specific I2 route, C4 model registry, native C4
context registry, cache binding, and owner-active operations described below do
not exist at this base. They are new implementation obligations in this exact
dependency order.

## 1. Genuine P1 prepared construction

The additive private C surface is:

```c
int64_t et_p1_private_construction_prepare_v1(
    void *context, void *construction);
int64_t et_p1_private_construction_commit_prepared_v1(
    void *context, void *construction);
int64_t et_p1_private_construction_abort_prepared_v1(
    void *context, void *construction);
```

Native state 3 is `PREPARED`. Existing state values do not change. Prepare
accepts only the exact active state-0 construction after the existing owner,
process, and complete token-integrity preflight, then changes only 0 to 3. Commit
accepts only that exact active state-3 construction, changes 3 to 1, releases the
enrollment array, and clears `active_construction`. Prepared abort accepts only
that exact active state-3 construction, invalidates its enrolled identities,
changes 3 to 2, releases the enrollment array, and clears the active pointer.

The existing native `et_p1_private_construction_seal_v1` and
`et_p1_private_construction_abort_v1` remain state-0-only. Existing direct C and
M3T behavior therefore stays unchanged.

P1 appends these trusted Eshkol operations after the currently occupied slots
0..70; it must not reuse the E3 64..68 or TR3 69..70 slots:

```scheme
(module-construction-prepare-eval-internal! identity) ; slot 71
(module-construction-seal-prepared-internal! identity) ; slot 72
```

Prepare-eval performs the existing P1 construction-seal fallible work, plus the
new exact-tree `eval` transition, while abort remains valid: schedule
recomputation and equality, unattached module and handle scans, canonical-path
planning and checked promotion, module finalization, resulting-mode verification,
and native P1 prepare. Its source record becomes `prepared` and remains the active
construction.
Open and prepared shells remain unobservable through ordinary P1 operations.

Seal-prepared accepts only `prepared`, invokes the native prepared commit, marks
the source record `sealed`, and clears the active construction. It performs no
allocation, graph walk, provider call, promotion, or recoverable branch.
`module-construction-abort-internal!` admits both `open` and `prepared`: open uses
the existing native abort and prepared uses the prepared-specific abort. The
existing one-shot train-mode construction seal remains unchanged for M3T.

## 2. One C4 route through the existing I2 ledger

The trusted C4-specific operations are:

```scheme
(i2-g3c4-construction-begin-internal native-owner)
(i2-g3c4-construction-prepare-eval-internal! construction)
(i2-g3c4-construction-seal-prepared-internal! construction)
(i2-g3c4-construction-abort-internal! construction)
```

Begin first checks the accepted C4 bridge availability, then retains the exact
native owner in the same rooted I2 construction record used by the current
constructor. The record gains a closed route tag and owner field; callers cannot
supply a callback, raw preflight function, or generic owner selector. The M3T
begin path has its existing fixed route and no owner field. Both routes share one
active record, so they cannot overlap.

Every unbound registration, bound registration, prepare, and abort preflight on
the C4 route calls the accepted three-argument bridge with the retained exact
owner. The current M3T route continues to call only its two-argument bridge.

The I2 record states are `open`, `prepared`, `sealed`, and `aborted`.
Prepare-eval accepts only `open`, calls the P1 prepare-eval operation, and changes
only the I2 phase to `prepared`; it retains the root, memberships, carriers,
previous module registry/count, anchor, route, and owner so prepared abort is
complete. Seal-prepared accepts only `prepared`, calls the P1 prepared seal, then
clears the retained rollback fields and active I2 record in a non-failing tail.
Abort admits `open` and `prepared`, preflights the complete retained set, revokes
P1 identities first, restores the prior I2 registry/count, invalidates carrier
authority, and clears the active record. It never destroys native C4 storage.

No second I2 registry, f32 registry, error record, or parameter admission path is
permitted.

## 3. Sole C4 model authority and atomic seeded construction

`g3c4-model-registry` is one process-lifetime Eshkol root list. It is distinct
from the C2 M3T registry because C4 has a different profile and owner type, but it
is the only Eshkol authority for a C4 model. A C4 owner never masquerades as an
M3T `OWNER`.

Each exact 12-slot model entry is:

| Slot | Value |
|---:|---|
| 0 | `#f` before P1 begin, then the canonical P1 root shell |
| 1 | `c4-model` |
| 2 | `pending`, `live`, or `dead` |
| 3 | authenticated native C4 owner |
| 4 | `seeded` |
| 5 | immutable original four-word vector |
| 6 | immutable successor four-word vector |
| 7 | `#f`; reserved and grants no restore authority |
| 8 | canonical vector of fourteen P1 handles |
| 9 | exact I2 construction while pending, otherwise `#f` |
| 10 | exact active private call entry, otherwise `#f` |
| 11 | `diagnostic-c4` |

The model identity is `eshkol-diagnostic-byte-decoder-c4-v1`. Its accepted owner
order and shapes are `[4,4]` four times, `[4,8]`, `[8,4]`, six `[4]` vectors at
indices 6..9 and 11..12, `[256,4]`, and position `[4,4]`: fourteen unique
parameters, fifteen paths, one token/head tie, 1,192 f32 values, and 4,768 unique
bytes.

Trusted `(g3c4-model-create-seeded-internal seed)` is the sole transaction in
this contract. It executes under the existing aggregate guard and follows this
order:

1. Root and canonically read back a `pending` model entry, handle vector, and
   cleanup ledger before native enrollment.
2. Create the accepted seeded C4 owner, begin the C4 I2 construction with that
   exact owner, install the canonical root, register/bind all fourteen parameters
   and the one tie, validate the exact schedule, initialize indices 0..13, and
   capture the accepted original/successor words.
3. Invoke I2/P1 prepare-eval. Then invoke
   `et_g3c4_private_model_owner_prepare_seal_v1`.
4. Preflight every remaining cleanup and publication action.
5. Run the no-allocation tail: I2/P1 prepared seal, owner
   `et_g3c4_private_model_owner_commit_seal_v1`, clear slot 9, then set the
   already-rooted model entry `live`.

No observer authenticates a pending entry. Atomicity is publication atomicity
under serialized execution: every fallible action precedes the final tail, and
the live store occurs only after both P1/I2 and owner commits complete.

On any failure before the tail, preserve the first error and unwind in this
order: C4-I2 complete preflight and P1/I2 abort/rollback; native owner abort;
pending model entry to `dead`. Owner abort must never run while live P1 handles
still name its parameters. Dead entries retain only slots 0..2. A live C4 model
has process lifetime; this contract adds no model destructor or restored
constructor.

## 4. Native C4 context, cache, and owner link

The native context is the address-stable generator context already named by the
earlier C4 design. This contract retains these proposed private boundary names;
none exists in the current source:

```c
void *et_g3c4_private_generator_seed_v1(
    void *c4_owner, int64_t seed, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos);
void *et_g3c4_private_generator_rng_v1(
    void *c4_owner, void *rng, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos);
int64_t et_g3c4_private_generator_close_v1(void *ctx);
int64_t et_g3c4_private_call_acquire_v1(
    void *ctx, int64_t call_kind, int64_t budget);
int64_t et_g3c4_private_call_prepare_end_v1(void *ctx);
int64_t et_g3c4_private_call_finish_v1(void *ctx);
int64_t et_g3c4_private_call_abort_v1(void *ctx);
```

This contract freezes only their context ownership and active-call behavior.
Policy, RNG, numerical frame, and result semantics remain downstream and cannot
be implemented from this document alone.

`g3c4-registry` is the one process-lifetime Eshkol context/transport root. For
the bounded substrate in this contract, its exact generator entry is:

| Slot | Value |
|---:|---|
| 0 | exact generator shell |
| 1 | `generator` |
| 2 | `pending`, `live`, or `dead` |
| 3 | authenticated native context |
| 4 | `#f`; no parent result |
| 5 | exact live entry from `g3c4-model-registry` |
| 6 | `#f`; tokenizer linkage remains downstream |
| 7 | immutable normalized constructor policy, opaque to this contract |
| 8 | `#f`; result auxiliary storage remains downstream |
| 9 | exact active private call entry, otherwise `#f` |
| 10 | rooted constructor/call cleanup ledger |
| 11 | `diagnostic-c4` |

Construction roots a pending generator entry and cleanup ledger before invoking
the native constructor. It installs the returned context only after native
registry enrollment, then marks the entry live. Failure marks the entry dead and
retains slots 0..2. Ordinary lookup authenticates exact shell, kind, and live
state before reading slots 3..11. The private call entry is rooted in this same
registry and carries the exact generator/model linkage; it creates no native
CALL kind or second context.

One native C4 transport registry authenticates contexts before dereference. A
context record contains a registry link, kind/state/busy fields, the exact sealed
C4 owner, one embedded accepted C4 pin record, one owned A2 cache handle, and
active-call phase. Later fixed scratch or policy fields extend this record; they
do not create another authority.

A constructor first authenticates the exact owner through the existing C4 owner
registry, requires `SEALED` and `owner.active == NULL`, completes every context
allocation, and creates its cache with exactly:

```c
et_a2_kv_cache_create_v1(1, 1, 2, 4, 2, &cache, &error);
```

The tuple is layers 1, batch 1, KV heads 2, capacity 4, and head dimension 2.
Only after the cache and complete context are ready does the constructor enroll
the context as live. The context retains the A2 handle; it does not copy A2
storage, lists, or registry logic. The cache handle never enters an Eshkol shell
or public API.

Close authenticates the exact context and is idempotent only for its retained
dead context tombstone. A live close requires no active call, IDLE pins, and no
context-owned cache transaction/view/borrow. It destroys the A2 cache through
`et_a2_kv_cache_destroy_v1`, releases context-owned storage, marks the context
dead, and clears its model link. It never destroys or changes the sealed C4
owner or P1 parameters.

## 5. Active-call tuple

There is no raw `owner.active` setter. Only a registry-authenticated context may
change that field through the call operations above. One active invocation has
this exact tuple:

```text
m3-call-state == #t
generator entry slot 9 == call entry
C4 model entry slot 10 == the same call entry
native owner.active == authenticated context
ctx->pins is FULL with held_mask == 0x3fff
```

The Eshkol wrapper authenticates the live generator entry, its exact live C4
model entry, and their native cross-links before calling acquire. Acquire requires
the context live and idle, the exact owner sealed and idle, the cache without an
active lease, and IDLE pins. It invokes the accepted C4 pin begin. Only after pin
begin succeeds does it publish `owner.active = ctx`; the wrapper then stores the
same rooted call entry in model slot 10 and generator slot 9. A recorded acquired
mask permits cleanup of only stores completed by this call.

Every later context boundary reauthenticates the context, owner, model link,
active phase, `owner.active`, and FULL pins. `call_prepare_end` is read-only and
proves pin integrity plus absence of nested cache leases and remaining fallible
cleanup before a commit tail. Finish and abort end pins, clear native
`owner.active`, then let the wrapper clear model slot 10 and generator slot 9.
The enclosing existing `m3-call` cleanup clears the aggregate guard last. No
independent G3 guard, pin implementation, cache registry, model registry, or
context authority is allowed.

## Dependency order and proof obligations

Implementation work must remain split and reviewed in this order:

1. P1 native/Eshkol prepare/commit/abort-prepared with slot and legacy behavior
   proofs.
2. C4-specific I2 route over the accepted three-argument bridge and the sole I2
   registry.
3. Pending/live/dead C4 model authority and seeded atomic transaction.
4. Native context registry and fixed A2 cache constructor/close.
5. Active-call acquire/prepare-end/finish/abort with the existing C4 pins and
   shared aggregate guard.

Each implementation gate must cover allocation failure at every new allocation,
open/prepared abort, exact rollback baselines, wrong/foreign/stale/cross-owner
identities, model pending invisibility, one active construction, context/cache
constructor cleanup, close while busy, every pin-prefix failure, active-tuple
partial publication and reverse unwind, deterministic repeat output, exact symbol
deltas, and ASan/UBSan/LSan. Ordinary P1, M3T, I2 restore/R2, C4 owner/pins, A2,
C2, E3, and TR3 manifests and focused gates remain required affected contracts.

This freeze does not claim a forward schedule, cache transaction orchestration,
tokenizer link, restored model, sampler, continuation, persistence, public model
or generation operation, package integration, CI registration, or full G3
completion.
