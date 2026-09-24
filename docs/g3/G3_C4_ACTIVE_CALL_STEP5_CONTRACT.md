# G3-C4 Step 5 native active-call contract

Status: **root accepted; implementation remains prohibited until the shared
slot is released**. Root accepted the independently reviewed draft at commit
`56737e2ec15b90cad059db853bb47fd04bea5af7`, tree
`a13dd41fbbf5e9c28903ae80fa2f9ed5b2b8ce47`, with all six decisions in
Section 10.

This contract is based exactly on the accepted Step 4a implementation at
commit `58c59699365cce8d0ef3dac1a8141dc481712ec9`, tree
`3702989fa35d2e722e3ab52ef882fe9b590fc530`. It narrows Step 5 of
[`G3_C4_MODEL_CONTEXT_SUCCESSOR_CONTRACT.md`](G3_C4_MODEL_CONTEXT_SUCCESSOR_CONTRACT.md)
to the native active-call acquire, read-only end preflight, finish, and abort
operations. It consumes the actual Step 4a context, the accepted fixed-14 C4
pins, `owner.active`, and the canonical A2 cache lease API.

It does not add an Eshkol generator registry, generator shell, model or
generator call-entry publication, seed/RNG or normalized-policy ownership,
frame state, cache transaction orchestration, forward roles, sampler, result,
persistence, public surface, package registration, CI entry, or G3-S source.

## 1. Existing authorities and exact dependencies

The accepted sources already provide the complete native authority required by
this leaf:

- `g3c4_model_owner.c` owns the sole process-lifetime owner and context
  registries, the sealed owner record, its fourteen canonical parameters and
  handles, `owner.active`, the Step 4a cache, and the shared C4 error triple.
- `m3_call_f32_integration.c`, compiled with
  `ET_G3C4_NATIVE_PINS_PRIVATE`, provides exactly:

  ```c
  int32_t et_g3c4_model_pins_begin_internal(
      et_f32_parameter *const parameters[14],
      const void *const identities[14],
      et_g3c4_model_pins_internal *pins, et_f32_tensor_error *error);
  int32_t et_g3c4_model_pins_check_internal(
      const et_g3c4_model_pins_internal *pins,
      et_f32_tensor_error *error);
  void et_g3c4_model_pins_end_internal(
      et_g3c4_model_pins_internal *pins);
  ```

  Begin authenticates all fourteen parameter/identity pairs and their fixed C4
  shapes, acquires prefixes in index order, and drains any failed prefix in
  reverse order while preserving the first f32 error. Check is allocation-free
  and read-only. End is a preflighted no-fail drain in reverse index order and
  zeroes the complete embedded record.
- A2 exposes no idle-query or nonallocating lease preflight. Its only existing
  read-only proof that the exact cache is live and has neither an active
  transaction nor read borrow is:

  ```c
  int32_t et_a2_kv_cache_read_borrow_begin_v1(
      et_a2_kv_cache *cache, et_a2_kv_cache_read_borrow **borrow,
      et_kernel_error *error);
  int32_t et_a2_kv_cache_read_borrow_end_v1(
      et_a2_kv_cache_read_borrow **borrow, et_kernel_error *error);
  ```

  For the fixed one-layer Step 4a cache, begin allocates the borrow record and
  one-element key- and value-descriptor arrays. End allocates nothing. A
successful begin immediately followed by end has no persistent cache-content
or lease effect. An existing transaction (including one with an active view)
or read borrow rejects begin as K1
`invalid-argument/provider-rejected`. This contract adds no A2 function and
reads no A2 private structure from production code.

The accepted Eshkol model authority has a reserved active-call slot 10 but no
admission function that may write it. There is no accepted generator registry,
generator entry, call-entry type, or seed/RNG constructor policy. Those are
separate dependencies. This native leaf neither observes nor claims the full
Eshkol tuple.

## 2. Source and build ownership

The four operations remain in the sole owner/context translation unit under a
new nested feature macro, `ET_G3C4_ACTIVE_CALL_PRIVATE`. Enabling it requires
both `ET_G3C4_CONTEXT_PRIVATE` and `ET_G3C4_NATIVE_PINS_PRIVATE`; an invalid
macro tuple fails compilation. The active block includes `m3_call_pins.h` and
extends the existing context record. It creates no second owner, context, pin,
or cache implementation.

The production source tuple compiles:

1. `g3c4_model_owner.c` with all three private macros above;
2. `m3_call_f32_integration.c` with `ET_G3C4_NATIVE_PINS_PRIVATE`; and
3. canonical `a2_kv_cache.c`.

The pin translation unit is the accepted replacement for the ordinary
`m3t_f32_integration.c` object and must not be linked beside that ordinary
object. Step 5 introduces no new production direct-source inclusion: the
accepted pin replacement retains its existing inclusion of
`m3t_f32_integration.c`, while owner and A2 remain separate objects.

Without `ET_G3C4_ACTIVE_CALL_PRIVATE`, the Step 4a context-enabled owner object
must remain byte-identical to commit `58c5969`. Without
`ET_G3C4_CONTEXT_PRIVATE`, the ordinary owner object must also remain
byte-identical. The active object adds exactly the four definitions in Section
4. Relative to the Step 4a context object, its new undefined dependencies are
the three accepted pin functions and A2 read-borrow begin/end only.

## 3. Exact context extension and states

The Step 4a fields remain in their accepted order. When the active-call feature
is enabled, the context appends only:

1. one embedded `et_g3c4_model_pins_internal`;
2. signed 64-bit `call_kind`;
3. signed 64-bit `budget`; and
4. a 32-bit native acquired mask.

The existing `busy` field becomes the active-call lifecycle: `0` is `IDLE` and
`1` is `ACTIVE`. No prepared, frame, transaction, view, result, policy, or RNG
field is added. The acquired bits are exactly `PINS = 1`, `METADATA = 2`,
`BUSY = 4`, and `OWNER = 8`; a valid active call has mask `15`. An idle context
has busy, call kind, budget, and acquired mask all zero, a byte-zero pin record,
and does not appear in `owner.active`.

The mask records only native publication completed by the current acquire. It
is never an Eshkol cleanup ledger and cannot authorize writes to model slot 10,
generator slot 9, or the aggregate `m3-call` guard.

On the reviewed x86-64 ABI, the accepted Step 4a record is 48 bytes and the
accepted pin record is 1,360 bytes. The proposed fields make the context record
1,432 bytes after alignment. Retaining 1,024 closed contexts therefore retains
1,466,368 logical record bytes; 8,192 retain 11,730,944. This remains intentional
linear process-lifetime tombstone retention. The implementation gate must
measure and report the actual reviewed size and counts; it makes no flat-memory,
allocator-overhead, or resident-set-size claim.

## 4. Exact private boundary and argument domain

The only new source-private functions are the already proposed names:

```c
int64_t et_g3c4_private_call_acquire_v1(
    void *context, int64_t call_kind, int64_t budget);
int64_t et_g3c4_private_call_prepare_end_v1(void *context);
int64_t et_g3c4_private_call_finish_v1(void *context);
int64_t et_g3c4_private_call_abort_v1(void *context);
```

The accepted argument tuples are exactly:

| `call_kind` | Meaning | Accepted `budget` |
|---:|---|---:|
| 0 | manual prefill | 0 |
| 1 | manual decode | 0 |
| 2 | generation | 0 or 1 |

Every other pair rejects before A2 or pin work as C4
`invalid-argument/selector`. These values authorize no frame, result, sampling,
or numerical behavior in this leaf.

All four calls reset the existing C4 diagnostic and require external
serialization. A foreign pointer rejects as C4
`invalid-argument/identity` before candidate dereference. A dead context,
wrong lifecycle, inactive/active mismatch, cross-owner state, or second acquire
rejects as C4 `invalid-state/lifecycle`. Magic, kind, mask, pin, or backlink
corruption after registry admission rejects as C4 `internal/invariant` where it
can be diagnosed without invoking the pin no-fail end.

Exact A2 errors are captured in domain K1. Exact pin begin/check errors are
captured in domain I2. Rollback preserves the first error triple. No new error
domain, category, code, string buffer, registry, or allocation ledger is added.

## 5. Common authenticated native tuple

Every active boundary authenticates, in order:

1. exact context registry membership before dereference, fixed magic/kind, and
   `LIVE` lifecycle;
2. the retained nonnull owner link through the exact owner registry;
3. exact `SEALED` owner lifecycle and the nonnull retained A2 cache;
4. busy `ACTIVE`, acquired mask 15, and one accepted call-kind/budget pair;
5. exact `owner.active == context`; and
6. FULL pins through `et_g3c4_model_pins_check_internal`, including held mask
   `0x3fff` and all fourteen exact parameter/identity/value/view back-links.

Acquire uses the corresponding idle tuple: live context, authenticated sealed
owner, nonnull cache, `owner.active == NULL`, busy and mask zero, zero call
metadata, and a byte-zero embedded pin record. Multiple Step 4a contexts may
share the owner while idle, but only one can acquire it.

## 6. Cache-idle probe

The private cache-idle probe initializes a local borrow slot to null, invokes
A2 read-borrow begin on the exact retained cache, and, on success, immediately
ends that exact borrow. It calls no layer accessor and reads or writes no cache
content. Begin failure preserves the exact K1 error and leaves no lease. End is
allocation-free and must succeed for the just-created private handle under the
required external serialization. A nonzero end status is an impossible
dependency-invariant failure and aborts rather than returning with an unowned
live lease; it is not a recoverable publication path.

Acquire probes before pin begin. Prepare-end probes after the complete native
tuple and pins check. Abort probes before its first cleanup write. The probe may
fail at any of its three A2 allocations; acquire remains idle, while
prepare-end and abort retain the exact active tuple for retry.

Finish deliberately does not allocate or reprobe A2 after a successful
prepare-end. Its future trusted caller must invoke it in the immediate no-fail
tail after the separately contracted commit and before releasing external
serialization. This is why Eshkol call-entry publication and frame commit remain
a required later contract rather than being inferred here.

## 7. Acquire and partial-publication rollback

`call_acquire` performs all fallible work before native publication:

1. authenticate the idle context and owner and validate the argument pair;
2. complete the cache-idle probe;
3. invoke pin begin with the owner's exact fourteen parameters and handles;
4. record `PINS`, then store call kind/budget and record `METADATA`;
5. set busy `ACTIVE` and record `BUSY`; and
6. publish `owner.active = context` and record `OWNER` last.

There is no allocator, callback, mapper, A2 call, pin check, or recoverable
branch after pin begin in production. Owner publication is the native commit
write. On success the acquired mask is 15.

The focused build adds test-only failure boundaries after each recorded native
component. A synthetic failure reports C4 `internal/invariant`, snapshots that
first error, and unwinds only recorded components in exact reverse publication
order: clear owner backlink, clear busy, clear call metadata, then end pins.
Pin begin's own fourteen prefix failpoints remain authoritative for pin-prefix
rollback. Every failed acquire returns to the exact idle context/owner/cache and
parameter baseline and restores the first error after cleanup.

## 8. Prepare-end, finish, and abort

`call_prepare_end` authenticates the full tuple, checks the pins, and completes
the cache-idle probe. It is read-only in net effect: it does not change busy,
mask, call metadata, pins, owner backlink, cache contents, or registry state.
Success certifies that all currently available native fallibility precedes the
future commit tail. It does not record a prepared phase or itself authorize a
commit.

`call_finish` is valid only for the full tuple and only in the future trusted
caller's immediate no-fail tail after a successful prepare-end and commit. It
reauthenticates the tuple and performs the allocation-free pin check, then runs
this no-fail cleanup order:

1. end the complete pin record;
2. clear `owner.active`;
3. clear call kind, budget, busy, and acquired mask.

`call_abort` authenticates the full tuple, checks pins, and completes the
cache-idle probe before mutation. It then runs the same no-fail cleanup order.
Thus an active nested A2 transaction, transaction view, or read borrow rejects
abort without releasing pins or owner authority. After the nested lease is
ended, abort can be retried.

Finish and abort have the same bounded native drain because this leaf contains
no frame, transaction, result, or RNG state. Their later semantic distinction
is postcommit success versus precommit failure; the frame contract must add and
authenticate those states without changing this pin/backlink drain order.
Repeated finish or abort rejects as inactive. Context close is strengthened in
the active build to require the complete idle tuple in addition to Step 4a's
existing A2 destroy preflight. It remains idempotent only after a valid close
has produced the retained dead context tombstone.

## 9. Exact implementation files and gate

After root acceptance, implementation is limited to:

- `src/eshkol_transformer/g3c4_model_owner.c` — nested active-call block and
  context extension;
- `src/eshkol_transformer/g3c4_context_internal.h` — the four declarations;
- `tests/g3c4/test_active_call.c` — focused intrusive native test;
- `scripts/check-g3c4-active-call.py` — fast source/closure check;
- `scripts/test-g3c4-active-call.sh` — supported native gate;
- `native/g3c4_active_call_source_closure.txt` — exact closure; and
- one implementation note under `docs/g3/`.

No Eshkol source, model extension, A2 source/header, pin source/header, owner
layout header, installed header, Makefile, package manifest, CI file, or G3-S
file is modified. The intrusive test may source-compose the accepted pin and
owner/context implementations and build A2 in a separate test object to inspect
lease registries. Production uses separate canonical objects.

The gate must prove:

- byte-identical ordinary and Step 4a context owner objects when the new feature
  macro is absent, exact four-symbol definition delta when present, and exact
  five-symbol new dependency delta for pin begin/check/end and A2 borrow
  begin/end;
- strict C and C++ private-header compilation and invalid macro-tuple rejection;
- all accepted and rejected call-kind/budget pairs;
- foreign, stale, dead, wrong-lifecycle, second-context, owner-busy, and corrupt
  native tuples before mutation;
- all three A2 probe allocation failures on acquire, prepare-end, and abort;
- active A2 transaction, transaction-view, and read-borrow rejection on
  prepare-end and abort, followed by successful retry;
- all fourteen accepted pin-prefix failures, exact reverse pin drain, exact I2
  first-error preservation, and every native partial-publication rollback
  boundary;
- full tuple acquire, read-only prepare-end, finish, abort, repeated cleanup
  rejection, close-while-active rejection, and close after cleanup;
- unchanged owner lifecycle, all fourteen identities, values, gradients, cache
  contents, and registry baselines across every failure and cleanup path;
- multiple contexts sharing one owner while idle but exactly one active;
- actual context size plus 1,024/8,192 tombstone counts and logical retained
  bytes, with explicit linear-retention limitation;
- deterministic repeat output and normal plus ASan/UBSan/LSan execution; and
- the unchanged Step 4a context, owner, pin, A2, C2, model-authority, I2, P1,
  E3, and TR3 structural/focused gates as affected dependencies.

Docker, AOT, Eshkol call-entry publication, package/CI integration, numerical
execution, and G3-S remain outside this unit.

## 10. Root acceptance decisions

Root acceptance fixes all of these decisions for implementation after the
shared slot is released:

1. accept only `(0,0)`, `(1,0)`, `(2,0)`, and `(2,1)` as native
   call-kind/budget pairs;
2. use the existing transient A2 read-borrow begin/end pair as the sole
   read-only idle proof, accepting its three fallible allocations and retry
   semantics instead of inventing an A2 API;
3. append the accepted embedded 1,360-byte pin record and native mask/metadata
   to each process-lifetime context tombstone, accepting the reviewed 1,432-byte
   x86-64 record and its linear retention cost;
4. publish `owner.active` last, test every partial native publication boundary,
   and retain the acquired mask as native cleanup authority;
5. preserve pin-end-before-owner-clear order for finish and abort; and
6. keep generator/model call-entry publication, enforcement of the
   prepare-end-to-finish trusted tail, seed/RNG policy, and all frame semantics
   as explicit later dependencies because no accepted API currently owns them.

If any decision changes, this contract must be revised and reviewed again. The
broader successor text alone authorizes no Step 5 implementation.
