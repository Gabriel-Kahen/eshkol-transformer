# G3-C4 Step 6 generator and call-entry dependency contract

**Review candidate; implementation is not authorized.** This bounded successor
starts from accepted Step 5 commit
`7d774d0255c589b0b65cdf47e5413ae5d0348ecb`, tree
`ae22dac77f979b2b41843fe4debdf50b89d0429c`. It specifies the next Eshkol
generator/model call-entry publication, its linkage to the existing aggregate
`m3-call` guard, and the normalized seed/RNG constructor boundary. It adds no
source implementation, public facade, model schedule, frame, sampler, tokenizer,
result, persistence, package, CI, or G3-S work.

This contract is intentionally dependency honest. The call-entry protocol is
fully specified below, but three prerequisite decisions remain unresolved in the
current source: there is no accepted authenticated G3 RNG owner for the proposed
RNG constructor; native abort can fail while the existing aggregate guard's
outer handler is about to clear itself; and the accepted context close has no
RNG/policy scrubbing transition. Section 8 gives the exact disposition required
before implementation. No subset may be implemented around any gap.

## 1. Existing authorities and exact boundary

This successor composes these authorities without copying them:

- `m3-call-state` in `native/m3_model_extension.esk` is the sole aggregate
  outer-call exclusion root. `m3-call` checks slot 0 for false, publishes `#t`,
  and clears it on normal return or an escaping exception. No C4 guard is added.
- `g3c4-model-registry` in `native/g3c4_model_extension.esk` is the sole Eshkol
  C4 model authority. Its exact twelve-slot live entry is already accepted;
  slot 10 is reserved for one exact active private call entry and is currently
  always `#f`.
- The model's native owner is admitted only by the existing C4 owner registry.
  Its fourteen canonical parameters and identities are the accepted C4 topology:
  four `[4,4]`, `[4,8]`, `[8,4]`, four `[4]`, `[256,4]`, two `[4]`, and position
  `[4,4]`. The fourteen values total 1,192 f32 elements and 4,768 unique bytes.
- The Step 4a/5 context registry owns the only native C4 context and its exact
  A2 cache `(layers=1,batch=1,kv-heads=2,capacity=4,head-dim=2)`. Contexts retain
  the exact sealed owner and have process-lifetime native tombstones.
- Step 5 owns the only native active-call lifecycle. Its accepted tuples are
  `(0,0)` manual prefill, `(1,0)` manual decode, and `(2,0|1)` generation. A full
  active tuple has pins mask `0x3fff`, acquired mask 15, and
  `owner.active == context`.
- The P1 and I2 registries, C4 fixed-14 pins, A2 registries, and existing C4
  error record remain sole authorities for their domains.

There is currently no `g3c4-registry`, generator entry, call entry, G3 RNG
entry, accepted sampling RNG native owner, constructor policy normalizer, or
Eshkol observer that may write model slot 10. Numeric initializer words, C2
training RNG metadata, M3T initializer shells, and N2 dropout state grant no G3
RNG authority.

## 2. One Eshkol transport registry and exact entry shapes

The only new Eshkol authority is one process-lifetime root:

```scheme
(define g3c4-registry (vector '()))
```

It owns generator, private call, and eventual G3 RNG identities. No sibling
generator registry, call registry, RNG registry, lookup table, weak map, or
copied model registry is permitted. Every lookup authenticates exact `eq?`
membership, kind, lifecycle, and cross-links before reading native payloads.
Shell tags and equal vectors are inert. Entries and shells are never reused.

Every entry has exactly twelve slots, preserving the accepted G3-T transport
layout:

| Slot | Generator | Private call | Eventual RNG |
|---:|---|---|---|
| 0 | exact generator shell | exact private shell | exact RNG shell |
| 1 | `generator` | `call` | `rng` |
| 2 | `pending`, `live`, `dead` | `pending`, `live`, `dead` | `pending`, `live`, `dead` |
| 3 | authenticated native context | `#f` | authenticated native RNG owner |
| 4 | `#f` | exact generator entry | `#f` |
| 5 | exact live C4 model entry | exact same model entry | `#f` |
| 6 | `#f` | `#f` | `#f` |
| 7 | immutable normalized policy | `#f` | immutable logical words snapshot |
| 8 | `#f` | `#f` | `#f` |
| 9 | exact active call or `#f` | `#f` | `#f` |
| 10 | rooted constructor ledger or `#f` | rooted call ledger | rooted release ledger or `#f` |
| 11 | `diagnostic-c4` | `diagnostic-c4` | `diagnostic-c4` |

Slot 6 remains `#f`: tokenizer admission is a later dependency and this unit
must not imply it. Slot 8 remains `#f`: frame/result auxiliary state is also
downstream. A dead entry retains only exact slots 0..2; every later slot is
cleared. Pending entries are never admitted by ordinary observers.

The eventual RNG row is frozen only as an authority shape. Its native owner and
constructor/release API are unresolved in Section 8 and therefore may not be
implemented by this unit.

## 3. Normalized policy and constructor input

The Eshkol normalizer consumes one detached proper flat list with exactly eight
key/value pairs:

```text
:profile           diagnostic-c4
:sampling          greedy | categorical
:temperature-bits  exact integer 0..4294967295
:top-k             exact integer 1..256
:top-p-bits        exact integer 0..4294967295
:max-new-tokens    exact integer 0..1
:eos               #f | exact integer 0..255
exactly one of      :seed nonnegative exact signed-i64
                    :rng exact live same-registry RNG shell
```

Unknown, duplicate, missing, improper, cyclic, overlong, or non-exact values
reject as `invalid-argument` before model or native work. The bit fields are
binary32 bit patterns, never ordinary real narrowing. Temperature must decode
to a finite value greater than zero; top-p must decode to a finite value in
`(0,1]`. Greedy requires temperature bits `0x3f800000`, top-k 256, and top-p bits
`0x3f800000`; inactive noncanonical controls reject instead of being ignored.
Categorical accepts the bounded ranges above. EOS normalizes from `#f` to native
`-1`, otherwise it remains 0..255. Determinism is mandatory.

The immutable policy stored in generator slot 7 is exactly:

```text
#(diagnostic-c4 mode temperature-bits top-k top-p-bits max-new eos)
```

where mode is 0 for greedy or 1 for categorical. It contains no source config
list, model shell, RNG shell, callback, native pointer, cache handle, or mutable
counter. The selected seed or RNG identity is carried separately through the
constructor transaction and must be copied into the context's owned RNG state.
The generator retains no source RNG owner after successful construction.

A seed is a nonnegative exact signed-i64 and constructs logical words
`[1,seed,0,0]`. An RNG source must be the exact live same-registry `rng` entry,
whose authenticated native owner represents four logical signed-i64 words
`[1,seed,counter-low,counter-high]`. The maximum unsigned 128-bit counter is a
valid exhausted state and is copied unchanged. Construction consumes no draw.
Greedy, manual calls, and zero-budget generation also consume no draw. This
contract defines no sampling or counter advance.

## 4. Private constructor boundaries

The earlier reviewed successor names remain the candidate native boundary:

```c
void *et_g3c4_private_generator_seed_v1(
    void *c4_owner, int64_t seed, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos);
void *et_g3c4_private_generator_rng_v1(
    void *c4_owner, void *rng, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos);
```

These are not accepted for implementation until Section 8 is resolved. If
accepted, both must use the existing context allocator, context registry, owner
registry, and A2 constructor. They must not wrap or call the public
`et_g3c4_private_context_create_v1` after separately allocating another context,
and must not create another native context or owner registry. The seed form owns
fresh `[1,seed,0,0]` state. The RNG form independently authenticates the exact
native RNG owner and deep-copies its words; it retains no source pointer.

Both constructors repeat the exact numeric policy checks, authenticate the C4
owner before dereference, require its accepted `SEALED`/idle state, allocate all
context/RNG policy storage, create the fixed A2 cache, and enroll only the
complete context. Every partial allocation is freed before return while the
first C4/K1 error is preserved. They do not pin parameters, write
`owner.active`, change model slot 10, consume RNG, or start an A2 transaction.

The Eshkol private constructor is one trusted operation:

```scheme
(g3c4-generator-create-internal model config)
```

It runs once under the existing `m3-call` macro. It normalizes the complete
config and authenticates the exact live twelve-slot C4 model before native
construction. Before any native allocation it roots and canonically reads back
the pending generator entry, normalized policy, and cleanup ledger. It then
selects exactly one native seed/RNG constructor, records the returned context,
marks the entry `live`, and clears its ledger in a no-allocation tail.

Every failure after pending-entry enrollment marks that entry dead and clears
its ledger before rethrowing the first E1 error. If a native context was created,
cleanup first calls the existing context close path and requires success or
terminates with exit 134 as an implementation invariant. If no native context
exists, no close is attempted. No pending entry becomes observable. Ordinary
live-generator close remains unresolved until the RNG-storage extension and its
scrubbing rule are accepted; this is an implementation hold in Section 8.

## 5. Aggregate guard and exact call tuple

One trusted lexical wrapper is proposed:

```scheme
(g3c4-with-call-internal operation generator call-kind budget body ...)
```

It is a source macro over trusted forms, not a runtime callback or public
operation. It enters the existing `m3-call` exactly once. The initial `#t` in
`m3-call-state` remains the aggregate admission token while construction of the
private call entry and native acquisition are pending. After complete
publication, slot 0 is upgraded from `#t` to the exact call entry. It is never
cleared by a C4 helper; the enclosing `m3-call` clears it last.

The full Eshkol/native active tuple is exactly:

```text
m3-call-state[0]          == call entry
generator entry slot 9   == call entry
C4 model entry slot 10   == call entry
call entry slot 4        == generator entry
call entry slot 5        == model entry
native owner.active      == authenticated generator context
native acquired mask     == 15
native pins held mask    == 0x3fff
native call kind/budget  == the admitted pair
```

No helper admits a call solely because the aggregate cell is truthy. Every
boundary checks the exact call identity and all generator/model cross-links.
No second aggregate boolean, nested `m3-call`, raw owner-active setter, or native
CALL kind is permitted.

## 6. Publication, success, and rollback order

Before native acquire, the wrapper authenticates the generator shell and exact
live generator/model entries, checks generator slot 9 and model slot 10 are
`#f`, checks their native owner/context cross-link, validates the call tuple,
allocates and roots a pending call entry plus a fixed cleanup ledger, and reads
both back canonically.

Publication is ordered:

1. Invoke `et_g3c4_private_call_acquire_v1`; its own commit publishes pins,
   metadata, busy, and `owner.active` in the accepted Step 5 order.
2. Record native acquisition in the rooted Eshkol ledger.
3. Store the exact call entry in model slot 10 and record that store.
4. Store the same entry in generator slot 9 and record that store.
5. Mark the call entry `live` and record publication.
6. Upgrade `m3-call-state[0]` from `#t` to the exact call entry last.

The body may use only separately accepted trusted call/frame operations. This
contract supplies no numerical role, frame state, cache transaction, result,
sampling, or public prefill/decode/generate loop.

The successful tail must call native `call_prepare_end`, execute only a later
accepted infallible frame/result/cache commit, and immediately call native
`call_finish` without allocation, handler installation, callback, observer, or
recoverable branch between prepare and finish. A nonzero finish status after
commit is an invariant defect and terminates with exit 134; it never enters the
precommit abort/rethrow path. Successful finish then clears model slot 10,
clears generator slot 9, marks the call dead, clears the call ledger, and returns
to `m3-call`, which clears the aggregate guard last. Until a frame contract
provides that infallible middle commit, the only valid development witness has
an empty commit and claims no generation behavior.

On a body failure before the commit tail, the inner call handler preserves the
first E1 error and performs:

1. If native acquisition completed, call native `call_abort` while the complete
   Eshkol tuple remains published.
2. If abort returns nonzero, terminate with exit 134. It must not let the error
   escape to the outer `m3-call` handler because that handler would clear the
   sole aggregate guard while Step 5 intentionally retains the native active
   tuple for retry.
3. Clear model slot 10 if recorded, then generator slot 9 if recorded.
4. Mark the call entry dead, clear its ledger, and rethrow the preserved error.
5. The enclosing `m3-call` handler clears the aggregate guard last.

Partial publication failure uses the same ledger and sequence. It clears only
stores recorded by this call. Native abort precedes all Eshkol active-cell
clears. A failure before native acquisition only deadens the pending call entry.
No cleanup error replaces the first recoverable error; an inability to restore
the cross-layer idle tuple is fail-stop.

## 7. Exact tests and retention gates

The implementation gate, after Section 8 is accepted, must prove:

- byte-exact preservation of the ordinary, Step 4a, and Step 5 native objects
  without the new constructor feature macro, plus exact new definition and
  undefined-dependency manifests on the supported compiler;
- exact twelve-slot generator/call/RNG shapes, pending invisibility, identity
  authentication before payload access, wrong/foreign/copied/dead/cross-model
  rejection, and dead entries retaining only slots 0..2;
- every malformed config class, both accepted modes, all scalar boundaries,
  canonical greedy controls, seed limits, EOS `#f`/0/255, max-new 0/1, and
  rejection of M3T initializer/C2 metadata/numeric word lists as RNG authority;
- allocation failure at every list/vector/ledger/native context/cache/RNG
  construction point with exact registry and guard baselines restored;
- model slot 10 and generator slot 9 failures at every recorded publication
  boundary, exact native acquire rollback, and the full tuple after success;
- nested M3/C4 exclusion through the one aggregate root, a busy second model,
  a second generator sharing one idle model, and no independent guard state;
- normal finish and all precommit failures, including all three A2 allocation
  failures in native abort. The first abort failure must take the explicit
  fail-stop subprocess path and must never return with `m3-call-state == #f`
  while native state remains active;
- prepare-end immediately followed by the empty development commit and finish,
  with source/IR evidence that no allocation or recoverable branch intervenes;
- unchanged Step 5, Step 4a, owner, pins, A2, C2, model-authority, I2, P1, E3,
  and TR3 affected gates; deterministic repeat and ASan/UBSan/LeakSanitizer;
- fresh supported Ubuntu 22.04/LLVM 21 evidence with exact source, runner,
  object, defined/undefined, and dependency manifests.

Retention is intentionally measured rather than hidden. Call-entry evidence
must report exact Eshkol vector/shell counts and logical retained bytes for 1,024
and 8,192 completed call cycles. Private call entries remain process-lifetime
strong-registry tombstones, and the gate must show their exact linear slope.
Generator retention cannot claim successful create/close cycles until the third
Section 8 decision freezes live-generator close and RNG/policy scrubbing. After
that decision is incorporated into a reviewed revision, report exact native
context/RNG record sizes, Eshkol vector/shell counts, logical retained bytes, peak
live native/cache bytes, and cumulative tombstone bytes for 1,024 and 8,192
successful generator create/close cycles. Make no flat-RSS or allocator-overhead
claim. No arbitrary lifetime cap, entry reuse, weak registry, or unreviewed
finalizer may be introduced to make the measurement appear bounded.

## 8. Unresolved decisions and implementation hold

Three decisions require root acceptance before any implementation from this
contract:

1. **Authenticated RNG carrier.** The proposed
   `et_g3c4_private_generator_rng_v1(void *rng, ...)` has no accepted producer,
   native kind, registry admission, clone, release, or error/lifetime contract in
   the current tree. G3-S is a held, registry-free numerical provider contract
   and does not fill this ownership gap. Root must either accept a single-registry
   native RNG owner compatible with the eventual RNG row above, or revise the
   constructor ABI in a separately reviewed contract. Numeric words alone are
   not an acceptable substitute.
2. **Abort-cleanup disposition.** Step 5 correctly leaves the full native tuple
   active when its fallible A2 idle probe fails. The current `m3-call` exception
   handler unconditionally clears the aggregate cell. This contract proposes an
   inner handler that treats unsuccessful native abort as exit 134, so inconsistent
   authority never returns to Eshkol. Root must explicitly accept that fail-stop
   rule or require a separately reviewed retryable aggregate-guard protocol.
3. **Live-generator close and scrubbing.** Existing context close destroys the A2
   cache and deadens the native context but knows no generator RNG/policy extension.
   Root must accept an exact native and Eshkol close transition, including idle
   admission, order of cache destruction, RNG/policy zeroing, model/generator
   cross-link removal, dead-tombstone contents, idempotence, first-error behavior,
   and the 1,024/8,192 create/close retention gate. Until then, successful live
   generators and their caches cannot be claimed released and neither constructor
   may be implemented.

The contract must be revised if any disposition changes the APIs, registry
shape, publication order, or cleanup semantics. G3-S remains on its existing
hold. Acceptance of this document alone authorizes no sampler, RNG numerical
provider, public generation operation, frame implementation, or G3-S work.
