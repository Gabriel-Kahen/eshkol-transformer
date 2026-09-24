# G3-C4 Step 6A generator, RNG, and call-entry contract

**Review candidate; implementation is not authorized.** This
bounded successor starts from accepted Step 5 commit
`7d774d0255c589b0b65cdf47e5413ae5d0348ecb`, tree
`ae22dac77f979b2b41843fe4debdf50b89d0429c`. It resolves the ownership and close
gaps identified in the first Step 6 draft while retaining its Eshkol call-entry
publication protocol.

Root has accepted one decision from that draft: if native precommit abort cannot
restore the idle tuple, the process exits 134 before the existing `m3-call`
handler can clear the aggregate guard. This revision proposes exact resolutions
for the remaining two prerequisites: one native transport registry containing
both contexts and immutable RNG owners, and an exact live-generator close and
scrub transition. It adds no implementation, public facade, model schedule,
frame, sampler, tokenizer, result, persistence, package, CI, or G3-S work.

## 1. Existing authorities and unchanged boundaries

This successor composes these authorities without copying them:

- `m3-call-state` in `native/m3_model_extension.esk` is the sole aggregate
  outer-call exclusion root. `m3-call` publishes one busy value and clears it on
  normal return or an escaping exception. No C4 aggregate guard is added.
- `g3c4-model-registry` in `native/g3c4_model_extension.esk` is the sole Eshkol
  C4 model authority. Its accepted twelve-slot live entry reserves slot 10 for
  one exact active private call entry.
- The native C4 owner registry remains the only authority for the sealed owner
  and its fourteen parameters and identities. Their accepted shapes are four
  `[4,4]`, `[4,8]`, `[8,4]`, four `[4]`, `[256,4]`, two `[4]`, and position
  `[4,4]`: 1,192 f32 values and 4,768 unique bytes.
- The Step 4a/5 context owns the exact A2 cache
  `(layers=1,batch=1,kv-heads=2,capacity=4,head-dim=2)`. Step 5 owns the only
  native active-call lifecycle and fixed-14 pins. Its accepted tuples are
  `(0,0)`, `(1,0)`, and `(2,0|1)`.
- P1, I2, the f32 registry, A2 transaction/view/borrow registries, and the C4
  error record remain sole authorities in their domains.

Numeric initializer words, C2 training metadata, M3T initializer shells, N2
state, and equal vectors grant no G3 RNG authority. The RNG record below owns
state only; it is not a numerical provider or a substitute for G3-S.

## 2. One native transport registry

A new feature macro, `ET_G3C4_GENERATOR_PRIVATE`, is nested under the accepted
Step 5 tuple and fails compilation unless `ET_G3C4_CONTEXT_PRIVATE`,
`ET_G3C4_NATIVE_PINS_PRIVATE`, and `ET_G3C4_ACTIVE_CALL_PRIVATE` are all enabled.
Without the new macro, ordinary, Step 4a, and Step 5 objects remain byte-identical
to their accepted objects.

With the macro, the current context registry is generalized in place into one
native C4 transport registry. It contains context and RNG records and replaces,
rather than accompanies, `et_g3c4_context_registry`. There is no native RNG
registry, generator registry, secondary context list, copied owner list, or
lookup map.

Both record kinds begin with one embedded common header whose x86-64 layout is:

```c
struct et_g3c4_transport_header_internal {
  struct et_g3c4_transport_header_internal *registry_next;
  uint64_t magic;
  uint32_t kind;
  uint32_t state;
  uint32_t busy;
};
```

The header is 32 bytes after natural padding. Kind 1 is CONTEXT and kind 2 is
RNG. State 1 is LIVE and state 2 is DEAD. Admission traverses the single header
list comparing pointer identity before dereferencing a candidate. After a match,
it validates the kind-specific magic, kind, lifecycle, and zero/reserved fields.
A same-registry wrong kind is `invalid-argument/identity`; malformed admitted
headers are `internal/invariant`.

The Step 5 context layout is preserved through its acquired-mask field. The
generator build appends exactly:

```c
uint32_t generator_kind;       /* 1 for a generator, 0 for a base context */
uint32_t generator_ready;      /* 1 live-owned state, 0 absent/scrubbed */
int64_t generator_policy[6];   /* mode, temperature, k, p, max-new, eos */
int64_t generator_rng_words[4];/* version, seed, counter-low, counter-high */
```

The accepted Step 5 record is 1,432 bytes on x86-64. The 88-byte extension makes
every context record in the generator build exactly 1,520 bytes. A generator has
`generator_kind == 1`; that subtype is permanent across close. `generator_ready`
and all ten signed words are zero before publication and after close. A base
context has both generator fields and all words zero and cannot enter a generator
API.

An RNG record is exactly the 32-byte header followed by four signed 64-bit words,
for a 64-byte x86-64 record. Its kind-specific magic remains present after
release; its words are zero in a valid dead tombstone. Records are never moved,
freed, or reused after enrollment.

## 3. One Eshkol transport registry and exact entry shapes

The only new Eshkol authority is:

```scheme
(define g3c4-registry (vector '()))
```

It owns generator, private call, and RNG identities. No sibling generator, call,
or RNG registry is permitted. Every lookup authenticates exact `eq?` membership,
kind, lifecycle, and cross-links before reading native payloads. Shell tags and
equal vectors are inert; entries and shells are never reused.

Every entry has exactly twelve slots:

| Slot | Generator | Private call | RNG |
|---:|---|---|---|
| 0 | exact generator shell | exact private shell | exact RNG shell |
| 1 | `generator` | `call` | `rng` |
| 2 | `pending`, `live`, `dead` | `pending`, `live`, `dead` | `pending`, `live`, `dead` |
| 3 | authenticated native context | `#f` | authenticated native RNG record |
| 4 | `#f` | exact generator entry | `#f` |
| 5 | exact live C4 model entry | exact same model entry | `#f` |
| 6 | `#f` | `#f` | `#f` |
| 7 | immutable normalized policy | `#f` | immutable algorithm/word snapshot |
| 8 | `#f` | `#f` | `#f` |
| 9 | exact active call or `#f` | `#f` | `#f` |
| 10 | constructor ledger or `#f` | call ledger | creation ledger or `#f` |
| 11 | `diagnostic-c4` | `diagnostic-c4` | `diagnostic-c4` |

Generator slot 6 remains `#f` because tokenizer admission is downstream. Slot 8
remains `#f` because frame/result auxiliary state is downstream. An RNG slot-7
snapshot is exactly:

```text
#(g3.philox4x32-10.categorical-f32.v1 1 seed counter-low counter-high)
```

It is immutable observation retained with the exact authenticated native record;
the vector alone grants no authority. A dead entry retains only slots 0..2 and
clears every later slot. Pending entries are never admitted by ordinary
observers.

## 4. Exact private native APIs and RNG lifetime

The source-private constructor and close boundary is:

```c
void *et_g3c4_private_generator_seed_v1(
    void *c4_owner, int64_t seed, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos);
void *et_g3c4_private_generator_rng_v1(
    void *c4_owner, void *rng, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos);
int64_t et_g3c4_private_generator_close_v1(void *context);
```

The immutable RNG-owner boundary is:

```c
void *et_g3c4_private_rng_seed_v1(int64_t seed);
void *et_g3c4_private_rng_clone_v1(void *rng);
int64_t et_g3c4_private_rng_word_v1(void *rng, int64_t index);
int64_t et_g3c4_private_rng_release_v1(void *rng);
```

All functions reset and use the existing C4 error record. They add no error
domain, category, or code. A foreign pointer rejects before dereference as
`invalid-argument/identity`; a same-registry wrong kind does likewise. Nonrelease
use of an exact dead owner rejects as `invalid-state/lifecycle`. An invalid word
index or seed rejects as `invalid-argument/selector`. Allocation failure is
`internal/allocation`. Corrupt admitted state is `internal/invariant`.

`rng_seed` accepts only a nonnegative signed i64, allocates a complete 64-byte
record, stores `[1,seed,0,0]`, and enrolls it last. `rng_clone` authenticates an
exact live RNG, allocates a new record, copies all four words, and enrolls the
clone last. The source is never retained or mutated. `rng_word` authenticates a
live RNG and returns index 0..3; callers distinguish a valid zero from failure by
checking the existing last-error category, exactly as the C4 owner-word path
already does.

RNG owners are immutable. `rng_release` authenticates the exact record; a valid
dead record with zero words is idempotent success. A live release writes all four
words to zero and marks DEAD in one allocation-free, no-fail tail. Clones and
generators previously copied from the source remain valid. No operation advances
a counter, computes Philox, checks a sampling environment, or selects a token.

The trusted Eshkol owner operations are source-private and uninstalled:

```scheme
(g3c4-rng-create-seeded-internal seed)
(g3c4-rng-clone-internal rng)
(g3c4-rng-release-internal! rng)
(g3c4-generator-create-internal model config)
(g3c4-generator-close-internal! generator)
```

RNG create/clone root a pending entry and cleanup ledger before native allocation,
read both back canonically, capture all four checked words into slot 7, publish
`live`, and clear the ledger. Every failure after enrollment conditionally
releases an actually created native RNG, deadens the entry, clears the ledger,
and rethrows the first E1 error. Release of an exact dead Eshkol RNG is idempotent;
live release calls native release before retaining only slots 0..2.

These internal RNG operations provide an authenticated `:rng` authority without
adding a numerical provider or public RNG-from-seed operation. A later output
contract may create nonzero-counter snapshots only through a separately accepted
same-translation-unit helper after an authenticated prepared sampling result. It
must not add another registry or a caller-supplied word constructor.

## 5. Normalized policy and generator construction

The normalizer consumes one detached proper flat list with exactly eight
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
reject as `invalid-argument` before model or native work. The bit fields represent
binary32 by bit copy. Their sign/exponent/fraction validity is classified with
integer bit operations, without floating arithmetic or fenv dependence.
Temperature must be finite and positive; top-p must be finite and in `(0,1]`.
Greedy requires temperature bits `0x3f800000`, top-k 256,
and top-p bits `0x3f800000`. Categorical accepts the bounded controls above. EOS
normalizes from `#f` to native `-1`; otherwise it is 0..255.

Generator slot 7 stores exactly:

```text
#(diagnostic-c4 mode temperature-bits top-k top-p-bits max-new eos)
```

where mode is 0 greedy or 1 categorical. It retains no config list, source RNG
shell, native pointer, callback, cache handle, or mutable counter.

Both native constructors repeat all numeric policy checks, authenticate the C4
owner before dereference, require the accepted SEALED/idle owner, allocate the
complete 1,520-byte generator context, create the fixed A2 cache, fill the
extension, and enroll only the complete context. The seed form owns
`[1,seed,0,0]`. The RNG form independently authenticates the exact live native
RNG through the one transport registry, revalidates version 1 and a nonnegative
seed, admits every signed encoding of the two 64-bit counter halves including the
maximum exhausted sentinel, and deep-copies all four words; it retains no source
pointer. Neither constructor consumes a draw, pins parameters, writes `owner.active`,
changes model slot 10, or starts an A2 lease.

`g3c4-generator-create-internal` runs once under the existing `m3-call`. It
normalizes the complete config and authenticates the exact live twelve-slot C4
model. Before native allocation it roots and reads back a pending generator entry,
normalized policy, and cleanup ledger. It selects exactly one seed/RNG native
constructor, records the context, marks the entry live, and clears the ledger in
an allocation-free tail.

Every failure after pending enrollment marks the entry dead and clears its ledger
before rethrowing the first E1 error. If a native context exists, cleanup first
calls generator close and requires success or exits 134. If no context exists,
no close is attempted. No pending entry becomes observable.

## 6. Exact live-generator close and scrub

`g3c4-generator-close-internal!` runs under `m3-call`. It first authenticates the
exact generator entry. An exact dead entry is idempotent success without reading
cleared payload slots. A live entry must have a live generator-native context,
exact live model cross-link, slot 9 `#f`, model slot 10 `#f`, immutable normalized
policy, and no constructor ledger.

Native generator close authenticates the context through the one transport
registry and requires `generator_kind == 1`, `generator_ready == 1`, LIVE, the
complete Step 5 idle tuple, exact sealed owner with `owner.active == NULL`, and a
nonnull cache. Base `context_close` accepts only `generator_kind == 0` in this
build; it cannot bypass generator scrubbing. Generator close is ordered:

1. Complete all context, owner, active-call, pin, policy, and RNG-word preflight.
2. Call `et_a2_kv_cache_destroy_v1` on the exact cache. It allocates nothing,
   rejects a live transaction/view/borrow before mutation, and nulls the cache
   only on success. On failure the full generator record remains live for retry
   and the exact K1 error is preserved.
3. After successful cache destruction, write all six policy words and four RNG
   words to zero, then set `generator_ready = 0`.
4. Clear the owner link and mark the context DEAD last. `generator_kind` remains
   1, and all Step 5 active fields and pins remain zero.

There is no allocation, handler, callback, registry traversal, or recoverable
branch after successful A2 destruction. A direct repeated native close admits
only the exact retained dead generator tombstone with null owner/cache, ready
zero, all policy/RNG/active fields zero, idle pins, permanent generator kind 1,
and valid header/magic; it returns success without mutation. Corruption is
`internal/invariant`. A base context or RNG owner is wrong-kind identity.

Only after native success does the Eshkol wrapper clear generator slots 3..11,
including the model cross-link and normalized policy, then set slot 2 to `dead`.
Those stores form an allocation-free tail. A native close failure leaves the
Eshkol entry and model cross-link live and retryable. Generator close never
changes or destroys the model, native owner, parameters, source RNG, or any RNG
clone. It publishes no tokenizer/result behavior.

## 7. Aggregate guard and call-entry publication

The one trusted lexical wrapper remains:

```scheme
(g3c4-with-call-internal operation generator call-kind budget body ...)
```

It is a source macro over trusted forms, not a runtime callback. It enters
`m3-call` exactly once. Initial `#t` remains the aggregate admission token while
the call entry and native acquire are pending. The full tuple is:

```text
m3-call-state[0]          == call entry
generator entry slot 9   == call entry
C4 model entry slot 10  == call entry
call entry slot 4        == generator entry
call entry slot 5        == model entry
native owner.active      == generator context
native acquired mask     == 15
native pins held mask    == 0x3fff
native call kind/budget  == admitted pair
```

No helper admits a call merely because the aggregate cell is truthy.

Before acquire, the wrapper authenticates all generator/model/native cross-links,
idle cells, tuple, and policy; roots and reads back a pending call entry and fixed
cleanup ledger; and installs its inner cleanup handler. Publication is:

1. native `call_acquire` publishes pins, metadata, busy, and owner backlink;
2. record native acquisition in the Eshkol ledger;
3. store the call entry in model slot 10 and record it;
4. store it in generator slot 9 and record it;
5. mark the call entry live and record it;
6. upgrade `m3-call-state[0]` from `#t` to the exact call entry last.

A successful tail calls native `call_prepare_end`, executes only a later accepted
infallible frame/result/cache commit, and immediately calls native `call_finish`.
No allocation, handler installation, callback, observer, or recoverable branch
may intervene. Nonzero postcommit finish exits 134 and never enters precommit
abort. Successful finish clears model slot 10, generator slot 9, deadens the call,
clears its ledger, and returns to `m3-call`, which clears the aggregate guard last.
Until a frame contract exists, the only development witness uses an empty commit
and claims no generation behavior.

On precommit failure, the inner handler preserves the first E1 error and calls
native `call_abort` while the complete published Eshkol tuple remains. A nonzero
abort exits 134, as root accepted; it never escapes to the outer handler with
native state active. Successful abort clears model slot 10, generator slot 9,
deadens the call, clears its ledger, and rethrows. The outer `m3-call` handler
clears the aggregate guard last. Partial publication uses the same recorded mask
and never clears another call's store.

## 8. Exact G3-S boundary

RNG seed, clone, word observation, release, generator deep copy, and scrub are
ownership operations over four signed integers. They invoke no K1 provider,
Philox round, float operation, fenv inspection, logits access, sampling policy,
or counter successor. They can therefore be implemented and reviewed without
G3-S after this contract is accepted.

No operation in Step 6A may advance the counter, manufacture a nonzero-counter
successor from caller words, test categorical exhaustion for a requested draw,
validate the deterministic sampling environment, or select a token. Those
operations require the accepted G3-S provider implementation and later G3-G
transaction contract. Categorical policy may be normalized and retained now,
but no categorical execution or determinism-availability claim follows. G3-S
remains on its existing hold.

## 9. Tests, failure cuts, and retention

After root acceptance, the implementation gate must prove:

- exact object identity for ordinary, Step 4a, and Step 5 builds without the new
  macro; exact generator-build definitions, dependencies, depfiles, and source
  closure; invalid macro tuples reject;
- one native registry only, context/RNG wrong-kind and foreign-pointer rejection
  before dereference, magic/state corruption, dead idempotence, and no second
  owner, context, RNG, pin, cache, or error authority;
- every native allocation cut: generator context plus all fixed A2 cache
  allocations, RNG seed allocation, RNG clone allocation, and every partial
  constructor cleanup with first-error preservation;
- every Eshkol shell/vector/list/ledger allocation cut before and after pending
  enrollment, exact pending-to-dead cleanup, canonical readback, and no pending
  observer;
- seed 0 and signed-i64 max, negative/inexact/overflow rejection; all four checked
  words; clone independence; source release before and after generator creation;
  repeated release; copied/forged/dead/cross-kind/cross-aggregate identities;
- all policy syntax and scalar boundaries, both modes, canonical greedy controls,
  EOS `#f`/0/255, max-new 0/1, and rejection of initializer/C2/list words as RNG;
- generator close with each active A2 transaction, transaction view, and read
  borrow; exact retry after ending the lease; cache destruction before scrubbing;
  all ten words zero, owner/cache cleared, permanent subtype, native and Eshkol
  idempotence, and model/parameters/source RNG unchanged;
- every call publication boundary, full exact aggregate tuple, normal empty-commit
  finish, all precommit failures, accepted exit-134 subprocesses for abort cleanup
  failure, and exit-134 postcommit finish defects;
- unchanged Step 5, Step 4a, owner, pins, A2, C2, model-authority, I2, P1, E3,
  and TR3 affected gates; deterministic repeat and ASan/UBSan/LeakSanitizer;
- fresh supported Ubuntu 22.04/LLVM 21 evidence with exact source/runner/object
  manifests. No runtime build is part of this contract review.

The generator build's native tombstone sizes are fixed and reported honestly:

| Record | 1,024 tombstones | 8,192 tombstones |
|---|---:|---:|
| 1,520-byte context | 1,556,480 bytes | 12,451,840 bytes |
| 64-byte RNG | 65,536 bytes | 524,288 bytes |

The gate runs 1,024 and 8,192 successful seed-generator create/close cycles and
RNG seed/release and clone/release cycles. It proves A2 live-cache counts and
bytes return to baseline while context and RNG registry tombstones increase
linearly by the exact counts above. RNG-source clones and generators survive
source release. It separately runs 1,024 and 8,192 completed call cycles and
reports Eshkol entry/shell/vector counts and logical retained bytes for generator,
RNG, and call tombstones, plus peak live native/cache bytes. It makes no flat-RSS
or allocator-overhead claim. No arbitrary cap, entry reuse, weak registry, or
unreviewed finalizer may disguise cumulative retention.

Acceptance of this revision would authorize only a later separately reviewed
Step 6A ownership/publication implementation. It would not authorize public
`transformer.generation`, tokenizer linkage, frames, numerics, sampling, results,
persistence, packaging, CI publication, G3-S, or full G3 completion.
