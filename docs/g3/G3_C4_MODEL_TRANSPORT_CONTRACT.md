# G3-C4 model and transport seam contract

**Proposed exact contract; implementation requires root acceptance.** This file
replaces the unresolved model/transport placeholders in
[the C4 profile proposal](G3_C4_PROFILE_PROPOSAL.md). It is derived from the
accepted integration tree `637cec09e00d6f9c2414e051836ab50665709606`, the
checked-promotion runtime contract at `222cad3aac68ddf48d09c1cdf322fa4c4e7b8296`
over implementation `714d20fea64d97286dde34b841fc0146bfdcc40b`, and the
pending SHARED-R2 implementation based on `19f404cf21632944e1f2d5d4959e1240f9a79f5f`.
It authorizes no implementation by itself. The restricted sampler source remains
outside this work.

The earlier proposal at `afb48dcb4770428d286bbe141b11e61e4d5328ba`
correctly fixes the C4 profile and public behavior, but it is not an exact seam
freeze. The current native construction bridge recognizes only M3T's singleton
`staged_owner`; P1/I2 seal combines fallible finalization with irreversible
publication; and P1 construction modules start in `train` while ordinary
`module-eval!` rejects unpublished construction shells. The additions below are
therefore required, not optional implementation details.

## Fixed model identity and Eshkol record

The model identity is `eshkol-diagnostic-byte-decoder-c4-v1`, profile
`diagnostic-c4`. It has the existing 15 logical paths and 14 unique parameters,
with one tie between `head/weight` and `token_embedding/weight`. Pin/owner order
is K, attention output, Q, V, FFN down, FFN up, norm1 beta/gamma, norm2
beta/gamma, tied token/head, final beta/gamma, position. Shapes are:

```text
[4,4] [4,4] [4,4] [4,4] [4,8] [8,4]
[4] [4] [4] [4] [256,4] [4] [4] [4,4]
```

That is 1,192 f32 values and 4,768 unique bytes. Every identity and storage is
distinct except the one named tie. A C4 owner is never an M3T `OWNER` and never
passes M3T's C2 profile admission.

`g3c4-model-registry` is one process-lifetime root list. Each exact 12-slot entry
is:

| Slot | Value |
|---:|---|
| 0 | `#f` before P1 begin, then the canonical P1 root-module shell |
| 1 | `c4-model` |
| 2 | `pending`, `live`, or `dead` |
| 3 | authenticated native C4 owner |
| 4 | `seeded` or `restored` |
| 5 | seeded original four-word vector, otherwise `#f` |
| 6 | seeded successor four-word vector, otherwise `#f` |
| 7 | `#f` (reserved; grants no R provenance authority) |
| 8 | canonical vector of 14 P1 handles |
| 9 | exact I2 construction while pending, otherwise `#f` |
| 10 | exact active private call entry, otherwise `#f` |
| 11 | `diagnostic-c4` |

The pending record, origin metadata, handle vector and cleanup ledger are promoted
and read back canonically before native ownership is staged. After native owner
creation and I2/P1 begin, the canonical root is installed in slot 0 through a
checked store while the transaction is still abortable. Registry lookup by exact
shell precedes every slot read from a live model. Pending entries are reachable
only through the rooted construction cleanup ledger. Dead
entries retain slots 0..2 only. Successful models have P1's process lifetime;
there is no model destructor and generator close does not release parameters.

## Exact native C4 owner surface

The following declarations live in a source-private header and are never
installed, boxed, serialized, or exposed through a public Eshkol import:

```c
void *et_g3c4_private_model_owner_create_seeded_v1(int64_t seed);
void *et_g3c4_private_model_owner_create_restored_v1(void);
void *et_g3c4_private_model_owner_parameter_v1(
    void *owner, int64_t index);
int64_t et_g3c4_private_model_owner_bind_v1(
    void *owner, int64_t index, void *exact_handle);
int64_t et_g3c4_private_model_owner_initialize_v1(
    void *owner, int64_t index);
int64_t et_g3c4_private_model_owner_word_v1(
    void *owner, int64_t index);
int64_t et_g3c4_private_model_owner_prepare_seal_v1(void *owner);
int64_t et_g3c4_private_model_owner_commit_seal_v1(void *owner);
int64_t et_g3c4_private_model_owner_abort_v1(void *owner);
int32_t et_g3c4_construction_parameter_preflight_internal(
    const void *exact_owner, void *parameter,
    const void *exact_handle, et_f32_tensor_error *error);
```

The two constructors prevent a caller-selected origin. Both allocate all 14
exact C4 parameters before enrolling the owner and setting the separate singleton
`staged_c4_owner`. Partial allocation failure destroys only local allocations and
enrolls nothing. Each parameter is created through the existing I2 parameter
constructor, so its same-shaped gradient starts at exact +zero with no active
borrow, plan pin, contribution, or retained restore source.

A seeded owner stores original words `{1, seed, 0, 0}` and initializes indices
0..13 in order. Matrix rows use the accepted N3K initializer; six norm vectors
use fixed zero/one bits. Its successor is exactly 292 Philox blocks later.
`model_owner_word` accepts indices 0..3 for original and 4..7 for successor,
only after the corresponding words exist. A restored owner has no initializer
identity or words and rejects `initialize` and `word`.

A restored owner is only an origin-correct allocation primitive. This contract
does not authorize a raw tensor restore-copy function. The later R freeze must
name an authenticated stage-owner tuple and the exact I2 builder/prepare/commit/
abort route that fills each index. Until that contract is accepted, the restored
constructor is not callable by an Eshkol wrapper and restored-model implementation
remains blocked. A bare tensor pointer, digest, file string or live I2 object is
never sufficient provenance.

The seeded constructor uses these exact private Eshkol aliases:

```text
g3c4-native-model-owner-create-seeded    arity 1
g3c4-native-model-owner-parameter        arity 2
g3c4-native-model-owner-bind             arity 3
g3c4-native-model-owner-initialize       arity 2
g3c4-native-model-owner-word             arity 2
g3c4-native-model-owner-prepare-seal     arity 1
g3c4-native-model-owner-commit-seal      arity 1
g3c4-native-model-owner-abort            arity 1
```

Each maps directly to the same-stem C declaration above. Trusted
`(g3c4-model-create-seeded-internal seed)` is the sole arity-1 source transaction
that composes those aliases with the P1/I2 construction operations and returns
the canonical pending root only after the no-failure seal tail. Public
`generation-c4-model-create/1` calls that binding under `m3-call`. The restored
constructor and construction-preflight helper have no direct Eshkol alias in
this contract: the former waits for the exact R adapter, and the latter is called
only by the three-argument I2 C bridge.

Owner states are `OPEN`, `PREPARED`, `SEALED`, and `ABORTED`. Bind is once per
index in `OPEN`; all handles are nonnull and pairwise distinct. Seed initialization
is once per index in order. A restored owner cannot become ready until the later
R contract supplies and records its exact 14-member copy transaction.
`prepare_seal` validates origin-specific
completion, all 14 exact bindings, shapes, finite values, idle parameter/value/
gradient controls, and the absence of an active call. It mutates only
`OPEN` to `PREPARED`. `commit_seal` accepts only that exact prepared staged owner,
sets `SEALED`, clears `staged_c4_owner`, and returns 0; it allocates nothing and
any invariant defect aborts the process. Abort accepts `OPEN` or `PREPARED`, is idempotent only
for `ABORTED`, and rejects `SEALED`. A sealed owner cannot be destroyed.

## Closed P1/I2 construction split

Preserve the existing C2/M3T `(parameter, exact_handle)` bridge and construction
entry points. C4 adds an exact-owner route rather than probing owner classes from
an arbitrary parameter:

```c
int64_t et_i2_private_g3c4_construction_available_v1(void);
int64_t et_i2_private_g3c4_construction_parameter_preflight_v1(
    void *exact_owner, void *parameter, void *exact_handle);

int64_t et_p1_private_construction_prepare_v1(
    void *context, void *construction);
int64_t et_p1_private_construction_commit_prepared_v1(
    void *context, void *construction);
int64_t et_p1_private_construction_abort_prepared_v1(
    void *context, void *construction);
```

Availability returns 0 only in the C4 source tuple and -1 otherwise. The
three-argument I2 bridge requires `exact_owner == staged_c4_owner`, state
`OPEN` or `PREPARED`, exact membership among its 14 parameters, and an exact
handle. A null handle is accepted only while that member is genuinely unbound.
It then proves teardown eligibility for the entire fixed owner. It never calls
the M3T adapter and never installs a callback or owner registry.

The exact trusted Eshkol additions are:

```scheme
(i2-g3c4-construction-begin-internal native-owner)
(i2-g3c4-construction-prepare-eval-internal! construction)
(i2-g3c4-construction-seal-prepared-internal! construction)
(i2-g3c4-construction-abort-internal! construction)

(module-construction-prepare-eval-internal! identity)
(module-construction-seal-prepared-internal! identity)
```

Begin retains the exact owner in the I2 ledger before enrolling parameters.
Every registration and abort preflight supplies that owner to the three-argument
bridge. P1 and I2 states are `open`, `prepared`, `sealed`, or `aborted`.
Registration is allowed only in `open`; ordinary observation and mutation reject
both `open` and `prepared` construction shells. C4 uses the existing singleton
I2 construction registry/root and module-count rollback ledger; it does not add a
parallel construction registry, and M3T/C4 construction cannot overlap.

Native P1 preserves its existing numeric states 0 `OPEN`, 1 `SEALED`, and 2
`ABORTED`; 3 is the new `PREPARED` state. The existing native
`et_p1_private_construction_seal_v1` and
`et_p1_private_construction_abort_v1` continue to accept only `OPEN`, preserving
all direct C callers and C2 tests. Prepare accepts only `OPEN` and changes 0 to 3.
The two new prepared-specific functions accept only 3 and change it to 1 or 2.
They use the same owner, process and token-integrity admission as the legacy
functions.

`module-construction-prepare-eval-internal!` performs all currently fallible seal
work while abort remains legal: recompute and compare the parameter/tie schedule,
reject unattached modules/handles, compute canonical-path and finalization plans,
install canonical paths through checked promotion, finalize the module tree, and
set every module in that exact tree to `eval`. It verifies the resulting modes,
then calls `et_p1_private_construction_prepare_v1`. This fixed operation is needed
because construction modules default to `train` and public `module-eval!` cannot
observe an unpublished shell.

`module-construction-seal-prepared-internal!` accepts only `prepared`. It calls
the native P1 commit-prepared function, marks the source ledger sealed, and clears the active
construction. It performs no allocation, graph traversal, provider call,
promotion, or recoverable branch. I2's prepared seal mirrors that tail and only
then drops its carrier/membership ledgers. Prepared abort calls the new native
abort-prepared function and revokes the same enrolled module/handle identities.
The existing C2 construction source and native seal/abort path remains unchanged:
it performs its current fallible train-mode finalization while open, then invokes
the legacy native seal directly. It never enters `PREPARED`, so its public
behavior, direct native tests and M3T owner eligibility remain unchanged.

## Construction ownership and exact ordering

Before native owner enrollment, each constructor owns and frees its partial C
allocations locally. After enrollment, one cleanup ledger owns the C4 owner,
I2/P1 construction, pending model entry, and any restored replay objects.

Recoverable abort order is fixed:

1. Snapshot the first error. C4-I2 preflights the exact owner, every carrier,
   binding, and all 14 teardown controls. It then invokes P1 abort, restores its
   prior I2 module registry/count, and invalidates its carrier authority. Under
   the closed no-callback constructor, failure of this cleanup preflight is an
   invariant defect and fail-stops; it is never exposed as a retryable error that
   could strand the process-global construction.
2. Native owner abort repeats the complete 14-member preflight, destroys all 14
   parameter controls, tombstones the native owner, and clears
   `staged_c4_owner`. After step 1 succeeds, a defect in this tail is fail-stop.
3. Mark the pending model entry dead and clear its rooted metadata without
   allocation. Release restored cache/RNG/scratch/output owners from their
   preallocated cleanup ledger, preserving the first error.

Native owner abort must never precede P1/I2 revocation, because live P1 handles
would then name destroyed carriers.

Seeded seal order is: promote all Eshkol roots; P1/I2 prepare-eval; owner
`prepare_seal`; preflight every tail cleanup; P1/I2 prepared seal; owner
`commit_seal`; set the already-enrolled model entry live. The last three actions
are one no-allocation/no-recoverable-error tail.

Restored construction follows the same order through both prepare operations,
then performs closed no-draw replay against the authenticated prepared owner
while it remains unpublished. Replay releases its pins/views and preflights all
temporary destruction before the final tail. That tail seals P1/I2, commits the
owner, then sets the pre-enrolled model and generator entries live through
ordinary nonraising stores. “Atomic” means no observer can authenticate either
pending entry before both stores complete under the outer guard; it does not
claim a multiword hardware transaction.

## Exact C4 pin tuple and active-call tuple

The shared f32 owner adds one sibling type in the same source lineage as
`m3_call_pins.h`; no second pin implementation is permitted:

```c
typedef struct et_g3c4_model_pins_internal {
  struct et_g3c4_model_pins_internal *self;
  et_f32_parameter *parameters[14];
  const void *identities[14];
  et_f32_tensor *values[14];
  et_kernel_tensor_view_v1 views[14];
  uint16_t held_mask;
} et_g3c4_model_pins_internal;

int32_t et_g3c4_model_pins_begin_internal(
    et_f32_parameter *const parameters[14],
    const void *const identities[14],
    et_g3c4_model_pins_internal *pins,
    et_f32_tensor_error *error);
int32_t et_g3c4_model_pins_check_internal(
    const et_g3c4_model_pins_internal *pins,
    et_f32_tensor_error *error);
void et_g3c4_model_pins_end_internal(
    et_g3c4_model_pins_internal *pins);
```

The table differs from C2 only at index 13 `[4,4]`; total bytes are 4,768.
Operation names are `g3c4-model-pins-begin` and `g3c4-model-pins-check`.
Canonical IDLE has every named field zero. FULL has `self == pins`,
`held_mask == 0x3fff`, and all snapshots/descriptors exact. Begin performs the
accepted complete 14-member preflight, then acquires 0..13 by setting parameter
`plan_pins`, the exact value borrow sentinel, and its bit. Prefix failure drains
in reverse and returns canonical IDLE without clobbering the first error. Check
is read-only. End first rechecks FULL, then drains 13..0 and zeroes the record;
IDLE is idempotent, and every other corrupt state fail-stops before release. The
internal end preflight operation name is `g3c4-model-pins-end`.

The C4 source successor owns one `g3c4-registry`, using the accepted G3-T
12-slot transport entry shape with only the model/profile substitutions below:

| Slot | C4 transport value |
|---:|---|
| 0 / 1 / 2 | exact shell / kind / `pending`, `live`, or `dead` |
| 3 | authenticated native owner, or `#f` for private `call` |
| 4 | parent generator for a pending result/call, otherwise `#f` |
| 5 | exact entry from `g3c4-model-registry` for generator/call, otherwise `#f` |
| 6 | authentic raw V256 tokenizer for generator/call, otherwise `#f` |
| 7 | immutable normalized C4 policy for generator, otherwise `#f` |
| 8 | output auxiliary `#(raw-candidates staging-candidates selected-G)`, otherwise `#f` |
| 9 | exact active private call entry for generator, otherwise `#f` |
| 10 | rooted pending-child/candidate cleanup ledger, otherwise `#f` |
| 11 | `diagnostic-c4` |

Kinds and native kind IDs are inherited exactly: 1 `generator`, 2 `input`, 3
`logits`, 4 `output`, 5 `ids`, 6 `lengths`, 7 `cache-lengths`, and 8 `rng`;
Eshkol additionally has private kind `call`. A native generator record owns a
registry header/state/busy flag, exact C4 owner, one embedded C4 pin record,
normalized policy/RNG, committed cache and logits, history/H/P0/stop, binding
snapshot, fixed scratch, pending output/candidate ledger, call kind/budget, and
frame phase/ordinal/readiness. The address-stable generator context is the native
call context; no separate native CALL kind is introduced.

Every C declaration in the next section has exactly one private extern alias
`g3c4-native-<stem>` with underscores changed to hyphens and the displayed arity.
Shell wrappers authenticate the exact `g3c4-registry` kind before invoking an
extern. Public `generation-c4-model-create/1` alone invokes the seeded model
constructor transaction. Existing generation operations route C4 only after
model/profile authentication; additive `generator-continue!/2` normalizes its
explicit budget then enters the same guarded C4 orchestration. No C2 wrapper,
extern alias, registry, kind, or arity changes.

One active invocation has this exact tuple:

```text
m3-call-state
generator entry slot 9
C4 model entry slot 10
native C4 owner.active
ctx->pins.self / ctx->pins.held_mask
```

`m3-call-state` is the boolean `#t` aggregate exclusion flag. Generator slot 9
and model slot 10 are `eq?` to the same rooted Eshkol call entry. That call entry
authenticates the native generator context, and `owner.active` equals that native
context; pins are FULL. After pin begin succeeds, publish native `owner.active`,
then model slot 10, then generator slot 9 using nonraising stores. A per-token
acquired mask records each completed store; a synchronous interruption first
ends pins, then rolls back only recorded stores in reverse. Every later boundary
reauthenticates the generator, model, native owner, exact cross-links, call
phase, and pins. Pin end clears pins only. Finish/abort then clears native
`owner.active`, model slot 10, and generator slot 9, in that order. The enclosing
`m3-call` handler clears the aggregate guard last. Pins stay held across every
token in one generate/continue call and are never reacquired between frames.

## Exact C4 transport boundary

C4 is a separate private prefix; none of the accepted 29 C2 G3-T signatures is
changed. All names below have prefix `et_g3c4_private_` and suffix `_v1`:

```c
void *et_g3c4_private_generator_seed_v1(
    void *c4_owner, int64_t seed, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos);
void *et_g3c4_private_generator_rng_v1(
    void *c4_owner, void *rng, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos);
void *et_g3c4_private_input_from_t1_v1(void *sealed_t1);
void *et_g3c4_private_input_from_token_v1(int64_t id);
int64_t et_g3c4_private_tensor_release_v1(void *owner);
void *et_g3c4_private_logits_reserve_v1(void *ctx);
void *et_g3c4_private_output_reserve_v1(
    void *ctx, int64_t prompt_length_or_zero, int64_t budget);
void *et_g3c4_private_output_ids_clone_v1(void *out);
void *et_g3c4_private_output_lengths_clone_v1(void *out);
void *et_g3c4_private_output_cache_lengths_clone_v1(void *out);
void *et_g3c4_private_output_rng_clone_v1(void *out);
int64_t et_g3c4_private_output_release_v1(void *out);
int64_t et_g3c4_private_rng_release_v1(void *rng);
int64_t et_g3c4_private_generator_close_v1(void *ctx);
int64_t et_g3c4_private_call_acquire_v1(
    void *ctx, int64_t call_kind, int64_t budget);
int64_t et_g3c4_private_frame_begin_v1(
    void *ctx, void *input_or_null, int64_t frame_kind);
int64_t et_g3c4_private_role_step_v1(void *ctx, int64_t ordinal);
int64_t et_g3c4_private_sample_v1(void *ctx);
int64_t et_g3c4_private_frame_prepare_v1(
    void *ctx, void *staged_result_or_null);
int64_t et_g3c4_private_frame_commit_v1(void *ctx);
int64_t et_g3c4_private_frame_rearm_v1(void *ctx);
int64_t et_g3c4_private_call_prepare_end_v1(void *ctx);
int64_t et_g3c4_private_call_finish_v1(void *ctx);
int64_t et_g3c4_private_call_abort_v1(void *ctx);
int64_t et_g3c4_private_output_prepare_v1(
    void *ctx, void *out, int64_t generated_count);
int64_t et_g3c4_private_output_copy_decode_ids_v1(
    void *ctx, void *out, void *staging_header);
int64_t et_g3c4_private_output_accept_text_v1(
    void *ctx, void *out, void *raw_header);
int64_t et_g3c4_private_last_error_domain_v1(void);
int64_t et_g3c4_private_last_error_category_v1(void);
int64_t et_g3c4_private_last_error_code_v1(void);
```

All native owner and transport calls use one thread-local C4 error snapshot.
Pointer functions return null on failure; status functions return 0 on success
and the nonzero category. Domains are 0 C4, 1 K1, 2 I1, and 3 I2. C4 categories
1..6 are invalid-argument, invalid-state, shape-mismatch, unsupported, internal,
and determinism-unavailable. Codes 1..13 retain the accepted G3-T meanings:
identity, lifecycle/busy/borrow, selector/config, shape/token range, alias,
topology, allocation, capability, stale binding, invariant, FP control,
required-draw exhaustion, and readiness. Each boundary snapshots its first
failure before cleanup; cleanup cannot overwrite it. P1/I2 source exceptions are
already bounded E1 conditions and are rethrown after the native snapshot has
been preserved. Malformed domain/category/code triples map to internal/invariant.
Every model-owner operation uses its displayed stem as the fixed operation name.

The exact shell-owner wrapper names/arities are inherited from G3-T under the
`g3c4-` prefix: `generator-create`(3), `generation-input-create`(1),
`generation-token-input-create`(1), `generation-tensor-release!`(1),
`generation-output-release!`(1), `generator-close!`(1),
`generation-rng-release!`(1), `generation-output-ids`(1),
`generation-output-lengths`(1), `generation-output-text`(1),
`generation-output-rng`(1), and `generation-output-cache-lengths`(1).
The fixed orchestration wrappers are `g3c4-call-acquire`(3),
`g3c4-logits-reserve`(1), `g3c4-frame-begin`(3),
`g3c4-role-step`(2), `g3c4-sample`(1),
`g3c4-frame-prepare`(2), `g3c4-frame-commit`(1),
`g3c4-frame-rearm`(1), `g3c4-call-prepare-end`(1),
`g3c4-call-finish`(1), `g3c4-call-abort`(1),
`g3c4-output-reserve`(3), `g3c4-output-prepare`(3), and
`g3c4-t1-decode-output!`(2). These are trusted definitions, not public exports.

`call_kind` is 0 manual prefill, 1 manual decode, 2 new generation, or 3
continuation. Manual calls require budget 0 and have no output reservation. For
new generation, the Eshkol wrapper first authenticates the prompt input, obtains
P=1..4, normalizes the stored max-new value as this call's budget, and checks
P+budget<=4. It then calls `output_reserve(ctx,P,budget)` before acquisition.
Continuation supplies its explicit normalized budget and calls
`output_reserve(ctx,0,budget)`, which authenticates the current committed H and
checks H+budget<=4. Reserve captures immutable P-or-H, call kind and budget and
preallocates an exact candidate for every G in 0..budget, plus one rooted output
envelope and cleanup ledger. It never allocates an overcapacity public tensor or
changes a tensor's shape.

`call_acquire` requires the linked pending reservation for kinds 2/3, repeats
its captured bounds, and requires its budget equal the reservation budget. Kind2
requires budget equal the generator's stored max-new value; kind3 uses only the
explicit continuation budget. `frame_begin` for kind2 requires the authenticated
input length exactly equal the reserved P. These duplicate checks prevent a
smaller claimed P or divergent candidate count from reaching pins/cache. Only
after those checks does acquisition begin.

`frame_kind` is 1 prefill or 2 decode. Manual prefill and new generation require
an authentic input; manual decode requires a single-token input. Generated and
continued decode requires null and consumes only the owned sampled-token scratch.
Each frame copies IDs before retaining history. `role_step` admits exactly the
fixed next ordinal 0..20. The C4 numerical row is chosen by the closed role,
actual T, and profile; callers supply no provider/operation/shape selector.

After a completed nonfinal token, `frame_prepare(ctx, NULL)` and `frame_commit`
publish cache, history, last logits, candidate RNG, generated count, and stop bit
for that token. `frame_rearm` is then the only legal transition to the next sample:
it allocates nothing, retains pins/output candidates/committed state, and clears
only frame-local readiness and scratch. It rejects before mutation in every
other phase. Final preparation names the pending manual result or follows the
ordered output protocol below.

`output_prepare(ctx, out, G)` is numeric-only. For G0 it requires committed
generated count 0 after a successful prefill, or the unchanged committed
continuation state. For a final sampled token it requires that token still staged
and `G == committed_generated_count + 1`; the final token is not committed yet.
It copies the complete generated sequence and candidate RNG/lengths into exact
candidate G, and preflights destruction of every unused candidate. It does not
decode text or set text readiness.

Source-private `(g3c4-t1-decode-output! call-entry output-entry)` then invokes
`output_copy_decode_ids` into the exact preallocated 8G-byte staging buffer,
calls the existing trusted raw T1 decoder into candidate raw[G], and invokes
`output_accept_text` to validate those bytes and set text readiness. Only after
all three succeed does `frame_prepare(ctx,out)` validate the final staged token
(or G0 result), followed by `call_prepare_end`. The final `frame_commit` then
publishes token cache/history/logits/RNG/stop and the selected output in one
no-failure tail; finish drains unused candidates and the active tuple. A failure
at any earlier point aborts the uncommitted token tail and every unpublished
output candidate, preserving the last complete generator boundary and emitting
no partial output.

## Ownership and release proof

The model owns its 14 parameters for process lifetime. The generator owns its
committed cache, f32[1,256] last logits, history[4], H, P0, stop bit, normalized
policy, typed RNG, exact binding identities/value bits, and fixed scratch.
Generator close requires idle, destroys those owners, and clears model/tokenizer
roots; it never destroys model parameters.

A call owns only its candidate cache/token tail, pending manual result or every
exact-G output candidate, nested views, and cleanup ledger. Frame commit transfers
only the named committed fields. Output publication transfers the selected
IDs/length/cache-length/RNG snapshot to the output; its Eshkol entry owns the
selected raw bytevector. Accessor clones are independent. Abort ends nested views,
destroys the uncommitted tail and every pending candidate, drains pins, clears the
active tuple, and rethrows the snapshotted first error. All cleanup capacity is
reserved before the first commit.

SAVE copies the 14 exact pinned spans into independent storage while the active
tuple is held. It then ends nested views and pins, clears native `owner.active`,
model slot 10 and generator slot 9, exits `m3-call`, and only then performs
detached encoding or potentially blocking file I/O. R snapshot
and parser staging must each have an explicit idempotent release operation;
reconstruction must have the abort order above. The semantic names in the R
proposal are not frozen C/Eshkol signatures, and no generic history, logits, or
candidate accessor is authorized by this contract.

## Dependency pins and implementation hold

SHARED-R2 must be accepted and merged as the genuine I2/K2 successor: provider
and verified entry `eshkol-transformer-f32`, version `1.1`, evidence
`I2:bounded-exact-f32-storage.copy-v2`, with rank0, rank1, and only exact rank2
`[2,4]`, `[4,4]`, `[4,8]`, `[8,4]`, `[256,4]`. It changes no construction,
owner, pin, or replay authority. Its inspected implementation is still uncommitted.

The transformer currently pins compiler `90cbd713`; checked-promotion adoption
is therefore not present. The complete reviewed Eshkol union at docs commit
`222cad3a` over implementation `714d20fe` must first be accepted and pinned.
P1 construction writes must compile through
`eshkol_region_write_barrier_checked_v1`; status failure must transfer through
the fixed emergency condition path so the construction guard executes the abort
order above. Constructor/handler allocation guards and downstream P1 failure
tests remain mandatory. Transformer code must not call this compiler-private ABI
as a substitute for a correctly compiled checked store.

Implementation remains blocked until root accepts this contract and the two
dependencies above are accepted. Exact R snapshot/stage native layouts, their
release signatures, capsule provenance fields, final package counts, symbols,
and source/object manifests still require a separate R/package freeze. They may
not be guessed from the semantic R table or used to weaken the seeded C4 model
and transport contract frozen here.
