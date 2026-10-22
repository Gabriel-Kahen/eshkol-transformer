# CPU4 typed native lease contract

Revision 2 adopted by integration for lease-only source implementation, 2026-10-22.
Independent review: `finish-20261022/CPU4-LEASE-R2-VERDICT.md`; original reviewed
proposal SHA-256 `f97e2177ae06bbe0e5de6031e1af3b0bf91b95c1281c6e3f1502be4cabdfe8ab`.
Exact source/test review and supported compiled/sanitizer admission remain required.
CPU4 input/result/graph/E3 signatures are a separate pending joint freeze.

 This document supersedes the older active-pointer/begin/end sketch in `CPU4-STATE-LEASE-MAP.md`. Compile new authority only with `ET_CPU4_TRAINING_PRIVATE`; provider rows use existing ABI without a feature macro. No installed export until owning aggregate/manifests are accepted.

## Whole-trainer exclusion

Add a forward-declared typed `et_g3c4_training_lease_internal` and `training_lease` pointer plus `live_contexts` counter to native C4 owner **under this flag**. Do not store a training pointer in `owner->active`, cast it to G3 context, or change existing C4 entry slot7. Existing `active` retains its generation-call meaning.

Count one published native context/control exactly once in `live_contexts`, whether reached through raw-context, seeded-generator, RNG-generator or wrapper entry. Reject `SIZE_MAX` before allocation/publication. Increment only in its common allocation-free native enrollment tail, not again in its wrapper. Decrement only after the first successful close preflight/cache destruction; failed or repeated close never decrements. Create/seed/rng/manual context routes must all reject a nonnull training lease. A trainer lease can publish only for a sealed, idle, fully authenticated C4 owner with `live_contexts==0`, no generation active call and no existing training lease. Thus an **idle live generator** excludes training too, not only an active request. Feature-off C2/C4 behavior stays unchanged.

The lease has its own registered opaque-control identity/magic, exact owner, captured fourteen parameter pointers and fourteen bound native identities, phase, and monotonic transaction ordinal. It owns no parameters, tensor carriers, P1 tree or generator. Original seed/config equality is checked against native owner word1 in the Eshkol sideledger before trainer publication. All native entrypoints first authenticate exact registry membership without dereferencing a foreign pointer, then owner/roster/state; no caller-imported parameter array.

## Exact declarations and phases

```c
/* All symbols private; null/nonzero uses the existing C4 error triple. */
void *et_g3c4_private_training_lease_create_v1(void *exact_c4_owner);
void *et_g3c4_private_training_lease_parameter_v1(void *lease, int64_t index);
void *et_g3c4_private_training_lease_identity_v1(void *lease, int64_t index);
int64_t et_g3c4_private_training_lease_check_v1(void *lease);
int64_t et_g3c4_private_training_lease_enter_v1(void *lease, int64_t phase);
int64_t et_g3c4_private_training_lease_leave_v1(void *lease);
int64_t et_g3c4_private_training_lease_prepare_release_v1(void *lease);
int64_t et_g3c4_private_training_lease_cancel_release_v1(void *lease);
int64_t et_g3c4_private_training_lease_commit_release_v1(void *lease);

/* Native-only typed transport hooks; not Eshkol/application-callable. */
int64_t et_g3c4_training_child_acquire_internal(
    void *lease, const void *exact_child, int64_t kind);
int64_t et_g3c4_training_child_release_internal(
    void *lease, const void *exact_child, int64_t kind);
```

`create` allocates/registers one control object, validates all fourteen native shape/membership/I2/P1 identities, then atomically publishes `owner->training_lease=lease` in state `IDLE`. Any setup failure frees only unpublished control storage and leaves the owner/pointers/values/gradients untouched. Parameter/identity selectors are exactly0..13 and return existing canonical objects only; getters do not transfer ownership. `check` authenticates current roster and owner lease identity without changing phase.

`enter` accepts only `phase=1 READ_FRAME` or `phase=2 WRITE_TRANSACTION`, only from `IDLE`, with no outstanding child. It increments a uint64 ordinal, rejecting overflow before any change. Read frames encompass one complete forward/graph/VJP or trainer evaluation transaction. They permit staged **atomic gradient contribution** to canonical gradient state only; parameter-value/O2/scheduler/control writes are forbidden. Write transactions encompass the existing prepared I2/O2 update/restore transaction, after step-local read pins/borrows have ended; existing canonical P1/I2/O2 plans still own write admission and no-fail commit. A second phase or nested independent evaluator rejects. Trainer evaluation uses its trainer's exact existing lease through the agreed internal borrow; it never creates another lease.

The two native-only child hooks maintain registered exact child/lease/ordinal/kind identities (closed kinds1=input,2=workspace,3=graph,4=logits,5=read-pin). They are called only by the owning native CPU4 transport after its own exact resource-registry authentication, before child publication and after successful child release respectively. Every input/workspace/graph/result is frame-local; no cached scratch survives `leave`. Acquisition requires active READ_FRAME, unique live child identity and current ordinal; allocation failure publishes no child. Release requires exact lease+child+kind+current ordinal, never an imported count or caller declaration; failed resource-release preflight does not unregister it. There is no Eshkol/application child-count setter. READ pins end before staged gradient contribution, and all registered children/pins/results must be gone before `leave` can restore `IDLE`. Thus `leave` itself rejects a live matching graph/workspace; the ordinal/child proof is native state, not a caller assertion. WRITE leave additionally validates the existing canonical parameter/I2 unborrowed boundary after its prepared plans finish. No parameter mutation or ownership transfer occurs in leave.

`prepare_release` requires `IDLE`, exact roster and unborrowed canonical storage plus prior CPU4 child-workspace/graph/result release preflights; transitions to `RELEASE_PREPARED`. `cancel_release` returns only that prepared exact lease to `IDLE` (cleanup before the commit tail), preserving all model state. `commit_release` requires exact prepared identity and uses only no-fail assignments: clear `owner->training_lease`, mark registered lease `DEAD`; repeated exact DEAD commit succeeds. Foreign/dead/nonprepared other operations reject with unchanged bytes/state. No C4 parameter destruction, generic destructor or generator substitution is permitted. A release-preflight failure preserves the live lease and supports same-state retry. Tombstone/control retention is measured separately from canonical C4 process-lifetime retention.

## CPU4 transport / E3 handoff

The distinct CPU4 binding points to this exact C4 lease; scratch/graph ownership is explicit and never enrolled as M3T `OWNER`. Four IDs are native `i64[1,4]`; positions0,1,2,3, keep16; full logits/upstream `f32[1,4,256]` (1024 words/4096 bytes). Input publication validates all four IDs before copying any. Logits seed/copy accepts exactly1024 words and rejects512/1023/1025 without writes. Full VJP stages all fourteen canonical-shaped gradients, including position `[4,4]` and tied head/token `[256,4]`, then uses the existing atomic canonical P1/I2 contribution transaction. No scalar numerical fallback.

CPU4 input/result/graph C signatures must be frozen alongside E3's frame before composition; their owning ledgers, byte lengths and read/write phase rules above are mandatory. G3 creates a fresh context only after successful training/restore lease release and canonical eval mode, never reuses a pre-update context. G3 separately owns real restored-candidate cleanup/retirement.

Required native tests: forged/foreign/dead/mixed-owner roster; wrong selector/phase; idle and active live generator exclusion; repeated/context close counter balance; allocation/setup failure before publication; pinned/borrowed release rejection followed by retry; phase/read/write overlap; release-cancel/retry/idempotent commit; pre/post exact fourteen identities, values and gradients. Required Eshkol tests bind canonical config+native seed and reject wrong context/seed/profile before trainer publication. This proposal grants no execution/admission claim.
