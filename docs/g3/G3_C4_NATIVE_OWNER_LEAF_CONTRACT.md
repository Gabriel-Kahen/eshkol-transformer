# G3-C4 seeded native-owner leaf contract

**Accepted implementation boundary.** This additive leaf is based on accepted
main `e899215cff22afeb00b9a5f56bddad9c4cc6c469`. It implements only the native
seeded C4 parameter owner. It adds no restored constructor, Eshkol alias,
P1/I2 construction transaction, transport, sampler, package manifest, CI entry,
or compiler/runtime pin.

The production leaf is
`src/eshkol_transformer/g3c4_model_owner_internal.h` and
`src/eshkol_transformer/g3c4_model_owner.c`, with focused native tests in
`tests/g3c4/test_model_owner.c`. The only shared-source amendment is a
conditional declaration and definition of
`et_f32_parameter_idle_preflight_internal` in `m3t_f32_scoped.h` and
`m3t_f32_integration.c`. It exists only under
`ET_G3C4_NATIVE_OWNER_PRIVATE`, uses the fixed operation name
`f32-parameter-idle-preflight`, allocates nothing, and verifies actual f32
registry membership plus parameter/value/gradient borrow and plan-pin state.
With the conditional absent, ordinary objects and symbol inventories are
unchanged. The existing M3T preflight and scoped operations are unchanged.

The source-private owner surface is:

```c
void *et_g3c4_private_model_owner_create_seeded_v1(int64_t seed);
void *et_g3c4_private_model_owner_parameter_v1(void *owner, int64_t index);
int64_t et_g3c4_private_model_owner_bind_v1(
    void *owner, int64_t index, void *exact_handle);
int64_t et_g3c4_private_model_owner_initialize_v1(
    void *owner, int64_t index);
int64_t et_g3c4_private_model_owner_word_v1(void *owner, int64_t index);
int64_t et_g3c4_private_model_owner_prepare_seal_v1(void *owner);
int64_t et_g3c4_private_model_owner_commit_seal_v1(void *owner);
int64_t et_g3c4_private_model_owner_abort_v1(void *owner);
int32_t et_g3c4_construction_parameter_preflight_internal(
    const void *exact_owner, void *parameter,
    const void *exact_handle, et_f32_tensor_error *error);
int64_t et_g3c4_private_last_error_domain_v1(void);
int64_t et_g3c4_private_last_error_category_v1(void);
int64_t et_g3c4_private_last_error_code_v1(void);
```

Calls require external serialization. A distinct process-lifetime C4 registry
authenticates every owner address before dereference and retains tombstoned
shells against address reuse. One `staged_c4_owner` may be `OPEN` or `PREPARED`.
States are `OPEN`, `PREPARED`, `SEALED`, and `ABORTED`; sealed owners retain
their parameters for process lifetime.

Creation accepts an exact nonnegative signed-i64 seed, allocates all fourteen
parameters before registry enrollment, and leaves no live allocation on any
failure. Shapes in owner order are `[4,4]` four times, `[4,8]`, `[8,4]`, six
`[4]` norm vectors at indices 6..9 and 11..12, `[256,4]` at index 10, and
`[4,4]` at index 13. Storage and nonnull bound identities are pairwise distinct.
Bind is once per index in `OPEN`. Initialize is once per index in order after all
bindings exist. N3K `n3k.matrix-init.uniform` initializes matrix indices
0..5,10,13 from original words `{1, seed, 0, 0}`. Norm beta indices 6,8,11 are
exact positive zero and gamma indices 7,9,12 are exact one. The successor is
exactly `{1, seed, 292, 0}`. Word indices 0..3 read the original and 4..7 the
successor only after it exists.

Prepare validates the exact staged owner, all fourteen bindings and members,
shapes, finite values, absent gradient metadata, idle controls, completed
initialization, and no active call. The exact positive-zero gradient storage is
the preserved I2 parameter-constructor invariant: no operation in this leaf can
obtain or mutate an absent gradient. Prepare then
changes only `OPEN` to `PREPARED`. Commit accepts only the exact prepared staged
owner, changes it to `SEALED`, and then clears the singleton without allocation
or a recoverable branch. Abort accepts `OPEN` or `PREPARED`, is idempotent for
`ABORTED`, rejects `SEALED`, authenticates membership and identity, and runs the
idle preflight over all fourteen parameters before destroying any of them. It
then destroys all parameters, tombstones the owner, and clears the singleton.
The whole-owner M3T preflight is never called. After complete preflight, any
commit-tail invariant defect or parameter-destruction failure is fail-stop; a
partially committed or partially destroyed owner is never returned as a
recoverable error.

This source owns the sole `_Thread_local` C4 `(domain, category, code)` snapshot
and the three getters. Its private header also freezes these helpers for future
C4 transport; future sources must reuse this storage:

```c
typedef struct et_g3c4_error_state_internal {
  int64_t domain;
  int64_t category;
  int64_t code;
} et_g3c4_error_state_internal;

void et_g3c4_error_reset_internal(void);
void et_g3c4_error_set_internal(
    int64_t domain, int64_t category, int64_t code);
et_g3c4_error_state_internal et_g3c4_error_snapshot_internal(void);
void et_g3c4_error_restore_internal(et_g3c4_error_state_internal snapshot);
```

Set and restore return no status: they store an accepted exact triple, while a
malformed triple is deterministically stored as C4/internal/invariant. Reset
stores the all-zero success triple and snapshot returns the current value.
Domains are C4=0, K1=1, I1=2, I2=3. C4 categories are OK=0,
invalid-argument=1, invalid-state=2, shape-mismatch=3, unsupported=4, internal=5,
and determinism-unavailable=6. C4 codes are OK=0, identity=1, lifecycle=2,
selector/config=3, shape/token-range=4, alias=5, topology=6, allocation=7,
capability=8, stale-binding=9, invariant=10, FP-control=11,
required-draw-exhaustion=12, and readiness=13. K1 and I2 triples retain their
actual accepted category and code. I1 triples likewise retain their actual
accepted I1 category and code. Unknown domains, out-of-range categories, or
out-of-range codes normalize to C4/internal/invariant. Each boundary snapshots
its first failure before cleanup; cleanup cannot overwrite it.

Focused tests cover all shapes, values, identities, gradients, the 292-block
successor, every allocation failpoint before enrollment, foreign/stale/singleton
admission, lifecycle transitions, malformed error mappings, and whole-owner
failure atomicity for every parameter/value/gradient borrow and plan-pin class.
Supported normal and sanitizer runs are reported separately; no broader package
or platform claim follows from this leaf.
