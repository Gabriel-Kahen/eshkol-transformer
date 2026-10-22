# CPU4 unpublished candidate producer and retirement contract

Adopted for bounded source implementation on 2026-10-22 after independent
`finish-20261022/cpu4-candidate-contract-assessment-v1/VERDICT.md` and its exact
OPEN_EMPTY/ABORTED-state correction. This is not compiled or aggregate admission.
The first closed Scheme producer supports no retained optimizer/generator children
(`#f` only); it rejects remaining live children. Native optional-child signatures
are not proof of a complete Eshkol restore/continuation teardown.

Scope: fixed CPU4 profile in root `docs/CPU_V1_PROFILE.md` at c3355f5b3. This is the minimum producer needed before checkpoint consumers. Ordinary C4 seeded construction, installed names, entry slot7=#f and context-two behavior remain unchanged. New native code requires `ET_CPU4_CANDIDATE_PRIVATE`, `ET_CPU4_TRAINING_PRIVATE`, and existing `ET_G3R_CANDIDATE_RETIRE_PRIVATE`. Default-off objects gain no candidate symbols.

## Native closed surface

All pointer admissions are registry-first; errors use existing C4 triples. A candidate capability is issued DURING construction, never adopted around an ordinary sealed model. The new native factory allocates capability control first, calls the existing seeded OPEN-owner factory, then enrolls the exact owner/capability through nonfailing writes before returning. It cannot allocate or invoke callbacks after owner enrollment.

```c
void *et_cpu4_private_candidate_begin_v1(int64_t seed);
void *et_cpu4_private_candidate_owner_v1(void *candidate);
int64_t et_cpu4_private_candidate_bind_graph_v1(
    void *candidate, void *p1_context, void *p1_construction);
int64_t et_cpu4_private_candidate_abort_open_v1(void *candidate);
int64_t et_cpu4_private_candidate_adopt_v1(
    void *candidate, void *exact_owned_generator_or_null);
int64_t et_cpu4_private_candidate_retire_prepare_v1(
    void *candidate, void *exact_optimizer_or_null,
    void *exact_owned_generator_or_null);
int64_t et_cpu4_private_candidate_retire_cancel_v1(void *candidate);
int64_t et_cpu4_private_candidate_retire_graph_commit_v1(void *candidate);
int64_t et_cpu4_private_candidate_retire_payload_commit_v1(void *candidate);
```

Native capability retains exact C4 owner, the existing native P1 context/construction, optional exact optimizer and optional one owned G3 generator. The owner receives a typed candidate pointer under the new flag, separate from CLI's typed training_lease/live_contexts and existing active pointer. States: CONSTRUCTING → UNPUBLISHED → ADOPTED or RETIRE_PREPARED → GRAPH_REVOKED → DEAD. Aborted preseal construction also reaches DEAD. Existing native owner commit-seal changes its attached capability from CONSTRUCTING to UNPUBLISHED in that same no-fail tail. ADOPTED is never admitted by failure retirement. Exact DEAD abort/payload-commit is idempotent; every foreign/cross-candidate/state mismatch rejects before mutation.

`bind_graph` runs before the first P1 token, once, and calls a minimal read-only native P1 helper:

```c
/* existing ET_G3R_CANDIDATE_RETIRE_PRIVATE; closed expected_phase:
 * 0 = exact active candidate OPEN/count0, entries==recorded roster base,
 *     context result==scope;
 * 2 = exact candidate ABORTED/entriesNULL/no-active, registry/context/pid valid.
 * Other phases reject. ABORTED never dereferences freed roster storage. */
int64_t et_p1_private_candidate_state_preflight_v1(
    void *context, void *construction, int64_t expected_phase);
```

It rejects an ordinary scope, another candidate's scope or any already populated/sealed scope. The C4 capability registry additionally prohibits binding that scope to a second capability. Open abort requires the exact native C4 OPEN/PREPARED staged owner, after successful exact P1/I2 abort: a bound capability MUST pass the phase2 native predicate before any payload destruction. An unbound capability may abort its fresh staged owner. It uses the existing owner abort, never the sealed path. Negative evidence must show bound abort rejection before P1 abort, genuine P1 abort, then successful retry.

Two native-only helpers complete real read-only whole-graph proof; neither is an Eshkol/installed API:

```c
/* in m3t_f32_integration.c, only ET_CPU4_CANDIDATE_PRIVATE:
 * parameter registry/child identities, distinct value/gradient, all plan/
 * borrow counts, existing immutable tensor backing destroy-readiness */
int32_t et_cpu4_parameter_destroy_preflight_internal(
    et_f32_parameter *parameter, et_f32_tensor_error *error);
/* in O2, same flag; arrays are the authentic C4 owner's native roster,
 * never imported from Eshkol. nonnull: existing O2 retire preflight plus
 * exact ordered fourteen parameter/handle equality; null: prove no live
 * optimizer/builder/state consumes this roster. */
int32_t et_cpu4_optimizer_roster_preflight_internal(
    et_o2_optimizer *optimizer_or_null,
    et_f32_parameter *const parameters[14],
    const void *const handles[14], et_o2_error_v1 *error);
```

Retire prepare validates the original sealed C4 owner, bound fourteen shapes/identities/tie, all value/gradient destroy-readiness, native P1 retained-roster preflight, exact O2 roster/moments/other-consumer exclusion, training_lease=NULL, and no active call. live_contexts must be zero if generator is null, or exactly one for the supplied registered, idle, same-owner candidate generator. Its full native close preflight is factored from the existing close; no guessed destructor. The Scheme ledger separately proves all TR3/transport/graph/result/borrow children are already released or included in its fixed preflighted cleanup plan. No retirement write occurs until ALL preflights succeed. Prepared state bars new training/context creation via one shared candidate-availability helper called alongside CLI's training exclusion. Cancel only exact prepared returns UNPUBLISHED.

Graph commit rechecks the complete preflight immediately before its first irreversible operation. It closes the exact optional generator through its accepted native close, commits the exact optional O2 optimizer via its accepted retire, then revokes this exact native P1 roster. Any unexpected nonzero after the first mutation exits134. It returns only GRAPH_REVOKED. The Eshkol closed tail then tombstones that exact child generator/O2 record, invalidates captured P1 shell/active-record nodes and unlinks only the captured I2 membership nodes/count, preserving unrelated graphs. Payload commit requires GRAPH_REVOKED, destroys all fourteen parameters through existing f32 destructor, zeros only owned payload/handle fields and leaves C4/capability control tombstones DEAD. Any unexpected failure exits134. There is no generic model destructor, M3T cast, native initializer object or postzero application callback.

Adopt requires UNPUBLISHED, sealed exact owner, no active/training lease and either zero contexts or the exact one owned idle generator. The Scheme caller authenticates complete config/restore/child provenance before this call. Its zero return is a no-fail publication boundary: fixed writes mark the original candidate ADOPTED and return preallocated result; no recoverable callback/allocation follows. Caller-owned adopted models cannot be cleaned up through this capability.

## Eshkol producer and retained authority

Closed private surface:

```scheme
(cpu4-candidate-model-create-internal! resolved) ; exact registered candidate
(cpu4-candidate-model-internal candidate)       ; same C4 shell, no transfer
(cpu4-candidate-adopt-internal! candidate generator-or-#f) ; publication
(cpu4-candidate-retire-internal! candidate)     ; closed preflight+tail
```

The source-private owning initializer installs one P1 bridge BEFORE the first construction, once. It appends trusted slot73 without changing slots0..72 or ordinary construction. It captures the fixed feature-local native candidate-begin and C4-bind-graph FFI closures in sealed lexical storage; no operation accepts an application callback, and reinstallation/rebinding rejects. Returned closed operations begin a candidate scope, retain exact shell records before seal, preflight retained identity/registry state, and perform only precomputed shell/active-record invalidation. Ordinary aggregates import no new feature-only native symbol. A feature-local constructor variant reuses C4's exact paths/shapes, owner initialization, I2 registration, prepared-eval and seal sequence; the ordinary constructor is byte-preserved.

Before native begin, root the candidate row and cleanup cells. Candidate P1 begin retains the existing native token roster; bind exact context/control to the native capability before first module token. The I2 candidate construction retains its original record and every membership/carrier as existing I2 creation enrolls it before native token publication. Before P1 seal, retain exact Scheme shell/active-record objects and I2 list-cell identities/successors in a private sidecar. No post-return wrapping or global native-token scan is permitted. Ordinary I2 seal may clear its own rollback fields; the candidate sidecar already owns the exact originals.

The candidate row is registered by identity, self-tagged and captures native capability/owner, original C4 entry/handles, P1/I2 controls, optional optimizer and one generator, all owned children and lifecycle. Factory errors before seal use existing exact open/prepared P1/I2/C4 abort; post-seal errors use this retirement. If cleanup is busy or indeterminate, retain the rooted unpublished candidate, first error and cleanup fault for authentic release/retry; never return a partial model or silently claim cleanup. List unlink plans are allocated/preflighted before mutation, preserve independent memberships and update count exactly once. Captured original nodes are rechecked before native first mutation.

## Actual gate

Implement producer before checkpoint consumer. Tests must construct a genuine same-registry C4 graph, exercise cuts before first token/at P1 enrollment/preseal/after seal, forged and cross-candidate controls, ordinary/adopted model denial, wrong tie/shape/value/gradient backing, every native and Scheme child borrow/pin/lease, second O2 consumer, idle live generator exclusion, and valid retry. Prove independent model/initializer/state/output unchanged; reclaimed parameter/O2/cache payload separately from retained tombstones. Inject faults after each irreversible stage: exit134 without publication. Normal/repeat/native sanitizer plus exact feature-off bytes/symbol closure and genuine Eshkol tests are required. Native-only tests do not admit aggregate cleanup or continuation. This producer is required source, not a replacement for the thin train→SAVE→reload/resume→eval→multi-token/EOS smoke.
