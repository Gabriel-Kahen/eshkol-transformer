# E3-P1 upstream checked-promotion proposal

**Design only; root review required before an upstream patch or implementation
freeze.** This proposes a bounded compiler/runtime prerequisite for issue #108.
It does not amend the accepted [P1 mode contract](E3_P1_MODES_CONTRACT.md), change
its five slots/statuses, add an Eshkol source intrinsic, or authorize a different
P1 implementation. The shared dependency checkout remains read-only.

## Observed blocker and source basis

The audited upstream revision is
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`. Paths and line numbers in this section
refer to that revision of the Eshkol repository, not this repository.

| Pinned location | Relevant behavior |
|---|---|
| `lib/core/runtime_regions.cpp:1247` (`evac_raw`) | A null target allocation returns the original younger pointer; successful copies immediately enter the forwarding map. |
| `lib/core/runtime_regions.cpp:1261` (`evac_object`) | A null allocation logs and returns the original pointer. Forwarding insertion precedes interior traversal/worklist completion. Zero/implausible sizes and unknown subtypes also have unsafe leave-in-place/shallow paths. |
| `lib/core/runtime_regions.cpp:1400` (`region_evacuate_value`) | Creates or clears the persistent map and changes its target before graph completion; STL map/worklist growth can allocate/throw. |
| `lib/core/runtime_regions.cpp:1867` | The write barrier has a void ABI and writes its result without a failure channel. |
| `lib/backend/codegen_context.cpp:22` | `emitRegionWriteBarrier` calls that void function and unconditionally loads its output. |
| `lib/backend/collection_codegen.cpp:2203` | `vector-set!` unconditionally stores the loaded result. |
| `lib/backend/codegen_context.cpp:117` | Ordinary compiler-generated error delivery allocates an exception and message before `eshkol_raise`. |
| `lib/backend/llvm_codegen.cpp:22997` | Explicit `(raise caught)` also constructs a new exception wrapper. |
| `lib/core/runtime_exceptions_hosted.cpp:787` | `eshkol_raise(existing_exception)` does not construct an exception, but it invokes dynamic-wind cleanup, promise unwind, region unwind, and then `longjmp`. |

The local diagnostic driver is
[`scripts/test-e3-p1-runtime-barrier.sh`](../../scripts/test-e3-p1-runtime-barrier.sh).
It links the actual pinned runtime and compiles the actual Eshkol `vector-set!`
publication path in
[`tests/e3_p1/runtime_barrier_publication.esk`](../../tests/e3_p1/runtime_barrier_publication.esk).
At the inspected local run, both the native and AOT successful controls survived
poisoned region exit. All six bounded-target exhaustion stages produced 6, 5, 4,
3, 2, and 1 original regional edges respectively. The driver correctly exited 1.
The linked runtime archive SHA-256 was
`580946c291d88e2008de02202b8e07942e7b85c0ac9c6b7f861397c1d115e240`.

The imposed remaining target capacities were 0, 48, 208, 240, 528, and 816 bytes.
These are deterministic synthetic bounded-arena allocation failures. They are
neither host OOM nor C++ map/worklist allocation failures. Failed graphs were
retired before region destruction, and no failed graph was read after destruction.
This is real local AOT evidence for the runtime publication defect, **not** full
P1 bind evidence, genuine E3 frame admission, or supported Ubuntu22/LLVM21 evidence.
The original logs and final candidate evidence must retain those distinctions.

## Chosen interface and source contract

Add a distinct, compiler-private C ABI in the upstream runtime header. Do not
change the return type of the existing void symbol in place: an old object would
still ignore the return value and publish unconditionally.

```c
/* Compiler/runtime promotion ABI v1; no application/provider authority. */
int32_t eshkol_region_write_barrier_checked_v1(
    eshkol_tagged_value_t *out,
    const void *destination_owner,
    const eshkol_tagged_value_t *value);

/* Never returns; transfers one fixed runtime-owned condition. */
void eshkol_region_promotion_raise_v1(int32_t status);

/* Returns only for an ordinary value; exact emergency identity is rethrown. */
void eshkol_region_promotion_rethrow_if_v1(
    const eshkol_tagged_value_t *value);
```

The header uses the upstream project's C/C++-compatible noreturn annotation for
`eshkol_region_promotion_raise_v1`. It does **not** mark
`eshkol_region_promotion_rethrow_if_v1` noreturn: the ordinary-value path returns.
Neither function is an installed Eshkol operation or a transformer native seam.

| Result | Meaning |
|---:|---|
| 0 | Complete promotion or a proven no-promotion case; `*out` contains the exact safe value. |
| 1 | Target allocation or temporary forwarding/worklist/bookkeeping allocation failed. |
| 2 | A traversed object has an unsupported/unprovable layout; leaving a younger pointer in place is forbidden. |
| 3 | A checked object/span/capacity calculation overflowed. |
| 4 | Invalid call/runtime state, including null mandatory pointers or no valid target for a required promotion. |

On every nonzero result, `*out` is byte-for-byte unchanged, the destination has
not been written, original source objects are unchanged, and persistent
forwarding entries/target and committed escape counts are unchanged. The function
returns normally after destroying its temporary C++ state; it never raises or
allows a C++ exception to cross the C boundary. Its inputs are live compiler-owned
runtime values, not an interface for validating arbitrary native pointers.
`out` and `value` may alias only as private staging storage; neither may alias the
destination slot or any object reachable from the input graph. The generated
caller uses two distinct entry-block stack slots and meets this precondition.
Checking native pointer authenticity is not added to this ABI.

The no-region, pointer-free immediate, null payload, already-root-owned and
same/longer-lifetime value cases remain zero-allocation paths. They do not create
an empty map, worklist, diagnostic object, or registration record. Existing symbol
lifetime rules remain unchanged. In particular, P1's immediate phase/status
writes and root-owned train/eval symbols must take these paths.

All span arithmetic and known object layouts encountered by the existing
evacuator must be checked before allocation/copy. A zero/implausible header or an
undeclared subtype cannot be called successful promotion merely by returning the
source address. Existing valid type coverage must be preserved; this proposal
does not add new object representations or broaden the accepted pointer grammar.

## Transactional forwarding and cleanup

Use one internal checked promotion engine with explicit status propagation.
Changing only the outer return type is insufficient. `evac_raw`, `evac_object`,
the worklist loop, and every raw-buffer/interior traversal must propagate the
first failure and stop traversal immediately. No later failure may replace it.

For this first bounded implementation, favor a staged replacement forwarding
map over rollback of a partially modified live map:

1. Read the current owner/map/target without modifying them. If the root already
   has a committed forwarding entry for this target, reuse that completed copy
   without allocating. Otherwise create a temporary candidate map. For the same
   target, copy the existing committed map into it; for a different target, start
   empty. Retain the original map and its target until commit. Failure constructing
   or copying the candidate leaves the original intact. A successful target change
   intentionally replaces the old target's cache, matching current semantics;
   this does not promise one canonical address across different target arenas.
2. Track every newly allocated target span in a temporary POD span ledger. Reserve
   capacity for its entry **before** requesting the target allocation. After a
   successful target allocation, recording its pointer/size cannot allocate or
   throw. Thus even failure inserting a forwarding entry or scheduling traversal
   cannot lose the only cleanup/accounting reference to a partial copy.
3. Insert new forwarding entries only into the candidate map. They may refer to
   not-yet-completed speculative copies to close cycles. Preserve aliases within
   the transaction and reuse prior committed copies; never rewrite interiors of
   prior committed copies or original objects. Every interior rewrite targets a
   newly copied span. If an interior buffer is already stable/committed, verify
   it needs no rewrite; do not mutate it as an incidental traversal destination.
4. Complete all graph traversal and all temporary/map allocations, validate the
   result and arithmetic for committed counters, then commit. Commit swaps the
   owner map pointer and target, updates committed copy counts, and writes `*out`.
   Those operations must be nonallocating/nonthrowing. Deleting the old map must
   not invoke user code or allocate. The C ABI returns 0 only after this commit.
   No callbacks, native provider calls, or source evaluation occur in this engine.
   Do not format or log an allocation-failure diagnostic inside the transaction;
   its numeric result is sufficient for the fixed error-transfer path.
5. On failure, use only the recorded spans to scrub the speculative copies; do
   not walk the partial graph, follow its raw fields, or scan arbitrary roots.
   Discard the candidate map/worklist/span ledger. Preserve the old map, target,
   output sentinel, destination and source bytes. Test every bookkeeping
   allocation as well as every target allocation.

Catch allocation exceptions from every STL operation inside the engine. Treat
`std::bad_alloc` as result 1 and checked capacity/length overflow as result 3.
Use the reviewed standard pointer hash/equality and POD bookkeeping only, with no
throwing application callbacks. Unexpected runtime defects must not be disguised
as successful promotion. The implementation review must account for any additional
throwing operation before deciding its exact failure disposition.

The map-copy choice has a declared cost: a promoting operation can need temporary
memory proportional to existing committed entries plus new entries. The existing
no-promotion and already-forwarded paths must bypass that cost. Measure its
effect on predecessor workloads; do not hide it behind the nonallocating P1 tail
claim. An optimized alternative requires a separately reviewed transaction proof;
this proposal does not authorize an unbounded merge or rehash after commit begins.

Failed speculative target bytes remain charged to the target arena until its
ordinary lifetime ends. **Do not rewind a shared target arena** or infer that
discarding the map frees its arena copies. Scrubbing removes inaccessible source
pointers but is not reclamation. Record target used-byte delta, reserved-byte and
block-count delta, attempted/successful copy bytes including headers/alignment,
peak temporary native bytes, and temporary live-byte return to baseline. Existing
`escape_count` remains a committed-copy count; expose failure counters only through
reviewed upstream test instrumentation. Repeated failed setups can accumulate
unreachable arena bytes; report the measured slope. No process-lifetime flatness
claim or new allocation quota is introduced.

## Compiler publication sequence

`emitRegionWriteBarrier` keeps its two entry-block stack slots and emits:

```text
store input into private value slot
status = checked_v1(private output slot, destination owner, private value slot)
if status != 0:
    promotion_raise_v1(status)     # noreturn; transaction temporaries are gone
    unreachable
success:
    canonical = load private output slot
    return canonical to the mutator
```

The destination store must be dominated by the successful result comparison.
The output load must also be dominated by that comparison; it is uninitialized
and intentionally unread on failure. Preserve these properties at optimized AOT
and emit negative IR/behavioral mutants that remove either branch. Keep existing
argument/index/type preflight and source evaluation order; do not mutate the
destination before the barrier.

The centralized helper is also used by scalar cons/global/vector-fill/hash
stores. Updating it changes their barrier calls, so inventory and test those
callers. This does not by itself prove a multiwrite operation transactional.
The P1 prerequisite's required operation is the ordinary single-slot
`vector-set!` path.

## Allocation-independent delivery to P1's guard

Use a finite table of process-lifetime, header-bearing, runtime-owned emergency
exceptions, one immutable identity for each nonzero result above, initialized
without arena/heap allocation. Their diagnostic text is static, they have no
irritants or arbitrary caught graph, and they carry no model/frame/pointer detail.
Use existing exception types; no new transformer error domain or public category
is added. Any upstream helper that can mutate exception metadata must recognize
these reserved identities and leave them immutable without allocating. A copied
record, equal message, numeric tag, or user-created exception never authenticates
as one of these conditions.

Each static entry must satisfy the normal exception payload alignment and exact
header-before-payload offset, proved by compile-time layout assertions. The tagged
payload pointer, header, static message and zero-irritant state remain valid for
the entire process, including after a thread exits. Region poisoning and exception
clear/pop must never free or rewrite this storage.

`promotion_raise_v1(status)` selects the exact exception identity, initializes
the existing raised-value state consistently, then uses the existing exception
transfer machinery. It must not call `eshkol_make_exception*`, build a message,
append an irritant, register a closure, or allocate a handler. Invalid status is
an internal runtime defect: zero or an unknown status selects the fixed status-4
internal condition, never success or an invented status extension.

Explicit `codegenRaise` must evaluate its input once, spill it, and invoke
`promotion_rethrow_if_v1` **before** any generic wrapper/message allocation or
replacement of the current emergency exception. That helper compares only exact
tagged runtime identities. For an emergency identity it transfers the same
exception and raised value without allocation; for all ordinary values it returns
and the existing raise semantics continue. A null input pointer also returns
without dereference; it cannot authenticate an emergency condition. Guard
fallthrough rethrow and other existing-exception paths require an audit so the
reserved identity and raised-value state are neither copied into a new wrapper
nor accidentally cleared.

The guarantee is deliberately local. For P1 bind, prove in actual generated code
and runtime instrumentation that:

- Its local cleanup guard and handler frame already exist before either publication
  attempt; handler construction/registration is outside the failing barrier.
- No dynamic-wind after-thunk, new promise evaluation, callback, provider call,
  continuation capture, or nested region entry occurs between that guard mark and
  the barrier. Transfer to this nearest guard therefore executes no user callback
  and requires no caught-graph promotion.
- On root-ledger publication failure, no canonical record was published and the
  guard does not dereference a speculative copy. If root publication succeeded but
  the later setup-box publication failed, the guard uses its already-read-back
  canonical record to clear slots 2–7 and mark it dead. Those writes take the
  zero-allocation barrier paths. Modes and caller output remain unchanged.
- `(raise caught)` in that cleanup takes the exact emergency rethrow path, preserving
  the original exception identity after the cleanup completes.

Do not label arbitrary `eshkol_raise` execution allocation-free. Existing
dynamic-wind ordering and callbacks remain intact. A user's after-thunk may
allocate or raise; the proposal neither skips it nor introduces a new fail-stop
rule merely to claim recovery. Region unwinding, pinned continuations, outer
handlers and callback-driven cleanup need their own existing semantics and tests.
Only P1's demonstrated nearest-guard path supports the allocation-independent
claim. The caller's eventual construction of a normal rich error is outside that
cleanup proof and may itself require available memory.

This preserves P1's accepted distinction: bind may raise before mode mutation;
slots 65–68 keep the exact immediate statuses and remain nonallocating. The new
upstream runtime conditions are not extra P1 return codes. This proposal alone
does not prove allocation failure of every ordinary Eshkol object constructor;
the complete P1 allocation-failpoint matrix remains required.

## Complete native caller inventory at the pin

The inventory below distinguishes actual calls from comments referring to the
native design. Each row must have an explicit disposition in the upstream patch;
no forwarding-map writer remains on an old implementation.

| File and function/callsite | Pin behavior | Required disposition |
|---|---|---|
| `runtime_regions.cpp`: `region_evacuate_value`, `evac_value`, `evac_object_ptr`, `evac_object`, `evac_raw` | One map shared by deep graph-copy callers, currently modified before completion. | Refactor this implementation into the single checked engine; remove all failure-as-source-pointer branches in this engine. |
| `runtime_regions.cpp`: `eshkol_region_write_barrier_into` | Scalar void adapter around deep evacuation. | Retain its ABI as a checked-engine adapter; stage privately, publish only after success, otherwise transfer the fixed emergency condition after cleanup. |
| `runtime_regions.cpp`: `region_escape_tagged_value_impl`, `region_escape_tagged_value`, `region_escape_tagged_value_into` | Return/overwrite escaped result, currently without a failure channel. | Same engine and cleanup-before-emergency rule; no result returned or out-slot overwritten on failure. These symbols retain their signatures. |
| `runtime_regions.cpp`: `eshkol_iter_nursery_recycle` | Promotes `vals` one at a time, then poisons/resets the nursery and clears the map. | Stage all kept values in one engine transaction before any result-slot write, poison, reset, or map invalidation. Failure cannot recycle the source. |
| `runtime_regions.cpp`: `eshkol_region_unwind_to`, reached by `eshkol_region_handle_close` | Retires handles, promotes values individually at each level, restores allocation slot, pops region. | At each level, stage the entire kept-value set before its handle retirement, slot writes, arena restoration and destruction. Failure leaves that level live until ordinary emergency exception transfer determines its cleanup. |
| `runtime_exceptions_hosted.cpp:815`, `runtime_continuations.cpp:225` | Raise and continuation transfer call `eshkol_region_unwind_to`. | Audit the updated unwind call and root-owned emergency value; preserve existing dynamic-wind/promise ordering. No separate evacuator or exception-graph shortcut. |
| `codegen_context.cpp`: `emitRegionWriteBarrier` | Calls the void barrier and loads its result. | New versioned checked ABI and status-dominated load/store sequence. |
| `collection_codegen.cpp`: `vectorSet` at 2203; `vectorFill` at 2808 | Scalar tagged mutation; fill promotes its single fill value before its write loop. | Use the central checked helper; prove the first actual write follows successful promotion. Preserve the distinct numeric tensor branches. |
| `hash_codegen.cpp`: `HashCodegen::hashSet` at 416/418 | Promotes key and value before `hash_table_set`. | Both promotions finish before table mutation. If the second fails, the first committed promotion may remain retained; table state remains unchanged. Hash-table allocation behavior is a separate operation contract. |
| `llvm_codegen.cpp`: global `set!` store at 13678; cons mutation at 39230 | Central helper protects the store. | Updated helper; no bypass to the old void call and no write before success. |
| `runtime_parameters_hosted.cpp`: `eshkol_make_parameter`, `eshkol_parameter_push`, `eshkol_parameter_set`, `eshkol_parameter_set_converter` | Four direct scalar calls; push increments `top` before promotion, and constructor allocates native stack before promotion. | Stage through the checked engine before publishing slot/top/converter. On constructor promotion failure, free its unpublished malloc stack and retire the unpublished control before emergency transfer. Retain separate constructor/realloc failure accounting. |
| `runtime_vector_mutation.cpp`: `eshkol_vector_copy_mutating`, calls at 97/113/145/155 | Vector↔vector and tagged dual-tensor branches copy first and barrier afterward. | Migrate all four callsites to the whole-range staging rule below before removing the old postwrite range entry point. |
| `collection_codegen.cpp`: `CollectionCodegen::vectorCopy` | Calls `eshkol_vector_copy_mutating`; existing statuses 0–3 describe shape/type admission. | Preserve those statuses. Promotion failure is raised inside the runtime helper only after all temporary native state has been released; do not map OOM to a type error. |
| `llvm_codegen.cpp`: nursery runtime declarations at 28741/28743 and region lowering at 35384 | Calls recycle/result-escape/unwind symbols. | Retain signatures but test the updated adapters and failure-before-reset/pop ordering in generated AOT. |
| `repl/repl_jit.cpp` resolver near 1060 | Explicitly registers result escape and nursery recycle. | Register the three new compiler-private v1 symbols alongside existing runtime discovery; test exact JIT/static linkage, not just dynamic lookup on one host. |

All runtime paths in the table are under upstream `lib/core/`, compiler paths
under `lib/backend/`, except the explicitly named REPL path. Update the companion
declarations in `lib/core/arena_memory.h` and
`inc/eshkol/backend/codegen_context.h`. A private C++ declaration, if needed to
share the checked batch implementation between core translation units, belongs
in `lib/core/runtime_region_promotion_internal.h`; it is not a registered C ABI,
installed source operation, or an additional authority surface.

The shallow `region_escape`, `region_escape_string`, and
`region_escape_tagged_cons_cell` functions at 828/857/889 neither read nor write
the persistent forwarding map. The audited production C/C++ sources contain no
calls to them outside their definitions/declarations; they are excluded from P1's
compiled execution proof, not relabeled deep checked promotion. Their standalone
allocation behavior remains separate. The bytecode VM uses its own heap/evacuation
implementation; comments naming native functions are not calls into this engine.
Browser shims in `web/eshkol-repl.js` and `site/static/eshkol-runtime.js` implement
nonreclaiming stubs. They do not establish this native prerequisite, and must not
silently resolve the new checked ABI to an old shallow/no-op shim. The initial
acceptance scope remains the supported hosted native compiler/runtime lane.

## Concrete coexistence and bulk-copy decision

This revision replaces the earlier open legacy-map coexistence choice with a
specific proposal: **one transactional forwarding engine for every native map
writer, safe compatibility adapters for scalar/result escape, and retirement of
the unrepairable postwrite range ABI at the explicitly reviewed toolchain
transition.** Root and the upstream owner must accept that transition before any
patch is dispatched. Keeping two evacuators or accepting old failed maps is not
an implementation option under this proposal.

Legacy scalar/escape wrappers preserve their successful return conventions and
null-pointer conventions where documented. Internally they call the checked
engine with private staging output. On failure, the engine returns after scrubbing
and releasing temporary state; only then does the wrapper invoke emergency
transfer. The wrapper never returns the original younger pointer as a failure
fallback. Native direct callers that currently pass their actual destination as
`out`, including parameters, must use a local staged value and commit their
container metadata only after success. This does not relax the new checked ABI's
private-staging precondition.

Since every deep caller uses the same transaction rules, the map contains only
fully committed copies irrespective of whether the call entered through a legacy
scalar name or the new checked name. A failed call leaves that map and forwarding
target identity unchanged; speculative target-arena bytes are scrubbed and remain
retained as accounted above. No map-version tag, second registry, opaque caller
token, serialized proof, or selective trust of old entries is needed. Start only fresh processes
with the rebuilt runtime; no old process/map is upgraded in place. Test alternating
legacy and checked calls, including failure/retry and target changes, in one
region. These are the required coexistence proofs; none is established by the
current pin or by assuming that failed legacy paths happen not to execute.

The internal engine accepts one or more roots for one destination lifetime and
commits their forwarding map once. The one-root C ABI is its scalar adapter.
Batch entry points are runtime-internal plumbing, not another Eshkol intrinsic.
The batch output is caller-private scratch; no true destination range is modified
until the whole transaction succeeds. Scratch allocations and copying use checked
counts and the same first-failure/cleanup discipline.

For `eshkol_vector_copy_mutating`, preserve all shape/type/overlap validation and
all four vector/tagged-dual conversion directions. When copied elements can need
promotion, first snapshot the complete source slice into temporary native tagged
storage, run a single batch promotion transaction for the destination lifetime,
then copy the completed slice to the true destination and release the scratch.
Thus overlap keeps memmove semantics, cycles/shared tails keep identity, and no
input/output alias can expose a partly promoted range. If snapshot allocation or
promotion fails, release all scratch before transferring the emergency exception;
the destination bytes are unchanged. Pointer-free numeric conversion and proven
no-promotion cases keep their existing direct paths; do not add allocation to them
just to reuse the staging implementation. No C++ destructor is bypassed by the
eventual `longjmp`.

The old void `eshkol_region_write_barrier_range(dst, slots, n)` cannot repair the
caller's original destination after that caller has already overwritten it; its
arguments contain no original bytes. Therefore remove that postwrite symbol and
declaration from the newly accepted native runtime after migrating every in-tree
caller. Do not keep a compatibility implementation that raises with younger
pointers already published, return a shallow success, or silently change the
meaning of its arguments. Old external objects requiring that symbol must fail
link/load under the new compiler-private runtime ABI. Existing self-contained old
executables remain old artifacts and are not evidence for the new pin. This is an
explicit runtime ABI migration requirement; if upstream will not accept it, root
must choose a separately reviewed alternative before patching, not leave mixed
failure semantics in the candidate.

For kept-value arrays in nursery recycle and region unwind, use the same internal
batch transaction before array writes and reclamation. An unwind is staged one
region level at a time; it does not promise to resurrect levels already completed
before a later failure. For the current level, handle retirement and arena
destruction follow successful staging. On failure, the fixed root-owned emergency
value can traverse the normal exception unwind without graph promotion; test this
actual path and nested-handler marks. Preserve dynamic-wind ordering and the
existing callback semantics rather than introducing a fail-stop or callback skip.

Concretely, replace the initial bulk `rh_retire_above(mark)` in
`eshkol_region_unwind_to` with retirement of reclaiming handles belonging to the
successfully staged current depth, immediately before that depth is destroyed.
An outer/lower-depth handle cannot be retired before its own promotion stage has
succeeded. The existing independent bookkeeping-only handle path keeps its own
documented sequence/cascade rules.

The batch engine must preflight the no-promotion case **before** constructing any
scratch array, span ledger, candidate map, or worklist. For the fixed root-owned
emergency exception, every unwind level takes this zero-allocation path. This is
mandatory: otherwise an allocation-disabled unwind could allocate batch scratch,
fail again, and recurse indefinitely through emergency transfer. Force an unwind
promotion failure with allocation still disabled through emergency retry; require
exactly one emergency transfer, a still-live current level until its successful
retry, no early handle retirement/reset/pop, and eventual delivery of the same
fixed condition. This test counts emergency transfers separately from an original
ordinary raise or continuation request.

New generated code references the versioned checked symbol; linking against the
old runtime fails rather than falling back. Rebuild compiler, runtime, all native
support libraries, canonical aggregates and AOT callers together, with explicit
new upstream revision/provenance and exact symbol closure checks. No old object
cache or postwrite range reference may survive in the accepted dependency set.

## Patch ordering and ownership proposal

One upstream implementation owner must be named by root. That owner owns the
whole runtime/compiler transition and integration order; independent testing and
lifetime review can be delegated, but no second owner patches a parallel
evacuator. Transformer agents do not edit shared `.deps` or adopt a speculative
pin. The following are review/commit boundaries in one dependency chain, not
permission to merge a partially migrated runtime:

| Order | Exact owned scope | Required gate before the next boundary |
|---|---|---|
| P0: checked transaction | `runtime_regions.cpp` evacuator and map lifecycle; private batch declaration if necessary; `arena_memory.h` new checked ABI. | Native scalar/multiroot transaction tests, complete allocation failpoints, retained-byte ledger, old-map preservation, cycles/aliases and target-switch negatives. No new publication path enabled yet. |
| P1: error transfer and scalar migration | `runtime_exceptions_hosted.cpp` fixed emergency objects/transfer and immutable metadata checks; `codegen_context.cpp/.h` checked branch; `llvm_codegen.cpp` exact rethrow fast path; parameter direct callers; safe legacy scalar/result adapters. | Actual AOT single-store suppression and nearest-guard cleanup/rethrow with allocation disabled; exact old/new scalar-map interoperability and parameter slot/top atomicity. Ordinary raise behavior preserved. |
| P2: remaining shared-map callers | `runtime_regions.cpp` recycle/unwind batches and per-successful-depth handle retirement; `runtime_vector_mutation.cpp` prewrite batch staging; remove old range declaration/definition; audit exception/continuation callsites. | Every copied-element failure leaves the full destination unchanged; no failed recycle/reset/pop; zero-allocation emergency unwind retry with exactly one emergency transfer; nested unwind tests; no production range references; all scratch freed before emergency transfer. |
| P3: integration and pin eligibility | REPL resolver, exact native symbol/linkage tests, upstream release/provenance surfaces; supported full upstream CI and independent lifetime review. | Fresh artifact closure, old-runtime/old-range mismatch fails explicitly, all existing successful-behavior regressions and supported sanitizer/leak evidence. Only the completed P0–P3 union is eligible for root's new-pin decision. |
| Transformer adoption | Root-coordinated toolchain lock/provenance transition, then canonical P1 append and inherited packages. | New pin independently accepted; constructor prerequisite below separately resolved; complete issue108 failpoint/lifetime/public/package tests and supported exact-candidate CI. |

Intermediate commits are reviewable upstream work, not installable transformer
prerequisites. P1's frozen draft stays blocked until the complete transition is
accepted. Root still has not assigned the sole upstream implementation owner.

## Separate ordinary-constructor prerequisite

Checked promotion handles allocations performed by the promotion engine and its
temporary bookkeeping. It does not automatically fix allocation of the original
token/record/vectors/list cell or construction of P1's cleanup handler.

The parent audit of local compiled IR found a separate null-check gap: the
two-slot `vector` allocator call is followed by an unconditional length store at
lines 2786–2787 of `.tmp/e3-p1/resume/constructor.o.ll`, and `make-vector17` has
the analogous call/store at 2804–2805. The pinned
`lib/core/runtime_object_alloc.cpp:239–258` allocator can return null. This is
static IR/source evidence, not an executed constructor-OOM test.

Final issue108 all-allocation acceptance therefore also needs a root-approved
adjacent audit/fix for the exact vector, cons/list-cell and guard-handler
constructors used by bind, with their null-check-before-initialization and
nonallocating failure-transfer paths. The emergency condition introduced here
must not be repurposed to constructor errors without that explicit design and
review. None of P0–P3 may be reported as proof of those constructors, and this
proposal does not casually expand into every language allocator. Parent/root
retains ownership of that adjacent scope.

No alternate native wrapper, raw region pin, whole-region retention workaround,
revised P1 token authority, compiler flag exception, sanitizer disablement, or
timeout increase is part of either proposal.

## Required upstream and downstream evidence

1. Native checked-ABI tests: every target/map/worklist/span-ledger allocation
   failure, invalid layout and checked overflow; unchanged output/destination/source
   and old forwarding state; no C++ exception crosses C; exact cleanup counters.
2. Successful and failed shared-tail/cyclic graphs, multiple publications sharing
   one map, prior successful publication followed by failure/retry, and target
   changes that fail without discarding prior committed mappings. Retries cannot
   retrieve incomplete copies from failed transactions.
3. Real Eshkol AOT `vector-set!` tests at each failure prefix: output sentinel
   unchanged, no store on failure, nearest guard entered, exact emergency identity
   caught/rethrown, every allocation disabled through cleanup, and poisoned-region
   survival only for fully committed graphs. Preserve the original diagnostic as
   a regression witness rather than relabeling its old failure as a pass.
4. IR proof and compiled mutants for status branch dominance, no failure output
   load/store, and emergency rethrow before generic exception construction.
   Exercise ordinary non-emergency raise unchanged, plus callback/unwind cases
   without claiming they are allocation-free. Exercise attempted location/irritant
   mutation of reserved exceptions, repeated catch/clear/rethrow and cross-thread
   lifetime, proving unchanged static metadata and no per-call allocation.
5. Fast-path zero allocations and no retained control growth at 1,024/8,192 calls;
   successful setup, failed promotion bytes, map-copy scratch and repeated failed
   setup cost measured separately. Record optimizer flags and hardware/runtime
   provenance; local LLVM22 is not supported LLVM21 evidence.
6. Independent upstream lifetime review, supported upstream CI and explicit new
   toolchain pin acceptance before transformer adoption. Then rerun the full P1
   contract, actual17-node mixed-mode/failpoint/reuse tests, and every inherited
   package/public-isolation/retention gate. The runtime AOT regression cannot
   replace those downstream proofs.

Required root decisions are the sole upstream owner and approval of the complete
P0–P3 scope, including retirement of the postwrite range ABI and the separate
constructor prerequisite. The proposed construction specifies how to prevent failed
map contamination; its implementation and mixed-call tests are still absent. Exact
upstream revision/PR, supported results, measured predecessor impact and the new-pin decision
remain blockers for transformer freeze. This refinement is design only and is not
patch authorization or a claim that the current draft is merge-ready.
