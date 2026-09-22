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

## Legacy ABI and rollout boundary

The old `eshkol_region_write_barrier_into` and range/region-escape symbols cannot
be reinterpreted as the new checked ABI. New generated code references the new
versioned symbol; linking it against the old runtime fails rather than silently
falling back. Require an accepted upstream revision, updated toolchain provenance,
and fresh compiler/runtime/package/AOT artifacts; mixing previously compiled
callers into candidate evidence is forbidden.

For the minimal prerequisite, retain legacy ABI symbols and document their existing
unaccepted failure behavior. Do not route the new checked caller through them.
An upstream decision to deprecate or convert old symbols needs its own compatible
caller audit; this proposal is not authority to change their failure policy.
In particular, `eshkol_region_write_barrier_range` is currently called **after**
`vector-copy!` has copied into its destination. Adding a per-slot status or raising
halfway through cannot establish atomicity: earlier writes have already happened.
Whole-range prepublication staging is a separate upstream contract. Likewise,
region-close/result escape, nursery recycle, and nonlocal-exit promotion are not
proved transactional by this single-store interface.

The new checked engine must not consume a persistent map polluted by a failed
legacy promotion. The accepted P1 execution closure must demonstrate that its
publication paths use only the checked engine and that no such legacy failure
continues into them. Upstream rollout review must resolve this coexistence boundary
explicitly; it must not infer safety from a versioned symbol alone. If the actual
P1 failing/rethrowing path necessarily traverses an unsafe legacy promotion,
acceptance stays blocked until that exact additional path is separately addressed.

No local patch to shared `.deps`, alternate native wrapper, raw region pin,
whole-region retention workaround, revised P1 token authority, compiler flag
exception, sanitizer disablement, or timeout increase is part of this proposal.

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

Open acceptance decisions are the exact upstream revision/PR and pin transition,
legacy-map coexistence proof, and measured predecessor impact of the staged map.
They remain blockers for patch/freeze approval, not implied authorization from
this design document.
