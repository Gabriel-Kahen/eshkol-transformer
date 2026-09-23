# Exact E3 error lifetime contract

**Exact design accepted by [root verdict5771845949](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/104#issuecomment-5771845949)
at reviewed head0a077a7.** This supersedes the proposal-state wording in the
source audit below; behavior is unchanged. Contract merge and explicit bounded
dispatch remain required; implementation and runtime evidence are not accepted.

The active implementation baseline is the reviewed compiler/runtime candidate
`Gabriel-Kahen/eshkol@222cad3aac68ddf48d09c1cdf322fa4c4e7b8296`.
The original contract audit inspected E1 core/internal/consumer code, M3
`m3-call`/`m3t-rethrow-raw`, and the predecessor runtime at
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`; that audit supplied no build/runtime
proof. The proposed prebuilt table is feasible with the corrections below.

## Decision

Use one package-root **13×12=156** prebuilt E1 error table, a same-catch-region preallocated single argument pair, existing `e1-internal-dispatch` for nonallocating predicate/category lookup, and an immediate first-error index in the frame. Preserve first mapped stage and exact A0 category; discard arbitrary original details/cause/exception identity. **Do not `(raise ...)` while any cleanup/exclusion obligation remains.** Return a failure sentinel normally through `m3-call`, then raise the selected root-owned error outside that guard after every state/output obligation is complete.

Six categories are insufficient: E1 admits12, and D2 emits io/corrupt-data/version-mismatch. Existing M3 dependency normalization preserves category, not only six transport-native codes. Fix category order to `#(invalid-argument shape-mismatch dtype-mismatch device-mismatch noncontiguous unsupported invalid-state io corrupt-data version-mismatch determinism-unavailable internal)`. Index is `12*stage + category-index`, with stage0..12. Closed enum comparisons only; never search by caller strings. Unknown/foreign raised value maps to internal at the first current stage.

## 1. Actual E1 construction and package-root ownership

`lib/transformer/error_internal.esk` provides `transformer-error-make/5` through `e1-internal-dispatch 'make`; it is real existing construction authority in the trusted root. E1 core validates/copies bounded fields and registers each opaque shell in its lexical root registry. It does not offer a reservable arbitrary-error promotion API.

Proposed file `native/e3_errors_extension.esk` is loaded by the trusted E3 root after inherited E1 initialization. Its top-level `e3-error-table` initializer allocates vector156 and fills it using `transformer-error-make category 'e3-evaluate-into! fixed-message fixed-details #f`. For each stage use fixed details `source-domain=e3-evaluation`, `source-code=stage`, `source-message=<fixed stage message>`; any extra category detail must also be a fixed declared field, never a native pointer or copied exception graph. No mutable entry/detail escapes; callers only receive registered E1 shells through raise/read-only observation. Existing details accessor returns a copy.

Construct at **package initialization**, not lazily per evaluation or per frame. `native/e1b_error_consumer_bridge.c` actually calls `__eshkol_lib_init__(get_global_arena_shared())` inside its initialization scope. Therefore table, shells, E1 metadata and details are constructed in true root storage before invocation regions. Root construction avoids uncertainty around `register-error` returning its original shell after a promotion barrier. Publish readiness only once the entire156-entry table is complete. Failure means package initialization failed; evaluation is unavailable, not a partially admitted table or fallback. Initialization retry/partial-root retention follows the existing initialization behavior and must be measured, not called zero-retention.

If frame-setup construction is chosen instead, it must run before mutation and explicitly prove root promotion/canonical shell readback; current `register-error` returns `shell` rather than rereading the registry. Package-init construction is the smaller, source-grounded safe choice.

## 2. Existing public readers allocate — use existing dispatch with a local pair

Actual core `transformer-error?` and `transformer-error-category` each construct `(list value)`. Public E1B consumer wrappers additionally construct input/output vectors. Do not call those wrappers from an allocation-free classifier.

Existing `e1-internal-dispatch 'predicate arguments` validates one proper argument then runs `find-entry`: exact-eq registry scan, no mutation/copy. The `category` branch does the same and returns the already-existing category symbol when the predicate succeeded. Its success path allocates nothing in source. Invalid input would bootstrap an error, so invoke category only after successful predicate and preserve the exact pair.

For each catch scope, allocate exactly `(cons #f '())` **in that same region before any operation for which this catch owns acquired state**. In a caught handler:

```scheme
(set-car! inspect-args caught)
(let ((category (if (e1-internal-dispatch 'predicate inspect-args)
                    (e1-internal-dispatch 'category inspect-args)
                    'internal)))
  ;; map the fixed category symbol; save immediate first-error index only
  (set-car! inspect-args #f)
  ...)
```

This is schematic control, not an implementation patch. Predicate/category do not mutate the pair. The pair cdr remains empty and inaccessible to caller code. A frame/root pair is forbidden: storing a younger caught object there could trigger arbitrary graph promotion. The original caught value must never be stored in the long-lived frame/error table.

Pinned `runtime_regions.cpp:1867` proves the relevant barrier branch: already-root values return unchanged, and `val_idx <= dst_idx` (same/older region than pair) returns unchanged. `set-car!` codegen invokes that barrier. Therefore a pair allocated in the current catching region safely accepts a caught value whose lifetime is that region or an ancestor. No evacuation is needed. This relies on catch placement, not a promise that every arbitrary exception is inherently root-owned.

**Nested batch handling:** A pair preallocated only in the invocation region is safe for an invocation-level handler after inner regions have unwound. It is not generally safe to fill it from a handler still inside a younger batch region. Use a fresh region-local pair for the inner per-batch catch, allocated before `next-batch`; classify/release the owned batch while its region remains live, save only immediate index/native status fields into the persistent frame, then leave the region normally with an immediate failure sentinel. This prevents both arbitrary error promotion and released batch-shell escape. An invocation-local pair handles failure opening/preparing a batch region (before that batch has acquired a carrier). D2's own inner shard unwinding may have already promoted its error before E3 catches; that is existing dependency behavior, not E3 allocation-free promotion evidence.

Reset the pair car to#f before its region ends. Confirm by pinned poisoned/allocation-disabled AOT that classification itself invokes no allocator and first native/domain/category/code snapshots survive cleanup.

## 3. Raise itself allocates on the pin

`llvm_codegen.cpp::codegenRaise` at22997 always emits `eshkol_make_exception_with_header`, even for a prebuilt E1 shell. `runtime_exceptions_hosted.cpp:155` allocates an exception header and message in `__repl_shared_arena`. Consequently, `(raise prebuilt-error)` is **not** an allocation-free terminal primitive. The exception dispatcher may also unwind/promote any in-flight younger value; prebuilt root shells avoid that promotion, but not the exception allocation.

Further, M3's `m3-call` catches, clears its guard and calls `m3t-rethrow-raw`; its enclosing `m3t-boundary` can catch/re-normalize again. Those paths call allocating error predicate/operation/details and create/register a new E1 error. They cannot be counted as part of an infallible, nonallocating E3 tail or as stable per-failure root retention.

Exact E3 control contract:

1. Within the shared guarded body, a closed inner handler classifies the first failure using its same-region pair, records immediate table index plus bounded native status fields, and performs all E3-owned view/batch/pin/cursor/mode cleanup without allocation.
2. Preserve model/outer exclusion through state restoration. On failure do not publish any scalar/counter; clear matching model tokens in the validated tail.
3. Return a fixed immediate failure sentinel normally from the body of existing `m3-call`. The existing normal path clears the sole outer guard and returns that sentinel. Do not manually clear the guard early and do not throw the normalized shell through M3's rewrapper.
4. Outside `m3-call` and after all obligation-bearing regions/resources have ended, read the immediate saved index, select the root error and invoke ordinary `(raise selected-error)`. This can allocate, but model/dataset/output state is already restored and all exclusions released. It is error delivery, not a rollback step.
5. On success, six-output publication and model/outer cleanup remain the nonfailing path; no error construction follows success.

Allocation failure during post-restoration delivery cannot be promised to preserve the chosen diagnostic object/category all the way to the caller. State/output rollback is already complete; label delivery-time runtime exhaustion honestly. If exact chosen-error delivery under allocation failure is required, a separately accepted noalloc runtime raise seam or status-return API is needed; the existing pin does not prove it.

## 4. Stage/category, first-error and ownership rules

Stage numbering0..12 must come from the consolidated operation inventory; set the current immediate stage before each possibly failing action. Do not overwrite a nonnegative first-error index once set. Native failure snapshots domain/category/code before any cleanup/native call changes private global status; map the exact admitted native category to A0 using the proposed fixed table. Preserve native raw fields only as private immediate diagnostic state; outward `source-code` is the documented stage, not falsely represented as the native provider code.

Table cells all use operation `e3-evaluate-into!`; they therefore already have the final bounded operation/category/stage semantics. Dropping arbitrary details and exception identity is an explicit E3 dependency-normalization contract, not “preserve original exception.” Cleanup invariant defects fail-stop and never overwrite the first recoverable mapped failure.

Error classification and table selection create no new E1 registry entries after package init. Dependency error construction and final ordinary raise can still retain root/runtime storage per failure. Repeated failure tests must count those existing E1/runtime allocations separately; do not claim flat failure-loop process retention merely because E3 uses a156-entry table. The same applies if any residual path still uses M3 rewrapping.

## 5. Proof obligations

- Table has all156 initialized authentic E1 cells, exact fixed operation/category/stage/cause#f and safe bounded details; no original details/cause copied. All12 A0 categories retain identity through mapping, including D2 io/corrupt/version.
- Poisoned nested batch and invocation regions: local argument pairs survive their catches; no caught pointer reaches persistent frame; selected root error survives all region ends.
- Allocation-disabled classifier and cleanup: both E1 dispatch calls and pair writes allocate zero; unknown/foreign values map internal without reader-bootstrap error.
- Stage first-failure mutation tests, native last-error clobbering tests, and failures after multiple batches preserve exact mode/cursor and all6 outputs.
- Failure returns normally through sole shared guard, clears matching model token exactly once, then raises outside; no post-restoration M3 error rewrap required.
- Count dependency construction and final raise allocations explicitly. Separate preparation/classification/cleanup guarantees from post-restoration error-delivery behavior.
