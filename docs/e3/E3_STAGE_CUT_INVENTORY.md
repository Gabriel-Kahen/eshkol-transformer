# E3 §9 stage-cut inventory and noninstalled test contract

Status: **source-only proposal** based on exact L lineage `7322632224f08dc123f585f9893519aa389eb6c2` / tree `c8591bbe14d1602c7d904429ddbd184c9f00e729`. This base includes the noninstalled first-evaluation owner/gradient overlay named below; current remote main `86f4eb36f2fcf079953049d3cdb11377cea5981c` lacks that tuple. No L source, recipe or evidence is changed. No new hook, runtime result, ABI, or §9 acceptance is supplied here. This inventory refines the R1/R2 rows of [the compiled acceptance matrix](E3_SECTION9_COMPILED_ACCEPTANCE_MATRIX.md) and the first-error rules of [the error-lifetime contract](E3_ERROR_LIFETIME_CONTRACT.md). The selected route remains the private bool evaluator in [§9](../E3_EVALUATION_PROPOSAL.md#9-required-references-failure-and-resource-proof).

## Existing source and authority, by exact symbol

| Owner | Existing callable and observer symbols | Boundary and limitation |
|---|---|---|
| E3 Eshkol | `e3-evaluate-into!`, `e3-evaluate-closed!`, `e3-run-one-batch!`, `e3-traverse!`, `e3-stage-set!`, `e3-note-error!`, `e3-note-native-error!`, `e3-note-caught-error!`, `e3-abort!`, `e3-restore-closed!` in `native/e3_private_extension.esk:80-470` | One first-error slot, stages 0–12, inner batch catch, and a normal-return cleanup path exist. None is a stage-selective fault setter or public diagnostic accessor. |
| E3 frame/I2 | `et_e3_private_frame_create_v1` (`src/eshkol_transformer/e3_frame.c:612-703`) creates 37 private tensors, discovers six K1 runtimes and copies epsilon **before** evaluation. `et_e3_private_frame_check_v1`, `et_e3_private_acquire_v1`, `et_e3_private_stage_inputs_v1`, `et_e3_private_stage_targets_v1`, `et_e3_private_stage_mask_v1`, `et_e3_private_forward_role_v1`, `et_e3_private_ce_v1`, `et_e3_private_reduce_v1`, `et_e3_private_correct_v1`, `et_e3_private_accumulate_v1`, `et_e3_private_finalize_v1`, `et_e3_private_cleanup_preflight_v1`, `et_e3_private_drain_v1`, `et_e3_private_publish_v1`, `et_e3_private_finish_v1` are the native evaluation spine (`e3_frame.c:705-1359`). | `e3_scope_begin` borrows real I2 tensors and `e3_dispatch` calls `et_kernel_runtime_dispatch`; no accepted invoke-inside fault. Existing `ET_E3_TESTING` exposes only `et_e3_test_frame_fail_alloc_after_v1` for frame construction, before an evaluable frame exists (`e3_frame.c:87-90,646-652`). I2's `et_f32_tensor_test_fail_alloc_after_v1` is a dependency allocator control, not an E3 stage selector (`native/f32_tensor.c:212-229`). Native first-error status is readable through `et_e3_private_last_error_domain_v1`, `et_e3_private_last_error_category_v1`, `et_e3_private_last_error_code_v1` (`e3_frame.c:217-219`) before cleanup changes it. |
| D2 | `d2-open-dataset-state`, `d2-current-capability`, `d2-token-dataset-next-batch`, `d2-token-tensor-with-view-internal` in `internal/d2/lib/d2_dataset.esk`; `e3-d2-prepare!`, `e3-stage-batch!`, `e3-d2-release-owned!`, `e3-d2-restore-preflight!`, `e3-d2-restore-commit!` in `internal/e3/lib/e3_d2_restore.esk` | `ET_D2_NATIVE_TESTING` defines `et_d2_batch_test_fail_stage_v1` and `et_d2_exact_read_test_fail_stage_v1`, plus live batch/borrow/FD/allocation counters (`native/d2_native.h:149-176`). A later real shard read fault is established; public `token-dataset-seek!` allocates and is excluded from E3's restoration tail. D2's test stages are D2-local, not E3 stage numbers. |
| P1/M3 | `p1-e3-mode-bind!`, `p1-e3-mode-prepare!`, `p1-e3-mode-enter!`, `p1-e3-mode-restore!`, `p1-e3-mode-unbind!` map to trusted slots 64–68 (`native/e3_p1_modes_extension.esk`). | The generated test variant's `p1-e3-test-mode-mask` and `p1-e3-test-fixed-mixed!` observe all 17 real modes (`scripts/generate-e3-p1-mode-test-variant.py:66-68`). The noninstalled mode/graph tuple observes M3 capture/enroll/release through `e3-mode-graph-cell` (`tests/e3_mode_graph/driver.esk:104-132`). These prove specific in-call witnesses, not every failure cut. |
| Native E3/K1 observers | `et_e3_test_retention_probe_v1` (29 selectors), `et_e3_test_model_word_count_v1`, `et_e3_test_model_word_v1`, `et_e3_test_gradient_metadata_v1`, `et_e3_test_seed_gradient_v1` are conditional in `e3_frame.c:1457-1584`; the first-evaluation test overlay adds `et_e3_test_first_eval_corrupt_v1` (`tests/e3_private/retention_first_eval_frame.c`). `native/e3_evaluation_metrics_provider.c:378-433` has `provider_validate` and `provider_invoke`. | These observe/seed authentic owners or inject one model mutation. `provider_validate` can reject malformed direct K1 calls; accepted validated `provider_invoke` is nonfailing. No copy/pin/discovery/preflight/post-provider selector or first-error-clobber hook is present in the private evaluator. |

## Stage map and smallest missing cut

`e3-frame-entry-live` rejects a foreign or released frame **before** `m3-call`, the stage-0 reset, and the mapped first-error path (`native/e3_private_extension.esk:58-69,449-459`). These are pre-entry admission negatives; assert their direct E1 error and unchanged state, without claiming an E3 stage-0 first-error-table cell. For an authenticated live frame that enters the call, stage 0 is initialized before `e3-evaluate-closed!`; later stages are set before their named actions. A stage-selective cut must fire at that exact entered-call boundary, once per armed invocation, and carry the existing mapped E1 stage/category. It must not fabricate logits, CE, metric bits, model state, or a provider result.

`e3-frame-bind!` and native frame construction are a **setup boundary outside stages 0–12**. Their allocation, six-provider discovery, epsilon copy and P1 bind failures require unchanged existing owners and no surviving partial frame; retry binds a fresh frame. They cannot be reported as an in-evaluation stage-3/4 cut. Conversely, native stage 3 pins the already constructed model and does not rediscover providers.

| E3 stage | Current operation/anchor | Existing witness; additional cut needed |
|---:|---|---|
| 0 | Authenticated-frame model/phase/native check, `e3-evaluate-closed!:370-401` | Wrong or busy model/phase and native frame-check denial are entered-call stage-0 cases. Foreign/released frame rejection occurs before stage 0 and stays a pre-entry negative. |
| 1 | `p1-e3-mode-prepare!`, `:403-410` | Real P1 rejection can be tested; a one-shot test-only prepare rejection needs exact token/state preservation proof. |
| 2 | `e3-d2-prepare!`, `:411-415`; D2 profile/remaining-row check at `e3_d2_restore.esk:112-170` | Genuine incompatible/busy/8,193 preflight. Dataset creation/open faults before the evaluator are separate setup negatives; public-seek fault may not stand in for rollback. |
| 3 | `e3-native-acquire`, `:416-432`; native storage check and 14-pin admission at `e3_frame.c:721-752` | Frame-construction allocator is outside the call. Inventory each invocation-owned borrow/pin and add precommit one-shot cuts only where a recoverable status exists. |
| 4 | `p1-e3-mode-enter!`, `:433-448` | Existing mode test variant observes mode state. Distinguish enter denial from provider dispatch preflight in stages 7–12; neither may run inside validated invoke. |
| 5 | `d2-token-dataset-next-batch`, `e3-run-one-batch!:332-341` | Existing D2 read-fail stage supports a later-shard cut after completed batches; record exact D2-local selector and E3 mapped stage separately. |
| 6 | `e3-stage-batch!`, `:342-344`; sequential input/target/mask view/copy in `e3_d2_restore.esk:241-263` | Existing D2 view and native staging errors; add distinct copy and borrow preflight cuts where reachable, never escape a view/shell. |
| 7 | 21-role `e3-forward-schedule-internal`, `:345-346` | Existing mode/graph event observer. Need named role/occurrence preinvoke cuts, including after a prior completed role/provider call. |
| 8 | `e3-ce!`, `:347`; native `et_e3_private_ce_v1` | Need CE preinvoke/preflight cut after complete forward; no L2 `invoke` failure promise. |
| 9 | `e3-reduce!`, `:348`; native `et_e3_private_reduce_v1` | Need L3S preinvoke/preflight cut after completed CE. |
| 10 | `e3-correct!`, `:349`; native `et_e3_private_correct_v1` | Need E3-N correctness preinvoke/preflight cut after completed L3S. |
| 11 | `e3-accumulate!`, `:350-351`; native `et_e3_private_accumulate_v1` | Need preinvoke/preflight cut with earlier completed batches and unchanged pre-call six outputs. |
| 12 | `e3-finalize!`, `e3-traverse!:362-368`; cleanup/publish in `e3-restore-closed!:269-307`, native `e3_frame.c:1214-1359` | Real finite-boundary overflow and publication borrow/preflight rejection are candidates. Cleanup invariant failure is fail-stop; no recoverable cut in the validated drain, D2 restore commit, P1 restore or four-scalar/counter commit tail. |

## Proposed closed test control, pending separate source review

Add only to a **new noninstalled** E3 §9 failure tuple. The test driver arms one authenticated `(stage, site, occurrence)` before `e3-evaluate-into!`; `site` is a closed enum drawn from the stage table, not an arbitrary callback/string. `occurrence` is a positive bounded count of that exact site in one invocation. The control records whether the site was reached and fired, then consumes itself before returning one fixed recoverable failure at a reviewed preinvoke/preflight boundary. Re-arm is required for retry. Unsupported site/stage combinations, zero/over-cap occurrence, stale/foreign frame, active frame, already armed state and unconsumed control reject before model/dataset/output mutation. D2's existing local read setter may be selected by the tuple but remains separate authority and is reset after each call. The test driver never changes production dispatch, K1 validated invoke, D2 cursor restore, or installed facades.

The same tuple snapshots **before** the call: exact saved D2 cursor bytes, all 17 modes, 14 parameter value/gradient images and gradient metadata, tie identities, initializer successor, typed RNG bits, four I2 destination bytes, both counters, native/Eshkol owner/view/pin/FD/arena counts and graph event counts. The postfailure comparison requires the first mapped E1 category and stage, identical state and six outputs, zero active batch/view/pin/frame/graph, and a same-authority retry matching a fault-free twin. Capture native domain/category/code before cleanup can clobber them. Dependency error construction and the final ordinary raise are measured separately from the noalloc restoration tail. Frame-bind failure uses a fresh frame on retry; a frame acquired for evaluation uses that same frame after rollback.

No fault can fire after the first scalar write. Before that commit, final publication must prevalidate all four destination/result I2 views and both counters; prospective failure leaves all six old values unchanged. After drain/restore, a new recoverable cut needs a separately reviewed proof that the frame can still abort and retry. A failure caused by an observer or cleanup invariant is **fail-stop**, never relabeled an E1 rollback pass. Ordinary post-restoration raise may allocate as specified by the error-lifetime contract.

## Isolation and negative matrix for the later implementation

| Boundary | Required negative/control |
|---|---|
| Source/ABI | Canonical private/public source and native object closures, defined/undefined symbols, exports and public strings remain byte-identical; proposed test control appears only in the new tuple manifests. Feature-off compile must lack every new test symbol and observer string. Reject ambient include, duplicate E3 root, forged extern and wrong arity. This document introduces **zero** source/object/ABI deltas now. |
| Selector/authentication | Reject each invalid/stale/foreign/busy/duplicate selector without arming a later invocation. Foreign/released-frame rejection is pre-entry and cannot supply a stage-0 assertion. Verify one fire, consumption, no silent skip and no second fire; distinguish exhausted occurrence from unreached site. |
| Stage and provider | Compile/admit a fault-free twin before each mutant. For an entered call, assert the exact first E3 stage and named preinvoke site; post-provider cuts occur only after a completed prior provider call. Direct-provider malformed K1 and equivalent controls retain their separate label. |
| State and publication | On every recoverable cut, byte-compare pre/post cursor, modes, parameter+gradient/RNG, all six outputs and owner counters; clear the fault and retry. Explicitly reject missed child mode, new graph, escaped view/shell, missed pin/borrow release, first-error overwrite and allocating rollback at the first meaningful assertion. |
| Resource | Run normal, poisoned optimized AOT, repeat and ASan/UBSan/LSan with leak detection on exact reviewed source; measure live/cumulative I2/frame controls and arena/RSS separately. The accepted one-invocation 8,192-batch horizon is a predecessor, not complete-call or all-cut evidence. |

Next gate: root reviews this inventory and the exact closed selectors/test-only compilation topology. A later source implementation must freeze the tuple's literal symbol/manifest delta and prove focused static negatives before one supported compiled failure/retry matrix. L's first-evaluation run and C6 work neither provide nor depend on these proposed controls.

## Exact symbol and preinvoke audit (2026-10-14)

This source audit is pinned to `842c98c273ac7118c50f0d11ba4a6f5d160a08c4`.
It adds no callable or fault control. The table separates an existing test
symbol from a cut the E3 evaluator can actually select for one invocation.

| Owner and exact symbols | Source guard and reachable scope | Missing E3 control |
|---|---|---|
| Frame: `et_e3_test_frame_fail_alloc_after_v1` | `ET_E3_TESTING` in `e3_frame.c:87-90,646-652`; only the 37-tensor frame constructor, before `e3-evaluate-into!`. `et_e3_private_last_error_{domain,category,code}_v1` reads native first-error status before the next native call clears it. | No stage-3 allocation index: `et_e3_private_acquire_v1` checks storage, zeros six sum tensors, then starts the 14-pin lease (`e3_frame.c:721-752`). A pin denial needs a reviewed precommit test seam; it cannot reuse frame construction. |
| D2: `et_d2_batch_test_fail_stage_v1`, `et_d2_exact_read_test_fail_stage_v1` | `ET_D2_NATIVE_TESTING` in `d2_native.c:487-497`; selectors 1-9 are D2 batch/control/view/borrow/mask/publish/setup stages (`d2_native.h:149-160`), read selectors 1-2 are read/close (`:48-50`). Each setter persists until explicitly reset to 0; it does not count E3 occurrences or consume itself. Live dataset, batch, borrow, FD, allocation and byte observers are `:165-176`. | A stage-5 later-shard witness needs a real corpus/read boundary and reset after the call. Stage-6 input/target/mask copy and view cuts need their own E3 occurrence/preflight authority; D2-local numbers must never be presented as E3 stages. |
| P1: `p1-e3-mode-bind!`, `p1-e3-mode-prepare!`, `p1-e3-mode-enter!`, `p1-e3-mode-restore!`, `p1-e3-mode-unbind!` | The private wrapper maps trusted slots 64-68 (`e3_p1_modes_extension.esk:4-13`). The generated `p1-e3-test-fixed-mixed!`/`p1-e3-test-mode-mask` variant sees the authentic token and 17 modes (`generate-e3-p1-mode-test-variant.py:65-118`). | No one-shot prepare/enter denial selector. Binding is setup; restore and unbind are noalloc/fail-stop tail obligations, not recoverable injected cuts. |
| I2/f32: `et_i2_test_fail_alloc_after_v1`, `et_i2_test_reset_allocator_v1`, `et_i2_test_live_builder_counts_v1`, `et_f32_tensor_test_fail_alloc_after_v1` | `ET_F32_TENSOR_TESTING` in `i2_wave2_package_bridge.c:97-147` and `f32_tensor.c:212-229`; the allocation setters arm/reset dependency allocation countdowns, while the live-builder observer counts current copy/reset/decode builders. Neither identifies an E3 site. `e3_scope_begin`/`e3_scope_end` borrow and release actual tensor views (`e3_frame.c:286-300`). | Stage-6 copy/borrow and stage-12 destination/result preflight need named, recoverable preinvoke cuts with all opened views released. Allocation countdown alone cannot identify a logical site or completed predecessor provider. |
| K1/native E3: `et_kernel_runtime_dispatch`, `provider_validate`, `provider_invoke` | `e3_dispatch` opens input/output views, calls K1 and closes all opened guards (`e3_frame.c:913-953`). Forward roles 0-20, CE, L3S and finalize use it; correct and accumulate have direct K1 dispatch with explicit guard unwind (`:1106-1191`). The metrics provider validates at `e3_evaluation_metrics_provider.c:378-392`, then invokes without a recoverable result at `:394-418`. | A site selector belongs before dispatch or in recoverable validation/preflight, never inside accepted `provider_invoke`. Role 0-20 must carry the role plus exact occurrence; CE, reduce, correct, accumulate and finalize each need their own site. A post-provider cut is only valid after a completed earlier call. |

The existing canonical tuple admits `#(packing? skipped-batches)` only
(`tests/e3_private/driver.esk:69-81`). Its bridge exposes destination helpers
and the fixed test run, but no fault arm/status entry. The retention tuple adds
29 owner selectors, exact model/gradient observers and a D2 read setter; it
does not turn those into a stage selector. `scripts/e3-private-package-policy.sh`
accepts only exact root/bridge/rename/export tuples, and
`scripts/build-e1b-consumer.sh` assigns native test flags by tuple. Thus a new
fault control requires a separate, complete noninstalled tuple, including
literal source, defined/undefined/export/string/object/closure manifests and
the selected test macros. It must not be spliced into the base or installed
diagnostic tuple.

For that later tuple, the first static negatives should reject mixed tuple
members, an ambient include root, duplicate E3 root, forged or wrong-arity
extern, and any unlisted symbol or string. Compare the canonical manifests and
feature-off object before and after the tuple addition; require the new symbol
only in the new tuple's localized definition and expected undefined set. The
existing canonical package policy already compares its exact manifests
(`scripts/test-e3-private-package.sh:17-33`) and checks a hostile
private-symbol link (`scripts/test-e3-private-package.sh:139-154`),
but those checks cannot prove an as-yet-undefined fault symbol is isolated.
No symbol or source test is added until the exact selector, ownership and
package topology receive separate review.
