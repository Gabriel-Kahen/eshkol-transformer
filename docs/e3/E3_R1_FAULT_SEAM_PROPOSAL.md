# E3 R1 noninstalled fault-seam proposal

**Status: source-anchored proposal for root decision; no hook or compiled
evidence.** The inspected basis is `582c6c7792e7d31ab9bfc66ae140a3a66299574b`
(tree `6f62fc7`) and the pending [§9 matrix](E3_SECTION9_COMPILED_ACCEPTANCE_MATRIX.md#failure-and-rollback-matrix).
Recheck every line anchor against the exact composed E3 tuple selected after the
H capture. Do not promote a cut from this proposal into that tuple by assumption.

## Existing cut inventory

| Boundary | Actual source and current mechanism | Proposed classification |
|---|---|---|
| Frame bind/setup | `src/eshkol_transformer/e3_frame.c:545–707` creates 37 owned tensors, discovers six K1 runtimes at 668–674 and copies epsilon at 679–681. `ET_E3_TESTING` has only `et_e3_test_frame_fail_alloc_after_v1` at 92–95, 651–658; it does not enumerate all owned allocations. | Constructor/setup proof only. A failed bind has no reusable admitted frame; retry with a freshly bound frame. Do not count discovery or epsilon-copy rejection as same-frame evaluation rollback. |
| Model pin/acquire | `src/eshkol_transformer/e3_frame.c:726–752` clears frame scratch, calls `et_g3t_model_pins_begin_internal` at 746–749, then marks model/frame active. `src/eshkol_transformer/m3_call_f32_integration.c:569–591` has an `ET_M3_CALL_TESTING` static pin ordinal and drains its own partial record. | Stage 3 partial-pin rejection can use the existing pin failure only if a reviewed noninstalled entry exposes that exact selector and its predecessor tests establish drain/idle behavior. A new E3 fault must not leave `model->active` or a held pin. |
| D2 copy | `internal/e3/lib/e3_d2_restore.esk:225–264` borrows input, target and mask views sequentially. `src/eshkol_transformer/e3_frame.c:839–900` validates each view, then performs one fixed-size `memcpy` and advances `stage_plane`. | Three distinct stage-6 **pre-copy** orchestration cuts, each after view validation and before its `memcpy`. A `memcpy` has no ordinary failure return; a test rejection must be named pre-copy, never misreported as a failed production copy. Previous scratch writes may exist but caller state must roll back. The setup epsilon copy remains a separate constructor case. |
| Provider discovery/preflight | Discovery is bind-time above. During evaluation `src/eshkol_transformer/e3_frame.c:918–958` opens scoped views and calls `et_kernel_runtime_dispatch`; correct and accumulate have direct dispatch at 1124–1151 and 1154–1215. The accepted validated K1 invoke is nonfailing. | Proposed stage-7..12 preinvoke cut occurs after authentic view preparation but before K1 dispatch. It proves E3 unwind at the provider boundary, not a real provider-validation rejection. Actual malformed/provider admission negatives remain a separate compiled-provider gate. No hook may interrupt validated invoke or change a provider result. |
| After completed provider | Common dispatch releases scoped guards at 957–958 before caller advances role/phase; correct and accumulate release their own guards at 1146–1151 and 1195–1215. | A stage-7..12 post-provider cut is eligible only after successful dispatch **and** all scoped views end, before the caller advances its semantic phase. It may reject the transaction with scratch outputs already written. Keep finalizer's `committed`/staged counts unset when its cut fires. A cut before view release is a separate scoped-release test, not post-provider. |
| Late read | `tests/e3_private/retention_driver.esk:12–13,132–169` uses `et_d2_exact_read_test_fail_stage_v1`, clears it and retries on the same authorities after verifying cursor/output/model/owner state. | Existing partial stage-5 witness; retain as a control. It does not prove copy, pin or provider cuts. |
| Restoration/publication | `native/e3_private_extension.esk:269–319` preflights, drains, restores cursor/modes, then publishes. `src/eshkol_transformer/e3_frame.c:1247–1264,1285–1344` preflights native cleanup and all four destinations/results before the four writes and two counter writes. | Stage-12 recoverable rejection is allowed only before the first caller write. Cleanup invariant failure, invalid release or a defect after the first write is fail-stop, never an injected recoverable E1 error. |

## Proposed control and observer boundary

One closed, noninstalled fixture may arm **one** tuple `(boundary, ordinal,
occurrence)` while the authentic frame, model and D2 dataset are idle. The
boundary is a fixed enum, not a caller string or function pointer; ordinals
refer only to source-reviewed preinvoke/pre-copy sites or the existing pin
ordinal. A matching event consumes the arm exactly once. Disarm and readback
occur outside `e3-evaluate-into!`; a missed arm is a test failure, not a pass.
No hook can supply logits, targets, masks, CE, expected values, provider
outputs, model parameters or a replacement exception. The canonical/private,
diagnostic and installed objects stay byte-identical with the feature off.

A bounded observer may increment fixed counters or append immediate
`(stage, boundary, ordinal, event)` words to a preallocated test-owned buffer;
it retains no view, batch shell, owner pointer, caught object or region value.
It performs no allocation, I/O, callback, E1 construction, reentry or mutation
of production data while the invocation is active. The fixture reads records
only after the call returns/raises and verifies each event against the first
mapped E1 category, source code and stage. Native error domain/category/code
must be copied before cleanup can replace them (`native/e3_private_extension.esk:83–102`).
Root must decide the exact test-only symbol/owner and whether the existing
`ET_M3_CALL_TESTING` pin selector can be composed without widening its ABI.

## Failure and resource proof required of an implementation

For every reachable arm, snapshot before the call all six published outputs,
cursor bytes, 17 P1 modes, 14 parameter values/gradient metadata and present
nonzero or absent gradient words, tie, initializer/RNG, native/Eshkol owners,
FDs and arena bytes. Require the first mapped stage/category, unchanged
snapshot after rollback, no live batch/view/pin/graph/busy frame, and same-frame
retry against a fault-free twin. Constructor cuts instead require zero partial
claim and fresh-bind retry. Run a cut after an earlier completed batch as well
as the first batch. An observer must detect missed child mode, graph creation,
escaped view, pin/borrow leak, clobbered first error and allocation in cleanup,
not merely a zero postcall graph count.

The cleanup tail is nonallocating and terminates on violated invariants;
`native/e3_private_extension.esk:269–316` orders native drain, D2 cursor,
P1 modes, publication and source finish. The accepted 156-cell E1 mapping
retains the first stage/category; ordinary `raise` occurs after restoration
and may allocate ([error-lifetime contract](E3_ERROR_LIFETIME_CONTRACT.md#decision)).
Count dependency error construction and that raise separately. The control
must have constant fixed storage, no per-call owner/root growth, no extra FD,
and the existing 1,024/8,192 normal, poisoned and LSan horizons must measure
caller-region bytes, live/retired controls, payload, peak/retained RSS and time.
No resource ceiling or sanitizer pass is inferred from source inspection.

Negative source/package tests must reject unknown boundary/ordinal, duplicate
or active arming, an unconsumed cut, wrong frame/dataset, cut during validated
K1 invoke, callback/pointer/value injection, observer reads during activity,
symbol escape into canonical or installed packages, foreign/duplicate roots,
and new ambient imports. Compare exact feature-off objects, defined/undefined
and public-export inventories and retained depfiles before any runtime claim.
The numerical F, mode B, retention G and H exact-tuple captures must receive
their own verdicts; this document supplies no §9 R1 or R2 acceptance.
