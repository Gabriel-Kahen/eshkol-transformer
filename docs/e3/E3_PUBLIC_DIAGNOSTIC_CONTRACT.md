# E3 fixed-model exact-report successor

Status: **source-private candidate**. The names below are provisional until an
independent source/package review and supported gate accept an installed surface.
This candidate does not complete E3, implement `trainer-evaluate!`, or revive the
deferred rank-zero scalar report.

## Proposed surface and result shape

The proposed module is `transformer.evaluation` with three fixed operations:

| Operation | Contract |
|---|---|
| `diagnostic-evaluate-fixed! model dataset` | Evaluate the admitted fixed M3 model from the D2 dataset's current cursor through EOS, restore cursor and all 17 modes on success or failure, and return one opaque immutable report. |
| `diagnostic-evaluation-f32-bits report key` | For exactly `loss`, `mask-weight`, `perplexity`, or `token-accuracy`, return the detached IEEE-754 binary32 word as an exact i64 in `0..4294967295`. |
| `diagnostic-evaluation-count report key` | For exactly `tokens` or `batches`, return the detached nonnegative exact i64 counter. |

The result shape is six rank-zero logical metrics. The four floating fields are
bits only: no ordinary binary64 value, rank-zero tensor handle, arithmetic API,
or implicit widening is exposed. `loss` is the ordered global CE numerator divided
by bool mask weight; `mask-weight` is that ordered weight; `perplexity` is finite
`expf(loss)`; `token-accuracy` is selected correct tokens divided by selected
tokens. `tokens` counts true mask positions and `batches` excludes EOS.

The evaluator accepts only the existing E3 fixed CPU-f32 profile: N1/T2/V256,
D4/Hq=Hkv2/Dh2/L1/F8, canonical byte tokenizer, finite unshuffled D2 input, bool
mask, and at most 8,192 remaining batches / 16,384 selected positions. Shapes are
the current private E3 shapes: inputs/targets/mask `[1,2]`, logits `[1,2,256]`,
per-token CE `[1,2]`, and four rank-zero I2 destinations.

## Composition, transaction, and failure

The source-private implementation uses the accepted `e3-frame-bind!`,
`e3-evaluate-into!`, selected loss/weight bit accessor, counter accessor, and I2
rank-zero create/copy-bits/destroy APIs. It reads perplexity and accuracy from the
two remaining bound destinations after successful restoration. The selected-bit
C seam receives the authenticated native pointer from the existing private E3
frame entry; the public Eshkol frame shell never crosses that boundary. It never
constructs an upstream model, dataset, frame, scalar, or cursor substitute.

Before evaluation it allocates and promotes a complete unpublished report record.
E3 alone owns mode/cursor restoration and atomic private destination publication.
After E3 returns idle, the adapter reacquires the existing M3 guard, copies all six
values, unbinds the frame, destroys all four temporary destinations, then publishes
the prevalidated report through an allocation-free registry tail. A recoverable
failure unbinds before destination destruction, abandons staging, publishes no
report, and rethrows the original structured error. Cleanup invariant failures are
fatal rather than reported as successful rollback.

The source-private failure witness records authority counters before and after a
genuine rejected evaluation. Acceptance requires reservations to advance by one,
successful reports to remain unchanged, staging to return inactive, the D2 cursor
to match byte-for-byte, and the caught value to be a categorized transformer error.

Report authentication uses an opaque fieldless condition identity and one hidden
registry. A report retains no model, dataset, frame, tensor, borrow, pointer, or
gradient graph. Unknown/non-symbol/wrong-family keys and foreign reports are
`invalid-argument`; construction reentry is `invalid-state`; exhaustion is
`unsupported`.

## Resource boundary and unsupported cases

The process admits exactly 8,192 lifetime invocation reservations, charged before
any per-invocation report, frame, or destination allocation and never recycled,
including failed calls. Therefore at most 8,192 reports can publish. Published
reports remain valid until process exit; there is no release operation. Each
successful report retains exactly three Eshkol control objects in the report
authority (one fieldless condition, one ten-slot record, one list cell) plus six
immediate payload words. Staging adds one promoted three-slot envelope while a call
is active. Because the root arena does not reclaim individual promoted objects, that
envelope may remain as unreachable retained storage after either success or failure;
failure still publishes no reachable report. Every reservation can also leave the
already accepted inert E3 frame/P1 mode/I2 identity controls after
unbind/destruction. The reservation cap bounds all of these per-invocation controls.

The source-private gate attributes global-arena retention at explicit allocation
boundaries. It measures the complete promoted report/staging graph at the root
publication barrier, then measures inherited transaction retention from immediately
after that reservation through completed unbind, destination destruction and
publish/abandon. The first source-private stats read lazily promotes one isolated
three-slot vector whose referents already have root lifetime; that is the exact
staging-envelope object size. Its one-time bytes remain in the raw outer horizon
delta and are reported separately, then excluded from the caller-witness remainder.
Therefore a
successful reservation attributes the promoted graph minus one envelope to reachable
report authority and one envelope to unreachable staging. A failed reservation
publishes no authority, so its complete abandoned promoted graph is unreachable
staging. `inherited_transaction_bytes` is the separately measured post-reservation
E3/P1/I2 cleanup bucket; on failure it also includes runtime error retention before
the outer rethrow. `witness_other_bytes` is the nonnegative remainder of the outer
arena delta, including caller dataset/result controls and the outer failure transfer.
These are retained global-arena bytes, not native heap totals or per-object live-heap
reachability from a tracing collector.

The pinned runtime records all four buckets at the 1,024 and 8,192 success and
failure horizons, with intermediate successful scaling points retained. Installed/
public acceptance remains blocked on review of the measurements and on a reviewed
exact package closure.

The intended first consumer is a one-shot CLI evaluation process. The candidate
does not support f32 masks, other model/dataset shapes, GPU, mixed precision,
shuffling, BPE, streaming datasets, concurrent calls, callbacks, scalar handles,
generic metrics maps, result release, trainer leases, or contamination checks.

## Source-private package successor

The package candidate uses the exact root
`native/e3_diagnostic_private_driver_root.esk`, bridge
`native/e3_diagnostic_private_bridge.c`, and additional native translation unit
`native/e3_diagnostic_destinations.c`. It source-composes the accepted private E3
tuple and adds no installed facade. Its only new global exports are the two test
driver entries `et_e3_diagnostic_test_run_v1` and
`et_e3_diagnostic_test_failure_v1`; frame, destination, report, and provisional
API authority remains localized.

`scripts/test-e3-diagnostic-private.sh` is the planned supported gate. It first
rebuilds and tests the unchanged base E3 tuple, then checks the diagnostic package's
exact source/native dependency closures, symbols, localization, undefined set and
archive membership. Separate poisoned-runtime processes cover one real success,
EOS rejection, canonical cursor restoration, foreign/non-symbol/wrong-family keys,
1,024/8,192 successful reports, 1,024/8,192 failed reservations, and the rejected
8,193rd success. The retention report records the total delta, reachable authority,
unreachable staging, inherited transaction bucket, caller-witness remainder, complete
promoted graph, calibrated envelope bytes, and the explicit one-time calibration
bytes. The instrumentation exposes no new installed operation and retains only
immediate counters plus one cleared calibration slot.
