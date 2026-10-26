# E3 §9 compiled acceptance matrix contract

This is a **pending full-matrix test contract**. Row statuses below cite
separate acceptance evidence where available. It refines the
required gates in [§9](../E3_EVALUATION_PROPOSAL.md) for the current private
N1/T2/V256/D4 bool evaluator. The source basis is
`native/e3_private_extension.esk`, `src/eshkol_transformer/e3_frame.c`,
`tests/e3_private/driver.esk`, `tests/e3_reference/{corpus,reference,transcript}.py`,
and `native/e3_evaluation_metrics_provider.c` at `94b8fecd`/tree `d28cffc1`.
The existing three-batch numerical, mode/graph and complete-call retention
gates are separate prerequisites. The exact `44c0418d` T2 numerical gate is
accepted for the bounded rows named below; mode/graph and complete-call
retention on that source remain pending. No prerequisite proves an unrelated
row by association.

## Evidence rule and frozen records

Every row has one of `compiled-private`, `compiled-provider`, `equivalent`,
`structural`, `unsupported`, or `open`. A compiled-private claim requires an
optimized AOT invocation of the real `e3-evaluate-into!` with authentic M3
model, raw T1 tokenizer, checksummed D1 resource, D2 dataset, E3 frame and
four I2 destinations. A compiled-provider claim exercises actual K1 dispatch
but is not full evaluator evidence. `open` cannot be turned into a pass by
relabelling a synthetic transcript, a parser mutation, a rejected compile, or
a skipped fixture. Each mutant must first pass its own compilation/admission
control and then fail the named value, state or atomicity assertion.

Freeze versioned, canonical, size-bounded development records **before**
looking at runtime output. Keep the existing `corpus-reference`,
`reference-bundle`, `observed-f32`, role TSV and verified transcript schemas
for the current five cases. A new versioned §9 case manifest must name its
independent case ID, model control and parameter SHA-256, T1 artifact and
fingerprint, D1 manifest/shard byte lengths and SHA-256, D2 configuration,
packing and start-cursor bytes, expected batch count, and the exact emitted
input/target/mask rows. Its oracle record must contain every row's 21 named
M3 role shapes and binary64 words, labeled binary32 logits and CE predictions
with their accepted mathematical tolerances,
L3S `(N_b,W_b,loss_b)`, correct `C_b`/active count, serial `(N,W,C,tokens,batches)`
after each batch, final `(loss,weight,perplexity,accuracy)` words, and the exact
end cursor. Distinguish mathematical tolerance for M3/L2/libm from bitwise
binary32 L3S, metric state, counters, masks and cursor. The independent
rational RN-even and Decimal-exp oracle supplies rounding/boundary witnesses;
it must not import production role descriptors, output, or provider code.

The observed record is separately tagged, names the compiled object/executable
and case, preserves raw stdout and stderr, and carries actual row/role/CE/
state/final words, cursor-before/after, six published outputs (four I2 scalar
bit patterns and two counters), mode/owner snapshots and diagnostic stage.
No expected field is copied from an observed record. Validate exact schema,
ordering, duplicate/extra-field rejection and complete row counts. Generate
the frozen oracle/corpus twice under the pinned Q0 Python/PyTorch versions and
compare bytes; seal the generator, independent D2/M3/rational sources, toolchain,
source tree, fixtures, outputs and transcripts. The compiled runtime contains
no Python, PyTorch, fixture-value carrier or unlisted observer.

Build two fresh private source/object/archive/AOT caches with the same
gate-owned stable generated-source pathname; compare bytewise objects,
archives, raw Eshkol and native depfiles, executable and stdout. Keep exact
source/undefined/public-export inventories, canonical-package observer
absence, negative tuple/import/ambient include/duplicate-root/alias tests,
normal and poisoned optimized AOT, and ASan/UBSan/LSan with leak detection.
Runtime stderr must be empty for successes; retain failure diagnostics. The
already accepted source-staging repair changes no numerical comparison rule.

## Genuine D2/M3 cases

`tests/e3_reference/corpus.py` currently materializes five layouts from the
same six bytes `(3,197,0,255,7,7)`. `driver.esk` admits only a two-field
`#(packing? skipped-batches)` scenario, always creates the seed-1729 model,
and returns four destination bits, two counters and two cursor bytevectors.
The accepted `44c0418d` T2 numerical AOT observes all five layouts. The
remaining cases in this table still require their own independently derived
resources and compiled observations.

| Required case/control | Compiled assertion and reachability rule |
|---|---|
| Three distinct batches, repeated/boundary IDs, unequal CE and valid counts, padded last row, nonzero starting cursor | Retain the existing authentic `11,11,10` packed case; run and verify every one of the five emitted layouts, including packed suffix. Assert all six outputs and cursor bytes, not final metrics alone. |
| Packed repartition; unpacked/causal boundary | Compare packed layouts only after independent D2 rows are byte-identical. Freeze separate expected rows/metrics for unpacked shard changes and for any causal boundary change; require the differences where the D2 equations predict them. |
| Empty, singleton, entry EOS, one valid final token | Generate checksummed D1 resources and independent D2 emissions for each. Empty/singleton/entry-EOS may be typed error or no-success cases under the current finalizer's `W>0` rule; assert the actual admitted outcome, six-output atomicity and cursor. Do not manufacture a successful zero-weight finalization. |
| Correct, wrong, tie, uniform and target offset | Search genuine finite raw-byte corpora against a pinned independently initialized model first. Freeze the selected corpus and prove all-correct/all-wrong, lowest-index tie and uniform `log(256)`/perplexity-near-256 from actual emitted logits. If no such model state is reachable through the accepted constructor, a separately reviewed **test-only** parameter-control seam is required before claiming a compiled-private row; direct-provider probes remain labelled provider-only. Nonuniform target-offset must use an input/next-target pair that changes observed CE. |
| Exact 8,192 and 8,193; payload/RSS | Reuse the authentic one-invocation 8,192 pattern and pin independently counted D2 rows. `internal/e3/lib/e3_d2_restore.esk:112–158` computes remaining rows from authentic D2 core fields 14–15 and rejects 8,193 at stage 2. `native/e3_private_extension.esk:402–432` calls this after P1 prepare but before native acquire/mode enter/traversal; assert exact stage-2 `unsupported`, unchanged caller modes/cursor and six outputs, no 8,193rd batch or publication. The later `e3-traverse!` cap is defensive, not the primary one-over witness. Record normal/poison/sanitizer payload/RSS and measured limits separately. |

For N1/T2 causal D2, successful masks are active prefixes `11` and `10`
(`tests/e3_reference/README.md`; `tests/e3_reference/corpus.py`). `01` is not
an authentic emitted row, and `00` cannot successfully finalize. Wrong targets
outside `[0,255]`, nonfinite masked logits, subnormal/negative-zero/fenv inputs
and arbitrary CE boundaries are not established by the seed-1729 model's
finite forward path. Keep these as compiled-provider admission/numeric rows
unless a genuine controlled M3 state and independent forward oracle demonstrate
reachability. A direct provider outcome never silently discharges a private
model traversal requirement.

## Compiled numerical mutations and controls

The accepted bounded private compiled kills include wrong-target CE,
mean-of-means, second normalization, wrong vocabulary axis, capacity
denominator, exp-before-global, and missing batch. Changed-source sum-order
is accepted as predecessor evidence. Extend the exact noninstalled tuple with
one named mutation at a time; do not alter canonical, diagnostic or installed
archives. Every value kill compares to a prior frozen independent transcript
and identifies the first different row/word. For a corpus-specific mutation,
first prove that the chosen authentic case makes the mutation observable.

| Mutation | Required first meaningful assertion |
|---|---|
| Input instead of next target; target offset | Per-position CE against real D2 target and actual logits, then L3S state. |
| Physical-capacity denominator; count padded position | `W_b`, active count and `C_b` on an authentic `10` row; final W/tokens. |
| Mean of batch means; second normalization | Serial `N,W` and final loss on unequal CE/count batches; no equal-count surrogate. |
| Missing or doubled batch | First divergent batch counter and serial N/W/C; require traversal to have emitted the expected third row. |
| `>=` tie winner; wrong vocabulary axis | Correct count on genuine tie/axis-distinguishing M3 logits. If unreachable with admitted model authority, retain the provider kill as provider-only and hold the private row open. |
| Exp before global aggregation | Final perplexity from global `N/W`, with a genuine unequal-loss witness. |
| Clamp overflow or f64 retry that changes outcome | At an actually reachable finite boundary/overflow row, compare typed failure/atomic output or changed final word. An identical-error retry is an equivalent control, not a kill. |
| Observable sum reorder or retained wider accumulator | A frozen three-term RN-even counterexample through genuine CE numerators. The synthetic/provider `64,2^-18,2^-18` witness proves the arithmetic distinction only. The separately accepted `ad4ad9ac` private gate supplies a genuine predecessor witness; an exact-`44c0418d` compiled carry remains open. |

Run provider-level malformed shape, dtype, device, noncontiguous layout,
index, mask, alias, metadata and fenv controls through actual K1 dispatch;
assert immutable inputs and byte-atomic outputs on early and late rejection.
Preserve separate source/IR review for no binary64, no hot scalar Eshkol loop
and no FMA outside accepted primitives. Two-term reversal, exact-f32 `W` versus
token-count conversion, canonical bool `0/1` FMA, local f64-add-then-cast and
identical-error f64 retry are explicit equivalence controls when outputs and
errors are identical; they must compile/run and compare equal, never be
reported as numerically killed. The existing independent M3/L2/L3S gradient
gates remain predecessor evidence; E3 introduces no backward promise.

At exact `44c0418d`/tree `70706c0a`, a bounded development sweep encoded and
parsed 513 checksummed C1 states with consecutive `norm_final/weight` f32 words
`41b17113..41b17313`, retaining the fixture's tied head and authentic D1/D2
`[3,0] -> [4,0]`, mask `10` row. The pinned mathematical M3/CE reference
produced 513 distinct final loss words. A local compiled C/libm probe found no
`expf(loss)` versus `(float)exp((double)loss)` word difference across them.
The adjacent boundary states `41b17251` and `41b17252` yielded losses
`42b17217` and `42b17218`: both calculations gave `7f7fff84` in the first
case and `7f800000` in the second. This is a bounded equivalent-result
disposition for these mathematical candidates, not a compiled-private E3
mutant result or a global equivalence proof. The existing private
finite/overflow fixture expects losses `42b0ffc6`/`42b1ffc6`; the latter's
f64 exponential likewise casts to infinity in the local probe. The f64-retry
M2 row stays open for a nonfinite-triggered retry that changes an outcome.
The sweep and source identity are retained under
`/home/gabe/.codex/evidence/eshkol-transformer/e3-m2-f64-retry-scout-20261025/SHA256SUMS`;
the gate above is not claimed by that evidence.

A separate finite-path f64 **recompute** candidate exists on this same
mathematical C1/M3/D2 route. In a new 257-word gamma sweep
`41ad01eb..41ad02eb`, state `41ad026a` produced C1 SHA-256
`3f6f6354655198266a42f508757c88307292692c1aebea1a659c7b8772b6e860`
and final loss `42ad0231`. On local GCC 16.2.1/glibc 2.44,
`expf(loss)` returned `7ddec091` while `(float)exp((double)loss)` returned
`7ddec090`; the independent 100/180-digit Decimal oracle rounds to
`7ddec090`. The one-at-a-time source proposal replaces only the finalizer's
`const float perplexity = expf(loss);` with
`const float perplexity = (float)exp((double)loss);` in a noninstalled,
hash-bound test overlay. Because `expf` is finite here, this is not a kill of
the nonfinite-triggered f64 **retry** row. On the same local host, an exhaustive
524,289-word scan `42ad0000..42b50000` found zero cases where nonfinite
`expf` became finite after f64 exponential and f32 cast. Both results are
bounded host controls, not a global equivalence proof. Before a finite
recompute mutant gate, the supported Ubuntu 22/LLVM 21 lane must show this
exact normal-versus-overlay word difference on the frozen C1/D2 fixture, then
independent normal/repeat/poison/sanitizer review must identify the first
changed final metric without accepting a mutant transcript. No such compiled
private run has occurred. The frozen candidate, Decimal check, host
compiler/libm identity and scan counts are sealed at
`/home/gabe/.codex/evidence/eshkol-transformer/e3-m2-f64-retry-extended-20261025/SHA256SUMS`.

## Failure and rollback matrix

For every recoverable injected cut, snapshot **before** the call: exact D2
cursor bytevector, all 17 P1 mode bits, all 14 parameter value words and
absent/present nonzero gradient words plus metadata, tie identity, initializer
successor and typed RNG bits, four caller I2 scalar bytes, two E3 counters,
native/Eshkol live/retired owner controls, FD count and arena used bytes.
Compare after failure: same exact cursor, modes, model/grad/RNG and six outputs;
no live D2 batch/borrow, I2 view, pin, graph owner or busy/active E3 frame.
Assert the **first mapped** E1 category and stage `0..12` with fixed operation
and source-code; the contract preserves mapped category/stage, not arbitrary
original exception identity/details. Clear the injected fault and retry with
the same dataset/model/frame/destinations; compare all six outputs and end
cursor against a fault-free twin. Count any dependency error construction and
post-restoration raise allocation separately from the noalloc cleanup tail.

| Actual boundary | Cut and required observation |
|---|---|
| Admission, P1 mode prepare (stages 0–1) | Wrong/foreign/stale/busy model/frame/destination authority and test-only P1 rejection; no transaction publication or changed modes. |
| D2 prepare/cursor authority (stage 2) | Authentic open/prepare fault and incompatible/busy cursor; preflight must leave saved cursor and six outputs unchanged. Public seek failure injection is **outside** E3's restoration tail and is a negative scope control. |
| Frame acquire, allocation, copy, pin and provider discovery/preflight (stages 3–4) | Exhaust each actual invocation-owned allocation index and each independently reachable copy/pin/provider admission cut; capture first native status before cleanup clobbers it. Frame-bind/setup allocation occurs before an evaluable frame exists: assert no surviving partial frame/destination claim, then retry with a freshly bound frame rather than pretending same-frame retry. A missing cut mechanism is `open`, not proof by a neighboring fault. |
| Later shard read and batch staging (stages 5–6) | Real fault after earlier successful batches, including late shard hide/restore; exact cursor and all six pre-call outputs survive, then same-authority retry succeeds. |
| Each forward role, CE, L3S, correct and accumulate (stages 7–11) | Test-only preinvoke/preflight failures before and after an earlier **completed** provider call; require named stage and complete rollback. Do not inject failure inside an accepted validated K1 invoke: its real contract is nonfailing. |
| Finalize, numeric overflow, cleanup preflight and final metric publication (stage 12) | Boundary overflow/invalid final metric must reject before any of four output writes; byte-atomic publication and original counters. Cleanup invariant failure is fail-stop, not a recoverable E1 error. |

An overflow value generated only by a direct provider input remains a
`compiled-provider` atomicity row. The private overflow row needs a real model
and D2 execution that reaches the boundary, or stays open; a hook that writes
the desired final metric would not meet it.

Include mutation witnesses for graph-producing evaluation, a missed child mode,
restoring a generation counter instead of its original value, escaped view or
shell, skipped pin/borrow release, overwritten first error, and allocating
rollback. Each witness must be rejected by an in-call event or exact state/
owner/arena assertion, not just a zero postcall graph count. The P1 token and
retention observers are precedents for modes/gradients/owners, but their
separate gates do not fill these stage cuts automatically. Preserve the
accepted 156-cell first-error table and after-restoration ordinary raise
semantics in [the error lifetime contract](E3_ERROR_LIFETIME_CONTRACT.md).

## Test-only seams and completion gate

The present `driver.esk` cannot choose a model parameter state, initialize
different destination sentinels, or request a stage cut; its scenario has only
packing and a skip count. The current native frame test macro exposes one
allocation countdown, not copy/pin/provider-preflight or post-provider cuts.
These are **upstream testability gaps for root review**. First assess the
registered idle-model `module-state-dict`/`module-load-state-dict!` path already
used by `tests/m3t/public_runtime.esk` (lines 139–140, 208–211); prove it
preserves exact M3 parameter identity, shapes, tied weight, leases and restored
bytes before using it for a zero/tie model. If it cannot express the required
state within the fixed source-composed package, propose a narrow test-guarded
real-owner parameter control with geometry, identity, lease and extent
preflight. Either route must run the actual M3 forward and independent oracle;
no hook may manufacture logits or CE. Before source work, inventory existing
D2, P1, I2, K1 and frame hooks by exact symbol and ownership; propose only
closed noninstalled scenario/observer and one-shot fault controls at proven
preinvoke boundaries. Do not introduce public/production/compiler ABI, alter
K1's nonfailing invoke, or let a test hook supply expected numerical values.
Expand source/defined/undefined/export/closure manifests and static negative
tests for every added test-only symbol; canonical object must remain
observer-free. Root must review any parameter-control contract before source
implementation.

The named next gates keep each still-open obligation visible:

| Row | Current status | Next proving gate |
|---|---|---|
| N1 three-batch exact roles/CE/states/outputs | Compiled-private T2 predecessor accepted at `44c0418d`/tree `70706c0a`: exact roles, CE, serial state, six outputs and cursor in normal A/B, poison and sanitizer. The earlier failed F attempt remains nonaccepting. Independent consumed-source closure review carries this accepted bounded result from `44c0418d` to docs-only `b4b82d2b` without a new compile (`t2-f-carry-44c0418-to-b4b82d2-independent-review-20261025/REVIEW.md`). | CPU4/current integration needs its own affected gate; no full §9 claim follows from source carry. |
| N2 five authentic layouts, cursor/repartition/boundary | The same independently reviewed F gate accepts all five genuine layouts, their D2 rows and the packed equivalence/different-layout assertions across 20 compiled streams. The independently reviewed consumed-source closure carries this bounded result to docs-only `b4b82d2b`; no new target compile ran. | New profile or D2 changes require fresh evidence. |
| N3 correct/wrong/tie/uniform/target offset | Bounded target-offset leaf passed F compiled gate and independent result review on exact `7bb38a56c` source/tree `370b6365`. A source-closure carry audit found three changed native files consumed by that gate at `44c0418d` (`e3_frame.c`, `e3_frame_internal.h`, `m3_call_f32_integration.c`), so that older F result is predecessor evidence. A fresh noninstalled `b482c9b`/tree `039d1b2c` target-offset compiled gate passed independent result review (`n3-target-offset-b482c9b-v3-independent-result-review-20261025/REVIEW.md`; 324-file raw seal `51a5df8b...`). Its consumed production/native closure equals `44c0418d`; genuine D2 `03c500`/`03c507` cases matched the independent oracle in normal A/B, poison and sanitizer, and the compiled wrong-target mutant failed first at `batches[0].ce_f32[0]` without a report. Exact `44c0418d` compiled all-correct/all-wrong A/B, poison and sanitizer passed independent result review (`correctness-a-44c0418-attempt2-review/REVIEW.md`, raw seal `8d984330...`), including the wrong-vocabulary-axis kill at first correct count. The uniform successor `3d59ad5` stopped before AOT on four unpinned actual imports and is nonaccepting. Tie/uniform and full N3 remain open. The `3aade9b1`/tree `dfb8fb3d` uniform/tie attempt 6 is independently NONACCEPTING (`n3-uniform-3aade9b-v5-attempt6-independent-review-20261025/REVIEW.md`; 236-file raw seal `22de375d...`). Q0 references, canonical and four private packages, plus normal A/B object/archive equality completed; the first AOT failed at `uniform_runtime.esk:44` on unknown local `e3-uniform-set-files!`. No runtime, sanitizer or tie mutant executed. Common Git config also drifted during the run, invalidating host custody. | Repair the test-only scenario/file interface, repin host Git identity and run one fresh reviewed compiled gate; then audit the bounded N3 union and remaining mixed-correctness gap. |
| N4 `01`, masked invalid, fenv, subnormal/signed-zero | The exact-`44c0418d` `make test-e3-metrics` attempt 6 passed independently reviewed provider/bridge numeric, fenv, malformed/atomicity, mutation, private numerical AOT and C provider/bridge ASan/UBSan/LSan checks (45-file raw seal `936cd9d2...`; `metrics-44c0418-attempt6-independent-review-20261025/REVIEW.md`). Its AOT bridge does not execute the private evaluator/frame/model, and temporary binaries/logs were removed. Attempt 5 guest PASS was host NONACCEPTING due an unrelated common-Git-config write; attempt 6 restored exact host custody. The [exact-44c source audit](E3_N4_PRIVATE_REACHABILITY_AUDIT.md) finds authentic D2 cannot emit mask `01` or an invalid target and E3 checks bounds at stage 6. A reviewed source/host audit at `03b698a7` finds the tied-head N3K four-term `+0` accumulation cannot produce negative-zero K1 logits under RN-even/FTZ-off/DAZ-off; unsupported external fenv rejects at stage-7 role 0 before K1. Genuine C1/M3 subnormal, masked-nonfinite and sticky-flag private witnesses remain unproven. | Retain `01` and invalid-target as provider-only unless a reviewed D2 seam is added; prove remaining private N4 cases with a genuine C1 state, independent M3 forward oracle and exact private AOT. Do not carry the C sanitizer result as full private-evaluator sanitizer. |
| N5 empty/singleton/EOS/one-token/8,192/8,193/resources | The `943b745` noninstalled edge/cap attempt 3 is independently authenticated but NONACCEPTING (`n5-edge-full-943b745-v5-attempt3-independent-result-review-20261025/REVIEW.md`; 229-file raw seal `f6691719...`). Its first normal cap process aborted with signal 6 after 1,239.90 seconds and 158,592 KiB peak RSS, before a result. The exact `0a86c35` stage-marker import discovery measured four output imports in normal and sanitized E1B objects, with unchanged sanitizer instrumentation (143-file raw seal `c7f8f2c8...`). The measured successor `2a39341c`/tree `971b5379` compiled canonical, normal A/B and sanitized packages and six cap/edge AOTs, but its normal A runtime aborted with signal 6 after the 8,192 control evaluation returned and before its batch/token/cursor checks completed (`n5-stage-2a39341-full-attempt2-independent-review-20261025/REVIEW.md`; 230-file raw seal `d1fb1653...`). Seed checks completed; no stage-2, edge, repeat, poison or sanitizer runtime ran. Its 1,255.59-second time and 155,360 KiB peak RSS belong to the failed process. The exact failing check is unproved. The test-only counter-order successor `bd2af693`/tree `b6ce66b2` corrected the private native publication selectors: selector 0 reports tokens and selector 1 reports batches. Its independently reviewed full attempt 2 passed six authentic empty, singleton, entry-EOS, one-token, 8,192 and 8,193 cases in normal A/B, poison and sanitizer (`n5-counter-order-bd2af693-full-attempt2-independent-review-20261025/REVIEW.md`; 259-file raw seal `3a29276d...`). The authentic corpora recorded 8,192 remaining rows/16,384 active tokens and 8,193 rows/16,386 tokens; E3 accepted the 8,192 case and rejected 8,193 at stage 2 with six outputs and unchanged cursor. All 15 stage markers and owned-container checks passed; sanitized cap peak RSS was 446,308 KiB under the 524,288 KiB limit. This accepts the bounded private N5 cap/edge/resource leaf, not full §9. | Bounded N5 is accepted. Carry only across independently reviewed source closure; full E3 §9 still needs the other open numeric, failure/retention and trained-path sanitizer gates. |
| M1 wrong target/mean of means | Both mutants compiled, traversed the actual evaluator and were rejected by the independent T2 numerical validator at `44c0418d`; their first divergent CE/state words are retained. The independently reviewed consumed-source closure carries this bounded result to docs-only `b4b82d2b`; no new target compile ran. | Other M1 profiles remain open. |
| M2 denominator/batch/normalization/axis/padding/tie/exp/overflow/reorder | The `44c0418d` F gate additionally kills second normalization at final metric words. Independent compiled reviews accepted two direct-44c noninstalled successors: capacity denominator `248f56c` first diverges at the authentic final `10` row's reduction weight `3f800000`→`40000000` (its correct count remains zero, so this does not prove a changed nonzero `C`); exp-before-global `af8cde4` first changes only final perplexity `43815d49`→`438160e2` (921 ULP). Both pass normal/repeat/poison/sanitizer controls and reject four mutants without transcript (reviews `m2-capacity-denominator-248f56c-independent-review.md`, SHA `fb3d7cf4...`, and `m2-exp-before-global-af8cde4-independent-review.md`, SHA `d1a46768...`). The independently reviewed noninstalled missing-batch successor `b396761`/tree `56f3213` passed a supported f31/fe9 normal A/B, poison and ASan/UBSan/LSan gate (370-file raw seal `bbbbb0a9...`, review SHA-256 `431f61d8...`). An authentic third D2 row kept input/target/mask/logits/CE/correct identical, while the mutant retained prior N/W and counters: normal `41de3b31`/`40a00000` and `[5,3]` versus mutant `41b195f4`/`40800000` and `[4,2]`; four mutant modes rejected first at `batches[2].state_f32` without a transcript. Cursor equality was checked but its raw bytes were not separately published. This accepts that bounded leaf, not exact-44c or full §9. The exact noninstalled sum-order subgate was separately accepted at `ad4ad9ac`/tree `aa17f1bf` by independent compiled RESULT review (`e3-sum-order-ad4ad9ac-a-independent-result-review-20261014/REVIEW.md`, SHA-256 `a58151c84401527909d63b2fb6f2d3cf5c4fc86b28c9008883e24b835fd5a26f`): genuine three-batch normal/repeat/poison/sanitizer runs passed and four mutants rejected first at third-batch numerator without transcript. That result is predecessor evidence because consumed native files changed by 44c. The earlier df9546 C mathematical witness failed and remains preserved. Direct provider/composition kills are separate. | Run and review the remaining one-at-a-time M2 mutants. Full §9 matrix remains open. |
| M3 equivalent controls and native malformed/atomicity | The exact-`44c0418d` metrics attempt 6 passed supported f31/fe9 with eligible host custody and independent review (`metrics-44c0418-attempt6-independent-review-20261025/REVIEW.md`; raw seal `936cd9d2...`). It reports 70,231 native and 135 fenv checks, three real BOOL L2/L3S/I1/I2 batches, rational/Decimal references including 102 exp inputs at 0 ULP, 12 numeric plus five admission/atomicity and five orchestration mutant kills, five equivalent controls, two fresh private numerical-bridge AOTs and C provider/bridge ASan/UBSan/LSan. This accepts the bounded exact-source provider/bridge gate. The bridge has no evaluator/frame/model authority and guest cleanup removed temporary AOT/sanitizer artifacts; private evaluator M3/full §9 remains open. Attempt 5 is preserved as a host-custody HOLD despite guest PASS. | Carry only the exact provider/bridge result; complete private evaluator equivalence and source/IR review with an authentic model traversal, then full sanitizer on that trained path. |
| R1 allocation/copy/pin/provider-preflight/read/final cuts and full rollback | Open; late D2 fault is a partial witness. The independently reviewed `8fec18f2` direct-D2 diagnostic passed its bounded compiled gate (`retention-d2-only-8fec18f-attempt1-independent-review-20261025/REVIEW.md`; 228-file raw seal `80a8a114...`): eight fresh fault/control cases exited zero, direct D2 faults retained 2,288 root-arena bytes through genuine E3 retry, and controls stayed flat. The prior full E3 fault retained 2,360 bytes; the 72-byte difference is not allocation attribution. The independently reviewed `d1f7b743` scoped-observer diagnostic passed eight compiled fault/control cases with public I2 counters zero, positive balanced private scoped counts, zero live guards and passing report/teardown postchecks in normal A/B, poison and sanitizer (`retention-scoped-d1f7b743-attempt1-independent-review-20261025/REVIEW.md`; 229-file raw seal `615aacb1...`). Its fault arms still retained 2,360 root-arena bytes and did not run complete-call arena equality. | Compare D2 and full-E3 fault-path growth without attributing the difference, repair root-arena retention, then run reviewed one-shot stage cuts and complete private failure/retry matrix. |
| R2 modes/grad/RNG/owners/graph/event/resource horizons | The noninstalled compiled mode/graph successor `dc565e1`/tree `67bee60a` passed independent result review (`mode-dc565e1-attempt6-independent-review/REVIEW.md`; 406-file raw seal `da61d695...`, review seal `91747722...`). Normal A/B, poison and ASan/UBSan runtime stdout matched for absent-gradient and seeded-present-gradient states; in-call mode/graph mutants were killed at their first assertions, and a fail-stop control exited 134 without success. This accepts bounded mode/graph and gradient-image preservation, not peak RSS or full R2. E3 evaluation is forward-only and has no VJP, reset, or gradient contribution contract; accepted M3/L2/L3S numerical-gradient checks remain predecessor evidence (`r2-gradient-contract-independent-review-20261025/REVIEW.md`). Its historical `/out:rw` mount weakens adversarial receipt isolation. Exact-source mode attempts 3/4 remain nonaccepting predecessors. Complete-call retention attempt 2 remains NONACCEPTING at `failure-arena-bytes` (`retention-44c0418-attempt2-review/REVIEW.md`). The diagnostic successor `4ab378a3` reused sealed packages and sampled eight fresh fault/control processes: all exited 134 at `balanced-i2-views-pins`; the fault arm alone gained 2,360 arena bytes after-call, while controls stayed flat yet failed the same predicate (`retention-arena-phase-4ab378a3-v2-attempt1-independent-review-20261025/REVIEW.md`). The first `c6f7217b` raw-I2 probe gate stopped before AOT on unpinned `fflush` (81-file raw seal `9562a62e...`). A corrected `6cc2d16d` import discovery measured the exact normal and sanitized inventories, but shared `.git/config` drift made its host result noneligible (142-file raw seal `4423fdd2...`). The reviewed `ef126033`/tree `c0b3a958` compiled diagnostic passed import, A/B object, AOT, and libc link checks, and sealed eight fresh fault/control cases (`retention-raw-i2-ef126033-attempt1-independent-review-20261025/REVIEW.md`; 228-file raw seal `f2855c4e...`). All eight printed public I2 begin/view/end counters zero before/after and then aborted at `balanced-i2-views-pins`; the assertion's positive-begin clause is false in fault and control paths. Fault arms alone retained 2,360 root-arena bytes through retry; allocation source is unproved. This accepts a bounded diagnostic only, not retention, full R2, or sanitizer. | Correct the observer contract while keeping owner/leak checks; independently trace the fault-path arena increase, then rerun complete-call retention, stage-selective rollback, resource horizons, and full sanitizer. |

The `1d916ffd` uniform/tie repair attempt 1 is independently NONACCEPTING
(`n3-uniform-1d916ff-attempt1-independent-review-20261026/REVIEW.md`;
248-file raw postexit seal `80ce2b64...`). All four private packages and AOTs
built with source and container custody intact. Only normal A runtime ran;
it exited zero with zero-state and numerical runtime markers, but the host
parser rejected its saved cursor before B, poison, sanitizer or tie-mutant
runtimes. The compiled D2 config sets `maximum-batch-bytes=34`, while the Q0
reference used its 1,048,576 default; that field is part of cursor identity.
Align the test-only reference to the authentic D2 bound, independently review
it, and run a fresh one-use gate. This receipt admits no N3 uniform/tie leaf.

The exact-`44c0418d` noninstalled sum-order compiled-private leaf passed
independent result review (`m2-sum-order-44c0418-attempt4-independent-review-20261025/REVIEW.md`;
435-file raw seal `afcfeec5...`, review seal `239ce432...`). Normal A/B, poison
and sanitizer results matched byte-for-byte; four compiled regrouping mutants
first diverged at the genuine third-row state numerator `41de3b31`→`41de3b30`
and published no mutant transcript. The wrapper classified only the exact
retained 291-byte Q0 NumPy warning. Attempt 3 remains host NONACCEPTING.
This admits the bounded current-source sum-order leaf, not all M2 or §9.

The independently reviewed `2e941bb7` mapper diagnostic passed a bounded
compiled 12-case gate (`retention-map-2e941bb-attempt1-independent-review-20261025/REVIEW.md`,
244-file raw seal `085cb296...`). Direct and mapped D2 faults each retained
2,288 root-arena bytes through retry, with flat controls. Adjacent mapper
samples showed zero **net active-region** used-byte change; they do not
attribute allocations or explain the separately measured full-E3 2,360-byte
fault growth. R1/R2 retention and full sanitizer remain open.

The source-reviewed `7827b8d` D2 fault-envelope repair has no compiled
retention result. Its first attempt is independently NONACCEPTING
(`retention-d2-escape-7827-attempt1-independent-review-20261025/REVIEW.md`;
19-file raw POSTEXIT seal `091f263c...`). The first private package stopped
before AOT on a stale test-generator D2 predecessor hash/size: repaired
`d2_dataset.esk` is 45,388 bytes versus the manifest's 45,336. A missing
guest Git alternates mount also warned. No paired runtime or canonical
complete-call arena assertion ran; repair effect remains unmeasured.

The manifest-repinned standalone successor `2b77481d`/tree `8c43cc62`
completed four private packages and AOTs, but its independently reviewed
full attempt 2 is NONACCEPTING (`retention-d2-escape-2b77481-attempt2-independent-review-20261025/REVIEW.md`,
330-file raw seal `daefe43e...`). Twelve direct/mapped D2 fault and control
runtimes exited zero across normal A/B, poison and sanitizer; fault arms
still retained 2,288 root-arena bytes through genuine retry while controls
were flat. Eight full-E3 fault/control runtimes aborted first at the invalid
public `balanced-i2-views-pins` assertion; fault arms retained 2,360 bytes,
without attribution of the 72-byte difference. The canonical 1,024-call
absent-gradient runtime aborted at `failure-arena-bytes` after 512 calls.
Retain R1/R2 as open. A fresh phase probe must distinguish D2 promotion,
exception construction, envelope and re-raise before another repair claim;
the full-E3 scoped observer and canonical equality need separate proof.

The `761cb395` one-shot R1 stage-3 attempt 1 is independently HOLD/nonadmitting
(`r1-stage3-oneshot-761cb395-attempt1-independent-review-20261026/REVIEW.md`;
85-file raw postexit seal `33d2b782...`). Corpus generation, normal fault
package/object/archive and normal arena-object build passed. Sanitized package
construction then failed because the wrapper set the retention-only sanitizer
selector for a fault tuple. No AOT or stage-3 normal, poison or sanitizer
runtime ran; no rollback, twin retry or pin-cut result is admitted. The owned
container was removed, with three absence checks. Repair the test wrapper and
freeze a new packet before a fresh compiled gate.

The accepted T2 numerical F evidence is
`/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/e3/numerical-a-review/REVIEW.md`:
supported f31/fe9 exit 0 after 3,750.864 seconds; 510-file raw seal
`f6b04ae125ae58710893ef5c32d3420de7063fabd2f844a140046e1c01d6e2c1`.
Its five layouts, three mutant kills and sanitizer observations are bounded to
that T2 source. Candidate-ON CPU4 and the full trained-path sanitizer remain
unadmitted until their own exact-source runs and independent result reviews.

The separately sealed original-source terminal-control exit 1 at `95070d729`
is classified against repair-only `51d5cc9c` PASS 62 in
`/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/terminal-managed-cell-ir-asan-control-trace-review-20261024/REVIEW.md`.
The original ASan trace strongly matches the predicted native-clone
borrow-header probe without final allocator/compiler fault attribution;
the repair-only result is a focused pass. Neither run was repeated for this
classification. This pair does not admit E3 §9, candidate-ON CPU4, or the full
trained-path sanitizer.

The trained-G2 sanitizer attempt at `55a56f2f` is separately authenticated as
NONACCEPTING in `/home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/team/integration-review-trained-g2-san-55a56f2f-failure/REVIEW.md`:
the first two-token run reports an ASan one-byte read seven bytes before a
48-byte terminal-ID clone allocation. Its partial SAVE/E3 prefix does not admit
the full trained path. The managed-cell focused successor at `266f2a5` is also
NONACCEPTING (`cpu4-trained-g2-san-managed-266f2a5-focused-v3-attempt1-independent-review-20261026/REVIEW.md`;
392-file raw seal `a2686963...`): all 37 native sanitizer units compiled,
but Clang hit the explicit 2,400-second bound while compiling the 1.73-GB
generated ASan IR. No sanitized evaluator object, linked runtime, two-token
case, or first-EOS case ran; source and container custody passed. The original
clone trace therefore remains unresolved for the trained path. A changed-bound
focused successor, independent result review, and then a fresh frozen full-path
gate remain separate prerequisites.

Completion requires independent source-and-test review, frozen oracle and
mutation inventory, every row's supported compiled proof or explicit accepted
unreachable/equivalent disposition, all pre/post state checks, and fresh
exact-source normal/poison/sanitizer gates after the numerical, mode/graph and
retention prerequisites. Report unsupported boundary rows rather than calling
§9 complete while any required compiled-private row is open.
