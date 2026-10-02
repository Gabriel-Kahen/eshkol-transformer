# E3 §9 compiled acceptance matrix contract

This is a **pending test contract**, not acceptance evidence. It refines the
required gates in [§9](../E3_EVALUATION_PROPOSAL.md) for the current private
N1/T2/V256/D4 bool evaluator. The source basis is
`native/e3_private_extension.esk`, `src/eshkol_transformer/e3_frame.c`,
`tests/e3_private/driver.esk`, `tests/e3_reference/{corpus,reference,transcript}.py`,
and `native/e3_evaluation_metrics_provider.c` at `94b8fecd`/tree `d28cffc1`.
The existing three-batch numerical gate, mode/graph gate and complete-call
retention gate remain separate prerequisites. Their preparation, execution or failure status
does not mark any row below proved.

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
The numerical AOT currently observes only packed-single. These rows therefore
specify **new** independently derived cases, not assertions about current
observations.

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

The existing private compiled kills are wrong-target CE and mean-of-means
state. Extend the exact noninstalled tuple with one named mutation at a time;
do not alter canonical, diagnostic or installed archives. Every value kill
compares to a prior frozen independent transcript and identifies the first
different row/word. For a corpus-specific mutation, first prove that the
chosen authentic case makes the mutation observable.

| Mutation | Required first meaningful assertion |
|---|---|
| Input instead of next target; target offset | Per-position CE against real D2 target and actual logits, then L3S state. |
| Physical-capacity denominator; count padded position | `W_b`, active count and `C_b` on an authentic `10` row; final W/tokens. |
| Mean of batch means; second normalization | Serial `N,W` and final loss on unequal CE/count batches; no equal-count surrogate. |
| Missing or doubled batch | First divergent batch counter and serial N/W/C; require traversal to have emitted the expected third row. |
| `>=` tie winner; wrong vocabulary axis | Correct count on genuine tie/axis-distinguishing M3 logits. If unreachable with admitted model authority, retain the provider kill as provider-only and hold the private row open. |
| Exp before global aggregation | Final perplexity from global `N/W`, with a genuine unequal-loss witness. |
| Clamp overflow or f64 retry that changes outcome | At an actually reachable finite boundary/overflow row, compare typed failure/atomic output or changed final word. An identical-error retry is an equivalent control, not a kill. |
| Observable sum reorder or retained wider accumulator | A frozen three-term RN-even counterexample through genuine CE numerators. The existing synthetic/provider `64,2^-18,2^-18` witness proves the arithmetic distinction only; it cannot be called a private model witness until real rows produce one. |

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
| N1 three-batch exact roles/CE/states/outputs | Numerical F failed sanitized import admission before AOT/runtime; no pass inferred | Independently reviewed inventory repair, fresh supported rerun, sealed evidence verdict. |
| N2 five authentic layouts, cursor/repartition/boundary | Open beyond reference-only and final-only checks | Expanded frozen-corpus private AOT transcript. |
| N3 correct/wrong/tie/uniform/target offset | Open; reachability/model authority unresolved | Reviewed idle-model control or test-only real-owner control, then independent M3 oracle and private AOT. |
| N4 `01`, masked invalid, fenv, subnormal/signed-zero | Provider-only unless independent model reachability is shown | Retained direct K1 provider numeric/admission gate; leave private requirement explicitly unsupported or open after reachability review. |
| N5 empty/singleton/EOS/one-token/8,192/8,193/resources | Horizon proves only a subset and lacks compiled 8,193 preflight | Authentic D2 private edge/cap and measured resource gate. |
| M1 wrong target/mean of means | Compiled source exists, supported F result pending | Numerical F exact value assertions. |
| M2 denominator/batch/normalization/axis/padding/tie/exp/overflow/reorder | Open in private tuple; direct provider/composition kills are separate | One-at-a-time compiled private mutant matrix with prior frozen witnesses. |
| M3 equivalent controls and native malformed/atomicity | Provider-level evidence only | Preserve compiled provider equivalence/admission plus source/IR review in full matrix evidence. |
| R1 allocation/copy/pin/provider-preflight/read/final cuts and full rollback | Open; late D2 fault is a partial witness | Reviewed one-shot test-only cuts and complete private failure/retry matrix. |
| R2 modes/grad/RNG/owners/graph/event/resource horizons | Mode B and retention G prepared but unrun; no complete failure matrix | Supported B/G verdicts, then stage-by-stage matrix and normal/poison/LSan horizons. |

Completion requires independent source-and-test review, frozen oracle and
mutation inventory, every row's supported compiled proof or explicit accepted
unreachable/equivalent disposition, all pre/post state checks, and fresh
exact-source normal/poison/sanitizer gates after the numerical, mode/graph and
retention prerequisites. Report unsupported boundary rows rather than calling
§9 complete while any required compiled-private row is open.
