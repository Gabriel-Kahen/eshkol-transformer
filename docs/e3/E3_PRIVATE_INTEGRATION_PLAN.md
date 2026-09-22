# E3 private integration readiness and test inventory

This is an implementation handoff, not a new contract or runtime acceptance.
The [accepted private contract](../E3_PRIVATE_CONTRACT.md) and its companions
remain authoritative. Snapshot: 2026-09-22, merged main
`ba0e37d06076d0a16c473ff742ab95723cb2cb89`, tree
`221e00c9c258841995195a051b9f44df515679b9` (PR #104).
E3 composition is held until prerequisites are accepted and merged and root
explicitly dispatches E3-PRIVATE. No prerequisite draft is consumed by this plan.

The baseline table below is historical. Shared guards are now accepted and
merged as `19f404cf`; [full provenance](../M3_SHARED_CALL.md#integration-provenance)
records supported and focused merged-head results. Independent source/integration
reviews approved the metrics and D2 candidates. Their combined PR #112 passed
supported [CI 35744832879](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35744832879)
at `b052385`, tree `2256194d2daeb233ec1745d0039b383d4358e3d5`, and merged
as `33a54ef7` with that exact tree. Main CI 35798192984 verified and reused this
evidence. Focused merged-head D2 checks passed; metrics closeout is underway.
The P1 owner completed the upstream runtime candidate at `222cad3a`, based on
the current pin `90cbd713`; independent integration review, upstream disposition,
downstream pin adoption and P1/guard acceptance remain required. No new compiler
pin or E3 runtime acceptance follows from this candidate.

## Dependency dispositions

| Prerequisite | Candidate evidence at this snapshot | Remaining gate |
|---|---|---|
| Shared call/pins, [#101 / PR #105](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/105) | Refreshed head `9ae35ecbae04031bb655925545d4176681742c3d`; [run 35736728847](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35736728847) in progress, topology and C2-format passed. Earlier head `e3d8566eaa33eec2c7fb2a4ef278b1c53b0d3acf` passed [run 35693360705](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35693360705), 19 suites / 27 commands. | Open and `MERGEABLE`; `UNSTABLE` while checks remain active. The previous `DIRTY` snapshot is superseded. Refreshed-head evidence, independent root acceptance and merge remain required; earlier candidate success is not a new-tree pass. |
| BOOL metrics, [#107 / PR #109](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/109) | `80d74f8564b39a92e025f8693cdc0cf6d1a224af`; supported [run 35697784630](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35697784630), 19 suites / 28 commands | Open; root disposition, coordinated merge and merged-head validation. |
| D2 identity/idle, [#106 / PR #110](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/110) | `f10acc953b5ab343ed87bdda19a48c3a783b1e40`; supported [run 35698763745](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35698763745), 19 suites / 28 commands | Open; root acceptance and merge. No frame authentication, staging, rollback or restore is supplied by this prerequisite. |
| P1 modes, [#108](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/108#issuecomment-5772515313) | New blocked checkpoint [`b61a0c53e24e548fefc621340a253c3451effb6a`](https://github.com/Gabriel-Kahen/eshkol-transformer/blob/b61a0c53e24e548fefc621340a253c3451effb6a/docs/E3_P1_MODES.md), parent `17648e2`: 749 ordinary checks passed at explicit O0 on CachyOS/LLVM22; no accepted runtime result or PR. | Corrected upstream promotion and constructor-failure prerequisites, allocation-disabled/short-region/optimized mode and reuse proofs, every affected inheritor regression, independent review and supported CI. |

The historical candidates below retain their original evidence scope. Main
`33a54ef7` has the accepted 19-suite / 29-command topology. PR #109 and PR #110
each added one distinct
top-level command, so their strict union is 19 suites / 29 commands; shared CG
adds no top-level command. Preserve both additions without dropping predecessor
coverage. This is the prerequisite union, before any separately reviewed E3-PRIVATE
gate registration. No full CI was launched for this documentation work.

[Upstream issue 713](https://github.com/tsotchke/eshkol/issues/713) blocks the
accepted checked setup/publication lifetime guarantee: under injected allocation
failure, the pinned runtime can return a young pointer or a partially promoted
graph normally. A guard or canonical root readback does not establish safe child
lifetime. The reproduction is deterministic bounded exhaustion, not measured
natural host OOM. Root selected an upstream contract-preserving prerequisite;
there is no authorized root-only relaxation or untested repin. D2 dependency
error promotion also remains relevant to the eventual real transaction proof.

P1's newer ordinary result includes genuine M3 construction plus a separate
parameterless 17-node I2 tree for observable mixed modes; it does not prove E3
frame admission or all 17 modes on the composed M3 evaluation path. Its ledger
also reports static IR evidence of unchecked vector-constructor allocation
results, without executing constructor exhaustion. Promotion repair alone does
not close all setup-allocation failpoints. These historical P1 results are not final-source adoption evidence. The frozen
upstream candidate and its owner are recorded above; no new pin or downstream
runtime acceptance is assumed.

## Integration sequence after root dispatch

1. Record the actual merged prerequisite SHAs/trees, acceptance decisions and
   supported evidence. Reconcile the CI command union and read actual accepted
   headers/source; do not implement against an unmerged candidate interface.
2. Implement the exact frame registry/storage and setup unwind, source authority,
   P1 wrapper integration, D2 restore/owned-batch staging, and error table from
   the accepted contracts. Keep one common guard and one native model authority.
3. Implement the literal 21-role Eshkol forward transcript, then L2 CE, L3S
   reduction, BOOL correct/accumulate and finalization using genuine D2/I2 owners
   and the accepted native i64/BOOL staging storage.
   Add closed private test observations and failpoints within the specified tuple.
4. Add the exact package-builder policy and capture actual raw dependency,
   symbol, compiler and object evidence. Generate candidate manifests separately;
   independent review and root acceptance precede freezing them.
5. Rebuild against frozen manifests in fresh paths; run the real numerical,
   transaction, adversarial and lifetime gates below, including leak-enabled
   sanitizers. Register the eventual focused gate only through root-coordinated CI
   integration; this plan does not invent an executable test command.
6. Hand off the exact commit/tree, measurements, failures and limitations to
   independent E3-PRIVATE-R and root. Private acceptance does not complete the
   public evaluator or authorize deferred weighted/public APIs.

## Numerical integration cases

The metrics prerequisite already exercises L2/L3S/E3 through genuine native
borrows, with independent numerical references and mutation tests. Its synthetic
logits do not exercise M3, D2, a frame or the Eshkol transaction. Extend evidence at
that boundary instead of rerunning a duplicate provider-only campaign.

| Case | Required real composition observation |
|---|---|
| Fixed model | Deterministic nontrivial parameters for N1/T2/V256/D4/Hq=Hkv2/Dh2/L1/F8; compare all 21 role outputs and final logits against an independent development reference. Verify learned positions, causal attention, tied token/head, exact-erf GELU and LayerNorm epsilon bits `0x3727c5ac`. |
| Reduction wiring | Observe L3S outputs in `Nb,Wb,mean` order, followed by `Cb` and exact active count; accumulate numerator/weight/correct totals in traversal order, never batch means. Compare the four final f32 values and exact i64 tokens/batches. |
| Real D2 fixture | Generate a small deterministic byte corpus at test time, including bytes 0 and 255 and repeated IDs. Exercise packed and unpacked shards, partial final batches and entry at a nonzero cursor. Reference each actual emitted row/mask and its shifted target. |
| Ordering and weighting | Preserve the accepted serial-rounding and mean-of-means witnesses in the numerical layer. At full-model level choose data/parameters with nonuniform losses and batch weights, and validate the actual emitted transcript; do not claim an unreachable synthetic-logit witness came from M3. |
| Reuse | Run success A, success B, success A on one frame; repeat suffix evaluation from a restored nonzero cursor. Verify both accumulator banks/counters reset, the first and last A match, and no stale readiness or prior dataset changes the result. |
| Caps and emptiness | Real EOS on entry, zero weight/all-masked input rejection where reachable, exactly 8192 batches / 16384 active tokens, and the first over-bound case. Reject the whole invocation without truncation or output changes. |
| Invalid inputs | Wrong model/tokenizer/profile/device/dtype/shape; malformed stage views, targets and masks; nonfinite logits even on masked rows; destination/control/model aliases. Map recoverable errors and preserve state/output bytes. Cases unreachable from valid D2 use explicit private fault injection, not fabricated loader behavior. |

BOOL-provider tests cover `00` rejection and admit `01`, `10`, `11`; valid unshuffled
N1/T2 D2 data emits active-prefix rows. The full loader test must not fabricate a
`01` case. A concrete corpus `[3,197,0,255,7,7]` can exercise three rows with
masks `11,11,10` and five active tokens; verify emitted rows before using this
as an oracle. Packed repartition comparisons require identical emitted rows;
unpacked shard boundaries require their own reference.

Separate mathematical and bitwise reference checks. Compare actual model logits
to the independent M3 equations under the accepted M3 tolerance and report measured
error. Derive exact lowest-index argmax independently from observed f32 logits;
an f64 reference winner is conclusive only when its top-two margin exceeds the
proved error interval. From observed f32 CE values and independently derived masks,
use rational round-to-nearest-even arithmetic to check physical L3S slots and
ordered accumulation. Independently compute CE from observed logits and expected
D2 targets under the accepted L2 tolerance, and compare end-to-end loss to the
mathematical model reference with propagated logit error; observed CE alone
cannot prove correct target wiring. Check expf against the independent Decimal oracle at the
observed final loss word (at most 2 ULP). Do not claim that an f64 model oracle
proves end-to-end f32 bit identity.

Add integration-specific mutations to the actual compiled Eshkol path: wrong
pin/role or staged plane, mean wired as numerator, stale accumulator bank, skipped
or repeated batch, and treating integer status zero as Scheme false. Require
transcript/value assertions and immediate stop after failure. Reuse predecessor
gradient and provider adversarial evidence; no new differentiable operation or
duplicate finite-difference campaign is introduced by this forward-only plan.

## Transaction and lifetime cases

For each recoverable failure, compare four destination byte spans, two prior
published counters, all 17 mode values, cursor semantics, model parameters,
gradients and RNG before/after. Assert no owned batch/view/pin survives and that
source/native model exclusion and the outer guard become idle in order.

| Case family | Required assertions |
|---|---|
| Setup/authentication | Fail every partial setup allocation; no half-live authority escapes. Reject copied, stale, foreign and cross-frame tokens/records before dereference; preserve unrelated owners. Unbind is exact-token/idempotent for dead tokens, retaining only specified tombstones. |
| Phase/failure matrix | Cover admission, P1 prepare, D2 prepare, acquisition, mode entry, next-batch, each staging copy, every forward role, CE, reduce, correct, accumulate, finalize and publication. Check incomplete native phases 1..7 versus already-drained phase 8 so cleanup is neither omitted nor repeated. |
| Modes/exclusion | Begin with mixed values across all 17 nodes; observe evaluation modes during traversal and exact restoration before publication. Hold model and outer exclusion through restore/publication; exercise reentry and unrelated idle-model behavior. Pins ending must not release the model token prematurely. |
| D2 restore/release | Start at zero/nonzero cursor and cross shard boundaries. Restore saved ordinal/logical state and next-batch behavior, but never rewind monotonic generation. Reject an entry-live batch without releasing it. Fail each sequential borrowed copy and later I/O; close its lease and release only the captured owned batch before region exit. |
| Six-value publication | Fail destination admission at each of four scopes and every staged-result/counter preflight before any write. Test success, then failure, then counter read: prior values remain readable through persistent `published_valid`. Impossible post-write cleanup defects are fail-stop, not recoverable partial publication. |
| Error lifetime | Check all 156 table cells structurally and reachable failures at stages 0..12; do not invent every stage/category combination. Classify with a pair in its catch region, retain only immediate diagnostics/root errors, preserve the first failure and clear temporary references. Return normally through the common guard before raising. Account separately for allocating final error delivery. |
| Poisoning/resources | Use real frame/model/D2 boundaries, poisoned regional memory and allocation observations. Measure live and cumulative controls, caller-region bytes, payload and FDs over 1024/8192 batches and repeated complete calls on one frame. Reconcile retained predecessor controls/tombstones instead of claiming flat process memory. |

No allocation-failure campaign may claim the promotion-dependent guarantee until
the upstream/P1 blocker is resolved. The final error raise still allocates; failure
there can lose the diagnostic after restoration, as specified by the accepted
error contract. No noalloc-raise or arbitrary promotion reservation API is assumed.

## Package evidence and acceptance record

Use exactly the contract's source root, bridge and test driver; 11 ordinary native
compile entries and three fixed provider objects; seven ordered include roots;
one aggregate object. Include each underlying model, D2 and f32 implementation
once. Preserve the predecessor public surfaces and require the exact inherited
M3 global set plus `et_e3_test_run_v1`, with all numerical/frame/idle seams local.

Capture unfiltered raw depfiles, command records, LLVM IR, prelocalization symbols,
objects/members and final symbols/strings. Validate the single generated D2 form
and its pinned hashes; normalize only the exact generated source to `@BUILD@`.
Candidate manifests must remain separate from frozen policy. Reconcile each
new private symbol against source, accept the manifest commit independently,
then compare two fresh builds. Reject extra/duplicate providers, altered tuple,
symlinks, include shadows, unknown symbols and unresolved private seams.

The final handoff must bind each test result to the actual tested SHA/tree,
toolchain/platform and supported or compatibility lane; include optimized and
ASan/UBSan/LSan results with leak detection enabled, full resource measurements,
independent review and root disposition. Preserve failures/retries in provenance.
No public evaluator, GPU, mixed precision, performance, gradient or resume proof
is supplied by this plan or by the private forward-only scope.
