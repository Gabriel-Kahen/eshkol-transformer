# CPU v1 completion — current acceptance index

Updated 2026-10-22. **Not admitted; PR #214 remains draft and unmerged.**
Active milestone: the first complete validated **Eshkol-native CPU-f32 version**.
The user narrowed scope on 2026-10-22; the team contract is
`finish-20261022/CPU_V1_SCOPE.md` beneath the receipt root. Existing in-flight gates
continue unchanged; queued legacy gates may be deferred before acquiring the heavy lock. GPU backends, mixed precision, distributed work, new
architectures, broad optimization and optional roadmap expansion remain backlog.

This index supersedes historical pending-status prose for the current finish effort;
the roadmap and linked contracts retain scope and measured limits.

## Frozen inputs and evidence

- Main: `ee24ef461ef7e313c36c6c6f97ba4f3d8155bac7`.
- Candidate: `c71d39a98411d2285fa099f63d1c56170069da3c`, tree
  `34ae8017fcb037548974073926bb26ef5fee7fdc`, [PR #214](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/214).
- Authenticated hosted CI: [run 38661209451](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/38661209451),
  exact candidate, completed/success, 24 successful checks. CLI job logs also
  authenticate the bounded cross-command/resume assertions; raw artifacts have expired.
  This is reusable hosted
  suite evidence, not private cross-workstream runtime admission.
- Local receipt root: `/home/gabe/.codex/evidence/eshkol-transformer/`.
  `wave3-pr214-c71-current-main-independent-review-20261015/REVIEW.md` accepts
  the prior source/docs review and hosted checks while holding runtime admission.
  The PR body's older `/tmp/eshkol-wave3-combined-source-review-9118321-20261014/RESULTS.txt`
  is absent; do not claim its checksum was reauthenticated.
- Current API/ref snapshots and checksum seal:
  `finish-20261022/integration-initial/SHA256SUMS` beneath the receipt root.
  `cli3-hosted-job.log` retains the exact-head CLI command execution.

## Accountable acceptance checklist

| Gate | Owner | Current accepted evidence | Remaining executable gate |
|---|---|---|---|
| Exact-head hosted suites | Integration | c71 run above: 24/24 success | Reuse for unchanged source; changed integration source needs affected gates and hosted checks. |
| G3-R replay / same-registry generation / authentic C2 SAVE | G3 | Reviewed source union; hosted C2 suites cover their declared scope | Authenticate retained receipts, then supported SAVE/reader normal, repeat, sanitizer and genuine replay/same-registry evidence. First entry point: `bash scripts/test-g3r-c2-authentic-save.sh FRESH_OUTPUT_DIRECTORY`. |
| E3 section 9 | E3 | Prior bounded leaves and native/provider checks only where exact source and seals authenticate | Complete [section 9 matrix](e3/E3_SECTION9_COMPILED_ACCEPTANCE_MATRIX.md): numeric/mode/gradient/failure/resource rows, poison and leak-enabled runs. First choose missing supported runner from retained evidence; `scripts/test-e3-numerical-transcript.sh`, `scripts/test-e3-mode-graph.sh`, and `scripts/test-e3-complete-call-retention.sh` remain separate gates. |
| Installed CLI3 chain and resume parity | CLI3 | Exact-c71 hosted job `116183228215` includes `CLI3-PRODUCTION-GENERATE-PASS` and `CLI3-CROSS-COMMAND-CHAIN-PASS`; retained authenticated log | `bash scripts/test-cli3-cross-command-chain.sh DISK_BACKED_EVIDENCE_DIR`: tokenizer/corpus → train → SAVE → resume vs uninterrupted → evaluate/generate, malformed/identity negatives and checkpoint immutability. |
| CPU multi-token/EOS terminal and data-only continuation | G3 + integration public adapter → CLI3 | Native terminal prerequisites and fixed one-byte CLI path only | Implement/compile accepted [same-call terminal contract](g3/G3_C4_P2_G2_ESHKOL_TERMINAL_CLOSE_CONTRACT.md), then public G2 gates and [data-only continuation](g3/G3_R_DATA_ONLY_CONTINUATION_CONTRACT.md); larger repeated-decode/EOS profile remains required by the full G3 contract. CLI adapter follows admitted exports. |
| Current-source CPU training/learning/resume acceptance | CLI3 | Bounded two-update CLI parity is distinct from training acceptance | Authenticate retained one-batch-overfit, held-out-improvement and longer interrupted-resume receipts against current source; execute genuinely missing declared gates and retain unsupported scope. |
| Combined source/tests/evidence and current-main union | Integration reviewer | Prior exact-c71 source review accepted; runtime union open | Review worker immutable commits, provenance, seals, results, scope limits and integration diff once the three lane results exist. |
| Installed CPU binary, artifacts and usable command walkthrough | Integration + CLI3 | Scripted chain exists; execution pending | Run and document an installed supported binary using the validated configuration and artifacts; retain actual JSON, parity and failure evidence. |
| Merge and final validation | Integration only | None; admission held | All rows above discharged, exact-head checks satisfied, reviewed main union, then merge and authenticate resulting main identity/checks. |

## Execution and handoff contract

Use isolated `codex/finish-<lane>-20261022` worktrees from c71; preserve the original
checkout. Lane IDs and current dispatch live in receipt-root
`finish-20261022/threads.json` and `TEAM.md`. Only integration edits the roadmap,
shared contracts and merge state. Publish focused reviewed commits without force-push.

Every local heavy gate acquires
`flock /home/gabe/.codex/evidence/eshkol-transformer/finish-20261022/heavy.lock`;
only one runs globally, with at most three compile jobs. Use the authenticated
pinned supported toolchain and a fresh disk-backed output directory. Record exact
source/tree/toolchain, command, exit status and post-exit checksum seal. No GPU
work or unsupported-host substitution. After two same-cause failures, repair the
cause or hand off exact evidence; do not restart unchanged gates.

Each owner writes `HANDOFF-<lane>.md` to the team receipt directory with immutable
source/tree/branch, command/results, sealed logs, measured limits and one next gate.
Integration consumes completion messages, uses one bounded team wait, and does
independent review/doc work while heavy gates run. Silence is never success.

## Current execution receipts

- Independent initial integration review: `finish-20261022/REVIEW-integration-initial.md`.
  Source composition and recovered bounded hosted CLI execution accepted; full admission held.
- E3 reports a supported fe9/Ubuntu22/LLVM21 native-frame control: 22,223 checks
  ordinary and ASan/UBSan/LSan plus cleanup-mutant rejection. The numerical
  transcript gate completed at exact `44c0418d2`, tree `70706c0a`, exit zero in
  3,750.864 seconds and passed independent result review. Five T2 layouts across
  normal A/B, poison and ASan/UBSan/LSan yielded twenty independently validated
  observations; three compiled numerical mutants failed at their intended
  validators. Raw 510-file/10.4-GB custody and post-source/Q0 checks authenticate.
  Six AOT compilers emitted the same retained vectorization warning; runtime
  stderr was empty. Receipt: `finish-20261022/e3/numerical-a-review/REVIEW.md`.
  This is a bounded frozen T2 prerequisite, not T4/full-matrix admission.
- Historical TR3 learning/resume observations authenticate retained bytes, but
  17 relevant source files differ from the newer retained run.
  `finish-20261022/TR3-PRIOR-AUTH.json` records missing generated artifacts and
  zero retained hash mismatches; current-source execution remains required.
- [Fixed-profile command walkthrough](WAVE3_WALKTHROUGH.md) provides the exact
  bounded recipe; the source-574b local scripted chain now passes with a retained
  binary/artifacts and independently reviewed checkpoint/JSON parity. Separate
  walkthrough execution and the new CPU4 workflow remain pending.

## Source integration (runtime admission still held)

The published integration source union `cad08034e94a383f0e1191949419d9191e89b928`
(tree `2f021cac225edad622edd6fae514a7a08a00789b`) selectively composes reviewed
E3 retention-import/test repairs, CLI corpus-validation assertions and G3 SAVE/replay
runners. No production numerical/generation source is changed by those leaves.
All 47 focused source/contract tests pass on the union; `git diff --check` passes.
Receipt: `finish-20261022/integration-source-union-tests.log`. Compiled producers
retain their original immutable lane identities; these static results do not
substitute for their supported runtime gates or final combined-source CI.

## Shared-profile critical path

The installed training/evaluation path in `native/cli3_b_extension.esk` admits
context two and sequence length two. The multi-token/EOS public G3 target uses
the distinct C4 owner with position parameters `[4,4]`. Its accepted
[model contract](g3/G3_C4_MODEL_CONTEXT_SUCCESSOR_CONTRACT.md) forbids treating it
as a C2 M3T owner and reserves its restore slot without authority. Public G2
ownership alone cannot make a trained C2 checkpoint a C4 model. Integration has selected the [single bounded CPU v1 profile](CPU_V1_PROFILE.md):
full context/sequence four with the existing C4 geometry, one canonical trained
model identity and explicit construction/restore/state-shape extension. CLI3 owns
upstream training support before its CLI consumer; G3 owns same-owner restore and
terminal/public continuation; E3 owns T4 evaluation after exact upstream contracts.
The existing C4 source-authority check passes; next existing executable prerequisite
is `bash scripts/test-g3c4-model-authority.sh`, then the frozen T4 forward/VJP
numerical gate supplied by the upstream owner. Do not substitute a fresh model,
convert legacy geometry or truncate inputs. In-flight gates remain running; future
heavy gates prioritize this shared-profile dependency sequence.

The bounded E3 P1-enter/D2-generation fault source repair is independently accepted
at worker `b34268a88cc5be80e98c5484f49b94c46c6cc590`; its original insertion-order
defect and repaired structural regression remain recorded in
`finish-20261022/e3/r1r2-source/REVIEW.md`. Integration preserves the strict M3CG
builder pin at `7ce3b14cb069de147d12d1c332fdf0279a8fe1200e8ae74a6dca4ecd7cc136a3`;
all nine M3CG contract tests and the full predecessor checksum manifest pass.
The later reviewed external-C2-header canonicalization repair is integrated
without replacing the fault-builder changes; the exact combined builder pin is
`e019798012bd6bc0f4ccfc13605c9f115a3f98edbfd613c5948a7f00a579a281`.
Its canonical/external/old-policy/alias/foreign-header regression and strict
M3CG checks pass. The public G2 union adds twelve package-flag lines while
preserving both repairs; its combined builder pin is
`225f2de63983c964d510e60765bba66d98fbd64385fad0cd023c3ca7697091ac`,
with all nine M3CG tests and full predecessor hash manifest passing. Prior pins
are historical.
Compiled failure/retry admission remains a separate queued gate.

## Adopted upstream CPU4 lease and active implementation

The [typed lease contract](CPU4_TYPED_LEASE_CONTRACT.md) revision two is adopted
for lease-only implementation after one independent assessment. The six-file native lease foundation is source-approved at worker
`063bc4c93ee2c1c62d108d0d4818182613c93a9f` and integrated as `6b62b596a`.
All six reviewed Git blobs were independently authenticated before integration.
The first supported lease gate stopped in fixture construction before any lease
operation: it interleaved binding and initialization, while C4 requires all
fourteen handles bound first. CLI3 is repairing the fixture in a new immutable
leaf; production guards and failed evidence are preserved.
The source implements the exact nine private methods and two native-only child hooks; no parameter owner
is copied or cast. Context accounting excludes even idle generation contexts from
training, native child/ordinal records enforce phase cleanup, and release preserves
the canonical C4 owner. The independently approved [thin transport contract](CPU4_THIN_TRANSPORT_CONTRACT.md)
uses one frame-local workspace with immutable snapshots and the existing Eshkol
21-forward/25-reverse schedule. CLI3 is implementing that producer; the narrow
E3 lease-bound [no-grad frame seam](CPU4_E3_FRAME_CONTRACT.md) is independently
accepted for implementation, with composite workspace and separate actual pin
control. Runtime acceptance remains pending. Joint canonical
C4/config/P1-mode binding remains an upstream dependency.

A2/N2/N3K T4 source and independent primitive-gradient fixtures are integrated
as `c7a3014d1` from source-approved worker `d5a5c33b`; the initial supported gate stopped at a negative error-classification assertion.
CLI3 is repairing the expected classification against the actual K1 stage;
failed evidence is retained. E3 L3S/metrics source `d5970d77` is independently source-approved and integrated
as `bda0c8605` after authentication of all thirteen reviewed blobs; ten combined
source/oracle tests and strict E3 predecessor hashes pass;
the supported provider gate completed at exact `d5970d77` in 15.931 seconds
and passed independent result review. Twenty-four native positive executions,
T2 regressions, a twenty-two-case exact oracle and five intended compiled-mutant
failures passed, with empty compiler/link/runtime diagnostics. Inner/outer seals,
source custody and terminal container cleanup authenticate. Receipt:
`finish-20261022/e3/cpu4-native-b-result-review/REVIEW.md`. This admits only the
native objective/metrics provider scope, with CPU4 frame/model integration open. The final public
G2 repair `0a292f416a3add5ef511c84bb7d89922e4a690ab` passed independent
source/tests review and is integrated at `10a1764f9` after authentication of all
264 sealed Git blobs. Actual-union source/C11/feature-off and host controls pass.
The bounded owner/builder merge assessment accepted the scoped union; supported installed
AOT/native/sanitizer execution remains pending. Simultaneous CPU4-training and
typed-result operation additionally requires the new lease/control span checks
in shared typed output alias validation; the independently source-approved fix
`11d3d5b3` is integrated at `16183055`. It protects known lease/child-record
extents; no arbitrary child payload extent is inferred. Its supported combined gate completed in 29.58 seconds at exact `11d3d5b3`:
690 identical normal/repeat/ASan+UBSan+LSan checks, empty diagnostics and exact
CPU4-off object equality. Independent result review and 37-file seal authenticate;
receipt `finish-20261022/CPU4-CONTROL-ALIAS-RESULT-REVIEW.md`. Earlier `3f53c54b` and successor source receipts remain held/preserved:
causal tests exposed output lifetime coupled to generator liveness and incomplete
final authority checks. The final repair reports 92 actual-core mutation controls;
host controls do not establish supported AOT/native/sanitizer admission. Receipts:
`finish-20261022/REVIEW-integration-public-g2-3f53c54b.md` and
`REVIEW-integration-cpu4-provider-contract.md` / `REVIEW-integration-cpu4-metrics-addendum.md`.

The [candidate producer/retirement contract](CPU4_CANDIDATE_CONTRACT.md) is adopted
with the independently required OPEN_EMPTY versus authentic ABORTED native P1
proof. The independently source-approved producer `3f0358d3` is integrated as
`09b5669f`, after authenticating all twenty-nine reviewed Git blobs and a bounded
composition review. Source/planner 1,239 checks and public C11/feature-off controls
pass on the union. Sixteen exact hash replacements across five predecessor
manifests retain their complete path lists. Supported native/Eshkol proving gates
are queued. It retains P1/I2 authority before sealing rather than wrapping a
sealed ordinary model. Initial Scheme cleanup rejects retained optimizer/generator children;
compound restore/continuation cleanup remains required before that acceptance row.

The frame-local training transport `2558dfdf1` is independently source-approved
and integrated as `e6d43612` after eight reviewed-blob checks. Its full-model
oracle covers all 1,192 parameter derivatives. Its initial native run stopped
at the same bind-before-initialize fixture defect before any transport operation;
the fixture repair and supported numerical/sanitizer rerun remain required. The integrated candidate parity runner now stages complete immutable Git
closures for provider `063bc4c9` and owner/control-alias `11d3d5b3`, retains depfiles
and rejects dependencies escaping each snapshot. Independent source/depchecker
controls accepted the repair at `8537341c`; no integrated object parity is claimed.

The next integrated execution priority is a thin supported same-owner
train → SAVE → reload/resume → evaluate → multi-token/EOS generation smoke after
foundations compile, then the complete numerical/negative/atomic/resource gates.

## CPU v1 completion rule

All checklist rows are required within the declared CPU profile: numerical and
gradient correctness; one-batch overfit and held-out improvement; complete-state
interrupted/resumed equivalence; accepted terminal/multi-token/EOS and persisted
continuation; evaluator mode/gradient preservation, atomic failures and bounded
resources; supported sanitizer/CI evidence; independent combined review; integration
and bounded merged-main validation. Deliver the actual binary, artifacts and
reproducible walkthrough with measured time/memory and unsupported cases. The
existing tiny one-byte proof alone does not satisfy CPU v1.

## Contract changes

Initial index and admitted source-only runner repairs change no API. Existing
admission requirements remain intact.

Accepted bounded private composition source target (runtime unadmitted), integration owner: port the already reviewed
public `diagnostic-c4-p2g2` adapter (budget two, explicit byte EOS) to the closed
terminal coordinator and current typed-owner ABI. Baseline terminal source is
`4506bc8f89bfe8083f5c47d71dcad74d968222c5`; existing public source is `cdf5f5e`
and typed wrapper `666d54eb`. Public provenance must be preallocated and completely
validated before native commit, with only fixed prevalidated state-marker writes
after success. No postcommit registration, arbitrary callback, new public name or
provenance-check relaxation is permitted. Independent source-contract review accepts this target with the following
constraints; exact changed source/tests must still pass independent review.
Preserve the original two-argument wrapper. An internal closed core may receive
explicit enrollment through a private three-argument public route; a factory must
be one-use inside its owning initializer, capture the secret bridge once and seal
installation before exposing a callable. Chosen source surface: one lexical
initializer owns the two-argument private wrapper over a closed
`core(generator,input,enrollment)`; its one-use factory seals installation before
returning the private three-argument public route. Dispatch installs this once
before exposing the public generator callable; a second install fails stop.
No rebind or ambient ticket setter.
Run every callable stage/verify before the core's final exact frame/marker/root
and ownership checks. Authenticate exact original/distinct ticket/row/anchor
carriers and pending roots before commit; set the success barrier first, then
exactly three fixed live-phase
markers. Restore pending roots only after successful abort/drain and exact-root
verification; mismatch or postzero error fails stop. The historical compiler
AD/ND failure used test-only `6803dbe2`, not the supported fe9 toolchain.

Any further contract change must be recorded here with owner, exact source and
review decision before integration.

## Supported-scope boundary

The frozen c71 candidate's bounded CLI acceptance uses context two. The current
CPU v1 target is full sequence/context four with the geometry in
[CPU_V1_PROFILE.md](CPU_V1_PROFILE.md); its compiled acceptance is pending.
A passing bounded chain does not establish arbitrary model sizes, GPU execution,
mixed precision, language quality, flat memory or practical scaling. Preserve
measured retention, resource and unsupported-case limits from each lane.
