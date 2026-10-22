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

## Current CPU4 foundations and next composition

All source below is published on `codex/finish-integration-20261022`. Source
approval, a bounded native result and complete CPU v1 admission are separate.
Initial failed sources/receipts remain preserved.

| Foundation | Integrated source and current result | Remaining gate |
|---|---|---|
| Typed C4 lease and training transport | [Lease contract](CPU4_TYPED_LEASE_CONTRACT.md), [transport contract](CPU4_THIN_TRANSPORT_CONTRACT.md). Reviewed `063bc4c9`/`2558dfdf1`, fixture repair `f935035b` integrated as `64877a8c`. Supported native normal/repeat/ASan+UBSan+LSan independently PASS: 318 lease and 4,219 transport checks, all 1,024 logits/1,192 full-model gradients and atomic fourteen-parameter contribution. | O2 update and same-model checkpoint composition; real binding and objective results are tracked below. |
| A2/N2/N3K exact T4 primitive rows | Source-approved `d5a5c33b`, test-only successors `615a0078`/`eaf48ccf` integrated through `df5d1a9f`. Independent supported result PASS in 22.44 s: 28 forward/VJP cases and 98,960 checks in normal/SAN, affected A2/N2/N3K normal/repeat/SAN suites. | Exact primitive/provider scope only; initial negative-classification/metadata failures stay preserved. |
| E3 T4 objective and metrics | Source-approved `d5970d77` integrated as `bda0c8605`. Supported native gate independently PASS in 15.931 s: 24 positive runs, T2 regressions, 22-case exact oracle, five intended compiled-mutant kills, empty diagnostics and authenticated cleanup/custody. | Native provider scope only; no full evaluator/P1/D2 admission. |
| E3 no-grad frame | [Accepted frame contract](CPU4_E3_FRAME_CONTRACT.md); six reviewed blobs at `393b5358` integrated as `9a7bfc67`. Fixture repairs through `c9613ef6` integrated as `98875504`; supported native result independently PASS: 22,890 assertions each in two normal/two ASan+UBSan+LSan runs, 21 roles/full 1,024 logits, gradient/owner preservation, atomicity/first-error/resource cases. | Real P1/D2/Eshkol consumer is implementing against actual binding. Failed mount and precondition fixtures remain preserved. No whole-frame bitwise or unbounded memory claim. |
| Candidate constructor/retirement | [Accepted contract](CPU4_CANDIDATE_CONTRACT.md); producer/FFI/bounded-helper repairs integrated through `f0df7708`, O2 diagnostic test-only `f94eeb43` as `22707205`. Independent final gate PASS: genuine producer normal/repeat84 via authenticated unchanged-source reuse; fresh native-provider SAN84; native805 normal/repeat/SAN, 34 exact fault exits, F32/P1/O2 repeats and eight feature-off pairs equal. | Typed Scheme child custody and factory/checkpoint cleanup. Initial Scheme cleanup admits null optimizer/generator only. Generated Eshkol IR/prebuilt runtime are uninstrumented; provider SAN is not whole-runtime coverage. |
| Public G2 ownership and aliasing | Reviewed `0a292f416` integrated as `10a1764f9`; actual-union C11/feature-off/host controls PASS. Control-alias `11d3d5b3` passed690 checks; candidate-record successor `0064e886` independently PASS732 normal/repeat/SAN, genuine live/DEAD capability ALIAS plus per-call byte invariance. Capability records are288 bytes; candidate-on I2 parameter controls are separately80 bytes. | Complete installed AOT/runtime/SAN and same-trained-model generation/continuation. Control observer protects known record extents; arbitrary child-payload extent is not inferred. |
| Canonical binding/P1/D2 | Nine reviewed blobs at `78f39c93` integrated as `02336953`, bounded P1 helper merge at `f0df7708`, trusted75-slot checker at `31530dfd`. Raw-tokenizer fixture `c267f0ef` integrated as `192792a1`; independent supported normal2/native-provider SAN2 PASS68: fourteen handle/carrier identities, position[4,4], seventeen modes, real T4 cursor and binding release. | Training/restore composition. Original V259-fixture rejection remains preserved; corrected gate uses genuine raw V256 identity. |
| Trainer snapshot/restore controls | Authenticated22-slot record, real WRITE lease and prepared controls-last tail at `fd1691ff`/`1a97ecc1`. Runtime recipe `eecf9396` and canonical bytewise comparison repair `502f0671` integrated through `948b8c31`; six source guards, four recipe checks and thirteen actual byte-helper host checks PASS. [Closed detach](CPU4_TRAINER_DETACH.md) `3580136f` integrated as `13af5c4b`. | Fresh controls and detach supported runtime gates; full C2/I2/O2/D2 transaction. Original constructor failure was canonical bytevector pointer equality in pinned fe9 and stays preserved. Detach is source-approved only. |
| Masked training objective and Eshkol schedule | [Objective contract](CPU4_TRAINING_OBJECTIVE.md), twelve reviewed blobs at `bb6ca081` integrated as `4df3500f`. Independent supported result PASS: native2,501 objective checks/full1,192 CE central derivatives and genuine Eshkol114 checks of21F→L2/L3S→25R→atomic fourteen-gradient contribution in normal2/native-provider SAN2,255.108s. | Real O2 update/full step, learning and same-model resume remain separate. Recorded toolchain is fe9/runner7dd254ba; generated IR is uninstrumented. |
| Installed G2 proving recipe | Independently reviewed `81704701` integrated as `b8e0d4e0`, preserving prior E3/C2-header changes. Combined source gate: eight unit/Q0 plus four public-source tests PASS. | Actual installed normal/SAN/coexistence NOT RUN. Supported native98a5 held-kind9 diagnostic independently PASS151; it supplies no Eshkol attribution. Old stage45 status134 remains unattributed; reviewed historical E attribution is staged but unqueued behind the CPU4 smoke priority. |

Latest binding and candidate result reviews are `team/CPU4-BINDING-RESULT-REVIEW.md`
and `cpu4-candidate-final-independent-evidence-audit-f94eeb43/AUDIT.md` beneath
`finish-20261022/`; objective result review is
`team/CPU4-OBJECTIVE-SCHEDULE-RESULT-REVIEW.md`. Their exact fixture deltas and review seals were authenticated
before integration; the complete combined source is not runtime-admitted.

The lease/transport fixture repair binds all fourteen handles before any
initialization; production guards were preserved. Its independently accepted
25.37-second result is `finish-20261022/CPU4-LEASE-TRANSPORT-RESULT-REVIEW.md`.
Maximum absolute logit/gradient errors were `3.0940474e-7` / `3.86504183e-6`.
Payload tensors return to baseline 56; retained controls measured eight transport
records/2,488 bytes plus eight lease-child records/320 bytes. This is a bounded
fixture census, not a long-run memory bound. The E3 provider and alias receipts
are `e3/cpu4-native-b-result-review/REVIEW.md` and
`CPU4-CONTROL-ALIAS-RESULT-REVIEW.md` beneath `finish-20261022/`.

The integrated candidate-off parity runner stages complete immutable Git closures
(provider `063bc4c9`, owner/control-alias `11d3d5b3`), retains depfiles and rejects
snapshot escapes; the source/depchecker repair is accepted at `8537341c`.
Integrated object parity remains unrun. The separately scoped actual native
candidate+typed record-alias result is sealed at
`team/integration-review-candidate-control-alias-receipt-attempt1/REVIEW.md`. Sixteen exact hash replacements across
five predecessor manifests preserve all path lists. Latest combined builder SHA:
`a7df3ba31c96069ce0fabb3dec0cab02a4c6dc0ea08946101a1249c5ce296706`;
nine M3CG tests and its full manifest pass. Previous builder pins are historical.

The next execution priority remains a thin supported same-owner
train → SAVE → reload/resume → evaluate → multi-token/EOS generation smoke,
then the remaining numerical/negative/atomic/resource gates. Joint C4/config/seed/
lease/P1-mode binding must come from the real producer. C2 owner casts, fresh-seed
substitution and shape-only checkpoint conversion are forbidden.

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

Current private checkpoint/factory target, owned by G3 with integration review:
caller retains tokenizer and D2; factory owns candidate/model, binding, O2 and
trainer. Snapshot uses a genuine WRITE parent. Restore detaches all values,
moments and later metadata before ending C2 borrow; stages and copies controls
before participants become busy; preflights every component before the first
write, then commits I2, O2/config/update, D2 and controls last. Prewrite blocked
cleanup retains exact retry ownership; unexpected postwrite failure fails stop.
No legacy TR3 trainer or M3T model authority is substituted.

Factory child tickets must be rooted before construction. Audit found existing
O2 native finish precedes fallible Scheme global registration; a caller ticket
cannot close that gap. G3 is implementing a closed CPU4 O2 constructor that roots
the actual envelope/registry before native finish and uses a fixed-store
publication tail. Public O2 remains unchanged. Exact typed child cleanup must
preflight the real native O2 before trainer detach, retire the unpublished native
candidate with that O2, then invalidate its Scheme shell/registry without a
second destructor. The existing shared I2/O2 restore-bindings constructors have
the same native-before-global-publication gap; G3 owns a bounded same-signature
repair that roots both ledgers and installs cleanup before native acquisition. These source targets still need immutable review and runtime
failure gates. Native adoption still requires trainer/binding release first.

Any further contract change must be recorded here with owner, exact source and
review decision before integration.

## Supported-scope boundary

The frozen c71 candidate's bounded CLI acceptance uses context two. The current
CPU v1 target is full sequence/context four with the geometry in
[CPU_V1_PROFILE.md](CPU_V1_PROFILE.md); its compiled acceptance is pending.
A passing bounded chain does not establish arbitrary model sizes, GPU execution,
mixed precision, language quality, flat memory or practical scaling. Preserve
measured retention, resource and unsupported-case limits from each lane.
