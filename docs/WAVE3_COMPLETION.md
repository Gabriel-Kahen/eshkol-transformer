# CPU v1 completion — current acceptance index

Updated 2026-10-25. **Not admitted; PR #214 remains draft and unmerged.**
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
| CPU-v1 resource census | CLI3 + integration | Reviewed producer stack through `b4b950af` and finite-cache verifier `4dfdec3e` are integrated; 48 resource host tests pass. Selected P1/candidate local transfers pass, but item 1 is `FAIL_TO_PROVE`, six-helper effects fail, and target runtime is not run. | Prove private initialization/dominance, holder identity/lifetime/ABI and transitive effects on authenticated current source; then run numbered resource gates and bounded runtime/census. Frozen c5f object/IR receipts do not transfer to this union. |
| Installed CLI3 chain and resume parity | CLI3 | Exact-c71 hosted job `116183228215` includes `CLI3-PRODUCTION-GENERATE-PASS` and `CLI3-CROSS-COMMAND-CHAIN-PASS`; retained authenticated log | `bash scripts/test-cli3-cross-command-chain.sh DISK_BACKED_EVIDENCE_DIR`: tokenizer/corpus → train → SAVE → resume vs uninterrupted → evaluate/generate, malformed/identity negatives and checkpoint immutability. |
| CPU multi-token/EOS terminal and data-only continuation | G3 + integration public adapter → CLI3 | Native terminal prerequisites and fixed one-byte CLI path only | Implement/compile accepted [same-call terminal contract](g3/G3_C4_P2_G2_ESHKOL_TERMINAL_CLOSE_CONTRACT.md), then public G2 gates and [data-only continuation](g3/G3_R_DATA_ONLY_CONTINUATION_CONTRACT.md); larger repeated-decode/EOS profile remains required by the full G3 contract. CLI adapter follows admitted exports. |
| CPU training/learning/resume acceptance | CLI3 | CPU4 two-update140 checks,250-update learning2010 checks, frozen-b0 normal/repeat FILE resume770 checks/375 real steps, and exact-a262 distinct fresh-process training resume with full A250/B250 C2 equality and three negatives independently accepted; synthetic held-out mean5.51855→0.00003266 | Complete affected current union, SAN/resources and installed workflow; tiny synthetic improvement is not language generalization or persisted generation continuation. |
| Genuine C4 T4 numerical bridge | Integration + CLI3 | Test-only `c629e378` uses canonical C4/P1/D2 21F/25R path and an independent 1,192-gradient/1,024-logit CE oracle; independent source review accepted its recipe and negatives. | Run supported normal/repeat/sanitizer gate under the shared lock and seal actual transcripts; no compiled or installed admission follows from source review. CLI3 C4 producer/consumer then depends on this result. |
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

- Integration exact-head [run 40156102186](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/40156102186)
  at `19096272` finished 20/24: CLI3 commands and canonical/model/parameter
  suites passed; shard-loader stopped at a stale E3-D2 source pin and native
  numerics at the unchanged N3K `cases==31u` assertion. The two aggregate
  failures are derivative. The E3-D2 pin was subsequently corrected at
  `a608141f`; the resource inventory and 20 exact package closure manifests
  have independently reviewed source corrections. No corrected-head canonical
  package or hosted acceptance exists yet.
- The N3K assertion failure in that run is a separate stale test-roster count:
  the provider advertises 55 operation/shape pairs while two downstream tests
  still expected 31. The independently reviewed two-value correction passes
  direct current-source Clang 21.1.8 normal and sanitizer N3K binaries (15,067
  negative and 47,546 carrier checks). Canonical package and hosted gates on
  the corrected union remain unrun.
- A distinct fixed-profile fresh-process training-resume result is independently
  accepted on immutable `a2626f12`, tree `fd9ab06b`: separate 250-step producer
  and A125→B250 consumer, complete 25,504-byte A250/B250 checkpoint equality,
  all 42 value/moment planes, nine metric words and three malformed/identity
  negatives. The first attempt's custody failure remains preserved. Receipt:
  `finish-20261022/team/cpu4-fresh-external-resume-attempt2-independent-review-20261023/REVIEW.md`,
  sealed `a9812a31b4a25a25ea6bc2b20cb72c29139afc85d96cf64bb2624a06e98430fa`.
  This test-only leaf is integrated here; the current native roster has 215
  members versus 211 on the accepted a262 source, so current-union execution
  and installed behavior remain open.
- Intermediate exact-head CI run `40140825549` on integration commit
  `9b281a96` failed `canonical-build` and `parameter-state` at the stale
  generated trusted P1 root check. The corrected source template and strict
  structural checkers pass locally; this does not replace a successful hosted
  exact-head rerun after the repair is published.

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

Latest CPU4 unblock: reviewed ordinary/TR3 O2 restore-provenance repair54df is
integrated through `b0af97d1` with exactly two strict common211 pin updates.
Its focused native attempt2 independently ACCEPTS all128 literal steps,16 probes,
13 stdout comparisons, strict negatives, native ASan/UBSan and OFF controls.
The original changed-option cases fail retirement with raw7/7/2; repaired
changed/same cases pass. Legacy retirement passes1841 checks and original
restore-OFF passes19696. The private TR3 plan measures1464→1472 bytes and two
retained plans2928→2944; combined I2/O2 long-cycle retention is unmeasured.
Receipt `finish-20261022/o2-restore-retirement-54df1d2a-attempt2` has outer seal
`32475690bcc91eccd1eeeb77a0d272158e25ac5cfd18755ed2ccc20f397a17cc`;
independent review `team/integration-review-o2-native-attempt2/REVIEW.md` beneath
the finish root has seal`bd3015abb6240d14d0c75b3704477b0f9c7ca5afeb7e564481d55d1f6764d658`.
Program/custody and wrapper exits are0 with complete dependency and inner seals.
Attempt1 remains failed because its Python-incompatible collector could not seal
the inventory. The repaired v3 collector uses portable hashing; original evidence
is preserved. Native acceptance does not admit FILE, resume or generation.

The FILEv5 gate on frozen `b0af97d1` independently ACCEPTS normal/repeat:
each reports region21/data96 and identical25506-byte C2 files, SHA
`d514e1440bbf5703f9544478435dcb52b8d66db5bb52b1cb0b4ebb3e0f028e56`.
Raw seal`8f9f0439f9315b4ff00a98f47737f8a5e62239a6e2ceddab83e79f8666ad2971`
and review`team/integration-review-file-b0af97d1-attempt1/REVIEW.md`, seal
`7c36a86be2a09c469e5fba83e1c15c1233587d6baed82b69f13bfdd13ea81218`,
authenticate source/dependencies/IR/BC/object/ELF, all exits0 and cleanup. Its
198-second duration is not throughput. This is untrained artificial coherent
FILE movement, not learning or resume evidence.

Plain trained resumev2 on frozen b0 independently ACCEPTS two exact770-check,
375-step executions. Trained A125 SAVE→A250 versus actual FILE-B125→B250 matches
all1192 values/2384 moment words, controls/RNG/cursors/config/tokenizer and fresh
same-B E3 reports. Mean CE5.518550395965576→0.0000326615190715529 satisfies the
fixed synthetic held-out criterion; perplexity249.2734→1.0000327, accuracy0→1.
Both25504-byte checkpoints have SHA
`499008516bce12c9ca07de19e97cee99ec373b0070cf13b34d035296889692cc`.
Raw`9240d74c5bbc88fca1fcae94cbf1a0d344ead51fa3f4844b52365e5858256932`
and `team/integration-review-resume-b0af97d1-attempt1/REVIEW.md`, seal
`89a8e435af91e940d0e08102df707122e649cbb0859605990c99f2f39aa1d0b9`,
authenticate emitted real-step backedges/full-plane comparisons, source/native
ownership, artifacts, all exits0 and cleanup.265s is gate duration. This is
normal/repeat within-process FILE resume; SAN, independent fresh-process
adoption, language generalization, installed and generation claims remain open.

Trained
same-B public-G2 source4c6 plus support-context55a is independently accepted;
its four-process functional gate independently ACCEPTS exact770 resume/375 step
plus19 G2 checks per process. EOS255 returns IDs/raw(3,41), length2/cache4;
EOS3 returns(3), length1/cache3; all actual result RNG words are(1,41,0,0).
Same B fourteen handles/4768 value bytes remain unchanged, all four checkpoint
files match SHA49900851 above. Raw seal
`ee00a454f7cbe1fa22503e6978cead10b2edc196b2b4d4d4e5b31a838fb11f36`
and `team/integration-review-trained-g2-55a56f2f-attempt1/REVIEW.md`, seal
`5d9356db7caa1875957284510f5cffb9bb57c5ae31cb7b2aed30e228f2c3dcc9`,
authenticate all37 native owners/38 MMD files/208 observed paths, real emitted
handoff/public-generation/release calls, all exits0 and cleanup.340s is gate
duration. Functional acceptance excludes SAN, history, persisted continuation,
independent fresh-process resume, installed and complete-retention claims.
Complete owner/byte census
is still blocked on missing observability:13424 is a partial source subtotal,
and the six existing native counts do not establish total retention. OS process
limits cannot substitute for that required census. No full CPU-v1 admission.

One-pass resource measurement policy is accepted from proposal-v2 seal
`386b4838c6137636e15fae2c648385bd5e93561588851f1ec6b4eeb409528711`
and independent review`team/cpuv1-resource-contract-review/REVIEW.md`, seal
`dbef04498658d71556d4c80dd1f551ca3cf55ab5fcddfea49b0acefd864bd795`.
CLI3 owns Stage A private observer producers; exact packets/source/tests require
review before a dependent sidecar is composed. Frozen history-OFF one-pass
limits: native requested growth128MiB (nested F32-retired64MiB/O2-step256KiB),
Scheme reservation plus metadata growth512MiB and tracked peak minus P0 current
512MiB, runtime peak RSS2GiB, normal-glibc backing2GiB and nonnegative deduplicated
residual growth32MiB. Absolute totals and thirteen family domains, all successful
allocation/free histories, retained controls, both arena chains, metadata and
semantic roots must reconcile at final P9 after D2 close. Native usable bytes,
allocator residual, backing and RSS are distinct columns. One original P1
runtime vector has75 slots0..74 (its AST list has76 elements including `vector`);
the proposed owning-scope census closure must preserve it exactly. No observer,
sizeof, measured fit, native transient peak, history-ON or retention admission
is established by this policy.

The reviewed resource producer stack through source `c5f50fd8` is integrated as
`b4b950af`; its finite-cache verifier `4e0173c1` is integrated as `4dfdec3e`.
Current-union source pins and trained-G2 preservation guards were refreshed
after independent review; 48 resource host tests pass. The verifier reports
`PASS_LOCAL_TRANSFER` for two selected holder transfers and `FAIL_ITEM1` for
the wider proof. Nineteen helper-global initializer selectors, full dominance,
indirect holder identity, lifetime and ABI remain open; separate six-helper
effect closure fails. Existing c5f object/IR/static receipts retain their
frozen source identity. Numbered runtime, owner-byte census and installed
admission remain **NOT RUN** on this union. Reviewed integration repair
`eefe25d1` repins the L3S current-source manifest and checks historical F32
preprocessing in normal/testing modes; supported Clang 21 source checks pass.
Current-head hosted L3S and installed behavior remain **NOT RUN**.

Binding-OOM witness attempt1 remains a preserved collector failure: `nm -g`
omitted genuine LOCAL lease symbols after successful native/AOT/link builds.
Raw seal `7e716949ee35e8babba7f934409e2c3f76ecd18b67eecfa8defd8e0d2fb7a6ea`
and failure audit `bcac9fd9bae24893a3a1df3fedfe9c1c2f2610d60d949883d412004d9c8866d4`
remain unchanged. Collector-only v2 retains strict global-wrapper checks and
proves exact LOCAL definitions through typed ELF, map and actual call addresses.
Clean attempt2 independently ACCEPTS two identical four-record runs in194s:
old publication leaves pending1/retry blocked; fixed early and post-publication
failure cases leave pending0 and each retry reaches real lease prepare/commit.
Raw seal `cccfaeff8b700036da2b1ba5f33318cdc609f7ac6e521f4160f74c1131878340`;
independent actual review
`team/review-cpu4-binding-witness-ce46e8d9-attempt2-actual-independent/REVIEW.md`,
seal `8bf8dbc36a293e74cf09a305c8c7517807f50ff258eaebe79a8b1954007a7556`.
All81 statuses/custody are0. Relocation remains UNPROVEN and retained bytes
UNMEASURED; this is normal native handler/lease evidence, not SAN or retention.

Private history/capture repair352 and exact composition78 remain accepted.
Focused fixture-only repair93db869d is integrated through `8e609c2a`; common
closure215 and all35 production TU flags remain unchanged, history OFF.
Original native78 attempt1 remains FAIL at undeclared fixture seed_generator,
raw `d48f81304240a85c1268efbc7282dbbd61ea0e1ed3b887df222e1001d347a397`;
its independent failure audit is sealed `5e050cf78b5acc95054ae0f50e23e97d6914ea70bfa52b7281694905f7c5b113`.
Fresh exact93 native attempt1 independently ACCEPTS in91s: normal, repeat and
native ASan+UBSan each pass history173/capture144/manual668 checks. Seven OFF
objects equal the accepted baseline; four ON layouts, expected compile failures,
all136 command outcomes,94 depfiles and654 sealed members authenticate.
Raw `a3dd8f232d7ec926314e3a85697aa3232d95a21cfb9e327bc15cb6df82b529e4`;
independent actual review
`history-capture-native-93db869d-attempt1-independent-resolution/RESOLUTION.md`,
seal `68e0d1d49c0a61e1e2796ce68198d04ae179b64ca213593acc4b4d93020bcc85`.
Genuine greedy and categorical replay of the three-token committed prefix
reproduces all16 K/all16 V values, keep/length,256 logits and RNG bit-for-bit
without another sampler call. Detached copies survive generator close while
the model stays test-owned; model destruction is not tested. SAME40 committed
frame lifetime, failure/alias/capture cuts pass. Static live/dead arithmetic is
not a measured retention slope. Typed history/RNG import, cache4 result,
closed public partial producer, wire and fresh-process continuation remain open.
See [CPU4_C4_HISTORY_CAPTURE.md](CPU4_C4_HISTORY_CAPTURE.md).

Trained-G2 SAN attempt1 is an independently authenticated FAILURE, not acceptance.
Recipe `e217824acd1472f4d6a69b0a760146cdd72e57d7685619a46aaa5b277356f45b`
and independent recipe review `725dc008a668544869ecc02c5e1bc6b277ab46aead4abf3d0c6ace6bf36399c2`
remain valid preparation. Raw receipt
`df95145c3798e5b743b8d2ca416f312f3734bf5d243715ab0bfc98ea841c5bf2`
and `team/integration-review-trained-g2-san-55a56f2f-failure/REVIEW.md`, seal
`035863de36e7da70630791ef7aeae872d86c2bae19d78e8989393cdfcd585fce`,
authenticate exact55a, all37 native ASan+UBSan TUs, reversible3,010-definition
generated-IR ASan rewrite, actual derived object/link/ELF and custody. Native,
IR compile/proof/link stages return0. First two-token process exits1: ASan detects
READ1 seven bytes before a48-byte terminal-IDs clone, in generated
runtime_lambda_60 through terminal-IDs release/typed publication/output-IDs.
Static machine code maps this to a pointer-minus-seven header probe; allocator,
source-wrapper and compiler attribution remain unresolved. Its seven stdout
records match accepted normal through B250 E3 and same-B handoff; A125 C2 bytes
match49900851. G2 RESULT/PASS, first checker, remaining three processes and all
output comparisons are NOT RUN. Program1/custody0 and owned CID cleanup pass;
2,315s is gate duration. Four LLVM vectorization and separate readelf DWARF
warnings occur in successful static stages. C7 runtime is uninstrumented;
generated-IR UBSan/full-runtime/leak-free claims remain excluded. Preserve the
failure; bounded causal reproduction, independent repair review and focused
sanitizer regression precede another full trained path. This memory defect is
the immediate correctness blocker; other new heavy gates are deferred.

The additive ON80 E3 source variation28e51cf2/README correction7fffd7a5 is
integrated through `cd966d44`, source-identical to reviewed7fff except roadmap
and this index. Root22 source tests pass. Three exact fixture projections keep
the original body/helper/assertions: current root once, genuine candidate create,
and candidate retirement after all original drainage. Original217/219 checks
and six metric words remain frozen;35 production TUs plus original helper once,
common215/caller13/resource4, historyOFF. Independent combined source/recipe
review `team/review-e3-cpu4-on80-source-and-normal-recipe-7fffd7a5-independent/REVIEW.md`
is sealed `af47d4e32e3c1b0233482d1610a9b08e664d89b9ac56bb649f0a083e2b5a8fe9`.
Corrected recipev3 `01f483cb7382558ae950c9b1f524d8112bd429b18a74e0152f2bbf37ed82d9b0`
authenticates actual checkout and normal producer/collector paths with fail-closed
controls. Failed preparationsv1/v2 stay NOT RUN/HOLD. The two-process normal
runtime matrix is NOT RUN and deferred to the memory-blocker priority; no OFF64
SAN/result promotion or retention admission follows.

The production CLI scalar/report contract v3 is accepted as a proposed API:
proposal `86c45f138b2bee3dea3bc6cdc5b1442fe371bc9d2e2b06bea9f757049e58ccb5`,
independent review `e72642b303b3e9cfbb9a8b3c6d9ffda195e53118397064cb9e9d5515335872fd`.
Its five native scalar helpers and closed Scheme report leaf are implemented
at68899222 and integrated as `ab654852`: three additive production files and
seven test/docs files, with all15 upstream producers/root/base35 unchanged.
Source review `e3/cpu4-cli-report-producer-source-review/REVIEW.md`, seal
`d3c11ceacbba38fac86393462b02159ecbc78b7b9023746f50a508a63a0729f3`,
and final-pin review `ee36ab0c53b6c1ccde87a8fd3b13d84f2606840d29361528b96c73a8e73c17b0`
authenticate source-only acceptance. Root Guile53 orchestration-double and
Python7 source checks pass. The prepared native/AOT unit recipe is sealed
`8f10d290f3bedd233edf5e106cd7835b36adb57ff1c852aff54736f0423c8e9a`:
base35+shim36, separate C unit/double-fault caller, genuine two-step/E3 full
state-preservation runtime normal/repeat. Compiler/native/AOT/SAN remain NOT RUN;
heavy execution is deferred to the immediate generation memory defect. A failed report after a committed
step must retain advanced U and permit report retry without repeating the step.
Per-step closed WRITE snapshots and32-byte scalar custody sidecars require a
separate resource-profile extension; the frozen55a resource counts do not cover
installed CLI commands. The current candidate-ON E3 consumer fault matrix is
also authorized as a bounded source variation; the older OFF64 matrix alone
does not establish ON80 failure/rollback behavior.

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
| E3 no-grad frame | [Accepted frame contract](CPU4_E3_FRAME_CONTRACT.md); six reviewed blobs at `393b5358` integrated as `9a7bfc67`. Fixture repairs through `c9613ef6` integrated as `98875504`; supported native result independently PASS: 22,890 assertions each in two normal/two ASan+UBSan+LSan runs, 21 roles/full 1,024 logits, gradient/owner preservation, atomicity/first-error/resource cases. | Real consumer `4c745a0f` independently PASS in1,280.106s: eight genuine normal/generated-IR-ASan absent/present/repeat runs,217/219checks. Exact1,774-definition ASan rewrite/object proof,17nativeTUs+helper and single F32/I1/D2 owners verified. Candidate-OFF64-byte direct uninstalled scope only; candidate-ON/trainer handoff/package and retention gates remain. Failed mount and precondition fixtures remain preserved. No whole-frame bitwise or unbounded memory claim. |
| Candidate constructor/retirement | [Accepted contract](CPU4_CANDIDATE_CONTRACT.md); producer/FFI/bounded-helper repairs integrated through `f0df7708`, O2 diagnostic test-only `f94eeb43` as `22707205`. Independent final gate PASS: genuine producer normal/repeat84 via authenticated unchanged-source reuse; fresh native-provider SAN84; native805 normal/repeat/SAN, 34 exact fault exits, F32/P1/O2 repeats and eight feature-off pairs equal. | Typed child/factory compound runtime and checkpoint cleanup. Reviewed child source is tracked below; earlier null-child-only admission does not admit it. Generated Eshkol IR/prebuilt runtime are uninstrumented; provider SAN is not whole-runtime coverage. |
| Public G2 ownership and aliasing | Reviewed `0a292f416` integrated as `10a1764f9`; actual-union C11/feature-off/host controls PASS. Control-alias `11d3d5b3` passed690 checks; candidate-record successor `0064e886` independently PASS732 normal/repeat/SAN, genuine live/DEAD capability ALIAS plus per-call byte invariance. Capability records are288 bytes; candidate-on I2 parameter controls are separately80 bytes. | Complete installed AOT/runtime/SAN and same-trained-model generation/continuation. Control observer protects known record extents; arbitrary child-payload extent is not inferred. |
| Canonical binding/P1/D2 | Nine reviewed blobs at `78f39c93` integrated as `02336953`; trusted75-slot checker retained. Independent supported normal2/native-C-SAN2 PASS68 covers14 handle/carrier identities, position[4,4],17 modes, real T4 cursor and release. D2 normalization `f0646f75` is separately validated by full-step. | Lower binding publication guard repair `ac9afc63` integrated as `c960dddc`: install handler before pending publication; cleanup rereads its canonical owner/pending row and preserves the exact error. Independent source3/host23 and existing40/66 checks PASS, including both intended baseline failures. Common211-member pin resolution `fd0db727` independently ACCEPTED. Genuine fe9 normal/repeat handler-OOM/native-lease retry witness independently ACCEPTS: rawcccfaeff/review8bf8dbc3; exact failed attempt retained. Relocation and retained bytes remain unmeasured. |
| Trainer snapshot/restore controls | Authenticated22-slot record, real WRITE lease and prepared controls-last tail at `fd1691ff`/`1a97ecc1`. Runtime recipe `eecf9396` and canonical bytewise comparison repair `502f0671` integrated through `948b8c31`; six source guards, four recipe checks and thirteen actual byte-helper host checks PASS. [Closed detach](CPU4_TRAINER_DETACH.md) `3580136f` integrated as `13af5c4b`. | Controls supported gate independently PASS104 checks in normal2/native-provider SAN2,259.021s. Detach fixture `be65fb8a` initial67-check result remains partial: reserved named-let `values` removed its preservation-loop backedge. Reviewed test-only `00ae5c50` integrated as `d94e056f`; corrected supported normal2/native-provider SAN2 independently PASS exactly93 checks each in247.599s, including actual loop backedge, all fourteen native value checksums, lifecycle/contention and same-trainer retry. This is not a full value-word/moment oracle or generation adoption. Full C2/I2/O2/D2 transaction remains separate. Original bytevector-pointer constructor failure stays preserved. |
| Masked training objective and Eshkol schedule | [Objective contract](CPU4_TRAINING_OBJECTIVE.md), twelve reviewed blobs at `bb6ca081` integrated as `4df3500f`. Independent supported result PASS: native2,501 objective checks/full1,192 CE central derivatives and genuine Eshkol114 checks of21F→L2/L3S→25R→atomic fourteen-gradient contribution in normal2/native-provider SAN2,255.108s. | Full-step component is accepted below; learning and same-model resume remain separate. Recorded toolchain is fe9/runner7dd254ba; generated IR is uninstrumented. |
| Full training step | [Step contract](CPU4_TRAINING_STEP.md); six reviewed new blobs at `a8e24ea7` integrated as `fdbe3a69`, seven host source guards PASS. Own exact frame drains READ microbatches before WRITE/O2 prepare; guard installed before first write; controls-last tail and retryable prewrite abort. Accumulation bounded at2^22. | Initial actual IR/ELF run stopped before O2 update: test helper used the absent-gradient state-snapshot API after expected gradient contribution. Reviewed moment-bit-observer fixture `2574d0fb` integrated as `7b49d17e`; seven host guards PASS. Its genuine retry exposed raw D2 cursor-decode error escape. Reviewed existing-error-adapter repair `f0646f75` integrated as `d9e59b38`; supported gate independently PASS four actual140-check runs in255.088s. Each reaches two real O2 updates on the same canonical C4 and compares all1,192 values/4,768 bytes plus2,384 moment words against ordinary O2, with rollback/EOS and exact cursor errors. Native C sanitizer/arena-poison scope; generated IR uninstrumented. Learning/checkpoint/resume remain separate. |
| Typed factory child custody | [Children contract](CPU4_CANDIDATE_CHILDREN.md), reviewed `de2e6d83` integrated as `2dd97345`: private O2 pre-roots its owner; cleanup preflights O2 before TRAIN detach/native retirement; strict75-slot checker unchanged. | Corrected exact-a262 component gate independently PASS: normal2/native-provider ASan+UBSan2, 537 checks in each of four runs, sealed `b11aaef9` review. Generated IR and C7 were uninstrumented; full trained sanitizer, resource and installed behavior remain open. Earlier opaque and EVAL-state failures stay preserved. |
| Existing-trainer checkpoint consumer | [Consumer contract](CPU4_CHECKPOINT_CONSUMER.md), five reviewed v2 files at `42438ce3` integrated as `c448e44b`; exact control dependency `502f0671`. Seven structural and34 actual-host/double checks PASS. Full14 values/28 moments/all metadata detached before C2 borrow-end; prepared controls precede busy participants; I2→O2→D2→controls-last. | Native/SAN compound snapshot/restore and nonzero resume. Shared I2/O2 acquisition `64fd89bc`+`fefbfa6d` integrated through `3c45186c`, independently source-accepted with nine structural/33 instrumented-host checks; actual regional ABI-shim AOT independently PASS19 checks in normal/repeat with empty diagnostics and real region enter/unwind/leave. It proves Scheme ledger promotion/drain against static shim identities, not real native payload transactions, faults or SAN. Real native compound gates remain required; factory/private O2 remains separate. |
| Fixed-profile checkpoint factory | Eight source-reviewed files at `23bc7a7c` integrated as `bac684a9`, exact existing C2 split-stage/K2 admission/reconstruct followed by canonical child creation and prepared restore. First-operation genuine `region_current` guard enforces nonregional serial entry. Recovery cell roots its ticket before acquisition; own images close before child/O2 cleanup; caller T2/D2 survive. Fourteen source,134 factory/34 consumer host and fifteen extra failure/retry checks PASS. | Positive FILE roundtrip four-file fixture `64ea7667` integrated as `596c46ed`. Frozen-b0 normal/repeat FILE and trained resume/held-out are independently accepted in their exact scopes; a262 fresh-process training resume is also accepted. Current 215-member union, SAN, general regional C2 and installed gates remain open. |
| Learning acceptance fixtures | [One-batch contract](CPU4_ONE_BATCH_OVERFIT.md), seven source-reviewed fixture files `722726b3` integrated as `76773cae`; five source/six checker tests PASS. Frozen250-update criterion: >=75% observed mean-CE reduction and last observed mean<=1.0. Held-out tokens `(197,197,3,41,7)` perturb training `(3,197,3,41,7)`; frozen actual E3 mean must improve>=10%, with full state/cursor/mode preservation. | Seven held-out fixture files `4bec1be5` integrated as `67382b41`; six source/four checker/eighteen helper checks PASS. Full-step is independently accepted. First overfit8cb attempt stopped before runtime at LLVM verification of a guarded named loop; exact fixture-only helper repair is source-reviewed, with all2010 assertions and frozen data/criteria unchanged. Reviewed two-file guard-placement repair `da0f0e55` integrated as `361aa91b`. Its supported gate is independently ACCEPTED: normal2/native-C-SAN2 PASS, exactly2,010 checks and250 updates each in345.239s; authenticated frozen train bytes, pre0 meanCE5.533246994→pre249 meanCE0.000003367656518 and both frozen thresholds. All fourteen parameter planes differ; live payload returns to56 tensors/14 parameters, finite F32 control growth5,449,344B and O2 controls24,000B remain within declared bounds; no held-out, generalization, whole-process flat-memory or final-union admission follows. |
| Candidate-ON common aggregate | [Private composition contract](../native/cpu4_v1_private_contract.md); original public generation root once, genuine T1/T2/C4 authorities and single native owners, candidate-ON80-byte I2 controls. Actual C2/K2 composition `e687388c` integrated as `4d2c05cc`. | The optimizer retirement provenance repair is independently accepted; frozen-b0 FILE normal/repeat and trained resume/held-out gates pass with genuine checkpoint bytes. Exact-a262 fresh-process training resume also passes. Their receipts retain fixed source identities; current 215-member union, full SAN, resource and installed package admission remain open. The earlier compiler and retirement failures remain preserved. |
| Same-model generation handoff | Six source-reviewed files `bade6126` integrated as `695e5238`, loaded/pinned by accepted`9b066e64`: closed real-public-G2 authentication, O2 preflight before TRAIN detach, same-model EVAL, fresh generator and fixed ownership-transfer tail. Source7/host105/legacy1,433 plus15 identity/retry probes PASS. | Actual trained/factory→public-G2 normal/SAN gate NOT RUN. Distinct genuine original T1 preserves training T2/D2. E3 must finish/unbind before detach; active READ rejection does not discover every idle E3 frame. |
| Trained FILE resume and held-out caller | Seven definitions-only files `5af5816e` integrated as `98e2c0bf`: A125 SAVE→A250 versus B FILE restore125→B250, complete value/moment/metadata/cursor equality and fresh B E3 on its same binding;770 checks/375 real step calls. Five-file actual frontend/normal-repeat driver `39ca9b9f` independently source-accepted/integrated as `21128e63`. | Frozen-b0 normal/repeat 770-check, 375-step FILE resume and synthetic held-out improvement are independently accepted. Exact-a262 distinct-process producer/consumer C2 equality and three negatives are also accepted. Current 215-member union, full SAN/resource and installed workflow remain open. No artificial seeding or generation API. |
| Persisted C4 generation continuation | Source-only maps authenticate existing APIs; proposed history/capture ABI v2 evidence seal`214ec8ab` has explicit early preflights, bounded pending proof, trusted ownership scope and detached lifetime. Root accepts this exact proposed contract after independent E3 delta review`09978127`; source implementation is authorized, runtime is not admitted. | Bounded native history/capture and bitwise no-draw prefix replay independently ACCEPT at93db869d (rawa3dd8f23/review68e0d1d). Authenticated typed RNG/history import, explicit C4 wire profile, cache4 result and fresh-process remaining decode remain missing. Existing profile1/history≤2 and result-RNG chaining do not discharge this row. |
| Installed G2 proving recipe | Independently reviewed `81704701` integrated as `b8e0d4e0`, preserving prior E3/C2-header changes. Combined source gate: eight unit/Q0 plus four public-source tests PASS. | Actual installed normal/SAN/coexistence NOT RUN. Supported native98a5 held-kind9 diagnostic independently PASS151; it supplies no Eshkol attribution. Old stage45 status134 remains unattributed; reviewed historical E attribution is staged but unqueued behind the CPU4 smoke priority. |

Actual E3 consumer result is `e3/cpu4-consumer-native-b-result-review/REVIEW.md`
beneath `finish-20261022/` (seal `3b954a2d…23bdf20`). All runtime diagnostics are
empty; generated-IR compilation has four retained nonfatal vectorization warnings.
This does not establish vectorization, performance, whole-runtime sanitizer
coverage, installed symbols or the final candidate-ON union.

Latest binding and candidate result reviews are `team/CPU4-BINDING-RESULT-REVIEW.md`
and `cpu4-candidate-final-independent-evidence-audit-f94eeb43/AUDIT.md` beneath
`finish-20261022/`; objective result review is
`team/CPU4-OBJECTIVE-SCHEDULE-RESULT-REVIEW.md`, controls result review is
`team/CPU4-TRAINER-CONTROL-RESULT-REVIEW.md`. Full-step and detach reviews are
`team/CPU4-TRAINING-STEP-RESULT-REVIEW.md`,
`team/CPU4-ONE-BATCH-OVERFIT-RESULT-REVIEW.md` and
`team/CPU4-DETACH-LOOP-RESULT-REVIEW.md`; the ABI-shim review is
`restore-region-fefbfa6d-independent-result-audit/AUDIT.md`. Their exact fixture deltas and review seals were authenticated
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

The one-batch learning, frozen-b0 FILE normal/repeat and trained resume/held-out
gates, and the exact-a262 distinct-process training resume gate are independently
accepted within their fixed finite scopes. The next integration priority is a
supported same-owner train → SAVE → reload/resume → evaluate → multi-token/EOS generation smoke,
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
cannot close that gap. The reviewed closed CPU4 O2 constructor roots
the actual envelope/registry before native finish and uses a fixed-store
publication tail. Public O2 remains unchanged. Exact typed child cleanup must
preflight the real native O2 before trainer detach, retire the unpublished native
candidate with that O2, then invalidate its Scheme shell/registry without a
second destructor. The existing shared I2/O2 restore-bindings constructors have
the same native-before-global-publication gap; G3 owns a bounded same-signature
repair that roots both ledgers and installs cleanup before native acquisition. These sources have immutable review; actual compound runtime
failure gates remain required. Native adoption still requires trainer/binding release first.
Evaluation must finish/destroy its frame and restore TRAIN; detach/release
binding while still TRAIN, then `module-eval!` the same model before creating
and adopting the fresh generator.
The shared acquisition repair uses canonical global-head readback after rooting,
not pre-promotion local identity. Its prepared actual Eshkol regional fixture
uses an explicit native ABI shim; that result cannot admit real payload plans.

The fresh file factory is assigned to G3: it owns its loaded C2 image/staging
and fresh children, while tokenizer/D2 remain caller-owned. Release its owned
source image before O2 retirement preflight; external live O2 state may still
block cleanup, which must retain exact retry ownership. Never destroy a caller
checkpoint or relax the native global-state guard to force cleanup.
Existing C2 reader/compose publication still has regional allocation windows.
The fixed serial CPU-v1 factory and CPU4 checkpoint entries will enforce no
active region before effects using the genuine pinned `region_current` symbol.
The sealed no-region checked-store proof is
`cpu4-nonregional-reader-source-proof-de2e6d83/PROOF.md`; guard source is accepted; genuine
regional rejection/ownership tests remain pending. General active-region C2 use
is not admitted by this narrowed target.

Accepted final trained-model public-G2 caller source target, G3 owner: extend the
actual A125 SAVE/A250 versus FILE-B125/B250 path at its live-B seam, after complete
B E3 unbind and before old cleanup. Preserve the original plain770-check caller;
new strict composite evidence retains all seven original records and375 steps.
One genuine handoff/public generator per process, distinct original T1, frozen
prompt bytes(3,197). Two fixed modes: EOS255 expects IDs/text(3,41), length2,
cache4; EOS3 expects IDs/text(3), length1, cache3. These are unmeasured criteria,
not results. Normal/repeat per mode must preserve every B value plane and caller
T2/D2, release output/input/detached copies and close the generator. The adopted
B/P1 model and sealed T1 encoding intentionally remain until one-pass process
exit; no new destructor, zero-owner or flat-memory claim. Plan`065592e2` plus
review`e9fb79ab` supports this source scope with root's explicit two-mode delta.
The existing read-only child observer and F32-testing flag may be added to the
sole owners beside O2-testing/held-out observer (37 TUs). Actual sizeof, complete
owner/byte census and bounded process resources remain separate required gates;
13,424 bytes is only the plan's partial source subtotal. No capture/import/wire
or continuation admission follows this caller.

Accepted bounded O2 retirement-provenance repair target, integration expert owner:
source cause map`733aca1e` shows legitimate ordinary LOAD and TR3 restore change
entry option words without updating the full-entry retirement roster. Preserve
all full-entry identity/tamper and global-state retirement guards. Authenticate
the original provenance before restore preparation/commit, then synchronize only
its option words in the existing fixed successful tail. Do not rewrite identity
fields, add fallible postzero work or change public/native ABI. Feature-OFF stays
unchanged; any private ON-plan size/retention delta must be explicit. Independent
source review and a focused genuine native causal test precede the next FILE run.

Accepted private C4 history/capture v2 contract, G3 owner, source-only implementation
authorized after proposal seal`214ec8aba6ceb8f946572cc2261c4c42f95b03ae4d67f55b8886af3960f9f062`
and independent E3 review seal`09978127a6615cebe527ae86262230c0f999252dbecfcad95d8c19d2c52f0b58`.
The exact proposal is pinned to30f2a02a native inputs, rechecked before implementation
against current supportfd0db727. Feature-OFF layouts and behavior remain unchanged;
complete feature-ON tuple adds80 context bytes and464 provenance bytes, including
40 bytes of authenticated pending custody. Four private detached-capture operations
create/copy-record/copy-parameters/release use a224-byte record plus4768 value bytes;
live5344 bytes, retired352 control bytes per capture. Capture leaves no source lease.
Early authentication/bounds precede providers/A2 mutation/sampling; fixed mirror
tails follow native commit, and abort preserves an already committed first prefix.
Pure queries and legal replacement retain existing admission. Detached operations
never dereference weak source addresses; exact DEAD validation precedes hash/free.
Authoritative native allocation links are trusted, SHA is not authority, and the
specified alias/TLS exception remains explicit. This changes no public interface
or continuation/import/wire/decode admission. Separate immutable implementation
review and genuine ON/OFF ordering/lifecycle/failure/alias/census gates are required.

Any further contract change must be recorded here with owner, exact source and
review decision before integration.

## Supported-scope boundary

The frozen c71 candidate's bounded CLI acceptance uses context two. The current
CPU v1 target is full sequence/context four with the geometry in
[CPU_V1_PROFILE.md](CPU_V1_PROFILE.md); its compiled acceptance is pending.
A passing bounded chain does not establish arbitrary model sizes, GPU execution,
mixed precision, language quality, flat memory or practical scaling. Preserve
measured retention, resource and unsupported-case limits from each lane.
