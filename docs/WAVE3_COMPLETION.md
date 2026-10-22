# CPU v1 completion — current acceptance index

Updated 2026-10-22. **Not admitted; PR #214 remains draft and unmerged.**
Active milestone: the first complete validated **Eshkol-native CPU-f32 version**.
The user narrowed scope on 2026-10-22; the team contract is
`finish-20261022/CPU_V1_SCOPE.md` beneath the receipt root. Existing in-flight gates
continue unchanged. GPU backends, mixed precision, distributed work, new
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
  transcript gate is running separately; no full-matrix pass is inferred.
- Historical TR3 learning/resume observations authenticate retained bytes, but
  17 relevant source files differ from the newer retained run.
  `finish-20261022/TR3-PRIOR-AUTH.json` records missing generated artifacts and
  zero retained hash mismatches; current-source execution remains required.
- [Fixed-profile command walkthrough](WAVE3_WALKTHROUGH.md) provides the exact
  bounded recipe; its fresh retained local execution remains pending.

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

None in the initial completion index. Existing admission requirements are retained.
Any proposed contract change must be recorded here with owner, exact source and
review decision before integration.

## Supported-scope boundary

The candidate's CLI acceptance uses the fixed CPU-f32 diagnostic profile (context
2, hidden size 4, one layer, two query heads, vocabulary 256, byte tokenizer).
A passing bounded chain does not establish arbitrary model sizes, GPU execution,
mixed precision, language quality, flat memory or practical scaling. Preserve
measured retention, resource and unsupported-case limits from each lane.
