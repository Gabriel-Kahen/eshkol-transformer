# CI-E: full-coverage efficiency

Status: **accepted / complete**.
Tracking: [issue #78](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/78).
Integration: PR #79 merged as `cbd0929`; PR #77 merged to main as
`913cdf4097db09d6c33769e9b0968c01ce0e1f55`. Both share accepted tree
`512a3355cea79583d73b49f690a46478fb1c772c`.
CI-E2 is tracked in issue #81. PR #82, including the auto-merged PR #80 state,
merged as `f9d8366b17ecf408a60d8f62e8bf6c4d4eebbbe1`; its tree
`d5137fca07ef71648e108ac5776072097b551df3` is identical to the independently
approved and supported-tested tree.
No runtime, API, ABI, checkpoint format, numerical tolerance, or Wave 3 work changes.

## Coverage-preserving changes

| Work | Before | Accepted |
|---|---|---|
| C2 format | Seven nested invocations | One full gate |
| C2 core | Four nested invocations | One full gate |
| C2 inspect, owner, O2 encode | Three each | One each |
| C2 SAVE and LOAD | Two each | One each |
| Other C2 gates | One each | One each |
| Purposeful repeat inside each gate | Fresh AOT/compiler/determinism repetitions | Unchanged |
| C2 scheduling | One serial chain | Six independent groups |
| Canonical clean build | Repeated within overlapping workflows | One explicit job per fresh full run |
| Manual acceptance after full CI | Repeats the suite | Verified exact-tree/run/attempt evidence reuse |
| Nightly acceptance | Fresh execution | Fresh execution of the shared full engine |

The six C2 groups are format/contracts, state ownership/encoding, SAVE, LOAD,
public packaging, and operational bounds. The intact LOAD → owner → three SAVEs →
release test and both fresh measured compilations remain in the operational job.
No memory result is constructed by combining separate processes. Fresh-build,
sanitizer, failpoint and wrong-arity/isolation tests stay inside their leaf gates.

`scripts/test-c2.sh` with no arguments still runs full coverage. Repeated
`--group c2-…` arguments select disjoint groups in canonical order; invalid,
duplicate or missing arguments reject before executing work. Core/load/operational
standalone defaults retain their regression tails; focused options suppress only
those independently scheduled tails. Lightweight tests assert the full leaf union,
failure propagation and unchanged default behavior.

## Prerequisite ownership

`scripts/ci-build-prerequisites.sh --plan SUITE` prints the build plan without
mutating a checkout. Build mode first validates every selector, then cleans and
configures once, builds each selected producer once, and checks expected artifacts.
The native-numerics suite retains K1/A2/L2/I1/I2/K2/N2/N3K/T2/D2/O2 because
collision and positive-link tests consume those archives even after O2 tests move
to their own job. Native-optimizer builds exactly K1/I2/T2/D2/O2. In particular,
D2 is not removed from reverse-import prerequisites just because a missing archive
could make a negative link test fail. Contracts retain D1; loader retains
K1/I1/D2; public C2 retains C2.
Several test-local-build suites need configuration only. Byte-tokenizer retains D2
conservatively. The canonical clean-build job remains a separate production-build
proof, not a cached test artifact.

## Evidence reuse and failure handling

Blocking CI and acceptance call the same reusable full engine. Manual acceptance
may reuse only successful full CI for the exact candidate revision and actual tree.
Every suite and clean-build job, evidence job and final CI status must succeed on
the same run attempt. The evidence is data read from authenticated GitHub job logs;
its commit tree is independently checked through GitHub. Failed, skipped, duplicate,
truncated, stale and malformed evidence is not accepted. A matching running CI
prevents duplicate dispatch. Nightly runs always execute fresh.

This does not permit copying a green status from a predecessor SHA, trusting a PR
head when a different merge tree was tested, or reusing prose-only skipped CI.
Job summaries expose prerequisite/test seconds. Raw logs retain exact checks and
resource results. Workflow selection/evidence and topology tests are development
Python only, outside all delivered Eshkol/runtime closures.

Hosted acceptance run 35392576803 did not reuse the matching completed CI and was
cancelled immediately after it started the full matrix. Selector-only run
35392784207 exposed the cause: GitHub CLI 2.100.0 rejects job-log responses that
contain terminal escape sequences unless its job-log opt-in flag is present. The
reader now feature-detects that flag and applies it only to the authenticated,
numeric job-log endpoint. Older CLI versions retain their existing invocation.
Log bytes remain bounded in a non-terminal temporary file and only the unique,
strict JSON evidence marker is parsed; raw log text is never rendered in stdout or
the job summary. Repaired selector-only run 35393115074 succeeded on the hosted
CLI and selected completed run 35384195050 with exact-tree reuse. This proves the
transport repair in isolation; final combined CI and acceptance are recorded below.

## N2 reference environment

The N2 development-oracle wrapper pins `ATEN_CPU_CAPABILITY=default` and
`MKL_CBWR=COMPATIBLE`. This is a reference-generation environment constraint only:
the O2, A2, and Q0 oracle commands still use the unwrapped pinned Python, and no
runtime, native kernel, frozen fixture, generator, dependency lock, or tolerance is
changed.

Diagnostic run 35390303234 reproduced the frozen oracle on two AMD EPYC 7763
hosts, but an Intel Xeon Platinum 8573C host differed in the same 18 words under
both before- and after-generation host probing. The differences were confined to
linear forward, weight-gradient, and input-gradient tensors; metadata was
unchanged. Controlled run 35390613831 sampled six hosts. Both baseline repetitions
failed with those same 18 word differences on Intel Xeon Platinum 8573C and 6973P-C
hosts, while the three AMD hosts matched. All 12 `MKL_CBWR=COMPATIBLE` repetitions
matched the frozen SHA-256
`a32e065db6d7654600121696ba680cb15654c6712696e37fdb4a5cfb51abca03` exactly.
These runs justify the accepted CI reference pin; production numerics did not
change.

## Measurements

| Supported baseline | Observed |
|---|---|
| C2 partition, run 35350092230 | 3h40m43s total; about 3h30m testing |
| Predecessor partition, same run | 44m03s build; cancelled at 240m during T2 boundary |
| Native rerun, run 35349163791 attempt 2 | 30m50s prerequisites + 60m47s tests |
| C2 partitioned critical path | 56m22s versus 220m43s; six partition jobs sum to 154m03s |
| Earlier optimized sample, run 35384195050 at `aec841b` | 88m10s wall / 441m59s summed jobs versus 224m58s / 464m00s: 60.8% wall / 4.7% summed reduction; retained as hardware/runner variance evidence |
| Final accepted run 35393213200 | 93m19s wall versus 224m58s: 131m39s / 58.5% lower; 421m23s summed jobs versus 464m00s: 42m37s / 9.2% lower |
| Selector transport repair | Hosted selector-only run 35393115074 selected exact-tree run 35384195050 |
| Final acceptance, run 35401088338 | Same-tree evidence verified; all full-matrix jobs skipped; success in 27s wall / 15 job-seconds |
| CI-E2 supported run 35407378830 | 55m22s wall / 409m13s summed runner time; 37m57s / 40.7% wall and 12m10s / 2.9% summed-runner reduction from accepted CI-E; 169m36s / 75.4% wall and 54m47s / 11.8% summed-runner reduction from baseline |
| CI-E2 live main reuse, run 35410997717 | Original run 35407378830 directly verified; exact three-job wrapper was topology success, full suites skipped, final success; 18s wall |

Validation must compare the same pinned Ubuntu 22.04 / LLVM 21.1.8 / oracle inputs,
record queue and job/step time separately, and retain every required count and
resource bound. Higher job concurrency reduces the critical path only if runner
capacity is available; summed runner time is an execution-time measure. These
public-repository GitHub-hosted runs make no billing or cost claim. Do not
claim the removed wrapper invocation count equals an identical percentage runtime
speedup. These measurements are individual hosted samples and remain subject to
runner, hardware, queue, and cache variance. The earlier sample establishes
variance; final run 35393213200 validates the combined engine, N2 reference, and
selector repairs.
The accepted exact-tree result covered all 15 suites and 23 commands plus canonical
smoke and benchmark. Independent authoritative review approved that tree. A bounded
post-merge compatibility retest passed C2 format/core-only, 69 CI unit tests, two
Python-isolation tests, topology, rebuilt compile smoke, and exact smoke output; it
was not a full local repository, public-AOT, or operational rerun. Automatic main
run 35401303514 subsequently completed successfully; it remains historical CI-E
evidence rather than the CI-E2 acceptance run.

## CI-E2 — native critical path and main-push reuse

Tracking: [issue #81](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/81).
Status: **accepted / complete**.

The full engine now assigns the unchanged O2 gate to `native-optimizer`, separate
from `native-numerics`. Both receive the pinned development oracle. The command
union is still exactly 23; the accepted full-evidence inventory is 16 suites.
Optimizer prerequisites are K1/I2/T2/D2/O2. The other native job retains its full
prerequisites, including O2: K2's positive collision/link checks consume them.
No intentional fresh-cache, deterministic rebuild, sanitizer, oracle, failure,
or memory proof is removed, and no test binary is shared between jobs.

In accepted run 35393213200 the original native job spent 28m40s on prerequisites
and 61m44s in tests; O2 accounted for about 31m07s of that testing. CI-E2 run
35407378830 measured native-numerics at 35m02s, native-optimizer at 52m35s, and the
critical C2 operational job at 54m44s. The complete run took 55m22s wall and
409m13s summed runner time. It improved on accepted CI-E by 37m57s / 40.7% wall
and 12m10s / 2.9% summed runner time, and on the uninterrupted baseline by
169m36s / 75.4% wall and 54m47s / 11.8% summed runner time. These are individual
hosted measurements subject to runner and hardware variance, not billing claims.
Wall time uses workflow `run_started_at` through the last job's `completed_at`;
the API update interval was 55m23s and is not the reported 55m22s measure.

Main-push reuse must verify a unique associated merged same-repository PR targeting
main and the exact merge commit, then validate its latest PR CI's full inventory,
run attempt, successful jobs and aggregation, and actual checkout tree through
GitHub. A green check or equal PR-head SHA alone is not evidence. A changed tree,
missing or ambiguous association, incomplete response, pending/failed source,
skipped matrix, or malformed report falls back to fresh full CI. No arbitrary
artifact is executed, and skipped/reused CI does not itself become full evidence.

Prose-only main pushes require an exact ordinary non-forced before/after diff,
checked-out HEAD agreement, ancestry, and the same prose allowlist as PRs. They
also require completed full coverage for the base code. This prevents a docs push
from cancelling an unfinished code run and then accepting its untested code.
If a base push reused PR evidence, the original full PR evidence must be verified
directly rather than following a chain of green statuses. Uncertain or incomplete
base coverage requires a fresh full run. Docs PRs retain their lightweight checks;
merge-queue, explicit fresh CI dispatch and nightly acceptance remain full.

Supported run 35407378830 at exact head
`22e54314078ffb133cd937bd454c89e6a7712282` passed all 16 suites, all 23
commands, canonical smoke and benchmark. Independent review approved the exact tree
in [PR #82 comment 5738034159](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/82#issuecomment-5738034159).
PR #82 merged the same tree as
`f9d8366b17ecf408a60d8f62e8bf6c4d4eebbbe1`. Main run 35410997717 then
directly verified original full run 35407378830 and succeeded in 18 seconds with
exactly topology success, full suites skipped, and final success. At this
documentation review, the live prose-base branch has not yet run; its subsequent
result is recorded in issue #81. No Wave 3 work is authorized by this extension.
