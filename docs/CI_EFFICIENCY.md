# CI-E: full-coverage efficiency

Status: implementation/review candidate; not accepted performance evidence.
Tracking: [issue #78](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/78).
Base: PR #77 scheduling head `e99b127b0b194c2a11d350a36d6b9f7507a66560`.
No runtime, API, ABI, checkpoint format, numerical tolerance, or Wave 3 work changes.

## Coverage-preserving changes

| Work | Before | Candidate |
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
The native suite retains K1/A2/L2/I1/I2/K2/N2/N3K/T2/D2/O2 because collision and
positive-link tests consume those archives. In particular, D2 is not removed from
reverse-import prerequisites just because a missing archive could make a negative
link test fail. Contracts retain D1; loader retains K1/I1/D2; public C2 retains C2.
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
These runs justify the candidate CI reference pin; they do not claim that this
branch has merged or that production numerics changed.

## Measurements

| Supported baseline | Observed |
|---|---|
| C2 partition, run 35350092230 | 3h40m43s total; about 3h30m testing |
| Predecessor partition, same run | 44m03s build; cancelled at 240m during T2 boundary |
| Native rerun, run 35349163791 attempt 2 | 30m50s prerequisites + 60m47s tests |
| CI-E supported wall time | Pending; no speedup claim |

Validation must compare the same pinned Ubuntu 22.04 / LLVM 21.1.8 / oracle inputs,
record queue and job/step time separately, and retain every required count and
resource bound. Higher job concurrency reduces the critical path only if runner
capacity is available; total billed compute is a separate measurement. Do not
claim the removed wrapper invocation count equals an identical percentage runtime
saving. Current running PR #77 checks are left untouched.
