# E3 stage-3 hosted resource qualification

Status: **NOT QUALIFIED / HOLD**. No hosted stage-3 compiled gate was added or
launched. This resource audit does not establish E3 §9 admission.

The reviewed stage-3 candidate is PR #223, commit
`03a315019254b0f5e4e5d98e339429b0a1dbe92a`, tree
`de658fb92c697785f86d9b4269f8b22df0f8f78e`. Its
[source and execution contract](https://github.com/Gabriel-Kahen/eshkol-transformer/blob/03a315019254b0f5e4e5d98e339429b0a1dbe92a/docs/e3/E3_STAGE3_ADMISSION_20261115.md)
remain attached to that candidate.

## Resource contracts

PR #223 requires at least **32 GiB free on the actual output/scratch
filesystem**. Its Docker limits are **6 GiB memory and 7 GiB memory plus
swap**, with three CPUs. These are experimental execution caps, not host
minimums or proof of sufficiency; the launcher records host available RAM and
free swap without imposing a host RAM/swap minimum.

The earlier, unlaunched stage-8 prepared bundle separately declared available
RAM **16 GiB**, free swap **8 GiB**, package/output filesystem free space
**32 GiB each**, and `/tmp` free space **4 GiB**. Its exact `run_once.py` and
README are retained in the evidence packet below. This audit preserves that
predeclared floor as a conservative hosted qualification constraint under the
current no-lower-floor instruction. It does not attribute those host minima to
PR #223 or reinterpret its 6/7 GiB caps as admission minima. No swap was added,
resource threshold lowered, or alternative scratch allocation assumed.

## Configured runner and measured result

Repository API inspection on 2026-11-15 found a public, personally owned
repository and zero registered repository self-hosted runners. All checked-in
workflow `runs-on` declarations on the exact PR #223 tree select
`ubuntu-22.04`; no larger runner label is configured. GitHub's
[standard runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[larger runner reference](https://docs.github.com/en/actions/reference/runners/larger-runners)
and [larger runner management guide](https://docs.github.com/en/actions/how-tos/manage-runners/larger-runners/manage-larger-runners)
describe runner options, not measured headroom for this gate.

The existing shared Wave 3
[resource-only run 43744457531, attempt 1](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/43744457531)
completed on source `34511e352c30342c66a6214099573d7c17462210`, tree
`2cfebfd07b688b83b0bcccbebee048d6d52406c4`, using
`.github/workflows/g3r-resource-probe.yml`. It measured Ubuntu 22.04.5,
x86_64, runner image `ubuntu22` version `20261111.360.2` before toolchain
installation. It did not check out or compile PR #223.

| Observation | Measured value | Preserved qualification floor | Result |
|---|---:|---:|---|
| `MemAvailable` | 15,462,760 KiB | 16,777,216 KiB (16 GiB) | Below floor |
| `SwapFree` | 3,145,724 KiB | 8,388,608 KiB (8 GiB) | Below floor |
| Workspace free | 92,600,385,536 bytes | 34,359,738,368 bytes (32 GiB) | Pass |
| Docker root free | 92,600,385,536 bytes | 34,359,738,368 bytes (32 GiB) | Pass |
| `/tmp` free | Not measured | 4,294,967,296 bytes (4 GiB) | Unmeasured |

Workspace was `/home/runner/work/eshkol-transformer/eshkol-transformer` and
Docker root `/var/lib/docker`. These disk observations are not a lock-time
measurement of an E3 evidence directory. Total RAM was 16,372,452 KiB,
itself below the 16,777,216 KiB available-memory floor on this sampled host;
freeing its guest services cannot close that gap. Total swap equaled free
swap. A runner's advertised total memory cannot substitute
for the required available-memory observation. The workflow succeeded in
recording its measurements, while its `resource_floor_met` result was
**false**. RAM and swap failures already prevent qualification; the missing
`/tmp` measurement and future scratch recheck cannot turn this result into a
pass. No unconfigured larger runner is treated as available.

## Evidence and next gate

The retained packet is
`/home/gabe/.cache/et-e3-hosted-qualification-20261115-evidence/`:
`repository.json`, `repository-runners.json`, `source-contract.json`, four
exact source-contract snapshots, `probe-run.json`, and
`hosted-probe/{resource-probe.json,SHA256SUMS}`. The probe JSON SHA-256 is
`83349591a1355fdf4a28d1a1587e42d8bdf682425d6cc83cec8b1827f2104eb5`.
The historical stage-8 launcher's SHA-256 is
`428d1e829c5fbfbdd1e959d73cb124d9e6f1589fc1e83a78ea09b691fe5cda98`.

No configured hosted option has qualified on the available measurements.
The local heavy-build hold remains in force. This bounded follow-up changes
documentation only; no compiler, runtime, new probe or stage-3 workflow ran.
The next execution gate requires an actually configured supported host with
measured headroom meeting the preserved qualification constraints, a fresh
actual-scratch check, and independent review of an exact-source launcher
recipe. Normal, repeat, poison and leak-enabled sanitizer evidence plus
independent result review remain required for bounded stage-3 admission.
