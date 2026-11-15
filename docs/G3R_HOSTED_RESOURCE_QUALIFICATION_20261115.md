# G3-R hosted resource qualification — 2026-11-15

**Measured result: NO for the standard supported runner against the unchanged
G3-R/C2 gate floors.** No toolchain installation, swap extension, aggregate
compile or generated Eshkol runtime was attempted.

[Probe run 43744457531, attempt 1](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/43744457531)
completed on Ubuntu 22.04.5 x86-64, image `ubuntu22` version
`20261111.360.2`, using source `34511e352c30342c66a6214099573d7c17462210`,
tree `2cfebfd07b688b83b0bcccbebee048d6d52406c4`.

| Resource | Measured | Required | Result |
| --- | ---: | ---: | --- |
| Available RAM, KiB | 15,462,760 | 16,777,216 | fail |
| Free swap, KiB | 3,145,724 | 8,388,608 | fail |
| Workspace free bytes | 92,600,385,536 | 34,359,738,368 | pass |
| Docker-root free bytes | 92,600,385,536 | 34,359,738,368 | pass |

The successful workflow status means the measurement completed. Its sealed
JSON explicitly reports `resource_floor_met: false`. Both disk observations
were taken before toolchain installation; they do not promise later capacity.
`/tmp` was not measured. Future scratch remains under the chosen disk-backed
workspace, and an actual launch must recheck its own paths while holding locks.

The downloaded artifact `g3r-resource-probe-43744457531-1` (ID
`14173593926`), raw job log, run metadata and independent measurement verdict
are retained under the integration worktree's
`build/evidence/g3r-launcher-hardening-20261115/`. The artifact's `SHA256SUMS`
verifies. The observation can be shared with E3 without establishing any E3
source, resource-policy or runtime admission.

The repository API reports zero self-hosted runners and owner type `User`.
GitHub's [larger-runner policy](https://docs.github.com/en/actions/how-tos/manage-runners/larger-runners/manage-larger-runners)
limits that feature to eligible organizations and enterprises. No configured
larger alternative was found for this personal repository. This is a verdict
on the measured available profile, not a claim about all possible provisioned
machines. Changing infrastructure remains separate from these source changes;
the existing acceptance limits are unchanged.
