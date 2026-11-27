# E3 stage-3 source on composed Wave 3

Status: **source candidate; compiled qualification HOLD**. This branch replays
the two existing PR #223 commits (`b4adb72e8`, then `03a315019`) onto the
deterministically composed Wave 3 source `4ff24845a`. The replay had no merge
conflicts. The E3 runtime/fault tuple, one-shot report, package and compiled
test source retain the PR #223 blobs. The launcher changes only its shared
heavy-build lock and the conservative host qualification check below; the
source pin, tracked digest, PR #216 ancestry, OCI image digest, dependency
identity, read-only mounts, cgroup caps, custody and admission checks remain.
The E3 leaf was reviewed separately from #221 and #218/#226.

## Fail-closed preflight

Freeze and review a clean commit and tree, then invoke the launcher directly.
It acquires `/home/gabe/.cache/eshkol-transformer-orchestration-build.lock`
followed by `/tmp/eshkol-transformer-heavy-build.lock` and holds both through
container cleanup. Under both locks it rechecks the exact HEAD/tree, clean
tracked source digest, and ancestry from
`70220bbe0d27a9a8110deeedaf873055f11ca9a6`. It requires the existing
read-only fe9 dependency bundle and Q0 Python environment. It records a fresh
host resource snapshot in the evidence directory and aborts **before Docker
create** unless all four preserved qualification floors pass:

| Observation at lock time | Required bytes |
|---|---:|
| `/proc/meminfo` `MemAvailable` | 17,179,869,184 (16 GiB) |
| `/proc/meminfo` `SwapFree` | 8,589,934,592 (8 GiB) |
| Actual `/home` package/output/scratch filesystem | 34,359,738,368 (32 GiB) |
| Host `/tmp` filesystem | 4,294,967,296 (4 GiB) |

These are the previously declared stage-8 **host qualification** floors, not
new claims about PR #223's experimental guest sufficiency. The existing guest
cap remains three CPUs, 6 GiB memory and 7 GiB memory plus swap. Docker must
use the existing full `f31d1db7...` image digest with `--pull never`, network
disabled, read-only source/common Git/dependencies/Q0 mounts, and output only
under `/home`. After container creation, the launcher verifies the image ID and
HostConfig caps; the guest verifies its cgroup limits, pinned toolchain and
distinct fe9 Git identity before compilation. Any source, resource, image,
dependency, cgroup, command, timeout, OOM or custody mismatch is a failure,
never an accepting result.

At `2026-11-25T21:15:44Z`, this machine had 15,305,170,944 bytes available
RAM and 5,085,978,624 bytes free swap: **both below floor**. The actual
`/home` scratch filesystem had 174,807,658,496 bytes free, `/tmp` had
8,885,141,504, and Docker root was under `/home`. The pinned Docker image,
fe9 checkout and Q0 environment existed. The repository API reported zero
registered self-hosted runners; all checked-in workflows use `ubuntu-22.04`.
The prior [hosted resource probe](E3_HOSTED_QUALIFICATION_20261115.md) also
fell below both RAM and swap floors. No currently measured authorized host
qualifies; this branch does not launch the generated gate or dispatch CI.

## Boxed-call diagnostic on the latest-main composition

A pinned AOT diagnostic on `d7bc9cff`/tree `58819dab` exercised the production
E1B box accessor with the Stage 3 caller's scenario, path and output carriers.
The diagnostic dispatch echoed the scenario. Normal, repeat, poisoned and
ASan/UBSan/LSan runs passed for both gradient flags with identical output.
Restoring the original unboxed call reproduced SIGILL. Peak compiler RSS
was 93,904 KiB.

The retained `2223dea6` packages match the box-only source delta across
100 audited closure paths. Their independent historical receipt covers the
old normal executable and corpus manifest. The latest-main composition
changes `internal/c1/lib/transformer/checkpoint_internal.esk` within the
package closure, so the next Stage 3 run requires a fresh package build.
The diagnostic receipt at
`/home/gabe/.codex/evidence/eshkol-transformer/e3-stage3-box-abi-d7bc9cff-20261127.gD8A9t/SHA256SUMS`
records the exact evidence and scope.

## Next executable gate

On an authorized supported host that meets **all four floors at lock time**,
use a fresh evidence directory and invoke the reviewed immutable head/tree:

```sh
python3 scripts/launch-e3-r1-stage3-compiled.py \
  <reviewed-40-hex-commit> <reviewed-40-hex-tree> \
  /home/gabe/.cache/et-e3-stage3-wave3-<unique-run>/
```

The next result must pass normal, repeat, poisoned and leak-enabled
ASan/UBSan runs for absent and present gradients, then receive independent
result/custody review. A resource-only preflight or an E3 guest marker alone
does not establish numerical correctness, sanitizer qualification or E3 §9
admission. Complete-call retention and the remaining R1/R2 cuts stay open.
