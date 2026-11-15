# A2 / PA4 supported native comparison, 2026-11-15

PA4 meets the **bounded forward speed criterion** for this fixed workload: exact
output parity is preserved, total forward K1 dispatch time is 9.675% lower and
prevalidated invocation time is 50.004% lower. This does not establish a uniform
win for every shape or PA4 integration admission. Four forward dispatch pair
ranges cross parity; backward invocation is 3.179% slower in aggregate. There is
no demonstrated process-memory reduction.

## Exact inputs and evidence

- Measured clean commit: `d1c5e4d7986649d9f1398b7e11d227218c9999d0` (`dirty: false`).
- Measured tree: `da32374fc4f7b43fcb9cd35fef4e721d828af584`.
- Composition inputs: B0 `14ca170b9a9647a0a85d7740a69a7cb7e45cb2ed` and PA4
  `a4087af4f729ca601ea0f3e5adeedd620a740e08`; their owning branches were not edited.
- Baseline: `ee24ef461ef7e313c36c6c6f97ba4f3d8155bac7` A2 source,
  SHA-256 `fdb4ca172ca7aa282e787ff42b2998592c8b77a30f67d35680f7e38b2f550b02`.
- Ubuntu 22.04.5 / Clang 21.1.8; Ryzen 7 3700X; Linux 7.2.7-1-cachyos host
  kernel; CPU affinity `[2]`. Container image actually inspected as
  `sha256:68e2e292a255c9b42206e2298d12770aea7594ff9b82219460cb27ee28dbdc99`.
- Container memory limit: 512 MiB, swap disabled; exit 0, `OOMKilled: false`.
- Default checked-in comparison definition: ten admitted shapes, both operations
  and scopes; 100 warmup calls per batch, 10,000 timed calls, 12 balanced timing
  pairs and four separate-process RSS pairs per case. No sample trimming.
- Report file SHA-256: `b8f874610e6c5d85f3df6c77b4d376fd1d72cced2d139c8d55c922878420bd21`.
- Retained report: `build/benchmarks/attention-compare-v1-report.json`;
  logs and actual container inspection: `build/pa4-comparison-evidence/`.
- Full definition/report contracts and rerun commands are in
  [ATTENTION_COMPARISON.md](ATTENTION_COMPARISON.md). Generated raw reports stay
  outside Git; the checksummed report retains all samples and output witnesses.

## Measured time

Each row receives equal call counts. The aggregate ratio is summed candidate
batch time divided by summed baseline time across the ten shapes; it describes
only this shape/input mix. Positive change means more time.

| Operation | Scope | A2 total ns | PA4 total ns | PA4 / A2 | Time change |
|---|---|---:|---:|---:|---:|
| forward | dispatch | 1758007348 | 1587920979 | 0.903250 | -9.675% |
| forward | invoke | 353236292 | 176604301 | 0.499961 | -50.004% |
| backward | dispatch | 4757671953 | 4765299587 | 1.001603 | +0.160% |
| backward | invoke | 619951362 | 639657933 | 1.031787 | +3.179% |

The following ranges are observed pair ratios, not confidence intervals.
Ratios below one mean less candidate time. Lower medians and AB/BA order
summaries follow the predeclared analysis; all raw arms remain in the report.

### Forward per shape

| Shape `[N,Hq,Hkv,Tq,Tk,Dh]` | Dispatch min / median / max | Dispatch AB / BA median | Invoke min / median / max |
|---|---:|---:|---:|
| `[1,2,2,1,1,1]` | 0.856096 / 0.950396 / 1.004456 | 0.943586 / 0.984452 | 0.519499 / 0.557780 / 0.562317 |
| `[1,2,2,2,2,1]` | 0.898780 / 0.945257 / 1.000499 | 0.945257 / 0.910988 | 0.571210 / 0.594421 / 0.633379 |
| `[1,2,2,1,2,1]` | 0.904946 / 0.934554 / 0.970719 | 0.947543 / 0.914276 | 0.540789 / 0.553767 / 0.565239 |
| `[1,2,2,1,2,2]` | 0.909158 / 0.962266 / 1.013806 | 0.953588 / 0.963043 | 0.469822 / 0.559922 / 0.682215 |
| `[1,2,2,2,2,2]` | 0.924035 / 0.957506 / 1.008420 | 0.955891 / 0.957506 | 0.568285 / 0.579986 / 0.954470 |
| `[1,4,2,1,1,2]` | 0.879788 / 0.904906 / 0.964647 | 0.904906 / 0.902811 | 0.497887 / 0.515666 / 0.830798 |
| `[1,4,2,3,3,2]` | 0.853675 / 0.874765 / 0.952139 | 0.865293 / 0.883397 | 0.420684 / 0.456847 / 0.529777 |
| `[2,4,2,2,3,4]` | 0.881845 / 0.896907 / 0.908878 | 0.894432 / 0.896907 | 0.493944 / 0.524903 / 0.637043 |
| `[2,4,2,3,3,2]` | 0.821647 / 0.839726 / 0.882371 | 0.832116 / 0.839726 | 0.396829 / 0.446096 / 0.489598 |
| `[2,4,2,1,3,2]` | 0.862083 / 0.900517 / 0.952645 | 0.907683 / 0.867765 | 0.447426 / 0.502863 / 0.612509 |

### Backward per shape

| Shape `[N,Hq,Hkv,Tq,Tk,Dh]` | Dispatch min / median / max | Dispatch AB / BA median | Invoke min / median / max |
|---|---:|---:|---:|
| `[1,2,2,1,1,1]` | 0.943334 / 0.976338 / 1.040506 | 0.976338 / 0.952304 | 0.979676 / 1.008332 / 1.050025 |
| `[1,2,2,2,2,1]` | 0.940093 / 0.971461 / 1.139790 | 0.954328 / 1.012800 | 1.006617 / 1.039984 / 1.362981 |
| `[1,2,2,1,2,1]` | 0.956777 / 0.999688 / 1.030703 | 0.985481 / 0.999688 | 0.821468 / 1.071636 / 1.089381 |
| `[1,2,2,1,2,2]` | 0.919277 / 0.993979 / 1.014922 | 1.007581 / 0.986317 | 0.824640 / 1.065535 / 1.223096 |
| `[1,2,2,2,2,2]` | 0.968015 / 0.989112 / 1.030273 | 0.989112 / 0.978866 | 0.902530 / 1.090006 / 1.116702 |
| `[1,4,2,1,1,2]` | 0.943246 / 0.993551 / 1.121015 | 1.049182 / 0.967751 | 0.765874 / 0.975611 / 1.350209 |
| `[1,4,2,3,3,2]` | 0.971445 / 0.990910 / 1.015428 | 0.992749 / 0.986906 | 1.009561 / 1.052861 / 1.102592 |
| `[2,4,2,2,3,4]` | 0.987196 / 0.991433 / 1.036123 | 0.988209 / 0.992329 | 0.972078 / 1.024676 / 1.084954 |
| `[2,4,2,3,3,2]` | 0.989343 / 1.012571 / 1.022730 | 1.015069 / 1.005814 | 0.979990 / 1.016906 / 1.053606 |
| `[2,4,2,1,3,2]` | 0.989608 / 1.006803 / 1.015510 | 1.002495 / 1.006803 | 0.964014 / 1.044551 / 1.121601 |

All ten forward per-shape paired medians are below one in both scopes.
Six forward dispatch shapes and all ten forward invoke shapes have every paired
ratio below one. The smaller dispatch benefits have order/run variation; they
are not individually established uniform wins. Backward is an unchanged
algorithm control, not an optimized feature; its variation and regressions
remain visible above and cannot be silently subtracted from forward results.
Code layout, frequency, scheduler and cache effects remain possible influences.

## Process RSS

| Provider | All selected-process samples, min–max KiB |
|---|---:|
| `a2.cpu-f32.serial` | 1960–2292 |
| `pa4.cpu-f32.row-scratch.candidate` | 1952–2292 |

Per-case provider ranges overlap and median differences have mixed signs.
The same binary links both implementations while initializing/executing only
one provider per RSS arm. Full-process RSS includes runtime, code, stack, setup,
warmup and serialization; it cannot isolate the fixed row scratch or an
incremental attention working set. These samples support no memory-win claim.

## Validation and remaining gates

- `bash scripts/test-b0.sh` in the supported native container: **59/59 passed**.
- Default comparison wrapper and report verification: **40 cases passed**;
  exact bits match across providers, repetitions, scopes and separate RSS arms.
- Supported candidate numerical/reference/gradient gate: **1,045 checks**;
  baseline numerical/gradient gate: **1,041 checks**; ten-shape forward/backward
  bitwise parity gate passed on unchanged production blobs.
- Independent source/test review approved, including the discovered-test import
  repair. The initial supported attempt failed test discovery before the
  performance run; its exit-1 log and non-OOM container inspection are retained.
- The bounded speed prerequisite is evidenced for forward attention on this
  exact recipe. Full PA4 admission still requires exact-head repository/AOT/hosted
  gates and integration review; this comparison does not provide those gates.
- Shapes remain tiny (`Tk <= 3`), masks include fully masked rows, the fixed
  mix is not representative training, CPU 2 was pinned but not reserved, and
  other host work continued. No large-context, GPU, training-throughput,
  cross-host, scratch-memory or universal speed claim follows.
