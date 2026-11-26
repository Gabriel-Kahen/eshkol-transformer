# C2 operational phase timing candidate

The completed green [#245 supported run 45718652932](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/45718652932/job/139571540247) used head `6e1049a957103da743052980d347de0264c96075`. Merged main `5744dc443dc02b61f67719694ef3defbf5b03c92` has the **same tree** (`76e6128de9aedcee6593a1e964d224f56e842d49`), so the checked-in C2 inputs are identical. Hosted runner load can still differ between runs.

The C2 suite step ran from 23:07:35 to 00:06:25 UTC (58m50s). Its runtime PASS records give these elapsed times:

| Mode | Clang A | Clang B | Combined share of suite |
| --- | ---: | ---: | ---: |
| joint | 1,052.540 s | 1,049.193 s | 59.5% |
| file | 439.025 s | 437.602 s | 24.8% |
| tensor | 71.791 s | 71.472 s | 4.1% |
| other seven modes | | 30.063 s combined | 0.9% |

The 20 measured runtime processes total 3,151.686 s (89.3% of the step). The remaining approximately 378 s covers fixture preparation, native/AOT compilation, comparisons, strict compiler and sanitizer checks, and command overhead; the completed log does not split those costs. The two joint processes dominate, but their existing success records do not expose whether LOAD or any of the three SAVE operations consumed the time. The existing test-only progress file already records monotonic timestamps at LOAD, each SAVE's borrow/preflight/measure/encode/publish/return, and release. Previously the driver printed that file only on failure, then deleted it on success.

This candidate prints each validated joint progress file to **driver stderr after the corresponding runtime and its checks succeed**. It reports `PROGRESS UNAVAILABLE` for missing or unsafe files. Runtime stdout, PASS and MEASURE records, progress-file schema, source workload, peak-RSS measurement, 1,800-second watchdog, and admission thresholds remain unchanged. A later supported gate can subtract adjacent `monotonic-ms` values to locate the slow phase on each fresh joint process, provided the corresponding phase boundaries are present. The unchanged validator accepts a syntactically valid prefix, so missing markers cannot be assigned a duration. This candidate itself is diagnostic only and does not qualify a new runtime result.

## Single-compiler native diagnostic

The exact timing candidate `2086112fbdd5175f6b3f0f809a5f241dc83c4b80`
(tree `ef1b5c7201f33460fe709789cd1e9b347223b3a7`) ran the existing
single-compiler development joint gate on network-disabled Ubuntu 22.04,
Clang 21.1.8, and Eshkol `fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`.
The gate passed 26 checks and verified both checkpoint outputs byte for
byte against the fixture. The joint process ran for 875.441 seconds with
438,112 KiB peak RSS. All 25 progress markers were present and monotonic.

| Phase | First save | Second save | Overwrite save | Total |
| --- | ---: | ---: | ---: | ---: |
| Measure | 123.131 s | 124.358 s | 123.727 s | 371.216 s |
| Encode | 128.335 s | 126.612 s | 132.756 s | 387.703 s |
| Publish | 0.046 s | 0.036 s | 0.217 s | 0.299 s |
| Entire SAVE | 251.524 s | 251.023 s | 256.714 s | 759.261 s |

LOAD took 116.155 seconds. Measure and encode together accounted for 86.7%
of the joint runtime. These timings identify the expensive phase boundaries.
Attribution within those phases needs narrower measurement. This run used
one local host and the single-compiler joint workload. The full supported
C2 gate remains a separate qualification.

The retained receipt `c2-joint-timing-2086112-20261125.niacTO` includes
the source and toolchain identity, command, 25 markers, and verified
fixture checks. The SHA-256 of `SHA256SUMS` is
`73d2c75dfcc0bc9daa7f1be354607385d5d49187ab8227d385d5b410baf64635`.
