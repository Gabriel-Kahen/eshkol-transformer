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
