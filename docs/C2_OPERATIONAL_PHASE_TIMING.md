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

## Development-only component seam diagnostic

An opt-in single-Clang overlay (`C2_OPERATIONAL_DEVELOPMENT_SEAM_TIMING=1`)
times the existing C2 model and O2 staging indirections, the actual native
LOAD measure/stage and SAVE measure/encode call sites, and the O2 LOAD parse
indirection. It generates source copies in the developer artifact directory;
the checked-in production source, ABI, and default operational runtime are
unchanged. The validator requires the exact 26-check joint PASS, 25 ordered
progress records, all 21 seam observations, and their order and containment
inside this run's own LOAD/measure/encode endpoints. An overflow is rejected
without intercepting the underlying call.

The pinned Ubuntu 22.04, Clang 21.1.8, Eshkol `fe9dfd5241` network-none
developmental run passed with both build locks. The strict AOT compile took
109 s; the joint runtime took 863,273 ms and peaked at 455,868 KiB, 68,420
KiB below the unchanged 524,288 KiB bound. Both 16,777,216-byte output files
match the deterministic fixture byte for byte (SHA-256 `23affcc0b05c8cc4...`).
Compile and runtime stderr were empty, the outer command had 5,267,960 KiB
peak RSS and zero swaps, and the 1,800-second runtime watchdog was unchanged.

| Pass | Measure parent | Model | O2 | Native measure | Encode parent | Model | O2 | Native encode and validate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First | 122,678 ms | 121,729 | 58 | 884 | 126,444 ms | 121,903 | 87 | 4,423 |
| Second | 122,631 ms | 121,692 | 59 | 880 | 126,119 ms | 121,625 | 57 | 4,413 |
| Overwrite | 122,862 ms | 121,922 | 58 | 882 | 126,287 ms | 121,792 | 58 | 4,414 |

The six model component calls total 730,663 ms, or 97.8% of the measured
747,021 ms measure and encode parent intervals. LOAD took 116,034 ms; its
native measure and stage calls took 1,357 and 1,331 ms, and the O2 parse
indirection registered 0 ms at the 1-ms clock resolution. These are nested
intervals from one instrumented run; they are not a speed comparison with the
earlier uninstrumented run. The model seam includes C1 serialization, payload
copying, and checksum work. This diagnostic does not isolate checksum time,
so it does not justify a checksum implementation change. The next narrow
measurement, if optimization is pursued, is inside C1 model encoding.

The retained receipt is
`/home/gabe/.codex/evidence/eshkol-transformer/c2-seam-timing-20261125-v3`.
Its `SHA256SUMS` digest is `522b5bc3019dff9ee4f3baa582f8d813f458f5a7d12d193c27f217ba13928d76`.
The run-time manifest has stale default AOT source/flag fields; the receipt's
`SUPPLEMENTAL_PROVENANCE.txt` pins the actual generated source and overlay.
This is developmental measurement, not full C2 qualification or production
admission.

## Development-only C1 checksum attribution

The next diagnostic extends the same opt-in single-Clang overlay. A generated
copy of `transformer.checkpoint_internal` times only the two existing C1 encode
checksum sites: each tensor record and the C1 container. For each site it
accumulates call count, input bytes, milliseconds spent assembling the SHA
input with `c1-concat`, and milliseconds inside the unchanged `c1-sha256`.
The validator compares those totals with the six measured model-encode calls
from the same process and reports the remaining model time separately. It
requires 372 tensor hashes and six container hashes with exact input totals
of 100,656,114 and 100,646,766 bytes, respectively, derived from the fixed
joint fixture and its byte-identical C2 outputs.

Input-list construction, timing and counter bookkeeping, payload copying,
and the rest of C1 encoding remain in the residual model interval. The clock
has millisecond resolution. The SHA interval includes its normal padding and
digest conversion. This is one instrumented diagnostic, not a speed
comparison or a production checksum change. The pinned executable compiler
does not emit an executable depfile; retained source and generated-overlay
hashes supply the exact input provenance.

The pinned Ubuntu 22.04, Clang 21.1.8, Eshkol `fe9dfd5241` network-none
joint gate passed 26 assertions, all 25 progress markers and 21 prior seam
observations, and exact fixture comparisons for both 16,777,216-byte C2
outputs. Its six model calls took 729,862 ms within 746,166 ms of SAVE
measure/encode parent intervals. The C1 split from that same process was:

| Existing C1 stage | Calls | Input bytes | Assembly | `c1-sha256` |
| --- | ---: | ---: | ---: | ---: |
| Tensor record | 372 | 100,656,114 | 28 ms | 356,496 ms |
| Container | 6 | 100,646,766 | 26 ms | 373,119 ms |
| Total | 378 | 201,302,880 | 54 ms | 729,615 ms |

The measured residual inside the six model calls is 193 ms. The SHA calls
occupy 99.966% of those model intervals in this instrumented run, subject to
the 1-ms clock resolution and probe overhead. The joint process took
862,071 ms, with 406,424 KiB peak RSS below the unchanged 524,288 KiB cap;
the AOT compile took 110 s. Both compiler and runtime stderr were empty.
The outer command used 5,291,556 KiB peak RSS and zero swaps. These numbers
locate the cost in the existing CPU SHA implementation; they do not measure
any replacement, speedup, or full supported C2 qualification.

The sealed receipt is
`/home/gabe/.codex/evidence/eshkol-transformer/c2-c1-sha-attribution-20261125`.

## Test-only native C1 checksum feasibility

The opt-in `C2_OPERATIONAL_DEVELOPMENT_NATIVE_SHA=1` successor generates a
private copy of `c2_checkpoint_format.c` with a whole-input digest entry point
using that file's existing SHA-256 core. It validates Eshkol bytevector headers
and length bounds before hashing payload bytes. A generated C1 overlay replaces
only the two timed SAVE checksum sites; checked-in C1/C2 production sources,
domains, checkpoint format, ownership, reader and writer gates, and default
execution are unchanged. Four focused tests cover known answers, padding edges,
invalid FFI boundaries, and the exact joint fixture bytes. Independent source
and test review found no remaining issues.

The pinned network-none Ubuntu 22.04, Clang 21.1.8, Eshkol `fe9dfd5241`
single-compiler developmental replay passed 26 checks and 25 ordered progress
markers. Both 16,777,216-byte outputs match the existing fixture exactly
(SHA-256 `23affcc0b05c8cc431f10357fa15e8ac9c2e0fa0a323a69a020efe586bdcef07`).
The joint process reported 138,540 ms elapsed and 396,464 KiB peak RSS, below
the unchanged 1,800-second and 524,288 KiB bounds. Across the six measured
model encodes, 378 checksum calls over 201,302,880 input bytes consumed
5,309 ms of 5,594 ms total model time; input assembly consumed 64 ms.
The strict AOT compile took 114 seconds. These are one-run developmental
measurements and do not establish a host-matched or production speedup.
Full supported C2 qualification and a reviewed production design remain open.

The retained receipt is
`/home/gabe/.codex/evidence/eshkol-transformer/c2-c1-sha-feasibility-20261125`.
