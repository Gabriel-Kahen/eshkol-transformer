# T2 private merge-path performance follow-up

The public tokenizer API, limits, streamed encoder/decoder, artifact format, and
training tie break are unchanged. Whole encode and training now compact only
newly owned token vectors in place. The scan invariant is `written <= input`:
both operands are read before a match is written, so no unread token is
overwritten, including when the operands are equal. Each training document
retains its own active length; stale vector suffixes never contribute pairs.
Whole encode also counts active token IDs and skips a rank when an operand is
absent. A replacement decreases each operand count by one and creates one
fresh result ID. Equal operands therefore decrease their count twice.
When both operands are present without adjacent matches, whole encode still
scans each rank, so its worst case remains O(input bytes × merge ranks).
The token-vector space is O(input bytes) rather than a new vector on every
rank; training still recounts every document's pairs at each rank.

The development-only workload is `tests/t2/performance_runtime.esk`, measured
by `tests/t2/run_performance.py`. The runner compares every output with the
independent Python T2 reference before recording time/RSS. It records the
actual core, workload, compiler and binary hashes. Compile the workload with
the pinned Eshkol AOT using `--strict-types --no-stdlib -I /workspace`, with
an isolated `XDG_CACHE_HOME`; invoke the runner inside the same container with
`--core-source`, `--source-sha256`, `--workload-source`, `--compiler`,
`--compiler-sha256`, `--image-digest`, `--source-commit`, `--binary`, and
`--output`. The baseline core is the exact `7291ef8` file (SHA-256
`3e3ea26ca39f2449a6a488e9d57d1ed6cdf784434f6272577cbdab619e5b1268`);
the candidate core SHA-256 is
`85ea8edd19460d4225f98ec5e9b9c8e291c95dfa631daf16a5a8a3c59f9e2c3a`.
The baseline workload changes only its load path to the preserved baseline
core; its SHA-256 is `3fcf6c6e4eb73534f3c6d6527ae434cdc6e479163f73ce45e58f43e7f8751cd7`
versus `1ad713563cef9df456b8636f1e668bc6adf15b6f36363ec793aa93039b2fabca`
for the candidate workload. The compiler is pinned `fe9dfd5`, SHA-256
`7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa`.
The container image digest is
`f568a6c2f1f11decd3bdf89034bb95f5a313185851d51c80764c79f12c56394d`.
Runs used no network, a two-CPU quota and a 6 GiB memory limit on a shared
desktop host (Ryzen 7 3700X, 8 cores/16 logical CPUs, Linux
7.2.7-1-cachyos) with Ubuntu 22.04 container userspace.

Medians of three separate AOT process runs, in seconds and KiB:

| Workload | Baseline time/RSS | Candidate time/RSS |
|---|---:|---:|
| 65,536 repeated bytes, 256 ranks, absent operand | 4.00 / 282,252 | 0.90 / 16,204 |
| 65,536 overlapping bytes, 256 ranks | 2.08 / 150,556 | 0.51 / 17,548 |
| 65,536 mixed bytes, all operands present, no adjacency | 4.03 / 280,672 | 4.00 / 18,496 |
| 65,536 bytes in 4,096 documents, 64 training merges | 3.59 / 128,248 | 1.98 / 28,056 |
| 65,536 bytes in one document, 64 training merges | 1.81 / 78,208 | 1.86 / 21,116 |
| 65,536 bytes, zero merges control | 0.91 / 17,480 | 0.87 / 15,588 |
| Three bytes, 256-rank control | 0.12 / 14,920 | 0.13 / 12,732 |

These are whole-process measurements: model construction, input preparation,
and output SHA-256 are included. GNU time provided RSS; a monotonic clock
measured elapsed time around each process. Mixed bytes show no measured speed
gain when all operands remain present. Small controls and the slight one-doc
timing difference are too short or small to support a speed claim. This is a
bounded 65,536-byte implementation; these measurements do not establish
large-corpus throughput or isolated kernel latency. The source and tests keep
the accepted streaming path and exact operational ceilings.
