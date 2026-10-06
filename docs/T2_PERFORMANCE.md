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

The development-only workload is `tests/t2/performance_runtime.esk`, measured
by `tests/t2/run_performance.py`. The runner compares every output with the
independent Python T2 reference before recording time/RSS. It records the
actual core, workload, compiler and binary hashes. Compile the workload with
the pinned Eshkol AOT using `--strict-types --no-stdlib -I /workspace`, with
an isolated `XDG_CACHE_HOME`; invoke the runner inside the same container with
`--core-source`, `--source-sha256`, `--workload-source`, `--compiler`,
`--compiler-sha256`, `--image-digest`, `--source-commit`, `--binary`, and
`--output`. The baseline core is the exact `7291ef8` file; the candidate core
is the file in this change. The compiler is pinned `fe9dfd5`, SHA-256
`7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa`.
The container image digest is
`f568a6c2f1f11decd3bdf89034bb95f5a313185851d51c80764c79f12c56394d`.
Runs used no network, two CPUs and a 6 GiB memory limit.

Preliminary medians of three separate AOT process runs, in seconds and KiB:

| Workload | Baseline time/RSS | Candidate time/RSS |
|---|---:|---:|
| 65,536 bytes, 256 ranks, no matching pair | 4.30 / 282,200 | 0.97 / 16,244 |
| 65,536 overlapping bytes, 256 ranks | 2.21 / 150,452 | 0.52 / 15,016 |
| 65,536 bytes in 4,096 documents, 64 training merges | 3.73 / 126,208 | 2.01 / 27,524 |
| 65,536 bytes in one document, 64 training merges | 2.01 / 80,168 | 1.93 / 19,332 |
| 65,536 bytes, zero merges control | 0.92 / 17,080 | 0.87 / 17,664 |
| Three bytes, 256-rank control | 0.12 / 13,060 | 0.10 / 14,952 |

These are whole-process measurements: model construction, input preparation,
and output SHA-256 are included. GNU time provided RSS and centisecond time;
the final runner additionally records monotonic nanosecond elapsed time around
each process. Small controls are too short to support a speed claim. This is a
bounded 65,536-byte implementation; these measurements do not establish
large-corpus throughput or isolated kernel latency. The source and tests keep
the accepted streaming path and exact operational ceilings.
