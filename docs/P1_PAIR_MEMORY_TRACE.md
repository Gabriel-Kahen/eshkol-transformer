# P1 paired-compiler memory trace

`P1_TWO_COMPILERS=1 P1_PAIR_MEMORY_TRACE=1 P1_KEEP_TMP=1` enables a
development-only trace during the existing P1 suite. The default serial
schedule and paired compiler commands, outputs, assertions, and GNU time lines
are unchanged. `P1_PAIR_MEMORY_TRACE=1` requires the paired schedule and a
readable Linux `/proc` and `P1_KEEP_TMP=1`; unsupported inputs fail before
compilation. The normal
P1 memory-headroom check still applies before each heavy pair.

Each pair writes `pair-N-memory.tsv` and `pair-N-memory.summary.json` beneath
the preserved P1 temporary evidence directory. The sampler reads the two Bash
compiler process groups, including their descendants, every 250 ms. Each row
contains a UTC and monotonic timestamp, both process-group IDs, each group's
resident bytes and process count, and their sum. The summary records the number
of samples, samples observing both groups, and maximum **sampled** pair RSS.
Short peaks between samples can be missed, and summing RSS can count shared
pages more than once. This is process RSS, not cgroup
`memory.current`, swap, private proportional memory, or a memory-cap proof.
The sampler itself is excluded from the pair sum.
The development sampler lives in `tests/p1/pair_memory_trace.py`; it is not a
delivered training-path dependency.

For a matched serial/paired comparison, use the same pinned source, toolchain,
runner, limits, host class, input paths and `P1_KEEP_TMP=1` in separate runs.
Record the exact source/tree and suite results; retain the compiler timing log
and all trace files. Compare wall time and per-command GNU time RSS alongside
the paired RSS series. Pair RSS is not directly comparable with the sum of
serial per-command peak RSS, since those peaks occur at different times.
Collect cgroup current/peak/swap separately on the same runner if a full
container-memory comparison is required. No hosted or serial comparison is
established by this source change.

The hosted trigger requests the six-pair memory trace from the exact pinned
P1 candidate using the reviewed branch-only diagnostic.
