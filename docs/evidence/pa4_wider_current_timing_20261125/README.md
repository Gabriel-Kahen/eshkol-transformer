# PA4-Wider current-source paired timing, 2026-11-25

The measurement used the unchanged source commit
`63a1d9f7a0fe473c7dff25ce82f63d2001c0062e` (tree
`3677bdd350ce0766cc9480b407182b5716c7cce6`). The provider SHA-256 is
`cf4b6b3f8b96cbaa827560c8ca2ce48bc62b5cfb0d0de156b9e626658f5e5c7c`.
`conditions-before.txt` records all source and benchmark blobs and hashes.

The exact outer command was:

```bash
flock -w 900 /home/gabe/.cache/eshkol-transformer-orchestration-build.lock \
  flock -w 900 /tmp/eshkol-transformer-heavy-build.lock \
  bash /home/gabe/.codex/evidence/pa4-wider-current-timing-20261125/run-locked.sh
```

It exited 0. `run-locked.sh` records the nested Docker build and benchmark
commands. The first lock acquisition attempt stopped before compilation
because an explicit container UID could not write the rootless Docker bind
mount; it produced no timing samples. The successful attempt used the image's
default user mapping. No unrelated processes were changed.

The image was Ubuntu 22.04.5 LTS with Clang 21.1.8, digest
`sha256:68e2e292a255c9b42206e2298d12770aea7594ff9b82219460cb27ee28dbdc99`.
The host was a Ryzen 7 3700X on Linux 7.2.7-1-cachyos. Docker pinned the
benchmark to logical CPU 6; CPU 14 is its SMT sibling. The six fixture shapes,
three operations, two call paths, 12 alternating AB/BA pairs, 15 warmups
per arm, and 100 iterations per sample are unchanged. The benchmark verified
all 864 timed samples against its untimed witnesses and reference tolerance.

The raw `SAMPLE` and `MEDIAN` records are in `benchmark-raw.log`.
`analyze.py` validates the full matrix and emits `summary.json`; the ratios
are online/reference, so values below one favor online. Per-iteration lower
medians and order-specific pair ranges are in the summary. Across all pairs,
the ratio ranged from 0.281660 to 0.705148. The outlying 0.281660 ratio was
one backward-invoke pair at T=64 whose reference arm took 0.102 seconds per
100 iterations, versus roughly 0.059 seconds in its other pairs. All 432 pairs
favored online in this run.

Host load averages at the benchmark start/end were 7.56/7.37 on 16 logical
CPUs. `/usr/bin/time` recorded 15.70 seconds user time over 15.81 seconds
elapsed, 99% job CPU, and 302 involuntary context switches. Between the
snapshots, CPU 6 was 3.98% idle and sibling CPU 14 was 95.91% idle. These
aggregate snapshots do not exclude brief interference. CPU frequency and
unrelated host load were not controlled. The local paired direction is
supported, but absolute latency and training throughput are not qualified.

`SHA256SUMS` seals the raw log, commands, source and host records, parser,
summary, and binary hashes. Binary files themselves are not committed.
