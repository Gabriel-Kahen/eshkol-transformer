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

## Hosted serial control

The paired receipt must first be verified for [PR #263](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/263).
Use trigger `6cd70932351dadfb44f8471b1c047174e6c2ba7e`, workflow base
`f9fd2de24ff10cd819a70b51f1697eb6e8fc3bfe`, and checked-out source
`15f2522946b8ee13b9f166967a99f2665263d4d0` at tree
`672e9da3d40061278e94ca0aa1954e98936afa7e`. Verify the exact hosted run and
attempt, the artifact seal, successful suite and collector, all six
overlapping trace TSV/JSON pairs, and the complete compiler timing inventory.
Keep the paired artifact, job JSON, and job log together.

Once the paired receipt is verified and the run is coordinated, a same-repo
documentation-only PR from `codex/p1-matched-serial-trigger-*` into
`codex/p1-matched-serial-receipt-20261127` can run the serial control. Verify
the exact base and trigger SHAs and that the trigger leaves the reviewed
workflow intact. The control uses the same pinned source, Ubuntu 22.04 x64
runner label, compiler and cache key, prerequisite clean/configure, limits,
sanitizer flags, and complete parameter-state suite. It sets
`P1_TWO_COMPILERS=0 P1_PAIR_MEMORY_TRACE=0 P1_KEEP_TMP=1`, rejects retained
paired traces, verifies the P1 pass and the complete GNU time inventory
including the expected compiler failures, and seals only the diagnostic
evidence. Either hosted job is diagnostic-only.

After both receipts are verified and extracted, save the original provider job
JSON and raw log for each exact run attempt. Fetch both using
`gh api repos/Gabriel-Kahen/eshkol-transformer/actions/jobs/<job-id>` and the
same path with `/logs` appended. Run the offline comparison from this branch.

```sh
python3 -m tests.ci.p1_serial_comparison \
  --pair "$pair_receipt" --pair-job "$pair_job_json" --pair-log "$pair_job_log" \
  --serial "$serial_receipt" --serial-job "$serial_job_json" --serial-log "$serial_job_log" \
  --serial-base "$serial_base" --serial-trigger "$serial_trigger"
```

Missing or mismatched evidence leaves the comparison unavailable. The gate
requires complete sealed receipts, matching job label and hosted runner image,
the same toolchain cache key and restore outcome, matching OS, source-input
hashes and toolchain provenance, and matching compiler commands and exit
statuses. It reports provider suite-step wall time and GNU time elapsed
seconds and peak RSS per compiler command. The randomized scratch directory
and generated compiler cache names are normalized for command matching. The
raw timing logs and pair traces remain in their respective receipts.

The pair and serial jobs run on separately provisioned hosted machines. The
runner image, cache restore and compiler identity checks establish a matched
configuration. Host placement, cache archive bytes, machine state and
contention can differ. Report the two observed wall times, per-command peak
RSS and sampled paired RSS with their own units and scope. A worker or
container memory comparison requires a separately controlled experiment that
records cgroup current, peak and swap for both schedules.
