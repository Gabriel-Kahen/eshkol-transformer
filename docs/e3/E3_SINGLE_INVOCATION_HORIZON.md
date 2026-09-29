# Fixed E3 single-invocation horizon witness

This bounded gate exercises the installed public `diagnostic-evaluate-fixed!`
on an authentic canonical-byte D1 corpus and packed D2 dataset. It does not
extend the fixed N1/T2/V256/D4 profile or the evaluator API. The existing
three-batch accepted reference remains the numerical oracle; this gate tests
the traversal horizon and a late D2 read failure through the real evaluator.

`tests/e3_horizon/generate.py` deterministically writes full 2,048-token D1
shards followed by one final shard containing a single token: two shards for
H=1,024 and nine for H=8,192. Independent D2 reference equations yield exactly
H packed `[1,2]` batches, all with two true positions. Only the final batch
reads the singleton shard for its second target. Nothing from these corpora is
committed. The generator validates the D1 resource and D2 row count, and its
unit test compares two independent materializations byte for byte at both
horizons.

The public caller opens D2 while all shards are valid, saves its cursor, and
uses a test-only C filesystem hook to hide the singleton final shard. It calls
the evaluator once. The only missing read is the final batch, so the expected
error is E3's `io` at its traversal stage. After restoring the shard, the
caller requires byte-identical cursor restoration and retries using the same
dataset/model authority. That retry is one evaluator invocation over exactly H
genuine batches, with 2H tokens, an exact binary32 mask-weight word, and the
cursor restored again. A leaked live batch or D2 borrow would reject the
retry. The test-only filesystem hook is linked only into the caller archive;
the installed package object is copied byte-for-byte without source changes.

The supported gate builds the fixed public package, compiles the caller at O2,
and runs both horizons with `ESHKOL_ARENA_POISON=1`. The per-process timeout is
3,600 seconds; the proposed operational peak RSS ceiling is 524,288 KiB. The
8,192 process traverses 8,191 batches before the fault and all 8,192 on retry.
The earlier two-shard fixture passed 1,024 in 345.14 seconds/162,400 KiB at
`05f4f48` and 356.70 seconds/163,372 KiB at `5fb359e`. Its 8,192 process
timed out at 1,800 and 3,600 seconds respectively; neither is a horizon pass.
In that layout, each D2 batch reread and validated the entire first shard:
16,592 bytes at H=1,024 and 131,280 bytes at H=8,192. The fixed-size shards
bound that per-batch read while preserving the same final-batch fault and full
retry. The exact-head 8,192 gate must still complete before acceptance. The
package compile retains its 900-second per-compile timeout and may need roughly
5 GiB peak RSS as prior E3 aggregate builds did; run it in the coordinated
disk-backed supported lane.
The caller prints root-arena deltas for failure and retry, and GNU time records
elapsed seconds and peak RSS. These values must be measured on the exact gate
head before acceptance; they are not asserted flat.

This leaf does not prove native cumulative allocation, balanced pins/views/FDs,
mode restoration, absent/present gradient invariance, independent 8,193-batch
preflight, sanitizer/LSan behavior, or the repeated-call retention horizons of
§9. The accepted E3 proposal keeps those gates open.
