# Fixed E3 single-invocation horizon witness

This bounded gate exercises the installed public `diagnostic-evaluate-fixed!`
on an authentic canonical-byte D1 corpus and packed D2 dataset. It does not
extend the fixed N1/T2/V256/D4 profile or the evaluator API. The existing
three-batch accepted reference remains the numerical oracle; this gate tests
the traversal horizon and a late D2 read failure through the real evaluator.

`tests/e3_horizon/generate.py` deterministically writes two D1 shards for each
admitted horizon H=1,024 or 8,192. The first has 2H tokens, the second has one
token. Independent D2 reference equations yield exactly H packed `[1,2]`
batches, all with two true positions. The second shard contains only the final
batch's second target. Nothing from these corpora is committed. The generator
validates the D1 resource and D2 row count, and its unit test compares two
independent materializations byte for byte.

The public caller opens D2 while both shards are valid, saves its cursor, and
uses a test-only C filesystem hook to hide the second shard. It calls the
evaluator once. The only missing read is the final batch, so the expected
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
On the pinned `05f4f48` gate, 1,024 completed in 345.14 seconds at 162,400 KiB,
while 8,192 reached the former 1,800-second timeout (exit 124, 160,436 KiB).
Linear scaling of the measured 1,024 runtime projects 2,761.12 seconds for
8,192, so the new bound leaves about 30% headroom. This projection is a timeout
choice, not an 8,192 runtime result; the exact-head gate must still complete.
The package compile retains its
900-second per-compile timeout and may need roughly 5 GiB peak RSS as prior E3
aggregate builds did; run it in the coordinated disk-backed supported lane.
The caller prints root-arena deltas for failure and retry, and GNU time records
elapsed seconds and peak RSS. These values must be measured on the exact gate
head before acceptance; they are not asserted flat.

This leaf does not prove native cumulative allocation, balanced pins/views/FDs,
mode restoration, absent/present gradient invariance, independent 8,193-batch
preflight, sanitizer/LSan behavior, or the repeated-call retention horizons of
§9. The accepted E3 proposal keeps those gates open.
