# E3 private missing-batch numerical source candidate

This noninstalled E3 §9 M2 candidate tests an omitted serial accumulation
without omitting a D2 emission. It reuses the accepted frozen three-batch
`sum_order_witness_v2.json` corpus and independent M3/CE oracle. The genuine
row masks are `11, 11, 10`; the third row is emitted with all 21 forward roles,
CE, reduction, and correctness observations before native accumulation.

Only the exact private numerical tuple can define
`ET_E3_NUMERICAL_MISSING_BATCH`. On native accumulate input `batches == 2`,
the test variant returns the prior `N,W,C,tokens,batches` as the result of that
otherwise successful call. The ordinary provider, evaluator, public package,
ABI, and other numerical mutants are unchanged. Without
`ET_E3_NUMERICAL_TESTING`, the mutant macro is a compile error.

The `sum-order-missing-batch` runtime case expects final counters `(4,2)` so
the executable reaches the independent oracle after consuming the genuine
third D2 row. The parser requires three ordered rows and 63 model roles.
`verify_missing_batch_mutant.py` first checks the normal observation against
the frozen model oracle. It then requires identical roles, D2 rows, CE,
reductions and first two states, plus the third result equal to the prior
serial state. The first named mutant difference is `batches[2].counters`
(`(4,2)` versus `(5,3)`); it checks third-row `N,W,C`, final counters and
metrics from independent binary32 arithmetic, and rejects the mutant before
writing a transcript. This corpus has third-row correct count zero, so `C`
remains exactly zero; the missing row is witnessed by the counter and `N,W`.
The runtime also requires cursor-byte equality.

The one-use `scripts/test-e3-missing-batch-mutant.sh` gate is prepared but has
not been run. It requires canonical isolation, exact object/archive/depfile
and symbol/closure inventories, two fresh private AOT builds, normal/repeat,
poison and ASan/UBSan runs, four normal transcripts and four causal mutant
rejections without transcripts. Independent source/test review must precede
that supported gate. No compiled-private, installed, public, gradient, or full
§9 acceptance follows from host source tests alone.
