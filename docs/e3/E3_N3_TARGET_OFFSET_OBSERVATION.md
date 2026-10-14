# N3 target-offset observation contract, source candidate v1

This is a test-only contract for the two genuine packed D2 rows emitted from
`03c500` and `03c507`. Both rows have inputs `[3,197]` and mask `[true,true]`;
their targets are `[197,0]` and `[197,7]`. The canonical private E3 entry still
receives `(packing?, skipped-batches)` only. The existing noninstalled numerical
observer (`ET_E3_NUMERICAL_TESTING`) emits the frame records while that entry
traverses D2. The target runtime supplies the case label, six caller outputs,
saved/restored cursor and pass marker after the call.

For each invocation stdout must contain, in exact order, one `E3ROW`, 21
`E3ROLE` records in the fixed M3 schedule, `E3CE`, `E3BATCH`, `E3STATE`,
`E3FINAL`, one `E3-TARGET-OFFSET` line, and
`E3-TARGET-OFFSET-RUNTIME-PASS`. The frame field syntax and binary32 word
encoding are the accepted numerical observer's syntax. The target line is
tab-separated: case name, four decimal binary32 output words, two decimal
counters, saved cursor hex and restored cursor hex. No extra records are
permitted. `parse_target_offset_observation.py` emits an `observed-f32` JSON
record, a complete role TSV, and `e3-target-offset-result-v1` JSON. The result
binds all six outputs to the frame final, the saved cursor to the generated D2
start cursor, and its provenance hashes to the exact corpus manifest and raw
stdout. The corpus manifest itself is `e3-target-offset-corpora-v1`, containing
the exact D1 bytes, D2 row and both D2 cursor snapshots.

For each case, parse with `python3 -B -m
tests.e3_private.parse_target_offset_observation` using `--stdout`, `--corpus`,
`--case`, `--observed`, `--roles` and `--result`. Then run
`python3 -B -m tests.e3_private.verify_target_offset` with `--reference`,
`--corpus`, `--report`, and `--stdout-CASE`, `--observed-CASE`, `--roles-CASE`,
`--result-CASE` for both `CASE=target-zero` and `CASE=target-seven`.
The verifier requires the pinned Q0 Python/PyTorch oracle environment. It
checks the genuine D2 row, all 21 roles and shapes against independent M3
values, observed logits and per-position CE, exact reductions and state,
six final outputs, restored cursor, corpus/reference/stream provenance and the
paired equal-logit/different-second-target CE witness. It writes a versioned
comparison report only on success.

The source-only parser tests do not establish a compiled observation. The next
gate must compile the target runtime against the accepted numerical test
package, run both genuine D1 corpora, compare under the pinned Q0 environment,
and independently review the exact package/source and negative mutations.
