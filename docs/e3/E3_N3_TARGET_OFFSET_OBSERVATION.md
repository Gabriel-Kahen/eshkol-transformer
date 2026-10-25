# N3 target-offset observation contract v1

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

The exact `7bb38a56c1dd54f5188795d6ca369ede485fd289` source/tree
`370b63652629354e8c2ca097664a0bfdb87733f1` passed the bounded F compiled
gate for this contract. Its independent result review is
`/home/gabe/.codex/evidence/eshkol-transformer/e3-n3-f-independent-gate-result-review-20261014.md`
(SHA-256 `825ed5cc7fe3b44e16915e3716ef76b3ccf4454eb01aad749388689057b8f1db`).
The gate exercised two authentic D2 corpora, pinned Q0 comparison, real private
AOT normal/repeat/poison/sanitized observations and wrong-target rejection.
Failed A–E raw and decisions remain preserved. This is one target-offset leaf
on that exact source. The current-main integration union requires independent
review and hosted checks; the other N3 cases and full E3 §9 remain open.

## 44c source successor

The source-only successor branches from `44c0418d2d2d8f3c1354c922d3eeee39b0d7e6ef`
(tree `70706c0a25cf2055815de363eec6539823a8fca8`). Its target-offset
runtime, paired D2 corpus generator, parser, verifier and Eshkol source closure
are byte-identical to the accepted F leaf. The compiled native closure changed
in `e3_frame.c`, `e3_frame_internal.h` and `m3_call_f32_integration.c`; the
successor pins their exact 44c bytes in
`tests/e3_private/test_target_offset_44c_successor.py`. The F result therefore
does not carry to this source. A separately reviewed, fresh supported compiled
gate must repeat the paired normal, repeat, poison and sanitizer observations
and wrong-target rejection. Preparation and source checks are not compiled
acceptance.
