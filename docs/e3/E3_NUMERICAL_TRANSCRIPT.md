# E3 compiled three-batch numerical transcript candidate

This noninstalled test tuple targets the three-batch numerical portion of
[E3 §9](../E3_EVALUATION_PROPOSAL.md). It is not a public evaluator or a
replacement for the separate complete-call retention, failure, and broader
numerical boundary gates. Independent source-and-test review approved
`30579532`/tree `44b8fbdb`; the composed tree still needs bounded integration
review and a supported exact-tree run before its observations can be accepted.

The exact `numerical_driver_root.esk` loads the unchanged private E3 driver.
The ordinary boxed `et_e3_test_run_v1` entry constructs the genuine fixed M3
model, T1 tokenizer, D2 dataset, frame, and four I2 destinations, then calls
the compiled evaluator on the packed-single D1/D2 corpus. Only the numerical
tuple compiles `ET_E3_NUMERICAL_TESTING`. At each successful native accumulate,
its guarded observer reads the staged D2 IDs, next-token targets, bool mask,
all 21 forward tensors, CE, L3S outputs, correct count, and selected
accumulator bank/counters from the authentic frame. After the six-output
publication it reads the four caller destinations and two counters. The
observer emits lowercase binary32 words and does not read expected values.

`parse_numerical_observation.py` only converts those tagged native lines into
the existing `observed-f32` development format and role TSV. The accepted
independent `tests/e3_reference/reference.py` regenerates the mathematical
model and D2 reference twice under pinned Q0 Python/PyTorch. The existing
`tests/e3_native_parity/verify_observation.py` checks every role against that
reference, authentic D2 rows/targets/masks, CE versus the actual logits and
targets, exact binary32 L3S and serial accumulator words, final metrics and
counts. The three rows include repeated/boundary IDs and active counts 2,2,1.

Two distinct compiled test mutants use the same tuple. `wrong-target` feeds
input IDs to the native CE dispatch instead of next-token targets;
`mean-of-means` adds each batch mean with unit batch weight in the native
metrics provider. Both must complete the evaluator and be rejected by CE or
accumulator value assertions, respectively. They are test-only compile modes;
the canonical private, diagnostic, installed and retention tuples never define
these macros.

`scripts/test-e3-numerical-transcript.sh` checks exact manifests, canonical
observer absence, two fresh numerical source/object/archive builds and raw
depfiles, two independent AOT caches and byte-identical runtime stdout. It
also runs poisoned optimized AOT and an ASan/UBSan/LSan build with leak
detection enabled. The runner records each failed mutant verifier transcript
without publishing a validated transcript. No supported numerical result or
full E3 §9 acceptance is claimed before the pinned gate passes.
