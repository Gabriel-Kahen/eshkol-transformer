# E3 compiled three-batch numerical transcript candidate

This noninstalled test tuple targets the three-batch numerical portion of
[E3 §9](../E3_EVALUATION_PROPOSAL.md). It is not a public evaluator or a
replacement for the separate complete-call retention, failure, and broader
numerical boundary gates. Independent source review and a supported exact-tree
run are required before its observations can be accepted.

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

The exact `5d642a2` fe9/f31 gate built both normal numerical packages and
both compiled-mutant packages, but stopped before executable/runtime proof
in the sanitized aggregate's final export check. Clang's
`___asan_globals_registered` was a COMMON cell; objcopy did not localize
COMMON, leaving this extra name global. Docker and launcher exited one,
with unchanged pre/post pins. The preserved 308-file failed evidence seal is
`/home/gabe/e3-numerical-transcript-5d642a2-fe9-f31-20261002-prepared-c/SHA256SUMS-FAILED`
(SHA-256 `8939a6bf7036f30bc3a4fa6af185519724e700f4770b5832b5ded7830941fc7f`).
The bounded repair materializes COMMON storage during the relocatable link
only for the already-authenticated noninstalled sanitized E3 tuple, then
applies the unchanged localization and exact export check. Normal/canonical
link arguments and every export/import manifest remain unchanged. A real
Clang ASan object regression reproduces the leak and checks the corrected
private registration cell and retained sanitizer imports. Independent repair
review and a fresh supported complete numerical gate remain required.

The later exact `94b8fecd` fe9/f31 gate F stopped at sanitized package import
admission before AOT runtime. Its launcher exited one with unchanged pins; the
retained scratch lists `bcmp` and `puts` as additional noninstrumentation
imports. Clang `-O1` lowers native `memcmp` equality to `bcmp` and generated
literal `printf` calls to `puts`. The bounded numerical-sanitized-only repair
pins both the exact added and absent import sets against the unchanged normal
manifest; sanitizer references and canonical/normal inventories retain their
checks. No numerical acceptance follows from this failed gate.

## Five-layout composition (source preparation)

The N2 composition combines the guarded numerical frame observer with the
frozen five-case physical D1/D2/M3 join. Before AOT, the gate regenerates the
corpus and mathematical bundle twice, compares their bytes, authenticates the
frozen envelope digests, and checks the physical T1 artifact, D1 shards, D2
rows and cursor endpoints. The compiled observer supplies each actual row,
all 21 forward roles, CE, reduction, correct count, and serial state. The Q0
parity verifier checks those observations against the independent reference for
each layout; the final join checks six boxed outputs and the evaluator's
restored entry cursor. It also compares actual packed repartition rows and
roles, and the packed suffix against the corresponding full-run rows and roles.
Unpacked layouts retain their distinct D2 boundaries.

This composition has passed source-only checks. Its exact-source AOT and
sanitized executions remain pending; the earlier numerical gate failures do
not establish an N2 verdict for this tree.
