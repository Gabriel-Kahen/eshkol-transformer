# E3 development reference tooling

This directory prepares small deterministic fixtures for later E3-PRIVATE tests.
It is not an evaluator, runtime API, native test hook, Q0 fixture family, installed
artifact or CI registration.

`corpus.py` creates a canonical raw/empty V256 T1 artifact and tiny checksummed D1
resources at a fresh output path. It derives packed, unpacked and nonzero-cursor
emissions through the accepted carrier-neutral D2 reference. All binary resources
and JSON transcripts are generated under a temporary or ignored build directory;
none is checked in.

`reference.py` requires the existing pinned Python 3.14.6 / PyTorch 2.13.0+cpu
development environment with `ATEN_CPU_CAPABILITY=default` and
`MKL_CBWR=COMPATIBLE`. It reuses the accepted M3 initializer and mathematical
forward equations, exposes a development-only 21-role trace, computes CE against
independently emitted D2 targets, and creates a clearly labeled synthetic f32
observation. `transcript.py` validates future standalone observations: model/CE
comparisons use mathematical tolerances, while reduction, accumulation, accuracy,
counters, loss and weight use exact rational binary32 arithmetic. Perplexity uses
the accepted independent Decimal reference with a two-ULP bound.

Focused commands from the repository root:

```bash
python3 -m unittest tests.e3_reference.test_corpus \
  tests.e3_reference.test_transcript tests.e3_reference.test_isolation -v

ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "$Q0_PYTHON" \
  -m unittest tests.e3_reference.test_reference -v

python3 -m tests.e3_reference.corpus --output build/e3-reference/corpus
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "$Q0_PYTHON" \
  -m tests.e3_reference.reference \
  --corpus-reference build/e3-reference/corpus/corpus-reference.json \
  --output build/e3-reference/reference.json
```

Valid N1/T2 D2 emissions have active-prefix masks `11` or `10`; `01` remains a
provider-only case and `00` cannot successfully finalize. Packed repartitioning
may preserve rows; unpacked shard boundaries deliberately change them. Synthetic
records test this development comparator only and make no native traversal,
transaction, lifetime, publication or public-evaluator claim.

`zero_model.py` adds the bounded §9 N3 development oracle. It builds four genuine
checksummed D1 corpora and derives their `11`/`10` rows and exact start/end cursor
bytes through the independent D2 reference. Fourteen distinct zero f32 parameter
owners (1,184 values/4,736 bytes) are independently shaped, with one logical
head/token-embedding tie. Both M3 forward equations run for every row; the
record stores all 21 role shapes and f64 words, f32 logits and CE, serial
`N/W/C` with token/batch counts, final four metric words, and D1/T1 provenance.
Its f32 transcript is explicitly a **prediction**, not a compiled observation.
Uniform logits make all next-zero targets correct and all next-nonzero targets
wrong; a mixed corpus and one-valid-token padded row exercise counts and masks.
They cannot witness a nonuniform target-offset mutation.

Run the focused reference gate under the pinned Q0 environment:

```bash
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "$Q0_PYTHON" \
  -m unittest tests.e3_reference.test_zero_model -v
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "$Q0_PYTHON" \
  -m tests.e3_reference.zero_model --output build/e3-reference/zero-model
```

The test generates two independent output trees and compares every file byte.
`load(output)` checks the canonical checksummed record plus T1/D1 physical bytes;
negative tests reject altered model metadata, D2 rows/cursors, roles,
accuracy, duplicate JSON fields, and a corrupt shard. Keep generated trees
under an ignored build or temporary directory. A real idle-model state-load
driver, parameter/tie/gradient/RNG observer, private AOT comparison and source
review are still required before the N3 compiled-private row can close.
