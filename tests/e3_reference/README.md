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

`sum_order_witness.py` performs a finite, single-threaded search of exactly 64
SHA-256-derived six-byte raw V256 corpora. Each candidate yields three actual
packed D2 batches; the registered seed-1729 M3 equations and f32 CE produce the
batch numerator words. The first genuine serial-versus-grouped RN-even
counterexample is frozen in `sum_order_witness_v1.json`, with exact source,
physical D1, D2 rows, parameter, 21-role-trace, Q0-lock, CE and numerator
checksums/words. The rational oracle recomputes both groupings from those words.
This is still a development reference; it does not prove native E3 traversal or
compiled-private M2 acceptance. Run the bounded source gate with a 30-second cap:

```bash
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE timeout 30s "$Q0_PYTHON" \
  -m unittest tests.e3_reference.test_sum_order_witness -v
```
