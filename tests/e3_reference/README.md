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

`join.py` checks the five decoded corpus/reference records against the physical
T1/D1 files and independent D2 traversal. `validate_five_case_join(root, corpus,
bundle)` returns each case's exact start/end cursor, digest, rows, roles, CE,
state and final expectations. It checks packed repartition equality, suffix
replay and unpacked boundary differences. Decode both canonical envelopes with
`transcript.decode` first; that rejects duplicate JSON keys. The join pins the
canonical corpus and reference-bundle envelope SHA-256 values from
`e3-private-composition-aa0adbf-package/{corpus/corpus-reference.json,reference.json}`.
Those pins bind the intermediate role words and model parameter digest, which
the structural and mathematical checks alone cannot authenticate. The
lightweight semantic tests use controlled zero-logit records with patched pins;
the unpatched checker was also exercised against the existing pinned output.
The pinned M3 generator and its separate test still require Q0. A re-pin needs
fresh Q0 generation and review. None of this establishes a compiled-private
N2 verdict.

`tests/e3_private/verify_layout_results.py` is a separate private-runtime
result join. It requires the actual AOT runtime's five ordered `E3RESULT` lines,
then binds each case's six boxed outputs and cursor bytes to this frozen join.
The E3 transaction restores the starting D2 cursor after success, so the
compiled post-call cursor must equal `start_cursor`; `end_cursor` describes the
independent D2 traversal and differs for these nonempty cases. The driver uses
the reference's `maximum-batch-bytes` config so cursor identity is bytewise
comparable. The join checks exact weight/accuracy/counters and Decimal exp of
published loss. It does not independently validate published loss; that still
needs authentic row/role/CE/state evidence. The existing private runtime loss
word serves as a regression control only.

Focused commands from the repository root:

```bash
python3 -m unittest tests.e3_reference.test_join tests.e3_reference.test_corpus \
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
