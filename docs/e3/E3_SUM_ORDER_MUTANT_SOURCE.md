# E3 genuine sum-order private mutant source

This is a noninstalled source candidate for the E3 §9 M2 sum-order row. The
`tests/e3_reference/sum_order_witness_v2.json` fixes compiled candidate 3,
the first distinguishing three-batch packed raw-V256 D2 corpus in the checked
SHA-256 domain. The sealed df9546 C executable observed CE numerators
`4131ddf7`, `41314df0`, `40b294f5`: serial `41de3b31`, grouped `41de3b30`.
The separately recorded PyTorch numerators have equal groupings. The earlier
index-2 mathematical witness failed the one-use compiled C gate because its
compiled groupings both equal `41dcde82`; that failure remains preserved.
`sum_order_compiled_probe_v1.json` pins the four native observations and
source/executable/library hashes. The rational oracle rechecks each CE
reduction and sum, and the fresh gate requires the exact compiled CE words.

`tests/e3_private/sum_order_corpus.py` materializes that exact physical D1
resource and verifies its file hashes, manifest trailer, tokenizer and D2 rows.
The existing numerical private tuple selects `sum-order` without changing the
public evaluator or package ABI. Only the test-only
`ET_E3_NUMERICAL_SUM_ORDER` frame variant retains the first two actual CE batch
numerators. On the third batch it supplies `first` and RN-even
`(second + third)` as the numerator operands to the existing K1 accumulate
dispatch; K1 writes the mutated frame state. Normal and canonical builds do
not contain the fields or branch.

`tests/e3_private/verify_sum_order_mutant.py` first checks the unmutated compiled
M3 roles, CE, row identity and serial state against the frozen model reference.
It then requires the *actual* native CE numerators to distinguish serial and
grouped order, identical pre-accumulate roles/reductions, and the first changed
word at third-batch `state_f32[0]`. A rejected mutant must publish no transcript.
If a newly built executable changes CE words, the gate fails before accepting
the mutant. The original C result is never rerun.

The prepared `scripts/test-e3-sum-order-mutant.sh` specifies two deterministic
normal and mutant builds, exact package/closure comparisons, optimized private
AOT, repeat output comparison, poison runs, ASan/UBSan/LSan runs and the
first-difference/no-transcript assertion. It has **not** been run with the
candidate-3 fixture. Source-only
checks are `tests.e3_private.test_sum_order_mutant`,
`tests.e3_private.test_numerical_tuple`, `bash -n` and C syntax admission.
The sealed old executables were locally rerun on the new corpus; normal passed
and mutant was rejected at the third state. Independent review, then a fresh
supported compiled gate and sealed outputs are required before M2 can advance.
