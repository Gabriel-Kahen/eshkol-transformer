# E3 genuine sum-order private mutant source

This is a noninstalled source candidate for the E3 §9 M2 sum-order row. The
accepted `tests/e3_reference/sum_order_witness_v1.json` fixes the first real
three-batch packed raw-V256 D2 corpus reached in the 64-candidate search. The
seed-1729 registered M3/f32 CE numerators make serial `41dcde81` and grouped
`41dcde80` in the independent rational RN-even oracle. Those are mathematical
reference words, not a claim about compiled native observations.

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
If native CE rounding erases the difference, the gate fails as an unresolved
reachability issue rather than relabeling the mathematical fixture as compiled
proof.

The prepared `scripts/test-e3-sum-order-mutant.sh` specifies two deterministic
normal and mutant builds, exact package/closure comparisons, optimized private
AOT, repeat output comparison, poison runs, ASan/UBSan/LSan runs and the
first-difference/no-transcript assertion. It has **not** been run. Source-only
checks are `tests.e3_private.test_sum_order_mutant`,
`tests.e3_private.test_numerical_tuple`, `bash -n` and C syntax admission.
Independent review, then a supported fresh compiled gate and sealed outputs
are required before the M2 row can advance.
