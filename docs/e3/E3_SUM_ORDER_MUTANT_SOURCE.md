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

The fresh supported `scripts/test-e3-sum-order-mutant.sh` gate exited 0 on
`ad4ad9acdd912d8a28343f0c5ee125e6adaa2ebe`/tree
`aa17f1bf8f457a325cd1b5614018a16a61fdbf32`. Its deterministic normal,
mutant, repeat, poison and ASan/UBSan private AOT runs completed three genuine
batches. The normal runs retained the frozen compiled CE numerator words; all
four mutant runs first diverged at the third-batch numerator and rejected
without publishing a transcript. The independent compiled RESULT review at
`/home/gabe/.codex/evidence/eshkol-transformer/e3-sum-order-ad4ad9ac-a-independent-result-review-20261014/REVIEW.md`
(SHA-256 `a58151c84401527909d63b2fb6f2d3cf5c4fc86b28c9008883e24b835fd5a26f`)
accepts this exact noninstalled M2 sum-order subgate only. The earlier df9546 C
failure remains preserved. Other M2 numerical mutants and the complete E3 §9
matrix remain open; the source is not merged or installed, and no GPU result is
claimed.
