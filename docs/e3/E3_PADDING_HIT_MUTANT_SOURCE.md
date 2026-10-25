# E3 M2 padded-hit source candidate

This is a noninstalled, test-only **source candidate** on parent
`44c0418d2d2d8f3c1354c922d3eeee39b0d7e6ef` (tree
`70706c0a25cf2055815de363eec6539823a8fca8`). It does not add a
production API or establish compiled E3 acceptance.

The independent raw T1/D1/D2 fixture uses six bytes `00 00 00 00 00 01`,
N1/T2 packed start ordinal zero, and genuine masks `11,11,10`. Its final
input/target/mask is `[0,0] / [1,0] / [1,0]`. The existing C1 zero-state
loader preserves the authentic M3 model and makes all 512 logits in every
batch zero. Lowest-index argmax is zero. Thus the final valid target 1 misses
and padded target 0 would hit. The independent oracle predicts normal
`(N,W,C,tokens,batches) = (5*log(256),5,4,5,3)` and accuracy `4/5`. The
test-only mutation changes only the `correct.bool` padded hit weight; its
active count still comes from the real D2 mask. The predicted first changed
word is final-row `correct_f32` `00000000 -> 3f800000`; final C becomes 5 and
accuracy 1.0, while L3S N/W, tokens, batches, M3 roles, CE, the other five
published outputs and D2 cursor must match the baseline. The verifier
requires the entire normal oracle first, then this exact projected mutant
tuple, and rejects without creating a mutant transcript.

This is distinct from the accepted padded-position gate, which feeds `11` to
both L3S and correctness and first changes final-row N/W. It is distinct from
the capacity-denominator candidate, which first changes final-row W. Passing
`11` only to correctness also changes active count to 2 while L3S W remains
1; K1's `wb == active` preflight rejects before accumulation, so that variant
cannot supply a successful six-output M2 witness. The existing seed-1729
sum-order and all-correct corpora have padded target 0 but observed padded
argmax 56; ignoring only the hit mask is identical on those exact corpora.
The bounded observation files are
`e3-m2-padded-aa3fbd41-static-20261014-a/result/padded-position-run/observed-normal-a.json`
(SHA-256 `cf60c45b2f7eb019f6dd2d653fcbe597f6beca624e05cf96e58d6bb65ca18bae`)
and `finish-20261022/e3/correctness-a-44c0418-attempt2/gate/observed-a-all-correct.json`
(SHA-256 `e34307d7878d98f4e78d39c35202a8169d8b780ba67957b2662d6907112b2199`),
under `/home/gabe/.codex/evidence/eshkol-transformer/`.
The all-zero uniform corpus gives a padded hit but also a valid hit, yielding
`C_b=2 > W_b=1` and K1 rejects. The selected final wrong target is required
to make a valid, distinct compiled comparison.

`scripts/test-e3-padding-hit-mutant.sh` prepares duplicate resource/reference,
closed canonical and noninstalled uniform packages, normal/mutant A/B,
poisoned and ASan/UBSan/LSan runs, complete role/row/state/output/cursor
comparisons, and first-word rejection with no mutant transcript. It is
**NOT_RUN** under the shared heavy lock. Host source checks passed 21/21
(`python3 -m unittest tests.e3_private.test_numerical_tuple
tests.e3_private.test_padding_hit_source tests.e3_private.test_uniform_tuple`);
`bash -n`, Clang private-macro admission/rejection and `git diff --check`
passed. Fresh pinned Q0 generation, exact
supported f31/LLVM21 package/AOT/runtime, independent source review and
independent compiled-result review are required before M2 admission. Full E3
§9 remains open.
