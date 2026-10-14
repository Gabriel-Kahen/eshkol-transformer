# E3 §9 exp-before-global host candidate

The committed [`exp_before_global_v1.json`](../../tests/e3_reference/exp_before_global_v1.json)
freezes one source-only candidate from the existing `packed-single` D2 resource and
seed-1729 M3 development reference. Its physical D1 file hashes, tokenizer hash,
model parameter hash, exact start cursor, pinned source/toolchain hashes, three
active masks, and exact binary32
L3S words bind it to the independent five-case corpus. The host oracle computes
`exp(N / W)` after serial aggregation. The specified mutant computes each
`exp(N_b / W_b)`, weights that result by `W_b` in binary32, sums serially, and
divides by global `W`. The two candidate perplexity words are `437f82e7` and
`437f8748`, 1,121 binary32 ULP apart. A changed final perplexity word is rejected
by the existing standalone transcript comparator.

This is a reachability candidate based on the pinned PyTorch reference CE, not a
compiled-private mutant kill. A later noninstalled AOT gate must first show the
authentic compiled CE and both admitted baseline/mutant executions on its exact
tree, then identify the first meaningful final-word difference against a frozen
independent transcript. No production API, provider, evaluator, or package
manifest changes here. Full M2 and §9 remain open.

Host check (with pinned Q0 Python/PyTorch and a disk-backed `TMPDIR`):

```bash
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "$Q0_PYTHON" -B \
  -m unittest tests.e3_reference.test_exp_before_global -v
```
