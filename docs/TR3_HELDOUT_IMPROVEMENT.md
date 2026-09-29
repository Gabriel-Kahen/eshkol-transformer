# Bounded installed held-out improvement witness

This test-only acceptance witness exercises the installed `trainer-create`,
`trainer-evaluate!`, `trainer-train!`, and `metrics-ref` APIs in the same
registry-owning TR3/E3 aggregate. The Eshkol caller uses the genuine fixed
N1/T2/V256/D4 diagnostic model and CPU-f32 AdamW update path. No test hook
changes parameters, gradients, the validation result, or the update count.
The only test helper converts an already published f32 metric into its exact
binary32 word for output checking.

`prepare_heldout.py` writes authentic D1 resources from the frozen TR3
mathematical reference: packed training shard `[3,197,41]` and unpacked
held-out shards `[5,197,197]`, `[6,41]`. The two active training transitions
and three held-out transitions are disjoint. Both datasets use the installed
byte tokenizer, D2 batch size 1, sequence length 2, no shuffle, and fixed
resource limits. The held-out dataset has two batches with mask weights
`[2,1]`, giving exact weight `3.0f` (`1077936128`), three active tokens, and
two batches per evaluation.

The caller first evaluates and checks that both D2 cursors are preserved. It
then performs exactly 32 updates using `trainer-train!` with an invocation
limit, requiring the summary counters 32 updates, 64 training tokens, and 31
EOS rewinds. It reevaluates the same validation dataset, checks cursor
preservation, and reads the first public result again to prove that later
publication did not mutate it. The separate checker rejects malformed or
extra output, incorrect f32 words/counts, missing updates, nonfinite or
nonpositive loss, and any loss drop below the **predeclared** `0.25f`
(`0x3e800000`) threshold. Its unit tests include a no-improvement control,
an insufficient-improvement control, wrong counts, changed prior metrics,
and nonfinite loss. The gate records the observed exact before/after words in
`heldout-check.stdout`; their values are learned from the linked run rather
than asserted from the development-only PyTorch reference.

Run the pinned supported package gate with
`scripts/test-tr3-public-installed-linked.sh`; it compiles the caller against
copied installed facades and the same raw aggregate used by the existing f32
inspector, runs with arena poisoning, and seals its output and genuine corpus
files. The implementation candidate has only passed focused source and checker
tests so far; the fe9 linked result is pending independent review and gate
execution. This fixed-profile witness does not establish arbitrary model,
tokenizer, dtype, device, corpus, training duration, generalization, full
checkpoint resume, or broad TR3 acceptance.
