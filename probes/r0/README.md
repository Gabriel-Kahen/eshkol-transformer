# R0 Eshkol capability probes

These probes audit the compiled Eshkol toolchain. Source or documentation presence
never establishes runtime support: only retained compile-and-run evidence can change
a capability from `untested-with-reason` to an observed classification.

The upstream revision is pinned in `eshkol.rev`. `run.sh` rejects another revision
or a dirty upstream checkout. It records toolchain/hardware metadata, command lines,
exit codes, stdout, and stderr beneath the supplied results directory. Capability
probes run twice in both AOT and JIT modes and stdout parity is checked. Every
subprocess has a wall-clock, core-dump, virtual-memory, and captured-stream limit so a
malformed allocation cannot hang the audit or fill the host filesystem.

Run from the repository root:

```bash
/usr/bin/bash probes/r0/run.sh \
  --eshkol-source /absolute/path/to/eshkol \
  --work-dir /tmp/eshkol-r0-work \
  --results-dir /tmp/eshkol-r0-results
```

The harness is expected to exit nonzero when it finds a missing or broken capability;
that is audit evidence, not a harness failure. Review `manifest.tsv`, retained
streams, `assertion-failures.txt`, and `parity-failures.txt`. A zero exit code
containing `FAIL` is a failed probe, and backend advertising is not proof that work
executed on that backend.

Probe groups:

- `tensor_core.esk`, `rank_nine.esk`, `view_alias.esk`, and
  `transpose_ownership.esk`: construction, shape/rank, indexing, mutation, reshape,
  transpose, and observed alias/copy behavior.
- `tensor_dot.esk` and `matmul.esk`: rank-1 dot and rectangular matrix multiplication.
- `dtype_*.esk` and `precision_f64.esk`: semantic integer/boolean behavior and a
  direct f64 precision discriminator; these do not infer physical dtype storage.
- `activations.esk`: ReLU, GELU, sigmoid, and axis softmax reference values.
- `autodiff_*.esk`: scalar forward/gradient finite differences, repeated inputs,
  tensor gradients, normalization/activation/attention gradients, and control flow.
- `rng.esk`: fixed-seed reproducibility.
- `file_io.esk`, `file_binary_io.esk`, and `file_rename.esk`: text/byte round trips
  and a rename prerequisite; these do not prove a durable atomic-write protocol.
- `memory_loop.esk`: bounded lifetime smoke; peak RSS alone does not prove flat RSS.
- `negative/*.esk`: malformed indexing, reshape, and matmul; each must fail.

No public syntax is guessed for f16/bf16/f32 storage, device transfer/identity,
checksums, or safe tensor serialization. Those remain `untested-with-reason` unless
the pinned executable exposes a reachable contract. Unsupported named candidates
(for example batched matmul or RMSNorm) remain in the suite so their diagnostics and
exit behavior are captured rather than inferred from source.
