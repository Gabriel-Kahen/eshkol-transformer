# R0 Eshkol capability probes

These probes audit the compiled Eshkol toolchain. Source or documentation presence
never establishes runtime support: only retained compile-and-run evidence can change
a capability from `untested-with-reason` to an observed classification.

The upstream revision is pinned in `eshkol.rev`. `run.sh` rejects another revision
or a dirty upstream checkout. It records toolchain/hardware metadata, command lines,
exit codes, stdout, and stderr beneath the supplied results directory. Capability
probes run twice in both AOT and JIT modes and stdout parity is checked. Every
recorded build or probe command has a wall-clock, core-dump, and output-file limit;
capability commands also have a virtual-memory limit. This bounds malformed probe
allocations without misrepresenting the uncapped provenance metadata helpers.

The harness accepts a clean canonical source checkout, records and validates its
origin and pinned commit, and validates the compiler identity. It can either build
from scratch or consume an existing F0 build. All compiler temporaries and relative
file-I/O probes run beneath the harness-owned work directory.

Run a representative probe from the repository root with an existing F0 build:

```bash
R0_RUN_ROOT="$(mktemp -d /home/gabe/.cache/eshkol-r0-canonical.XXXXXX)"
/usr/bin/bash probes/r0/run.sh \
  --eshkol-source /absolute/path/to/canonical-eshkol \
  --existing-build /absolute/path/to/f0-eshkol-build \
  --work-dir "$R0_RUN_ROOT/work" \
  --results-dir "$R0_RUN_ROOT/results" \
  --probe tensor_core
```

Omit `--existing-build` for a clean full build. Default run/compile/build timeouts
are 90/300/1200 seconds and may be changed with the corresponding command-line
options or `R0_*_TIMEOUT_SECONDS` variables. Every retained `.command` file records
the effective timeout, working directory, output cap, and virtual-memory cap. Build
and discovery phases intentionally have no virtual-memory cap; command streams are
retained up to 2 MiB each.

The harness is expected to exit nonzero when it finds a missing or broken capability;
that is audit evidence, not a harness failure. Review `manifest.tsv`, retained
streams, `assertion-failures.txt`, and `parity-failures.txt`. A zero exit code
containing `FAIL` is a failed probe, and backend advertising is not proof that work
executed on that backend.

Probe groups:

- `tensor_core.esk`: construction, shape/rank, indexing, mutation, reshape,
  transpose, a broadcast candidate, and integer/boolean/f64 values.
- `matmul.esk`: rectangular matrix multiplication and reference values.
- `activations.esk`: ReLU, GELU, sigmoid, and axis softmax reference values.
- `autodiff_*.esk`: scalar forward/gradient finite differences, repeated inputs,
  tensor gradients, normalization/activation/attention gradients, and control flow.
- `rng.esk`: fixed-seed reproducibility.
- `file_io.esk`: basic non-executable text write/read prerequisite.
- `memory_loop.esk`: bounded lifetime smoke; peak RSS alone does not prove flat RSS.
- `negative/*.esk`: malformed inputs; each must produce an actionable compile-time
  or runtime rejection without timing out or terminating by signal.

No public syntax is guessed for f16/bf16/f32 storage, device transfer/identity,
checksums, or safe tensor serialization. Those remain `untested-with-reason` unless
the pinned executable exposes a reachable contract. Unsupported named candidates
(for example batched matmul or RMSNorm) remain in the suite so their diagnostics and
exit behavior are captured rather than inferred from source.
