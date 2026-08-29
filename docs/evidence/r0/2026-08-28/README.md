# R0 canonical evidence index

All retained executions used canonical Eshkol
`https://github.com/tsotchke/eshkol.git` at
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`. The compiler identified as
`Eshkol Compiler v1.3.4-evolve` with SHA-256
`caa295b19a6e9388963aa0def99dade63656d2dcbffccad421bd1daaa1db3750`.

## Full suite

`full/` is the complete result directory from this exact command, invoked from
the repository root:

```bash
/usr/bin/bash probes/r0/run.sh \
  --eshkol-source /home/gabe/.codex/worktrees/49f7/eshkol-transformer/.deps/eshkol-src \
  --existing-build /home/gabe/.codex/worktrees/49f7/eshkol-transformer/.deps/eshkol-build-minimal \
  --work-dir /home/gabe/.cache/eshkol-r0-canonical-full-work-20260828 \
  --results-dir /home/gabe/.cache/eshkol-r0-canonical-full-results-20260828
```

Outer exit status: `1` (expected audit gaps). The harness completed and wrote
`failures=38`, 183 command records, and 184 manifest lines including the header.
Every runnable positive probe executed twice under AOT and twice under JIT. The
RMSNorm candidate was rejected before an AOT executable existed, so the harness
recorded three rather than five commands for it.

## Supplemental probes

`supplemental/` contains eight later probes found while reconciling the pinned
compiler's advertised dtype, GPU-alias, persistence, and transformer-activation
surfaces. Each used this command shape with a separate clean result/work pair:

```bash
/usr/bin/bash probes/r0/run.sh \
  --eshkol-source /home/gabe/.codex/worktrees/49f7/eshkol-transformer/.deps/eshkol-src \
  --existing-build /home/gabe/.codex/worktrees/49f7/eshkol-transformer/.deps/eshkol-build-minimal \
  --work-dir /home/gabe/.cache/eshkol-r0-canonical-supplemental-20260828/PROBE-work \
  --results-dir /home/gabe/.cache/eshkol-r0-canonical-supplemental-20260828/PROBE-results \
  --probe PROBE
```

`tensor_serialization`, `model_serialization`, `file_checksum`,
`precision_casts`, `gpu_aliases`, `activation_transformer`, and `autodiff_silu`
each exited 0 with six recorded commands and exact AOT/JIT repeat stdout parity.
`serialization_corruption` exited 1 with `failures=4`: both corrupt loaders
returned truthy values in all four executions.

## Other direct evidence

`backend-inventory.*` records the direct `--features` and `--abi-fingerprint`
observation. `independent-review.md` records the independent agent's representative
rerun and overclaim audit. `probe-source-sha256.tsv` binds the checked-in probe and
harness files to this evidence handoff.

Generated text had trailing whitespace removed after capture for repository
hygiene. Command tokens, exit codes, substantive stdout/stderr, and ANSI diagnostics
are otherwise retained verbatim.
