# Agent instructions

This project is being developed through coordinated Codex tasks.

## Working method

- Use `gpt-5.6-sol` with high reasoning for project tasks.
- Use subagents for independent implementation, tests, review, or documentation when
  the task has separable work.
- Work from the roadmap dependency graph. Do not implement a dependent subsystem
  against an invented upstream API.
- Keep changes scoped to the assigned workstream. Coordinate contract changes with
  the orchestrator before merging.
- Preserve user and other-agent changes. Never reset or overwrite unrelated work.
- Make focused commits with tests and update the relevant status in `docs/ROADMAP.md`.

## Engineering standards

- Correctness before optimization. Every differentiable operation needs numerical
  gradient checks and a documented tensor-shape contract.
- Never hide a CPU fallback, scalar fallback, approximate gradient, unsupported dtype,
  or unavailable device. Detect it and report it explicitly.
- Do not claim GPU execution, mixed precision, determinism, resume equivalence, or
  performance without a test or measurement that proves it.
- Hot numerical paths must use Eshkol tensor operations or reviewed native kernels,
  not recursive scalar list/vector loops.
- Training artifacts use versioned, checksummed, non-executable formats. Do not load
  arbitrary code from checkpoints.
- Reproducibility state includes model, optimizer, scheduler, RNG, tokenizer identity,
  dataset cursor, configuration, and token count.
- Python and PyTorch are allowed only in development tests, reference generators, and
  interoperability tools. The delivered training path must be Eshkol-native.
- Add negative tests for malformed shapes, indices, shards, and checkpoints.
- Avoid large generated fixtures in Git. Generate deterministic fixtures in tests.

## Required handoff

Each implementation task must report:

1. APIs and contracts added or changed.
2. Tests and commands run, with results.
3. Measured limitations and unsupported cases.
4. Follow-up dependencies or risks.
5. Commit or pull-request reference suitable for integration.

