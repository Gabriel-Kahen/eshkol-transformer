# Agent instructions

This project is being developed through coordinated Codex tasks.

## Working method

- Use `gpt-6-sol` with high reasoning for project tasks.
- Use subagents for independent implementation, tests, review, or documentation when
  the task has separable work.
- Default to one implementer and one independent reviewer for a bounded change.
  The reviewer covers source and tests together; add a specialist only for a
  concrete unresolved risk. Do not repeat an unchanged review.
- Coordinate long builds by completion-triggered handoff. Use one bounded wait for
  relevant tasks and avoid repeated polls of unchanged processes or logs. Check
  a live build only when its result, resource condition, or a decision needs it.
- For hosted CI, avoid `gh run watch` and repeated status queries. Check once
  after the expected full-suite duration, then only when a result is needed;
  stop querying if the GitHub API reports a rate limit.
- Read only the relevant roadmap section and source contracts for a bounded task;
  do not reload the full project history on each follow-up. Keep implementation
  handoffs under 500 words: accepted contract, changed files, exact commit/tree,
  test result, unresolved finding, and next completion gate. Link sealed logs
  rather than pasting them into the conversation.
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

Keep the handoff short: exact contract and changed files, command/result evidence,
remaining blocker, next gate, and immutable commit/tree. Link to detailed logs
instead of pasting them or reconstructing the full project history.
