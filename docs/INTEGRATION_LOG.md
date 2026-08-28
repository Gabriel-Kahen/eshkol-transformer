# Integration log

This repository-side ledger mirrors contract decisions recorded in
[issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1).
Only the integration owner changes a proposed decision to `accepted` after review.

## Schema

| Field | Content |
|---|---|
| Date | ISO 8601 date |
| Workstream / issue | Roadmap ID and tracking issue |
| Decision | `proposed`, `accepted`, `rejected`, or `superseded` |
| Contract | Stable command, path, API, ABI, or file-format surface |
| Evidence | Exact command/result, CI run, review, or pending reason |
| Dependencies / retest | Affected downstream workstreams and gates |
| Reference | Commit, PR, or issue URL |

## 2026-08-28 — F0 / issue #2

- **Decision:** proposed.
- **Contract:** Canonical entry points are `make toolchain`, `make configure`,
  `make build`, `make test`, and `make smoke`, invoked with `/usr/bin/bash`.
  Eshkol source lives under `src/eshkol_transformer/`; tests under `tests/`;
  developer tooling under `scripts/`; compatibility pins under `toolchain/`.
  The initial supported CI lane is Ubuntu 22.04 x86-64, LLVM/Clang 21.1.8,
  and Eshkol commit `90cbd7130f47b8184bcc77b8d5c1b0026da980de`.
- **Evidence:** `/usr/bin/bash -c 'make clean && make configure && make build &&
  make test && make smoke'` passed locally as an explicitly unsupported CachyOS
  x86-64 / LLVM 22.1.6 compatibility probe. The test runner performed two fresh AOT
  compilations and executions with byte-identical `eshkol-transformer-smoke:v1\n`
  output and an actionable missing-toolchain negative. The supported Ubuntu 22.04
  x86-64 / LLVM-Clang 21.1.8 lane passed toolchain, fresh configure, build, test, and
  smoke in [CI run 33220286433](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33220286433).
- **Dependencies / retest:** B0 may consume these commands after F0 merges.
- **Reference:** issue #2; [PR #8](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/8);
  reviewed head `cf0752afd5e48b4043498ffc1abbcad34a709bb7`.
