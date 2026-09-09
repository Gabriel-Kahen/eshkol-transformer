# Wave 2 primitive acceptance

Integration closeout, 2026-09-09. A2, T2, L2, I2, and N2 are accepted and
complete within their documented bounded contracts. This supersedes their earlier
active/proposed/review status, not their capability or lifetime limitations.
D2, O2, C2, the complete model, and the first-release training gates have separate
acceptance decisions; this closeout does not mark Wave 2 as a whole complete.

## Independent review and supported evidence

For each row, the reviewed head, supported PR checkout, and actual implementation
merge have the same tree. The post-merge runs use Ubuntu 22.04 / LLVM-Clang
21.1.8 and pass build, the integrated test matrix, smoke, and benchmark. Benchmark
success is repository smoke evidence, not a numerical acceleration claim.

| Workstream | Implementation PR / merge | Independent approval | Supported PR run | Post-merge evidence |
|---|---|---|---|---|
| A2 | [#52](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/52), `c4e52b8` | [A2-R](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/52#issuecomment-5514404661) | [33540065067](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33540065067) | [Exact-merge run 33667505816](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33667505816) |
| T2 | [#54](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/54), `3006c5c` | [T2-R](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/54#issuecomment-5542400784) | [33822681026](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33822681026) | [Exact-merge run 33906135154](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33906135154) |
| L2 | [#56](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/56), `8ff3d7e` | [L2-R](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/56#issuecomment-5546458897), [CI condition cleared](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/56#issuecomment-5547571678) | [33917785648](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33917785648) | [Exact-merge run 33981828638](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33981828638) |
| I2 | [#53](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/53), `309de72` | [I2-R](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/53#issuecomment-5556348520) | [33988137736](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/33988137736) | [Exact-merge run 34013850063](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/34013850063) |
| N2 | [#57](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/57), `231f935` | [N2-R](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/57#issuecomment-5562021565) | [34046782925](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/34046782925) | [Superseding successor-tree run 34301118384](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/34301118384/job/102308027982), qualified below |

These approval comments record independent Codex/subagent review. They are not
formal approvals attributable to separate GitHub accounts: the authenticated
account also owns the PRs, so GitHub disallowed owner self-approval.

## N2 post-merge failure and superseding retest

The immediate N2 merge-push [run 34062447715](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/34062447715)
failed because freshly generated oracle bytes differed from the frozen fixture.
Smoke and benchmark were skipped. The cause is not established; this closeout
does not call that run successful or attribute the mismatch to a particular host
or tool.

Later supported run 34301118384 tested synthetic merge
`58ec324e9fc983f20bea9e0094eff1367dcbeff7`, whose tree
`69eca436397cfe8a2ff9cd3e2c8f2764af2d4f35` is identical to actual main merge
`d805024764a8661815be3b2c3036695195b7075f`. All 27 N2-owned documentation,
header, native source, build, test, fixture, generator, and oracle-lock paths have
identical blobs and modes to N2 merge `231f935`.

That run installed CPython 3.14.6 and PyTorch 2.13.0+cpu and passed all 31 N2
reference/gradient tests without skips, including fresh byte-identical oracle
regeneration, followed by the complete N2 gate and integrated tests, smoke, and
benchmark. An earlier intentional no-oracle isolation pass in the same job skipped
14 development-oracle tests; it is not the successful pinned-oracle evidence.
Integration accepts the later unchanged-owned-tree execution as the superseding
supported post-merge retest, while retaining the historical failure above.

## Scope preserved

- I2 proves its dense CPU-f32 ownership, parameter/gradient, and atomic-plan
  substrate; it does not implement an optimizer or model.
- T2 proves its bounded deterministic BPE and streaming contract, not an unbounded
  tokenizer or reclaimed public token identities.
- N2, A2, and L2 prove only their exact advertised operation/shape domains and
  documented explicit provider interfaces. They do not prove a composed public
  transformer, general autodiff, GPU execution, or training/resume equivalence.
- Existing error, resource, sanitizer, packaging, numerical, and capability gates
  remain unchanged. Historical integration-log entries remain chronological and
  are superseded only in status by this closeout.
