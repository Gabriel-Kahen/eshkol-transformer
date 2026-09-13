# Wave 2 primitive acceptance

Integration closeout, 2026-09-09. A2, T2, L2, I2, and N2 are accepted and
complete within their documented bounded contracts. This supersedes their earlier
active/proposed/review status, not their capability or lifetime limitations.
D2 has a separate acceptance record. The O2 addendum below records its acceptance
on 2026-09-10. C2 and its K2 prerequisite remain unfinished; the complete model and
first-release training gates remain separate. This does not mark Wave 2 complete.

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

## O2 merge and bounded post-merge retest

O2 is accepted and complete within [its bounded contract](O2_OPTIMIZER.md).
[PR #58](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/58) merged as
`884a0744d5adc21f95577c1f8d6904c842a019cf` after
[independent O2-R approval](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/58#issuecomment-5627783349),
successful Ubuntu 22.04.5 / LLVM-Clang 21.1.8
[blocking CI 34533652814](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/34533652814),
and [exhaustive acceptance 34533681600](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/34533681600).
Reviewed head `1a22fbd95e0644373d352b7f9f74715c12c2e836`, synthetic merge
`1ac1815eecfe7406f48efac36283f02ce0a1dd00`, and the actual merge have identical tree
`b84ff6b0c1f828600ab21f15483ece1f748cf56a`. Whole-tree comparisons were empty;
the dedicated O2 path-manifest SHA-256 was
`60769957a00f9b26ceaabf0268a661fc472a716a7294f959b1327489c5857e30` for all three.

The [merged-main retest](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/46#issuecomment-5628426969)
ran in a clean detached checkout of that actual merge using explicit non-login
`/usr/bin/bash`. It passed:

- `make test-ci-topology`: eight suites cover all 21 full-test commands exactly once.
- `python3 -m unittest discover -v -s tests/ci -p 'test_*.py'`: 17/17.
- `python3 -m unittest -v tests.q0.test_python_isolation`: 2/2.
- `make test-o2` with explicit compatibility mode, LLVM/Clang paths,
  `P1_LSAN=1`, `I2_ASAN_DETECT_LEAKS=1`, `O2_ASAN_DETECT_LEAKS=1`, and the pinned
  O2 oracle interpreter: rebuilt shared prerequisites, native optimizer, exactly
  6,201 adversarial checks, config/state AOT, PyTorch 2.13.0+cpu reference parity,
  exact 53-global successor and authority isolation, ASan/UBSan/LSan, and production
  isolation. The linked evidence preserves the complete absolute command.
- `/usr/bin/bash scripts/smoke.sh`: exactly `eshkol-transformer-smoke:v1`.

The local retest was on unsupported CachyOS / LLVM-Clang 22.1.6. Supported-host
claims rely on the identical-tree CI runs above, not on that compatibility lane.
This bounded post-merge retest does not claim a new full local repository test run.
Final HEAD/tree and clean worktree were unchanged.

O2 defines no C2 bytes, token-count schedule, trainer cadence, generic release
authority, or live-optimizer destroy operation. Live optimizer receivers and their
moments remain process-local until exit; snapshot release frees snapshot-owned
moment storage, not live optimizer storage. Identity tombstones remain cumulative
with linear lookup. C2 consumes the merged logical projection and scoped synchronous
borrow/release contracts; TR3 retains cadence, token counters/schedules and
mid-accumulation checkpoint policy. No full training trajectory or generation
equivalence is claimed by this acceptance.

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
