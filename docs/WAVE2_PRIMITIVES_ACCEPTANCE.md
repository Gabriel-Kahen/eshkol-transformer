# Wave 2 primitive acceptance

Initial primitive closeout, 2026-09-09; bounded Wave 2 component closeout,
2026-09-18. A2, T2, L2, I2, and N2 are accepted and
complete within their documented bounded contracts. This supersedes their earlier
active/proposed/review status, not their capability or lifetime limitations.
D2 has a separate acceptance record. The O2 addendum below records its acceptance
on 2026-09-10. [K2 was accepted on 2026-09-17](K2_CAPABILITY_FACADE.md#verification).
C2 was subsequently accepted and merged as recorded below, so the bounded Wave 2
component work is complete. Wave 3 work remains paused; this closeout starts no
Wave 3 tasks. Complete-model and first-release training gates remain separate.

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
Smoke and benchmark were skipped. This closeout does not call that run successful,
and no retained per-word diagnostic supports retroactively assigning its mismatch.
Later controlled evidence diagnosed the same class of fresh-oracle failure on
recent Intel runners as an MKL cross-vendor dispatch difference in 18 linear tensor
words. The accepted reference-only `MKL_CBWR=COMPATIBLE` wrapper pin reproduced
the unchanged frozen fixture on all 12 controlled runs without changing runtime,
generator, lock, fixture, or tolerance.

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

## C2, CI-E, and Wave 2 closeout

C2 is accepted and complete for detached component-state continuation and atomic
file publication. PR #79 merged as `cbd0929`, and PR #77 merged to main as
`913cdf4097db09d6c33769e9b0968c01ce0e1f55`; both share reviewed and tested tree
`512a3355cea79583d73b49f690a46478fb1c772c`. Supported run 35393213200 passed all
15 suites, all 23 top-level commands, canonical smoke, and benchmark. Independent
authoritative [review](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/77#issuecomment-5733708650)
approved the exact tree. Acceptance run 35401088338 verified
the same-tree evidence, skipped the full matrix, and completed successfully in 27
seconds. The two final joint operational runs measured 521,500 and 521,576 KiB,
below the unchanged 524,288 KiB ceiling.

The merged-tree integration owner also passed the C2 format and core-only groups,
69 CI unit tests, two Python-isolation tests, 15-suite/23-command topology, rebuilt
compile smoke, and exact `eshkol-transformer-smoke:v1` under explicit non-login
Bash on an unsupported CachyOS/LLVM 22.1.6 compatibility host. This was a bounded
retest, not a full local repository, public-AOT, or operational rerun; supported
full evidence remains the run above. C2 does not claim TR3 joint live restoration
or full-trajectory equivalence, and it does not claim G3 generation equivalence.

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
