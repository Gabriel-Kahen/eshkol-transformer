# G3-N bounded forward primitives

Status: **complete for the exact bounded forward provider**.
Independent approval, supported CI, merge and root focused retest are recorded
in [integration provenance](#integration-provenance); G3-S/T and generation remain
unaccepted.
Supported exact-head run results are tracked in [issue #93](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/93).
This implements issue #93 and the exact ABI accepted in
[decision 5751933660](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5751933660),
merged in contract PR #92 at `4353bd2f14ae4fb50c877c27fc73b8edef8e2f30`.
The binding [G3-N ABI contract](g3/G3_N_ABI_PROPOSAL.md) remains unchanged.

## Boundary

`include/eshkol_transformer/g3n_primitives_abi.h` declares ABI 1.0 and the sole
`et_g3n_kernel_provider_v1` accessor. The immutable `g3n.cpu-f32.serial` descriptor
has six capabilities, seven forward operations and eleven exact operation/row
pairs, with evidence identity `G3-N:cpu-f32-c2-forward-v1`. It has no canonical
K1 resolver, registry, allocation, retained borrow, carrier, public Eshkol API,
backward operation, RNG, graph or cache. Explicitly link
`build/g3n/libeshkol_transformer_g3n.a` before K1 and `-lm`.

The provider admits both T1 embeddings, four bias-free linears, LayerNorm,
residual, split/merge and no-cache attention. All operands, aliases, error
categories and numerical order follow the exact contract. Epsilon accepts any
positive finite f32; positions accept A2's `[0,16777215]` domain. Downstream model
policy must restrict these separately. Existing N2 owns `gelu.forward [1,1,8]`.

Complete K1 generic admission plus provider validation is required before invoke.
Validation restores the complete incoming floating environment and changes no
input, output or metadata. Inputs, descriptors and supported floating controls
must remain stable until invoke returns. Output may be observed only afterward.
A single calculation serves dry validation and invoke; its recoverable numerical
failures have already been excluded before publication. Unsupported x87/MXCSR
controls are rejected before floating input reads, including signaling NaNs.

T1 head layout has identity flat order with different ranks. Tests prove exact
shape/direction admission and bit preservation, not a nontrivial permutation.
Single-key attention still evaluates QK and rejects score overflow. A masked or
future key produces exact positive zero without score evaluation, while all
floating inputs must remain finite. Only complete residual skip/branch aliases
and complete attention query/key-position aliases are admitted.

## Source provenance and packaging

All copied helpers come from the accepted tree
`4353bd2f14ae4fb50c877c27fc73b8edef8e2f30`:

| Source | Helpers |
|---|---|
| `native/n3k_primitives_provider.c` | embedding, bias-free linear, residual addition and token/head index mapping |
| `native/n2_primitives_provider.c` | two-pass biased LayerNorm statistics, epsilon completion and affine forward |
| `native/a2_attention_provider.c` | score, causal/mask admission, maximum-subtracted softmax and increasing-key forward accumulation |

Sixteen standalone helper bodies are compared mechanically to frozen predecessor
sources (only A2's internal `admitted` name changes). Residual/layout branches
retain their numerical/index statements. A2 forward preflight is adapted with an
optional output pointer, explicit weighted-product finiteness check, and empty-row
positive-zero publication. Frozen predecessor source/header hashes and old
provider reports remain checked. The implementation does not link predecessor
private functions.

The production input closure is exactly the G3-N source/header and unchanged K1
header. The deterministic archive has one `g3n_primitives_provider.o` member and
one defined global. Local LLVM 22 measures nine undefined symbols:
`__stack_chk_fail`, `et_kernel_error_clear`, `expf`, `fegetenv`, `fesetenv`, `memset`,
`snprintf`, `sqrtf`, `strcmp`. The accepted ceiling additionally permits `memcpy`;
the supported LLVM 21.1.8 G3-N job 106598514947 confirmed the same exact subset
at the final reviewed candidate. Its complete gate is authenticated as a reused
success in supported run 35681252651 attempt 2.
Sanitizer object inventories are separate and never shipped.

The immutable K1 report is 4,979 bytes excluding NUL, SHA-256
`b342924ddfcecb20f552c7dcdb2f9391ccc910ccab16e0d57df3f40a35efef83`.
Its independent reconstruction checks every capability, operation, row and field;
two emissions must match. K1's provider-free baseline and explicitly selected
N2/N3K/A2 reports remain unchanged even with the G3-N archive force-linked.

## Verification

Run `make test-g3n` for the focused gate. It verifies the pinned compiler, builds
only the canonical inputs read by this gate and runs `scripts/test-g3n.sh`.
The normal build includes the archive, and normal/acceptance test lists include
the focused gate. The `g3n-forward` supported CI suite adds to all existing suites;
its prerequisites are K1, A2, I1, I2, N2, N3K and G3-N. Main's accepted L3S
addition brings its baseline to 18 suites/26 commands; the G3-N union is 19
suites/27 commands. Future integration must union with actual main coverage,
not replace concurrent additions.

The local compatibility gate measures 3,712 numerical/metadata checks, 36,245
adversarial checks, and 26,493 carrier/allocation checks (26,494 with the additional
instrumented live-count assertion), plus fifteen compiled mutants. The gate includes:

- Independent long-double mathematical and literal f32 references for all eleven
  pairs; all embedding indices, matrix channels/orientation/reduction, LayerNorm
  reachable overflow stages, signed zeros, rank-sensitive layouts and both
  attention heads/mask states/position relations and score-overflow rejection.
- Fifteen compiled mutants: weight transpose, reversed linear reduction,
  linear FMA, unbiased variance, reversed LayerNorm reduction, affine FMA, fixed
  epsilon, omitted attention score, ignored mask, reversed position predicate,
  ignored unselected embedding nonfinite input, erased residual zero sign, raw
  embedding-row reordering, nonzero dimension reserved bytes and provider name.
- Per-operation malformed prefixes/tables/shapes/dtypes/devices/spans/alignment,
  exact/partial input and output overlaps, scalar/value errors, whole-fixture
  failure atomicity, independent x87/MXCSR flags and unsupported controls,
  capture/restore fail injection and future-tail/reserved-byte preservation.
- Real I1/I2 scoped borrowed dispatch, active-borrow mutation/release exclusions,
  raw-view parity, allocation/free-disabled accessor/validate/invoke/dispatch,
  repeated carrier cleanup and immutable metadata. ASan/UBSan/LSan instrument
  provider, K1 and carrier paths together.
- Exact source/depfile/archive/export/undefined inventories, C11/C++17 clients,
  private/canonical-symbol link negatives, production dependency isolation,
  unchanged predecessors, two fresh deterministic objects/archives and two fresh
  pinned-Eshkol private AOT binaries/runs exercising all rows and negatives.

Local CachyOS/LLVM 22 results are compatibility evidence only. LSan cannot run
under the sandbox ptrace environment; the same binaries must run with permitted
unsandboxed execution and leak detection enabled. Supported Ubuntu 22.04 /
Clang-LLVM 21.1.8 full CI, independent review, merge and proportional merged
retest passed with the distinct provenance below. No cross-libm bit guarantee,
performance, model execution, generation completion, wider context or gradients
are claimed. G3-T and G3-M composition remain downstream.


## Integration provenance

[Independent G3-N-R approval 5771670331](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/97#issuecomment-5771670331)
accepted head `15014b81bbfb32dece08b204d8bf52173a6a5c32`, tree
`90e83c02d13eb7851b0d0a5a6a5eabb64bf37983`. The separate numerical, native/fenv/
ownership and packaging reviewers reproduced the focused gate and authenticated
the complete supported result. No actionable findings remained.

[Supported run 35681252651, attempt 2](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35681252651/attempts/2)
passed all 19 component suites / 27 commands, topology, evidence and final
aggregation on Ubuntu 22.04 / Clang-LLVM 21.1.8. `CI_EVIDENCE_V1` records that
run/attempt, every suite and checkout `dfe5675a7c4333042d6c1520f976dc7633ca8e2a`,
whose tree equals the reviewed candidate above.

Attempt 1 failed only P1 parameter-state job 106598514896: `scripts/test-p1.sh`
returned timeout status 124 and Make returned 2. The last visible stage was
`identity-sanitize`; later output was redirected and not retained, so the exact
timed-out subcommand and root cause remain unknown. This is not a proved
infrastructure-flake claim. All other 18 components passed.
[Root disposition 5770842334](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/97#issuecomment-5770842334)
authorized exactly one failed-job-only retry of the unchanged SHA. New P1 job
106618260262 passed in 44m46s with `P1_LSAN=1`; dependent evidence 106626773388 and
final aggregation 106626802030 passed. The 19 cloned success records reuse the
original topology and 18 component executions; they are not fresh executions.
Independent review authenticated their original timestamps, steps, runner/head
and checkout logs. No source, timeout, assertion, leak gate or workflow changed.
The original failure remains recorded. Older run 35679799935 was cancelled while
incomplete as superseded; its successful G3-N component is historical partial
evidence, not full acceptance.

[PR #97](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/97) merged as
`ca3880f6e5da4850bafbf349a846de7829203e51`, tree
`68e6063612eaba06843d7f37d0d4184dffd0e085`. This differs from the supported
tested tree only in seven accepted L3S/E3 Markdown files. The trees are distinct;
the merged tree is not relabeled as the exact supported-tested tree.

[Root merged-head verification 5771719405](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/93#issuecomment-5771719405)
held that clean merge fixed and ran `scripts/ci-build-prerequisites.sh g3n-forward`,
`scripts/test-g3n.sh`, `python3 tests/ci/topology.py` and all 99 CI checker tests.
Every command exited 0. The canonical K1/A2/I1/I2/N2/N3K/G3-N prerequisites were
rebuilt; all focused counts above, all 15 mutants, sixteen helper-provenance
comparisons, two fresh deterministic private AOT builds/runs and ASan/UBSan/LSan
with `G3N_ASAN_DETECT_LEAKS=1` passed. Topology remained 19 suites / 27 commands.
This separate retest used the read-only pinned Eshkol source/compiler
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`, explicit non-login `/usr/bin/bash`,
and CachyOS / LLVM 22.1.6 compatibility mode; it does not substitute for supported
host evidence or claim performance.

Subsequent shared-contract documentation PR #102 advanced main to
`2ac0b5bc5a385e79700f46bf899bab841f33d1cb` through four Markdown files only.
The merge's main run 35691068144 was cancelled; the latest
[main run 35691356855](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35691356855)
was still in progress at this closeout's preparation. Neither is claimed as a
completed supported-main result. This documentation adds no runtime/test/CI
change or new numerical campaign. G3-S/T, G3-M composition and public generation
remain separate downstream work.
