# G3-N bounded forward primitives

Status: **review; independent root acceptance pending**.
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
the supported LLVM 21.1.8 job 106594118935 confirmed the same exact subset
at candidate `aca9c4a37333c2e52de7ed4f197fe0259584d1a0`. The full union
candidate still requires complete supported CI before acceptance.
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
Clang-LLVM 21.1.8 full CI, independent root review, merge and proportional merged
retest remain separate acceptance requirements. No cross-libm bit guarantee,
performance, model execution, generation completion, wider context or gradients
are claimed. G3-T/M composition remains downstream.
