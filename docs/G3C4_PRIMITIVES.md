# G3-C4 bounded forward primitives

Status: **implementation candidate; runtime acceptance pending**.
[Issue #115](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/115)
authorizes the [exact standalone ABI](g3/G3_C4_N_ABI_PROPOSAL.md) accepted at
`67e3cd8919a54e80448e764493e3ac53708d0a6f`. Independent review approved the implementation candidate; supported CI and
root acceptance remain pending. This provider does not depend on C4 model
construction, shared model pins, transport, persistence or sampler review.

## Contract

The sole public native accessor is `et_g3c4_kernel_provider_v1`, declared in
`include/eshkol_transformer/g3c4_primitives_abi.h`. Its immutable K1 ABI 1.0
provider identity is `g3c4.cpu-f32.serial`, version `1.0`, evidence
`G3-C4:cpu-f32-c4-forward-v1`. Seven capabilities expose eight operations and
28 exact operation/shape pairs: six embedding, eight linear, two LayerNorm,
two GELU, one residual, four head-layout and five attention pairs. The binding
ABI document defines every row, operand, layout, alias, error and arithmetic rule.

The implementation copies the accepted G3-N forward machinery and N2 exact-erf
GELU. Both attention softmax passes explicitly reject a nonfinite
score-minus-maximum before `expf`; finite exponential underflow to zero is valid.
T3/T4 layouts materialize the token/head permutation, and attention covers
multiple keys and query rows. Existing predecessor providers remain unchanged.

Complete K1 generic admission and provider validation are required before
invoke. Validation restores the complete incoming floating environment and
leaves inputs, outputs and metadata unchanged. The same calculation performs
nonwriting validation and publication. The caller must preserve descriptors,
input bytes and supported floating controls until synchronous invoke finishes.

The provider owns no storage, graph, cache or RNG, retains no carrier borrow,
and adds no Eshkol public operation, canonical resolver, backward operation or
model policy. The primitive accepts any finite positive LayerNorm epsilon and
positions in `[0,16777215]`; downstream C4 policy restricts these separately.

## Build and verification

`make test-g3c4` builds the audited canonical prerequisites and runs
`scripts/test-g3c4.sh`. The normal archive is
`build/g3c4/libeshkol_transformer_g3c4.a`, with one object and one global export.
Its production source closure is the provider C source, its ABI header and the
unchanged K1 header. Link it explicitly before K1 and `-lm`.

The full and acceptance-predecessor test tiers append the gate immediately
after G3-N. The existing `g3n-forward` suite runs both gates, retains its
75-minute budget and requires both leak-check settings. Its audited producer
order is K1, A2, I1, I2, N2, N3K, G3-N, G3-C4. This base has 19 shared suites
and 28 commands; integration with the pending two-command PR #112 produces
19 suites and 30 commands. Root owns that union and supported CI dispatch.

Local verification uses the unchanged pinned Eshkol commit
`90cbd7130f47b8184bcc77b8d5c1b0026da980de` with an already-built, provenance-
verified compiler. CachyOS/Clang/LLVM 22.1.6 results are compatibility evidence;
they do not establish the supported Ubuntu 22.04/LLVM 21.1.8 result. No compiler
repin or compiler-source change is part of this candidate.

## Reachable numerical boundaries

A validated attention probability lies in `[0,1]`: the largest exponential
weight is one, every weight is nonnegative, and serial positive accumulation
makes the denominator at least each individual weight. Dividing and rounding
cannot produce a probability greater than one. Multiplying by finite f32 V
therefore cannot overflow. The copied weighted-product guard remains enforced
by helper provenance. Root accepted this independently reviewed unreachability argument as an issue
#115 proof refinement; the numerical ABI and production guard remain unchanged.

The increasing-key output reduction can overflow because rounded probabilities
need not sum to exactly one. A local four-key witness uses Q `[1,0]`, K first
components `0`, `-0x1.e2993p+1`, `-0x1.db81d8p+2`, `-0x1.1a793ep+3`, zero
second components, and all four first V components `FLT_MAX`. Query position3,
key positions0..3 and keep bytes1 admit all keys. Every weighted product stays
finite; the serial sum overflows. Actual K1 dispatch rejects it with
`INVALID_ARGUMENT/PROVIDER_REJECTED`. The focused suite proves individually finite products and no-write failure
atomicity for this witness.

## Acceptance evidence

The first complete local `scripts/test-g3c4.sh` run passed with
`G3C4_ASAN_DETECT_LEAKS=1` on 2026-09-22. It records 16,194 numerical checks,
100,116 adversarial checks, 60,145 I1/I2 checks (60,146 with the instrumented live-
count assertion), and 20 detected compiled semantic mutants. Normal test runs
are repeated with identical stdout; sanitizer results agree. The gate also
passed two fresh identical native builds and private Eshkol AOT callers,
C11/C++17 clients, exact inventories/closure, forbidden-link checks, immutable
reports, frozen predecessor sources, helper provenance, and ASan/UBSan/LSan.
All 100 CI Python tests passed, as did the audited prerequisite verification
and 19-suite/28-command topology check. Full CI was not dispatched.

The provider report has 5,679 bytes excluding NUL, SHA-256
`b4b96727bc3a44ffeeaa1a0855a7dc51b742fc7820b98a6e3d845f4d8f6fba01`.
The normal local object has one global export and ten undefined symbols:
`__stack_chk_fail`, `erff`, `et_kernel_error_clear`, `expf`, `fegetenv`,
`fesetenv`, `memset`, `snprintf`, `sqrtf`, `strcmp`. The accepted ceiling also
permits `memcpy`. Sanitizer inventories are separate. These exact local
inventories still require confirmation on the supported compiler lane.

Independent Sol/high reviewer `c4n_review` approved with no remaining findings.
The reviewer separately ran all three native tests and 20 mutants, checked
source/envelope/arithmetic/metadata, and verified deterministic packaging,
reports, header clients and negative links. Root ran the complete AOT/sanitizer
suite. Review fixes strengthened test fixtures, arithmetic-order witnesses,
descriptor/fenv coverage and real-carrier admission; they required no production
change. The weighted-product proof refinement is recorded in
[issue #115 comment 5779652587](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/115#issuecomment-5779652587).

Integration base is `19f404cf21632944e1f2d5d4959e1240f9a79f5f`.
Focused commits carry the accepted ABI (`d2668fb`), provider/header/build
(`6ad4e54`), proof refinement (`59de933`), and tests/package (`5a60775`).
The full local gate log is `/tmp/g3c4-focused-gate.log`, SHA-256
`2bca0480832e002fcab6afd4f2e3fcb53f93dc5ddaffc3bb9169b025e4bb8ae9`.
Root retains current-main integration, compiler evidence freeze, supported CI,
and merged-head verification. C4 model construction, shared-storage admission,
generation and continuation/save-load remain separate workstreams.
