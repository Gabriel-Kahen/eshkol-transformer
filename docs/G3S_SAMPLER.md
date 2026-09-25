# G3-S bounded native sampler

G3-S implements the [accepted ABI1.0 contract](g3/G3_S_ABI_PROPOSAL.md) for
issue [#94](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/94).
The exact PR #96 head has green supported CI and independent source approval.
The later Wave 3 union still needs fresh supported gates and integration. This
provider does not establish public generation or G3 completion.

## API and execution contract

The sole native export is `et_g3s_kernel_provider_v1`, declared in
`include/eshkol_transformer/g3s_sampling_abi.h`. The registry-free provider
`g3s.cpu-f32.serial` advertises exactly `g3s.greedy` and `g3s.categorical`, each
with its single `.forward` operation and exact f32 CPU request `[1,256]`.
Both ordinary deterministic request values execute the same path. K1 ABI,
predecessor providers and installed Eshkol exports are unchanged.

Greedy consumes logits f32[1,256] and numeric RNG i64[4]; categorical consumes,
in order, logits, rank-zero f32 temperature, rank-zero i64 top-k, rank-zero f32
top-p, and numeric RNG. Both return selected-token i64[1] and RNG i64[4].
All input payloads are disjoint; outputs are disjoint from every input and each
other. Caller metadata and borrowed inputs remain stable through synchronous
dispatch; writable outputs must also be disjoint from metadata and error storage.
The caller error record is intentionally written and excluded from output
preservation. The accepted contract retains exact generic-K1 versus
direct-provider error precedence.

Greedy chooses the lowest ID attaining the numeric maximum, including signed-zero
ties, and copies RNG unchanged even when exhausted. Categorical follows the
specified binary32 maximum subtraction, temperature division, `expf`, ID-order
normalization, probability/ID insertion sort, top-k renormalization, shortest
top-p prefix and strict-CDF endpoint order. Domain-tagged Philox4x32-10 uses lane0
high24 bits, consumes exactly one block, permits the final consumable block to
produce the exhausted sentinel, and never draws from that sentinel.

Validation checks structural schemas, disjointness and x87/MXCSR controls before
floating payload classification, rejects exhausted required draws before numerical
work, then dry-runs the complete computation into automatic scratch. Validation
restores the observed incoming fenv; failed restoration returns INTERNAL and makes
no preservation claim. Invoke recomputes the candidate
and complete successor before either output store. Stable inputs and controls are
a K1 caller obligation; publication is logical under serialized calls, not a
concurrent hardware-store transaction.

## Verification and resource evidence

Run `make test-g3s` after configuring the pinned compiler. The focused gate includes
independent literal-order and Decimal80 references, production mutations,
malformed descriptors/shapes/dtypes/devices/layouts/spans/aliases, fenv controls,
signaling-NaN subprocesses, byte preservation, real scoped I1/I2 borrowed views,
allocation-disabled dispatch, ASan/UBSan/LSan, and actual private Eshkol AOT.
[Numeric evidence](../tests/g3s/NUMERICAL.md) and
[packaging evidence](../tests/g3s/PACKAGING.md) define the individual checks.
Generated full-vocabulary fixtures remain temporary development artifacts.

The normal archive `build/g3s/libeshkol_transformer_g3s.a` contains only
`g3s_sampling_provider.o`. Exact manifests test its one export, source and header
closure, provider report, predecessor hashes, and normal undefined-symbol subset.
The accepted ceiling remains nine external symbols; observed local LLVM22 uses
seven: `__stack_chk_fail`, `et_kernel_error_clear`, `expf`, `fegetenv`, `fesetenv`,
`snprintf`, `strcmp`. Supported LLVM21.1.8 must independently confirm that subset.
Sanitized dependencies are separately inventoried and never shipped canonically.

Local compatibility verification passes 59,028 numerical checks across 48 fixtures,
23/23 independently compiled numerical mutation rejections, and 37,796 adversarial
checks with zero allocation attempts during disabled dispatch. Two fresh private
Eshkol executables and their outputs are byte-identical. The initial sandboxed
LSan attempt failed because of ptrace; leak-enabled verification is recorded in
the integration log separately.

Compiler stack-usage includes spills and statically bounds every simultaneous
provider/helper path by conservatively summing the acyclic emitted frames plus
return/red-zone allowances. The measured local compiler bound is 5,968 bytes; painted runtime high-water is
4,504 bytes. The accepted maximum is 16,384 bytes. Caller/K1 and
external libc/libm frames are excluded from this compiler bound. A separate painted
normal-runtime stack measurement includes the test callback and external library
frames and also enforces 16,384 bytes. No recursion, VLA, `alloca`, allocator or
`qsort` occurs in the provider. Sanitizer frames have no normal-build bound claim.

The dedicated 75-minute `g3s-sampling` CI suite adds the focused gate to the actual
main coverage union. Canonical build, serial local test and predecessor test entry
points include the sampler while retaining every predecessor command. At the
contract base4353bd2, this is 19 suites /26 commands. Integration must preserve
independently merged L3S/G3-N additions when refreshing main.

## Limits and downstream obligations

Local CachyOS/Clang22 runs are compatibility evidence only. Exact `expf` bits and
determinism claims are scoped to supported Ubuntu22.04/LLVM21.1.8 and its libm.
No performance, accelerator, mixed-precision, gradient or wider-shape claim is
made. Real I2 carrier tests account separately for accepted process-lifetime
identity tombstones; G3-S owns no carrier or retained identity.

The numeric RNG is data, not authenticated ownership or a serialization format.
G3-T owns authentic transport and lifetime; G3-G owns early required-draw admission
before initial prefill and the joint token/cache/RNG commit. Candidate sampler
success cannot establish model/cache rollback, generation completion, public
zero-budget behavior or save/reload equivalence. G3-N is independent; no unmerged
L3S or downstream G3-T/M/G/R implementation is required or included.
