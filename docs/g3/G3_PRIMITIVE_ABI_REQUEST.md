# G3-N / G3-S: exact ABI decision request

**Accepted for implementation** by [decision 5751933660](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5751933660)
at `d2b0858a39d99954396ade51b019f6094a00a0f3`, reviewed request SHA-256
`43b68b2c58cd8e4862339b27ef880ab78ffa6bbbcc453aa1c228660160a1c12a`.
The [binding C2 design](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5748532295)
is unchanged. This historical request records the accepted primitive contracts;
it does not install headers, claim numerical evidence or
authorize dependent implementation against unmerged APIs.

| Contract | G3-N | G3-S |
|---|---|---|
| Exact proposal | [Forward rows](G3_N_ABI_PROPOSAL.md) | [Sampler](G3_S_ABI_PROPOSAL.md) |
| Header | `g3n_primitives_abi.h` | `g3s_sampling_abi.h` |
| Semantic version macros | `ET_G3N_PRIMITIVES_ABI_MAJOR/MINOR` = 1/0 | `ET_G3S_SAMPLING_ABI_MAJOR/MINOR` = 1/0 |
| Sole native export, arity0 | `const et_kernel_provider_v1 *et_g3n_kernel_provider_v1(void)` | `const et_kernel_provider_v1 *et_g3s_kernel_provider_v1(void)` |
| Provider/implementation | `g3n.cpu-f32.serial` | `g3s.cpu-f32.serial` |
| Version / evidence ID | `1.0` / `G3-N:cpu-f32-c2-forward-v1` | `1.0` / `G3-S:cpu-f32-v256-v1` |
| Matrix | 6 capabilities / 7 operations / 11 exact pairs | 2 capabilities / 2 operations / 2 exact pairs |
| Production source / archive member | `native/g3n_primitives_provider.c` / `g3n_primitives_provider.o` | `native/g3s_sampling_provider.c` / `g3s_sampling_provider.o` |
| Archive | `build/g3n/libeshkol_transformer_g3n.a` | `build/g3s/libeshkol_transformer_g3s.a` |

Both return immutable K1 ABI1.0 providers, required features0, f32/cpu compute
and deterministic1; request deterministic0 still runs the same checked path.
No K1 layout/enum change, new public Eshkol name, canonical resolver, registry,
owned carrier, backward operation, source dependency on another provider or
serialization is added. VERIFIED metadata is contingent on implementation proof.

G3-N exact operation rows:

- `g3n.embedding.forward`: [1,1,256,4], [1,1,2,4]; 2 inputs → 1 output.
- `g3n.linear.forward-no-bias`: [1,1,4,4], [1,1,4,8], [1,1,8,4],
  [1,1,4,256]; 2 → 1.
- `g3n.layer-norm.forward`: [1,1,4]; x/gamma/beta/scalar epsilon → y.
- `g3n.residual.forward`: [1,1,4]; 2 → 1.
- `g3n.heads.merge.forward` and `g3n.heads.split.forward`: [1,1,2,2]; 1 → 1.
- `g3n.causal-attention.forward`: [1,2,2,1,1,2]; Q/K/V/query-position/
  key-position/bool-mask → y. True uncached T1; no new cache rows.

The N proposal gives each exact capability name and ordered operand shape. Numeric
epsilon admits positive finite f32 and positions admit A2's [0,16777215]; G3-T/M
enforces the fixed model epsilon and positions0/1. Input spans are disjoint except
equal complete residual inputs and equal complete query/key position views.
No partial alias or Q/K/V overlap is admitted. Existing T1 GELU routes to N2.

G3-S schemas both request [1,256]:

- `g3s.greedy` / `g3s.greedy.forward`: f32[1,256] logits, i64[4] RNG →
  i64[1] token, unchanged i64[4] RNG.
- `g3s.categorical` / `g3s.categorical.forward`: logits, rank0 f32 temperature,
  rank0 i64 k, rank0 f32 p, i64[4] RNG → token, successor RNG.

Policy scalars are bit-materialized from accepted config; K1 has no auxiliary
parameter field. Native RNG words are data, not authenticated G3 ownership.
All inputs are pairwise disjoint; outputs are disjoint from inputs/each other.
The accepted arithmetic, tag/lane/high24/one-block and exhausted/no-draw rules
are unchanged. Categorical native exhaustion is INVALID_ARGUMENT/PROVIDER_REJECTED;
G3-G separately rejects an authenticated required draw as public invalid-state
before prefill. Greedy preserves the exhausted sentinel.

Both proposals fully specify generic K1 versus provider errors, prefix/tail
compatibility, exact descriptor counts, span/alignment checks, x87/MXCSR admission,
complete validation fenv restoration, and allocation-free dry-validation/void-invoke
publication. No output mutates on a recoverable failure. Invoke requires stable
borrowed inputs/metadata/controls and has no recoverable branch after publication.

Each normal archive has one source/member/export and a three-project-file source
closure (C source, its header, K1 header). Both dependency ceilings are exactly
`__stack_chk_fail`, `et_kernel_error_clear`, `expf`, `fegetenv`, `fesetenv`,
`memcpy`, `memset`, `snprintf`, `strcmp`; N additionally allows `sqrtf`.
The actual supported-compiler subset must be measured and frozen during review,
not invented now. S additionally proposes 16,384 bytes maximum simultaneous
provider/helper automatic storage; no recursion, VLA or alloca. External libc/libm
and caller/K1 frames are excluded and reported; provider/helper compiler spills
remain counted.

## Disposition requested (now accepted)

Ratify (1) exact identities/capability matrices/descriptor arities; (2) primitive
scalar domains and alias exceptions; (3) native error/fenv/lifetime/publication
rules; and (4) one-source/export inventories, dependency ceilings and sampler
automatic-storage bound. These are the outstanding ABI deltas. Do not reopen
the accepted C2 profile, six additions, EOS, sampler algorithm or token transaction.

Independent [packaging/adversarial review](G3_ABI_REVIEW.md) has no open blocker.
Authors also cross-reviewed the opposite numerical proposal: no remaining
numerical/alias/fenv/schema conflict was found. Required future acceptance includes
independent mathematical/literal-vector and mutation checks, all exact-row and
descriptor/alias/fenv/exhaustion negatives, input/output preservation, I1/I2 borrows,
allocation-disabled execution, ASan/UBSan/LSan, exact archives/closures/symbols,
fresh private AOT, supported exact-head CI and proportional merged-head tests.
These are inventories of required tests, not executed results.

[G3-T guard/pin/decoder/RNG seams](G3_T_SEAM_PLAN.md) remain a separate contract
hold. No full builds, new numerical runs or CI were performed in this follow-up;
the earlier compiled reachability evidence and its limitations are unchanged.
