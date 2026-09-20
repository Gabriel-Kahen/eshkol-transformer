# G3-N / G3-S independent packaging and adversarial ABI review

Status: **independent contract review passed; integration ABI disposition remains required**.
This is an independent Astra/high source review under binding
[decision 5748532295](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5748532295).
It adds no production code and runs no build, numerical test, or CI.

## Authoritative seams inspected

- `include/eshkol_transformer/kernel_abi.h`, `native/kernel_abi.c`, and
  `docs/K1_KERNEL_ABI.md`: actual provider/request/call/tensor layouts,
  capability matching, discovery, generic admission and two-phase dispatch.
- N2/N3K/A2 public headers, numerical provider implementations, build scripts and
  native tests: explicit accessor symbols, operation schemas, floating controls,
  dry validation, exact manifests, no-allocation checks and baseline isolation.
- `scripts/m3-native-inputs.sh`, `scripts/build-e1b-consumer.sh` and M3 manifests:
  one future aggregate incorporates selected native objects and localizes private
  authority; predecessor owning aggregates cannot simply be linked together.

## Required conformance boundaries

1. **No invented K1 parameter field.** ABI 1.0 has no `params`, parameter schema,
   parameter bytes or user context field in requests/calls. Epsilon, sampling
   controls and RNG words must use explicitly ordered input tensor descriptors,
   or a separately accepted ABI. Mixed i64/bool operands do not change the f32
   compute request. The logical request-shape row is distinct from tensor ranks.

2. **Capabilities describe Cartesian products.** Every listed operation must be
   implemented for every listed shape alternative, dtype and device. Use exact
   `min=max` alternatives, not a bounding rectangle over reviewed examples. Give
   new providers distinct capability names; never concatenate predecessor rows
   or operation lists under one shared capability and thereby widen its meaning.

3. **Prefix compatibility is mandatory.** Frozen K1 prefixes are provider96,
   capability120, call88, request56 and tensor-view72 bytes. Accept compatible
   appended tails; reject truncated prefixes before reading later fields. Repeated
   descriptors require aligned `count/stride/bytes/base`, exact nonoverflowing
   byte span and `prefix <= element.struct_size <= stride`. Null/zero empty
   tables remain the K1 rule. Higher compatible provider minor is accepted when
   required-feature bits are known. Provider semantic version is separate from
   K1 ABI major/minor. Current K1 does not inspect reserved bytes; do not claim
   generic reserved-zero rejection or interpret them as new controls.

4. **Separate generic from provider admission.** K1 matches capability/operation
   and validates descriptor tables, byte lengths, device/layout and output data
   aliases. Provider checks exact arity and operand schema, natural data alignment,
   data-pointer spans, input-input alias rules, IDs/policies/state and finite
   numerical intermediates. Missing rows reject as unsupported; malformed operands
   on an admitted row are not evidence for broader capability. Direct provider
   invoke requires all generic and provider checks, stable storage and FP controls.
   Readable pointers and immutable call/metadata/input storage are C caller
   obligations; tests must not promise arbitrary hostile pointers are safe.

5. **Invoke cannot fail recoverably.** K1's `invoke_call` returns `void`. All
   recoverable numerical, policy, exhaustion and FP-control errors must occur in
   non-mutating validation. A dry run followed by the same ordered arithmetic is
   an existing pattern. Every output is fully overwritten on successful invoke;
   every rejected call preserves input/output/RNG bytes. No output-dependent
   arithmetic or hidden allocation is admitted. Validation must restore incoming
   complete x87/MXCSR flags and controls, including signaling-NaN paths. Unsupported
   controls reject rather than being silently repaired.

6. **Keep native and Eshkol errors distinct.** K1 has no
   `determinism-unavailable` enum. Use an existing exact native category/code and
   specify later E1/G3 mapping separately. K1 output aliases use
   `INVALID_ARGUMENT/ALIASING_OUTPUT`; provider input-alias rejection uses
   `INVALID_ARGUMENT/PROVIDER_REJECTED`. Unknown state-version and exhausted-draw
   errors need explicit rows. Native word tensors are not authenticated G3 owners.
   Binding decision permits authentic exhausted state for all no-draw operations;
   generator preflight rejects a required categorical draw before prefill commits.

7. **One explicit resolver per registry-free numerical object.** Standalone N/S
   accessors return immutable provider metadata and retain no caller storage.
   They must not export `eshkol_transformer_kernel_provider_v1`, register global
   model/tensor/RNG identities, load plugins dynamically, or install Eshkol public
   generation names. A future fixed private resolver selects complete capability
   owners and rejects duplicates. Whole-archive linking the provider must leave
   K1's provider-free baseline unchanged.

8. **Measure the artifact actually delivered.** Require exact archive member,
   defined-symbol, allowed-undefined-symbol, provider report and actual compile
   depfile closure inventories; pin copied predecessor helper sources. Inspect
   actual object exports and dependencies, not just companion manifest files.
   Exclude malloc/free, dynamic loading, Python, test adapters and canonical
   provider symbols. Match N2/N3K FP build controls explicitly; A2's older build
   flags alone do not establish the stricter new sampler contract. Inspect IR and
   machine code for forbidden FMA/binary64/fast-math widening. Exact undefined
   inventories are established on the supported toolchain, never guessed from
   host compatibility artifacts.

9. **Keep acceptance layers separate.** C/C++ header, native dispatch/reference,
   carrier integration, AOT linkage, future public Eshkol execution and supported
   exact-head CI prove different things. N/S alone cannot claim genuine no-grad
   model execution, whole-cache publication, typed RNG authentication, generated
   output lifetime or save/reload. Those remain T/M/G/R obligations.

## Independent draft findings and disposition

Reviewed [G3-N](G3_N_ABI_PROPOSAL.md) and [G3-S](G3_S_ABI_PROPOSAL.md) after the
following author corrections. No production change or executable check was used
as evidence for this review.

| Proposal | Exact reviewed boundary | Disposition |
|---|---|---|
| G3-N | `g3n.cpu-f32.serial`; six capabilities, seven operations, eleven exact operation/row pairs; embedding2/linear2/LayerNorm4/residual2/layout1/attention6 inputs, one output each; one source/object/archive and sole `et_g3n_kernel_provider_v1` export | No remaining packaging or K1-schema blocker |
| G3-S | `g3s.cpu-f32.serial`; two capabilities/operations, each exact `[1,256]`; greedy2/categorical5 inputs, token plus numeric-state outputs; one source/object/archive and sole `et_g3s_kernel_provider_v1` export | No remaining packaging or K1-schema blocker |

Resolved findings:

- G3-N originally implied reserved-byte rejection by generic K1. It now states
  provider-emitted zeros and unchanged consumer semantics explicitly, including
  ignored compatible tails. Its error rows now separate alignment, exact-byte
  mismatch and address wrap, avoiding two incompatible codes for one failure.
- G3-S originally blurred generic K1 errors and direct provider errors. It now
  states generic precedence and the exact device, shape-product, byte-length and
  address-wrap distinctions. Public E1 error names remain later transport work.
- G3-S now identifies writable-data versus metadata/control disjointness as the
  caller's existing stable-storage precondition, rather than claiming a new
  arbitrary metadata-alias detector. Tensor payload alias validation remains an
  enforced operation contract.
- G3-S replaced broad dependency families with a nine-symbol normal-object
  ceiling and concrete C11/O2/PIC/stack-protection/FP/deterministic-archive build
  requirements. Both proposals distinguish the ceiling from the exact supported
  object inventory, which still requires independent implementation review.

The final narrow follow-up reviewed G3-S's proposed 16,384-byte maximum
simultaneous normal-build provider/helper automatic storage, including compiler
spills. It forbids recursion, VLA and dynamic alloca and requires future compiler
stack-usage plus runtime high-water evidence. Excluding external libc/libm and
caller/K1 frames is explicit; this is not a total process/thread stack limit or
an executed measurement. The compact [ABI request](G3_PRIMITIVE_ABI_REQUEST.md)
matches the reviewed identities, matrices, descriptors, dependency ceilings and
G3-T hold. Its final stack-summary wording explicitly excludes and reports
external libc/libm and caller/K1 frames while counting provider/helper compiler
spills; the initially ambiguous “toolchain” exclusion was corrected.

The authors' separate cross-reviews of the opposite numerical proposal are
additional source review, not a substitute for numerical or supported-platform
tests. This review's disposition covers packaging and K1-schema conformance.

The exact metadata Cartesian products are internally consistent; no backward
row or unreviewed predecessor capability is implied. Their policy fields use
actual K1 typed operand tables. Their native states are borrowed numeric bytes,
not authenticated tensor or RNG identities. Their `void` invoke paths are
constrained to completely prevalidated, allocation-free writes and synchronous
storage ownership. G3-S's exhausted/no-draw behavior and future G3-G early
exhaustion check match binding decision 5748532295.

For source-only review reproducibility, the reviewed draft SHA-256 values were:

- G3-N: `ae7cbf18a1675e8305e494045f3af8ff1c6715b5c137081c518b1691e660a377`.
- G3-S: `a27f281bba82359995a6d13bc8061990fb9fe5b196199572427c9dd81d584f16`.
- Compact ABI request: `43b68b2c58cd8e4862339b27ef880ab78ffa6bbbcc453aa1c228660160a1c12a`.

**Disposition:** ready for root's bounded ABI decision, with no open independent
packaging/schema correction. This is not authority to install the API, advertise
VERIFIED capabilities, or begin dependent implementations before their required
contracts/implementations merge. Numerical algorithms, literal RNG vectors,
allocation/fenv/failure behavior, actual exports/undefined symbols, fresh objects,
carrier/AOT execution and supported exact-head CI remain unmeasured future gates.
G3-T's private shared-guard, liveness/pin, typed-owner and synchronous T1-borrow
seams still need their own exact contract; this review does not freeze them.
