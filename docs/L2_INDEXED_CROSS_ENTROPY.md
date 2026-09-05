# L2 fused indexed token cross-entropy

Status: **review**. This document fixes the proposed L2 native ABI and numerical
contract for review. It does not claim acceptance, merge, supported-host evidence,
an A0 model graph, or an owned Eshkol tensor implementation.

## Scope and dependency boundary

L2 implements the existing K1 capability `kernel.indexed-cross-entropy` as one
explicitly discovered CPU binary32 provider. It adds no A0 public name or arity and
no persistent format. It does not implement masking, a scalar reduction, target
gradients, tensor allocation, tensor ownership, device transfer, or reverse-mode
graph integration.

K1 supplies borrowed tensor views and two-phase dispatch, not storage or an Eshkol
autodiff node. I1 supplies exact CPU i64 storage only. The pinned Eshkol runtime was
measured to expose neither an accepted physical f32/i64 carrier nor a usable direct
gradient through the required operation. L2 therefore remains carrier-neutral.
Issue #49 tracks I2, the coordinated shared Wave-2 storage substrate. L2 can consume
accepted I2 dense borrowed K1 views without retaining or reinterpreting them, but
the private L2 AOT proof context is never a production carrier.

## Version and discovery

The independent L2 ABI version is 1.0. The C header is
`include/eshkol_transformer/indexed_cross_entropy.h`. Its only exported functions
are:

```c
int32_t et_l2_indexed_cross_entropy_abi_major_v1(void);
int32_t et_l2_indexed_cross_entropy_abi_minor_v1(void);
const et_kernel_provider_v1 *et_l2_indexed_cross_entropy_provider_v1(void);
```

The provider descriptor is immutable for process lifetime and uses K1 ABI 1.0,
provider name `eshkol-transformer-l2-cpu`, version `1.0.0`, and evidence ID
`L2:cpu-f32-indexed-cross-entropy-v1`. A caller explicitly returns this descriptor
from its K1 resolver. L2 does not define `eshkol_transformer_kernel_provider_v1`,
load a library, search a path, inspect an environment variable, register globally,
or alter K1's provider-free baseline.

The one verified entry is:

| Field | Exact value |
|---|---|
| capability | `kernel.indexed-cross-entropy` |
| operations | `indexed-cross-entropy.backward`, `indexed-cross-entropy.forward` |
| compute dtype | `f32` |
| device | `cpu` |
| deterministic | true |
| request shape | rank 3; every extent in `[1,1664510]` |

The per-axis maximum is an ABI-1.0 admission bound. It conservatively keeps a cubic
binary32 byte product within a 64-bit `size_t`; K1 still validates the exact actual
byte products. L2 advertises no `tensor.f32`, `tensor.i64`,
`tensor.contiguous`, or `autodiff.reverse` capability. K1 currently admits one
provider callback pair per runtime; I2/provider aggregation must compose accessors
explicitly rather than inventing a canonical provider symbol.

`make build` creates `build/l2/libeshkol_transformer_l2.a` with exactly one archive
member, `indexed_cross_entropy.o`. Native consumers compile against the L2 and K1
headers and link the L2 archive, K1 archive, and the platform math library. L2 ships
no module or facade under `lib/` or `src/`, and it is not part of the Wave-1 E1B
aggregate. The test gate constructs a temporary archive containing the private
Eshkol bridge, L2 object, and K1 object solely for AOT evidence; that archive and its
symbols are never installed or accepted as provider/carrier composition.

## Tensor schemas and ownership

Let `N`, `T`, and `V` be the request shape. Every tensor is CPU, dense row-major,
zero-offset, naturally aligned, and has the exact K1 byte span.

| Operation | Borrowed inputs in exact order | Caller-owned output |
|---|---|---|
| `indexed-cross-entropy.forward` | logits `f32[N,T,V]`; targets `i64[N,T]` | per-token loss `f32[N,T]` |
| `indexed-cross-entropy.backward` | logits `f32[N,T,V]`; targets `i64[N,T]`; upstream `f32[N,T]` | logit gradient `f32[N,T,V]` |

Inputs are read-only and must remain immutable and live for the complete K1
validate-plus-invoke dispatch. Their storage ranges must be pairwise disjoint. The
output must be disjoint from every input and other output under K1's generic rule.
L2 retains no descriptor, pointer, storage, owner, or registry identity after return.
It performs no allocation, materialization, copy, cast, transfer, or output
accumulation. A successful invocation fully overwrites the output.

There is no mask or reduction option. The output is exactly reduction `none`; later
model/trainer composition owns A0's
`sum(mask * per-token-loss) / sum(mask)` rule and supplies that objective's upstream
gradient to the direct backward.

## Numerical contract

For each row, in increasing vocabulary-index order, forward computes:

```text
m = max(logits)
s = left-to-right f32 sum(expf(logit[j] - m))
loss = logf(s) + (m - logits[target])
```

The parentheses in the final expression are contractual: they preserve `log(V)` for
equal extreme logits instead of losing it by adding `logf(s)` to an extreme `m`
first. Storage and intermediates are IEEE-754 binary32 C `float`; L2 requires
`sizeof(float)=4`, radix 2, 24-bit significand, and maximum exponent 128. It uses
`expf`/`logf`, fixed serial traversal, FP contraction disabled, and no fast-math or
hidden higher-precision path.

Backward directly recomputes the same stable row maximum and exponential sum, then
writes, for every vocabulary index:

```text
upstream * (expf(logit[j] - m) / s - (j == target ? 1.0f : 0.0f))
```

No one-hot tensor is constructed. Backward does not invoke Eshkol autodiff, a
finite-difference procedure, repeated forward perturbations, or an approximate
gradient. Finite differences exist only in the development test.

All logits and backward upstream values must be finite. NaN and either infinity are
rejected; signed zero and subnormals are accepted. A finite forward input whose loss
is not representable as finite f32 is rejected during validation. Backward likewise
preflights every output gradient for finiteness. `V=1` returns exact positive-zero
loss and exact zero logit gradient.

## Errors and failure atomicity

The native boundary uses K1 ABI-local errors:

| Failure | Category |
|---|---|
| target `< 0` or `>= V`; operand rank/extent mismatch | `shape-mismatch` |
| operand dtype differs from its schema | `dtype-mismatch` |
| operand/request device mismatch | `device-mismatch` |
| non-dense layout or nonzero offset | `noncontiguous` |
| nonfinite input, unrepresentable result, input alias, misalignment | `invalid-argument` |
| absent/unverified/nonmatching capability, including a zero-extent request outside the advertised range | `unsupported` |
| truncated ABI descriptor | `version-mismatch` |

Provider-specific rejections use stable K1 code `provider-rejected`, except a
misaligned or overflowing data range uses `invalid-buffer` and input aliasing uses
`aliasing-output`. K1 validates generic table/descriptor/output-alias rules first.
Because capability matching precedes provider validation, a raw zero-extent K1
request is `unsupported`, not `shape-mismatch`. A future A0 Eshkol adapter must
prevalidate public shapes and map that case to A0 `shape-mismatch`; L2 does not claim
that adapter today.

K1 completes generic validation, then L2 scans the complete schema, every target,
every floating input, and every prospective numerical result without mutation.
Only success enters the non-failing invocation. Consequently every recoverable
failure leaves all input and output bytes unchanged. Process termination, hardware
failure, and mutation racing the borrowed inputs are outside this in-process rule.

## Eshkol evidence and isolation

`tests/l2/eshkol_bridge.c` and `tests/l2/eshkol_runtime.esk` form a private,
fixed-arity, test-only transport. The bridge owns fixed native test buffers,
explicitly discovers the production L2 provider, and dispatches real forward and
backward calls. The Eshkol AOT program checks known values and an invalid-target
failure-atomic case twice. The bridge is never installed under `lib/`, never enters
the production archive, owns no shared registry, and is not an I2 or A0 tensor API.

The frozen development oracle is
`tests/q0/fixtures/indexed_cross_entropy_v1.json`, generated by exact PyTorch
`2.13.0+cpu` with deterministic algorithms and one CPU thread. Its SHA-256 is
`022b12c20b83f54908fdc4b711e92625a2db2eee263f11111d5445d010f1beff`.
The strict Q0 reader verifies its canonical bytes, checksum, source identity, lock
identity, operation identifiers, shapes, dtypes, roles, and tolerances. Python and
PyTorch remain only under `tests/`; production source, header, object, archive, and
AOT execution neither import nor invoke them.

## Verification

Run `make test-l2` after the configured K1/L2 build. The focused gate checks:

- C11/C++17 headers and the exact three-symbol public object allowlist;
- absence of the canonical provider symbol and dispatch-time allocator dependencies;
- provider-free versus explicit-provider capability reports in `C` and `C.UTF-8`;
- frozen PyTorch forward/direct-backward parity;
- extreme finite logits, `V=1`, fixed-order repeated determinism, gradient row sums,
  and a development-only central finite-difference check;
- negative/out-of-range targets, nonfinite input, finite-loss overflow, schema,
  layout, device, alias, zero-extent, and poisoned-output failure atomicity cases;
- two fresh Eshkol AOT compilations/executions with byte-identical output; and
- AddressSanitizer and UndefinedBehaviorSanitizer execution.

Local CachyOS/LLVM 22.1.6 results are compatibility evidence only. Supported Ubuntu
22.04 x86-64 / LLVM-Clang 21.1.8 exact-head CI is required before review handoff can
claim supported evidence.

## Measured limitations and downstream requirements

- No accelerator, f16, bf16, f64, non-CPU, strided, offset, broadcast, masked,
  reduced, ignore-index, target-gradient, or gradient-accumulation mode exists.
- The serial reference kernel is correctness-first and unbenchmarked; L2 makes no
  optimized or accelerated performance claim.
- L2 supplies no owned tensor, lifetime manager, registry, public Eshkol operation,
  or reverse-mode graph node. Private AOT evidence is not model integration.
- Accepted I2 storage/provider composition is required before N2/A2/M3 may use this
  primitive through a shared Eshkol training carrier. M3 must additionally prove
  direct-backward graph composition and A0 error mapping.
- ABI operation, schema, extent, numerical-order, nonfinite, ownership, alias, or
  error changes require issue #1 coordination and a version decision.
