# M3 fixed-profile model composition

M3 implements the bounded [composition contract](M3_COMPOSITION_PROPOSAL.md)
accepted in [integration decision 5744129957](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5744129957).
Numerical, ownership and package-identity checks pass locally, but the canonical
8192-forward retention gate exceeds its 600-second cap. See the
[measured failure and scoped remedy proposal](M3_RETENTION_GATE.md). Complete
retention, supported CI and independent M3-R remain required for acceptance.
The accepted M3T evidence establishes its
dependencies; it is not evidence that this model implementation passes.

## API and execution

The new installed `transformer.model` facade provides eight operations:
`model-forward`, `model-output-logits`, `model-output-loss`, `model-output-rng`,
`model-output-release!`, `diagnostic-output-vjp!`,
`diagnostic-model-logits-bits` and `diagnostic-model-logits-release!`.
The proposal specifies their exact arities, validation order and ownership.
The first four implement a bounded A0 subset. Construction and input ingress
remain the accepted M3T operations.

The profile is CPU f32, N1/T2/V256/D4/Hq=Hkv2/Dh2/L1/F8, learned positions,
pre-LayerNorm and final LayerNorm, bias-free projections, exact-erf GELU,
tied embedding/head and no dropout or cache. There are 14 unique parameter
tensors, 15 logical paths and 1,184 parameter values. Inputs own i64[1,2];
logits own f32[1,2,256]. Optional targets/masks and RNG ingress reject as
unsupported after option syntax validation. Both modes and both boolean
determinism options use this same fixed execution path. Loss and RNG accessors
return `#f` for a live output.

`native/m3_schedule.esk` contains the complete 21-role forward and 25-role reverse
schedule. All numerical operations call the accepted fixed-role M3T transport
and N3K/N2/A2 kernels. C handles graph storage, identity, lifetime and the final
I2 transaction. The delivered model path has no Python dependency.

## Graph ownership and contributions

Each forward snapshots 14 parameter tensors, IDs, logits, fixed constants, mode
and model/initializer identities into an independent immutable graph. It resets
the model's one private reusable workspace before publishing the output.
Separate outputs can coexist. Each logits accessor creates a new immutable
tensor with a graph lease; releasing the originating output preserves the
tensor and its ability to invoke VJP. Observed bytevectors are detached copies.

VJP restores the saved graph, replays the Eshkol forward schedule, snapshots the
finite seed and executes the analytic reverse schedule. It checks exact live
parameter values, mode and canonical identities again immediately before one
14-entry I2 prepare/commit/release. The tied gradient is head plus embedding
exactly once; Q/K/V and residual contributions use the accepted ordered sums.

The seed is an already-weighted numerator. The positive finite f32 weight is
metadata supplied as exact integer bits; it does not scale or normalize the
gradient. Each successful call adds every unique gradient once and increments
every contribution count once. The expected ordinal must match all destinations.
Zero seed still creates present-zero gradients or adds zero to existing values
while advancing metadata. Only explicit `module-zero-grad!` resets them.
Exact parameter/mode restoration permits reuse of an older graph.

The ten source-private seams stay localized inside the M3 archive. Its reset
preflight uses the M3-only nonallocating I1 registry check and existing I2 scoped
guards. It does not use ordinary M3T reset's allocating I1 leases. All cleanup
eligibility is checked before prepare; the post-commit tail releases the exact
plan and clears preallocated native/Eshkol fields. An impossible internal tail
defect terminates rather than returning a misleading retryable error.

## Packaging and validation commands

Use the pinned toolchain and the shared prerequisite planner:

```sh
make configure
/usr/bin/bash scripts/ci-build-prerequisites.sh model-composition
/usr/bin/bash scripts/test-m3.sh
```

The suite requires an absolute `M3_ORACLE_PYTHON` pointing to the pinned
Python 3.14.6 / PyTorch 2.13.0+cpu development interpreter or wrapper. Oracle CPU
controls are set before importing Torch. Native, reference, instrumented public
numerical and canonical installed-package gates run separately.

The development reference is frozen by `tests/m3/reference_manifest.json`, a
dedicated version-1 manifest separate from Q0. It records the exact profile,
parameter shapes, four cases, source/lock identities and generated JSON checksum.
Ordinary generation must match the entire manifest; a proposed regeneration is
an explicit separate mode. The direct PyTorch graph imports no production role
descriptors or schedule. Numerical checks cover every one of the 1,184 values
at two central-difference steps on two fixtures (4,736 comparisons), plus causal,
head-layout, Q/K and tied-edge mutations. Public parity checks compare each
parameter separately so larger tensors cannot mask a missing small Q/K gradient.

The M3 archive has one registry-owning member, seven installed facades and an
exact 93-global / 87-export surface. Source/native closures, undefined symbols,
archive membership and public-name strings have checked-in manifests. The
instrumented development archive exposes only its separately audited observers;
those symbols are absent from the production archive. Fresh package checks also
exercise hostile ambient paths, private symbols, arities, duplicate registries
and reversed facade import order.

The CI topology adds model-composition while retaining every predecessor gate:
18 suites and 25 top-level test commands. Local compatibility measurements and
supported Ubuntu 22.04 / LLVM 21.1.8 CI results must be identified separately.

The initial canonical archive audit passed on the pinned Eshkol compiler using
the explicit local CachyOS / Clang-LLVM 22.1.6 compatibility host. Its installed
public caller passed for initializer seeds 1729, 0 and signed-i64 maximum, with
distinct, boundary and repeated token IDs respectively. Maximum absolute logits
errors against the independent mathematical reference were `1.34583584e-08`,
`9.4684347e-09` and `2.24634212e-08`. These are preliminary local observations;
they do not substitute for the all-parameter VJP, lifetime, fresh-package or
supported-CI gates.

## Retention and remaining scope

A live graph owns 6,800 tensor payload bytes: 4,736 saved parameters, 16 IDs and
2,048 logits. Fixed constants add 24 semantic bytes; headers, shapes, strides,
native controls and Eshkol identities are separate. Each independent logits
copy adds 2,048 payload bytes. Final lease release destroys these payloads.

Ordinary I2 retains dead identity and plan shells. Output/VJP/reset histories
therefore have cumulative costs, even after payload release. The retention gate
reports exact Eshkol arena bytes at 1,024 and 8,192 iterations for forward,
logits, VJP, VJP/reset and failed VJP, with separate native counters and peak
RSS. Neither RSS nor fixed-workspace reuse proves flat training memory.

There is no claim of arbitrary shapes, JIT execution, compiler autodiff, GPU,
mixed precision, loss/RNG ingress, optimizer steps, generation, resume
equivalence or performance. Exactly one registry-owning aggregate may be linked
into a process. D2/O2/K2/C2 interoperability, rank-two checkpoints and a complete
training aggregate remain downstream. TR3 must independently establish its
retention behavior and training/resume evidence.
