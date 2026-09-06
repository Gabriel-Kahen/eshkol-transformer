# eshkol-transformer

Eshkol-native transformer construction, pretraining, evaluation, and generation.

The project is intended to train language models from random initialization without
using Python as the runtime training framework. Python/PyTorch may be used only as
development oracles for numerical parity tests and conversion tools.

## Project status

Pre-alpha. The repository contains the implementation plan, architecture boundaries,
quality gates, and an Eshkol-native build/smoke foundation. No training-capable public
API is stable yet.

## Build and test

The initial supported lane is Ubuntu 22.04 x86-64 with Clang/LLVM 21.1.8. The
Eshkol compiler/runtime is built from commit
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`, which reports version
`1.3.4-evolve`. Exact compatibility inputs are in `toolchain/eshkol.lock`; package
requirements and limitations are in `toolchain/README.md`.

From a clean checkout on the supported lane, run:

```bash
/usr/bin/bash -c 'make toolchain'
/usr/bin/bash -c 'make clean && make configure'
/usr/bin/bash -c 'make build'
/usr/bin/bash -c 'make test'
/usr/bin/bash -c 'make smoke'
```

Run the focused T1 tokenizer gate with:

```bash
/usr/bin/bash -c 'make test-t1'
```

Run the focused T2 deterministic BPE and streaming gate with:

```bash
/usr/bin/bash -c 'make test-t2'
```

Run the focused A2 causal-attention, RoPE, and transactional KV-cache gate with:

```bash
/usr/bin/bash -c 'make test-a2'
```

`toolchain` clones and builds only the pinned Eshkol revision. `configure` rejects a
missing, wrong-revision, wrong-version, or unsupported toolchain instead of falling
back to Python or another runtime. `build` performs an explicit AOT compile,
requires the compiler depfile to contain the Eshkol library source, and leaves the
explicit-link K1 archive at `build/k1/libeshkol_transformer_k1.a`; its public header
remains at `include/eshkol_transformer/kernel_abi.h`. It also builds D1's single
localized E1B/D1 archive at `build/d1/libeshkol_transformer_d1.a`, with exact
symbol-policy evidence beside it; the D1 test path consumes that same artifact.
`test` performs
two fresh AOT compilations and executions, compares output bytes including the final
newline, and verifies an actionable missing-toolchain failure. `smoke` runs the built
native artifact and expects `eshkol-transformer-smoke:v1`.

Production Eshkol entry points belong under `src/eshkol_transformer/` and reusable
modules under `lib/transformer/`; tests belong under `tests/`, developer entry points
under `scripts/`, and compatibility pins under `toolchain/`. F0 defines no transformer
API, runtime capability, numerical-oracle, or model-training contract. B0's
reproducible host-process smoke benchmark is
documented in [docs/BENCHMARK_FORMAT.md](docs/BENCHMARK_FORMAT.md); run it with
`make benchmark` after `make build`.

The build also leaves I1's separate exact signed-i64 CPU container archive at
`build/i1/libeshkol_transformer_i64.a`, with its ABI 1.0 header at
`include/eshkol_transformer/i64_tensor.h`. Its explicit K1 provider verifies only
bounded deterministic `tensor.i64` / `storage.copy` requests; see
[docs/I1_I64_TENSOR.md](docs/I1_I64_TENSOR.md).

I2 is integrating the shared ABI 1.0 owned dense CPU-f32 carrier, explicit borrowed
K1 views, and P1-bound value/accumulated-gradient substrate required by N2 and O2.
Its explicit provider accessor verifies only bounded deterministic `tensor.f32` /
`storage.copy` and never defines K1's canonical provider symbol. The native archive
is `build/i2/libeshkol_transformer_f32.a`; the one-member localized P1L/C1
integration aggregate is `build/i2/libeshkol_transformer_wave2.a`. The aggregate
retains the existing E1/X1/P1/D1/C1/T1 public surface and localizes every I2 seam.
Run the focused gate with `make test-i2`; see
[docs/I2_F32_TENSOR.md](docs/I2_F32_TENSOR.md).

L2's carrier-neutral deterministic CPU-f32 fused indexed cross-entropy provider is
at `build/l2/libeshkol_transformer_l2.a`, with its isolated ABI 1.0 header at
`include/eshkol_transformer/indexed_cross_entropy.h`. It exposes only explicit K1
provider-accessor discovery, per-token forward, and direct backward; it does not
claim an owned tensor, Eshkol autodiff graph, global provider, or I2 carrier. See
[docs/L2_INDEXED_CROSS_ENTROPY.md](docs/L2_INDEXED_CROSS_ENTROPY.md).

The build leaves A2's carrier-neutral serial CPU-f32 provider and fixed-capacity
transactional cache in `build/a2/libeshkol_transformer_a2.a`. Consumers obtain the
provider only from `et_a2_kernel_provider_v1`; the archive does not define K1's
generic resolver. Cache reads expose full-capacity dense K/V, exact lengths, and a
canonical false-outside-length bool mask. A2 does not supply a production tensor
carrier, P1 binding, provider aggregate, accelerator path, or public training module.
Its exact numerical and lifetime contracts are in
[docs/A2_ATTENTION.md](docs/A2_ATTENTION.md). Source-tree native consumers link the
A2 archive before `build/k1/libeshkol_transformer_k1.a` and `-lm`; there is no A2
install or dynamic-discovery contract.

X1's public `transformer.config` source stub links explicitly against the single
prelocalized E1B/X1 artifact at `build/x1/libeshkol_transformer_x1.a`. The archive
exports only the six E1 accessors and six fixed package-specific configuration
wrappers; its trusted implementation source and evidence are not application include
roots. See [docs/CONFIG_FORMAT.md](docs/CONFIG_FORMAT.md).

The P1 structural module/state-tree gate is `make test-p1`. Its logical in-memory
state schema, deterministic UTF-8 path ordering, tie semantics, strict loading,
provider 2.0 exact-once ownership, explicit `state-dict-release!`, read-only
state-backed handles, and tensor-runtime limitations are documented in
[docs/P1_MODULE_STATE.md](docs/P1_MODULE_STATE.md). It defines no checkpoint file or
numerical tensor capability.
The narrow process-local native identity boundary used only to enforce P1's
public/trusted compile separation is documented in
[docs/P1_IDENTITY_ABI.md](docs/P1_IDENTITY_ABI.md).

C1's internal data-only checkpoint container and local atomic-I/O boundary are
specified in [docs/CHECKPOINT_FORMAT.md](docs/CHECKPOINT_FORMAT.md). Run its
deterministic format, corruption, ownership, native ABI, failpoint, sanitizer, and
production-isolation gate with `make test-c1`. C1 intentionally exposes no public
trainer checkpoint API and no production tensor codec; C2 owns that composition.

T1's Eshkol-authored byte tokenizer, special-token rules, canonical artifact,
fingerprint, C1-backed persistence limits, and exact-I1 output lifetime are specified
in [docs/TOKENIZER_FORMAT.md](docs/TOKENIZER_FORMAT.md). The build creates one
canonical `build/t1/libeshkol_transformer_wave1.a` aggregate from trusted source
inputs and localizes it once. Its public boundary is exactly 47 globals: six E1
error accessors, eighteen P1 module/state wrappers, eight D1 data wrappers, six X1
configuration wrappers, one C1 persistence-policy wrapper, and eight T1 tokenizer
wrappers. The installed `transformer.persistence` surface contains only
`persistence-policy`; C2 checkpoint operations remain unavailable. The authoritative
runtime test is compiled Eshkol AOT; Python participates only as an independent
development oracle and never in the production archive or execution path.
Tokenizer, policy, and successful encoded-tensor identities are strongly retained in
append-only aggregate registries until process exit. Their lookup cost is linear and
their memory cost is cumulative, so applications should construct/load once, reuse
identities, serialize T1 calls, and use a bounded worker process when a process-exit
reclamation boundary is required. Exact per-artifact format limits do not bound this
cumulative process-lifetime cost; see the lifecycle guidance in the T1 contract.

T2 adds a distinct, versioned deterministic BPE artifact without changing T1 bytes
or the eight tokenizer names/arities. The build leaves the successor aggregate at
`build/t2/libeshkol_transformer_wave2.a`; applications link either that aggregate or
the Wave-1 aggregate, never both. Wave 2 preserves the same 47 public globals while
adding localized Eshkol-only training, rank-stage streaming, and bounded D1
composition contracts. Python is a development oracle only. See
[docs/BPE_TOKENIZER_FORMAT.md](docs/BPE_TOKENIZER_FORMAT.md).

## First release criterion

The first release must deterministically train a byte-level decoder-only transformer,
resume it exactly from a checkpoint, reduce held-out loss, and generate text through
an Eshkol-authored model and training loop.

See:

- [Architecture](docs/ARCHITECTURE.md)
- [Development roadmap](docs/ROADMAP.md)
- [Quality gates](docs/QUALITY_GATES.md)
- [Benchmark format](docs/BENCHMARK_FORMAT.md)
- [Native-kernel ABI and capability report](docs/K1_KERNEL_ABI.md)
- [Exact signed-i64 tensor container](docs/I1_I64_TENSOR.md)
- [Dense CPU-f32 tensor and parameter-gradient substrate](docs/I2_F32_TENSOR.md)
- [Fused indexed token cross-entropy](docs/L2_INDEXED_CROSS_ENTROPY.md)
- [Causal attention, RoPE, and KV-cache substrate](docs/A2_ATTENTION.md)
- [Checkpoint container format and atomic I/O](docs/CHECKPOINT_FORMAT.md)
- [Configuration and resolved-run format](docs/CONFIG_FORMAT.md)
- [Byte tokenizer format and runtime contract](docs/TOKENIZER_FORMAT.md)
- [Deterministic BPE tokenizer and streaming contract](docs/BPE_TOKENIZER_FORMAT.md)
- [Token corpus format](docs/TOKEN_SHARD_FORMAT.md)
- [Integration log](docs/INTEGRATION_LOG.md)
- [Contributing](CONTRIBUTING.md)
