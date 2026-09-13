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
/usr/bin/bash -c 'python3.14 -m venv "$(pwd)/.tmp/q0-venv"'
/usr/bin/bash -c '"$(pwd)/.tmp/q0-venv/bin/python" -m pip install -r tests/q0/requirements-oracle.lock'
/usr/bin/bash -c 'printf '\''#!/usr/bin/bash\nexport ATEN_CPU_CAPABILITY=default\nexec "%s" "$@"\n'\'' "$(pwd)/.tmp/q0-venv/bin/python" > "$(pwd)/.tmp/n2-oracle-python" && chmod 0500 "$(pwd)/.tmp/n2-oracle-python"'
/usr/bin/bash -c 'Q0_PYTHON="$(pwd)/.tmp/q0-venv/bin/python" N2_ORACLE_PYTHON="$(pwd)/.tmp/n2-oracle-python" O2_ORACLE_PYTHON="$(pwd)/.tmp/q0-venv/bin/python" A2_ORACLE_PYTHON="$(pwd)/.tmp/q0-venv/bin/python" ESHKOL_RUN="$(pwd)/.deps/eshkol-build/eshkol-run" make test-after-build'
/usr/bin/bash -c 'make smoke-after-build'
```

Code pull requests, pushes to `main`, and merge-queue runs partition the complete
test command set across eight parallel blocking suites. Full P1, T1, D2, C1,
native-numerics including N2/O2/Q0, contract/data, T2 runtime, and T2 boundary gates run independently,
while suite-specific build targets avoid
outer builds whose artifacts the full scripts immediately rebuild. The serial
`Exhaustive acceptance` workflow also runs nightly and through manual dispatch;
after one clean build it uses no-rebuild test, smoke, and benchmark entry points.

[Supported run 34373099684](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/34373099684)
took 3h59m10s including queue and spent about 72 minutes in redundant full builds.
The earlier reduced run 34057603751 took 11m55s but did not carry the same coverage.
[The first parallel run](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/34400156724)
passed five suites (native numerics 21m16s, parameters 31m35s, loader 29m39s,
contracts 12m04s, checkpoint 6m45s), but both tokenizer jobs lacked the D2 aggregate
needed by their reverse-import checks. The revised prerequisites and separate T2
runtime/boundary jobs still need hosted validation; that failed run does not establish
a successful full-suite duration.

PRs changing only `README.md`, `CONTRIBUTING.md`, or Markdown under `docs/` run the
CI topology and selector checks without launching compiler suites. `AGENTS.md`,
unknown paths, mixed changes, empty diffs, or unavailable history require full CI.
Renames are checked as both a deletion and an addition. Every push to `main` and
merge-queue run still requires all suites, and the final status check accepts a
skipped matrix only for an explicitly selected documentation-only PR.

The pinned oracle environment requires Python 3.14.6.
Run the serial test phase locally after a build with the same four absolute oracle
variables shown above and `make test-after-build`.

Run the focused T1 tokenizer gate with:

```bash
/usr/bin/bash -c 'make test-t1'
```

Run the focused T2 deterministic BPE and streaming gate with:

```bash
/usr/bin/bash -c 'make test-t2'
```

D2's accepted memory-bounded loader contract provides the ten A0 dataset/batch
operations plus explicit `token-batch-release!`, the canonical `ESHKDCU1` cursor,
and a source-composed 58-global/52-export canonical aggregate. Its compiled public,
carrier, exact-resume, corruption, resource, sanitizer, and frozen-Q0 gates run with
`make test-d2`. `token-batch-release!` deterministically invalidates the generation
and frees its native `17*N*T` carrier; the pinned runtime does not individually
reclaim caller Eshkol shell allocations. Long-running callers must therefore keep
each next/use/release interval in one lexical `with-region`, unless they deliberately
retain/promote aliases and account for that caller-owned storage. This lifetime
clarification is
[accepted with live-shell conditions](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5608140148).
Stale/idempotent-alias guarantees require a live shell allocation; accessing freed
lexical-region storage has no safety guarantee, and keeping a raw reference alone
does not preserve that allocation's lifetime.
Shell authentication uses two fixed compiled-constructor code identities as a
per-process private implementation ABI; it is neither public nor serializable.
D2 is complete after independent D2-R approval, supported CI, merge, and a focused
merged-main retest. See
[docs/D2_SHARD_LOADER.md](docs/D2_SHARD_LOADER.md).

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

I2 supplies the shared ABI 1.0 owned dense CPU-f32 carrier, explicit borrowed K1
views, and P1-bound value/accumulated-gradient substrate required by N2 and O2.
Its explicit provider accessor verifies only bounded deterministic `tensor.f32` /
`storage.copy` and never defines K1's canonical provider symbol. The native archive
is `build/i2/libeshkol_transformer_f32.a`; the one-member localized P1L/C1
integration aggregate is `build/i2/libeshkol_transformer_wave2.a`. The aggregate
retains the existing E1/X1/P1/D1/C1/T1 public surface and localizes every I2 seam.
Run the focused gate with `make test-i2`; see
[docs/I2_F32_TENSOR.md](docs/I2_F32_TENSOR.md).

N2's ABI 1.0 deterministic serial CPU-f32 primitive provider is
`build/n2/libeshkol_transformer_n2.a`; its only public symbol is the explicit
`et_n2_kernel_provider_v1` accessor declared in
`include/eshkol_transformer/n2_primitives_abi.h`. It implements only the exact
embedding, linear, LayerNorm, GELU, ReLU, dropout, and residual rows documented in
[docs/N2_PRIMITIVES.md](docs/N2_PRIMITIVES.md). The focused `make test-n2` gate
exercises those kernels through accepted I2 f32 and I1 exact-i64 borrows. N2 adds
no carrier, canonical K1 resolver, compiler-autodiff claim, accelerator, mixed
precision, or fallback.

O2 adds Eshkol-native dense CPU-f32 AdamW, canonical parameter groups, optional
global-L2 clipping, I2 accumulation consumption, constant/linear successful-update
schedules, and explicit optimizer-snapshot release. Its successor aggregate is the
one-member `build/o2/libeshkol_transformer_wave2.a`. It exposes exactly 53 globals:
the inherited 47-global I2 boundary plus six fixed optimizer wrappers with arities
2/1/1/1/2/1. Applications link this aggregate instead of, never together with, an
I2, T1, T2, or other registry-owning aggregate. Run `make test-o2`; see
[docs/O2_OPTIMIZER.md](docs/O2_OPTIMIZER.md). O2 optimizer snapshots release their
owned moment carriers explicitly. Live optimizer receivers and their moments have no
v1 destroy operation and remain process-local until exit; identity tombstones are
cumulative and registry lookup is linear.

N3K adds a separate explicit ABI 1.0 provider for the accepted two-token diagnostic
profile at `build/n3k/libeshkol_transformer_n3k.a`, discovered only through
`et_n3k_kernel_provider_v1`. Its exact embedding, bias-free linear, GELU, residual,
layout/VJP, ordered-sum, and explicit-state matrix-initializer contracts are in
[docs/N3K_PRIMITIVES.md](docs/N3K_PRIMITIVES.md). Run `make test-n3k` for independent
numerical/gradient references, owned I1/I2 borrows, native failure atomicity,
private pinned-Eshkol AOT, sanitizers, and package isolation. This provider adds no
public model, RNG transport, initializer registry, or provider composition.

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
- [AdamW optimizer, schedules, and logical state](docs/O2_OPTIMIZER.md)
- [Fused indexed token cross-entropy](docs/L2_INDEXED_CROSS_ENTROPY.md)
- [Causal attention, RoPE, and KV-cache substrate](docs/A2_ATTENTION.md)
- [Checkpoint container format and atomic I/O](docs/CHECKPOINT_FORMAT.md)
- [Configuration and resolved-run format](docs/CONFIG_FORMAT.md)
- [Byte tokenizer format and runtime contract](docs/TOKENIZER_FORMAT.md)
- [Deterministic BPE tokenizer and streaming contract](docs/BPE_TOKENIZER_FORMAT.md)
- [Token corpus format](docs/TOKEN_SHARD_FORMAT.md)
- [D2 memory-bounded shard-loader contract](docs/D2_SHARD_LOADER.md)
- [Integration log](docs/INTEGRATION_LOG.md)
- [Contributing](CONTRIBUTING.md)
