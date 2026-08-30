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

`toolchain` clones and builds only the pinned Eshkol revision. `configure` rejects a
missing, wrong-revision, wrong-version, or unsupported toolchain instead of falling
back to Python or another runtime. `build` performs an explicit AOT compile,
requires the compiler depfile to contain the Eshkol library source, and leaves the
explicit-link K1 archive at `build/k1/libeshkol_transformer_k1.a`; its public header
remains at `include/eshkol_transformer/kernel_abi.h`. `test` performs
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

X1's public `transformer.config` source stub links explicitly against the single
prelocalized E1B/X1 artifact at `build/x1/libeshkol_transformer_x1.a`. The archive
exports only the six E1 accessors and six fixed package-specific configuration
wrappers; its trusted implementation source and evidence are not application include
roots. See [docs/CONFIG_FORMAT.md](docs/CONFIG_FORMAT.md).

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
- [Configuration and resolved-run format](docs/CONFIG_FORMAT.md)
- [Integration log](docs/INTEGRATION_LOG.md)
- [Contributing](CONTRIBUTING.md)
