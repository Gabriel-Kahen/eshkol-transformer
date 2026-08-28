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
back to Python or another runtime. `build` performs an explicit AOT compile and
requires the compiler depfile to contain the Eshkol library source. `test` performs
two fresh AOT compilations and executions, compares output bytes including the final
newline, and verifies an actionable missing-toolchain failure. `smoke` runs the built
native artifact and expects `eshkol-transformer-smoke:v1`.

Production Eshkol source belongs under `src/eshkol_transformer/`, tests under
`tests/`, developer entry points under `scripts/`, and compatibility pins under
`toolchain/`. F0 defines no transformer API, runtime capability, numerical-oracle,
or benchmark contract.

## First release criterion

The first release must deterministically train a byte-level decoder-only transformer,
resume it exactly from a checkpoint, reduce held-out loss, and generate text through
an Eshkol-authored model and training loop.

See:

- [Architecture](docs/ARCHITECTURE.md)
- [Development roadmap](docs/ROADMAP.md)
- [Quality gates](docs/QUALITY_GATES.md)
- [Integration log](docs/INTEGRATION_LOG.md)
- [Contributing](CONTRIBUTING.md)
