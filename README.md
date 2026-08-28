# eshkol-transformer

Eshkol-native transformer construction, pretraining, evaluation, and generation.

The project is intended to train language models from random initialization without
using Python as the runtime training framework. Python/PyTorch may be used only as
development oracles for numerical parity tests and conversion tools.

## Project status

Pre-alpha. The repository currently contains the implementation plan, architecture
boundaries, and quality gates. No training-capable public API is stable yet.

## First release criterion

The first release must deterministically train a byte-level decoder-only transformer,
resume it exactly from a checkpoint, reduce held-out loss, and generate text through
an Eshkol-authored model and training loop.

See:

- [Architecture](docs/ARCHITECTURE.md)
- [Development roadmap](docs/ROADMAP.md)
- [Quality gates](docs/QUALITY_GATES.md)
- [Contributing](CONTRIBUTING.md)

