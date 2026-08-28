# Quality gates

## Required on every numerical component

- Shape, dtype, device, and error-contract tests.
- Known-value forward tests.
- Central finite-difference gradient checks where differentiable.
- Independent reference parity for high-risk kernels.
- Repeated-input and accumulation cases.
- NaN, Inf, empty, boundary, and malformed-input cases.

## Required integration gates

1. Tokenizer encode/decode round-trip and deterministic vocabulary training.
2. Shard corruption detection and exact loader-cursor resume.
3. Tiny transformer forward and gradient parity against a frozen oracle fixture.
4. One-batch overfit.
5. Small-corpus loss reduction and held-out evaluation.
6. Bitwise-identical next training step after checkpoint reload where supported;
   otherwise a documented tolerance with a proved cause.
7. AOT/JIT parity.
8. Fixed-seed reproducibility.
9. Flat resident memory across a long bounded training loop.
10. Save/reload generation equivalence.

## Performance evidence

Benchmarks record commit, hardware, OS, compiler, backend, dtype, tensor shapes,
warmup, repetitions, throughput, latency, and peak memory. A backend is not called
accelerated until execution on that device is directly observed and tested.

## Merge policy

- No red required gates.
- No undocumented fallback or unsupported case.
- No public format change without a version/migration decision.
- No performance rewrite without correctness parity.
- Cross-workstream API changes require orchestrator review.

