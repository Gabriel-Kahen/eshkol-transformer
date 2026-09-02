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
   BPE evidence includes exact merge tie-breaking, admitted document-order and
   chunk-partition invariance, whole/stream rank-stage parity, all-byte fallback,
   every strict UTF-8 split plus F0/F4 scalar boundaries, and measured exact-limit
   admission. The decoder maximum includes both one 73,728-ID chunk and 73,728
   one-ID omit-only chunks under count-pinned time/RSS/no-warning gates. Delivered
   compiled parsers must reject every frozen header/payload/order invariant, and the
   compiled D1 seam must preserve malformed/truncated/checksum corruption categories
   while rejecting fingerprint or vocabulary mismatch.
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

## Required configuration gates

- Strict, bounded, non-executable parsing with duplicate and unknown keys rejected
  before a configuration value is constructed.
- Every explicitly present source leaf is validated before overrides, so an override
  cannot mask an invalid source type, range, enum, or policy. Defaults, valid input
  values, explicit overrides, and absent-field derivation are tested at their
  documented precedence boundary, including incompatible combinations.
- Repeated fresh compilation of the configuration test, plus byte-identical canonical
  manifests and fingerprints across fresh processes, working directories, and
  hostile-environment runs.
- Golden canonical bytes, direct external-fingerprint recomputation,
  resolved/provenance mutation, malformed manifest, and unsupported version/feature
  tests.
- Production dependency inspection proving no Python/PyTorch runtime, evaluator,
  include expansion, environment interpolation, or hidden execution fallback.
- Successor aggregate boundary inspection proving hostile include/path isolation,
  exact repository tuple admission, localized private/native symbols, no archive
  index leakage, deterministic localized objects/archives/evidence/AOT binaries,
  public-caller closure, and duplicate registry ownership rejection.

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
