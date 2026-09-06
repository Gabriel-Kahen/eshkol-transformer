# D2 memory-bounded shard-loader candidate

Status: **proposed; public contract not accepted**. The current-main proposal is
recorded on [integration issue #1](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5557201280).
Until that decision is accepted, D2 provides only a carrier-neutral Eshkol semantic
core, development references, and a private native lifetime/transport candidate. It
adds no public wrapper and freezes no 58-global/52-export aggregate.

## Candidate semantics

D2 consumes the ordered flat token stream described by accepted D1. Packed mode may
cross canonical shard boundaries; unpacked mode treats each shard as a packing
boundary but does not invent document semantics. For token length `L` and positive
sequence length `T`, a stream contributes zero rows when `L < 2`, otherwise
`1 + floor((L - 2) / T)`. Row `r` begins at `r*T`, inputs copy source positions,
targets copy the one-token shift, and unused positions are token zero with false
loss mask. Every candidate batch is fixed `[N,T]`; unused rows in the last batch are
fully masked. D2 emits no causal mask—A2/model code owns causal attention policy.

Optional seeded ordering partitions logical row ordinals into consecutive windows
of at most `B` rows and applies descending Fisher–Yates inside each window. Draws use
domain-separated SHA-256 and rejection sampling over unsigned 64-bit words, removing
modulo bias within each window. It does not claim a globally uniform permutation.
Absent seed and `B=1` are canonical identity order. One finite permutation is
defined; epoch reseeding is outside D2.

The internal state stages a complete batch plan before advancing its next ordinal.
Failure leaves the ordinal unchanged. The proposed cursor is detached, canonical,
checksummed data binding the D1 manifest digest, tokenizer identity/vocabulary,
normalized loader options, total logical rows, and next ordinal. Exact format bytes
remain proposed rather than accepted; C2 must not serialize them yet.

## Candidate carrier and resource bound

The private native candidate owns two dense rank-2 CPU `i64` planes and one dense
rank-2 CPU one-byte `bool` plane, all shape `[N,T]`. Their exact semantic payload is
`17*N*T` bytes. Views use the verified K1 descriptor layout and are valid only for
an explicit synchronous borrow. One dataset/current-generation design permits at
most one live batch; release invalidates it before the fixed destruction tail. No
scalar list, f32 substitution, cast, transfer, Python runtime, or allocation fallback
is provided.

The proposed maximum semantic working payload is:

```text
maximum_manifest_bytes + maximum_shard_bytes + 17*N*T + 8*min(B,M)
```

where `M` is the logical row count. This is computed evidence, not another config
option. Control structures, K1 view metadata, allocator overhead, stacks, shared
libraries, and compiler/runtime RSS are reported separately. A contract-conforming
implementation may
retain at most one bounded manifest window, one bounded shard, one batch, and one
bounded shuffle window, independent of total corpus token bytes.

## Evidence boundary and limitations

`make test-d2` is the focused candidate gate. It validates the stdlib-only frozen Q0
fixture, carrier-neutral Python oracle/resource tests, two fresh strict Eshkol AOT
builds and runs, and the private C ABI when present. The gate must additionally prove
warning-clean C/C++, explicit release/borrow behavior, failure cleanup, sanitizer
results, Python-runtime isolation, bounded RSS/file descriptors, and exact/one-over
admission before D2 can move to review.

The current candidate reports semantic payload, fixed carrier metadata, lifecycle
allocation baselines, and file-descriptor stability. It does not claim an integrated
loader RSS bound or independence from total corpus size while the public contract
and Eshkol wrapper remain unaccepted and unimplemented.

D1 retains its documented trusted-directory, symlink, concurrent namespace mutation,
and power-loss limitations. D2 is CPU-only, serialized, nonreentrant, and finite. It
does not claim a GPU path, alternate dtype, global-uniform shuffle, document reset,
automatic epoch progression, C2 checkpoint field, O2 integration, or public ABI.

The merged T2 and I2 aggregates are sibling 47-global/41-export registry owners:
T2 includes BPE internals but not I2, while I2 starts from T1 and omits T2. A future
D2 aggregate therefore needs one newly reviewed source-composed root; linking either
localized archive or loading both trusted roots is invalid. The proposed 58/52
boundary is name arithmetic, not present aggregate evidence.
