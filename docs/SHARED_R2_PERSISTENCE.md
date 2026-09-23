# SHARED-R2 bounded persistence admission

Status: **independently reviewed integration candidate with local owner gates
complete** under issue
[#114](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/114). Root-owned
supported integration CI remains required for acceptance.

SHARED-R2 keeps K1 ABI 1.0, every I2 native symbol and public arity, the provider
name `eshkol-transformer-f32`, and C1/C2 serialized bytes. The I2 provider and its
sole verified `tensor.f32` entry advance to version 1.1 with evidence
`I2:bounded-exact-f32-storage.copy-v2`. Rank zero and rank one
`[0,4611686018427387903]` remain unchanged. Exactly these rank-two shapes are
added, in descriptor order: `[2,4]`, `[4,4]`, `[4,8]`, `[8,4]`, `[256,4]`.
No other rank-two or higher-rank shape is admitted.

`storage.copy` still requires matching f32 CPU dense row-major zero-offset
request/input/output shapes, exact checked byte lengths, aligned nonwrapping data
spans, and disjoint input/output storage. Success is one exact-bit `memcpy`.
There is no cast, reshape, allocation, fallback, retained input pointer, or new
lifetime authority.

K2 still discovers the actual accessor-returned I2 provider through K1, then
exact-audits the complete provider descriptor and all eleven live report rows.
Its source constraint snapshot is a fresh deep copy on every accessor call. The
ten unverified rows are unchanged. The deterministic canonical report is 3,133
bytes with SHA-256
`50c7078af4d3e495c4d668b6b8aaab29024cbab6a775fa6e98c9e9d10124389e`.

The C2 wire format is unchanged. Deterministic tests generate a fully checksummed
M3-schema state with 15 logical parameter paths, 14 unique tensors, the
`head/weight`–`token_embedding/weight` tie, 4,736 unique model bytes, and 28
correctly shaped O2 moments. The operational source-composed lane asserts the
exact 43-request shape multiset, codec calls, native live counters, and release
lifecycle. A separate installed-facade lane uses only
`transformer.capabilities`, `transformer.persistence`, and `transformer.trainer`:
public `checkpoint-load/3` reconstructs each owner, `checkpoint-save!/4` produces
a byte-identical round trip, and `trainer-state-release!/1` is idempotent. A
second schema/storage witness
changes only `position_embedding/weight` from `[2,4]` to `[4,4]`, yielding 4,768
unique model bytes; it does not claim a C4 runtime. A checksummed unsupported
rank-two near-miss remains an early K2 rejection with zero codec calls and flat
live storage.

Independent review approved implementation commit `8047cec9` and tree
`e80930797d556edac83fed12ea18c6c4dabcaa2b` with no remaining source or gate
design blocker. On the local CachyOS/LLVM-Clang 22 compatibility host,
`scripts/test-i2.sh`, `scripts/test-k2.sh`, and `scripts/test-c2-public.sh` pass.
The public C2 retention gate is flat at 5,832,704 bytes for both 1,024 and 8,192
iterations. This compatibility evidence does not replace the root orchestrator's
single supported Ubuntu 22.04/LLVM-Clang 21.1.8 integration CI dispatch.

Historical I2, K2, and C2 acceptance runs remain evidence only for their original
bounded contracts.
