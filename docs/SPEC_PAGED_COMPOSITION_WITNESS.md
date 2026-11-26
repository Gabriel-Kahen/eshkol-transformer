# Speculative verifier and paged KV composition witness

`make test-spec-paged-witness` is a development-only native C11 gate. It
composes the existing `et_spec_verify_{greedy,sample}_v1` verifier with the
existing paged KV sequence APIs. The verifier sees deterministic finite mock
logits, and the paged manager sees deterministic finite mock K/V tensors.

Each case starts with a three-token prefix in a block-size-two pool and forks
an anchor, uninterrupted control, and speculative sequence. The speculative
sequence stages two draft tokens with `et_paged_kv_append_tokens`. The verifier
returns an accepted count and emitted span. The witness then truncates the
speculative sequence to `prefix_len + accepted_count` and appends the emitted
correction or bonus token through `et_paged_kv_append_token`.

The four cases cross greedy and seeded stochastic verification with full
acceptance and a rejection after one accepted token. The rejected draft tail
occupies a physical block before truncation. The gate checks the committed
prefix immediately after truncation, then checks every logical K/V word against
the uninterrupted control, the unchanged anchor, the block-sharing pattern,
each physical block's refcount against its table references, pool accounting,
and zero-leak cleanup. The stochastic fixture uses top-k one, so the selected
tokens are deterministic while the verifier still consumes its documented RNG
draws.

On 2026-11-25, `make test-spec-paged-witness` passed in the pinned
`eshkol-r0-debug:20260928` Ubuntu 22.04 container with Clang 21.1.8.
All four cases passed in optimized native and ASan/UBSan/LSan builds with
byte-identical stdout. The independent source-and-test review found no
remaining blocker after the rejected-tail physical-release assertion was added.

This proves the bounded rollback handoff for mock data. It does not provide a
target or draft model forward, establish how live target K/V is produced for a
correction or bonus token, connect public generation/FFI, or measure throughput.
