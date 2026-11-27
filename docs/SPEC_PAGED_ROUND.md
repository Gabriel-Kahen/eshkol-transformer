# Greedy speculative decoding with paged K/V

`et_spec_paged_round_greedy_v1` in
`include/eshkol_transformer/spec_paged_round_abi.h` composes the accepted SPEC5
greedy verifier and PKV5 sequence ownership APIs in an opt-in C11/C++17 leaf.
`make build-spec-paged-round` builds
`build/spec_paged_round/libeshkol_transformer_spec_paged_round.a`. Link that
archive before the existing SPEC5 and PKV5 core archives and `-lm`. The
existing `scripts/build-spec-decode.sh` and `scripts/build-paged-kv.sh` build
those core archives without running Eshkol AOT tests.

The caller supplies two distinct model-scoped paged pools and live parents at
the same token position. The caller binds each parent to the same token
prefix, and provides each producer's K/V for all `K` candidate tokens plus
`[K + 1, V]` target logits. Logit row `i` predicts the next token after the
parent and the first `i` candidate tokens. Each pool can have a different K/V
geometry. The per-side candidate layout is
`[K, num_layers, num_kv_heads, head_dim]` in binary32.

The coordinator forks both parents, stages the candidate suffixes, runs the
greedy verifier, and truncates both children to the accepted length. If there
is a correction or bonus, the callback receives read-only views of the two
truncated children and the token and position to compute. It returns one
K/V token per model in caller-owned scratch that remains valid and unchanged
until the round returns. The round calls it once for a correction or bonus
even if that token is EOS. When EOS itself was accepted from the draft,
truncation is sufficient and callers can omit the callback.

On success both returned child IDs are caller-owned, and both child caches
contain the entire emitted suffix. Parents remain caller-owned. Callers can
publish new model state after success. The callback may stage private model
state and scratch. It must leave both pools and externally visible model state
untouched, return normally, and release the temporary prefix views when it
returns. Calls require exclusive use of both pools and disjoint backing storage
for the pools, round result, and caller inputs.

An error leaves the result unchanged and releases all temporary sequences
and live block references. Existing live sequences retain their K/V and block
tables. Staging can increment cumulative CoW telemetry and write bytes that
remain in freed blocks. The return value identifies the failing cache, verifier,
or callback and preserves its status code.

`make test-spec-paged-round` exercises full acceptance, first and later
rejection, all three EOS paths, malformed inputs and callback K/V spans,
second-pool slot and block exhaustion, callback failure, and retry after a late
second-pool append fails. The fixture verifies every surviving block refcount,
parent table and K/V contents, both model-specific result caches, and cleanup.
The native gate links the separate archives in C11 and C++17 and runs the full
fixture under ASan, UBSan, and leak detection.

The API consumes caller-supplied CPU binary32 logits and K/V. Live target and
draft forward adapters, sampling, GPU support, and public generation dispatch
remain outside this leaf.
