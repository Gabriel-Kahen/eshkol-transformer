# P1 private identity ABI 1.1

## Status and scope

This is a private, process-local P1/C1/I2 integration ABI. It is not an A0, K1, I1,
checkpoint, serialization, or installed native ABI. Version 1.1 requires
64-bit pointers and `size_t`; every call has fixed arity and uses only fixed-width
`i64`, `ptr`, and bounded byte spans. There is no generic opcode dispatcher.

The public product is
`public/libeshkol_transformer_p1_identity.a`. It contains exactly:

- `et_p1_public_identity_abi_major_v1()`;
- `et_p1_public_identity_abi_minor_v1()`;
- `et_p1_public_token_kind_v1(token)`;
- `et_p1_public_token_live_v1(token)`.

The trusted product has the same archive basename under a mutually exclusive
`trusted` directory. It is a replacement, never an additive library. Its 31
`et_p1_private_*_v1` functions have hidden ELF visibility and cover one private
context, fixed-kind identity creation, immutable provider admission witnesses,
exact state/provider binding, state-backed tensor identity and release transitions,
unpublished-identity cleanup, error/result access, and deterministic live/tombstone counts. The trusted product is not installed by
the normal build.

## Frozen layout and ownership

Compile-time assertions freeze these ABI-v1 facts:

| Item | Size / offset |
|---|---:|
| opaque token allocation | 264 bytes |
| caller-visible token nonce words | offsets 0 / 8 |
| private registry record | 256 bytes |
| registry callback-token array | offset 32 |
| registry token kind | offset 88 |
| registry provider-ID bytes | offset 128 |
| private context | 344 bytes |
| error record | 272 bytes |
| error operation/message | offsets 16 / 80 |

Tokens and contexts are bridge-owned. Callers hold opaque pointers only and must
not read, copy, free, or mutate their storage. The registry retains token memory as
a live identity or permanent tombstone, so an allocator address cannot be reused to
restore authority. Successful provider/module/tree/handle shells live until trusted
cleanup where supported or process exit. The legacy ownerless native
`et_p1_private_state_entry_create_v1` symbol is retained for ABI compatibility only:
the production trusted Eshkol root never calls it, it carries no Eshkol value or
carrier authority, and it cannot establish state provenance. Every entry shell
actually exposed from a state uses the state-scoped constructor and binds to one
exact state; successful clone-on-bind replacement, rollback, or explicit state
release makes the affected entries permanent inert tombstones. Explicit state
release likewise makes the state and every dependent state-tensor shell an inert tombstone. Tombstones
retain no Eshkol carrier, tensor storage, or raw pointer. There is no concurrent-
mutation or cross-process claim; use is serialized and post-fork use rejects.

Each caller-visible token contains only the nonzero 128-bit `getrandom` identity and
reserved bytes. Its bridge-private registry record authoritatively owns kind, owner
context, origin PID, duplicate integrity value, provider bytes, callback roles, and
state binding. Recognition finds the registered pointer before dereferencing it and
then verifies the nonce against that record. Null, foreign, copied, mutated,
wrong-kind, cross-context, stale, and post-fork tokens reject without mutation.
Entropy and allocation failure are explicit; there is no random, time, address,
file, Python, scalar, CPU, or other fallback.

Provider ID input is an exact byte span with maximum
`ET_P1_IDENTITY_MAX_PROVIDER_ID_BYTES == 127`. Eshkol validates the semantic symbol
before calling C. C permits zero length, requires a nonnull pointer for nonzero
length, rejects negative/oversized spans, and copies exact bytes before publishing
the token. It never truncates, calls `strlen`, treats NUL specially, performs text
normalization, or returns a bridge-owned byte pointer.

Provider interface 2.0 has eight callbacks. The original seven callback identities
remain in the frozen callback-token array; the eighth `release-owned!` identity uses
the existing record binding field, so ABI 1.1 changes no record size or frozen
offset. Actual Eshkol callbacks remain in a hidden immutable Eshkol snapshot.
Release-aware sealing validates the complete eight-token set, then atomically records
and back-references it; repeat sealing with the exact set is idempotent and any
substitution rejects. Legacy seven-callback seal/match entrypoints cannot admit or
downgrade a release-aware provider. A sealed callback cannot be revoked or
transplanted to another provider.

A state-scoped entry or state-tensor token binds to one exact live state token.
State-scoped entry creation prevents equal raw entries exposed through different
states from sharing identity authority. The ABI-only legacy ownerless entry token is
never accepted as state provenance. Validation rejects foreign,
copied, mutated, stale, wrong-kind, cross-context, and cross-state pairs before any
Eshkol carrier is returned. `et_p1_private_state_release_begin_v1` is the native-first
linearization point: the first valid call tombstones every dependent state-entry and
state-tensor and then the state, clears native bindings, and returns result `1`; a repeated call on the
same intact tombstone returns result `0` without allocating or mutating. P1 checks
for an active scoped borrow before this call and invokes provider destruction only
after the native transition, so stale handles cannot authorize freed storage.

Callback revoke and provider abort are unpublished failed-admission cleanup calls.
They accept only live, exact-kind, same-context, unreferenced/unsealed identities.
Repeat, stale, sealed, published, wrong-kind, or foreign cleanup rejects atomically.
Provider admission creates all allocations and recoverable decisions before seal;
after seal, trusted Eshkol performs only preallocated nonraising registry writes.

## Status and error ownership

Status values are `0 ok`, `1 invalid-argument`, `2 unsupported`,
`3 invalid-state`, and `4 internal`. The context owns one fixed-size error/result
record. A failed call publishes no new identity or stale success result; it may only
replace this diagnostic record. Error access returns context-owned immutable text
valid until the next private call. The trusted P1 root does not forward those
pointers or create an intermediate error object. It snapshots the numeric status
and code before cleanup, maps them to fixed bounded text, and directly invokes
E1B's five-value nonreturning raise seam. Canonical data-only details contain
`native-category`, `source-domain` set to `p1-private-bridge`, `source-code`, and
`source-message`; cause is `#f`. Unknown values and impossible status/code pairings
map to bounded `internal` diagnostics.

Native code is intentionally unable to validate state schema, walk Eshkol data,
call a provider, inspect a tensor, prepare/commit a load, serialize state, or infer a
capability. Arbitrary malicious native object injection is outside the Eshkol-module
threat model. Fresh-cache AOT tests prove arbitrary compiled Eshkol linked only with
the public package cannot resolve private Eshkol names or private native symbols.
