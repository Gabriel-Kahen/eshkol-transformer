# P1 private identity ABI 1.0

## Status and scope

This is a private, process-local P1/C1/I1 integration ABI. It is not an A0, K1,
I1, checkpoint, serialization, or installed native ABI. Version 1.0 requires
64-bit pointers and `size_t`; every call has fixed arity and uses only fixed-width
`i64`, `ptr`, and bounded byte spans. There is no generic opcode dispatcher.

The public product is
`public/libeshkol_transformer_p1_identity.a`. It contains exactly:

- `et_p1_public_identity_abi_major_v1()`;
- `et_p1_public_identity_abi_minor_v1()`;
- `et_p1_public_token_kind_v1(token)`;
- `et_p1_public_token_live_v1(token)`.

The trusted product has the same archive basename under a mutually exclusive
`trusted` directory. It is a replacement, never an additive library. Its 25
`et_p1_private_*_v1` functions have hidden ELF visibility and cover one private
context, fixed-kind identity creation, immutable provider admission witnesses,
exact state/provider binding, unpublished-identity cleanup, error/result access,
and deterministic live/tombstone counts. The trusted product is not installed by
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
restore authority. Successful state/provider/module/tree/handle/entry shells live
until trusted cleanup where supported or process exit. There is no concurrency or
cross-process claim; post-fork use rejects.

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

Seven callback identities are inert native tokens. Actual Eshkol callbacks remain
in a hidden immutable Eshkol snapshot. Sealing first validates the complete token
set, then atomically records and back-references it; repeat sealing with the exact
set is idempotent and any substitution rejects. A sealed callback cannot be
revoked or transplanted to another provider.

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
`source-message`; cause is `#f`. Unknown status/code values map to bounded
`internal` diagnostics.

Native code is intentionally unable to validate state schema, walk Eshkol data,
call a provider, inspect a tensor, prepare/commit a load, serialize state, or infer a
capability. Arbitrary malicious native object injection is outside the Eshkol-module
threat model. Fresh-cache AOT tests prove arbitrary compiled Eshkol linked only with
the public package cannot resolve private Eshkol names or private native symbols.
