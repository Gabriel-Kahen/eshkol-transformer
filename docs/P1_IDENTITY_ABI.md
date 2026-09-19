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
`trusted` directory. It is a replacement, never an additive library. Its 36
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

## Unpublished construction extension v1

The accepted M3T extension adds exactly five private native calls: construction
`begin(context)`, `module_create(context, construction)`,
`handle_create(context, construction)`, `seal(context, construction)`, and
`abort(context, construction)`, all with the `et_p1_private_construction_` prefix
and `_v1` suffix. Results use the existing context result pointer. The four public
inspection functions, token kinds, ABI 1.1 layouts, and existing signatures are
unchanged. Construction identities are separately registered before dereference;
they are not public P1 token kinds.

Only one unpublished construction may be active. Begin reserves 8,192 enrollment
slots (the 4,096 module and 4,096 parameter bounds) before returning. Scoped creates
reserve capacity before allocating a token, and enroll each successful token without
further allocation. Seal and abort validate all exact enrolled records before a
nonallocating transition. Abort revokes only those modules and handles; it cannot
revoke arbitrary existing tokens. Abort is idempotent; sealed ledgers reject abort.
Both terminal transitions free the enrollment array and retain a small ledger
identity tombstone. Repeated failures consume identity tombstones but no live-token
capacity. Successful modules retain the existing process-local lifetime.

The trusted Eshkol surface appends construction begin, root, seal, abort, and
parameters at slots 59–63, preserving all preceding slots and all ten C2 seams.
Its exact surface is 18 public plus 46 private names. Begin takes an admitted
provider name and returns the registered construction identity. Root returns the
fresh root shell. Registration can only attach exact same-construction modules
and handles; public observation and buffer registration reject before seal.
Parameters returns detached `#(rows ties)`, with a path-sorted list of
`#(path canonical-handle shape dtype device)` rows and ordinary P1 tie groups.
The retained independent schedule is compared to a new actual traversal at seal;
mutating the returned metadata cannot alter that schedule. Seal also rejects
unattached enrolled modules or handles. It returns the canonical finalized root.

I2's wrapper binds its ledger to this exact P1 construction. M3T owns all native
parameter storage. Begin promotes all ledger/membership graphs before native
scope creation and retains the exact preconstruction I2 registry/count. A
preallocated parameter0 anchor is filled by the fixed trusted caller immediately
after I2 begin returns, using the already-created native owner and only scalar
pointer stores. Abort validates
that anchor and every canonical carrier/handle binding and preflights all 14 native
parameters, including zero/partial P1 enrollment, borrows and plan pins. It then
revokes P1 authority and restores the saved I2 registry/count without allocation
before M3T performs its fixed nonfailing storage teardown. Both terminal states
clear the anchor and saved baseline references. No provider
callback selects a destructor. The source-only I2 preflight bridge is available
only in the M3T aggregate; ordinary I2/C2 construction begin explicitly reports
unsupported before allocating or enrolling anything. Its `NULL,NULL` sentinel
queries that fixed availability and confers no parameter authority.

The focused `scripts/test-m3t-construction.sh` witness exercises each of the 14
positions at zero and partial enrollment. Five blockers are reachable on unbound
parameters: value/gradient ordinary borrows, value/gradient scoped borrows, and a
tensor copy plan. Gradient-reset and gradient-contribution plans require a bound
identity; the witness checks their explicit invalid-state rejection and unchanged
live counts on unbound parameters, then exercises all seven modes on the two
canonically bound partial-enrollment parameters. Rejected aborts preserve exact
P1/I2 authority and native live counts, and release permits retry. Native P1 begin
allocation failpoints also prove cleanup before the caller receives an I2 ledger.
