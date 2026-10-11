# G3-C4 fixed P2/G2 public profile and result contract

**Proposal for independent review; public P2/G2 is not implemented or
accepted.** This leaf depends on the installed fixed C4 P2/G1 public package,
the [private P2/G2 terminal close](G3_C4_P2_G2_ESHKOL_TERMINAL_CLOSE_CONTRACT.md),
and the [logical I1 clone](G3_C4_P2_G2_LOGICAL_I1_CLONE_CONTRACT.md).
The latter two have source in this tree, but their same-aggregate compiled,
fault, and integration gates remain open. This document grants no public
admission before those gates and the typed-result prerequisite below pass.

## Fixed admission and dispatch

Add a distinct public profile symbol **`diagnostic-c4-p2g2`** on the existing
eight-pair config and existing `generator-create` / `generator-generate!`
arities. It admits only an authentic public C4 model, the same-aggregate raw
V256 T1 tokenizer, an authentic C4-owned dense CPU-i64 I1 `[1,2]` input
copied from a sealed two-byte T1 input, CPU-f32 execution, `N=1`, and
`:max-new-tokens` exactly `2`. `:eos` is `#f` or one exact integer byte ID
`0..255`; an EOS in the prompt does not terminate generation. Greedy keeps
the installed exact temperature/top-k/top-p canonical values. Categorical
keeps the installed accepted ranges and exactly one `:seed` or authenticated
public C4 result `:rng`. The first emitted EOS is included in the result
with logical `G=1`, cache length `3`; otherwise exactly two IDs are emitted
with `G=2`, cache length `4`, even if the second ID is EOS. Sampling consumes
one Philox block per emitted categorical ID and none for greedy. Capacity
remains four learned positions; EOS does not relax full-request admission.

The public C4 model and input authority already exists. Extend only its
exact C4 model branch: validate the public profile/config before translating
`diagnostic-c4-p2g2` to the private `diagnostic-c4` policy. Preserve
`diagnostic-c4-p2g1` as exactly `max_new=1`, `eos=#f`, and its existing
producer. An authentic dead C4 model/generator/input stays on the C4 branch
and rejects there; only identities absent from both C4 registries enter the
unchanged C2 route. At generate, authenticate the protected generator policy
and exact C4 input, perform the private full-request preflight, then dispatch
by the authenticated budget/profile to
`g3c4-p2g2-terminal-generate-internal` for budget two. The existing
`g3c4-p2g1-generate-internal` remains the budget-one route. Mutable shell
tags or caller-provided sampled IDs never choose the route. Any private
ownership gate needed by the installed archive must be activated only for
this authenticated P2/G2 branch, with a reviewed feature guard; loading a
source file's top-level initializer cannot be assumed to run in the archive.
Neither the native call ABI nor public package exports need a new name.

The private coordinator owns one call region, genuine P2 prefill and first
sample, actual same-registry T1 decode, first-EOS terminal commit or genuine
non-EOS prefix plus retained-logit second sample/forward, and one terminal
publication. The public wrapper returns only the authentic terminal shell
after that region closes. It must not call the P2/G1 producer, publish a
first-token partial result, rerun prefill, synthesize logits, or perform
fallible work after the private native-zero commit boundary. The profile
requires the private terminal and detached read/clone sources to be composed
once in the same registry-owning G3-G archive before dispatch is enabled.

## Public result and typed owner prerequisite

Keep the five installed arity-one output accessors and three typed releases.
For an exact live P2/G2 terminal output they return:

| Accessor | Fresh detached result |
|---|---|
| `generation-output-ids` | Singleton list of owned CPU-i64 rank-one I1 with exact shape `[G]` and the `G` emitted IDs |
| `generation-output-lengths` | Owned CPU-i64 I1 `[1]` containing `G` |
| `generation-output-cache-lengths` | Owned CPU-i64 I1 `[1]` containing `2+G` |
| `generation-output-text` | Singleton list of a new bytevector of exact length `G`, equal to genuine raw T1 bytes |
| `generation-output-rng` | New immutable, authenticated, releasable C4 RNG owner with all four final words |

Each accessor reauthenticates the distinct `terminal-output` entry and its
surviving original-allocation anchor, takes the reviewed detached terminal
snapshot, and checks `ID[0..G)` against the anchored text, `G=1/2`, cache
`3/4`, and the RNG schema before publishing a clone. The physical terminal
ID storage remains `[2]`; its zero second slot on `G=1` is never exposed.
Independent accessor calls return fresh owners or bytes. Clones remain valid
after source output release and generator close. The private snapshot is a
value descriptor, not itself a public tensor or RNG owner.

**Typed I1 prerequisite:** the private terminal logical clone is an opaque
kind-9 control, not the installed scalar result kinds 5/6/7. Its private
copy entry yields bytes and its release entry owns its own kind-9 registry;
neither makes it an installed public I1. Before returning it as a public
tensor, review and prove a public rank/device/element-access path for exact
logical `[1]` or `[2]` that recognizes this owner, including lease and error
semantics. Alternatively add a reviewed C4 native typed result constructor
that copies the authenticated logical prefix into a fresh public I1 `[G]`,
with a matching exact-owner release path. The existing scalar constructor
`et_g3c4_private_result_tensor_create_v1(kind,value)` cannot construct
`[2]`, and its validators hard-code values `1/3` for lengths/cache lengths.
Extend or add reviewed typed constructors for `G=1/2` and `2+G=3/4` before
using those accessors. Do not cast kind 9 to I1, feed it to
`generation-tensor-release!`'s current scalar branch, or relabel it as a
G3-T/C2 owner. The chosen path must supply a documented tensor shape and
element contract and prove independent clone/release ownership.

Each existing public output and release wrapper must check both the C4
transport registry and the separate C4 model registry, including tombstones,
before any C2 fallback. If either registry contains the identity, the C4
branch then checks exact kind and liveness; a C4 model is the wrong output,
tensor or RNG kind even though it has no transport entry. Only identities
absent from both registries reach the unchanged C2 branch. The distinct
terminal output must not enter P2/G1 copy-out or scalar result publication.
Public output release uses the private terminal native-first, borrow-aware,
idempotent release;
tensor/RNG releases use only the matching newly reviewed typed owners.
Protected native identity and original anchors prevent swapped shell,
record, text, or clone controls from releasing another owner. Held genuine
I1 leases reject before mutation and permit release/retry. Wrong-kind,
forged, copied, and cross-profile owners are `invalid-argument`; authentic
dead owners are `invalid-state` for access, with exact repeated release
idempotent. Preserve the invoked public operation, bounded data-only E1
details and `cause #f`. Device mismatch may be reported only from a
reviewed live-device discriminator; other violated native invariants keep
their typed error rather than being guessed from generic `internal`.

Every accessor is a separate read/clone transaction after generation, never
an extension of the irreversible terminal tail. Preallocate public shells,
ledger and return list before creating a native clone; reauthenticate the
source and new owner after all recoverable reads and allocation, then mark
only the completed result live. On failure, scrub detached scratch bytes,
release any created native clone, tombstone the pending entry, and leave
the source output and earlier clones unchanged. Cleanup failure after an
ownership mutation is fail-stop. No `m3-call` may nest around an existing
private snapshot/clone `m3-call`.

## Required evidence and exclusions

First finish and integrate the private native terminal, same-call Eshkol
coordinator, detached snapshot/logical clone, and their accepted source and
compiled gates. Then independently review the typed I1 prerequisite and
this public profile/dispatch/adapter. Compile one registry-owning archive
and fresh installed public callers. Test two genuine P2 prompts, greedy and
nontrivial categorical sampling, first EOS, second EOS, and no EOS; verify
exact IDs/raw T1 text, `[1]`/`[2]` tensor shapes and elements, `G=1/2`,
cache `3/4`, P3/T4 same-provider logits/K/V, and all four final RNG words
against an independent G3-S/Philox oracle including carry, singleton,
final-block and exhausted-second-draw cases.

Cut every recoverable generation operation before native terminal zero:
model/input admission, I1/A2 borrows, prefill, both samplers/forwards and
T1 decode. Before first commit, abort preserves the documented entry or P2
state and original RNG; after non-EOS prefix commit, failure preserves cache
`3/1110`, first RNG advance and binding but publishes no result or partial
text. Prove valid repair/retry where specified and fail-stop for impossible
failures after native terminal zero. Separately cut postpublication snapshot
and typed clone allocation/copy: these failures leave the committed terminal
output, cache and successor RNG intact, roll back only a pending accessor
clone, and permit read retry. Cut clone/output release before mutation; a
recoverable held-lease rejection leaves the relevant live owner intact for
release retry. Exercise malformed shapes/storage/IDs, wrong order and owner
swaps, equal-length carrier substitutions, held leases,
dead/forged/cross-output handles,
source release before clone read, two independent clones, exact repeated
release, and feature-off symbol/route rejection. Re-run installed P2/G1
and C2 public tests unchanged; verify source/export/undefined manifests,
private-link negatives, pinned normal/repeat/sanitizer parity and sealed
evidence. Source checks alone do not satisfy this gate.

This fixed leaf adds no `N>1`, P1/P3 admission, `G>2`, general generation
loop, GPU/mixed precision, cache/no-cache parity, persistence, continuation,
or CLI3 `generate` claim.
