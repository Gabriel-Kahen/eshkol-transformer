# Private P2/G2 second-frame and terminal ownership contract

**Proposed, unimplemented contract for independent review.** This is the
next dependency after the authentic, source-private P2/G2 T1 prefix
coordinator. It covers one `(P,max G,N)=(2,2,1)` diagnostic-C4 CPU-f32/V256
request, owned I1 `[1,2]` prompt, registered raw T1 tokenizer and A2 capacity
four. It adds no public profile, provider row, package export, CLI command or
persistence claim. The [fixed private generation proposal](G3_C4_P2_G2_PRIVATE_GENERATION_PROPOSAL.md),
[prefix ownership contract](G3_C4_P2_G2_T1_PREFIX_OWNERSHIP_CONTRACT.md),
and [prefix coordinator leaf](G3_C4_P2_G2_T1_PREFIX_COORDINATOR_LEAF.md)
remain authoritative for first-token admission and provenance.

## Existing boundary and proposed private ABI

The current first-frame carrier, `token_frame_begin_last_v1` and
`token_forward_v1` admit budget two only before prefix commit at position 2.
The accepted carrier also rejects a sampled first EOS **before** calling
`token_forward_v1`; the current prefix coordinator therefore cannot reach
an EOS forward. A reviewed EOS-capable first-frame carrier route is a
separate prerequisite. It must sample and forward the genuine selected EOS
with the accepted P2 input/pending-output, bytevector header/extent,
native-owned alias and prefill-binding checks, and write the authentic ID
only after successful forward and reauthentication. One guarded successor
bridge may handle both selected EOS and non-EOS and branch only after that
single real draw; an EOS-rejected frame must never be replayed to discover
which route to take. The existing non-EOS bridge and prefix behavior must
remain unchanged when this feature is off.
The accepted prefix commit retains the genuine first selected ID/raw byte,
`[1,256]` next logits, K/V at position 2 and successor RNG at `3/1110`, but
leaves the sole requested-capacity-two output pending and unready. There is
no accepted post-prefix sampler/carrier. Passing Eshkol bytevector `ptr` to
the native raw `float[256]` or `i64*` arguments would address its length
header; passing a pointer to retained context logits would leak private
storage. A **new, separately reviewed, feature-gated native second-frame
bridge** is therefore proposed, with exact ABI name/signature still open.
It must admit only the original active `(kind,budget)=(2,2)` call, same model
pins/prefill binding, committed non-EOS prefix, idle frame, exact sole pending
output and committed cache `3/1110`. It samples directly from the context's
retained 256 next-logit words through the accepted G3-S route, using native
scratch, and writes the selected ID only as little-endian payload bytes into
an authenticated, unaliased Eshkol eight-byte staging carrier with intact
length header. It must never replay P2 prefill, synthesize four-row logits,
expose a raw pointer, select from caller-supplied logits or begin A2 before
checking that a categorical second draw is available. Greedy consumes no
RNG; categorical consumes exactly one Philox block per emitted token,
including EOS and singleton selection. On any pre-frame failure, the
authentic prefix and successor RNG remain unchanged.

The second frame opens one width-one A2 transaction at position 3, runs the
same accepted 21-role T1 forward using the selected integer, and retains its
candidate K/V, `[1,256]` next logits and successor RNG until terminal
preflight. The existing forward may gain a **narrow post-prefix state route**
only after review of its budget-two guards and provenance; its P2/G1 and
first-frame behavior must remain byte-for-byte equivalent. The Eshkol
composer decodes the bridge's selected ID through the *same registered raw
T1 owner* into a captured one-byte second-token carrier, checks the seven
upper staging bytes are zero, and reauthenticates the original call,
generator, native owner, input, output, registry/core, model eval/pins and
all child, auxiliary and carrier identities/known extents after every
recoverable boundary. A caller-supplied ID/raw byte cannot replace the
bridge and decode. Selected `-1/256`, malformed headers, aliasing or changed
bytes reject before commit.

## Terminal forms and commit ordering

Requested capacity two and **actual emitted count** must be separate native
and Eshkol owner facts. The preallocated native output ID I1 has physical
shape `[2]`; pending Eshkol raw/staging payloads have capacities two/sixteen
bytes. Before initial prefill, allocate and authenticate all terminal
ID/raw/staging/text representations needed for either logical `G=1` or
`G=2`, including two distinct private one-byte raw/eight-byte ID staging
carriers. Captured original identity, bytevector header and allocator extent
are checked before pointer use and at each recoverable boundary. No
allocation, T1 decode, output-ID borrow, text encoding, provider dispatch,
recoverable release or shell construction may occur after the terminal
closed-tail boundary. Physical unused ID/raw/staging slots stay zero and
unexposed. Native and Eshkol live output accessors must report and clone
only the exact logical count, raw text and final four-word RNG; existing
P2/G1 copy-out continues to reject this owner type.

Two **proposed**, mutually exclusive terminal routes are required:

1. **First selected EOS:** the newly reviewed EOS-capable first-frame carrier
   must complete genuine position-2 forward, then same-registry T1 decodes
   its selected-ID payload. Both the accepted first-frame bridge and
   `p2g2_prefix_commit_v1` reject EOS; this branch needs the new carrier
   route **and** a distinct first-frame terminal preflight/commit, not a
   fabricated non-EOS prefix or a bypass of either rejection.
   Prepare exact logical `G=1` ID/text/output, verify staged A2 K/V and
   successor RNG, then commit position 2 and publish the sole output in one
   no-failure tail. Final cache is `3/1110`; no second sampler/frame runs.
2. **Non-EOS prefix then second token:** use the authentic retained-logit
   bridge and T1 forward above. Both second EOS and budget exhaustion
   produce logical `G=2`, including the selected second ID. Prepare exact
   two-ID/two-byte output and verify the view-free staged A2 transaction,
   cache length `4`, mask `1111`, binding, selected IDs/raw bytes and final
   RNG. A dedicated terminal commit then appends position 3, advances RNG
   once, clears private continuation, and publishes the sole output in one
   no-failure tail. No intermediate G1 output is ever live.

The current `output_prepare_v1` rejects budget two; ID copy/text acceptance
assume requested `generated_length==budget` and only implement the one-ID
conversion; P2/G1 `generation_frame_prepare/commit`, `call_prepare_end` and
`call_finish` reject the prefix-committed state. None is silently reused or
generalized. A reviewed private terminal output/finish ABI must bind the
exact pending output and actual length, verify all output I1 and A2 leases
are absent, preflight the native call/output/owner state, and define the
closed tail through native publication, Eshkol ledger/shell publication and
pin drain. Any impossible failure after the first terminal mutation is
fail-stop. Native output becomes independently live only at that tail;
Eshkol must finish the same protected `m3-call` region before returning the
sole shell. Output release is exact-owner, native-first, borrow-aware,
retryable and idempotent. No recoverable step or partial shell may separate
the native and Eshkol publication writes.

The protected record's trusted phase and child ledger must distinguish
`before-prefix`, `prefix-committed` and terminal completion: first EOS moves
directly from the first to terminal completion, while a non-EOS first token
moves through the second. The phase, exact output child edge, native
published state and sole shell must agree. Before the terminal tail the
output stays pending; after it, the call has no pending output, live frame
or generator/model/tokenizer edge, and the published output is independently
owned. These bounded ledger/phase writes belong in the same no-failure tail,
not a recoverable cleanup after native publication.

## Abort, lease and ownership invariants

Before first commit, failure discards the staged frame and pending output,
scrubs captured carriers and leaves entry or committed P2 `2/1100` cache,
binding and original RNG according to where prefill failed. After the
non-EOS prefix but before second terminal commit, failures in second
sampling, A2 begin/forward/view, T1 decode, output preparation, preflight
or an exhausted draw preserve first K/V and `3/1110` cache, binding,
parameters, gradients and successor RNG. Abort discards only the second
candidate transaction and pending output; it scrubs captured raw/staging
storage and clears private prefix continuation. It publishes no native
output, Eshkol shell, partial ID/text or generated count. A later attempt
is a new full request through normal admission, not an implicit continuation
or RNG rewind.

The protected scope must remain open through recovery. At each abort or
release, reauthenticate original call/output/native/input, model/generator
and tokenizer owners, child ledger and every auxiliary/carrier identity and
extent. Preflight genuine pending-output I1 and staged A2 leases **before**
any mutation. A held lease rejects with all owners and links live; the
in-region caller may release the genuine lease and retry. Forged/dead,
cross-owner or equal-length substituted edges never authorize release of
another owner. If an escaping guard cannot successfully abort, it fails
stop. Once terminal commit has begun, no recoverable abort path applies.

## Required proof and dependency order

First accept the narrow EOS-capable first-frame carrier and post-prefix
second-frame admission/carrier/forward routes against authentic owner and
prefix state; then accept the EOS-aware native terminal output/finish leaf
and its exact logical-length ownership model; finally
compose both with same-registry T1 and the protected Eshkol ledger/shell
tail. The ABI and ownership leaves are coupled by the exact active call,
pending output and prefix state, so neither may define a contradictory
owner or terminal phase. The first-EOS terminal route may be tested without
a second frame, but cannot be inferred from the non-EOS prefix route.

Supported proof must include two distinct P2 prompts, greedy and
nontrivial categorical policies, exact actual IDs/raw bytes and all four
RNG words against an independent G3-S/Philox oracle (carry, final block,
singleton and exhausted second draw included). Check position-2 P3 and
position-3 T4 bitwise next logits/K/V, cache masks `2/1100`, `3/1110`,
`4/1111`, unchanged model parameters/gradients/prefill binding, first EOS,
second EOS and no EOS. Cover malformed/aliased headers, altered carrier
bytes, stale registry/core/eval/pins, wrong output/ledger/auxiliary identity,
sampled `-1/256`, falsified logits/K/V, wrong order,
provider/forward/decoder/output allocation and precommit
cuts, held I1/A2 lease rejection/release/retry, exact output release, and
impossible first-EOS/second-terminal closed-tail fail-stop. Feature-off,
P2/G1/manual/C2 and public rejection require compiled preservation checks;
run pinned normal/repeat and ASan+UBSan+LSan Eshkol/native gates. This is a
generation path: no training operation or differentiable change is added,
so numerical gradient checks are not a substitute for the required
forward/K/V and parameter/gradient preservation proofs.
