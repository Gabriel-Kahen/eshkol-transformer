# G3-G capacity-two public orchestration and ownership proposal

**Proposed for review; no public implementation or acceptance is claimed.** This
contract narrows the accepted [A0 public API](../PUBLIC_API_CONTRACT.md#13-generator)
and [G3 diagnostic-C2 design](../G3_GENERATION_PROPOSAL.md) to P1/G1 and
P2/G0 generation. It consumes the accepted [G3-T private contract](G3_T_PRIVATE_CONTRACT.md),
G3-N rows and G3-S sampling ABI. Every G3-G facade, package name, wrapper,
constructor, clone and production operation below is a **new proposed seam**
until separately reviewed and implemented. The G3-T development witness is not
an installed operation. Nothing here admits P2/G1, capacity three, repeated
generation, persistence or a CLI.

## Fixed surface and package tuple

The sole new installed facade is `lib/transformer/generation.esk`. Its exact
provided names and arities are:

| Name | Arity | Result or effect |
|---|---:|---|
| `generator-create` | 3 | live generator retaining authentic model and tokenizer, with owned RNG and no cache |
| `generator-prefill!` | 2 | new owned, no-grad CPU f32[1,256] last logits; cache replacement, no draw |
| `generator-decode-step!` | 2 | new owned, no-grad CPU f32[1,256] last logits; one manual append, no draw |
| `generator-generate!` | 2 | opaque live output after P1/G1, P1/G0 or P2/G0 |
| `generation-output-ids` | 1 | new one-element list of owned CPU i64[G], G=0 or 1 |
| `generation-output-lengths` | 1 | new owned CPU i64[1], value G |
| `generation-output-text` | 1 | new one-element list of detached raw bytevectors of length G |
| `generation-output-rng` | 1 | new independent, immutable G3 RNG owner |
| `generation-output-cache-lengths` | 1 | new owned CPU i64[1], value P+G |
| `generation-input-create` | 1 | copy authentic same-aggregate sealed T1 i64[P] to owned CPU i64[1,P], P=1 or 2 |
| `generation-token-input-create` | 1 | one exact byte ID to owned CPU i64[1,1] |
| `generation-tensor-release!` | 1 | typed release of G3 input, last-logit or output tensor clone |
| `generation-output-release!` | 1 | typed output release; detached clones survive |
| `generator-close!` | 1 | close cache, scratch and owned RNG; retain no released payload |
| `generation-rng-release!` | 1 | typed independent RNG release |

These are the nine unchanged A0 declarations plus six accepted G3 design
additions, not fifteen extra names beyond M3. `generator-generate!` accepts only
an authentic live G3 input owner. Its configured budget is zero or one; require
P+G<=2 **before** call acquisition, model pinning, cache replacement or RNG
sampling. P2/G1 rejects even if EOS might stop the draw. The accepted profile,
eight-pair config, exact bit-valued sampling policy, optional byte EOS, model
evaluation, tokenizer fingerprint and admission order remain as in G3; no
additional public option is proposed. P1/G0, manual P1/P2 prefill and manual
P1 decode are included in the same C2 semantics, but their production schedules
remain separate G3-G implementation work.

Proposed sibling package: one `libeshkol_transformer_g3g.a` containing one
registry-owning `g3g_package.o`; compose M3/M3T/I2/P1/T1/E1/G3-T/N/S in one
source root, never link their owning archives alongside it. Install exactly the
existing seven M3 facades plus `transformer/generation.esk`. Preserve all 87 M3
boxed exports and append one boxed export for each of the fifteen names above:
**eight facades, 102 boxed exports, 108 globals/public-name strings**. Candidate
tuple names are `native/g3g_package_{root.esk,bridge.c,private_renames.txt,public_exports.txt}`
and `scripts/build-g3g.sh`; they are proposed names, not existing artifacts.
Proposed generated private/public symbol pair for each row is
`et_e1b_private_g3_<stem>_cabi_v1` /
`et_e1b_public_g3_<stem>_v1`, with these exact stems in table order:

`generator_create`, `generator_prefill`, `generator_decode_step`,
`generator_generate`, `generation_output_ids`, `generation_output_lengths`,
`generation_output_text`, `generation_output_rng`,
`generation_output_cache_lengths`, `generation_input_create`,
`generation_token_input_create`, `generation_tensor_release`,
`generation_output_release`, `generator_close`, `generation_rng_release`.

Only those fifteen new boxed public symbols may appear in the successor export
manifest. The native G3-T/N/S/A2/cache helpers, registries, production role
helpers and test observers remain local/hidden. Exact source, object, undefined,
defined and string manifests must be measured and reviewed on the implementation
candidate; the counts above are an acceptance target, not build evidence.

## Admission, ownership and detached results

`generation-input-create` first authenticates the exact same-aggregate T1 shell
and sealed rank-one raw-byte IDs through T1's registry, then checks P in {1,2}
and every ID in 0..255. It copies synchronously into a distinct G3-owned I1
i64[1,P] and retains no additional T1 borrow; the sealed T1 shell keeps its
own internal borrow. A copied/tagged T1 shell,
foreign aggregate, M3T input, output clone or shape-compatible generic tensor
cannot acquire G3 input authority. The accepted private `input_from_t1` stem
and same-aggregate wrapper now have focused source-private evidence; their
public mapping remains proposed. Existing scalar and pair constructors are
private witnesses, not alternate public constructors. Manual scalar token
input still requires an owned I1[1,1] implementation to meet this proposed
public result contract: the current private scalar witness stores inline IDs.

Output is a separately authenticated live owner with no parent-call, model or
tokenizer roots after publication. It holds the exact generated IDs, G and P+G
lengths, final RNG words and raw bytes as immutable snapshots, independent of
subsequent generator mutation, replacement or close. `generation-output-ids`,
`-lengths`, `-cache-lengths` and `-rng` authenticate the output, call the four
accepted source-private G3-T clone stems, enroll fresh
closed G3 owners and return detached copies; clone failure enrolls no live shell
and leaves the output unchanged. IDs are a fresh one-element list around the
owned i64[G] clone. Text is a fresh list and bytevector copy; caller changes to
one returned list or bytevector affect no later accessor. No output accessor
returns a borrowed native header, cache view, T1 shell or G3-T test observer.
The four native clone stems have focused private witnesses; the Eshkol wrappers,
package linkage and public accessor behavior described here remain proposed.

For each ID, length, cache-length or RNG accessor, allocate and root its future
Eshkol shell, pending entry, registry cons cell and (for IDs) one-element result
list **before** requesting the native clone. Native clone construction must
allocate its full detached payload before enrolling a live native owner. On
failure, discard the pending Eshkol wrapper without a live enrollment. After
clone success, attach the pointer and publish the already allocated wrapper
and list in a no-allocation tail. If a later Eshkol operation can still raise,
its handler must release the exact newly cloned native kind, clear the pending
wrapper and re-raise the first error; no orphan owner may remain. Text similarly
preallocates and roots both its bytevector and one-element list, copies the
output-owned bytes, then publishes without further allocation. Every injected
allocation cut leaves the source output and both Eshkol/native live-registry
membership unchanged; a failed accessor returns no partially live result.

Generator, output, tensor clones and RNG snapshots have separate lifetime and
registry identities. `generation-tensor-release!` accepts only G3 input, last
logits and ID/length/cache-length clone kinds; `generation-rng-release!` only
G3 RNG snapshots; output release only output; generator close only generator.
Each exact released identity is a retained tombstone and repeated release is
idempotent. Wrong kind, copied/forged identity and cross-aggregate shell are
`invalid-argument`; authentic dead non-release use is `invalid-state`. Busy or
active-borrow release rejects before payload mutation. Closing the generator
does not close outputs or accessor results and never destroys the retained M3
model or T1 tokenizer. Dropping an Eshkol reference does not release a payload.

## Single-token transaction and failure cuts

G3-G supplies the production P1 21-role prefill and one 21-role T1 append
schedule using only the accepted fixed G3-N/N2/N3K/A2/G3-S routes. No model
forward with a discarded graph, shape relabel, scalar numeric loop, fallback
provider or caller-selected resolver is allowed. Prefill uses position zero;
the sampled token uses position one and is copied from I1[1] into distinct
I1[1,1] decode scratch. The accepted G3-T call/14-pin/frame protocol is the
only cache publication authority.

1. Authenticate generator then input and its stored policy, validate P/G and
   `P+G<=2`, eval/profile/model state and any required categorical draw capacity.
   Allocate the rooted Eshkol output envelope, ID/text/RNG capacity and cleanup
   ledger before initial prefill publication. Rejection before call acquisition
   preserves the entry cache, binding, RNG and old outputs; no pin is acquired.
   Greedy, G0 and manual calls accept an authentic exhausted RNG without a draw.
2. Acquire the shared call and fourteen canonical pins; reserve its native
   output against that active call before prefill. Begin an unpublished candidate
   cache and run all 21 prefill roles. End views and preflight old cache
   destruction. Initial prefill commit replaces cache/binding. For G0,
   prepare empty IDs/raw text and unchanged RNG, preflight, commit, finish and
   publish the output with cache length P. A prefill failure keeps the prior
   cache/binding/RNG; no output escapes.
3. For P1/G1, leave the output pending after the initial prompt commit. Sample
   exactly once from the true f32[1,256] last logits. Greedy takes no block;
   categorical stages one successor G3 Philox block, including singleton or
   EOS. Run the complete generated-token frame and stage matching K/V at
   absolute position one. A sample or append failure aborts/scrubs the tail,
   destroys pending output, retains prompt-only cache/binding and old RNG, and
   raises without a partial output.
4. Stage ID, lengths, successor RNG and raw byte. Validate all 256 logits and
   the raw T1 byte decode before text acceptance. `output_prepare`, decode-ID
   copy, text acceptance, `frame_prepare` and `call_prepare_end` complete before
   final publication. The no-failure `frame_commit` atomically publishes
   cache/ID/RNG/output; `call_finish` immediately drains pins and active tokens,
   then Eshkol seals the already-rooted output. Emitted EOS follows this same
   append and is present in ID, text and cache. No allocation, callback or
   recoverable error is allowed between final native commit and finish.

All cuts must be injectable in a development-only closure, absent from the
installed archive: malformed/foreign/dead input; P2/G1 and length>2 before
pins; output shell/I1/RNG/text allocation and every accessor wrapper/list/
bytevector/native-clone allocation; each of 21 prefill and append roles;
A2 creation, stage/view/close and old-cache preflight; sampler nonfinite and
required-draw exhaustion; ID/raw-byte mismatch; once-only prepare/finish;
active borrow and stale model binding. At each cut assert exact cache, binding,
RNG, output registry, pin and guard state, then retry. Compare all 256 logits
and both K/V positions with the accepted no-cache/reference witnesses. Repeat
normal and instrumented runs; sanitizer, exact package manifests, private-link
negatives and AOT public callers are gates before public acceptance.

## Public error and release boundary

Each facade authenticates its receiver/liveness first, then arguments left to
right, syntax/scalars, metadata rank/extents/dtype/device/layout, byte values,
profile/provider/model/cache state and numerical preflight. For release, exact
identity authentication precedes a dead-owner idempotent return, then busy/borrow
preflight. Errors are E1 with the **invoked public operation**, bounded data-only
source domain/category/code/message and `cause #f`; no pointer, raw hidden owner
or earlier private operation leaks. The accepted G3-T bounded native mapping
applies: wrong-kind/forgery/config and scalar token range `invalid-argument`;
dead/busy/stale and required categorical draw exhaustion `invalid-state`;
T1 tensor shape/ID and context shape `shape-mismatch`;
dtype/device/layout their dedicated categories; unavailable
profile/provider `unsupported`; explicit FP determinism failure
`determinism-unavailable`; allocation/invariant `internal`. Output storage
inconsistency preserves A0 `device-mismatch`. Snapshot the first native error
before cleanup, then normalize at the outer facade. No recoverable E1 error may
follow a successful joint commit; impossible postcommit defects fail-stop.

This proposal does not change the accepted G3-T/N/S ABI or public predecessor
arity. Review must freeze the fifteen wrapper mappings, exact package closure,
authentic T1 constructor and detached clone ownership before dependent code.
