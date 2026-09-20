# G3 independent cache and lifetime review

Status: proposed only. This review changes no API, ABI, serialized format, or
capability claim. It audits merged M3/M3T, A2, I1/I2, T1 and P1 for issue #89.
The smallest candidate below is a diagnostic prerequisite for G3, not completion
of the roadmap generation workstream. No builds were run by this reviewer.
Historical review of `c354cb2`; [binding decision 5748532295](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5748532295)
now settles its high-level choices, including all six additions with independent
RNG release. Exact [G3-T seam planning](G3_T_SEAM_PLAN.md) remains separately gated.

## Findings from the implementation

| Existing contract or code | Consequence for generation |
|---|---|
| `native/m3_model_extension.esk:m3-public-model-forward` always runs the 21-role schedule and captures a graph; `src/eshkol_transformer/m3_model.c` clones 14 saved parameters, IDs and logits. | Eval mode is not a no-grad path. Calling M3 forward and discarding its graph does not establish the proposed no-grad ownership contract. |
| `src/eshkol_transformer/m3t_transport.c` has fixed IDs `[1,2]`, token activations `[1,2,4]`, logits `[1,2,256]`, positions `{0,1}`, and mask bytes all one. | Neither single-token prefill nor append positions are reachable through the existing public transport. Fixed T2 execution cannot silently stand in for T1 by padding. |
| M3T workspace begin snapshots all 14 canonical parameter values and mode; one model has one active frame; exact-primals checking includes value bits, canonical identities and constants. | A generation successor can reuse the reviewed identity discipline, but must introduce its own persistent cache binding and no-grad execution contract. |
| A2 owns separate K/V `[L,N,Hkv,C,Dh]`, lengths `[N]`, mask `[N,C]`; transactions stage every layer and expose full-capacity views. | Its native transaction is usable infrastructure, but it is not an Eshkol generator or an I2 tensor owner. Cache views must stay A2-owned. |
| A2 commit validates all layers and absence of nested views before publishing lengths/mask; abort scrubs staged tails to positive zero. | Stage/cache cleanup must finish before a generator commit point. Generator RNG, output staging and cache commit still require a coordinated prepare/commit protocol. |
| A2 cache documentation requires post-RoPE keys; M3 has learned positions and no RoPE. | Accept an explicit bounded clarification admitting keys from the model's declared position transform, with learned-position K unchanged after projection, or require a distinct cache contract. Do not apply RoPE to M3 or silently reinterpret the existing prose. |
| P1 module registration borrows parameters for module lifetime and has no model-release API. | G3 retains the genuine model; closing G3 must never destroy parameters or claim to reclaim the model. |
| T1 tokenizer, policy and successful encoded-ID shells are held by strong append-only registries. T1 decode accepts only its own same-aggregate sealed rank-one shell. | General I1 tensors, G3 tensors, M3T inputs and copied/tagged T1 shells are not interchangeable. Repeated ephemeral T1 encoding is not bounded-memory ingress. |
| M3 output/logits release frees payloads but retains identity history; I2 has retained dead control shells. | Payload release does not prove flat process memory. G3 needs separate retained-shell/arena/native payload accounting. |
| A0 defines nine generation names, no generator/output/tensor release names. | A release design is a pre-freeze decision, not something implementation can invent privately while exposing owned public tensors. |

Sources: `docs/M3_COMPOSITION_PROPOSAL.md`, `docs/M3T_TRANSPORT_PROPOSAL.md`,
`docs/A2_ATTENTION.md`, `include/eshkol_transformer/a2_kv_cache.h`,
`native/a2_kv_cache.c`, `docs/I1_I64_TENSOR.md`, `docs/I2_F32_TENSOR.md`,
`docs/TOKENIZER_FORMAT.md`, `docs/P1_MODULE_STATE.md`, and A0 sections 4/13.

## Smallest bounded successor to disposition

Keep the existing model's CPU-f32 learned-position N1/V256/D4/Hq=Hkv2/Dh2/L1/F8
architecture and 14 parameter tensors. Fix cache capacity at C=2. Admit prompt
length P in `{1,2}`, absolute positions starting at zero, and
`0 <= max-new-tokens <= 2-P`. Require eval mode at generator creation and each
operation; reject train mode without changing it. No dropout, padding, batching,
position offset, sliding window, growth, eviction, mixed precision, accelerator,
or cast/shape/scalar fallback is admitted. This candidate can demonstrate at most
one emitted token; it cannot prove multi-token continuation, later-token failure,
batched EOS behavior, or full G3 acceptance. A C4 successor would need a separate
position-table/initializer/profile decision and additional exact numerical rows.

Private carrier shapes for this candidate:

| Value | Exact shape and ownership |
|---|---|
| Prompt | borrowed authenticated CPU i64 `[1,P]`, P=1 or 2; copied to generator staging before execution |
| Decode input | borrowed authenticated CPU i64 `[1,1]`; token in `[0,256)` |
| Full-sequence reference execution | no-grad f32 logits `[1,S,256]`, S=1 or 2, independently owned scratch; no graph or gradient metadata |
| Public last logits | newly owned immutable CPU f32 `[1,256]`; separate payload from workspace/cache and any other result |
| Cache K/V | two A2-owned f32 `[1,1,2,2,2]` allocations; distinct identities |
| Cache metadata | A2-owned i64 `[1]` lengths and bool `[1,2]` mask |
| Per-call Q and staged K/V | f32 `[1,2,A,2]`, A=P for prefill and A=1 for decode |
| Position operands | exact i64 query `[1,A]` and key `[1,2]`; canonical key positions `{0,1}` even under false tail mask |
| Attention mask | exact bool `[1,A,2]` materialized from A2 validity mask; no implicit broadcast |
| Output | immutable owned generated IDs i64 `[G]`, lengths i64 `[1]`, cache lengths i64 `[1]`, decoded raw bytes and final RNG snapshot; `G=0` or 1 |

All attention rows must resolve through K1, including full-capacity Tq=1/Tk=2
for prefill P=1. No pretending that the capacity-strided cache is a dense shorter
tensor. The independent numerical review owns the exact missing N2/N3K rows;
acceptance of a schema-valid direct provider call is not capability support.
The independent full-sequence path must share model parameters and position
semantics but execute without cache; compare every exposed position, not only
the selected token. Numerical work remains an Eshkol schedule over reviewed
kernels. C may copy, allocate, authenticate, materialize exact metadata and manage
leases; it may not become an unreviewed numerical implementation.

## Operations and commit points

`generator-create/3` validates the exact model/tokenizer identities and profile,
V256, raw byte decode policy, context, sampling policy and explicit RNG authority.
It retains model/tokenizer references and a detached configuration. It publishes
an idle generator only after all allocation and validation succeeds. It does not
consume RNG, create gradients, change mode or mutate model/inputs. Initial cache
is absent. A private in-call guard excludes reentrancy and shares the M3 model
active-frame exclusion. Mutable use is serialized throughout.

`generator-prefill!/2` admits P=1 or 2. Stage a completely new candidate cache,
model snapshot binding, prompt, positions, logits result and cleanup resources.
Execute all layers against the candidate, close every view, and validate all
commit eligibility. Publish the candidate and newly owned `[1,256]` result in one
infallible tail; dispose the previous cache only under prevalidated exclusive
ownership. Any failure leaves the previous cache/binding and RNG unchanged and
destroys the candidate. Successful prefill replaces any prior context and never
consumes RNG. A prefill may intentionally establish a new binding after model
values changed between calls; stale existing cache never authorizes decode.

`generator-decode-step!/2` requires a committed cache and `length < 2` before
numerical work. Check current canonical parameter identities, bitwise parameter
values, model topology/profile and eval mode against the cache binding. Exact
restoration may pass, matching M3's existing value-based semantics. Stage the
token at absolute position `length`, compute the returned next-token logits,
prepare the result and end every nested view. Then commit cache length and
publish result in one infallible tail. On any precommit failure, abort/scrub the
tail and preserve committed prefix, length, RNG and previous outputs. This manual
operation appends the caller's token regardless of configured EOS and consumes
no sampling RNG. A successful decode at length 1 returns logits at position 1
even though no further append is possible.

`generator-generate!/2` always starts by the same atomic prefill semantics; it
does not continue an existing context. Validate the full request and capacity
budget before replacing the old cache. With zero new-token budget, successful
prefill still replaces the cache; return an empty generated-ID tensor/list,
length zero, empty raw decoded bytes, cache length P, and unchanged RNG. Do not
special-case it into a no-op or allow an empty prompt.

For a nonzero budget, sample from prefill's last logits into private staging,
then run the sampled token through the model and stage its K/V. A0 requires an
emitted token, including EOS, to be represented in cache length; sampling alone
is not the commit point. Preallocate all final output storage and text capacity
before the first emitted token, and prepare the immutable successor RNG before
committing its associated token. End leases and preflight cleanup before commit.
Commit cache append, generated-ID prefix, length metadata and RNG successor as
one no-failure publication tail. Never return a sampled token whose K/V append
failed, and never advance RNG on a failed token attempt.

A0 promises per-token atomicity, not whole-call rollback. A failure after
successful prefill but before the first token leaves the prompt-only cache and
original RNG. A later-token failure in a future larger profile must retain the
last committed cache/RNG and raise without a partial output. Do not attempt to
restore the old pre-generation cache. C2 cannot exercise that later-token case;
the larger-profile gate must. Root should explicitly ratify prefill as the
initial generation commit point because A0 specifies atomic prefill separately
but leaves this composition edge implicit.

Raw V256 byte decoding has a useful bounded property: each generated ID maps to
one byte, so detached output bytes and list storage can be staged at the maximum
length before token commits. Strict UTF-8 or special-token decode errors would
introduce a separate post-generation failure surface; they are outside this
candidate. An optional explicit numeric EOS ID in `[0,255]` or `#f` avoids
inventing a named special absent from the baseline tokenizer. Include the EOS ID
and its actual byte in IDs/text/cache length. A prompt ending in that byte does
not stop generation; only newly emitted tokens trigger stop. Named-special EOS
or omit/error semantics require a separate tokenizer/model-vocabulary decision.

## Lifetime seams requiring acceptance

The generator owns cache, persistent model-binding snapshot, reusable workspace,
committed prompt/history metadata and one current RNG authority. It borrows
inputs only synchronously. Each returned result must survive later prefill,
decode, generation, generator close and model mutation. Outputs must own or lease
immutable snapshots, never views into cache/workspace. Each accessor returns the
new detached object required by A0; caller mutation of lists/bytes cannot alter
other accessors or the generator. No output retains an analytic graph or gradient
destination. Sampling RNG is separate from M3T initializer authority and cannot
be forged from inspected initializer words or C2 metadata lists.

The preferred disposition is explicit release for generator, output, returned
last logits and every newly owned ID/length tensor, with authenticated tombstones,
idempotent release for exact issued identities and `invalid-state` on subsequent
non-release use. Closing a generator frees its cache/scratch and releases its
references; it never frees the borrowed P1 model or process-lifetime T1 tokenizer.
Do not claim that finalizers or loss of Eshkol reachability perform release.

Additive **candidate names only**, for root to consolidate before any ABI freeze:

- `generation-input-create/1`: same-aggregate sealed T1 i64 `[P]` to new owned
  CPU i64 `[1,P]`, P=1 or 2, copying values; no retained T1 lease.
- `generation-token-input-create/1`: one exact integer `[0,255]` to new owned
  i64 `[1,1]`, providing manual decode ingress without per-token T1 encoding.
- `generation-tensor-release!/1`: closed union of these input owners, generator
  last-logits results, and generation ID/length accessor tensors; exact kind
  authentication; no generic I1/I2/native-pointer release authority.
- `generation-output-release!/1` and `generator-close!/1`: release independent
  owners, with sibling accessor snapshots surviving output release.

This is five proposed additions to the nine A0 operations. The independent RNG
snapshot returned by `generation-output-rng/1` must also survive output/generator
release. Its final carrier decision may require a sixth addition,
`generation-rng-release!/1`, or a reviewed caller-region representation with no
native payload; process-lifetime RNG identity enrollment would instead require
explicit bounded-retention acceptance. Output-owned RNG that becomes stale when
the output is released is not an adequate independent snapshot contract.
A different admitted
closed-carrier lifecycle is possible; leaving caller-owned tensor lifetime
implicit is not. Repeated input-create from one reused encoded prompt is bounded
in T1 enrollment but still requires explicit input release and measured shell
retention. Repeated `tokenizer-encode` remains process-lifetime retention.

For `generation-output-text`, add a reviewed source-private T1 decoder seam that
borrows already-authenticated G3-owned rank-one I1 IDs for one synchronous decode
under the generator's retained same-aggregate tokenizer. It must validate range,
return detached raw bytes, release all borrows on every path and make **no** T1
sealed-shell enrollment. It accepts neither arbitrary native pointers nor a
caller-selected decoder/provider. Root should prefer this to creating a permanent
T1 shell for every generated result. Existing `tokenizer-decode/2` remains
unchanged and continues rejecting G3 tensors. An alternative dual-shell admission
requires its own exact ownership/release proof before use; retagging is invalid.

## Serialization and prerequisite order

No A2 cache handle, memory address, live view, graph, callback or generator is
serializable. Preserve T1/T2/X1/C1/C2 formats. Save/reload generation equivalence
must restore accepted model data and tokenizer/config identity, restore an
accepted explicit sampling RNG state, then reconstruct cache by no-grad prefill
from retained prompt/generated IDs; it must not reload cache bytes. Replaying
tokens by manual decode consumes no sampling RNG. C2's current public rank-zero/
rank-one copy capability does not admit M3 rank-two parameter reload, and its
data-only RNG fields do not themselves create live generator RNG authority.
Those are explicit prerequisites, not permissions to extend C2 incidentally.

Suggested independently reviewable work IDs (not roadmap status claims):

1. **G3-C2-N**: exact S1/no-cache and A1/cache numerical rows, plus learned-position
   cache interpretation; no broader rectangular shape advertisement.
2. **G3-C2-T**: closed S1/S2 no-grad Eshkol schedule, exact input/output carriers,
   finite checks, position/mask transport and cache-bound model identity.
3. **G3-C2-L**: generator/output/tensor lifetimes, T1 private decode borrowing,
   cache/RNG/output atomic publication and explicit release candidates.
4. **G3-C2-R**: separately authenticated sampling RNG, fixed successor preparation,
   exhaustion/rollback behavior and deterministic sampling contract.
5. **G3-C2-P**: one source-composed registry owner containing these reviewed seams;
   installed public facade/private-symbol and cross-aggregate rejection gates.
6. **G3-SAVE**: existing-format model rank-two reload plus an accepted RNG
   construction/restore route; deterministic reconstruction evidence.
7. **G3-C4** or a separately named broader profile: multi-token continuation and
   later-token failure evidence before full G3 acceptance.

G3-C2-N precedes T; L and R contracts may be reviewed independently but their
joint commit design precedes implementation of generation. P depends on all
runtime predecessors. SAVE can be separately coordinated; C2 diagnostic evidence
does not discharge it. No dependency on unmerged L3S is introduced.

## Required evidence and disposition questions

Tests must cover old-output survival across replace/close, exact parameter mutation
and restoration, active M3 frame exclusion, forged/cross-registry/wrong-kind/dead
receivers, release idempotence, zero budget, P2 exact capacity, append overflow,
malformed rank/dtype/device/ID/position/mask, finite validation, source aliasing,
every allocation failpoint, every staged-layer/view failure and cleanup retry.
Record exact unchanged cache prefix/length/storage IDs and RNG on rejection.
Prove no gradient buffer/count change even when nonzero gradients already exist.
Require ASan/UBSan/LSan, detached-accessor alias tests and retained arena/native
payload/control-shell counts at 1,024 and 8,192 create/release or reuse cycles;
RSS alone is insufficient. Full-sequence/cache parity uses an independent oracle,
and save/reload compares logits, IDs, bytes, lengths and RNG bitwise on the
supported deterministic platform. Larger-profile tests must cover failures after
multiple committed tokens and EOS at every possible step.

Root decisions required: accept the C2 diagnostic limit separately from G3;
ratify learned-position A2 cache interpretation; accept numeric-byte EOS/raw decode;
ratify prefill commit and zero-budget behavior; choose explicit release/ingress
names and the private T1 decoder borrowing seam; accept exact-value cache binding
with exact restoration; select the separate model/RNG reload prerequisite.
None of these decisions are claimed by the existing runtime.
