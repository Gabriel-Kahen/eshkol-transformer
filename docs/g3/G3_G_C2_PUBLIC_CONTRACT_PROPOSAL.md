# G3-G public diagnostic-C2 P1/G1 proposal

**Proposed for independent review; no public implementation or package is
accepted.** This is one installable slice of the [A0 generator
contract](../PUBLIC_API_CONTRACT.md#13-generator), not its full generation
API. It consumes the reviewed private
[P1/G1 schedule](G3_M_PRIVATE_SEEDED_P1_G1_PROPOSAL.md) and its
[implementation leaf](G3_M_PRIVATE_SEEDED_P1_G1_LEAF.md). The source-private
[P1/G0](G3_T_ZERO_BUDGET_LEAF.md),
[P2/G0](G3_T_P2_ZERO_BUDGET_LEAF.md), and
[manual decode](G3_M_MANUAL_DECODE_COMPOSER_PROPOSAL.md) routes establish
separate numerical and transaction behavior; this facade calls none of them.
There is no public prefill/decode operation, G0 output, P2 prompt, P2/G1,
continuation after any committed prefix, second sample, N>1, persistence, or
CLI in this proposal.

## Exact proposed facade and package boundary

Install one `lib/transformer/generation.esk` facade, with these thirteen names
and arities only. Each maps to an existing source-private G3-T/G3-M operation
or owner; the public wrapper and package mapping remain to be implemented.

| Name | Arity | P1/G1 behavior |
|---|---:|---|
| `generator-create` | 3 | Authentic model, tokenizer and diagnostic-C2 config; live generator with private cache and RNG, configured for budget exactly one. |
| `generator-generate!` | 2 | One authentic live P1 G3 input; run the private `g3m-generate-p1-g1!` transaction and return one live opaque output. |
| `generation-input-create` | 1 | Copy a same-aggregate sealed T1 rank-one one-byte ID into a distinct owned CPU i64 `[1,1]` input. |
| `generation-token-input-create` | 1 | Copy one exact byte ID into a distinct owned CPU i64 `[1,1]` input. |
| `generation-output-ids` | 1 | Fresh one-element list containing an owned CPU i64 `[1]` generated-ID clone. |
| `generation-output-lengths` | 1 | Fresh owned CPU i64 `[1]`, value `1`. |
| `generation-output-text` | 1 | Fresh one-element list containing a detached raw one-byte bytevector. |
| `generation-output-rng` | 1 | Fresh independent immutable G3 RNG owner with the post-generation words. |
| `generation-output-cache-lengths` | 1 | Fresh owned CPU i64 `[1]`, value `2`. |
| `generation-tensor-release!` | 1 | Typed G3 input or ID/length/cache-length clone release. |
| `generation-output-release!` | 1 | Typed output release. |
| `generation-rng-release!` | 1 | Typed independent RNG snapshot release. |
| `generator-close!` | 1 | Close generator cache, scratch and owned RNG; retained model/tokenizer stay caller-owned. |

The public constructor accepts only the already accepted diagnostic-C2/M3T
profile, CPU f32 model, authentic retained T1 byte tokenizer, exact policy and
seed or independently owned RNG source. For this facade, its configured budget
must be `1`; a budget `0` or greater than `1` is `invalid-argument` before a
live generator is published. It does not silently route budget zero to P1/G0
or P2/G0. A valid optional byte EOS is accepted but cannot suppress the one
append: if the sampled byte equals EOS, it remains in IDs, text and cache.
Greedy chooses the lowest ID on a max tie and consumes no draw; categorical
sampling consumes exactly one G3-S Philox block, even for singleton or EOS.
The accepted G3 configuration, capability, tokenizer identity, model-eval and
sampling bit contracts are unchanged.

The proposed sibling archive is one `libeshkol_transformer_g3g.a` with one
registry-owning `g3g_package.o`. It source-composes the existing M3/M3T/I2/P1/
T1/E1/G3-T/N/S closure; no second owning archive is linked alongside it. It
installs exactly the existing seven M3 facades plus
`transformer/generation.esk`. The measured M3 baseline is 93 globals, 87
boxed exports and 93 public-name strings. The proposed boundary appends one
boxed export and one public name per table row: **eight facades, 100 boxed
exports, 106 globals and 106 public-name strings** as a target to verify,
not an observed artifact. Candidate files remain
`native/g3g_package_{root.esk,bridge.c,private_renames.txt,public_exports.txt}`
and `scripts/build-g3g.sh`; no such public package is present yet. The
proposed private/public boxed symbol pair is
`et_e1b_private_g3_<stem>_cabi_v1` /
`et_e1b_public_g3_<stem>_v1`, for these thirteen stems in table order:

`generator_create`, `generator_generate`, `generation_input_create`,
`generation_token_input_create`, `generation_output_ids`,
`generation_output_lengths`, `generation_output_text`,
`generation_output_rng`, `generation_output_cache_lengths`,
`generation_tensor_release`, `generation_output_release`,
`generation_rng_release`, `generator_close`.

The existing native G3-T/N/S/A2 helpers, registries, role helpers, private
composer and testing hooks remain localized. Exact source, object,
undefined/defined-symbol, string and AOT manifests must be measured on the
candidate. The two deferred A0 operations `generator-prefill!` and
`generator-decode-step!` get no wrapper, boxed export or installed name here;
adding them later requires a separately reviewed package version/tuple.

## Admission, transaction and publication

`generation-input-create` authenticates the sealed same-aggregate T1 shell
through its registry, proves rank one/length one and byte `0..255`, then uses
the accepted private T1 input constructor to copy into a distinct G3-owned
I1 `[1,1]`. It retains no additional T1 borrow; the T1 shell retains its own
internal borrow. A T1 length two, copied/tagged/foreign shell, output clone,
M3T input or shape-compatible generic tensor cannot acquire public input
authority. `generation-token-input-create` accepts only an exact integer byte
`0..255`; no inexact coercion. Both constructors publish a live wrapper only
after the native owner is complete. The caller retains and separately releases
the G3 input.

`generator-generate!` authenticates the generator shell and input kind/live
identity left to right. It rejects budget mismatch as `invalid-argument`/native
code `3` before the existing full-request preflight. That native preflight
reads the authenticated input's stored P/G/ID and policy, and rejects a
committed prefix, busy call, `P+G>2`, malformed byte or unavailable required
categorical draw before call acquisition or model pins. The public wrapper
must not derive P from an Eshkol tag, caller tensor, test observer or old T1
value. The generated route copies its authenticated inline ID; an underlying
read-only I1 borrow is permitted, while typed input release rejects until
that borrow ends. This differs from manual decode's unborrowed typed-I1
admission. A generator with a prefix from manual P1/P2, P1/G0, manual decode
or an earlier G1 is ineligible; native preflight rejects it. No public reset
or cache replacement transition is implied.

On success, the wrapper delegates exactly once to the reviewed private
operation: one outer guarded M3 call; 14 canonical model pins; pending output
reservation; P1 frame with roles `0..20`; prompt-only A2 commit and true
last-row logits; one G3-S sample; generated frame with roles `0..20` at
absolute position one; ID/raw T1 text, lengths and successor RNG readiness;
final preflight; adjacent no-failure commit/finish. The existing private
output reservation roots its Eshkol envelope before prompt publication; the
public wrapper returns that already-live shell without postcommit allocation
or a fallible conversion. It never exposes intermediate logits or a pending
owner. It may not
run a scalar model path, a second sample or a different provider route.

A failure before prompt commit leaves the entry cache, binding, RNG and older
outputs unchanged. A sampling, append, A2, ID or text failure after prompt
commit aborts the tail and pending output, retaining only the new prompt cache
and binding with the old generator RNG. No generated output escapes. A
successful final commit publishes cache length `2`, one owned ID, generated
length `1`, raw byte and successor RNG together. Native abort/finish failure
or an impossible postcommit exception is fail-stop. The wrapper snapshots the
first recoverable native error before cleanup; cleanup must not replace it.
The private testing closure has a binding cut after sample, but the production
composer exposes no interposition hook for every ID/T1/text error. Existing
lower-level output/text and final-publication tests cover those cuts; the
public package gate must describe precisely which faults it can inject.

## Detached ownership, accessors and public errors

The generator owns cache/binding and mutable RNG; model and tokenizer remain
retained caller-owned references. The output is a distinct authenticated
owner with immutable ID, raw byte, lengths and RNG snapshots and no parent
call/model/tokenizer root after publication. It survives generator close and
later mutation. Each accessor authenticates the output and returns a new
independent payload. IDs, lengths and cache lengths use the reviewed native
clone stems and existing private Eshkol wrappers; RNG uses the reviewed RNG
clone; text copies the raw byte into a new bytevector and list. The accessors
return neither a cache view nor a native header. Mutating a returned list or
bytevector cannot affect the output or a later accessor.

Each accessor must root its future Eshkol shell, pending entry, registry cell
and, where needed, one-element list before the native clone request. Native
clone creation completes detached storage before owner enrollment. On a
recoverable failure, no live wrapper or orphan native clone remains and the
source output is unchanged. After clone success, publication is a no-allocation
tail; any remaining fallible wrapper step must release that exact new native
kind before re-raising the original error. Text similarly preallocates its
bytevector/list, copies, then publishes. This allocation ordering requires a
public-wrapper cut witness; private source checks alone do not prove it.

Input, output, tensor clone, RNG snapshot and generator have distinct
registry identities and release domains. Exact already-released identity is a
retained tombstone and repeated typed release is idempotent. Wrong kind,
forged/copy or foreign shell is `invalid-argument`; authentic dead non-release
use, busy receiver or active-borrow release is `invalid-state`. Typed tensor
release accepts only this slice's input and ID/length/cache-length clone
kinds; kind-3 manual last-logit ownership is private until its public facade
is separately proposed. Generator close leaves outputs, clones, independent
RNG snapshots, model and tokenizer alive. Dropping an Eshkol reference does
not release a payload.

Every facade raises E1 with the **invoked public operation**, bounded data-only
source domain/category/code/message and `cause #f`. It authenticates receiver
identity/liveness before later arguments; constructor scalar, T1 and model
validation follows the accepted native order. Wrong-kind/forgery/config and
scalar byte range map to `invalid-argument`; dead/busy/stale and categorical
draw exhaustion to `invalid-state`; T1 tensor shape and context extent to
`shape-mismatch`; unsupported profile/provider to `unsupported`; explicit FP
determinism rejection to `determinism-unavailable`; allocation/invariant to
`internal`. Dtype, device and layout preserve their dedicated E1 categories;
inconsistent retained output storage maps to A0 `device-mismatch`. Native
error domain/code remains authoritative. No raw pointer, private operation
name, hidden owner or previous error leaks. No recoverable E1 error follows a
successful joint commit.

## Acceptance and deferred work

A source-composed package candidate must fix the thirteen exact E1 mappings,
closed source/symbol/string manifests, one registry owner and eight installed
facades. Genuine T1-backed and scalar P1/G1 callers must prove all 256 P1 and
append logits plus both K/V positions against independent M3T, exact G3-S
sample/RNG, EOS equality/inequality, byte `0`/`255`, borrowed input/release
behavior, failure rollback and retry, detached output/accessor lifetime and
wrong-kind/dead/foreign owner rejection. Public-wrapper allocation cuts must
cover constructor/envelope/accessor preallocation and native-clone failure;
role/provider/A2 and post-sample binding cuts retain their existing private
witnesses. Run normal/repeat/ASan+UBSan+LSan, Q0, production feature-off
inventory, private-link negatives, fresh-cache AOT public callers, package
localization and supported CI on the exact package tree before acceptance.
This proposal neither claims those tests have run nor freezes a public ABI.

P1/G0, P2/G0 and manual P1/decode facades, cross-operation cache replacement,
P2/G1 capacity-three transport, N>1 and a general generation loop remain
separate contracts and gates. They cannot be inferred from this P1/G1
publication path.
