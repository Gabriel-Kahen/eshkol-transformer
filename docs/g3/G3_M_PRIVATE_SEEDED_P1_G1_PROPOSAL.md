# G3-M private seeded P1/G1 schedule proposal

Status: **pending independent review; contract only**. This draft is based on
reviewed private composer `86bd5a5` and the existing G3-T generated call in
`src/eshkol_transformer/g3t_transport.c`. It proposes one private Eshkol
operation, `(g3m-generate-p1-g1! generator input)`, arity two. It adds no native
ABI, installed G3-G facade, package export, loop or N>1 claim. The generator
may have been created from a seed or an accepted independent RNG owner; this
operation neither constructs nor returns a generator.

## Admission and scope

Require an authentic live G3-T diagnostic-C2/M3T generator configured with
budget `1` and an authentic live G3-T kind-2 input with exactly one byte ID.
The existing T1-backed or owned-token constructors supply distinct CPU
i64 `[1,1]` storage; the pair constructor's inline-only P2 input is
ineligible. `g3t-generator-find` authenticates shell identity;
`g3t-call-acquire` checks generator/model liveness. Under one outer `m3-call`,
authenticate generator kind/liveness and input kind/liveness, then require
budget `1` from the authenticated generator policy (private
`invalid-argument`/code `3` on a budget mismatch). Bind the **existing** native
`et_g3t_private_full_request_preflight_v1(context,input)` in the private
Eshkol closure and call it before acquiring a call or pins. It checks stored
P/G, `P+G<=2`, byte IDs, idle/no committed prefill, and required categorical
draw capacity (`g3t_transport.c:605-632`). The native frame owns length
validation; no shell field or caller token value supplies P. Use the native
error fields on rejection. The separate existing P1/G0 and P2/G0 operations
remain the zero-budget routes.
No generation continuation after manual P1/P2 prefill, manual decode, P1/G0,
or a prior G1 commit is admitted: native generated preflight rejects an
already committed prefix. Manual `g3m-prefill-p1!` and `g3m-decode-one!`
demonstrate numerical and A2 seams but are not called by this G1 transaction.

## One generated-token transaction

Install a rollback guard before `g3t-call-acquire(generator-entry,2)` and
reserve the pending output with `g3t-output-reserve(call,1)`. This roots the
wrapper, I1[1] ID owner, eight-byte ID staging and one-byte raw text capacity
before any prompt commit (`g3t_prefill_sample_extension.esk:198-220`). Begin
`g3t-frame-begin(call,input,1)` and run ordered roles `0..20` once; this is
the same accepted T1 numerical route as manual P1, but under generated kind 2.
`g3t-frame-prepare(call,#f)` then `g3t-frame-commit(call)` commits the P1 A2
cache, exact model binding and all 256 last-logit words while the output stays
pending and RNG stays unchanged (`g3t_transport.c:2250-2279`). Do not finish
or expose an output at this intermediate point.

Call `g3t-sample(call)` exactly once on those true f32 `[1,256]` logits. The
reviewed G3-S greedy route selects max/lowest ID without a draw and copies the
RNG; categorical stages exactly one Philox successor, including singleton or
EOS, but changes neither generator RNG nor output yet (`g3t_transport.c:2313-
2370`, `G3_S_ABI_PROPOSAL.md`). Begin the generated frame with
`g3t-frame-begin(call,#f,2)`: native copies the sampled ID into its frame,
without constructing or borrowing a separate public token input. Run ordered
T1 roles `0..20` again; role 10 stages the A2 append at absolute position 1.
Then call `g3t-output-prepare(call,output)` to copy the sampled ID to its
owned I1[1], set generated length `1`, cache length `2`, and stage successor
RNG. Call `g3t-t1-decode-output!(call,output)` to copy the eight-byte ID into
rooted staging, decode the raw byte using the retained T1 tokenizer, and
accept matching text. Finally call `g3t-final-commit!(call,output)`, which
performs frame prepare, cleanup preflight, atomic A2/ID/RNG/output commit,
and immediate nonrecoverable finish without a fallible interposed action
(`g3t_final_publication_extension.esk:24-54`). Return only its live opaque
output. The sampled byte is included in ID, raw text and cache even when it
equals configured EOS; budget one or EOS stops after this single append.
Prompt EOS has no stop effect. There is no second sample or speculative
second frame.

## Ownership, failures and proof gate

The generator owns its A2 cache, binding and mutable RNG; the caller retains
the input. A successful output owns its I1 ID, raw byte and final RNG words,
with generated/cache lengths. It survives generator close; existing private
accessors return independent ID/length/cache-length/RNG clones and copied
raw text. Typed output, tensor and RNG releases remain distinct. Greedy, G0
and manual calls preserve even an exhausted RNG;
categorical G1 must reject exhausted draw state before prompt publication.

Before the first prompt commit, any exception aborts the acquired call and
preserves the entry cache/binding/RNG and older outputs. After that commit,
sampling, append, A2, ID or text failure aborts the pending output and staged
tail, retaining the new prompt-only cache/binding and old generator RNG; there
is no partial generated output. The native final preflight checks matching
owned ID, raw byte, successor RNG, length/mask and binding before the no-failure
commit/finish tail (`g3t_transport.c:2026-2311`). Native abort/finish failure
or an impossible postcommit exception is fail-stop; preserve the first
recoverable native error through cleanup and the outer operation name in E1
normalization. No callback, scalar model forward, AD graph, alternate sampler,
or public token staging is introduced.

Implementation requires a private source file, exact local-symbol/closed
source manifest and checker fixing preflight-before-acquire, one outer guard,
two explicit 21-role schedules, one sample, readiness and final-tail order.
Run genuine seed and RNG-source greedy/categorical P1/G1 normal/repeat/
ASan+UBSan+LSan, with independent M3T parity for all 256 P1/append logits
and both A2 K/V positions; exact G3-S sample/RNG oracle; ID/text/length/
cache-length/RNG clone and detached-lifetime checks. Cover EOS equality and
inequality, byte 0/255, exhausted draw and P2/G1 pre-pin rejection, malformed/
dead/borrowed owners, stale binding, all meaningful provider/A2/ID/T1/text
cuts, old-cache borrow and retry, plus P1/P2 manual, manual decode, P1/G0,
P2/G0 and inherited P1/G1 regressions. Record reached cuts, pin/guard drain,
production feature-off symbols, Q0 and sealed exact-tree evidence. Public
G3-G error wrappers/package, cross-operation continuation, persistence and
N>1 remain separate dependency gates.
