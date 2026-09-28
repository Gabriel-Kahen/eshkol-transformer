# G3-M private manual decode composer proposal

Status: **accepted for bounded private implementation** after independent
source review. This is the next
bounded private consumer of the reviewed G3-T native manual decode source
`a4f7c53` and Eshkol wrapper source `f117987` (tree `972087b`). It adds no
implementation, native symbol, public generation facade, sampler call, draw,
or repeated append beyond the capacity-two `1→2` route. The accepted
[native transport](G3_T_MANUAL_DECODE_TRANSPORT_PROPOSAL.md) and
[wrapper](G3_T_MANUAL_DECODE_WRAPPER_PROPOSAL.md) contracts retain admission,
numerical, A2 and publication authority.

## Proposed source-private operation

Add only `(g3m-decode-one! generator input)`, arity two, alongside the
existing G3-M P1/P2 composers. The generator must be an authentic live G3-T
C2/M3T owner with one committed prefix position, from either
`g3m-prefill-p1!` or the accepted P1/G0 route. The input must be an authentic
live kind-2 owner containing unborrowed canonical CPU i64 dense `[1,1]`
storage, eight bytes and byte ID `0..255`. Existing
`g3t-generation-token-input-create` or one-token
`g3t-generation-t1-input-create` supplies it; no new input constructor,
inline-only pair, null input or shell-inferred shape is admitted. G3-M uses
`g3t-generator-find` for shell identity and checks input kind and liveness
before acquisition. `g3t-call-acquire` checks generator/model liveness before
publishing the call; the reviewed native `frame_begin` proves typed I1 bits,
inline agreement,
binding and exact A2 prefix (`g3t_transport.c:1473-1562`). A missing prefix,
full length-two cache, stale binding, borrowed or malformed input rejects.
The caller retains input ownership; native copies the token during admission.

Use exactly one outer `m3-call` and the existing P1/P2 guard pattern
(`g3m_prefill_p1_extension.esk:3-48`,
`g3m_prefill_p2_extension.esk:3-46`). Install a `call=#f`/
`committed=#f` rollback guard before `g3t-call-acquire(generator-entry,1)`
takes fourteen parameter pins. Then call `g3t-logits-reserve(call)`,
`g3t-frame-begin(call,input,2)`, and exactly one explicit ordered
`g3t-role-step(call,n)` for each integer `n=0..20`. Do not add a callback,
runtime role loop, scalar numerical route or T2 role selection. Native role
0 uses the copied ID, role 1 uses learned position one, role 10 appends via
A2, and the other roles follow the reviewed T1 path
(`g3t_prefill_roles.inc:213-252`).

Finish with `g3t-frame-prepare(call,logits)`,
`g3t-call-prepare-end(call)`, `g3t-frame-commit(call)`, set
`committed=#t`, and immediately `g3t-call-finish(call)` with no allocation
or recoverable action in between. Return only the detached kind-3 owned CPU
f32 `[1,256]` last-logits shell. Native preflight proves the result bits,
both K/V positions, lengths rank-1 `[1]` value `2`, mask rank-2 `[1,2]`
values `[1,1]`, and unchanged binding before atomic publication. The cache
remains the same A2 owner; RNG, model parameters and older detached results
remain unchanged. The result outlives the call and generator until its
typed release. This operation emits no generated token or text.

## Failures and implementation gate

Preserve the P1/P2 error order: outer guard, generator lookup, input registry
kind/liveness, call acquisition, pending-result reservation, then native
frame/role/preflight checks. Do not replace native domain/category/code
from I1, F32, A2 or providers. If acquisition fails, there is no linked call
to abort. Every later precommit exception aborts exactly the acquired call,
drains pins, kills pending logits, retains the length-one prefix/binding/RNG
and older detached outputs, then re-raises the original exception. A failed
role is retryable at the transport level; G3-M aborts the failed call, so
retries use a fresh call. A borrowed pending result can block native abort;
because the composer never exposes that result before finish, its normal
failure path cannot create such a borrow. Native abort/finish failure and any
impossible postcommit exception are fail-stop, as in the existing wrappers.

Implementation should add one private Eshkol file and exact local-symbol list,
then extend the reviewed wrapper source closure with that file, its focused
checker and this contract/leaf evidence. The checker must fix the one outer
guard, kind-1/frame-2 route, ordered 21 explicit calls, and adjacent
commit/finish tail. A genuine operation-level witness must run after both P1
manual and P1/G0 predecessors, compare all 256 final-logit words and both
committed K/V positions bitwise with independent M3T, check length/mask,
unchanged RNG, older-result retention, input release, detached result after
parent close, and fresh-call repeat behavior. Cover forged/wrong-kind/dead,
P2/inline-only, borrowed/malformed input, no/full prefix, busy context, stale
binding, F32 allocation, provider/A2 cuts and rollback. Preserve P1/P2
manual and P1/G0, P2/G0, P1/G1 regressions. Run Q0, closed source checker,
pinned network-disabled normal/repeat/ASan+UBSan+LSan aggregate, and
production feature-off symbol inventory; seal exact commit/tree and logs.
Only after independent source/test review may downstream private seeded
generation work proceed; this proposal installs no public G3-G facade.
