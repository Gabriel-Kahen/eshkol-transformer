# G3-T private manual P2 prefill transport

Status: implementation candidate pending independent review.
The accepted [manual P2 contract](G3_T_MANUAL_P2_TRANSPORT_PROPOSAL.md)
defines this source-private extension to G3-T's existing kind-0 native call.
It adds no public facade, new native symbol, G3-M orchestration, or manual
decode route.

`ET_G3T_MANUAL_P2_PREFILL_PRIVATE` requires the accepted manual P1, T2
generation roles and T1-owned input route. A live kind-2 input must own a
canonical CPU i64 `[1,2]` tensor. Native frame admission rejects the old
inline-only pair constructor, copies both authenticated IDs into the frame,
and releases the typed borrow before creating an unpublished A2 candidate.
The existing 21 T2 roles stage both K/V positions. Publication copies row one
of the 512-logit scratch to a detached `[1,256]` result and commits a
length-two cache only after full preflight. Precommit failure preserves the
old cache, binding, RNG and older detached results; the accepted abort path
drains the candidate, pending result and pins.

The focused aggregate proves P1-to-P2 and repeated P2 transitions, complete
M3T row-one logit and both K/V position parity, input release after frame
begin, malformed typed view and inline-staging rejection, candidate and
attention A2 cuts, a test-only simulated provider-dispatch rejection/retry,
borrow and binding failures, abort and ownership. It retains P1 and P2/G0
regressions. The pinned f31/LLVM21 network-none normal, repeat and
ASan+UBSan+LSan gate passed 46,984 byte-identical checks in each run, with
empty compiler/runtime stderr, 18 source contracts, Q0 4/4 and production
test-symbol exclusion. One sanitizer launch used `detect_leaks=1`. Evidence
is recorded at
`/home/gabe/.codex/evidence/eshkol-transformer/g3t-manual-p2-native-candidate-20260928`.
The synthetic dispatch cut tests error propagation and retry; it is not a
measured failure from an actual provider. This leaf does not implement G3-M
P2 orchestration or a public facade.
