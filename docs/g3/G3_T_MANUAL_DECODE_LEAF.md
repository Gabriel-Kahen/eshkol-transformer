# G3-T source-private native manual decode leaf

Status: isolated implementation candidate. The accepted
[manual decode contract](G3_T_MANUAL_DECODE_TRANSPORT_PROPOSAL.md) is the
authority. Under `ET_G3T_MANUAL_DECODE_PRIVATE`, native call kind 1 accepts
an authentic owned CPU i64 `[1,1]` input only after a bound length-one
prefix. It copies the ID and prefix K/V bits into the frame, releases both
borrows, runs the existing 21 position-one T1 roles and stages one A2 append.
The preflight checks detached f32 `[1,256]` logits, both K/V positions,
lengths `[1]` value 2, mask `[1,2]` values `[1,1]`, and unchanged binding.
Commit appends in place and detaches logits; abort scrubs the staged tail and
preserves the prefix, binding, RNG and older results. No new native symbol,
Eshkol wrapper, G3-M operation or public generation API is added.

The focused test runs after the existing P1/P2 and G0/G1 aggregate, comparing
all 256 final logits and both K/V positions to independent M3T bits. It covers
manual P1 and P1/G0 predecessors, missing/full prefix, dead/wrong-length/
malformed/borrowed input, stale binding, F32 reservation and A2 cuts,
synthetic provider dispatch failure/retry, staged-tail abort, mask and owner
release. The synthetic cut proves dispatch error propagation, not a measured
failure from an actual provider.

The pinned f31/LLVM21 network-none normal/repeat/ASan+UBSan+LSan aggregate
passed 47,141 byte-identical checks in each mode with empty compiler/runtime
stderr, 20 source contracts, Q0 4/4, and production test-symbol exclusion.
One sanitizer launch used `detect_leaks=1`. Sealed external evidence is at
`/home/gabe/.codex/evidence/eshkol-transformer/g3t-manual-decode-candidate-20260928`.
Independent source review and hosted integration CI remain pending.
