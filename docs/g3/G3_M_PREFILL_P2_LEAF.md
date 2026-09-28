# G3-M source-private P2 prefill orchestration leaf

`(g3m-prefill-p2! generator input)` is a private arity-two operation. It
accepts an authentic live G3-T generator and an authentic live same-aggregate
kind-2 input copied from a sealed T1 two-token shell into canonical owned CPU
i64 `[1,2]` storage. The reviewed G3-T native `frame_begin` authenticates the
owned I1 shape, values, inline shadow and unborrowed state. The existing
inline-only pair input has no owned I1 and is ineligible. A successful call
returns a detached kind-3 owned CPU f32 `[1,256]` last-row logits shell.

One outer `m3-call` guard installs rollback before acquiring fourteen parameter
pins. The operation reserves a pending result, begins an unpublished length-two
A2 frame, executes exactly the 21 ordered T2 roles, prepares its last logits
row, preflights publication, commits the A2 cache and detached result, then
finishes immediately. All precommit failures abort the call, retaining the
prior cache, binding, RNG and older detached results. An impossible
postcommit failure is fail-stop. Native P2 admission, numerical routing,
bitwise preflight and atomic publication remain authoritative; the composer
adds no new native ABI, numerical scalar loop, random draw or public facade.

The focused witness compares all 256 output words and both committed K/V
positions against independent M3T bits; replaces P1 with P2, repeats P2,
retains detached results, and covers forged, dead, wrong-kind, inline-only,
malformed, borrowed and busy inputs, I2/A2 allocation, synthetic provider
dispatch, old-cache borrow, rollback and owner release. The aggregate includes
P1/P2-G0 regressions, normal/repeat/sanitizer runs, Q0 isolation, exact source
closure and production test-symbol exclusion. This leaf does not add manual
decode, public generation or N>2.

The pinned f31/LLVM21 network-none normal/repeat/ASan+UBSan+LSan aggregate
passed 47,016 byte-identical checks in each run, with empty compiler and
runtime stderr, 19 source contracts, Q0 4/4, and production native test-symbol
exclusion. One sanitizer launch used `detect_leaks=1`. The synthetic dispatch
cut tests error propagation and abort, not a measured failure from an actual
provider. Sealed external evidence is at
`/home/gabe/.codex/evidence/eshkol-transformer/g3m-prefill-p2-candidate-20260928`.
Independent source review and hosted integration CI remain pending.
