# Private P2/G2 EOS-capable first-frame carrier

This candidate adds one source-private prerequisite for the fixed diagnostic
CPU-f32/V256 P2/G2 request. `ET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE` requires
the accepted [carrier bridge](G3_C4_P2_G2_CARRIER_BRIDGE_LEAF.md) and exposes
`et_g3c4_private_p2g2_eos_first_frame_carrier_v1(context, input,
staging_header, carrier_bytes)`. Its signature and owner, pending-output,
input, header, alignment, backing-extent precondition and full owned-storage
alias checks are exactly those of the existing first-frame carrier. Eshkol
must still authenticate its actual eight-byte bytevector allocation and
retained identity before passing the complete 16-byte header/payload carrier.
A raw pointer and self-declared header do not prove allocation extent.

The new route runs genuine P2 prefill, one native last-logit sample, and one
selected-token forward, accepting a genuine selected EOS as well as non-EOS.
The ID is written as eight little-endian payload bytes only after READY-frame
and owner/binding/header reauthentication. The length header is unchanged.
No rejected frame is replayed to discover whether EOS was selected. Greedy
leaves the four-word RNG unchanged; categorical stages the genuine successor
from its single draw. The committed generator RNG remains unchanged in both
cases until a future separately reviewed terminal commit.

The original carrier uses the same numerical body but retains its existing
EOS rejection before forward, including when the new feature is enabled.
The existing non-EOS prefix commit still rejects EOS. This leaf does not
publish numeric IDs, text, an output, or cache/RNG. Successful EOS remains
an active READY transaction at position 2 with an unready capacity-two
pending output. A held transaction view prevents abort; releasing that
view permits authenticated abort, leaving the committed P2 cache `2/1100`
and original RNG. The future logical-G1 terminal preflight/commit and
same-registry T1 decode remain separate dependencies. No second-token,
terminal result, public admission, CLI, persistence or full G2 claim follows.

Preflight errors preserve entry cache and carrier. Sampler/forward errors
after successful prefill preserve P2 cache on abort and leave the carrier
untouched. Callback header tampering after genuine forward is rejected before
the sole payload write. Tests exercise greedy/categorical EOS on two prompts,
non-EOS through the new route, old-route EOS rejection, all four RNG words,
unchanged parameter values, malformed extent/header/alignment, native/f32
aliases, forged IDs, provider cuts, repeated calls, pending output and abort
leases. Candidate next logits and staged K/V are compared bitwise with a
separate genuine three-token full-prefix forward. Captured real logits/IDs
and RNG are checked by the independent G3-S/Philox oracle for both prompts.

The checker reconstructs and hashes the exact reviewed predecessor native
source/header. The runner compares feature-off preprocessed source bytewise,
checks the sole private symbol addition and isolated-macro rejection, runs
the carrier predecessor gate, and compares normal/repeat/ASan+UBSan+LSan
execution. Its fresh build scratch is retained for evidence inspection.

The independently source-and-test-reviewed `a9c0e2c2c52942018c2dd3326842bd91ecfb50aa`
(tree `4b078fabe1d73db3c5f9bb5b8811a7128c072cc7`) passed its exact pinned
fe9/f31 Clang 21 gate on 2026-10-02 from 02:56:59 to 02:58:22 UTC.
Docker and launcher exited zero, with matching clean pre/post source,
compiler, runner, runtime/static-archive and image pins. Normal/repeat and
ASan+UBSan+LSan each passed 2,051 pending-only checks with identical stdout
and empty compile/runtime stderr. Both independent two-mode/256-logit
oracle pairs, all carrier/prefix/first-frame predecessors, exact private-symbol
delta, isolated-feature rejection, reviewed predecessor hashes and feature-off
preprocessor identity passed. Independent evidence review accepted the
verified 77-file seal at
`/home/gabe/g3c4-p2g2-eos-first-frame-a9c0e2c-fe9-f31-20261002-prepared-a/SHA256SUMS-RUN`
(SHA-256 `1287d2a520ed7de09c61f87b69b77a1ae13736ef9b737ff4ae924b64ab6e3b2d`).
Hosted exact-head CI and integration remain pending. Carry/exhaustion, later
draws, Eshkol T1 provenance and terminal publication still require their own
subsequent gates; this accepted native gate does not prove those behaviors.
