# Private P2/G2 same-registry T1 prefix coordinator candidate

This stacked source candidate follows the accepted [protected ownership scope](G3_C4_P2_G2_PROTECTED_SCOPE_LEAF.md). It adds an explicit same-aggregate owned P2 input constructor, idle native budget-two preflight, and a one-region coordinator wrapper that anchors the exact preflighted input entry/native and generator. Inside the protected `m3-call`, it reserves the exact pending `I1[2]` child, passes the original 16-byte staging allocation to the native first-frame carrier bridge, checks the sampled little-endian V256 ID and non-EOS policy, decodes through the retained T1 tokenizer into the original one-byte raw allocation, reauthenticates every call/input/output and carrier edge, and invokes native prefix commit with that raw header. Only trusted phase and ledger writes follow a successful native commit. The scope still requires an in-region abort and never publishes an output or second token.

The new compiled fixture is intended to exercise two genuine P2 prompts with greedy and categorical sampling, native sampled ID/T1 raw parity, pending zero `I1[2]`, committed `3/1110` cache and successor RNG, and bitwise direct P3 next-logit/K/V reference. A fresh native P2 prefill on the same unchanged owner, prompt and policy emits reference last logits for an independent Python G3-S/Philox check against the two **actual coordinator** selected IDs and RNG states. The fixture cuts prefill, sampling, forward, invalid-ID, selected-EOS, native prefix precommit lease, malformed header, alias and byte-mismatch paths; probes same-region decoder/core/registry/model/pin drift and equal-length carrier-byte changes; verifies exact entry/P2/post-prefix binding snapshots, cache/RNG/model parameter/gradient preservation and captured-buffer scrub; challenges post-prefix output/ledger/auxiliary identity before abort; and retries held output-I1 and A2 abort leases inside the call region. A forked native commit failure checks the impossible closed tail fails stop without damaging the parent’s real staged frame. A separate feature-off fixture rejects coordinator preflight. The native carrier, prefix and protected-scope predecessor gates remain dependencies of the supported runner.

Independent source-and-test review approved `d8e58859`/tree `806a7df6`,
but its first pinned supported gate stopped at the feature-off Eshkol link
before running either new fixture. The runner compiled the T1 shell and M3
I1 owner without their exact-pair feature macro and omitted the existing
I1 borrow-view test wrapper from the link. This follow-up aligns those
real owner/wrapper flags and checks their object symbols before the fixture
link; it changes no production API or numerical path. The next pinned gate
linked and ran the feature-off fixture (five checks) but stopped while
generating enabled Eshkol IR: one test-only precommit probe nested its
after-decode lambda inside the before-decode lambda, giving the four-argument
probe function only three operands. This follow-up closes that lambda and
checks both probe call arities statically. A fresh pinned
normal/repeat/sanitizer Eshkol/native gate remains required before enabled
runtime or integration acceptance. Parent PR #174 is merged. Second-token
generation, EOS terminal handling, output publication, public ABI/profile
and CLI remain later gates.
