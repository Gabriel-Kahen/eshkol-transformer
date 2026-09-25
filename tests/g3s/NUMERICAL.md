# G3-S numerical test boundary

`reference.py` is development-only and uses no production provider code. Its
integer Philox implementation uses Python arbitrary-precision products, explicitly
splits their halves, and freezes domain-tagged literals plus the standard
zero-counter/zero-key Philox4x32-10 vector. Its literal sampler explicitly rounds
each arithmetic statement to binary32. Only `expf` is delegated to the execution
lane's libm, so this reference does not establish cross-libm bit identity.

A second oracle uses 80-digit Decimal exponentials, sums and divisions. It checks
all 256 softmax probabilities and mathematically ranked top-k renormalized
weights against an absolute tolerance of 2e-5. This is an
absolute mathematical tolerance for the documented serial 256-term f32 sum (whose
standard gamma-256 bound is approximately 1.53e-5), not a token/bit tolerance.
The literal oracle separately requires every inspected intermediate and token to
match exactly. Decimal arithmetic does not select expected literal boundary tokens.
Generated fixture headers are temporary test artifacts, never shipped or committed.

The deterministic fixture inventory includes uniform and asymmetric distributions,
signed-zero maximum ties, highest/lowest selected IDs, k=1 and k=256, p equality
and both adjacent floats, p=1 early crossing and full-sum fallback, exponential
subnormal/zero tails, minimum-positive temperature with equal and distinct logits,
minimum-positive p, largest-finite temperature and logits, rounded probability
ties between distinct logits, low-counter carry, final consumable block, and
multiple sequential raw-sampler calls. Those raw calls are numerical continuation
evidence and do not claim longer model context or generation commit semantics.

Greedy checks use the exhausted numeric state. Categorical singleton cases still
advance one block. The raw byte vocabulary gives no distinguished EOS to the
native sampler; the high-ID singleton's ordinary draw also applies when a later
caller designates that ID as its EOS byte. Public EOS handling, authenticated RNG
save/reload, zero-budget generation, prefill admission, and model/cache rollback
remain downstream G3-T/G3-G gates.

Literal boundary witnesses are also asserted independently of generated tables:

| Inputs | Required result |
|---|---|
| 256 equal logits, k=256 | Every a and b is `0x3b800000` (1/256) |
| Equal logits, p=0.5 or its predecessor | Prefix length 128 |
| Equal logits, p=successor(0.5) | Prefix length 129 |
| Equal logits, p=1, u=143/256 | Token 143 (strict crossing) |
| Seven equal retained probabilities | Each b is `0x3e124924`, total `0x3f7fffff`; p=1 retains all seven |
| Logits `[0,-17,-1000,…]`, p=1 | Prefix length one despite k=256 |

The frozen seed-zero counters 9,773,243 and 3,850,303 produce lane-zero words
`0x00000031` and `0xffffff89`, respectively, proving admitted u=0 and maximum-u
behavior through ordinary K1 dispatch. Counter 207,528 produces `0x8f00006d`,
whose high24 mapping is exactly 143/256. These counters were found by a bounded
independent C integer scan and cross-checked by the Python integer oracle.

A separate static-helper endpoint test deliberately supplies u=1 to exercise the
specified defensive last-positive fallback with a zero-weight tail. It is not a
claim that the public high24 RNG ever emits u=1. Normal dispatch tests retain the
actual RNG path without injection.

`mutations.py` compiles a fresh isolated provider source for each of 23 production
mutations, then requires the real numerical executable to reject it. Compilation
failure never counts as a mutation kill. The mutations cover reverse-ID and
binary64 sums, descending-ID ties, sorting logits before probability rounding,
missing top-k renormalization, top-p before renormalization, strict/epsilon/p=1
prefix variants, nonstrict CDF, last-zero fallback, wrong Philox domain/lane/key/
counter mapping/round count/uniform scale, low24 mapping, double advance, missing
carry, singleton no-draw, greedy last-ID ties, and repaired exponential underflow.
The original checkout and canonical artifact remain unchanged.
