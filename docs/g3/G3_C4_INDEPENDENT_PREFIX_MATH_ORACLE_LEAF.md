# G3-C4 independent prefix mathematical oracle source leaf

The reviewed `g3c4-independent-prefix-oracle-contract-20261009-b` contract,
with its reviewed ROADMAP addendum C, is implemented only under `tests/g3c4`.
No production Eshkol source or public API changes. The initializer fixes the
fourteen canonical storage paths, token/head tie, 1192 f32 words, and Philox
successor `[1,seed,292,0]` for seeds 0, 1729, and signed-i64 maximum. It
checks the 13 direct source pins and all 61 full-prefix closure members before
generation or comparison. The only historical member exception is the exact
reviewed ROADMAP status paragraph; the original Git blob and successor bytes
are both pinned. The accepted G3-S development reference is separately pinned
for diagnostic selection and supplies no production math.

The pinned CPU PyTorch 2.13.0+cpu / Python 3.14.6 oracle computes float64
equations for legal prefix lengths 1–4: learned token and position embedding,
Q/K/V and causal two-head attention, three biased-variance norms, exact-erf
GELU FFN, residuals, and tied V256 head. It records all logits and K/V for
boundary and repeated IDs, both current two-ID prompts, and independently
predicted conditional three/four-ID terminal extensions. Predicted selected
IDs remain diagnostic until an explicit tolerance-margin proof. No expected
field is copied from native observations. Generate the fixture twice in
separate pinned Q0 processes and compare its bytes before native observation.
The future observer must provide that frozen fixture's exact SHA-256; the
checker verifies source pins again.

The fixed mathematical comparison is
`abs(native - oracle) <= 2e-6 + 2e-5*abs(oracle)` for every logit and K/V
entry. Observed errors may be reported but cannot widen it. Initializer bits,
state words, record schema, shapes, CPU/f32/f64 dtype, and frozen fixture hash
are exact. Source tests use a **synthetic** f32 transcript to exercise the
parser; it is not native parity evidence. Mutated equations, source pins,
manifest order, and wrong transcripts reject.

Compiled same-call proof still needs an independently captured native
transcript and admitted source/toolchain/runtime pins. This oracle alone proves
no VJP/gradient, EOS ownership, sampler correctness on observed logits, clone,
emitted no-allocation tail, null ABI, symbol closure, sanitizer closure, or
public generation. A future adapter must capture authentic full-model and
K/V/logit images without using them to generate expected values.
