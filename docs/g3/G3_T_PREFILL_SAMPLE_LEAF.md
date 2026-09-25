# G3-T private P1/G1 prefill and sampler leaf

This leaf extends the sealed seeded M3T/C2 generator constructor at `824f91e`.
It accepts one byte-ID input, reserves a pending G1 output, executes the fixed
21-role T1 prefill through G3-N, N2 and A2, prepares the exact 14-identity/
4,736-byte model binding, commits the prompt cache and 256 logits, and invokes
G3-S greedy or categorical sampling. The candidate token and
successor RNG remain speculative; the generator RNG is not advanced. The private
Eshkol `g3t-with-call` wrapper roots the call/output under the shared `m3-call`
guard and aborts the pending result on exit. The compiled witness issues the
roles explicitly. There is no token frame, text decode, output publication or
public generation API in this leaf.
`ET_G3T_PREFILL_SAMPLE_PRIVATE` selects the new provider route in the native
translation unit, so the predecessor context/constructor archives retain their
original source and link dependencies. The local Eshkol helper names are pinned
in `native/g3t_prefill_sample_local_symbols.txt`.

P1/G1 fits C2 exactly (`P+G=2`). Other lengths, further generate calls and
categorical draw exhaustion are rejected before prefill. A role or cache
allocation failure leaves the old cache and RNG intact; abort drains all pins,
destroys a candidate cache, and tombstones the pending output. Once prefill has
committed, abort retains the prompt cache and binding while discarding the
sampled candidate. Preparation rechecks the exact model/eval/T1 links and pins;
sampling rechecks all bound parameter bytes. A later token-frame leaf must
revalidate this binding before continuation.

The numerical witness compares all 256 prefill logits bitwise with the first
position of the independent M3T two-token forward schedule, checks an actual
G3-S sample in both modes, and exercises negative ordering and rollback,
including attention failure after nine successful numerical roles. The default pinned
f31/LLVM21 runner builds normal and ASan/UBSan objects, runs normal/repeat/
sanitizer callers, checks private test-hook absence from a production object,
and records a source closure. `G3T_ONLY_NORMAL=1` is a development shortcut;
it is not the final gate.

The predecessor's sanitizer model-seal crash remains an independent limitation:
the previously sealed constructor and the earlier native-context baseline both
showed intermittent crashes in `eshkol_bignum_compare` during model creation,
before generator construction. This leaf does not claim to fix that path.
