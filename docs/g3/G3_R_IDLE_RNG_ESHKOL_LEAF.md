# Private Eshkol idle RNG equality admission

Status: source-only candidate. No compiled Eshkol result, package admission,
public API, or resumed-generation proof is claimed.

`native/g3r_idle_rng_equal_extension.esk` adds
`(g3r-idle-rng-words-equal-internal! generator version seed low high) -> #t`.
It is loaded only in a future same-registry G3-R restore aggregate after the
G3-T constructor and prefill extensions. The four scalars must first come from
an authenticated G3RCV1/C2 pair; the entry itself does not read a file or
accept a record/vector as authority.

The entry checks exact signed i64 values, version 1 and nonnegative seed
before foreign conversion. It finds the exact live kind-`generator` shell in
the G3-T registry, requires idle Eshkol ledger/output/call slots and the
diagnostic-C2 profile, reauthenticates its canonical eval M3T model and T1
registry root, then passes the registered native handle to the existing
feature-gated `et_g3t_private_idle_rng_words_equal_v1`. Native zero means the
generator is idle with no active frame, model link, pin or pending output and
all four words match bit for bit. Any nonzero status follows existing G3-T
error mapping. The entry mutates no generator, RNG, cache or output state.

`tests/g3r/test_idle_rng_equal_source.py` checks the FFI signature, admission
order, model/T1 roots, native status mapping, private export boundary, and
negative source mutations. The next gate must build one feature-local owning
aggregate with `ET_G3R_G3T_RNG_EQUAL_PRIVATE`, run normal/repeat and supported
ASan/UBSan/LSan with genuine pair-derived words, manual P1/P2 replay, idle
success, each-word mismatch, busy/pin/pending/wrong-owner cases, and exact
source/symbol closure. The accepted native predicate's test-local Eshkol
extern is not this owning wrapper's compiled proof. Candidate retirement,
model restore, pair file publication, no-draw replay and public continuation
remain independent dependencies.
