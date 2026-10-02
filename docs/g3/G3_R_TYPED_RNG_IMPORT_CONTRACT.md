# G3-R source-private typed RNG import contract

**Proposed dependency contract only.** No word-import ABI, Eshkol constructor,
checkpoint restore, or continuation is accepted here. This leaf targets the
`diagnostic-c2` G3-T aggregate and the four signed i64 G3-S words at offsets
`88..119` of the accepted [G3RCV1 record](G3_R_DATA_ONLY_CONTINUATION_CONTRACT.md).
The separate fixed-record codec candidate still requires its supported gate.
An authenticated, immutable C2 image/companion pair and a trusted restore
transaction are preconditions for turning decoded words into a live owner;
the data-only codec alone cannot grant that authority.

## Existing boundary

`et_g3t_private_output_rng_clone_v1` makes a native kind-8 `g3t_rng_clone`
only from a complete, independently published G3-T output. It copies four
inline signed words, enrolls a live record in the G3-T native registry, and
`et_g3t_private_rng_release_v1` zeroes and tombstones that record. The Eshkol
`g3t-generation-output-rng` maps it to a same-aggregate pending kind-`rng`
shell and cleanup ledger, then publishes the shell live. Typed release is
idempotent; the clone survives output and generator release. No production
G3-T API reads arbitrary clone words or constructs that clone from saved words.

The accepted private `g3t-generator-create/3` `:rng` route admits an exact live
kind-`rng` shell in its own `g3t-registry`; native
`et_g3t_private_generator_rng_v1` independently admits the kind-8 pointer
and copies all four words into a newly allocated idle generator. Its `:seed`
route initializes both counters to zero and cannot restore a successor.
`et_g3c4_private_result_rng_create_v1` accepts four words but enrolls a
different C4 kind and registry; it is not a G3-T import path. The native
`et_g3t_test_rng_word_v1` can inspect generator words in tests only.

## Proposed private ABI and ownership

Add the guarded native ABI
`et_g3t_private_rng_words_create_v1(int64_t version, int64_t seed,
int64_t counter_low, int64_t counter_high) -> void *` under a new
`ET_G3T_RECORD_RNG_IMPORT_PRIVATE` build guard that requires the existing
`ET_G3T_OUTPUT_RNG_CLONE_PRIVATE` kind-8 type and release. It validates
`version == 1` and `seed >= 0` before allocation. Both counter words accept
every signed i64 bit pattern, including low/high `-1/-1` exhaustion, `-1/0`
carry, and the sign bit. It copies the four values as exact i64 words, with
no float/string conversion, source RNG, seed reset, sampling, or cache work.
Allocate the complete clone, set kind/live state and words, then enroll once;
failure before enrollment leaves no live record. The existing native release
remains the sole destroy path and must scrub all words even after failed
Eshkol publication.

Add a source-private Eshkol wrapper with the proposed signature
`(g3r-rng-owner-from-checked-words version seed counter-low counter-high)`.
It accepts four exact signed i64 scalars, validates the same domain, and
returns only a newly enrolled G3-T kind-`rng` shell. It neither accepts a raw
pointer as authority nor exposes the input vector as an owner. In one
serialized `m3-call`, it preallocates the shell, pending 12-slot registry
entry, ledger and cons cell before native creation. On success it stores the
native pointer and changes the exact pending entry to live as the final
non-failing step. On any recoverable failure it releases only the newly
created native clone and tombstones that pending entry; an impossible native
release failure fails stop. It retains no decoded record, C2 image, model,
tokenizer, output or source RNG. The existing `:rng` constructor copies its
words; the imported shell remains independently releasable.

The wrapper is **not** a standalone file loader or public factory. A future
reviewed restore transaction must first authenticate the exact G3RCV1 SHA,
whole immutable C2-image SHA, profile/policy, model and T1 identities, then
copy all four decoded signed words into this wrapper within the same owning
aggregate. Scalar validation here cannot authenticate that pair, and a raw
four-word vector or equal seed must never be relabeled as a live RNG owner.
The private extension stays out of installed/public G3-G roots until that
transaction is reviewed. No change to the public generator API is proposed.

## Acceptance and dependencies

Native and Eshkol enabled fixtures must compare all four imported words with
independent G3-S/Philox expectations and with a genuine output-clone
predecessor. Cover seed and counter high bits, `-1/0` carry, final consumable
counter, `-1/-1` exhaustion rejected before a categorical draw/A2 mutation,
greedy no-draw behavior, source-shell release independence, and exact retry.
Reject wrong version, negative/overwide seed or counters, forged/wrong-kind,
stale, busy and cross-aggregate shells, allocation cuts and malformed decoded
records; check native zeroization, Eshkol pending-ledger cleanup, idempotent
typed release, and unchanged existing model/cache/RNG/outputs on failure.
Compile a separate feature-off fixture without the import guard or Eshkol
extension and prove the import symbol absent. Require exact source/symbol
closure, Q0, supported normal/repeat and ASan+UBSan+LSan evidence.

The expected implementation touch set is `src/eshkol_transformer/g3t_transport.c`
and `.h`, one new private Eshkol extension loaded only in a trusted test
aggregate, focused off/on native and Eshkol tests, a checker/runner/closure,
and the scoped roadmap leaf. Existing public roots, C4 registry, C2 wire and
G3RCV1 record layout stay unchanged. This owner import still does not supply
the missing stable generation-save snapshot, authenticated pair publication,
live restored model/T1 binding, or full-history no-draw cache/logit replay.
Those are separate prerequisites for fresh-process continuation and the
larger repeated-decode profile required by full G3-R.
