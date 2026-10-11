# G3-R private checked-pair to typed RNG bridge candidate

This bounded source candidate composes the existing file-pair admission and
accepted four-word G3-T RNG import in one private aggregate. It adds
`(g3r-pair-rng-admit-internal c2-path companion-path policy report destination)`.
The destination is an exact ten-slot mutable general vector. On success it
receives `[live C2/SHA receipt, independent G3-T RNG owner, policy, RNG words,
history, prompt length, generated count, manual count, T1 fingerprint,
C2 SHA]`. The caller owns both typed owners and releases them separately.

The bridge passes the paths to `g3r-file-pair-admit-internal` using a private
nine-slot scratch vector. Its decoded RNG words never arrive from a caller
vector. It copies all four checked signed i64 words through
`g3r-rng-owner-from-checked-words`, which enrolls a same-aggregate live kind-8
owner; the existing G3-T `:rng` constructor can then authenticate and copy it.
No seed reset or vector-as-owner route exists. The pair's prepublication fault
hook now receives `#f`, preventing it from mutating the checked staged receipt
or RNG after validation. It remains available for raise-based failure cuts.

All file checks, C2/receipt operations and RNG allocation precede one checked
ten-slot publication. On import or publication failure the bridge releases the
new RNG owner, then the receipt; a cleanup defect fails stop. The accepted
fe9 whole-vector promotion rule leaves caller slots unchanged on failed copy.
The candidate requires serialized access to these private registries and to
the caller destination.

`tests/g3r/pair_rng_bridge_test.esk` includes malformed-pair, native RNG
allocation and destination-promotion negatives, a post-validation hook attack
on receipt/RNG slots, raw-vector rejection, independent word ownership, and
separate release. It also writes a same-aggregate M3T eval-model/raw-T1
`g3t-generator-create :rng` case checking all four copied words before and
after releasing the source owner. `tests/g3r/test_pair_rng_bridge_source.py`
checks source order and fixture coverage. These are source checks only here;
the Eshkol fixture is unexecuted. Aggregate load/symbol closure, the new
`:rng` case and bridge-specific Eshkol allocation cuts remain unverified by
execution. The accepted typed-import leaf exercised its own aggregate only.
No live model restore, tokenizer owner binding, generation replay, public
LOAD/SAVE, or continuation equivalence is claimed.
