# G3-R capture partial C2 claim source fixture

**Source prepared; compiled gate HOLD.** This test-only fixture is based on
commit `3b8ea59a4527ab564d0f894ca6412ec62474cc34` and tree
`86832ef9f1edb20383f8e7b1107ba85c97582fff`. The preparer checks the
exact 96-file production Eshkol closure and its byte manifest, C2/caller/O2
source hashes, manual P1 prefill helpers, and a dedicated, byte-pinned
FE9/F31/LLVM 21 compiler identity manifest. The recovered 81298 TR3 candidate
is neither staged nor accepted as compiler authority. This identity file is
source data, not a release of a compiled executable. The preparer copies
production leaves without editing them, generates a deterministic four-token
D2 corpus, and seals a read-only staging directory.

The runtime constructs one genuine X1/T2/D2/M3/O2/G3/capsule graph. Its
test-only P1 wrapper delegates to the real claim before recording the exact
P1 state and owner token. The O2 wrapper checks that token against the
genuine issued shell at C2's existing second-claim boundary and raises a
distinct typed error before real O2 assignment. Counting observers delegate
to real C2 P1 rollback and G3 O2/P1 abandon releases. In normal mode,
the result checks
the original error, one of each transition, empty C2 request output,
unchanged C2 registry, 14 parameter value and absent-gradient bit images,
28 O2 moment bit images and update count, 17 G3 policy/RNG/history words,
capsule controls/epoch, D2 cursor, tied alias, one-use denial, and a fresh
valid capture. Wrong token/shell, absent or premature O2 hook, uncertain
rollback and uncertain abandon are separate negative modes. The
`rollback-uncertain` mode exits 134 before the real P1 rollback;
`abandon-uncertain` exits 134 before the real G3 abandon, after the real P1
rollback returns. Neither emits a release marker or reaches post-capture
checks. Status 134 proves process failure only; completed cleanup, recovery,
and C2 publication behavior remain unproved for these modes.

`scripts/test-g3r-capture-partial-claim.sh --check` runs only source
predicates. `--prepare EXPECTED_HEAD ABSOLUTE_FRESH_OUTPUT` verifies exact
HEAD, committed fixture/tool bytes, and whole-file lock/FE9 identity pins
before writing sealed inputs from that immutable commit.
The prepared 113-entry SHA manifest includes the dedicated FE9 identity
file and its exact Git-object bytes; `PREPARED.tsv` records its path and hash.
`--run` deliberately exits 74 until an independently reviewed exact
current-main native union and supported compiled runner are approved. No
compiled behavior, runtime capacity, or integration is claimed. The
source-only native observer passed local Clang 22 syntax checking; this
does not substitute for pinned Clang 21/container execution.
