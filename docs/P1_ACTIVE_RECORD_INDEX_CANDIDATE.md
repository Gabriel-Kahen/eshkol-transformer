# P1 active-record index candidate

This isolated candidate adds two advisory lists of canonical shell-registry
records: one for published states and one for admitted modules, including open
and prepared construction modules. The append-only shell registry and its
native-token index remain the only shell identity authority. The ownership
scans still apply their existing kind, raw, lifecycle, provider, and storage
comparison predicates. If the active index is invalid, they scan the full
registry.

Before creating a native state/module shell, P1 stores a prepared registry
spine and active-list node in a root-owned staging holder. This promotes any
regional graph. The construction guard is installed before staging, so a
staging or native-creation failure clears any unpublished stage. After native
creation P1 writes the token into the canonical record, marks the advisory
index invalid, publishes the prepared registry
spine, links the prepared node, clears staging, and restores the prior validity
bit. The linked values are already global; the pinned checked write barrier
has a zero-allocation path for them. A post-registry failure leaves the index
invalid, so reads fall back to the authoritative registry.

Successful native state-release admission unlinks the state before provider
cleanup callbacks. State-construction rollback also unlinks before releasing
provider-owned carriers. Successful construction abort unlinks module records
as their raw fields become dead markers. Sealed and direct modules remain
admitted because P1 has no module-release API. The active module list is
therefore bounded only by admitted modules, though the current TR3-C retention
fixture creates its modules before its SAVE/restore cycle loop.

This does not accelerate raw-to-shell creation, native `find_record`, or the
terminal-registry compaction scan. The inherited E3/P1 structural checker
already rejects the unmodified `49dcaed` base at its E3 lexical-definition
hash, so it cannot serve as this candidate's gate until separately repaired.
Integration also requires reviewed repinning of source-hash checks after
focused behavioral and performance evidence.
The candidate is not a claim of near-linear end-to-end performance or of the
open 8,192-cycle horizon.

The pinned strict-O0 focused proof passed poisoned nested/sibling publication
(17 checks), construction abort/seal (23 checks), and registry atomicity
(169 checks). Logs and the compressed generated IR are sealed under
`/home/gabe/.codex/evidence/eshkol-transformer/p1-active-record-candidate-focused-20260925/`.
