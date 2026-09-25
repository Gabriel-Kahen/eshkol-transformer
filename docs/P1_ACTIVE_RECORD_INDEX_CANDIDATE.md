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

## State-tensor and entry lookup leaf

The separate raw-identity candidate extends the same advisory root with two
heads of canonical promoted records: live state tensors and live state entries.
`shell-for-raw` uses the tensor head; `state-entry-shell-for-raw` uses the entry
head and still rechecks exact raw and owner-state identity. Both fall back to
the append-only registry whenever the shared validity bit is false. No moving
raw object is used as a hash-table key, and native `find_record` is unchanged.

Tensor shell creation resolves its owner state before preparing the registry
snapshot and stage. Entry and tensor records are linked only after registry
publication; any failed preparation/native creation clears the pending root.
Successful state release and pre-clone entry tombstoning unlink records before
changing their raw markers. A lost advisory link leaves the validity bit false
and full-registry lookup authoritative. All existing shell validation and
release predicates remain on the read paths.

The pinned LLVM 21 strict-O0 focused proof passes 24 poisoned publication,
forged and stale checks, plus 11 test-only diagnostic checks. The diagnostic
copy inserts only an extra private slot to simulate a stage failure and missing
advisory heads: it verifies pending cleanup, exact retry identity, and
full-registry fallback while invalid. The production root has no diagnostic
slot or public ABI change. The native normal/sanitized suite passes, including
139 allocation-failpoint checks for state-entry and state-tensor token/record
creation. E3/P1, TR3 fixed-set, prepared-split and generated-root checks pass.
Evidence is under
`/home/gabe/.codex/evidence/eshkol-transformer/p1-raw-identity-focused-20260925/`.
The clean `83eedb2` full P1 package gate passed in 3,701 seconds with
3,983,988 KiB peak RSS; its 232-file seal is
`p1-raw-identity-full-package-83eedb2-20260925/SHA256SUMS` (`44071bc6...`).
The genuine 1,024-cycle SAVE/LOAD/joint-restore continuation passed 2,075
exact-image and ownership checks with no sanitizer diagnostic. It took
17:17.86 at 1,988,812 KiB peak RSS and grew the arena by 1,258,757,248
bytes. Native `find_record` still consumed 128.16 seconds. This does not show
an overall speed gain against the earlier 16:41.49 diagnostic, which was a
different run. Its 53-file seal is
`p1-raw-identity-1024-83eedb2-20260925/SHA256SUMS` (`b92b7967...`).
The 8,192-cycle run was not started: linear projection exceeds the 4 GiB
heap and the 2,400-second runtime cap. A measured retention/time improvement
or separately justified resource bound is needed before that horizon.

## Native exact-token lookup leaf

The integrated leaf adds an intrusive address-ordered AVL to P1's private
native records. It accelerates exact lookup of registered token addresses;
the permanent newest-first `records` chain still owns every record and all
tombstones. Lookup compares only integer forms of pointers until an exact
registered address matches, so forged pointers are never dereferenced. A
missing address or invalid index takes the full chain. Node insertion and
rotation allocate nothing and follow authoritative record publication, before
the result token is returned. Any unexpected duplicate address invalidates
the index and restores full-chain lookup.

This changes only private record size and test-only invalidation API. Public
token layout, status categories, native symbol manifests, and opaque surface
slots remain unchanged. The focused native suite covers 2,828 indexed and
fallback identity checks plus its inherited forged/stale, 139 allocation
failpoints, construction rollback, deterministic builds, and sanitizer checks.
It does not change the separate live-capacity or nonce-uniqueness scans in
`create_token`; terminal record retention and the Eshkol global-arena slope
are expected to remain.

The clean `2a6ece7` full supported P1 package gate passed in 1:01:42 at
3,985,304 KiB peak RSS. It includes the 2,828 native-index checks,
failpoints, sanitizers, publication proof and deterministic rebuilds; the
evidence is sealed at
`p1-native-index-full-package-2a6ece7-20260925/SHA256SUMS`
(`3d7b3224...`). Separate genuine 128/256-cycle samples measured lower
native `find_record` self-time but no overall speed or memory improvement.
The 8,192-cycle retention horizon remains unproven.
