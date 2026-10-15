# G3-R data-only composition boundary at PR #214 source head

This is a source audit of `c71d39a98411d2285fa099f63d1c56170069da3c`
(tree `34ae8017fcb037548974073926bb26ef5fee7fdc`). It adds no runtime API,
SAVE/LOAD transaction, or continuation claim.

The installed leaves can copy an idle generator's committed 136-byte history,
policy and RNG (`native/g3r_committed_save_snapshot_extension.esk`), encode or
decode an in-memory 288-byte G3RCV1 record, import typed G3-S words, load a
C2 owner with a same-image SHA receipt, and read a companion against that
receipt. `src/eshkol_transformer/g3t_transport.h` also declares a guarded
native 17-word same-owner hold. These are separate private seams. The
136-byte Eshkol snapshot returns after releasing its model pins and A2 borrow;
it cannot attest that a later C2 SAVE captured the same owner state.

At this head, `native/` has no Eshkol original-owner held coordinator, G3-R
same-owner C2 SAVE image receipt, or owning file-pair writer. The authentic
public C2 SAVE fixture in `scripts/test-g3r-c2-authentic-save.sh` produces a
fixed-profile image for a read-side gate; it does not save a G3 owner or bind
one to that image. `native/cli3_generate_extension.esk` restores a checkpoint
and constructs a fresh seed-based one-byte generator; it has no history/RNG
replay entry or candidate publication. The proposed candidate model/O2/P1-I2
retirement and G3-T teardown needed after a failed replay are not installed.

The first integration gate is a reviewed, same-registry Eshkol coordinator
that holds the **original** G3/model/cache and O2 authority while C2 clones
and adopts the exact 14-path model/optimizer state. It must check the 17
committed-history words and fixed14 value/gradient bits before and after
transfer, bind a C2 SAVE image receipt to that held snapshot, then release
all holds before file I/O. Reject busy, held, foreign, stale and partial-G1
owners before any caller/file publication; test partial O2 claim, post-adoption
failure, and retry without changing the original owner. The divergent
`origin/codex/g3r-original-owner-coordinator-20261014` source is a candidate
for this gate, not an installed or admitted transaction at this head.

After that gate, pair publication and a fresh-process loader still need
restored model identity, no-draw manual cache replay, candidate-only rollback,
and a single assignment-only publication. A file-read witness or equal-seed
CLI3 generation cannot establish resumed generation.
