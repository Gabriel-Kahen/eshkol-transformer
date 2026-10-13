# Private training-controls capsule source candidate

`g3r-controls-trainer-create` takes the genuine five TR3 inputs after X1/M3
and T2/D2 lineage admission. It returns a private capsule, not a trainer shell.
`g3r-controls-step!` and `g3r-controls-restore!` use the existing trainer
transactions; direct mutation of an enrolled trainer is denied before receiver
writes. Enter checks the live copied controls against the prior issuer journal
before changing phase. The lower entry consumes one operation grant, so a
reentrant direct call in the same phase cannot reuse it. Direct installed
`trainer-train!` is denied before metrics staging. Fixed journal assignments
occur only in the existing irreversible
step and restore control tails, after their successors were copied and rooted.
Each staged journal epoch is checked below signed-i64 maximum before native
commit. An active enrolled commit with no exact staged successor fails stop.
After native commit, the source wrapper uses only a preallocated fixed return
cell; impossible divergence fails stop, and no ordinary lookup or exception
handler can report the committed mutation as a recoverable failure.

`g3r-controls-capture-release!` copies RNG, current/epoch cursor, tokenizer,
configuration and counters while the genuine trainer is still live. It then
releases that trainer and verifies only the separately retained live model,
dataset and optimizer identities and control values before publishing.
`g3r-controls-captured-copy` returns detached control bytes. The private row
is finite (64 cumulative issuances); `g3r-controls-retire!` clears its owner
references. A release whose completion cannot be authenticated stops the
process. A failed ordinary operation retains its original error after cleanup.

These are source-level ordering and provenance claims. Generated package
behavior, callback reentry, no-work emitted tails and sanitizer cleanup remain
unproved. In particular, the capsule does not authenticate same-handle F32
parameter or gradient bits. The later native SAVE reservation must supply an
independent bit witness under its held owner lease before C2 publication.
