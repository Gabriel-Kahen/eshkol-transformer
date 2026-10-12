# G3-R trainer-to-generator owner handoff prerequisite

**Source audit on main `1590fcd9`; no owner bridge or SAVE API is admitted.**
The separate published `4fe964ed` held native capture and `e80ff5e1`
bytevector FFI candidates descend from `09f7ed52`; neither is on this main
lineage. They hold one G3-T generator, its native M3T model and cache while
copying committed generation state. They do not authenticate a trainer, P1,
O2, X1 or D2 owner.

## First available owner graph and the lifecycle boundary

`tr3-lease-record-for-shell-internal` in
`native/tr3_lease_core_extension.esk` authenticates an **enrolled** trainer
record. Its exact edges include the M3T model and entry, 14 P1 handles and
carriers, O2 optimizer and record, D2 dataset and state, X1 resolved object,
T1 fingerprint/vocabulary/byte-profile identity, cursor, RNG and counters.
Creation receives an exact tokenizer shell and stages its registry entry,
but the enrolled record retains only that identity triple. Even in CLI3,
where `cli3_generate_t2_bridge.esk` maps T1 entry lookup onto the shared T2
registry, a tokenizer shell later supplied to a release-time caller cannot
prove it is the original creation shell. `tr3-c-trainer-state-internal` in
`native/tr3_c_snapshot_extension.esk` uses that live record to clone P1/O2,
capture controls and compose a detached C2 owner under the trainer's
`snapshotting` phase. This is the first current-main callable graph with the
required training components. It does not contain a G3-T generator.

`tr3-p1-fixed-recheck-internal` requires an idle train-mode model, while
`g3t-model-entry-live` requires eval mode. The production combined CLI3
route in `native/cli3_generate_extension.esk` calls `trainer-release!`,
then `module-eval!`, then `g3g-public-generator-create`. Release unenrolls
the exact trainer record; its former shell no longer authenticates the
P1/O2/controls graph. The CLI3 native aggregate
`native/cli3_generate_native_owner.c` intentionally shares one M3 owner
between E3 and G3-T, so native registry separation is **not** the blocker.
The blocker is the missing authenticated ownership transition across that
train-to-eval release barrier. Lexical `model` and `tokenizer` variables in
CLI3 establish only which values its command passes onward; they are not a
registered SAVE lineage or a lease over the detached C2 image and G3 state.

The historical divergent `3e817b` sources contain `tr3-controls-route` and
`g3r-generator-lineage-require` plus a broader SAVE caller. Neither symbol
is callable in main `1590fcd9` or the `4fe`/`e80` lineage. Their ownership
rules are design evidence, not APIs to copy into this branch.

## Required next contract, signature unselected

An owning aggregate must introduce one source-private, registered handoff
authority. Trainer creation must retain the exact tokenizer shell and entry,
or a reviewed equivalent provenance token, in its enrolled owner graph.
The handoff must begin from that graph while the trainer is exact and idle,
then survive the reviewed trainer release, model eval transition and
exact G3-T generator creation. Its inputs must come from the canonical
M3T/P1/O2/X1/D2 graph, the authenticated tokenizer root and generation
policy; its output must retain exact identity edges to the
successor generator and an immutable P1/O2/control capture or detached C2
owner. The signature, result ownership and transfer order remain unselected
until ROOT reviews the irreversible release and C2 claim boundaries.

The handoff must bind the successor G3-T generator to the same M3T model and
raw V256 T1 root and compare all 14 parameter identities and bits against
the captured P1 image. It must preserve the exact O2 state, D2 cursor, X1
configuration, tokenizer fingerprint and training controls; then a held
G3-T copy can authenticate policy, four RNG words, ordered history,
partition and terminal classification before the handoff is consumed. A
detached C2 owner, equal bytes, equal seed or caller-supplied component list
alone cannot mint this authority. No file writer should use the separate
published held-capture candidates until they are composed with this owning
aggregate and reviewed for the same stable interval. A detached C2 owner
captured before trainer release is not by itself proof of the later G3
generator's owner or unchanged model image.

The finite next gate is a source-private compiled CLI3-style aggregate with
one shared E3/G3 M3 owner, an exact trainer snapshot and handoff into an
authentic G3 generator. Positive cases cover P1/G0 and completed P1/G1,
the train-to-eval transition, bitwise comparison of all 14 parameters and
release of the native hold. Negatives reject a wrong trainer, model or one
of the 14 handles; a distinct T1 shell with the same fingerprint; duplicate
or stale handoff and ordinary-release bypass; busy trainer or failed release;
model bit drift across train-to-eval-to-G3; wrong G3 native owner, active
call or frame, pending G1, pending output or logits, and cache or parameter
borrow conflicts; and O2 assignment, update, moments, X1, D2 cursor or
control drift. A detached published output may remain live. Every failure
cut before and after trainer unenrollment must leave no leaked P1/O2/C2 or
G3 authority. Run feature-off, normal, repeat and sanitizer gates on one
exact composed source head before considering a SAVE writer. This audit adds
no source implementation.
