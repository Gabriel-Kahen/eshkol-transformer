# Closed CPU4 unpublished trainer detach

Load `native/cpu4_trainer_detach_extension.esk` after the actual control producer.
`cpu4-trainer-detach-internal!(trainer)` is a single-shot source-private cleanup
seam for a caller-owned failed/unpublished factory. It authenticates the exact
idle CPU4 registry row, canonical binding/P1/config, genuine O2 identity/absent
gradients, idle D2 identity/cursor/count state, and absence of pending/active
parents or restore stages. The actual binding release performs native prepare
(IDLE/no children/unborrowed storage) then its accepted no-fail commit. All
fallible checks precede release; failed preflight retains the exact trainer and
binding for retry. The final bounded store marks the trainer dead. Existing
record access rejects it and overlap scans already skip dead tombstones.

O2, D2, tokenizer, model and canonical P1 parameters remain caller-owned and
usable according to their own contracts. G3 owns candidate child enrollment,
optimizer retirement and shell invalidation order. This seam destroys none of
those components. Repeat detach rejects as stale; the G3 cleanup ledger must
mark successful detach complete before proceeding to its next cleanup phase.
The binding model accessor remains available after lease release.

Source approval does not establish runtime or factory cleanup acceptance.
