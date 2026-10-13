# G3-R current-history held SAVE: proposed private contract

**Reviewed historical proposal; no SAVE admission.** This contract was
published at `07638286`/tree `9250179d` and is carried onto merged main
`86f4eb36` (tree `10e48904`), which includes the committed-history snapshot
candidate from PR #203. That candidate returns
detached data after releasing its M3T pins and A2 read borrow. A joint C2/model
capture needs those original-owner leases to remain live across P1 cloning,
O2 issue/claim, and the final comparison. Calling the existing 136-byte
copyout before or after those steps cannot establish one state.
The bounded source-private G3/A2 held API has since been implemented and
independently source/tests accepted at `57a8e80a`/tree `8c5fcf94`; this
proposal remains the contract for a later joint capture, not that API's
admission into a SAVE writer or coordinator.

## Source conflict and migration boundary

The divergent same-owner producer `c9bad736` (tree `086d8e95`) is source-only.
Its native G3-T branch uses `ET_G3R_G3T_SAVE_LEDGER_PRIVATE`, `c->ledger`, a
staged `candidate_ledger`, and an 18-word registered idle snapshot. The
current branch uses `ET_G3R_COMMITTED_HISTORY_PRIVATE`,
`history_ids/history_*`, and `history_pending_g1`. It commits prompt and
append history at different native sites, and its idle snapshot is 17 words.
An exact merge conflicts in `g3t_transport.c` and `.h`; the old ledger and its
save-begin/check/end entrypoints must not be copied into this owner. The
current committed-history rules, including rejection of an aborted G1 after
P1 prefill, are authoritative.

The old coordinator's **ownership pattern** is reusable after review: root
the original generator, M3T, O2, X1, T1, D2, and trainer-control capsule;
pin the same fourteen M3T handles and their value/absent-gradient bit
witnesses; hold O2 and G3/A2; clone P1 and issue O2 once; recheck each original
witness before child transfer; drain all leases on failure. Its Eshkol
adoption/bundle authentication and successful C2 owner transfer are also
port candidates. Their compiled post-adoption longjmp/root-cell and actual
partial-O2-claim rollback tests remain unrun. None is accepted as a
current-main implementation by this proposal.

## Proposed held native owner contract

At proposal time, names and ABI awaited independent review; the bounded
private `et_g3r_private_g3t_save_*_v1` leaf now covers the G3/A2 hold.
The later joint capture remains subject to the following owner contract.
Acquisition takes the exact registered original generator and model plus the
coordinator's fourteen freshly authenticated parameter handles/identities;
caller-supplied history, RNG, or copied tensor bytes confer no authority. It
must reject foreign/stale or busy owners, wrong diagnostic-C2 CPU-f32 profile,
wrong tied handle, pending G1, invalid partition/terminal, active frame/call,
pending output/logits, and incompatible existing leases **before** granting
the hold. It pins the original M3T graph and A2 cache, holds the generator
and O2 in a documented order, and captures the current 17-word committed
snapshot under those leases. The 14 bit witnesses include the exact primary
value and absent-gradient bits and tied alias; detached P1/O2 copies cannot
substitute for this original-handle comparison.

While held, a check reauthenticates registry identities, model/cache binding,
all fourteen witnesses, the current native committed-history invariant,
no pending G1, and equality to the captured **17 words**. Check before and
after each P1/O2 transfer boundary and before returning a detached C2 owner.
The hold is not a callback that lets caller code mutate the original graph.
No file I/O occurs while it is live. A successful release drains all native
leases exactly once; a failed pretransfer path retains the first typed source
error and drains owned children in reverse order. A failed or uncertain
release/partial claim must retain rooted authority and fail stop rather than
returning a recoverable result. No destination or file may be published on a
recoverable prewrite failure.

The current native structure is 136 bytes, seventeen signed 64-bit words:

| Indices | Current owner source | Later G3RCV1 field |
|---|---|---|
| 0–5 | `policy[6]` | policy[6] |
| 6–9 | `rng[4]` | RNG[4] |
| 10–11 | `history_ids[2]` | ordered raw history, truncated by word 12 |
| 12 | `history_length` | history length, 1 or 2 |
| 13–15 | prompt/generated/manual counts | partition |
| 16 | generated-EOS terminal | checked provenance, not a G3RCV1 field |

The old snapshot had a version prefix at word 0, then policy at 1–6, RNG
at 7–10, IDs at 11–12, counts/terminal at 13–16, and length at 17. That
offset shift is a migration hazard, not a second accepted format. Only after
the held owner proves the 17-word image may a private adapter project the
six policy words, four RNG words, `history_ids[0:length]`, and the three
counts into the seven-field exact-image pair publisher. Recompute and compare
terminal provenance; do not infer it from a detached output or an EOS-valued
prompt/manual byte.

The adapter must separately authenticate the **original** raw-V256 T1/T2
tokenizer and compare its fingerprint with the fingerprint in borrowed C2
controls (`controls[1][0]`, vocabulary 256) and the exact PR #203 raw-V256
fingerprint before publishing. The 17 words contain no tokenizer authority.
The captured trainer-control epoch and fixed profile must be revalidated
against their issuer and the resulting C2 controls; neither a detached epoch
nor an equal-seed model is sufficient. The detached C2 owner must remain
rooted and be released on success and every publisher failure, including a
first-file orphan or post-rename sync/close error. These are later adapter
obligations, not implemented here.

## Required next gates

1. The current-history native G3/A2 hold has independent source/tests
   acceptance (receipt SHA-256
   `da6b77bfe5227713e55c01da40a8213ce6622ee29e30cbf5adca54efbc74557c`).
   The private F32/M3/G3 source union is also accepted with integration HOLD
   (`g3r-current-main-57a8-union-independent-review-20261013-a/SHA256SUMS-REVIEW`,
   manifest SHA-256 `cf7dce7a0a0f1059576904bc693b1506cd9e4bb865d385dbe9a0725f534bcb81`).
   A joint coordinator and O2 repair remain separate gates.
2. Focused native normal/repeat/sanitizer cuts pass for the G3/A2 leaf;
   compiled Eshkol aggregate and remaining failure cuts must cover:
   P1/P2/manual/generated/EOS, pending G1, changed policy/RNG/history,
   fourteen value/gradient bits, foreign/stale owners, lease conflicts,
   allocation failures, post-adoption longjmp, partial O2 claim, and release
   uncertainty. Preserve caller destinations and original owners on every
   recoverable cut.
3. Independently review the adapter against the admitted producer and exact
   C2 callback bytes; test C2 release on success, prewrite failure, orphan,
   and post-rename status. Then prove current-main union, filesystem cuts,
   cold C2/model restore and fresh-process no-draw replay. No public SAVE,
   restore, or data-only continuation follows from this proposal.
