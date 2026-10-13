# G3-R current-history G3/A2 held-owner leaf

**Source-private candidate; integration HOLD.** This leaf composes the
`86f4eb36` merged committed-history owner with the reviewed F32 and M3
prerequisites. The O2 hold prerequisite has an unresolved output/error alias
finding and is excluded from this published branch. No coordinator, Eshkol
caller, SAVE writer, restore, replay or continuation is admitted.

`ET_G3R_SAME_OWNER_SAVE_PRIVATE` enables four `et_g3r_private_g3t_save_*_v1`
functions. `begin` requires a registered idle generator, the identical sealed
model's fourteen handles and identities, a committed nonempty history with no
pending G1, valid diagnostic CPU-f32 policy/RNG, and matching current A2
length. It pins the original model graph, holds the original A2 cache read
borrow, captures its key/value/mask bits, and makes the generator/model busy.
The captured 136 bytes are seventeen signed words: policy[6], RNG[4], ordered
raw IDs[2], history length, prompt/generated/manual counts, and generated-EOS
terminal. The terminal comes from committed native history, not a detached ID.

`check` reauthenticates the registered hold, model and fixed pins, binding
bits, committed-history invariant and all seventeen words, plus the borrowed
cache's shape, dtype, device, length and exact bits. `copy` writes only a
disjoint 136-byte destination after a successful check. Its destination cannot
overlap any registered M3 or G3-T control shell, including M3 graph/logits
records, separately rostered M3 workspace metadata, another model owner, or
a partial record span. `end` checks once,
then drains the A2 borrow and model pins and leaves an inert registered shell;
release uncertainty fails stop. The first typed G3 error is restored if the
final check fails. This API does no file I/O and grants no caller mutation.

Focused `scripts/test-g3r-current-history-held-owner.sh` passes feature-off,
normal, repeat and ASan+UBSan native cuts, including pending-G1 rejection,
17-word offsets, changed history, changed RNG, stale hold, and aliased/short
copy output. Foreign-owner, M3 graph/logits, M3 workspace-roster and
partial-owner/transport aliases reject without changing bytes or releasing the
active hold; an enrolled M3 graph does not reject disjoint output. LSan is
unavailable in the ptraced local runner and was disabled for the sanitizer
cut. Independent source/tests
review is accepted at `57a8e80a`/tree `8c5fcf94`
(`/tmp/g3r-held-57a8-independent-review-20261013-a/SHA256SUMS-REVIEW`,
manifest SHA-256 `da6b77bfe5227713e55c01da40a8213ce6622ee29e30cbf5adca54efbc74557c`).
The independent private F32/M3/G3 source union is also accepted with
integration HOLD (`g3r-current-main-57a8-union-independent-review-20261013-a/SHA256SUMS-REVIEW`,
manifest SHA-256 `cf7dce7a0a0f1059576904bc693b1506cd9e4bb865d385dbe9a0725f534bcb81`).
Repaired O2 review, post-adoption cleanup, true partial-O2-claim rollback,
and compiled aggregate cuts remain gates before any held SAVE integration.
