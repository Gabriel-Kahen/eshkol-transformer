# C4 one-token SAVE boundary and second-token entry dependency

Status: source-grounded blocker contract only. No closed-call one-token C4
owner, SAVE snapshot, idle-P3 second-token entry, or fresh-process continuation
is implemented here. A data-only C4 record's one-token phase remains
provisional until these transitions are reviewed and proved.

## Existing exact boundary

The private P2/G2 prefix commit admits an active `(kind,budget)=(2,2)` call,
its sole pending capacity-two output and a ready position-two frame
(`src/eshkol_transformer/g3c4_model_owner.c:7400-7435`). Its successful
no-failure tail commits A2 position two, advances the real generator RNG,
copies the next 256 logits and selected ID/raw byte into the **active
context**, resets the frame and marks the prefix committed (`:7512-7525`).
The resulting cache is P3, length 3/mask `1110`; it is not a published G1
output. The Eshkol scope requires this prefix to stay inside one `m3-call`
and forbids returning an active call or detached continuation
(`G3_C4_P2_G2_T1_PREFIX_OWNERSHIP_CONTRACT.md:44-58`).

Neither existing close operation can make that prefix saveable. Ordinary
`et_g3c4_private_call_finish_v1` rejects `p2g2_prefix_committed` before it
drains pins and clears `owner->active` (`g3c4_model_owner.c:8121-8129,
8132-8147`). `et_g3c4_private_call_abort_v1` discards the pending output,
clears the prefix ID/raw/next logits and then drains the call (`:8272-8302`).
Abort preserves committed P3 cache/RNG but deliberately does **not** retain
continuation authority (`G3_C4_P2_G2_T1_PREFIX_OWNERSHIP_CONTRACT.md:112-118`).
The G3-R SAVE snapshot contract requires an idle generator with no active
call/frame, pending output/logits, model call link, pins or aborted provenance
(`G3_R_COMMITTED_SAVE_SNAPSHOT_PREREQUISITE.md:83-94`). Saving the active
prefix violates that rule; treating the post-abort P3 cache as a completed
one-token result loses the selected ID/raw and next logits.

The accepted second-frame carrier also cannot begin at idle P3. It calls
`et_g3c4_admit_active_call` and requires the same pending output, budget-two
call and `p2g2_prefix_committed=1` (`g3c4_model_owner.c:2181-2199,
7135-7157`). It samples from `context->p2g2_prefix_next_logits`, consumes
the successor RNG and forwards the selected second token at position three
(`:7167-7188`). The terminal path likewise requires the active capacity-two
output and call (`:8398-8436`). A completed P2/G1 output has requested budget
one and is not a cumulative budget-two prefix; changing its policy words
would not recreate the missing call, cache/history binding or retained logits.

## Required transaction 1: stable one-token close and SAVE hold

Review a distinct feature-local transition from the genuine non-EOS prefix
to an **idle, same-owner, authenticated** one-token state. Before crossing a
closed no-failure tail, it must preallocate and validate any stable record,
rooted Eshkol ledger and cleanup targets; authenticate the original model,
generator, tokenizer, policy, P2 prompt IDs, selected first ID and raw T1
byte, exact fourteen parameter identities, P3 A2 K/V/mask, next-logit bits
and all four successor RNG words. It must prove no held I1/A2/model leases,
foreign/forged handle or changed pending-output edge. Publication must
transfer the committed prefix facts to a stable same-owner record, remove
the sole pending capacity-two output, drain native/Eshkol call and pins,
and make the generator idle **without** calling ordinary finish or abort.
Recoverable pre-tail failures leave the active committed prefix available
for same-region retry or authenticated abort; uncertainty after the first
closed-tail mutation fails stop. No partial output or detached scalar tuple
can serve as authority.

A C4-specific read-only SAVE hold must then authenticate that stable record,
the idle generator, model/T1 binding and P3 cache under leases, returning
ordered P2 plus one generated non-EOS ID, original requested budget two,
terminal=false, complete policy and four RNG words. It must bind this to a
same-trained-model C2 image before a companion is published. The current
capacity-two G3-T snapshot is diagnostic-C2 and cannot be cast into this
C4 owner. No file I/O runs while native leases are held.

## Required transaction 2: genuine idle-P3 second-token entry

Review a separate feature-local entry that consumes the exact idle stable
state (or an equivalently authenticated fresh-process no-draw replay of it),
not caller-supplied ID/logits/RNG bytes. It must reauthenticate the same
model, tokenizer, policy, fourteen parameters, first ID/raw, P3 K/V/mask,
256 last logits and four RNG words; acquire a new call with one remaining
token while retaining the original cumulative requested budget two. Its
pending output and ledger must be new and exact. It may reuse the accepted
second-frame numerical route only after reconstructing its active native
preconditions from that authenticated state; the current entrypoint cannot
be called unchanged. It samples **once** from retained/rechecked P3 logits,
forwards only position three, includes a selected second EOS, and publishes
the correct cumulative two-token terminal result. It never repeats P2
prefill, resamples token one, resets RNG from seed, or relabels P2/G1.
Precommit failure preserves the stable one-token state for retry; successful
publication retires that state exactly once. Distinct first-EOS terminal and
full-context G2 terminal states reject this entry without a draw.

## Minimum next gate

First independently review the native/Eshkol stable-close ownership and
failure-atomic contract against the exact current source and P2/G2 leases.
Then implement a private native stable-prefix record/close plus same-region
Eshkol ledger transition, with normal/repeat/native ASan+UBSan and feature-off
tests. Prove original P3 K/V/logits/ID/raw/RNG bit parity, idle/no-pending
state, held-lease rejection, every fallible cut, post-tail fail-stop, abort
contrast and P2/G1 isolation. Only after that receipt should the idle-P3
entry and SAVE hold receive separate API review and implementation. A later
fresh-process trained-model replay/continuation gate remains mandatory.
