# G3-R original-owner joint hold: native source gate

Status: **source prerequisite only**. No aggregate loads this C file; there is
no Scheme coordinator, C2 claim, SAVE pair, receipt, or replay evidence.

`et_g3r_joint_hold_begin_v1` authenticates the fourteen original parameter and
identity pointers through the registered sealed M3T owner. It acquires the
G3/M3/A2 held owner with that roster, then the O2 held owner with the **same**
roster. The private record is allocated before either hold and registered only
after both succeed. If O2 acquisition fails, G3 drains before begin returns.
The record is kept as an inert tombstone after end, so stale addresses cannot
alias a later hold. The leaf is capped at 64 registered records.

`et_g3r_joint_hold_check_v1` checks G3 and exact O2 optimizer owner binding.
`issue_v1` permits one held O2 issue, stores the exact returned pointer in the
registered record before rechecking, and refuses a second issue even after
release. `issued_match_v1` rejects a distinct bit-equal state by pointer
identity before invoking the O2 held bit check. `copy_words_v1` copies the
current 17-word G3 history while both holds remain live. No file I/O occurs.

Preclaim `release_issued_v1` probes the exact native lifecycle after a failed
release: LIVE retains the pointer for controlled retry, DEAD clears it while
reporting the provider defect, and RELEASING or an unprovable lifecycle fails
stop. `end_v1` refuses to drain while this record owns an issued state; once
it is released, end drains O2 before G3 and preserves the first typed error.
The original model/optimizer linkage is established by the one native roster,
not by matching detached state bits.

The next Scheme aggregate must authenticate the exact generator, model, and
optimizer shells before calling begin; preallocate its pair envelope, pointer
cell, C2 request and child cleanup ledger before issuing; adopt this record's
exact pointer into the existing registered O2 Scheme shell; resolve partial
P1/O2 C2 claims; and release the original holds before receipt I/O. The
existing `g3r-held-o2-adopt!` issues through an O2 hold directly. The new
joint-issued source successor below handles the exact native-to-Scheme pointer
handoff. No source here transfers a state to C2.

`scripts/check-g3r-joint-hold-source.py` checks the source ordering with five
mutation controls. Strict feature-on C syntax passes locally. Native runtime,
sanitizer, combined Eshkol compilation, and failure-cut evidence remain open.

## Joint-issued Scheme adopter source successor

`et_g3r_joint_hold_optimizer_match_v1` now checks the exact native optimizer
pointer obtained from a registered Scheme optimizer shell, then rechecks the
joint G3/O2 holds. The feature-private
`native/g3r_joint_issued_o2_adopter_extension.esk` preallocates its Scheme
shell, state record, registry cons, exact-pointer cell, error cell, and ledger;
it installs cleanup protection before asking the joint record to issue once.
The joint record roots the returned pointer natively before returning it;
Scheme stores that same pointer immediately, checks the exact joint pointer
and owner, and only then enrolls its shell in `o2-states`.

The adopter's exception path releases the issued state with lifecycle-aware
native status, retains the first caught error, and leaves the original holds
for the caller's outer cleanup. Its explicit preclaim drain records the first
release defect, consumes the issued state, then calls joint end to drain O2
before G3. Native end itself refuses any still-owned issued pointer. Distinct
bit-equal state pointers and foreign optimizer pointers fail the exact native
queries. This remains source-only; the Scheme extension is not loaded into a
compiled aggregate or exercised at runtime.

No C2 transfer occurs. The next coordinator must preallocate its P1/O2 request
and pair envelope, authenticate original tokenizer/cursor/trainer controls,
claim exactly once, reconcile partial P1/O2 transfer, release both original
holds, then gate exact-image receipt and publication.
