# C4 Eshkol same-region stable-prefix ledger candidate

Status: source/local candidate. No compiled Eshkol admission, SAVE hold,
idle-P3 second-token entry, or trained-C4 continuation is claimed.

The feature-local `g3c4-p2g2-stable-prefix-close!` runs only after the
accepted `g3c4-p2g2-prefix-commit!` inside the **same**
`g3c4-with-p2g2-prefix-call` / `m3-call` dynamic region. Before native close
it reauthenticates the original protected call, input anchor, live pending
capacity-two output, model/generator/tokenizer/policy, non-EOS selected ID
and matching raw T1 byte. It allocates a stable record and independent
anchor in advance. Both are rooted in the existing generator entry's
previously unused slots 8 and 10 only after native close succeeds. They
contain identity edges and phase, not authoritative ID/logit/RNG copies.
The native context remains data authority.

`et_g3c4_private_p2g2_stable_prefix_close_v1` performs the final native
lease and P3 validation. A recoverable native refusal leaves the original
active Eshkol ledger and pending output intact for same-region abort/retry.
The fail-stop handler is installed before native close. After native success,
the Eshkol tail does only checked fixed-length byte scrubbing, vector writes
and head-list drops; any exception fails stop. Source inspection shows no
post-close heap constructor, but compiled allocation/failure behavior still
needs a pinned supported gate before this transaction is admitted.
It kills the pending output and call records, clears model slot 10 and
generator slot 9, removes the protected record/anchor, and installs the
stable record/anchor in generator slots 8/10. No active call or output can
escape the region. The prefix input anchor is removed by the existing
outer prefix-call macro as the body returns.

The ordinary generator-live checker accepts this exact dual-slot shape
under the feature gate. Call-ready rejects stable state; generator close
may retire it through the already reviewed native close. The read-only
stable check reauthenticates the Eshkol edges and native status but returns
only success/failure. Feature-off generator slots remain `#f`.

This native/Eshkol closed state is not a SAVE point yet: there is no
C4-specific read-only SAVE hold, checkpoint association, or file IO protocol.
The distinct idle-P3 second-token entry remains a separate reviewed gate.
